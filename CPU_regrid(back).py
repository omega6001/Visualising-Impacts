import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import trimesh
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d


@njit(parallel=True)
def get_blackbody_rgb_numba(T_kelvin):
    T = np.clip(T_kelvin, 1000.0, 40000.0)
    t = T / 1000.0
    r = np.empty_like(t)
    g = np.empty_like(t)
    b = np.empty_like(t)

    for i in prange(len(t)):
        ti = t[i]
        
        # Red
        if ti <= 66.0:
            r[i] = 255.0
        else:
            val = 329.698727446 * (max(ti - 60.0, 1e-6) ** -0.1332047592)
            r[i] = min(max(val, 0.0), 255.0)

        # Green
        if ti <= 66.0:
            val = 99.4708025861 * np.log(max(ti, 1e-6)) - 161.1195681661
            g[i] = min(max(val, 0.0), 255.0)
        else:
            val = 288.1221695283 * (max(ti - 60.0, 1e-6) ** -0.0755148492)
            g[i] = min(max(val, 0.0), 255.0)

        # Blue
        if ti >= 66.0:
            b[i] = 255.0
        elif ti <= 19.0:
            b[i] = 0.0
        else:
            val = 138.5177312231 * np.log(max(ti - 10.0, 1e-6)) - 305.0447927307
            b[i] = min(max(val, 0.0), 255.0)

    rgb = np.stack((r, g, b), axis=-1) / 255.0
    return rgb.astype(np.float32)

@njit(parallel=True)
def scale_colors_keep_ratio_numba(colors, scale_factor):
    n = colors.shape[0]
    scaled_colors = np.empty_like(colors)

    for i in prange(n):
        max_channel = np.max(colors[i])
        if max_channel == 0:
            scaled_colors[i, :] = 0.0
        else:
            normalized = colors[i] / max_channel
            scaled = normalized * (max_channel * scale_factor)
            for j in range(3):
                scaled_colors[i, j] = min(max(scaled[j], 0.0), 1.0)
    return scaled_colors

def laplacian_smooth(mesh, iterations=4, lam=0.2):
    verts = mesh.vertices.copy()
    adj = mesh.vertex_neighbors
    for _ in range(iterations):
        new_verts = verts.copy()
        for i, neighbors in enumerate(adj):
            if not neighbors:
                continue
            avg = np.mean(verts[list(neighbors)], axis=0)
            new_verts[i] = verts[i] + lam * (avg - verts[i])
        verts = new_verts
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

@njit
def match_particles_numba(ids_a, ids_b):
    id_to_idx_b = {}
    for idx in range(len(ids_b)):
        id_to_idx_b[ids_b[idx]] = idx

    matched_idx_a = []
    matched_idx_b = []
    for idx_a in range(len(ids_a)):
        pid = ids_a[idx_a]
        if pid in id_to_idx_b:
            matched_idx_a.append(idx_a)
            matched_idx_b.append(id_to_idx_b[pid])
    return np.array(matched_idx_a), np.array(matched_idx_b)

@njit(parallel=True, fastmath=True)
def batch_hermite_positions(pos_a, pos_b, vel_a, vel_b, dt, t_vals):
    """
    Compute Hermite positions for all particles and all t_vals.
    pos_a, pos_b, vel_a, vel_b: (N, 3)
    dt: scalar
    t_vals: (M,)
    Returns: positions (N, M, 3)
    """
    N = pos_a.shape[0]
    M = t_vals.shape[0]
    positions = np.zeros((N, M, 3), dtype=np.float64)
    for i in prange(N):
        p0 = pos_a[i]
        p1 = pos_b[i]
        v0 = vel_a[i]
        v1 = vel_b[i]
        for j in range(M):
            t = t_vals[j]
            t2 = t * t
            t3 = t2 * t
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            positions[i, j, 0] = h00 * p0[0] + h10 * v0[0] * dt + h01 * p1[0] + h11 * v1[0] * dt
            positions[i, j, 1] = h00 * p0[1] + h10 * v0[1] * dt + h01 * p1[1] + h11 * v1[1] * dt
            positions[i, j, 2] = h00 * p0[2] + h10 * v0[2] * dt + h01 * p1[2] + h11 * v1[2] * dt
    return positions

@njit(parallel=True, fastmath=True)
def compute_arc_lengths(positions):
    """
    Compute cumulative arc length along Hermite curve for each particle.
    positions: (N, M, 3)
    Returns: arc_lengths (N, M)
    """
    N, M, _ = positions.shape
    arc_lengths = np.zeros((N, M), dtype=np.float64)
    for i in prange(N):
        total_length = 0.0
        arc_lengths[i, 0] = 0.0
        for j in range(1, M):
            dx = positions[i, j, 0] - positions[i, j-1, 0]
            dy = positions[i, j, 1] - positions[i, j-1, 1]
            dz = positions[i, j, 2] - positions[i, j-1, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += dist
            arc_lengths[i, j] = total_length
    return arc_lengths

@njit(parallel=True, fastmath=True)
def get_uniform_reparam_indices(arc_lengths, n_uniform):
    """
    For each particle, find indices in arc_lengths that correspond to uniform arc-length spacing.
    arc_lengths: (N, M)
    n_uniform: scalar number of uniform points desired
    Returns: indices (N, n_uniform) integer indices into arc_lengths positions
    """
    N, M = arc_lengths.shape
    indices = np.zeros((N, n_uniform), dtype=np.int64)
    for i in prange(N):
        total_len = arc_lengths[i, M-1]
        if total_len == 0.0:
            # Degenerate case: just sample first position repeatedly
            for u in range(n_uniform):
                indices[i, u] = 0
            continue
        for u in range(n_uniform):
            target_len = total_len * u / (n_uniform - 1) if n_uniform > 1 else 0.0
            # Binary search for closest arc length index
            low = 0
            high = M - 1
            while low < high:
                mid = (low + high) // 2
                if arc_lengths[i, mid] < target_len:
                    low = mid + 1
                else:
                    high = mid
            indices[i, u] = low
    return indices

@njit(parallel=True)
def pointcloud_to_grid_numba(positions, values, grid_size, global_min, global_max):
    grid = np.zeros(grid_size, dtype=np.float32)
    scale = 1.0 / (global_max - global_min + 1e-10)
    n = positions.shape[0]
    
    for i in prange(n):
        norm = (positions[i] - global_min) * scale
        idx = (norm * (np.array(grid_size) - 1)).astype(np.int32)
        x, y, z = idx
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
            grid[x, y, z] += values[i]
    return grid



def get_global_bounds(folder):
    min_all = np.full(3, np.inf)
    max_all = np.full(3, -np.inf)
    min_en = np.inf
    max_en = -np.inf
    max_density = -np.inf
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                energy = f["PartType0/InternalEnergies"][:]
                dens = f["PartType0/Densities"][:]
                min_en = min(min_en, energy.min())
                max_en = max(max_en, energy.max())
                max_density = max(max_density, dens.max())
    #print(min_all, max_all)
    min_all = np.array([250,250,250])
    max_all= np.array([400,400,400])
    return min_all, max_all, min_en, max_en, max_density


def edge_blend_weights(grid_shape, padding_voxels):
    """
    Create a smooth blending weight mask for a 3D grid that fades from 1 inside
    to 0 at the edges over the padding region.

    Args:
        grid_shape (tuple of int): Shape of the 3D grid (nx, ny, nz).
        padding_voxels (int): Number of voxels over which to blend near edges.

    Returns:
        weights (np.ndarray): 3D array of shape grid_shape with values in [0,1].
                              Values are 1 in the inner region and fade to 0 at edges.
    """
    nx, ny, nz = grid_shape

    def fade1d(size):
        """1D fade from 1 to 0 near edges over padding_voxels"""
        w = np.ones(size, dtype=np.float32)
        # Linear fade at start edge
        w[:padding_voxels] = np.linspace(0, 1, padding_voxels, endpoint=False)
        # Linear fade at end edge
        w[-padding_voxels:] = np.linspace(1, 0, padding_voxels, endpoint=False)
        return w

    wx = fade1d(nx)
    wy = fade1d(ny)
    wz = fade1d(nz)

    # Outer product to get 3D weights
    weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    return weights



def process_with_interpolated_marching_cubes_cpu(
    hdf5_folder, out_folder, bounds_min, bounds_max, globalmin_en, globalmax_en, global_thresh=0.001,
    grid_size=(900, 900, 900), coarse_grid_size=(100, 100, 100), interp_steps=1,
    blur_sigma=0.5, padding_voxels=8
):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')])
    frame_counter = 0

    #global_thresh = None  # Initialize global threshold

    for frame_idx in range(len(files) - 1):
        fname_a, fname_b = files[frame_idx], files[frame_idx + 1]
        print(f"Interpolating between {fname_a} and {fname_b}")

        with h5py.File(os.path.join(hdf5_folder, fname_a), 'r') as fa, \
             h5py.File(os.path.join(hdf5_folder, fname_b), 'r') as fb:

            ids_a = fa['PartType0/ParticleIDs'][:]
            pos_a, vel_a = fa['PartType0/Coordinates'][:], fa['PartType0/Velocities'][:]
            dens_a, ie_a = fa['PartType0/Densities'][:], fa['PartType0/InternalEnergies'][:]

            ids_b = fb['PartType0/ParticleIDs'][:]
            pos_b, vel_b = fb['PartType0/Coordinates'][:], fb['PartType0/Velocities'][:]
            dens_b, ie_b = fb['PartType0/Densities'][:], fb['PartType0/InternalEnergies'][:]

            time_a = fa.attrs.get('Time', 0.0)
            time_b = fb.attrs.get('Time', 1.0)
            dt = time_b - time_a

            idx_a, idx_b = match_particles_numba(ids_a, ids_b)
            pos_a, pos_b = pos_a[idx_a], pos_b[idx_b]
            vel_a, vel_b = vel_a[idx_a], vel_b[idx_b]
            dens_a, dens_b = dens_a[idx_a], dens_b[idx_b]
            ie_a, ie_b = ie_a[idx_a], ie_b[idx_b]
            
            # Step 1: define fine sampling along Hermite curves
            M = 150  # number of fine samples for arc length calculation
            t_vals = np.linspace(0.0, 1.0, M)

            # Step 2: compute batch Hermite positions (N particles x M samples x 3)
            hermite_pos = batch_hermite_positions(pos_a, pos_b, vel_a, vel_b, dt, t_vals)

            # Step 3: compute cumulative arc lengths along each curve
            arc_lengths = compute_arc_lengths(hermite_pos)

            # Step 4: find indices for uniform arc length spacing for interp_steps points
            uniform_idx = get_uniform_reparam_indices(arc_lengths, interp_steps)

            N = pos_a.shape[0]

            # Step 5: for each interpolation step, gather interpolated positions directly
            for s in range(interp_steps-1):
                interp_pos = hermite_pos[np.arange(N), uniform_idx[:, s]]

                t = s / (interp_steps - 1) if interp_steps > 1 else 0.0
                # Now interpolate scalar fields linearly as before
                interp_dens = (1 - t) * dens_a + t * dens_b
                interp_ie = (1 - t) * ie_a + t * ie_b
                # (halo_temperature_kelvin, marching cubes, saving .npz, etc.)
                halo_temperature_kelvin = interp_ie * 1.0e9
                #print(max(halo_temperature_kelvin), min(halo_temperature_kelvin), np.mean(halo_temperature_kelvin))
                halo_colors = get_blackbody_rgb_numba(halo_temperature_kelvin)
                halo_colors = np.clip(scale_colors_keep_ratio_numba(halo_colors,1.0), 0, 1)

                grid_indices = np.floor((interp_pos - bounds_min) / (bounds_max - bounds_min) * coarse_grid_size).astype(int)
                grid_indices = np.clip(grid_indices, 0, np.array(coarse_grid_size) - 1)
                flat_indices = grid_indices[:, 0] * coarse_grid_size[1] * coarse_grid_size[2] + \
                               grid_indices[:, 1] * coarse_grid_size[2] + grid_indices[:, 2]

                coarse_dens_grid = pointcloud_to_grid_numba(
                    interp_pos, interp_dens, coarse_grid_size, bounds_min, bounds_max
                )
                coarse_dens_grid = gaussian_filter(coarse_dens_grid, sigma=blur_sigma)

                #coarse_volume = np.prod(coarse_grid_size)
                #fine_volume = np.prod(grid_size)
                #scale_ratio = fine_volume / coarse_volume
                #global_thresh = (np.max(coarse_dens_grid) * 0.001) / scale_ratio
                #global_thresh = global_thresh / scale_ratio
                print(f"Using global marching cubes threshold: {global_thresh:.6e}")

                active_cells = np.argwhere(coarse_dens_grid > np.max(coarse_dens_grid) * 0.001)

                def process_cell(cell):
                    cx, cy, cz = cell
                    cell_min_norm = cell / np.array(coarse_grid_size)
                    cell_max_norm = (cell + 1) / np.array(coarse_grid_size)
                    cell_min = bounds_min + cell_min_norm * (bounds_max - bounds_min)
                    cell_max = bounds_min + cell_max_norm * (bounds_max - bounds_min)
                    cell_extent = cell_max - cell_min
                    sim_extent = bounds_max - bounds_min
                    pad = np.maximum((padding_voxels / np.array(grid_size)) * sim_extent * 0.2,
                                     (padding_voxels / np.array(grid_size)) * cell_extent)
                    region_min = np.clip(cell_min - pad, bounds_min, bounds_max)
                    region_max = np.clip(cell_max + pad, bounds_min, bounds_max)

                    mask = np.all((interp_pos >= region_min) & (interp_pos <= region_max), axis=1)
                    if mask.sum() < 50:
                        return None

                    local_pos_full = interp_pos[mask]
                    local_dens_full = interp_dens[mask]
                    local_ie_full = interp_ie[mask]
                    local_colors_full = halo_colors[mask]
                    local_flat_indices = flat_indices[mask]

                    owning_cell_idx = cx * coarse_grid_size[1] * coarse_grid_size[2] + cy * coarse_grid_size[2] + cz
                    ownership_mask = local_flat_indices == owning_cell_idx
                    if ownership_mask.sum() < 50: ##ignores cells with less than this many particles
                        return None

                    halo_pos = local_pos_full[ownership_mask]
                    halo_colors_out = local_colors_full[ownership_mask]

                    dens_grid = pointcloud_to_grid_numba(local_pos_full, local_dens_full, grid_size, region_min, region_max)
                    weights = edge_blend_weights(dens_grid.shape, padding_voxels)
                    dens_grid *= weights
                    dens_grid = gaussian_filter(dens_grid, sigma=blur_sigma)

                    if dens_grid.max() < global_thresh: #####needs to return only halo particles
                        # Return only the halo data if no surface can be extracted
                        return None, None, None, halo_pos, None, halo_colors_out, None


                    verts, faces, normals, _ = marching_cubes(dens_grid, level=global_thresh)
                    scale = (region_max - region_min) / (np.array(grid_size) - 1)
                    verts_world = verts * scale + region_min
                    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
                    smoothed_mesh = laplacian_smooth(mesh, iterations=2, lam=0.4)

                    verts_world = smoothed_mesh.vertices
                    faces = smoothed_mesh.faces
                    normals = smoothed_mesh.vertex_normals

                    energy_grid = pointcloud_to_grid_numba(local_pos_full, local_ie_full, grid_size, region_min, region_max)
                    energy_grid = gaussian_filter(energy_grid, sigma=blur_sigma)

                    grid_coords = ((verts - 1e-4) / (np.array(grid_size) - 1) * (np.array(grid_size) - 1)).astype(int)
                    grid_coords = np.clip(grid_coords, 0, np.array(grid_size) - 1)

                    grad_x = np.gradient(dens_grid, axis=0)
                    grad_y = np.gradient(dens_grid, axis=1)
                    grad_z = np.gradient(dens_grid, axis=2)
                    grad_grid = np.stack([grad_x, grad_y, grad_z], axis=-1)

                    density_at_verts = dens_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
                    grad_at_verts = grad_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
                    grad_mag = np.linalg.norm(grad_at_verts, axis=1)
                    vertex_ie = energy_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]

                    weight = np.clip(2000 * (density_at_verts - 0.001), 0.0, 1.0)
                    weight[grad_mag < 1e-6] = 0.0

                    light_dir = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
                    light_dir /= np.linalg.norm(light_dir)
                    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)####################################################################

                    normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
                    grad_normals_unit = grad_at_verts / (grad_mag[:, None] + 1e-12)
                    shading_normals = (1.0 - weight[:, None]) * normals_unit + weight[:, None] * grad_normals_unit
                    shading_normals /= np.linalg.norm(shading_normals, axis=1, keepdims=True) + 1e-12

                    dot_nl = np.einsum('ij,j->i', shading_normals, light_dir)
                    facing_light = dot_nl > 0
                    dot_nl_clamped = np.clip(dot_nl, 0.0, 1.0)
                    diffuse = dot_nl_clamped
                    reflect_dir = 2 * dot_nl_clamped[:, None] * shading_normals - light_dir
                    reflect_dir /= np.linalg.norm(reflect_dir, axis=1, keepdims=True) + 1e-12
                    dot_rv = np.einsum('ij,j->i', reflect_dir, view_dir)
                    specular = np.clip(dot_rv, 0.0, 1.0) ** 12
                    specular *= 0.6 * facing_light
                    ambient_color = np.full((verts.shape[0], 3), 0.15, dtype=np.float32)  # You can tweak 0.1 to control intensity
                    base_color = np.full((verts.shape[0], 3), 0.35, dtype=np.float32)
                    lit_color = ambient_color + base_color * diffuse[:, None] + specular[:, None]

                    temperature_kelvin = vertex_ie * (grid_size[0]**3) * 3200
                    #print(max(temperature_kelvin), min(temperature_kelvin), np.mean(temperature_kelvin))
                    raw_emission = get_blackbody_rgb_numba(temperature_kelvin)
                    emission_strength = np.clip(temperature_kelvin / 8000.0, 0.0, 1.0) ** 2
                    blackbody_emission = raw_emission * emission_strength[:, None]
                    raw_color = lit_color + blackbody_emission
                    raw_color = np.clip(raw_color, 0.0, 1.0)

                    blend_weights = weights[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
                    vertex_blackbody_color = scale_colors_keep_ratio_numba(raw_color, 1.0) * blend_weights[:, None]

                    return verts_world, faces, normals, halo_pos, vertex_ie, halo_colors_out, vertex_blackbody_color

                all_verts, all_faces, all_normals = [], [], []
                all_local_pos, all_vertex_ie = [], []
                all_local_colors, all_vertex_blackbody_color = [], []

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_cell, active_cells))

                vert_count = 0
                for res in results:
                    if res is None:
                        continue

                    verts_world, faces, normals, local_pos, vertex_ie, local_colors, vertex_blackbody_color = res

                    # Store halo particle data (always present if res is not None)
                    all_local_pos.append(local_pos)
                    all_local_colors.append(local_colors)

                    if verts_world is not None and faces is not None and normals is not None:
                        # Only store mesh data if it exists
                        faces_offset = faces + vert_count
                        all_verts.append(verts_world)
                        all_faces.append(faces_offset)
                        all_normals.append(normals)

                        if vertex_ie is not None:
                            all_vertex_ie.append(vertex_ie)

                        if vertex_blackbody_color is not None:
                            all_vertex_blackbody_color.append(vertex_blackbody_color)

                        vert_count += verts_world.shape[0]

                # Save if we have at least the halo data
                if all_local_pos:
                    save_path = os.path.join(out_folder, f"mesh_{frame_counter:04d}.npz")
                    np.savez(save_path,
                            verts=np.vstack(all_verts).astype(np.float32) if all_verts else np.empty((0, 3), dtype=np.float32),
                            faces=np.vstack(all_faces).astype(np.uint32) if all_faces else np.empty((0, 3), dtype=np.uint32),
                            normals=np.vstack(all_normals).astype(np.float32) if all_normals else np.empty((0, 3), dtype=np.float32),
                            halo_positions=np.vstack(all_local_pos).astype(np.float32),
                            vertex_internal_energy=np.concatenate(all_vertex_ie).astype(np.float32) if all_vertex_ie else np.empty((0,), dtype=np.float32),
                            halo_blackbody_color=np.vstack(all_local_colors).astype(np.float32),
                            vertex_blackbody_color=np.vstack(all_vertex_blackbody_color).astype(np.float32) if all_vertex_blackbody_color else np.empty((0, 3), dtype=np.float32))
    
                    print(f"Saved mesh frame {frame_counter} to {save_path}")
                    frame_counter += 1




                                   
                
if __name__ == "__main__":
    input_folder = "outs2/"
    output_folder = "meshes_interpolated/"
    bounds_min, bounds_max, globalmin_en, globalmax_en, dense_max = get_global_bounds(input_folder)
    global_thresh = dense_max*0.0002 #for 10^6 this is 1.382e-5 (or 0.001*)#dense_max*0.001

    process_with_interpolated_marching_cubes_cpu(
        input_folder,
        output_folder,
        bounds_min,
        bounds_max,
        globalmin_en,
        globalmax_en,
        global_thresh,
        grid_size=(400,400, 400),         # High-res per region
        coarse_grid_size=(10, 10, 10),  # Coarse global scan
        interp_steps=5,
        blur_sigma=1.2,
        padding_voxels=8                   # Padding to reduce continuity artifacts
)