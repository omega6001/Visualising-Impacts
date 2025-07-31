import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import trimesh
from numba import njit, prange


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

def laplacian_smooth(mesh, iterations=8, lam=0.5):
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



def hermite_interp_vec(p0, p1, v0, v1, t, dt):
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (h00 * p0.T + h10 * dt * v0.T + h01 * p1.T + h11 * dt * v1.T).T

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

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                coords = f["PartType0/Coordinates"][:]
                min_all = np.minimum(min_all, coords.min(axis=0))
                max_all = np.maximum(max_all, coords.max(axis=0))
                energy = f["PartType0/InternalEnergies"][:]
                min_en = min(min_en, energy.min())
                max_en = max(max_en, energy.max())
    return min_all, max_all, min_en, max_en

def process_with_interpolated_marching_cubes_cpu(
    hdf5_folder, out_folder, bounds_min, bounds_max, globalmin_en, globalmax_en,
    grid_size=(900, 900, 900), coarse_grid_size=(100, 100, 100), interp_steps=1,
    blur_sigma=0.8, padding_voxels=8
):

    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')])
    frame_counter = 0

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

            for s in range(interp_steps):
                t = s / interp_steps
                interp_pos = hermite_interp_vec(pos_a, pos_b, vel_a, vel_b, t, dt)
                interp_dens = (1 - t) * dens_a + t * dens_b
                interp_ie = (1 - t) * ie_a + t * ie_b
                halo_temperature_kelvin = interp_ie * 1e9
                halo_colors = get_blackbody_rgb_numba(halo_temperature_kelvin)
                halo_colors = np.clip(halo_colors ** 0.8, 0, 1)

                # === FIRST PASS: COARSE GRID OCCUPANCY ===
                coarse_grid_size = tuple(coarse_grid_size)
                coarse_dens_grid = pointcloud_to_grid_numba(
                    interp_pos, interp_dens, coarse_grid_size, bounds_min, bounds_max
                )
                coarse_dens_grid = gaussian_filter(coarse_dens_grid, sigma=1.0)
                threshold = coarse_dens_grid.max() * 1e-6                
                active_cells = np.argwhere(coarse_dens_grid > threshold)
                print(f"Found {len(active_cells)} active coarse grid cells")

                # Prepare accumulators for this frame
                all_verts = []
                all_faces = []
                all_normals = []
                all_vertex_ie = []
                all_vertex_blackbody_color = []
                all_local_pos = []
                all_local_colors = []

                vert_count = 0

                for cell in active_cells:
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
                    
                    if mask.sum() < 500:
                        continue  # Skip sparse regions
                    print(f"Particles in region: {np.sum(mask)} / {interp_pos.shape[0]}")
                    local_pos = interp_pos[mask]
                    local_dens = interp_dens[mask]
                    local_ie = interp_ie[mask]
                    local_colors = halo_colors[mask]

                    local_bounds_min = region_min
                    local_bounds_max = region_max
                    local_grid_size = tuple(grid_size)

                    dens_grid = pointcloud_to_grid_numba(local_pos, local_dens, local_grid_size, local_bounds_min, local_bounds_max)
                    dens_grid = gaussian_filter(dens_grid, sigma=blur_sigma)
                    if dens_grid.max() <= 0:
                        continue

                    print(f"Checking subgrid cell {cell}")
                    print(f"Region min: {region_min}, Region max: {region_max}")
                    mc_thresh = dens_grid.max() * 0.001
                    verts, faces, normals, _ = marching_cubes(dens_grid, level=mc_thresh)
                    grid_shape = np.array(local_grid_size)
                    scale = (local_bounds_max - local_bounds_min) / (grid_shape - 1)
                    verts_world = verts * scale + local_bounds_min
                    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
                    smoothed_mesh = laplacian_smooth(mesh, iterations=10, lam=0.4)

                    verts_world = smoothed_mesh.vertices
                    faces = smoothed_mesh.faces
                    normals = smoothed_mesh.vertex_normals

                    energy_grid = pointcloud_to_grid_numba(local_pos, local_ie, local_grid_size, local_bounds_min, local_bounds_max)
                    energy_grid = gaussian_filter(energy_grid, sigma=blur_sigma)

                    local_grid_size_arr = np.array(local_grid_size)
                    norm_verts = verts / (local_grid_size_arr - 1)
                    verts_scaled = norm_verts * (local_bounds_max - local_bounds_min) + local_bounds_min
                    grid_coords = ((verts_scaled - local_bounds_min) / (local_bounds_max - local_bounds_min + 1e-8) * (local_grid_size_arr - 1)).astype(int)
                    grad_x = np.gradient(dens_grid, axis=0)
                    grad_y = np.gradient(dens_grid, axis=1)
                    grad_z = np.gradient(dens_grid, axis=2)
                    grad_grid = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape: (X, Y, Z, 3)

                    # Interpolate density & gradient at vertex positions
                    grid_coords = ((verts_scaled - local_bounds_min) / (local_bounds_max - local_bounds_min + 1e-8) * (local_grid_size_arr - 1)).astype(int)

                    density_at_verts = dens_grid[
                        grid_coords[:, 0],
                        grid_coords[:, 1],
                        grid_coords[:, 2]
                    ]
                    grad_at_verts = grad_grid[
                        grid_coords[:, 0],
                        grid_coords[:, 1],
                        grid_coords[:, 2]
                    ]
                    grad_mag = np.linalg.norm(grad_at_verts, axis=1)
                    vertex_ie = energy_grid[
                        grid_coords[:, 0],
                        grid_coords[:, 1],
                        grid_coords[:, 2]
                    ]

                    # === Weight from density gradient + density
                    weight = np.clip(2000 * (density_at_verts - 0.001), 0.0, 1.0)
                    weight[grad_mag < 1e-6] = 0.0
                    #####################################LIGHTING#######################################
                    light_dir = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
                    light_dir /= np.linalg.norm(light_dir)
                    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # assuming camera looks along -Z

                    # === Mix shading normal: blend between mesh normal and gradient normal
                    normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
                    grad_normals_unit = grad_at_verts / (grad_mag[:, None] + 1e-12)
                    shading_normals = (1.0 - weight[:, None]) * normals_unit + weight[:, None] * grad_normals_unit
                    shading_normals /= np.linalg.norm(shading_normals, axis=1, keepdims=True)

                    # Normalized vectors
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    view_dir = view_dir / np.linalg.norm(view_dir)
                    shading_normals /= np.linalg.norm(shading_normals, axis=1, keepdims=True) + 1e-12

                    # Light-facing check
                    dot_nl = np.einsum('ij,j->i', shading_normals, light_dir)
                    facing_light = dot_nl > 0
                    dot_nl_clamped = np.clip(dot_nl, 0.0, 1.0)

                    # Diffuse
                    diffuse = dot_nl_clamped

                    # Specular reflection direction
                    reflect_dir = 2 * dot_nl_clamped[:, None] * shading_normals - light_dir
                    reflect_dir /= np.linalg.norm(reflect_dir, axis=1, keepdims=True) + 1e-12

                    dot_rv = np.einsum('ij,j->i', reflect_dir, view_dir)
                    specular = np.clip(dot_rv, 0.0, 1.0) ** 12
                    specular *= 0.6
                    specular *= facing_light  # prevent glints on back side

                    # Base color and emission
                    base_color = np.full((verts.shape[0], 3), 0.3, dtype=np.float32)
                    lit_color = base_color * diffuse[:, None] + specular[:, None]

                    # Emission (blackbody)
                    temperature_kelvin = vertex_ie * (grid_size[0]**3) * 1.4625
                    raw_emission = get_blackbody_rgb_numba(temperature_kelvin)
                    emission_strength = np.clip(temperature_kelvin / 8000.0, 0.0, 1.0) ** 2
                    blackbody_emission = raw_emission * emission_strength[:, None]
                    brightness_scale = 1.05
                    # Combine lighting and emission
                    raw_color = lit_color + blackbody_emission
                    raw_color = np.clip(raw_color, 0.0, 1.0)
                    vertex_blackbody_color = scale_colors_keep_ratio_numba(raw_color, brightness_scale)
                    # Offset faces for concatenation
                    faces_offset = faces + vert_count

                    # Accumulate all results
                    all_verts.append(verts_world)
                    all_faces.append(faces_offset)
                    all_normals.append(normals)
                    all_local_pos.append(local_pos)
                    all_vertex_ie.append(vertex_ie)
                    all_local_colors.append(local_colors)
                    all_vertex_blackbody_color.append(vertex_blackbody_color)

                    vert_count += verts_world.shape[0]

                # Save combined mesh for this frame (if any subgrid was processed)
                if all_verts:
                    combined_verts = np.vstack(all_verts)
                    combined_faces = np.vstack(all_faces)
                    combined_normals = np.vstack(all_normals)
                    combined_local_pos = np.vstack(all_local_pos)
                    combined_vertex_ie = np.concatenate(all_vertex_ie)
                    combined_local_colors = np.vstack(all_local_colors)
                    combined_vertex_blackbody_color = np.vstack(all_vertex_blackbody_color)

                    save_path = os.path.join(out_folder, f"mesh_{frame_counter:04d}.npz")
                    np.savez(save_path,
                             verts=combined_verts.astype(np.float32),
                             faces=combined_faces.astype(np.uint32),
                             normals=combined_normals.astype(np.float32),
                             halo_positions=combined_local_pos.astype(np.float32),
                             vertex_internal_energy=combined_vertex_ie.astype(np.float32),
                             halo_blackbody_color=combined_local_colors.astype(np.float32),
                             vertex_blackbody_color=combined_vertex_blackbody_color.astype(np.float32))
                    print(f"Saved mesh frame {frame_counter} to {save_path}")
                    frame_counter += 1

                    
                    
                    
                    
                    
if __name__ == "__main__":
    input_folder = "outs/"
    output_folder = "meshes_interpolated/"
    bounds_min, bounds_max, globalmin_en, globalmax_en = get_global_bounds(input_folder)

    process_with_interpolated_marching_cubes_cpu(
        input_folder,
        output_folder,
        bounds_min,
        bounds_max,
        globalmin_en,
        globalmax_en,
        grid_size=(300,300, 300),         # High-res per region
        coarse_grid_size=(10, 10, 10),  # Coarse global scan
        interp_steps=5,
        blur_sigma=0.8,
        padding_voxels=8                   # Padding to reduce continuity artifacts
)