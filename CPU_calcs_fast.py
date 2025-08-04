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
                energy = f["PartType0/InternalEnergies"][:]
                min_en = min(min_en, energy.min())
                max_en = max(max_en, energy.max())
    min_all = np.array([100,200,150])
    max_all= np.array([450,450,400])            
    return min_all, max_all, min_en, max_en


def process_with_interpolated_marching_cubes_cpu(
    hdf5_folder, out_folder, bounds_min, bounds_max, globalmin_en, globalmax_en,
    grid_size=(500, 500, 500), interp_steps=5, blur_sigma=1.2
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
                halo_temperature_kelvin = interp_ie * 1e9  # Use your existing scaling
                halo_colors = get_blackbody_rgb_numba(halo_temperature_kelvin)
                gamma = 0.8  # Adjust to taste; <1.0 boosts midtones
                halo_colors = np.clip(halo_colors ** gamma, 0, 1)

                dens_grid = pointcloud_to_grid_numba(interp_pos, interp_dens, grid_size, bounds_min, bounds_max)
                dens_grid = gaussian_filter(dens_grid, sigma=blur_sigma)
                if dens_grid.max() <= 0:
                    print(f"Skipping interpolated frame {s} due to zero density")
                    continue

                threshold = dens_grid.max() * 0.001
                print(dens_grid.max(),dens_grid.min(), np.mean(dens_grid))
                verts, faces, normals, _ = marching_cubes(dens_grid, level=threshold)

                scale = (bounds_max - bounds_min) / (np.array(grid_size) - 1)
                verts_world = verts * scale + bounds_min
                mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
                smoothed_mesh = laplacian_smooth(mesh, iterations=10, lam=0.4)

                # Replace with smoothed vertices
                verts_world = smoothed_mesh.vertices
                faces = smoothed_mesh.faces  # should be the same, but included for robustness

                energy_grid = pointcloud_to_grid_numba(interp_pos, interp_ie, grid_size, bounds_min, bounds_max)
                energy_grid = gaussian_filter(energy_grid, sigma=blur_sigma)

                norm_verts = verts / (np.array(grid_size) - 1)
                verts_scaled = norm_verts * (bounds_max - bounds_min) + bounds_min
                grid_coords = ((verts_scaled - bounds_min) / (bounds_max - bounds_min + 1e-8) * (np.array(grid_size) - 1)).astype(int)

                vertex_ie = energy_grid[
                    grid_coords[:, 0],
                    grid_coords[:, 1],
                    grid_coords[:, 2]
                ]

                # === Density gradient (for shading) ===
                density_grad = np.stack(np.gradient(dens_grid), axis=-1)  # shape: (X, Y, Z, 3)
                grad_at_verts = density_grad[
                    grid_coords[:, 0],
                    grid_coords[:, 1],
                    grid_coords[:, 2]
                ]   

                grad_mag = np.linalg.norm(grad_at_verts, axis=1) + 1e-12
                density_at_verts = dens_grid[
                    grid_coords[:, 0],
                    grid_coords[:, 1],
                    grid_coords[:, 2]
                ]

                # === Weight from density gradient + density
                weight = np.clip(2000 * (density_at_verts - 0.001), 0.0, 1.0)
                weight[grad_mag < 1e-6] = 0.0

                # === Mix shading normal: blend between mesh normal and gradient normal
                normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
                grad_normals_unit = grad_at_verts / grad_mag[:, None]
                shading_normals = (1.0 - weight[:, None]) * normals_unit + weight[:, None] * grad_normals_unit
                shading_normals /= np.linalg.norm(shading_normals, axis=1, keepdims=True)

                # === Lighting
                light_dir = np.array([1.0, -1.0, 1.0])
                light_dir = light_dir / np.linalg.norm(light_dir)
                view_dir = np.array([0.0, 0.0, -1.0])  # Assuming camera looking along Z

                diffuse = np.clip(np.einsum('ij,j->i', shading_normals, -light_dir), 0.0, 1.0)

                # === Specular
                reflect_dir = 2 * np.einsum('ij,j->i', shading_normals, light_dir)[:, None] * shading_normals - light_dir
                specular = np.clip(np.einsum('ij,j->i', reflect_dir, view_dir), 0.0, 1.0) ** 12
                specular *= 0.6

                # === Base color
                base_color = np.full((verts.shape[0], 3), 0.3, dtype=np.float32)

                # === Apply lighting
                lit_color = base_color * diffuse[:, None] + specular[:, None] * np.array([1.0, 1.0, 1.0])

                # === Emission (from blackbody)
                temperature_kelvin = vertex_ie *(1e8) #assumes cube grid(equal boxes of L,W,H) (grid_size[0]**3)*1.4625
                ####1e8 for 400x400x400
                print(np.max(temperature_kelvin),np.min(temperature_kelvin), np.mean(temperature_kelvin))
                raw_emission = get_blackbody_rgb_numba(temperature_kelvin)
                emission_strength = np.clip(temperature_kelvin / 8000.0, 0.0, 1.0) ** 2
                blackbody_emission = raw_emission * emission_strength[:, None]

                #   === Combine lit + emission
                raw_color = lit_color * 1.0 + blackbody_emission  # combine lit and emission colors
                raw_color = np.clip(raw_color, 0.0, 1.0)

                brightness_scale = 1.2  # tweak this value for desired brightness

                vertex_blackbody_color = scale_colors_keep_ratio_numba(raw_color, brightness_scale)

                save_path = os.path.join(out_folder, f"mesh_{frame_counter:04d}.npz")
                np.savez(save_path,
                         verts=verts_world.astype(np.float32),
                         faces=faces.astype(np.uint32),
                         normals=normals.astype(np.float32),
                         halo_positions=interp_pos.astype(np.float32),
                         vertex_internal_energy=vertex_ie.astype(np.float32),
                         halo_blackbody_color=halo_colors.astype(np.float32),
                         vertex_blackbody_color=vertex_blackbody_color.astype(np.float32))
                frame_counter += 1
                print(f"Saved mesh frame {frame_counter} to {save_path}")


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
        grid_size=(400, 400, 400),
        interp_steps=1,
        blur_sigma=0.8
    )
