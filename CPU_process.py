import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import numba as nb

def match_particles(ids_a, ids_b):
    id_to_idx_b = {pid: idx for idx, pid in enumerate(ids_b)}
    matched_idx_a = []
    matched_idx_b = []

    for idx_a, pid in enumerate(ids_a):
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

def pointcloud_to_grid(positions, densities, grid_size=(1000, 1000, 1000), global_min=None, global_max=None):
    positions = np.array(positions)
    densities = np.array(densities)
    norm_pos = (positions - global_min) / (global_max - global_min + 1e-10)
    indices = (norm_pos * (np.array(grid_size) - 1)).astype(int)
    grid = np.zeros(grid_size, dtype=np.float32)
    for idx, dens in zip(indices, densities):
        x, y, z = idx
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
            grid[x, y, z] += dens
    return grid

def get_global_bounds(folder):
    min_all = np.full(3, np.inf)
    max_all = np.full(3, -np.inf)
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                coords = f["PartType0/Coordinates"][:]
                min_all = np.minimum(min_all, coords.min(axis=0))
                max_all = np.maximum(max_all, coords.max(axis=0))
    return min_all, max_all

def process_with_interpolated_marching_cubes_cpu(
    hdf5_folder, out_folder, bounds_min, bounds_max,
    grid_size=(500, 500, 500), interp_steps=5, blur_sigma=1.0
):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')])
    frame_counter = 0

    for frame_idx in range(len(files) - 1):
        fname_a = files[frame_idx]
        fname_b = files[frame_idx + 1]
        print(f"Interpolating between {fname_a} and {fname_b}")

        with h5py.File(os.path.join(hdf5_folder, fname_a), 'r') as fa, \
             h5py.File(os.path.join(hdf5_folder, fname_b), 'r') as fb:

            ids_a = fa['PartType0/ParticleIDs'][:]
            positions_a = fa['PartType0/Coordinates'][:]
            velocities_a = fa['PartType0/Velocities'][:]
            densities_a = fa['PartType0/Densities'][:]

            ids_b = fb['PartType0/ParticleIDs'][:]
            positions_b = fb['PartType0/Coordinates'][:]
            velocities_b = fb['PartType0/Velocities'][:]
            densities_b = fb['PartType0/Densities'][:]

            time_a = fa.attrs.get('Time', 0.0)
            time_b = fb.attrs.get('Time', 1.0)
            dt = time_b - time_a

            idx_a, idx_b = match_particles(ids_a, ids_b)

            pos_a_matched = positions_a[idx_a]
            pos_b_matched = positions_b[idx_b]
            vel_a_matched = velocities_a[idx_a]
            vel_b_matched = velocities_b[idx_b]
            dens_a_matched = densities_a[idx_a]
            dens_b_matched = densities_b[idx_b]

            for s in range(interp_steps):
                t = s / interp_steps
                interp_positions = hermite_interp_vec(pos_a_matched, pos_b_matched, vel_a_matched, vel_b_matched, t, dt)
                interp_densities = (1 - t) * dens_a_matched + t * dens_b_matched

                grid_interp = pointcloud_to_grid(interp_positions, interp_densities, grid_size=grid_size, global_min=bounds_min, global_max=bounds_max)
                grid_interp = gaussian_filter(grid_interp, sigma=blur_sigma)

                if grid_interp.max() <= 0:
                    print(f"Skipping interpolated frame {s} due to zero density")
                    continue

                threshold = grid_interp.max() * 0.001
                verts, faces, normals, _ = marching_cubes(grid_interp, level=threshold)

                scale = (bounds_max - bounds_min) / (np.array(grid_size) - 1)
                verts_world = verts * scale + bounds_min

                # Save mesh and halo data to disk as npz for GPU rendering
                save_path = os.path.join(out_folder, f"mesh_{frame_counter:04d}.npz")
                np.savez(save_path,
                         verts=verts_world.astype(np.float32),
                         faces=faces.astype(np.uint32),
                         normals=normals.astype(np.float32),
                         halo_positions=interp_positions.astype(np.float32))

                frame_counter += 1
                print(f"Saved mesh frame {frame_counter} to {save_path}")

if __name__ == "__main__":
    input_folder = "outs/"
    output_folder = "meshes_interpolated/"
    bounds_min, bounds_max = get_global_bounds(input_folder)

    process_with_interpolated_marching_cubes_cpu(
        input_folder,
        output_folder,
        bounds_min,
        bounds_max,
        grid_size=(1000, 1000, 1000),
        interp_steps=5,
        blur_sigma=1.2
    )
