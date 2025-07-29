import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
import scipy.spatial
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import numba as nb

# Fixed camera settings
campos = np.array([320, 320, 420], dtype=np.float32)
lookat = np.array([320, 320, 320], dtype=np.float32)#(250,250,250)


def pointcloud_to_grid(positions, densities, grid_size=(500, 500, 500), global_min=None, global_max=None):
    positions = np.array(positions)
    densities = np.array(densities)

    norm_pos = (positions - global_min) / (global_max - global_min + 1e-10)
    indices = (norm_pos * (np.array(grid_size) - 1)).astype(int)

    grid = np.zeros(grid_size, dtype=np.float32)
    for idx, dens in zip(indices, densities):
        x, y, z = idx
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

def hermite_interp_vec(p0, p1, v0, v1, t, dt):
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    # p0, p1, v0, v1 are (N,3), so operate on transpose and transpose back for broadcasting
    return (h00 * p0.T + h10 * dt * v0.T + h01 * p1.T + h11 * dt * v1.T).T

def match_particles(ids_a, ids_b):
    """
    Return indices to match particles from frame A to frame B by ID.
    For unmatched particles, ignore or handle accordingly.
    """
    # Create dict for fast lookup of id in frame B
    id_to_idx_b = {pid: idx for idx, pid in enumerate(ids_b)}
    matched_idx_a = []
    matched_idx_b = []

    for idx_a, pid in enumerate(ids_a):
        if pid in id_to_idx_b:
            matched_idx_a.append(idx_a)
            matched_idx_b.append(id_to_idx_b[pid])

    return np.array(matched_idx_a), np.array(matched_idx_b)

@nb.njit(parallel=True)
def sph_density_grid_fast(
    positions, masses, smoothing_lengths,
    grid, grid_min, grid_max
):
    nx, ny, nz = grid.shape
    dx = (grid_max - grid_min) / np.array([nx - 1, ny - 1, nz - 1])

    for i in nb.prange(len(positions)):
        px, py, pz = positions[i]
        m = masses[i]
        h = smoothing_lengths[i]
        h_inv = 1.0 / h
        h_inv3 = h_inv ** 3

        # Compute kernel support radius (2h) in grid units
        r_grid = (2 * h / dx).astype(nb.int32)

        # Compute center grid index
        ix = int((px - grid_min[0]) / dx[0])
        iy = int((py - grid_min[1]) / dx[1])
        iz = int((pz - grid_min[2]) / dx[2])

        for x in range(max(ix - r_grid[0], 0), min(ix + r_grid[0] + 1, nx)):
            for y in range(max(iy - r_grid[1], 0), min(iy + r_grid[1] + 1, ny)):
                for z in range(max(iz - r_grid[2], 0), min(iz + r_grid[2] + 1, nz)):
                    gx = grid_min[0] + x * dx[0]
                    gy = grid_min[1] + y * dx[1]
                    gz = grid_min[2] + z * dx[2]
                    rx = px - gx
                    ry = py - gy
                    rz = pz - gz
                    r = (rx**2 + ry**2 + rz**2) ** 0.5

                    q = r * h_inv
                    if q <= 2.0:
                        if q <= 1.0:
                            w = (1 - 1.5 * q**2 + 0.75 * q**3)
                        else:
                            w = 0.25 * (2 - q)**3
                        w *= h_inv3
                        grid[x, y, z] += m * w

    return grid

def sph_density_grid_wrapper(
    positions, masses, smoothing_lengths,
    grid_size=(500, 500, 500),
    global_min=None, global_max=None
):
    grid = np.zeros(grid_size, dtype=np.float32)
    return sph_density_grid_fast(
        np.array(positions, dtype=np.float32),
        np.array(masses, dtype=np.float32),
        np.array(smoothing_lengths, dtype=np.float32),
        grid,
        np.array(global_min, dtype=np.float32),
        np.array(global_max, dtype=np.float32)
    )

def render_mesh_with_halo(verts_world, faces, halo_positions, mesh_normals, output_path, img_size=(1920, 1080)):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Camera
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    # Light
    light = gfx.DirectionalLight()
    light.local.position = (1, 1, 1)
    scene.add(light)

    # Mesh
    geometry = gfx.Geometry(
        positions=verts_world.astype(np.float32),
        indices=faces.astype(np.uint32)
    )
    material = gfx.MeshPhongMaterial(color=(1, 0.5, 0.2), shininess=20)
    mesh = gfx.Mesh(geometry, material)
    scene.add(mesh)

    # Nearest mesh vertex for each halo point
    tree = scipy.spatial.cKDTree(verts_world)
    _, nearest_indices = tree.query(halo_positions, k=1)
    matched_normals = mesh_normals[nearest_indices]

    # Lighting for halo particles
    light_dir = np.array([1, 1, 1], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.clip(np.dot(matched_normals, light_dir), 0, 1)
    brightness = 0.3 + 1 * (diffuse ** 1.5)

    base_color = np.array([1.0, 0.6, 0.0], dtype=np.float32)
    glow_colors = (brightness[:, None] * base_color[None, :]).astype(np.float32)

    # Halo glow
    if len(halo_positions) != 0:
        glow_geometry = gfx.Geometry(
            positions=halo_positions.astype(np.float32),
            colors=glow_colors
        )
        glow_material = gfx.PointsMaterial(
            color_mode="vertex",
            size=2.5,
            opacity=0.3,
        )####################potentially remove halo particles to increase grid size?#####################D
        glow_material.blending = "additive"
        glow_material.depth_test = True
        glow_material.depth_write = False

        glow = gfx.Points(glow_geometry, glow_material)
        scene.add(glow)

    # Render and save
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())
    Image.fromarray(image_data, mode="RGBA").save(output_path)

def interpolate_voxel_grids(grid_a, grid_b, steps=4):
    """Yield interpolated voxel grids between two frames."""
    for i in range(steps):
        t = i / steps
        yield (1 - t) * grid_a + t * grid_b

def process_with_interpolated_marching_cubes(
    hdf5_folder, out_folder, bounds_min, bounds_max,
    grid_size=(1200, 1200, 1200), interp_steps=5, blur_sigma=1.0
):
    global p
    os.makedirs(out_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')])

    for frame_idx in range(len(files) - 1):
        fname_a = files[frame_idx]
        fname_b = files[frame_idx + 1]

        print(f"Interpolating between {fname_a} and {fname_b}")

        # Load frames A and B
        with h5py.File(os.path.join(hdf5_folder, fname_a), 'r') as fa, \
             h5py.File(os.path.join(hdf5_folder, fname_b), 'r') as fb:

            ids_a = fa['PartType0/ParticleIDs'][:]        # Assuming this dataset exists
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

            # Match particles by ID
            idx_a, idx_b = match_particles(ids_a, ids_b)

            # Extract matched particles
            pos_a_matched = positions_a[idx_a]
            pos_b_matched = positions_b[idx_b]
            vel_a_matched = velocities_a[idx_a]
            vel_b_matched = velocities_b[idx_b]
            dens_a_matched = densities_a[idx_a]
            dens_b_matched = densities_b[idx_b]

            for s in range(0,interp_steps):
                t = s / (interp_steps)
                print(s)
                

             # Hermite interpolate positions and velocities for matched particles
                interp_positions = hermite_interp_vec(
                    pos_a_matched, pos_b_matched,
                    vel_a_matched, vel_b_matched,
                    t, dt
                )

                # Linear interpolate densities (or optionally hermite if you have density derivatives)
                interp_densities = (1 - t) * dens_a_matched + t * dens_b_matched

                # Build the SPH grid from these interpolated particles only
                grid_interp = pointcloud_to_grid(
                    interp_positions, interp_densities,
                    grid_size=grid_size,
                    global_min=bounds_min,
                    global_max=bounds_max
                )

                # Optional Gaussian blur
                grid_interp = gaussian_filter(grid_interp, sigma=blur_sigma)

                # Skip if empty
                if grid_interp.max() <= 0:
                    print(f"Skipping interpolated frame {s} due to zero density")
                    continue

                threshold = grid_interp.max() * 0.001
                verts, faces, normals, _ = marching_cubes(grid_interp, level=threshold)

                scale = (bounds_max - bounds_min) / (np.array(grid_size) - 1)
                verts_world = verts * scale + bounds_min

                #output_file = f"{os.path.splitext(fname_a)[0]}_interp{s:02d}.png"
                output_file = f"frame_interp{p:04d}.png"
                output_path = os.path.join(out_folder, output_file)

                render_mesh_with_halo(
                    verts_world,
                    faces,
                    interp_positions,
                    normals,
                    output_path
                )
                p=p+1

global p
p=0
if __name__ == "__main__":
    input_folder = "outs/"
    output_folder = "frames_interp/"
    bounds_min, bounds_max = get_global_bounds(input_folder)

    process_with_interpolated_marching_cubes(
        "outs/", "frames_interp", bounds_min, bounds_max,
        grid_size=(500, 500, 500),
        interp_steps=5,
        blur_sigma=1.2
    )   

    # Create video
    output_video = 'output_mesh1e6_interp.mp4'
    os.chdir(output_folder)

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '20',
        '-i', 'frame_0%03d.png',
        '-vf', 'scale=1920:1080',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '../' + output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)