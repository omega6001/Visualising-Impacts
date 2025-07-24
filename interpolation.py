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

# Fixed camera settings
campos = np.array([320, 320, 420], dtype=np.float32)
lookat = np.array([320, 320, 320], dtype=np.float32)#(250,250,250)

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

def pointcloud_to_grid(positions, densities, grid_size=(400, 400, 400), global_min=None, global_max=None):
    positions = np.array(positions)
    densities = np.array(densities)

    norm_pos = (positions - global_min) / (global_max - global_min + 1e-10)
    indices = (norm_pos * (np.array(grid_size) - 1)).astype(int)

    grid = np.zeros(grid_size, dtype=np.float32)
    for idx, dens in zip(indices, densities):
        x, y, z = idx
        grid[x, y, z] += dens
    return grid

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
    material = gfx.MeshPhongMaterial(color=(1, 0.5, 0.2), shininess=100)
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
    brightness = 0.2 + 1 * (diffuse ** 1.5)

    base_color = np.array([1.0, 0.6, 0.0], dtype=np.float32)
    glow_colors = (brightness[:, None] * base_color[None, :]).astype(np.float32)

    # Halo glow
    glow_geometry = gfx.Geometry(
        positions=halo_positions.astype(np.float32),
        colors=glow_colors
    )
    glow_material = gfx.PointsMaterial(
        color_mode="vertex",
        size=2.5,
        opacity=0.3,
    )
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

def process_with_marching_cubes_interp_blur(
    hdf5_folder,
    out_folder,
    bounds_min,
    bounds_max,
    grid_size=(600, 600, 600),
    interp_steps=4,
    blur_sigma=1.0
):
    os.makedirs(out_folder, exist_ok=True)
    frame_files = [f for f in sorted(os.listdir(hdf5_folder)) if f.endswith('.hdf5')]

    for i in range(len(frame_files) - 1):
        fname_a = frame_files[i]
        fname_b = frame_files[i + 1]

        # Load frame A
        with h5py.File(os.path.join(hdf5_folder, fname_a), 'r') as f:
            positions_a = f['PartType0/Coordinates'][:]
            densities_a = f['PartType0/Densities'][:]

        grid_a = pointcloud_to_grid(positions_a, densities_a, grid_size, bounds_min, bounds_max)

        # Load frame B
        with h5py.File(os.path.join(hdf5_folder, fname_b), 'r') as f:
            positions_b = f['PartType0/Coordinates'][:]
            densities_b = f['PartType0/Densities'][:]

        grid_b = pointcloud_to_grid(positions_b, densities_b, grid_size, bounds_min, bounds_max)

        # Generate interpolated + blurred grids
        for s, grid_interp in enumerate(interpolate_voxel_grids(grid_a, grid_b, interp_steps)):
            if grid_interp.max() <= 0:
                continue

            # Apply Gaussian blur
            blurred_grid = gaussian_filter(grid_interp, sigma=blur_sigma)

            if blurred_grid.max() <= 0:
                continue

            threshold = blurred_grid.max() * 0.001
            verts, faces, normals, _ = marching_cubes(blurred_grid, level=threshold)

            scale = (bounds_max - bounds_min) / (np.array(grid_size) - 1)
            verts_world = verts * scale + bounds_min

            output_file = f"{os.path.splitext(fname_a)[0]}_interp{s:02d}.png"
            output_path = os.path.join(out_folder, output_file)

            render_mesh_with_halo(
                verts_world,
                faces,
                positions_a,  # Optional: still using original positions for halo
                normals,
                output_path
            )

    print("Interpolation with blur complete.")

if __name__ == "__main__":
    input_folder = "outs/"
    output_folder = "frames_mesh1e6"
    bounds_min, bounds_max = get_global_bounds(input_folder)

    process_with_marching_cubes_interp_blur(
        input_folder,
        output_folder,
        bounds_min,
        bounds_max,
        grid_size=(600, 600, 600),
        interp_steps=4,      # Increase for smoother animation
        blur_sigma=1.2       # Tune blur as needed (0.5–2.0 range works well)
    )

    # Create video
    output_video = 'output_mesh1e6.mp4'
    os.chdir(output_folder)

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '5',
        '-i', 'impact1e6_0%03d.png',
        '-vf', 'scale=1920:1080',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '../' + output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)
