import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
import scipy.spatial

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

def pointcloud_to_grid(positions, densities, grid_size=(1000, 1000, 1000), global_min=None, global_max=None):
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
    material = gfx.MeshPhongMaterial(color=(1, 0.5, 0.2), shininess=50)
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

def process_with_marching_cubes(hdf5_folder, out_folder, bounds_min, bounds_max, grid_size=(1000, 1000, 1000)):
    os.makedirs(out_folder, exist_ok=True)

    for frame_idx, fname in enumerate(sorted(os.listdir(hdf5_folder))):
        if not fname.endswith('.hdf5'):
            continue

        path = os.path.join(hdf5_folder, fname)
        with h5py.File(path, 'r') as f:
            positions = f['PartType0/Coordinates'][:]
            densities = f['PartType0/Densities'][:]

        print(f"Processing {fname} (frame {frame_idx}) with {len(positions)} particles")

        grid = pointcloud_to_grid(
            positions, densities,
            grid_size=grid_size,
            global_min=bounds_min,
            global_max=bounds_max
        )

        if grid.max() <= 0:
            print("Skipping frame due to zero density")
            continue

        threshold = grid.max() * 0.008

        verts, faces, normals, _ = marching_cubes(grid, level=threshold)

        # Convert verts to world coordinates
        scale = (bounds_max - bounds_min) / (np.array(grid.shape) - 1)
        verts_world = verts * scale + bounds_min

        # Low-density particles → glow
        halo_mask = densities < threshold
        halo_positions = positions[halo_mask]

        # Save image
        output_file = os.path.splitext(fname)[0] + ".png"
        output_path = os.path.join(out_folder, output_file)

        render_mesh_with_halo(
            verts_world,
            faces,
            halo_positions,
            normals,
            output_path
        )

if __name__ == "__main__":
    input_folder = "outs/"
    output_folder = "frames_mesh1e6"
    bounds_min, bounds_max = get_global_bounds(input_folder)

    process_with_marching_cubes(input_folder, output_folder, bounds_min, bounds_max)

    # Create video
    output_video = 'output_mesh1e6_test.mp4'
    os.chdir(output_folder)

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '8',
        '-i', 'impact1e6_0%03d.png',
        '-vf', 'scale=1920:1080',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '../' + output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)
