import os
import h5py
import numpy as np
from skimage.measure import marching_cubes
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess

# ✅ Fixed camera settings (same for all frames)
campos = np.array([250, 250, 380], dtype=np.float32)
lookat = np.array([250, 250, 250], dtype=np.float32)

def get_global_density_range(folder):
    dmin, dmax = np.inf, -np.inf
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                d = f["PartType0/Densities"][:]
                dmin = min(dmin, d.min())
                dmax = max(dmax, d.max())
    return dmin, dmax

def get_global_bounds(folder):
    """Compute the global bounding box across all frames."""
    min_all = np.full(3, np.inf)
    max_all = np.full(3, -np.inf)

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                coords = f["PartType0/Coordinates"][:]
                min_all = np.minimum(min_all, coords.min(axis=0))
                max_all = np.maximum(max_all, coords.max(axis=0))
    return min_all, max_all

def pointcloud_to_grid(positions, densities, grid_size=(700, 700, 700), global_min=None, global_max=None):
    positions = np.array(positions)
    densities = np.array(densities)

    if global_min is None or global_max is None:
        global_min = positions.min(axis=0)
        global_max = positions.max(axis=0)

    # Normalize to global bounding box
    norm_pos = (positions - global_min) / (global_max - global_min + 1e-10)
    indices = (norm_pos * (np.array(grid_size) - 1)).astype(int)

    grid = np.zeros(grid_size, dtype=np.float32)
    for idx, dens in zip(indices, densities):
        x, y, z = idx
        grid[x, y, z] += dens
    return grid, global_min, global_max

def render_mesh(vertices, faces, output_path, img_size=(1920, 1080)):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Fixed camera
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    # Lighting
    light = gfx.DirectionalLight()
    light.local.position = (1, 1, 1)
    scene.add(light)

    # Mesh
    geometry = gfx.Geometry(
        positions=vertices.astype(np.float32),
        indices=faces.astype(np.uint32)
    )
    material = gfx.MeshPhongMaterial(color=(1, 0.5, 0.2), shininess=50)
    mesh = gfx.Mesh(geometry, material)
    scene.add(mesh)

    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())
    Image.fromarray(image_data, mode="RGBA").save(output_path)

def process_with_marching_cubes(hdf5_folder, out_folder, bounds_min, bounds_max, grid_size=(700, 700, 700)):
    os.makedirs(out_folder, exist_ok=True)

    for frame_idx, fname in enumerate(sorted(os.listdir(hdf5_folder))):
        if not fname.endswith('.hdf5'):
            continue

        path = os.path.join(hdf5_folder, fname)
        with h5py.File(path, 'r') as f:
            positions = f['PartType0/Coordinates'][:]
            densities = f['PartType0/Densities'][:]

        print(f"Processing {fname} (frame {frame_idx}) with {len(positions)} particles")

        grid, _, _ = pointcloud_to_grid(
            positions, densities,
            grid_size=grid_size,
            global_min=bounds_min,
            global_max=bounds_max
        )

        # Set threshold as 10% of max density in grid
        if grid.max() <= 0:
            print("Skipping frame due to zero density")
            continue
        threshold = grid.max() * 0.0005

        verts, faces, _, _ = marching_cubes(grid, level=threshold)

        # Transform back to world coordinates
        scale = (bounds_max - bounds_min) / (np.array(grid.shape) - 1)
        verts_world = verts * scale + bounds_min

        output_file = os.path.splitext(fname)[0] + ".png"
        render_mesh(verts_world, faces, os.path.join(out_folder, output_file))

if __name__ == "__main__":
    input_folder = "data/"
    output_folder = "frames_mesh"
    bounds_min, bounds_max = get_global_bounds(input_folder)

    process_with_marching_cubes(input_folder, output_folder, bounds_min, bounds_max)

    # Make MP4 from PNGs
    output_video = 'output_mesh.mp4'
    os.chdir(output_folder)

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '8',
        '-i', 'earth_impact_0%03d.png',
        '-vf', 'scale=1920:1080',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '../' + output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)
