import os
import h5py
import numpy as np
import pygfx as gfx
import matplotlib.cm
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
import pylinalg as la


campos = np.array([250, 250, 400], dtype=np.float32)
lookat = np.array([250, 250, 250], dtype=np.float32)
light_dir = (0, 1, 0)

def get_global_density_range(folder):
    dmin, dmax = np.inf, -np.inf
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                d = f["PartType0/Densities"][:]
                dmin = min(dmin, d.min())
                dmax = max(dmax, d.max())
    return dmin, dmax


def normalize_array_global(arr, dmin, dmax, contrast=1.5):
    norm = (arr - dmin) / (dmax - dmin + 1e-11)
    return np.clip(norm**contrast, 0, 1)


def get_depth_sorted(positions, colors, camera_matrix):
    """Return positions and colors sorted by view-space depth (Z)."""
    # Convert to homogeneous coordinates
    pos_h = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float32)], axis=1)

    # Transform to camera/view space
    view_pos = pos_h @ camera_matrix.T  # shape: (N, 4)

    # Use negative Z (depth) for sorting — we want farthest first
    depth = -view_pos[:, 2]
    sort_idx = np.argsort(depth)[::-1]  # farthest to nearest

    return positions[sort_idx], colors[sort_idx]


def render_and_save(positions, densities, output_path, img_size=(1920, 1080)):
    # Setup canvas and renderer
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # Scene setup
    scene = gfx.Scene()
    scene.add(gfx.DirectionalLight(light_dir))

    # Normalize densities (full range) with contrast boost
    
    norm_dens = normalize_array_global(densities, dmin, dmax, contrast=0.3)
    cmap = matplotlib.cm.get_cmap("inferno")
    colors = cmap(norm_dens)[:, :3].astype(np.float32)  # RGB only

    # Camera setup
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    # Sort particles by distance from camera (farthest to nearest)
    distances = np.linalg.norm(positions - campos, axis=1)
    sort_idx = np.argsort(distances)[::-1]
    sorted_positions = positions[sort_idx]
    sorted_colors = colors[sort_idx]

    # Create geometry and material
    geometry = gfx.Geometry(positions=sorted_positions.astype(np.float32), colors=sorted_colors)
    material = gfx.PointsMaterial(color_mode="vertex", size=1.0)
    points = gfx.Points(geometry, material)
    scene.add(points)

    # Render once
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())  # (H, W, 4) uint8 array

    # Save as PNG with PIL
    image = Image.fromarray(image_data, mode="RGBA")
    image.save(output_path)


def process_folder(hdf5_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    for fname in sorted(os.listdir(hdf5_folder)):
        if not fname.endswith('.hdf5'):
            continue
        path = os.path.join(hdf5_folder, fname)
        with h5py.File(path, 'r') as f:
            positions = f['PartType0/Coordinates'][:]  # shape: (N, 3)
            densities = f['PartType0/Densities'][:]   # shape: (N,)

        print(f"Rendering {fname} with {len(positions)} particles")
        output_file = os.path.splitext(fname)[0] + ".png"
        render_and_save(positions, densities, os.path.join(out_folder, output_file))


if __name__ == "__main__":
    dmin,dmax = get_global_density_range("data/")
    process_folder("data/", "frames")
    image_folder = './frames'
    output_video = 'output_wgpu.mp4'

    os.chdir(image_folder)

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
