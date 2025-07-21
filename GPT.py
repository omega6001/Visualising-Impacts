import os
import h5py
import numpy as np
import pygfx as gfx
import matplotlib.cm
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import os
import subprocess


campos = (250,250,350)
light_dir = (0,1,0)


def normalize_array(arr):
    return (arr - arr.min()) / (np.ptp(arr) + 1e-8)


def render_and_save(positions, densities, output_path, img_size=(1920, 1080)):
    # Setup canvas and renderer
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # Scene
    scene = gfx.Scene()
    scene.add(gfx.DirectionalLight((light_dir)))

    # Normalize densities and map to RGB colors
    norm_dens = normalize_array(densities)
    cmap = matplotlib.cm.get_cmap("magma")
    colors = cmap(norm_dens)[:, :3].astype(np.float32)  # RGB only

    # Create geometry
    geometry = gfx.Geometry(positions=positions.astype(np.float32), colors=colors)
    material = gfx.PointsMaterial(color_mode = "vertex", size=2.0)
    points = gfx.Points(geometry, material)
    scene.add(points)

    # Camera setup
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at((250, 250, 250))

    # Render once
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())  # shape (H, W, 4), dtype uint8

    # Convert to PIL Image and save
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
    # Example usage:
    process_folder("data/", "frames")
    image_folder = './frames'  # folder where PNGs are stored
    output_video = 'output_wgpu.mp4'

# Change to the folder where the images are
    os.chdir(image_folder)

    # Construct ffmpeg command
    # Assumes images are named like frame_000.png, frame_001.png, etc.
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '8',             # frames per second
        '-i', 'earth_impact_0%03d.png',
        '-vf', 'scale=1920:1080',      # input filename pattern
        '-c:v', 'libx264',              # use H.264 codec
        '-pix_fmt', 'yuv420p',          # pixel format for wide compatibility
        '../'+ output_video            # output file (saved one level up)
    ]

    # Run the command
    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)
