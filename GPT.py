import os
import h5py
import numpy as np
import pygfx as gfx
import matplotlib.cm
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
from scipy.cluster.vq import kmeans2

# Camera and lighting setup
campos = np.array([250, 250, 400], dtype=np.float32)
lookat = np.array([250, 250, 250], dtype=np.float32)
light_dir = (-1, 1, -1)

def get_global_density_range(folder):
    """Scan all frames to determine global density min/max for consistent normalization."""
    dmin, dmax = np.inf, -np.inf
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                d = f["PartType0/Densities"][:]
                dmin = min(dmin, d.min())
                dmax = max(dmax, d.max())
    return dmin, dmax

def normalize_array_global(arr, dmin, dmax, contrast=0.7):
    norm = (arr - dmin) / (dmax - dmin + 1e-11)
    return np.clip(norm**contrast, 0, 1)

def compute_fake_normals_from_clusters(positions, k=1):
    """Assign normal vectors from particle to its cluster center."""
    if len(positions) == 0 or k < 1:
        return np.zeros((len(positions), 3), dtype=np.float32)

    k = min(k, len(positions))  # avoid k > n
    centroids, labels = kmeans2(positions.astype(np.float32), k, minit='points')

    vectors = centroids[labels] - positions
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0

    return (vectors / norms).astype(np.float32)

def render_and_save(positions, densities, output_path, img_size=(1920, 1080), dmin=None, dmax=None, frame_idx=0):
    # Create canvas and renderer
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Normalize and color-map densities
    norm_dens = np.clip(((densities - dmin) / (dmax - dmin + 1e-12)) ** 0.9, 0, 1)
    cmap = matplotlib.cm.get_cmap("autumn")
    base_colors = cmap(norm_dens)[:, :3].astype(np.float32)
#################
    base_colors *= 1.3  # Brighten base colors slightly
    base_colors = np.clip(base_colors, 0, 1)
    # Choose k clusters based on frame number
    if frame_idx <= 7:
        k=2
    elif frame_idx > 7 and frame_idx <45:
        k = 1
    else:
        k=1
    normals = compute_fake_normals_from_clusters(positions, k=k)

    # Lighting calculation
    light = np.array(light_dir, dtype=np.float32)
    light /= np.linalg.norm(light)

    diffuse = np.clip(np.dot(normals, light), 0, 1)
    ambient = 0.1
    brightness = ambient + (1 - ambient) * diffuse**1.5
    shaded_colors = base_colors * brightness[:, None]
    shaded_colors = np.clip(shaded_colors, 0, 1).astype(np.float32)

    # Camera
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    # Sort for proper blending
    distances = np.linalg.norm(positions - campos, axis=1)
    sort_idx = np.argsort(distances)[::-1]
    sorted_positions = positions[sort_idx]
    sorted_colors = shaded_colors[sort_idx]

    # Geometry + rendering
    geometry = gfx.Geometry(positions=sorted_positions.astype(np.float32), colors=sorted_colors)
    material = gfx.PointsMaterial(color_mode="vertex", size=1.0)
    points = gfx.Points(geometry, material)
    scene.add(points)

    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())
    Image.fromarray(image_data, mode="RGBA").save(output_path)

def process_folder(hdf5_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    dmin, dmax = get_global_density_range(hdf5_folder)

    for frame_idx, fname in enumerate(sorted(os.listdir(hdf5_folder))):
        if not fname.endswith('.hdf5'):
            continue
        path = os.path.join(hdf5_folder, fname)
        with h5py.File(path, 'r') as f:
            positions = f['PartType0/Coordinates'][:]
            densities = f['PartType0/Densities'][:]

        print(f"Rendering {fname} (frame {frame_idx}) with {len(positions)} particles")
        output_file = os.path.splitext(fname)[0] + ".png"

        render_and_save(positions, densities, os.path.join(out_folder, output_file),
                        dmin=dmin, dmax=dmax, frame_idx=frame_idx)

if __name__ == "__main__":
    dmin, dmax = get_global_density_range("data/")
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
