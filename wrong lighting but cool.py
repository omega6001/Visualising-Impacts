import os
import h5py
import numpy as np
import pygfx as gfx
import matplotlib.cm
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

# Camera and lighting setup
campos = np.array([320,320,400], dtype=np.float32)
lookat = np.array([320,320,320], dtype=np.float32)
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


from scipy.spatial import cKDTree
import numpy as np

def compute_smoothed_weighted_normals(positions, k=8, sigma=0.2):
    """
    Compute consistently oriented, weighted-smoothed normals using cross products and neighbors.
    Normals are oriented outward from each particle's local neighborhood center.
    """
    if len(positions) < 3:
        return np.zeros((len(positions), 3), dtype=np.float32)

    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)

    raw_normals = np.zeros_like(positions, dtype=np.float32)

    # Step 1: Raw normals from two nearest neighbors
    for i in range(len(positions)):
        p0 = positions[i]
        p1 = positions[indices[i][1]]
        p2 = positions[indices[i][2]]

        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n /= norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Local reference direction: from particle to neighbor centroid
        neighbor_centroid = positions[indices[i][1:]].mean(axis=0)
        to_centroid = p0 - neighbor_centroid  # want normals pointing *away* from cluster center
        if np.dot(n, to_centroid) < 0:
            n = -n

        raw_normals[i] = n

    # Step 2: Weighted smoothing using Gaussian falloff
    smoothed_normals = np.zeros_like(raw_normals)
    for i in range(len(positions)):
        neighbor_idxs = indices[i]
        neighbor_normals = raw_normals[neighbor_idxs]
        neighbor_dists = dists[i]

        weights = np.exp(- (neighbor_dists**2) / (2 * sigma**2))
        weighted = neighbor_normals * weights[:, None]

        n = np.sum(weighted, axis=0)
        norm = np.linalg.norm(n)
        smoothed_normals[i] = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return smoothed_normals


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
    normals = compute_smoothed_weighted_normals(positions, k=100, sigma=0.5)

    # Lighting calculation
    light = np.array(light_dir, dtype=np.float32)
    light /= np.linalg.norm(light)
    # Flip normals that point away from light
    dot_l = np.dot(normals, light)
    flip_mask = dot_l < 0
    normals[flip_mask] *= -1

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
    dmin, dmax = get_global_density_range("outs/")
    process_folder("outs/", "frames2")

    image_folder = './frames2'
    output_video = 'output_1e6_cool.mp4'
    os.chdir(image_folder)

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
