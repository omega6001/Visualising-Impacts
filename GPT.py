import os
import h5py
import numpy as np
import pygfx as gfx
import matplotlib.cm
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import subprocess
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

campos = np.array([250, 250, 400], dtype=np.float32)
lookat = np.array([250, 250, 250], dtype=np.float32)
light_dir = (-1, 1, -1)

def get_global_density_range(folder):
    dmin, dmax = np.inf, -np.inf
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder, fname), "r") as f:
                d = f["PartType0/Densities"][:]
                dmin = min(dmin, d.min())
                dmax = max(dmax, d.max())
    return dmin, dmax

def normalize_array_global(arr, dmin, dmax, contrast=1.0):
    norm = (arr - dmin) / (dmax - dmin + 1e-11)
    return np.clip(norm**contrast, 0, 1)

def cluster_in_3d_with_merging(positions, max_clusters=3, merge_thresh=5.0):
    # Step 1: KMeans in 3D space
    kmeans = KMeans(n_clusters=max_clusters, n_init="auto", random_state=42)
    initial_labels = kmeans.fit_predict(positions)

    # Step 2: Get initial centers
    centers = np.array([positions[initial_labels == i].mean(axis=0)
                        for i in range(max_clusters)])

    # Step 3: Merge close centers
    merged = []
    used = np.zeros(len(centers), dtype=bool)

    for i in range(len(centers)):
        if used[i]:
            continue
        close_group = [centers[i]]
        for j in range(i+1, len(centers)):
            if used[j]:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < merge_thresh:
                close_group.append(centers[j])
                used[j] = True
        merged.append(np.mean(close_group, axis=0))
        used[i] = True

    final_centers = np.array(merged)

    # Step 4: Reassign all points to nearest merged center
    tree = cKDTree(final_centers)
    _, final_labels = tree.query(positions)

    return final_labels, final_centers

def compute_normals_from_cluster_centers(positions, labels, centers):
    """Return normalized vectors from particle to its assigned cluster center."""
    cluster_vecs = centers[labels] - positions
    norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (cluster_vecs / norms).astype(np.float32)

def render_and_save(positions, densities, output_path, img_size=(1920, 1080), dmin=None, dmax=None, n_clusters=2):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    # Normalize and stretch contrast of densities
    norm_dens = normalize_array_global(densities, dmin, dmax, contrast=0.7)
    cmap = matplotlib.colormaps.get_cmap("autumn")
    base_colors = cmap(norm_dens)[:, :3].astype(np.float32)

    # Lighting normals from radial distance clustering
    labels, centers = cluster_in_3d_with_merging(positions, max_clusters=3, merge_thresh=18.0)
    print(f"Detected {len(centers)} merged centers this frame")

    normals = compute_normals_from_cluster_centers(positions, labels, centers)

    # Lighting calculation
    light = np.array(light_dir, dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(np.dot(normals, light), 0, 1)

    ambient = 0.15
    brightness = ambient + (1 - ambient) * diffuse**1.2
    gamma = 0.8  # Slight nonlinear enhancement of light contrast
    brightness = brightness ** gamma

    shaded_colors = np.clip(base_colors * brightness[:, None], 0, 1).astype(np.float32)

    # Camera
    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    # Sort by distance to camera
    distances = np.linalg.norm(positions - campos, axis=1)
    sort_idx = np.argsort(distances)[::-1]
    sorted_positions = positions[sort_idx]
    sorted_colors = shaded_colors[sort_idx]

    geometry = gfx.Geometry(positions=sorted_positions.astype(np.float32), colors=sorted_colors)
    material = gfx.PointsMaterial(color_mode="vertex", size=1.0)
    points = gfx.Points(geometry, material)
    scene.add(points)

    # Render and save
    canvas.request_draw(lambda: renderer.render(scene, camera))
    image_data = np.asarray(canvas.draw())
    Image.fromarray(image_data).save(output_path)

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

        print(f"Rendering {fname} with {len(positions)} particles")
        output_file = os.path.splitext(fname)[0] + ".png"

        # Dynamic number of clusters
        if frame_idx < 10:
            n_clusters = 2
        else:
            n_clusters = 2

        render_and_save(positions, densities, os.path.join(out_folder, output_file),
                        dmin=dmin, dmax=dmax, n_clusters=n_clusters)

if __name__ == "__main__":
    dmin, dmax = get_global_density_range("data/")
    process_folder("data/", "frames")

    image_folder = './frames'
    output_video = 'output_test.mp4'

    os.chdir(image_folder)
    ffmpeg_cmd = [
        'ffmpeg', '-framerate', '8',
        '-i', 'earth_impact_0%03d.png',
        '-vf', 'scale=1920:1080',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '../' + output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print("MP4 video created:", output_video)
