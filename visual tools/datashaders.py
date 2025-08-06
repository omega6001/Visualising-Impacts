import h5py
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import os
import subprocess
from glob import glob
from matplotlib.colors import LinearSegmentedColormap

# === CONFIGURATION ===
HDF5_GLOB = "outs/*.hdf5"  # <-- UPDATE THIS
FRAME_DIR = "frames"
VIDEO_FILE = "output.mp4"
FRAMERATE = 10
Z_SLICE_WIDTH = 1080.0  # Large enough to skip slicing
PLOT_SIZE = 1920

# === COLOR MAP ===
bwr_cmap = LinearSegmentedColormap.from_list("bwr_custom", ["blue", "white", "red"])

# === CAMERA FUNCTIONS ===
def normalize(v):
    return v / np.linalg.norm(v)

def get_camera_matrix(eye, target, up):
    forward = normalize(eye - target)
    right = normalize(np.cross(up, forward))
    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=1)
    T = -R.T @ eye

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R.T
    view_matrix[:3, 3] = T
    return view_matrix

def project_points(coords, view_matrix):
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N,4)
    transformed = coords_h @ view_matrix.T
    return transformed[:, :2]  # (N,2)

# === DATA FUNCTIONS ===
def load_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as f:
        coords = f['PartType0/Coordinates'][:]  # (N, 3)
        densities = f['PartType0/Densities'][:]   # (N,)
    return coords, densities

def save_frames(file_list, output_dir, eye, target, up):
    os.makedirs(output_dir, exist_ok=True)
    view_matrix = get_camera_matrix(np.array(eye), np.array(target), np.array(up))

    for i, filepath in enumerate(sorted(file_list)):
        coords, densities = load_hdf5_data(filepath)

        # No slicing; full 3D
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'density': densities
        })

        projected = project_points(df[['x', 'y', 'z']].values, view_matrix)

        # Project the look-at point too — we want this to be at the center
        target_proj = project_points(np.array([target]), view_matrix)[0]
        center_x, center_y = target_proj

        df['x_proj'] = projected[:, 0] - center_x
        df['y_proj'] = projected[:, 1] - center_y

        cvs = ds.Canvas(plot_width=PLOT_SIZE, plot_height=PLOT_SIZE, x_range = (-50,50),y_range = (-50,50))
        agg = cvs.points(df, 'x_proj', 'y_proj', ds.mean('density'))

        img = tf.shade(agg, cmap=bwr_cmap, how='linear')
        img.to_pil().save(f"{output_dir}/frame_{i:04d}.png")
        print(f"Saved frame {i} from {os.path.basename(filepath)}")

# === VIDEO FUNCTION ===
def make_video(output_dir, output_file, framerate):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", f"{output_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file
    ]
    print("Encoding video with ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"Video saved to {output_file}")

# === MAIN ===
def main():
    hdf5_files = sorted(glob(HDF5_GLOB))
    if not hdf5_files:
        print("No HDF5 files found.")
        return

    eye = [320, 320, 400]
    target = [320, 320, 320]
    up = [0, 1, 0]

    save_frames(hdf5_files, FRAME_DIR, eye, target, up)
    make_video(FRAME_DIR, VIDEO_FILE, FRAMERATE)

if __name__ == "__main__":
    main()