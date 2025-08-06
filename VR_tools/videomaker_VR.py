import os
from PIL import Image
import subprocess

left_folder = "VR tools/rendered_frames/left_eye"
right_folder = "VR tools/rendered_frames/right_eye"
combined_folder = "VR tools/rendered_frames/vr_sbs"
os.makedirs(combined_folder, exist_ok=True)

# Assumes filenames like frame_0000.png, frame_0001.png, ...
frame_files = sorted(os.listdir(left_folder))

# Step 1: Combine left/right images side-by-side
for fname in frame_files:
    if fname.endswith(".png"):
        left_path = os.path.join(left_folder, fname)
        right_path = os.path.join(right_folder, fname)

        if not os.path.exists(right_path):
            print(f"Skipping {fname}: missing right eye image.")
            continue

        left_img = Image.open(left_path)
        right_img = Image.open(right_path)

        # Ensure both are same size
        w, h = left_img.size
        right_img = right_img.resize((w, h))

        # Create side-by-side (SBS) image
        combined = Image.new("RGBA", (w * 2, h))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (w, 0))

        combined.save(os.path.join(combined_folder, fname))

print("Combined stereo image sequence saved.")

# Step 2: Render video using FFmpeg
os.chdir(combined_folder)

ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "10",  # You can adjust frame rate
    "-i", "frame_%04d.png",
    "-vf", "scale=3840:1080",  # 2x width for side-by-side 1080p per eye
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "../output_vr_sbs50.mp4"
]

subprocess.run(ffmpeg_cmd, check=True)
print("VR (SBS) video created: output_vr_sbs50.mp4")