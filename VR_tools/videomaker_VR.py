import os
from PIL import Image
import subprocess

left_folder = "VR_tools/rendered_frames/left_eye"
right_folder = "VR_tools/rendered_frames/right_eye"
combined_folder = "VR_tools/rendered_frames/vr_sbs"
os.makedirs(combined_folder, exist_ok=True)

# Assumes filenames like frame_0000.png, frame_0001.png, ...
frame_files = sorted(os.listdir(left_folder))

# Combines left/right images side-by-side
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

# Render video using FFmpeg
os.chdir(combined_folder)

ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "36",  # 36 FPS
    "-i", "frame_%04d.png",  # Input SBS frames
    "-vf", "scale=7680:1920",  # Full SBS resolution
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "../../output_vr_sbs_7680x1920_36fps2.5.mp4" # name of output file
]
subprocess.run(ffmpeg_cmd, check=True)
print("VR (SBS) video created: output_vr_sbs5.mp4")