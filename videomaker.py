import os
from PIL import Image
import subprocess
input_folder = "frames_interp"
output_folder = "frames_interp_renamed"
os.makedirs(output_folder, exist_ok=True)

# Sort by file creation time or filename
frame_files = sorted(os.listdir(input_folder))

#for i, fname in enumerate(frame_files):
    #if fname.endswith(".png"):
        #src = os.path.join(input_folder, fname)
        #dst = os.path.join(output_folder, f"frame_{i:04d}.png")
        #Image.open(src).save(dst)
        
os.chdir("frames_interp")

ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "15",
    "-i", "frame_interp%04d.png",
    "-vf", "scale=1920:1080",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "../output_interp.mp4"
]

subprocess.run(ffmpeg_cmd, check=True)
print("MP4 video created: output_interp.mp4")
