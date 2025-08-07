import os
from PIL import Image
import subprocess
input_folder = "rendered_frames" #change input here


# Sort by file creation time or filename
frame_files = sorted(os.listdir(input_folder))     
os.chdir(input_folder)#input folder

ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "15",#framerate
    "-i", "frame_%04d.png",#input frames
    "-vf", "scale=1920:1080",#video resolution
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "../output_interptest.mp4" ###change the output name
]

subprocess.run(ffmpeg_cmd, check=True)
print("MP4 video created")
