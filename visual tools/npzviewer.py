import os
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image


# Camera configuration
campos = np.array([320, 320, 420], dtype=np.float32)
lookat = np.array([320, 320, 320], dtype=np.float32)


def init_renderer(img_size=(1920, 1080)):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])
    camera.local.position = campos
    camera.look_at(lookat)

    return canvas, renderer, scene, camera


def render_particles_frame(
    canvas, renderer, scene, camera,
    all_positions, all_colors, output_path
):
    for obj in scene.children:
        scene.remove(obj)

    if all_positions is not None and len(all_positions) > 0:
        geometry = gfx.Geometry(
            positions=all_positions.astype(np.float32),
            colors=all_colors.astype(np.float32),
        )
        material = gfx.PointsMaterial(
            color_mode="vertex",
            size=2.5,
            opacity=1.0,
        )
        material.blending = "additive"
        material.depth_test = True
        material.depth_write = False

        particles = gfx.Points(geometry, material)
        scene.add(particles)

    renderer.render(scene, camera)
    image = canvas.draw()

    try:
        image_data = np.asarray(image)
        Image.fromarray(image_data, mode="RGBA").save(output_path)
        print(f"Saved particle frame to {output_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")


def main(mesh_folder="meshes_interpolated/", output_folder="rendered_particles/", limit=None):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.npz')])

    if limit is not None:
        files = files[:limit]

    canvas, renderer, scene, camera = init_renderer()

    for idx, filename in enumerate(files):
        output_file = f"particles_{idx:04d}.png"
        output_path = os.path.join(output_folder, output_file)
        print(f"Rendering particle frame {idx} from {filename}")
        data = np.load(os.path.join(mesh_folder, filename))
        all_positions = data.get("all_positions", None)
        all_colors = data.get("all_colors", None)

        if all_positions is None:
            print(f"Missing 'all_positions' in {filename}, skipping")
            continue

        if all_colors is None or len(all_colors) != len(all_positions):
            all_colors = np.full_like(all_positions, 0.8, dtype=np.float32)

        render_particles_frame(
            canvas, renderer, scene, camera,
            all_positions, all_colors, output_path
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_folder", type=str, default="meshes_interpolated/")
    parser.add_argument("--output_folder", type=str, default="rendered_particles/")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to render")
    args = parser.parse_args()

    main(
        mesh_folder=args.mesh_folder,
        output_folder=args.output_folder,
        limit=args.limit
    )