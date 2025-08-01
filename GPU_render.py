import os
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
from scipy.spatial import cKDTree


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


def render_frame(
    canvas, renderer, scene, camera,
    verts_world, faces, halo_positions, mesh_normals,
    vertex_colors, output_path,halo_colors=None
):
    # Clear previous objects (keep light)
    for obj in scene.children:
        scene.remove(obj)

    # --- Create mesh ---
    geometry = gfx.Geometry(
        positions=verts_world.astype(np.float32),
        indices=faces.astype(np.uint32),
        colors=vertex_colors.astype(np.float32)
    )
    material = gfx.MeshBasicMaterial(color_mode="vertex")
    mesh = gfx.Mesh(geometry, material)
    scene.add(mesh)

    # --- Add glow particles ---
    if halo_positions is not None and len(halo_positions) > 0 and halo_colors is not None:
        if len(halo_positions) != len(halo_colors):
            print("Mismatch between halo_positions and halo_colors. Skipping glow.")
        else:
            glow_geometry = gfx.Geometry(
                positions=halo_positions.astype(np.float32),
                colors=halo_colors.astype(np.float32)
            )
            glow_material = gfx.PointsMaterial(
                color_mode="vertex",
                size=4.5,
                opacity=0.1,
            )
            glow_material.blending = "additive"
            glow_material.depth_test = True
            glow_material.depth_write = False

            glow = gfx.Points(glow_geometry, glow_material)
            scene.add(glow)


    # --- Render and save image ---
    renderer.render(scene, camera)
    image = canvas.draw()

    try:
        image_data = np.asarray(image)
        Image.fromarray(image_data, mode="RGBA").save(output_path)
        print(f"Saved frame to {output_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")


def main(mesh_folder="meshes_interpolated/", output_folder="rendered_frames/", limit=None):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.npz')])

    if limit is not None:
        files = files[:limit]

    canvas, renderer, scene, camera = init_renderer()

    for idx, filename in enumerate(files):
        print(f"Rendering frame {idx} from {filename}")
        data = np.load(os.path.join(mesh_folder, filename))
        verts = data['verts']
        faces = data['faces']
        normals = data['normals']
        halo_positions = data['halo_positions']
        halo_colors = data.get('halo_blackbody_color', None)
        vertex_colors = data['vertex_blackbody_color']

        output_file = f"frame_{idx:04d}.png"
        output_path = os.path.join(output_folder, output_file)

        render_frame(
            canvas, renderer, scene, camera,
            verts, faces, halo_positions, normals,
            vertex_colors, output_path, halo_colors
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_folder", type=str, default="meshes_interpolated/")
    parser.add_argument("--output_folder", type=str, default="rendered_frames/")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to render")
    args = parser.parse_args()

    main(
        mesh_folder=args.mesh_folder,
        output_folder=args.output_folder,
        limit=args.limit
    )
