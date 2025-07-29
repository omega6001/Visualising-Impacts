import os
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import scipy.spatial

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


    # Add one persistent light
    light = gfx.DirectionalLight()
    light.local.position = (1, 1, 1)
    scene.add(light)

    return canvas, renderer, scene, camera

def render_frame(canvas, renderer, scene, camera, verts_world, faces, halo_positions, mesh_normals, output_path):
    # Preserving only the first object (e.g., light)
    for obj in scene.children[1:]:
        scene.remove(obj)
    # Add mesh
    geometry = gfx.Geometry(
        positions=verts_world.astype(np.float32),
        indices=faces.astype(np.uint32)
    )
    material = gfx.MeshPhongMaterial(color=(1, 0.5, 0.2), shininess=20)
    mesh = gfx.Mesh(geometry, material)
    scene.add(mesh)

    # Add halo glow points
    if len(halo_positions) > 0:
        tree = scipy.spatial.cKDTree(verts_world)
        _, nearest_indices = tree.query(halo_positions, k=1)
        matched_normals = mesh_normals[nearest_indices]

        light_dir = np.array([1, 1, 1], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(np.dot(matched_normals, light_dir), 0, 1)
        brightness = 0.3 + 1 * (diffuse ** 1.5)

        base_color = np.array([1.0, 0.6, 0.0], dtype=np.float32)
        glow_colors = (brightness[:, None] * base_color[None, :]).astype(np.float32)

        glow_geometry = gfx.Geometry(
            positions=halo_positions.astype(np.float32),
            colors=glow_colors
        )
        glow_material = gfx.PointsMaterial(
            color_mode="vertex",
            size=2.5,
            opacity=0.3,
        )
        glow_material.blending = "additive"
        glow_material.depth_test = True
        glow_material.depth_write = False
        
        glow = gfx.Points(glow_geometry, glow_material)
        scene.add(glow)

    # Render and save image
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

        output_file = f"frame_{idx:04d}.png"
        output_path = os.path.join(output_folder, output_file)

        render_frame(canvas, renderer, scene, camera, verts, faces, halo_positions, normals, output_path)

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
