import os
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import math
import pylinalg as la  # add this import near the top


def rotate_vec_x_neg90(v):
    # v is (3,) numpy array
    rx = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)
    return rx @ v

def init_renderer(campos, lookat, img_size=(3840, 1920)):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    root = gfx.Group()
    root.local.rotation = la.quat_from_euler((-np.pi / 2, 0, 0))  # rotate scene -90° X
    scene.add(root)

    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])

    camera.local.position = campos
    camera.look_at(lookat)

    return canvas, renderer, scene, camera, root

def chunk_data(verts, faces, colors, max_faces=100000):
    chunks = []
    num_faces = faces.shape[0]
    for i in range(0, num_faces, max_faces):
        f = faces[i:i+max_faces]
        unique_indices, inverse_indices = np.unique(f.flatten(), return_inverse=True)
        verts_chunk = verts[unique_indices]
        colors_chunk = colors[unique_indices]
        faces_chunk = inverse_indices.reshape((-1, 3))
        chunks.append((verts_chunk, faces_chunk, colors_chunk))
    return chunks

def render_glow_particles(root, positions, colors, chunk_size=200_000):
    for i in range(0, len(positions), chunk_size):
        pos_chunk = positions[i:i+chunk_size]
        col_chunk = colors[i:i+chunk_size]
        glow_geometry = gfx.Geometry(
            positions=pos_chunk.astype(np.float32),
            colors=col_chunk.astype(np.float32)
        )
        glow_material = gfx.PointsMaterial(
            color_mode="vertex",
            size=5.5,
            opacity=0.02,
        )
        glow_material.blending = "alpha"
        glow_material.depth_test = True
        glow_material.depth_write = True
        glow = gfx.Points(glow_geometry, glow_material)
        root.add(glow)

def render_frame(
    canvas, renderer, scene, camera, root,
    verts_world, faces, halo_positions, mesh_normals,
    vertex_colors, output_path, halo_colors=None
):
    # Clear previous objects inside root, not scene
    for obj in root.children[:]:
        root.remove(obj)

    mesh_chunks = chunk_data(verts_world, faces, vertex_colors)
    for verts_chunk, faces_chunk, colors_chunk in mesh_chunks:
        geometry = gfx.Geometry(
            positions=verts_chunk.astype(np.float32),
            indices=faces_chunk.astype(np.uint32),
            colors=colors_chunk.astype(np.float32)
        )
        material = gfx.MeshBasicMaterial(color_mode="vertex")
        mesh = gfx.Mesh(geometry, material)
        root.add(mesh)  # add to root

    if halo_positions is not None and len(halo_positions) > 0 and halo_colors is not None:
        if len(halo_positions) != len(halo_colors):
            print("Mismatch between halo_positions and halo_colors. Skipping glow.")
        else:
            render_glow_particles(root, halo_positions, halo_colors)  # add to root

    renderer.render(scene, camera)
    image = canvas.draw()

    try:
        image_data = np.asarray(image)
        Image.fromarray(image_data, mode="RGBA").save(output_path)
        print(f"Saved frame to {output_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

def main(mesh_folder="meshes_interpolated3/", output_base="rendered_frames3/", dx=10.0, limit=None):
    eyes = {
        "left_eye": -dx,
        "right_eye": dx
    }

    original_lookat = np.array([315, 315, 315], dtype=np.float32)

    for eye, offset in eyes.items():
        output_folder = os.path.join(output_base, eye)
        os.makedirs(output_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.npz')])
        if limit is not None:
            files = files[:limit]

        original_campos = np.array([315+offset,220,340], dtype=np.float32)
    
        # Rotate camera and lookat vectors to match root rotation
        campos = rotate_vec_x_neg90(original_campos)
        lookat = rotate_vec_x_neg90(original_lookat)

        canvas, renderer, scene, camera, root = init_renderer(campos, lookat)

        for idx, filename in enumerate(files):
            print(f"Rendering {eye} frame {idx} from {filename}")
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
                canvas, renderer, scene, camera, root,
                verts, faces, halo_positions, normals,
                vertex_colors, output_path, halo_colors
            )

if __name__ == "__main__":
    mesh_folder = "meshes_interpolated/"
    output_folder = "VR_tools/rendered_frames/"
    dx = 5
    limit = None

    main(mesh_folder, output_folder, dx, limit)