import os
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
from scipy.spatial import cKDTree
import h5py

# Camera configuration
with h5py.File("outs/impact1e6_0000.hdf5",'r') as f:
    pos = f["PartType0/Coordinates"][:]
    
    
dz = 20 # distance along z axis to view from
campos = np.array([np.mean(pos[:][0]),np.mean(pos[:][1]),np.mean(pos[:][2])+dz], dtype=np.float32)
lookat = np.array([np.mean(pos[:][0]),np.mean(pos[:][1]),np.mean(pos[:][2])], dtype=np.float32)
#centers camera to look at mean position (this will tend to be inside the larger body)

def init_renderer(img_size=(1920, 1080)):
    canvas = WgpuCanvas(size=img_size, pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()

    camera = gfx.PerspectiveCamera(60, img_size[0] / img_size[1])#defines camera with 60FOV and an aspect ratio (16x9) currently
    camera.local.position = campos
    camera.look_at(lookat)

    return canvas, renderer, scene, camera

def chunk_data(verts, faces, colors, max_faces=100000): # splits the data into chunks to prevent an error that occurs with larger files
    chunks = []
    num_faces = faces.shape[0]
    for i in range(0, num_faces, max_faces):
        f = faces[i:i+max_faces]
        # Get unique vertices used in this face batch
        unique_indices, inverse_indices = np.unique(f.flatten(), return_inverse=True)
        verts_chunk = verts[unique_indices]
        colors_chunk = colors[unique_indices]
        faces_chunk = inverse_indices.reshape((-1, 3))
        chunks.append((verts_chunk, faces_chunk, colors_chunk))
    return chunks

def render_glow_particles(scene, positions, colors, chunk_size=200_000):
    for i in range(0, len(positions), chunk_size):
        pos_chunk = positions[i:i+chunk_size]
        col_chunk = colors[i:i+chunk_size]
        #defines the halo particles
        glow_geometry = gfx.Geometry(
            positions=pos_chunk.astype(np.float32),
            colors=col_chunk.astype(np.float32)
        )
        #keeping a low opacity and a larger size makes the particles seem more like a glowy effect
        glow_material = gfx.PointsMaterial(
            color_mode="vertex",
            size=5.5,
            opacity=0.02,
        )
        glow_material.blending = "additive"
        glow_material.depth_test = True
        glow_material.depth_write = True
        glow = gfx.Points(glow_geometry, glow_material)
        scene.add(glow)

def render_frame(
    canvas, renderer, scene, camera,
    verts_world, faces, halo_positions, mesh_normals,
    vertex_colors, output_path,halo_colors=None
):
    # Clear scene
    for obj in scene.children:
        scene.remove(obj)

    mesh_chunks = chunk_data(verts_world, faces, vertex_colors)
    for verts_chunk, faces_chunk, colors_chunk in mesh_chunks:
        #define the mesh colours and vertices
        geometry = gfx.Geometry(
            positions=verts_chunk.astype(np.float32),
            indices=faces_chunk.astype(np.uint32),
            colors=colors_chunk.astype(np.float32)
        )
        material = gfx.MeshBasicMaterial(color_mode="vertex")#basic mesh allows for more custom lighting
        mesh = gfx.Mesh(geometry, material)
        scene.add(mesh)
    # --- Add glow particles ---
    if halo_positions is not None and len(halo_positions) > 0 and halo_colors is not None:
        if len(halo_positions) != len(halo_colors):
            print("Mismatch between halo_positions and halo_colors. Skipping glow.")
        else:
            render_glow_particles(scene, halo_positions, halo_colors)


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
    parser.add_argument("--mesh_folder", type=str, default="meshes_interpolated/")#input folder
    parser.add_argument("--output_folder", type=str, default="rendered_frames/")#output folder
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to render")
    args = parser.parse_args()

    main(
        mesh_folder=args.mesh_folder,
        output_folder=args.output_folder,
        limit=args.limit
    )
