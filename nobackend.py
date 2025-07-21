import os
import numpy as np
import wgpu
import wgpu.utils
from wgpu.gui.offscreen import WgpuCanvas
from PIL import Image
import h5py
import asyncio



async def get_adapter():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    return adapter


# Camera matrix helpers
def look_at(eye, target, up):
    forward = (target - eye)
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    up_corrected = np.cross(side, forward)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = up_corrected
    m[2, :3] = -forward
    trans = np.identity(4, dtype=np.float32)
    trans[:3, 3] = -eye
    return m @ trans

def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return proj

def color_map(density):
    """Map density to RGBA color (simple grayscale for now)."""
    norm = (density - density.min()) / (np.ptp(density) + 1e-9)
    colors = np.stack([norm] * 3 + [np.ones_like(norm)], axis=-1)
    return colors.astype(np.float32)

def render_particles(positions, colors, output_path, width=1024, height=1024):
    canvas = WgpuCanvas(size=(width, height))
    adapter = asyncio.run(get_adapter())
    device = adapter.request_device()

    # Camera setup
    eye = np.array([350, 250, 250], dtype=np.float32)
    target = np.array([250, 250, 250], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    view = look_at(eye, target, up)
    proj = perspective(np.radians(45), width / height, 0.1, 1000.0)
    view_proj = (proj @ view).T  # Transpose for WGSL column-major

    # Create buffers
    pos_buffer = device.create_buffer_with_data(
        data=positions.astype(np.float32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    col_buffer = device.create_buffer_with_data(
        data=colors.astype(np.float32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    uniform_buffer = device.create_buffer_with_data(
        data=view_proj.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_code = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> pos: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> col: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> view_proj: mat4x4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(pos[index], 1.0);
    out.position = view_proj * world_pos;
    out.color = col[index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"""
    shader = device.create_shader_module(code=shader_code)

    # Create pipeline
    bind_group_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "uniform"}},
    ])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        {"binding": 0, "resource": {"buffer": pos_buffer}},
        {"binding": 1, "resource": {"buffer": col_buffer}},
        {"binding": 2, "resource": {"buffer": uniform_buffer}},
    ])
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    format = canvas.get_preferred_format()

    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": format}]},
        primitive={"topology": "point-list"},
    )

    # Create render target
    texture = device.create_texture(
        size=(width, height, 1),
        format=format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # Encode rendering
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(color_attachments=[{
        "view": texture_view,
        "clear_value": (0.0, 0.0, 0.0, 1.0),
        "load_op": "clear",
        "store_op": "store",
    }])
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind_group)
    render_pass.draw(len(positions), 1, 0, 0)
    render_pass.end()
    device.queue.submit([encoder.finish()])

    # Get image
    buffer = wgpu.utils.capture_texture(device, texture, (width, height))
    image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(output_path)
    print("Saved:", output_path)

def process_hdf5_folder(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".hdf5"))
    for i, fname in enumerate(files):
        print(f"[{i}] Processing {fname}")
        with h5py.File(os.path.join(folder, fname), "r") as f:
            pos = f["PartType0/Coordinates"][:]
            dens = f["PartType0/Densities"][:]
            colors = color_map(dens)
            render_particles(pos, colors, f"frame_{i:04d}.png")

# Run it
process_hdf5_folder("data/")
