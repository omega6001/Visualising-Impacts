import pygfx as gfx
import h5py
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run



campos = (250,250,350)
light_dir = (0,1,0)


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.DirectionalLight((light_dir)))


with h5py.File("data/earth_impact_0000.hdf5",'r') as f:
    pos = f["PartType0/Coordinates"][:].astype(np.float32)
    dens = f["PartType0/Densities"][:].astype(np.float32)

N = len(dens)
densemax = max(dens)
col = []
for x in range(0,N):
    frac = dens[x]/densemax.astype(np.float32)
    col =np.append(col,(0.8,0.5-frac,0,1))
    
geometry = gfx.Geometry(positions=pos, sizes= np.ones(N).astype(np.float32), colors=col)
material = gfx.PointsMaterial(color_mode="vertex", size_mode="vertex")
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(60, 16 / 9)
camera.local.position = campos
camera.look_at((250, 250, 250))


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()