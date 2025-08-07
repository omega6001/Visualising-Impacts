Code made by Thomas Power (sbkz64@durham.ac.uk) ,(after June 2026 email thomasp7000@gmail.com for any questions)
this is designed to visualise planetary impacts calculated by swift SPH simulations by renderiing both a mesh and halo particles together
this also provides tools to render the simulations into VR 4k video to be run on a virtual reality headset. 
all of the code is written in python using the module pygfx as a wrapper for webGPU which itself is a wrapper for Vulkan and hence any rendering using this will need Vulkan support.
Inside the python environment you will need the following modules: pygfx, numba, woma, trimesh, h5py, scipy, skimage-measure, pillow, wgpu, datashader (maybe more but the error should be good enough to figure out)
  
examples:
<img width="3840" height="1920" alt="frame_0196" src="https://github.com/user-attachments/assets/66919f16-a2ae-4f16-b036-daa689342d6c" />



https://github.com/user-attachments/assets/aab62db3-b1c7-4787-9ee0-4ed4da2a2a00



https://github.com/user-attachments/assets/c0e855c8-bd04-4751-9171-211e4fcc37d7



    
Included in this repository is:  
CPU_program.py  
GPU_render.py  
videomaker.py  
visual tools  
VR_tools  
WoMatesting  

CPU_program.py :  
this is the main body of calculations in the program, this takes the data from the hdf5 files and interpolates between frames, calculates colours, generates the mesh vertices using marching cubes, includes halo particles
then exports all of the data needed to a .npz file for GPU_render.py to use. The way the space is split up into a grid is in two main steps. A first inital rough pass is done on a coarse grid, this detects which cells are active
and which ones arent including enough particles for us to be bothered to render. Then each active cell is split up further into a much finer grid for use in marching cubes algorith to generate a smoother mesh. There are a few
constants in this code that could need some tweaking depending on your input file. I'd recommend running this on a batch queue on a HPC node as this could take a long time depending on how many interpolated frames you want/grid size.
You will need to define the grid bounds you want to use for your render as finding the apropriate bounds was tricky from the data files themselves due to rogue particles. If there appears to be artifacts over the fine grid cell boundaries, you can increase the number of padding cells and this should help, also increasing blur_sigma can help this.  
    
GPU_render.py :  
takes in the .npz files and renders all of the data into mesh and halo particles. Due to memory limits in the offscreen renderer, the data is split into chunks to be rendered piece by piece into the same .png. In this file you can
change the viewpoint and the opacities, material properties etc. This is designed to run on a GPU node on a HPC server, however this can also run using modern CPU's but this is much slower.  

videomaker.py :  
This takes the .png files from GPU_render.py and stitches them together to form a video using ffmpeg. This runs faster on a CPU system however the difference can be small for smaller videos.  
  
visual tools :  
these are a set of various bits of code I've used a few times to help with general bugfixing, finding necessary camera positions etc. This also includes datashaders.py which uses the datashade python module to render a basic 2D animation
from the data files (note this doesnt include interpolation). This is useful to give a brief look at what you are first trying to get and to see through the materials and track densities (coloured by density).  

VR_tools :  
This contains GPU_render_VR.py and videomaker_VR.py. These work almost identical to their non VR versions. GPU_render_VR.py takes extra arguments for the eye seperation for the VR video and renders frames for both right and left eye.
it is worth noting that you need to run CPU_program.py to get the necessary .npz files for this to work, however this can cause erros to occur with the specular lighting as this is calculated for viewing along Z axis, this error is however minimal.
This code currently stores the .png's as a 3840x1920 size.  This code also rotates the rednered frame so that the the z axis points up and you are looking along (and a bit above) the plane of the collision for dramatic effects for VR.
videomaker_VR.py works almost the exact same as its non VR version, however it has an extra process where it sticthes the 2 sets of right and left eye PNGs together to form a larger combined PNG (7680 x 1920).

WoMatesting :  
This includes one file with an example of how my initial conditions were set up for a collision between Uranus and Earth.  

  
Method for general use:  
when running this I would first get the general placement of the bodies I want to collide (these varied between my data sets) and choose my own bounds for the grid in CPU_program.py (tend on the generous side). I would also think of where you want to be viewing this from, the WoMa initial conditions maker sets the collisions to occur along the x axis. You would ideally want to set the view direction in CPU_program.py to match for the most realistic lighting. You can also vary the direction of the light.
  
The general structure of commands is intended to be as follows:  
setup IC's -> run swift -> run CPU_program.py (most likely on batch queue ~10hrs for 600 frames) -> ssh GPU cluster -> run GPU_render.py -> ssh back to login node -> run videomaker.py  
  
However these can be combined together to run on one massive batch queue command, this still technically works and could be easier for some. Using the split code is easier to bug fix and runs more optimally.



Ideas for future work:  
The Hysteresis mechanic designed to stop particles flickering between states of mesh and halo needs fine tuning over a few runs especially for more particles.  
testing the optimal eye distance for the VR videos is hard without access to a VR headset.  
The halo particles and mesh colours so far are treated differently, and as such when they transition from one to another this can be visually odd.  
There will be a method somehow on how to create the grid efficiently and automatically.


