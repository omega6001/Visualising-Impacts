import h5py
import numpy as np
from matplotlib import pyplot as plt

file_path = 'jupiter_sat_2e5.hdf5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as f:
    # Print all the groups and datasets at the root level
    # Example: Read a dataset (update 'your_dataset_name' with a real one)
    if 'PartType0/Coordinates' in f:
        data = f['PartType0/Coordinates'][:]
        
        
x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection = "3d")
#ax.set_xlim(45,55)
#ax.set_ylim(45,55)
#ax.set_zlim(45,55)
#ax.set_axis_off()


ax.scatter(x,y,z,color='black',s=0.01, marker= "o")
plt.show()