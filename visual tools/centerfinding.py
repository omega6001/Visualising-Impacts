import numpy as np
import h5py
import matplotlib.pyplot as plt
import time


#this code is helpful for visualising your first file so you know camera postions, lookat vectors etc


with h5py.File("outs/impact1e6_0000.hdf5",'r') as f:
    pos = f["PartType0/Coordinates"][:]
    
    
x = pos[:][0]
y = pos[:][1]
z = pos[:][2]  
    
    

print(pos)
print(np.mean(pos[:][0]),np.mean(pos[:][1]),np.mean(pos[:][2]), "pos")
mags = np.linalg.norm(pos, axis = 1)
print("max")
print(max(mags),min(mags))
print(np.mean(mags))

plt.hist(mags, bins=200)
plt.title("Histogram of Distances from Origin")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.show()
time.sleep(7)
