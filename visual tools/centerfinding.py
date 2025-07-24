import numpy as np
import h5py
import hdbscan
import matplotlib.pyplot as plt





with h5py.File("outs/impact1e6_0000.hdf5",'r') as f:
    pos = f["PartType0/Coordinates"][:]
    
print(pos)
mags = np.linalg.norm(pos, axis = 1)
print("max")
print(max(mags),min(mags))
print(np.mean(mags))

plt.hist(mags, bins=100)
plt.title("Histogram of Distances from Origin")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.show()