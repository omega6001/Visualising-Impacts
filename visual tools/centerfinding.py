import numpy as np
import h5py
import matplotlib.pyplot as plt





with h5py.File("outs2/pE_T20_Th_T20_B45v100_1e7_000000.hdf5",'r') as f:
    pos = f["PartType0/Coordinates"][:]
    
print(pos)
mags = np.linalg.norm(pos, axis = 1)
print("max")
print(max(mags),min(mags))
print(np.mean(mags))

plt.hist(mags, bins=200)
plt.title("Histogram of Distances from Origin")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.show()