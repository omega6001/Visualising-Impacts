import h5py
import numpy as np
from matplotlib import pyplot as plt
#just a small section of code to read the data in the hdf5 file (can be useful)
file_path = 'init_impact_1e6.hdf5'

with h5py.File(file_path, 'r') as f:
    data = f['PartType0/Pressures'][:]
    
print(data)
        
        

