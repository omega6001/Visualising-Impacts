import h5py
import numpy as np
from matplotlib import pyplot as plt

file_path = 'init_impact_1e6.hdf5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as f:
    # Print all the groups and datasets at the root level
    # Example: Read a dataset (update 'your_dataset_name' with a real one)
    data = f['PartType0/Pressures'][:]
    
print(data)
        
        

