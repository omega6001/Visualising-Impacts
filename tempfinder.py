import numpy as np
import h5py
from scipy.interpolate import interp1d

### === REGULAR-GRID PARSER FOR IRON AND FORSTERITE ===
def parse_regular_aneos(filename):
    with open(filename, 'r') as f:
        densities = np.array(list(map(float, f.readline().split())))
        temperatures = np.array(list(map(float, f.readline().split())))
        
        n_density = len(densities)
        n_temp = len(temperatures)
        
        # Read remaining lines — expecting n_density * n_temp lines, each with 4 values
        data = []
        for line in f:
            if not line.strip():
                continue
            vals = list(map(float, line.split()))
            if len(vals) != 4:
                raise ValueError("Expected 4 columns per data row")
            data.append(vals)
        
        data = np.array(data)
        
        if data.shape[0] != n_density * n_temp:
            raise ValueError(f"Expected {n_density*n_temp} data rows, got {data.shape[0]}")

        # Now reshape
        # Assuming order: for each temperature, all densities in order
        # So shape (n_temp, n_density, 4)
        data = data.reshape((n_temp, n_density, 4))
        
        # Extract internal energy grid shape (n_density, n_temp) for your interpolation
        ie_grid = data[:, :, 0].T  # transpose to get (n_density, n_temp)
        
    return densities, temperatures, ie_grid



### === TEMPERATURE CALCULATOR FOR REGULAR GRID ===
def find_temperature_for_particle_regular(rho_particle, ie_particle, densities, temperatures, ie_grid):
    # Find density bracket indices
    if rho_particle <= densities[0]:
        i_low = i_high = 0
    elif rho_particle >= densities[-1]:
        i_low = i_high = len(densities) - 1
    else:
        i_high = np.searchsorted(densities, rho_particle)
        i_low = i_high - 1
    
    rho_low, rho_high = densities[i_low], densities[i_high]
    
    def interp_temp_for_density(ie_row):
        # Sort IE and temperature to ensure monotonicity
        sorted_idx = np.argsort(ie_row)
        ie_sorted = ie_row[sorted_idx]
        temp_sorted = temperatures[sorted_idx]
        
        # Clamp ie_particle within ie grid range
        ie_clamped = np.clip(ie_particle, ie_sorted[0], ie_sorted[-1])
        
        f = interp1d(ie_sorted, temp_sorted, bounds_error=False,
                     fill_value=(temp_sorted[0], temp_sorted[-1]))
        return f(ie_clamped)
    
    temp_low = interp_temp_for_density(ie_grid[i_low])
    if i_low == i_high:
        return temp_low
    
    temp_high = interp_temp_for_density(ie_grid[i_high])
    
    # Linear interpolate temperature between density slices
    temp_particle = temp_low + (temp_high - temp_low) * (rho_particle - rho_low) / (rho_high - rho_low)
    
    T_MIN = 1.0  # Set minimum temperature threshold
    return max(temp_particle, T_MIN)


### === VECTORISED TEMPERATURE CALCULATOR BASED ON MATERIAL ID ===
def find_temperatures_with_material(densities, internal_energies, material_ids,
                                    eos_iron, eos_forsterite):
    temps = np.empty_like(densities)
    
    is_iron = material_ids == 400
    is_forsterite = material_ids == 401
    
    # Iron
    if np.any(is_iron):
        rho_iron = densities[is_iron]
        ie_iron = internal_energies[is_iron]
        dens_grid, temp_grid, ie_grid = eos_iron
        temps[is_iron] = np.array([
            find_temperature_for_particle_regular(rho, ie, dens_grid, temp_grid, ie_grid)
            for rho, ie in zip(rho_iron, ie_iron)
        ])
    
    # Forsterite
    if np.any(is_forsterite):
        rho_fort = densities[is_forsterite]
        ie_fort = internal_energies[is_forsterite]
        dens_grid, temp_grid, ie_grid = eos_forsterite
        temps[is_forsterite] = np.array([
            find_temperature_for_particle_regular(rho, ie, dens_grid, temp_grid, ie_grid)
            for rho, ie in zip(rho_fort, ie_fort)
        ])
    
    return temps


### === MAIN EXECUTION ===

# File paths
filename_iron = "EoSTables/ANEOS_iron_S20.txt_cleaned.txt"
filename_fort = "EoSTables/ANEOS_forsterite_S19.txt_cleaned.txt"
file_path = "outs/impact1e6_0003.hdf5"

# Parse EOS tables
eos_iron = parse_regular_aneos(filename_iron)
eos_forsterite = parse_regular_aneos(filename_fort)

# Load particle data from HDF5
with h5py.File(file_path, "r") as g:
    densities = g["PartType0/Densities"][:]
    internalenergies = g["PartType0/InternalEnergies"][:]
    matID = g["PartType0/MaterialIDs"][:]

# Compute temperatures
temps = find_temperatures_with_material(densities, internalenergies, matID,
                                        eos_iron, eos_forsterite)

print(f"Temperature range: min={temps.min()}, max={temps.max()}, mean={temps.mean()}")
