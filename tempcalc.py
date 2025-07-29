import numpy as np
from scipy.interpolate import interp1d
import h5py

def parse_aneos_file(filename):
    with open(filename, 'r') as f:
        densities = np.array(list(map(float, f.readline().split())))
        
        temperatures = np.array(list(map(float, f.readline().split())))

        n_density = len(densities)
        print(n_density)
        n_temp = len(temperatures)

        ie = []
        pressure = []
        velocity = []
        entropy = []

        for line in f:
            if line.strip() == '':
                continue
            vals = list(map(float, line.split()))
            if len(vals) != 4:
                raise ValueError(f"Unexpected number of columns in data line: {line}")
            ie_val, p_val, v_val, s_val = vals
            ie.append(ie_val)
            pressure.append(p_val)
            velocity.append(v_val)
            entropy.append(s_val)

        total_points = n_density * n_temp
        if len(ie) != total_points:
            raise ValueError(f"Expected {total_points} data lines, got {len(ie)}")

        ie = np.array(ie).reshape((n_density, n_temp))
        pressure = np.array(pressure).reshape((n_density, n_temp))
        velocity = np.array(velocity).reshape((n_density, n_temp))
        entropy = np.array(entropy).reshape((n_density, n_temp))

    return  densities, temperatures, ie, pressure, velocity, entropy


# Usage


def find_temperature_for_particle(rho_particle, ie_particle, densities, temperatures, ie_grid):
    # Find indices bracketing the particle density
    if rho_particle <= densities[0]:
        i_low = i_high = 0
    elif rho_particle >= densities[-1]:
        i_low = i_high = len(densities) - 1
    else:
        i_high = np.searchsorted(densities, rho_particle)
        i_low = i_high - 1

    rho_low, rho_high = densities[i_low], densities[i_high]

    # Interpolate temperature for ie_particle at rho_low
    ie_slice_low = ie_grid[i_low, :]
    # Make temp vs ie function (invert ie->temp)
    f_low = interp1d(ie_slice_low, temperatures, bounds_error=False, fill_value="extrapolate")
    temp_low = f_low(ie_particle)

    if i_low == i_high:
        # Particle density is exactly on grid, no need to interpolate density
        return temp_low

    # Interpolate temperature for ie_particle at rho_high
    ie_slice_high = ie_grid[i_high, :]
    f_high = interp1d(ie_slice_high, temperatures, bounds_error=False, fill_value="extrapolate")
    temp_high = f_high(ie_particle)

    # Interpolate temperature between low and high densities
    temp_particle = temp_low + (temp_high - temp_low) * (rho_particle - rho_low) / (rho_high - rho_low)

    return temp_particle

def find_temperatures_for_particles(rho_particles, ie_particles, densities, temperatures, ie_grid):
    temps = np.empty_like(rho_particles)
    for i, (rho, ie) in enumerate(zip(rho_particles, ie_particles)):
        temps[i] = find_temperature_for_particle(rho, ie, densities, temperatures, ie_grid)
    return temps


filename = "EoSTables/ANEOS_forsterite_S19.txt_cleaned.txt"
dens, temp, ie_grid, p_grid, v_grid, s_grid = parse_aneos_file(filename)
file_path = "outs/impact1e6_0003.hdf5"

with h5py.File(file_path,"r") as g:
    densities = g["PartType0/Densities"][:]
    internalenergies = g["PartType0/InternalEnergies"][:]
    matID = g["PartType0/MaterialIDs"][:]
    particleID = g["PartType0/ParticleIDs"][:]
    
    


temps = find_temperatures_for_particles(densities, internalenergies, dens,temp, ie_grid)
print(temps)
print(min(temps),max(temps))   