import woma
import numpy as np
import h5py
####WoMa file for generating initial conditions in swift simulations
#currently set up as a collision between uranus and earth
#included as an example if wanted
woma.load_eos_tables()
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2
# Example masses and radii of the target and impactor
M_t = 0.887 * M_earth
M_i = 0.133 * M_earth
R_t = 0.96 * R_earth
R_i = 0.57 * R_earth

# Mutual escape speed
v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i))

# Initial position and velocity of the target
A1_pos_t = np.array([0., 0., 0.])
A1_vel_t = np.array([0., 0., 0.])


#Set initial separation of 4 Earth radii
# Default inputs: dimensionless impact parameter and SI speed
A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_t(
    b           = 45,
    v_c         = 1, 
    t           = 10000, 
    R_t         = R_t, 
    R_i         = R_i, 
    M_t         = M_t, 
    M_i         = M_i,
    units_b     = "B", 
    units_v_c   = "v_esc",
)



# Centre of mass
A1_pos_com = (M_t * A1_pos_t + M_i * A1_pos_i) / (M_t + M_i)
A1_pos_t -= A1_pos_com
A1_pos_i -= A1_pos_com


# Centre of momentum
A1_vel_com = (M_t * A1_vel_t + M_i * A1_vel_i) / (M_t + M_i)
A1_vel_t -= A1_vel_com
A1_vel_i -= A1_vel_com

print(A1_pos_t / R_earth, "R_earth")
print(A1_vel_t, "m/s")
print(A1_pos_i / R_earth, "R_earth")
print(A1_vel_i, "m/s")

target = woma.Planet(
    A1_mat_layer    = ["HM80_rock", "HM80_ice", "HM80_HHe"],
    A1_T_rho_type   = ["power=0", "power=0.9", "adiabatic"],
    P_s             = 1e5,
    T_s             = 70,
    M               = 14.5 * M_earth,
    A1_R_layer      = [None, 3 * R_earth, 4 * R_earth],
)

# Generate the profiles

impactor = woma.Planet(
    A1_mat_layer    = ["ANEOS_Fe85Si15", "ANEOS_forsterite"],
    A1_T_rho_type   = ["entropy=1500", "adiabatic"],
    P_s             = 1e5,
    T_s             = 1000,
    M               = M_earth,
    R               = R_earth
)

# Generate the profiles
target.gen_prof_L3_find_R1_given_M_R_R2()

impactor.gen_prof_L2_find_R1_given_M_R()
particles_t = woma.ParticlePlanet(target, 3e5, N_ngb = 48, verbosity = 0)
particles_i = woma.ParticlePlanet(impactor, 1e5, N_ngb = 48, verbosity = 0)

particles_t.A2_pos += A1_pos_t
particles_i.A2_pos += A1_pos_i

particles_t.A2_vel += A1_vel_t
particles_i.A2_vel += A1_vel_i
print(particles_t.A2_pos[122])

with h5py.File("impact_5e5_ice_v15_b45.hdf5","w") as f:
    woma.save_particle_data(
        f,
        np.append(particles_t.A2_pos, particles_i.A2_pos,axis = 0),
        np.append(particles_t.A2_vel, particles_i.A2_vel,axis = 0),
        np.append(particles_t.A1_m, particles_i.A1_m),
        np.append(particles_t.A1_h, particles_i.A1_h),
        np.append(particles_t.A1_rho, particles_i.A1_rho),
        np.append(particles_t.A1_P, particles_i.A1_P),
        np.append(particles_t.A1_u, particles_i.A1_u), 
        np.append(particles_t.A1_mat_id, particles_i.A1_mat_id),
        boxsize=100*R_earth,
        file_to_SI=woma.Conversions(m=M_earth,l=R_earth,t=1)   
    )
