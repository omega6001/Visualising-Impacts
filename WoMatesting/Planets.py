import woma
import numpy as np
import h5py

woma.load_eos_tables()
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2
# Example masses and radii of the target and impactor
M_t = 317.8 * M_earth
M_i = 95 * M_earth
R_t = 10.97 *R_earth
R_i = 9.13 *R_earth

# Mutual escape speed
v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i))

# Initial position and velocity of the target
A1_pos_t = np.array([0.0, 0.0, 0.0])
A1_vel_t = np.array([0.0, 0.0, 0.0])


# Set time until contact of xxxxx
# Instead: impact angle in degrees and speed in units of the escape speed
A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_r(
    b       = np.sin(0 * np.pi/180), 
    v_c     = v_esc*1.5, 
    r       = 30 * R_earth, 
    R_t     = R_t, 
    R_i     = R_i, 
    M_t     = M_t, 
    M_i     = M_i,
)



# Centre of mass
A1_pos_com = (M_t * A1_pos_t + M_i * A1_pos_i) / (M_t + M_i)
A1_pos_t = np.array(A1_pos_t, dtype=np.float64)
A1_pos_com = np.array(A1_pos_com, dtype=np.float64)
print(A1_pos_com,A1_pos_t)
A1_pos_t -= A1_pos_com
A1_pos_i -= A1_pos_com


# Centre of momentum
A1_vel_com = (M_t * A1_vel_t + M_i * A1_vel_i) / (M_t + M_i)
A1_vel_t = np.array(A1_vel_t, dtype=np.float64)
A1_vel_com = np.array(A1_vel_com, dtype=np.float64)
A1_vel_t -= A1_vel_com
A1_vel_i -= A1_vel_com

print(A1_pos_t / R_earth, "R_earth")
print(A1_vel_t, "m/s")
print(A1_pos_i / R_earth, "R_earth")
print(A1_vel_i, "m/s")

target = woma.Planet(
    A1_mat_layer    = ["ANEOS_iron","CD21_HHe"],
    A1_T_rho_type   = ["power=2","power=2"],
    P_s             = 4e6,
    T_s             = 180,
    M               = M_t,
    R               = R_t,
)

impactor = woma.Planet(
    A1_mat_layer    = ["ANEOS_iron"],
    A1_T_rho_type   = ["power=2"],
    P_s             = 1.4e6,
    T_s             = 140,
    M               = M_i,
)

# Generate the profiles
target.gen_prof_L2_find_R1_given_M_R(verbosity=1)
impactor.gen_prof_L1_find_R_given_M(R_max= 10*R_earth)






particles_t = woma.ParticlePlanet(target, 1e5, N_ngb = 48, verbosity = 1)
particles_i = woma.ParticlePlanet(impactor, 1e5, N_ngb = 48, verbosity = 1)

particles_t.A2_pos += A1_pos_t
particles_i.A2_pos += A1_pos_i

particles_t.A2_vel += A1_vel_t
particles_i.A2_vel += A1_vel_i
print(particles_t.A2_pos[122])

with h5py.File("jupiter_sat_2e5.hdf5","w") as f:
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
        boxsize=200*R_earth,
        file_to_SI=woma.Conversions(m=M_earth,l=R_earth,t=1)   
    )
