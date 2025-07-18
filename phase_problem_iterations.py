import numpy as np
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm
from scipy.special import erf, erfc
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #

Tm      = 54.        # *C # melting temperature
T1      = 35.        # *C # right boundary temperature
L       = 2.1e5      # J/kg
rho_l   = 780.       # kg/m^37
rho_s   = 860.       # kg/m^3
lam_l   = 0.15       # W/K m
lam_s   = 0.24       # W/K m
c_l     = 2100.      # J/kg K
c_s     = 2900.      # J/kg K
w       = 50e-3      # m # simulated from 0 to w distance
h_air   = 2.66       # W/m^2 K
R       = 7e-3/2     # m

a_s = lam_s / (rho_s * c_s)
a_l = lam_l / (rho_l * c_l)

# ---------------------------------------------------------------------------- #
#                                   Formula's                                  #
# ---------------------------------------------------------------------------- #

# ---------------------------- Discrete formula's ---------------------------- #

# the fraction of liquid in the mushy zone
@njit
def theta_l(T, dT):
    return (T - Tm + dT) / 2 / dT

# full_theta_l the total fraction liquid in all three zones
@njit
def full_theta_l(T, dT):
    # different zones
    mush_mask = (Tm - dT < T) & (T < Tm + dT)
    solid_mask = T <= Tm - dT
    liquid_mask = T >= Tm + dT

    theta = np.zeros_like(T)
    theta[solid_mask] = 0
    theta[liquid_mask] = 1
    theta[mush_mask] = theta_l(T[mush_mask], dT)
    return theta

# apparent heat capacity
@njit
def c_A(T, dT):
    return (1 - theta_l(T, dT)) * rho_s * c_s \
        + theta_l(T, dT) * rho_l * c_l \
        + ((rho_l * c_l - rho_s * c_s) * T + rho_l * L) / 2 / dT

# discrete timestep of liquid zone
@njit
def dT_liquid_loss(T, dt):
    return 2 * dt * h_air / (rho_l * c_l * R) * (T - T1)

@njit
def dT_liquid(T, dt, dx):
    return a_l * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1]) - dT_liquid_loss(T[1:-1], dt)

# discrete timestep of solid zone
@njit
def dT_solid_loss(T, dt):
    return 2 * dt * h_air / (rho_s * c_s * R) * (T - T1)

@njit
def dT_solid(T, dt, dx):
    return a_s * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1]) - dT_solid_loss(T[1:-1], dt)

# mushy zone heat conductivity
@njit
def lam_phi(T, dT):
    return full_theta_l(T, dT) * lam_l + (1 - full_theta_l(T, dT)) * lam_s

# discrete timestep of mushy zone
@njit
def dT_mush_loss(T, dT, dt):
    return 2 * dt * h_air \
        / (((1 - theta_l(T, dT)) * rho_s * c_s + theta_l(T, dT) * rho_l * c_l) * R) * (T - T1)

@njit
def dT_mush(T, dT, dt, dx):
    return dt / (2 * dx**2 * c_A(T[1:-1], dT)) * ((lam_phi(T[2:], dT) + lam_phi(T[1:-1], dT)) * T[2:] \
        + (lam_phi(T[:-2], dT) + lam_phi(T[1:-1], dT)) * T[:-2] \
        - (lam_phi(T[2:], dT) + lam_phi(T[:-2], dT) + 2 * lam_phi(T[1:-1], dT)) * T[1:-1]) \
        - dT_mush_loss(T[1:-1], dT, dt)

# ---------------------------------------------------------------------------- #
#                                  Simulation                                  #
# ---------------------------------------------------------------------------- #

# generator for timesteps of the temperature
# see https://en.wikipedia.org/wiki/Generating_function for definition of generator functions
# or https://youtu.be/5jwV3zxXc8E for python generator functions tutorial
@njit
def solve_equation(x, T0, dT, dt, dx):
    T = np.full(x.size, T1)
    T[0] = T0 # boundary condition
    T_new = T.copy()

    while True:
        # differentiations of zones
        mush_mask = (Tm - dT < T) & (T < Tm + dT)
        solid_mask = T <= Tm - dT
        liquid_mask = T >= Tm + dT
        
        # mushy zone if there is also a liquid zone and a minimum amount of mush
        if np.any(mush_mask):
            left = np.where(mush_mask)[0][0] # find start of mushy zone
            right = np.where(mush_mask)[0][-1] # find end of mushy zone
            T_new[left:right+1] = T[left:right+1] + dT_mush(T[left-1:right+2], dT, dt, dx) # update mushy zone
        else: # otherwise add mushy zone to solid zone
            solid_mask |= mush_mask

        if liquid_mask.sum() > 1: # liquid zone starts at 1. 0 is considered fixed boundary
            left = 1 # starts at 1
            right = np.where(liquid_mask)[0][-1] # right side of liquid zone
            T_new[left:right+1] = T[left:right+1] + dT_liquid(T[left-1:right+2], dt, dx) # update liquid zone

        left = max(np.where(solid_mask)[0][0], 1) # find start of solid zone
        right = len(T)-2 # solid zone end 1 from the last cell
        T_new[left:right+1] = T[left:right+1] + dT_solid(T[left-1:right+2], dt, dx) # update solid zone

        # T_new[-1] = T_new[-2] # uncomment voor geïsoleerd

        T, T_new = T_new, T # swap T and T_new
        yield T # return T iteration

# loading generator just saving frames that are viewed
@njit(nogil=True)
def load_frames(frames, x, N, T0, dT, dt, dx, progress):
    solve_equation_gen = solve_equation(x, T0, dT, dt, dx) # load the generator
    T_list = np.empty((frames.size, x.size))
    T = np.empty(x.size)
    
    frame = 0
    for i in range(N):
        T = next(solve_equation_gen)

        if frame < len(frames) and i == frames[frame]: # if the iteration is going to be shown
            T_list[frame] = T # save the iteration
            frame += 1

        progress.update(1) # update progressbar
    
    return T_list

# ---------------------------------------------------------------------------- #
#                              Running simulation                              #
# ---------------------------------------------------------------------------- #

t1 = 3600. # simulation time to run for
# dT = [.5,     1.,   5.,   5.,   .5,  .05,   .5,   5.,  .05, .075,   .1,   .5] # *C # mushy zone region. Higher boundary condition requires higher dT. Check theta_l graph for realistic distribution
# dx = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5]       # m
# dt = [1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]       # s
dT = [  .5, .05] # *C # mushy zone region. Higher boundary condition requires higher dT. Check theta_l graph for realistic distribution
dx = [1e-4, 1e-5]       # m
dt = [1e-5, 1e-5]       # s
T0 = [120.] * len(dT)
T = [] # results array

fps = 1 # fps for animation

for i in range(len(dT)):
    print(f"\nCalculating T0={T0[i]}, dT={dT[i]}, dx={dx[i]:.0e}, dt={dt[i]:.0e}...")
    
    x = np.arange(0, w, dx[i]) # all cell positions
    N = int(t1 / dt[i]) # number of iterations for simulation time
    t_fps = 1 / fps / dt[i] # number of iterations between each frame
    frames = np.where(np.arange(N, dtype=int) % int(t_fps) == 0)[0] # all iterations that are displayed
    
    with ProgressBar(total=N) as progress: # start progressbar
        T.append(load_frames(frames, x, N, T0[i], dT[i], dt[i], dx[i], progress)) # start simulation

    np.savetxt(f"results,T0={T0[i]},dT={dT[i]},dx={dx[i]:.1e},dt={dt[i]:.1e}.csv", T[-1], delimiter=",")