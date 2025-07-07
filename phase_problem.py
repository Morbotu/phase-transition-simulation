import numpy as np
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm
from scipy.special import erf, erfc
from scipy.optimize import root, curve_fit
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #

Tm      = 54.        # *C # melting temperature
T1      = 30.        # *C # right boundary temperature
dx      = 1e-4       # m
dt      = 1e-4       # s
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

# --------------------------- Analytical formula's --------------------------- #

# function for finding k of analytical solution, syntax with lambda is called currying https://en.wikipedia.org/wiki/Currying
# root_function = lambda Tw: lambda k: np.exp(-k**2) / erf(k) \
#     - c_s / c_l * np.sqrt(a_s / a_l) * (T1 - Tm) / (Tw - Tm) * np.exp(-k**2 * (a_l / a_s)) / erf(k * np.sqrt(a_l / a_s)) \
#     - k * L * np.sqrt(np.pi) / c_l / (Tw - Tm)

root_function = lambda Tw: lambda k: np.exp(-k**2) / erf(k) \
    + lam_s / lam_l * np.sqrt(a_l / a_s) * (T1 - Tm) / (Tw - Tm) * np.exp(-k**2 * (a_l / a_s)) / erfc(k * np.sqrt(a_l / a_s)) \
    - k * L * np.sqrt(np.pi) / c_l / (Tw - Tm)

# position of interface according to analytical solution
def X_i(t, k):
    return 2 * k * np.sqrt(a_l * t)

# temperature according to analytical solution
def T_analytical(x, t, Tw, k):
    if t == 0: # first frame is in initial temperature because requirement for formula is that t>0
        return np.full_like(x, T1)
    
    if x < X_i(t, k): # liquid region
        return Tw + (Tm - Tw) * erf(x / 2 / np.sqrt(a_l * t)) / erf(k)
    
    # solid region
    return T1 + (Tm - T1) * erfc(x / 2 / np.sqrt(a_s * t)) / erfc(k * np.sqrt(a_l / a_s))

T_analytical = np.vectorize(T_analytical)

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
def dT_liquid_loss(T):
    return 2 * dt * h_air / (rho_l * c_l * R) * (T - T1)

@njit
def dT_liquid(T):
    return a_l * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1]) - dT_liquid_loss(T[1:-1])

# discrete timestep of solid zone
@njit
def dT_solid_loss(T):
    return 2 * dt * h_air / (rho_s * c_s * R) * (T - T1)

@njit
def dT_solid(T):
    return a_s * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1]) - dT_solid_loss(T[1:-1])

# mushy zone heat conductivity
@njit
def lam_phi(T, dT):
    return full_theta_l(T, dT) * lam_l + (1 - full_theta_l(T, dT)) * lam_s

# discrete timestep of mushy zone
@njit
def dT_mush_loss(T, dT):
    return 2 * dt * h_air \
        / ((1 - theta_l(T, dT)) * rho_s * c_s \
        + theta_l(T, dT) * rho_l * c_l * R) * (T - T1)

@njit
def dT_mush(T, dT):
    return dt / (2 * dx**2 * c_A(T[1:-1], dT)) * ((lam_phi(T[2:], dT) + lam_phi(T[1:-1], dT)) * T[2:] \
        + (lam_phi(T[:-2], dT) + lam_phi(T[1:-1], dT)) * T[:-2] \
        - (lam_phi(T[2:], dT) + lam_phi(T[:-2], dT) + 2 * lam_phi(T[1:-1], dT)) * T[1:-1]) \
        - dT_mush_loss(T[1:-1], dT)


# ---------------------------------------------------------------------------- #
#                                  Simulation                                  #
# ---------------------------------------------------------------------------- #

# generator for timesteps of the temperature
# see https://en.wikipedia.org/wiki/Generating_function for definition of generator functions
# or https://youtu.be/5jwV3zxXc8E for python generator functions tutorial
@njit
def solve_equation(x, T0, dT):
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
            T_new[left:right+1] = T[left:right+1] + dT_mush(T[left-1:right+2], dT) # update mushy zone
        else: # otherwise add mushy zone to solid zone
            solid_mask |= mush_mask

        if liquid_mask.sum() > 1: # liquid zone starts at 1. 0 is considered fixed boundary
            left = 1 # starts at 1
            right = np.where(liquid_mask)[0][-1] # right side of liquid zone
            T_new[left:right+1] = T[left:right+1] + dT_liquid(T[left-1:right+2]) # update liquid zone

        left = max(np.where(solid_mask)[0][0], 1) # find start of solid zone
        right = len(T)-2 # solid zone end 1 from the last cell
        T_new[left:right+1] = T[left:right+1] + dT_solid(T[left-1:right+2]) # update solid zone

        # T_new[-1] = T_new[-2] # uncomment voor geïsoleerd

        T, T_new = T_new, T # swap T and T_new
        yield T # return T iteration

# loading generator just saving frames that are viewed
@njit(nogil=True)
def load_frames(frames, x, N, T0, dT, progress):
    solve_equation_gen = solve_equation(x, T0, dT) # load the generator
    T_list = np.empty((frames.size, x.size))
    
    frame = 0
    for i in range(N):
        if frame < len(frames) and i == frames[frame]: # if the iteration is going to be shown
            T_list[frame] = next(solve_equation_gen) # save the iteration
            frame += 1
        else: # else don't save the iteration
            next(solve_equation_gen)[0:3]

        progress.update(1) # update progressbar
    
    return T_list

# ---------------------------------------------------------------------------- #
#                              Running simulation                              #
# ---------------------------------------------------------------------------- #

x = np.arange(0, w, dx) # all cell positions
t1 = 3600. # simulation time to run for
N = int(t1 / dt) # number of iterations for simulation time
# T0 = np.arange(100, 1000 + 50, 50) # *C # left boundary condition temperatures
T0 = [120.]
dT = [.5] * len(T0) # *C # mushy zone region. Higher boundary condition requires higher dT. Check theta_l graph for realistic distribution
T = [] # results array
T_analytics = []

playback = t1 / 30 # playback speed
fps = 20 # fps for animation
t_fps = 1 / fps / dt # number of iterations between each frame
frames = np.where(np.arange(N, dtype=int) % int(t_fps * playback) == 0)[0] # all iterations that are displayed

fit_func = lambda t, k: 2 * k * np.sqrt(a_l * t)
t = frames * dt
ks = []
for i in range(len(T0)):
    print(f"\nCalculating T0={T0[i]}...")
    
    with ProgressBar(total=N) as progress: # start progressbar
        T.append(load_frames(frames, x, N, T0[i], dT[i], progress)) # start simulation
    
    s = [x[np.abs(T[-1][i] - Tm).argmin()] for i in range(len(frames))]
    popt, popv = curve_fit(fit_func, t, s)

    ks.append(popt[0])

    np.savetxt(f"returnsT0={T[-1][0][0]},dT={dT[-1]}.csv", T[-1], delimiter=",")

out = np.array([T0, ks]).T
np.savetxt(f"ks.csv", out, delimiter=",")

for T0_i in set(T0):
    T_exact = np.zeros((frames.size, x.size))
    
    k = root(root_function(T0_i), 1)["x"][0] # find the value of k for the analytical solution

    for i, frame in enumerate(frames):
        T_exact[i] = T_analytical(x, frame * dt, T0_i, k)

    T_analytics.append(T_exact)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ymin, ymax = 0, max(T0)
img_temp = []
img_temp_analytical = []
img_theta_l = []
vlines = []

# plot the data and save the plots for animation
for i in range(len(T0)):
    img_temp.append(ax[0].plot(x, T[i][0], label=f"T0={T[i][0][0]}, dT={dT[i]}")[0]) # plot the temperature as a function of position
    img_theta_l.append(ax[1].plot(x, full_theta_l(T[i][0], dT[i]), '.')[0]) # plot fraction of liquid on the right
    vlines.append(ax[0].vlines(x[abs(T[i][0] - Tm).argmin()], ymin, ymax, "red", "dashed")) # the position of the interface

for i in range(len(set(T0))):
    img_temp_analytical.append(ax[0].plot(x, T_analytics[i][0], label=f"T0={T[i][0][0]} analytical")[0])

title = fig.suptitle("t = 0 s")
# ax[0].set_xlim(0, 30e-3)
ax[0].legend(loc="upper right")
ax[0].set_xlabel("$x$ (m)")
ax[0].set_ylabel("$T$ ($\\degree$C)")
ax[1].set_xlabel("$x$ (m)")
ax[1].set_ylabel("$\\theta_l$")
# ax[1].set_xlim(0, 30e-3)

# update animation
def update(frame):
    for i in range(len(T0)):
        new_x = x[np.abs(T[i][frame] - Tm).argmin()]

        # iterate over the frames in the animation
        vlines[i].set_segments([[(new_x, ymin), (new_x, ymax)]])
        img_temp[i].set_data(x, T[i][frame])
        img_temp[i].set_label(f"T0={T[i][0][0]}, dT={dT[i]}, mush={((Tm - dT[i] < T[i][frame]) & (T[i][frame] < Tm + dT[i])).sum():.4f}")
        img_theta_l[i].set_data(x, full_theta_l(T[i][frame], dT[i]))
    
    for i in range(len(set(T0))):
        img_temp_analytical[i].set_data(x, T_analytics[i][frame])

    title.set_text(f"t = {frames[frame] * dt:.2f} s")
    legend = ax[0].legend(loc="upper right")
    return *img_temp, *img_temp_analytical, *img_theta_l, *vlines, title, legend

# run animation
ani = animation.FuncAnimation(fig, update, frames=frames.size, blit=False, interval=1e3/fps)

print("\nSaving animation...")
with tqdm(total=len(frames)) as pbar:
    ani.save('phase_problem.mp4', fps=fps, dpi=300, progress_callback=lambda i, n: pbar.update(1))

plt.tight_layout()
plt.show()