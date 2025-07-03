import numpy as np
from numba import njit, typeof
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
T1      = 20.        # *C # right boundary temperature
dx      = 1e-5       # m
dt      = 1e-4       # s
L       = 2.1e5      # J/kg
rho_l   = 780.       # kg/m^3
rho_s   = 860.       # kg/m^3
lam_l   = 0.15       # W/K m # lambda_l
lam_s   = 0.24       # W/K m # lambda_s
c_l     = 2100.      # J/kg K
c_s     = 2900.      # J/kg K
w       = 10e-3      # m # simulated from 0 to w distance

a_s = lam_s / (rho_s * c_s)
a_l = lam_l / (rho_l * c_l)

t_analytic = 0       # s # initial simulation using analytical solution

# ---------------------------------------------------------------------------- #
#                                   Formula's                                  #
# ---------------------------------------------------------------------------- #

# --------------------------- Analytical formula's --------------------------- #

# function for finding k of analytical solution, syntax with lambda is called currying https://en.wikipedia.org/wiki/Currying
root_function = lambda Tw: lambda k: c_l * (Tw - Tm) * np.exp(-k**2) / erf(k) \
    - np.sqrt(a_s / a_l) * c_s * (T1 - Tm) * np.exp(-k**2 * (a_l / a_s)) / erf(k * np.sqrt(a_l / a_s)) - k * L * np.sqrt(np.pi)

# root_function = lambda Tw: lambda k: np.exp(-k**2) / erf(k) \
#     + lam_s / lam_l * np.sqrt(a_l / a_s) * (T1 - Tm) / (Tw - Tm) * np.exp(-k**2 * (a_l / a_s)) / erfc(k * np.sqrt(a_l / a_s)) \
#     + k * L * a_l * np.sqrt(np.pi) / lam_l / (Tw - Tm)

# position of interface according to analytical solution
@njit
def X_i(t, k):
    return 2 * k * np.sqrt(a_l * t)

# temperature according to analytical solution
@njit
def T_analytical(x, t, Tw, k):
    if t == 0: # first frame is in initial temperature because requirement for formula is that t>0
        return T1
    
    if x < X_i(t, k): # liquid region
        return Tw + (Tm - Tw) * erf(x / 2 / np.sqrt(a_l * t)) / erf(k)
    
    # solid region
    return T1 + (Tm - T1) * erfc(x / 2 / np.sqrt(a_s * t)) / erfc(k * np.sqrt(a_l / a_s))

# ----------------------------- python analytical ---------------------------- #

# position of interface according to analytical solution
def X_i_python(t, k):
    return 2 * k * np.sqrt(a_l * t)

# temperature according to analytical solution
def T_analytical_python(x, t, Tw, k):
    if t == 0: # first frame is in initial temperature because requirement for formula is that t>0
        return np.full_like(x, T1)
    
    if x < X_i_python(t, k): # liquid region
        return Tw + (Tm - Tw) * erf(x / 2 / np.sqrt(a_l * t)) / erf(k)
    
    # solid region
    return T1 + (Tm - T1) * erfc(x / 2 / np.sqrt(a_s * t)) / erfc(k * np.sqrt(a_l / a_s))

T_analytical_python = np.vectorize(T_analytical_python)

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
# @njit
# def c_A(T, dT):
#     return (1 - theta_l(T, dT)) * rho_s * c_s \
#         + theta_l(T, dT) * rho_l * c_l + (rho_l * c_l * T - rho_s * c_s * T + rho_l * L) / 2 / dT

# second version
# @njit
# def c_A(T, dT):
#     return (1 - theta_l(T[1:-1], dT)) * rho_s * c_s \
#         + theta_l(T[1:-1], dT) * rho_l * c_l \
#         + ((T[:-2] < Tm + dT) & (T[2:] > Tm - dT)) *  ((rho_l * c_l - rho_s * c_s) * T[1:-1] + rho_l * L) * (full_theta_l(T[2:], dT) - full_theta_l(T[:-2], dT)) / (T[2:] - T[:-2])

@njit
def c_A(T, dT):
    return (1 - theta_l(T, dT)) * rho_s * c_s \
        + theta_l(T, dT) * rho_l * c_l
        #+ ((rho_l * c_l - rho_s * c_s) * T + rho_l * L) / 2 / dT


# discrete timestep of liquid zone
@njit
def dT_liquid(T):
    return a_l * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1])

# discrete timestep of solid zone
@njit
def dT_solid(T):
    return a_s * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1])

@njit
def lam_phi(T, dT):
    return full_theta_l(T, dT) * lam_l + (1 - full_theta_l(T, dT)) * lam_s

# discrete timestep of mushy zone
@njit
def dT_mush_old(T, dT):
    return (lam_l - lam_s) / 2 / c_A(T[1:-1], dT) / dT * dt * ((T[2:] - T[:-2]) / 2 / dx)**2 \
        + ((1-theta_l(T[1:-1], dT)) * lam_s + theta_l(T[1:-1], dT) * lam_l) / c_A(T[1:-1], dT) * dt / dx**2 * (T[2:] + T[:-2] - 2 * T[1:-1])

# second version
@njit
def dT_mush(T, dT):
    return dt / (2 * dx**2 * c_A(T[1:-1], dT)) * ((lam_phi(T[2:], dT) + lam_phi(T[1:-1], dT)) * T[2:] \
        + (lam_phi(T[:-2], dT) + lam_phi(T[1:-1], dT)) * T[:-2] \
        - (lam_phi(T[2:], dT) + lam_phi(T[:-2], dT) + 2 * lam_phi(T[1:-1], dT)) * T[1:-1])


# ---------------------------------------------------------------------------- #
#                                  Simulation                                  #
# ---------------------------------------------------------------------------- #

# generator for timesteps of the temperature
# see https://en.wikipedia.org/wiki/Generating_function for definition of generator functions
# or https://youtu.be/5jwV3zxXc8E for python generator functions tutorial
@njit
def solve_equation(x, T0, dT, k, old):
    T = np.full(x.size, T1)
    T[0] = T0 # boundary condition
    T_new = T.copy()
    i = 0

    while True:
        # analytical solution for time until t_analytic
        if i * dt < t_analytic:
            for j in range(len(T)):
                T[j] = T_analytical(x[j], i * dt, T0, k)

            T, T_new = T_new, T
            yield T
            continue

        # differentiations of zones
        mush_mask = (Tm - dT < T) & (T < Tm + dT)
        solid_mask = T <= Tm - dT
        liquid_mask = T >= Tm + dT
        
        # mushy zone if there is also a liquid zone and a minimum amount of mush
        if np.any(mush_mask):
            left = np.where(mush_mask)[0][0] # find start of mushy zone
            right = np.where(mush_mask)[0][-1] # find end of mushy zone
            if old:
                T_new[left:right+1] = T[left:right+1] + dT_mush_old(T[left-1:right+2], dT)
            else:
                T_new[left:right+1] = T[left:right+1] + dT_mush(T[left-1:right+2], dT) # update mushy zone
            # print(T_new[left-1], T_new[left], T_new[left+1], dT_mush(T[left-1:left+2], dT), left)
        else: # otherwise add mushy zone to solid zone
            solid_mask |= mush_mask

        if liquid_mask.sum() > 1: # liquid zone starts at 1. 0 is considered fixed boundary
            left = 1 # starts at 1
            right = np.where(liquid_mask)[0][-1] # right side of liquid zone
            T_new[left:right+1] = T[left:right+1] + dT_liquid(T[left-1:right+2]) # update liquid zone

        left = max(np.where(solid_mask)[0][0], 1) # find start of solid zone
        right = len(T)-2 # solid zone end 1 from the last cell
        T_new[left:right+1] = T[left:right+1] + dT_solid(T[left-1:right+2]) # update solid zone

        # T_new[-1] = T_new[-2] # geïsoleerd

        T, T_new = T_new, T # swap T and T_new
        i += 1
        yield T # return T iteration

# loading generator just saving frames that are viewed
@njit(nogil=True)
def load_frames(frames, x, N, T0, dT, k, progress, old):
    solve_equation_gen = solve_equation(x, T0, dT, k, old) # load the generator
    T_list = np.empty((frames.size, x.size))
    
    frame = 0
    for i in range(N):
        if frame < len(frames) and i == frames[frame]: # if the iteration is going to be shown
            T_list[frame] = next(solve_equation_gen) # save the iteration
            frame += 1
        else: # else don't save the iteration
            next(solve_equation_gen)[2]

        progress.update(1) # update progressbar
    
    return T_list

# ---------------------------------------------------------------------------- #
#                              Running simulation                              #
# ---------------------------------------------------------------------------- #

x = np.arange(0, w, dx) # all cell positions
t1 = 30. # simulation time to run for
N = int(t1 / dt) # number of iterations for simulation time
T0 = [100., 100., 100.] # *C # left boundary condition temperatures
dT = [1., 10., 30.] # *C # mushy zone region. Higher boundary condition requires higher dT. Check theta_l graph for realistic distribution
old = [False, False, False]
T = [] # results array
T_analytics = []

playback = 3 # playback speed
fps = 20 # fps for animation
t_fps = 1 / fps / dt # number of iterations between each frame
frames = np.where(np.arange(N, dtype=int) % int(t_fps * playback) == 0)[0] # all iterations that are displayed

for i in range(len(T0)):
    print(f"\nCalculating T0={T0[i]}...")

    k = root(root_function(T0[i]), 1)["x"][0] # find the value of k for the analytical solution

    # k = 0.45352843824427835
    
    with ProgressBar(total=N) as progress: # start progressbar
        T.append(load_frames(frames, x, N, T0[i], dT[i], k, progress, old[i])) # start simulation

for T0_i in set(T0):
    T_exact = np.zeros((frames.size, x.size))
    
    k = root(root_function(T0_i), 1)["x"][0] # find the value of k for the analytical solution

    for i, frame in enumerate(frames):
        T_exact[i] = T_analytical_python(x, frame * dt, T0_i, k)

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
ax[0].set_xlim(0, 0.005)
ax[0].legend(loc="upper right")
ax[0].set_xlabel("$x$ (m)")
ax[0].set_ylabel("$T$ ($\\degree$C)")
ax[1].set_xlabel("$x$ (m)")
ax[1].set_ylabel("$\\theta_l$")
ax[1].set_xlim(0, 0.005)

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

np.savetxt("foo.csv", T[0], delimiter=",")

# print("\nSaving animation...")
# with tqdm(total=len(frames)) as pbar:
#     ani.save('phase_problem.mp4', fps=fps, dpi=300, progress_callback=lambda i, n: pbar.update(1))

plt.tight_layout()
plt.show()

# s = [x[np.abs(T[0][i] - Tm).argmin()] for i in range(len(frames))]
# t = frames * dt

# fit_func = lambda t, k: 2 * k * np.sqrt(a_l * t)

# popt, popv = curve_fit(fit_func, t, s)

# print(f"k = {popt[0]}")

# plt.plot(t, s)
# plt.plot(t, fit_func(t, *popt))
# plt.plot(t, 2 * k * np.sqrt(a_l * t))

# plt.show()