import numpy as np
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm
from scipy.special import erf, erfc
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import animation

Tm      = 54.        # *C
Tw      = 500.        # *C
dT      = 25.         # *C
T1      = 20.        # *C
dx      = 1e-5       # m
dt      = 1e-4       # s
L       = 2.1e5      # J/kg
rho_l   = 780.       # kg/m^3
rho_s   = 860.       # kg/m^3
lam_l   = 0.15       # W/K m
lam_s   = 0.24       # W/K m
c_l     = 2100.      # J/kg K
c_s     = 2900.      # J/kg K
w       = 5e-3       # m

a_s = lam_s / (rho_s * c_s)
a_l = lam_l / (rho_l * c_l)

# root_function = lambda k: np.exp(-k**2) / erf(k) + lam_l / lam_s * np.sqrt(a_s / a_l) * (Tm - T1) / (Tm - Tw) * np.exp(-k**2 * (a_s / a_l)) / erfc(k * np.sqrt(a_s / a_l)) + k * L * np.sqrt(np.pi) / c_s / (Tm - Tw)
# root_function = lambda k: np.exp(-k**2) / erf(k) \
#     + lam_s / lam_l * np.sqrt(a_l / a_s) * (Tm - T1) / (Tm - Tw) * np.exp(-k**2 * (a_l / a_s)) / erfc(k * np.sqrt(a_l / a_s)) \
#     - k * L * np.sqrt(np.pi) / c_l / (Tm - Tw)
root_function = lambda k: c_l * (Tw - Tm) * np.exp(-k**2) / erf(k) \
    - np.sqrt(a_s / a_l) * c_s * (T1 - Tm) * np.exp(-k**2 * (a_l / a_s)) / erf(k * np.sqrt(a_l / a_s)) - k * L * np.sqrt(np.pi)
k = root(root_function, 1)["x"][0]

def X_i(t):
    return 2 * k * np.sqrt(a_l * t)

def T(x, t):
    if t == 0:
        return np.full_like(x, T1)
    
    if x < X_i(t):
        return Tw + (Tm - Tw) * erf(x / 2 / np.sqrt(a_l * t)) / erf(k)
    
    return T1 + (Tm - T1) * erfc(x / 2 / np.sqrt(a_s * t)) / erfc(k * np.sqrt(a_l / a_s))

T = np.vectorize(T)

x = np.linspace(0, w, 1000)
t1 = 30
playback = 3

fig, ax = plt.subplots()

ymin, ymax = 0, Tw

img = ax.plot(x, T(x, 0))[0]
vline = ax.vlines(X_i(0), ymin, ymax, "red", "dashed")

fps = 20
t_fps = 1 / fps

def update(frame):
    # vline.set_segments([[(X_i(t_fps * frame * playback), ymin), (X_i(t_fps * frame * playback), ymax)]])
    img.set_data(x, T(x, t_fps * frame * playback))
    return img,

ani = animation.FuncAnimation(fig, update, frames=int(t1 * fps / playback), blit=True, interval=1e3/fps)

print("\nSaving animation...")
with tqdm(total=int(t1 * fps / playback)) as pbar:
    ani.save('analytische_oplossing.mp4', fps=fps, progress_callback=lambda i, n: pbar.update(1))

plt.show()