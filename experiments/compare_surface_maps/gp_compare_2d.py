import logging
import os
import random

from pathlib import Path

import GPy as gpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from noise import pnoise1
from tqdm import tqdm

from stmmap import RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_2d, gen_meas_2d

from .elevation_maps import ElevationMap2D
from .gp_maps import GPMap

sns.set(style="ticks")

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Set noise seed
seed = (int)(np.random.rand() * (2.0 ** 30))
# seed = 540398136
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# env_type = "perlin"
env_type = "simonsberg"
func = gen_env_fn_2d(env_type)
env_func = lambda x: 1 * func(x)

directory = Path(__file__).resolve().parent / "results" / "gp_comp_2d" / env_type / str(seed)
if not os.path.exists(directory):
    os.makedirs(directory)

# XXX Setup logging
level = logging.DEBUG
output_file = directory / "output.log"
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

loggers = [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]
loggers += [logging.getLogger("GPy")]

for logger in loggers:
    logger.setLevel(level)

    # create file handler which logs even debug messages
    if output_file is not None:
        fh = logging.FileHandler(output_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# XXX: Flags
n_depths = 10  # Repeat for different grid depths. default=10
divisions = 2
N = int(10 * divisions ** (n_depths))  # Number of measurements
nstds = 1  # Number of standard deviations to plot for height estimates

# XXX Test points
N_test = 10 * divisions ** n_depths
test_pts = np.zeros((N_test, 2, 1))
test_pts[:, 0, 0] = np.linspace(0, 1, N_test)
test_pts[:, 1, 0] = env_func(test_pts[:, 0, 0])

# scale = 0.05  # Accurate
scale = 0.3  # Inaccurate
m_gen, C_gen = gen_meas_2d(env_func, N, scale=scale)
invalid = np.logical_or(m_gen[:, 0, 0] < 0, m_gen[:, 0, 0] > 1)
keep = ~invalid
m_gen = m_gen[keep, :, :]
C_gen = C_gen[keep, :, :]
m_raw = 1 * m_gen

# Make sure there is a constant amount of measurements in each element
sub_div = divisions ** (n_depths)
bins = np.linspace(0, 1, sub_div + 1)
bin_alloc = (np.digitize(m_gen[:, 0, :], bins) - 1).flatten()
sorted_indices = np.argsort(bin_alloc)

y_min, y_max = (1.1 * np.min(m_gen[:, 1, 0]), 1.1 * np.max(m_gen[:, 1, :]))

fig = plt.figure(figsize=(9, 3), constrained_layout=True)
ax = plt.gca()

axins = zoomed_inset_axes(ax, 3, loc="upper left")
axins.set_visible(True)
axins.set_yticklabels([])
axins.set_xticklabels([])
# axins.tick_params(direction="in")
axins.get_xaxis().set_ticks([])
axins.get_yaxis().set_ticks([])
plt.setp(axins.spines.values(), color="0.5")

mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

m_z = np.copy(m_gen)
C_z = np.copy(C_gen)

depth = 6
sub_div = divisions ** (depth)

# Calculate factor beliefs

# print("Updating Elevation Map")
# elev_map = ElevationMap2D(depth)
# elev_map.update(1 * m_z, 1 * C_z)
print("Updating STM Map")
stm_map = RelativeSubmap(depth, dim=2, tolerance=1e-5, max_iterations=1000)
stm_map.insert_measurements(1 * m_z, 1 * C_z)
stm_map.update()
print("Updating GP Map")
gp_map = GPMap(dim=2)
gp_map.update(1 * m_z, 1 * C_z)

# NOTE: Luckily this is only a chain
bin_nums = np.arange(sub_div, dtype=np.uint32)
bin_vars = np.vstack((bin_nums, bin_nums + 1)).T

# # XXX: Analysis
# # Elevation
# edges = (bin_vars / sub_div).flatten()
# heights = np.zeros_like(edges)
# stds = np.zeros_like(edges)
# log_like_elev = elev_map.loglike(test_pts)
# mse_elev = elev_map.mean_squared_error(test_pts)
# for i, values in enumerate(elev_map):
#     height, variance = values
#     heights[2 * i : 2 * (i + 1)] = 1 * height
#     stds[2 * i : 2 * (i + 1)] = nstds * (variance) ** 0.5

# tmp = ax.plot(edges, heights, "C0", label="Elevation Map", zorder=5)
# col = tmp[0].get_color()
# ax.fill_between(
#     edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=0
# )
# print("Elevation MSE", mse_elev)
# print("Elevation LL", log_like_elev)

# GP Map
gpy.plotting.change_plotting_library("matplotlib")
gp_map.gp_model.plot_mean(
    ax=ax, plot_limits=[0, 1], resolution=1000, label="GP Map", color="C0", zorder=5, linewidth=1.5
)
gp_map.gp_model.plot_confidence(
    ax=ax,
    plot_limits=[0, 1],
    resolution=1000,
    lower=50 - 34.1,
    upper=50 + 34.1,
    label=None,
    color="C0",
    zorder=5,
    linewidth=1,
)
gp_map.gp_model.plot_mean(
    ax=axins,
    plot_limits=[0, 1],
    resolution=1000,
    label="GP Map",
    color="C0",
    zorder=5,
    linewidth=1.5,
)
gp_map.gp_model.plot_confidence(
    ax=axins,
    plot_limits=[0, 1],
    resolution=1000,
    lower=50 - 34.1,
    upper=50 + 34.1,
    label=None,
    color="C0",
    zorder=5,
    linewidth=1,
)

log_like_gp = gp_map.loglike(test_pts)
mse_gp = gp_map.mean_squared_error(test_pts)
print("GP MSE", mse_gp)
print("GP LL", log_like_gp)


# STM Map
edges = (bin_vars / sub_div).flatten()
heights = np.zeros_like(edges)
stds = np.zeros_like(edges)
log_like_stm = stm_map.loglike(test_pts)
mse_stm = stm_map.mean_squared_error(test_pts)
for i, surfel in enumerate(stm_map):
    heights[2 * i : 2 * (i + 1)] = surfel.bel_h_m.reshape(-1)
    stds[2 * i : 2 * (i + 1)] = nstds * (surfel.bel_v_b / (surfel.bel_v_a)) ** 0.5

tmp = ax.plot(edges, heights, "C1", label="STM Map", zorder=5, linewidth=1.5)
col = tmp[0].get_color()
ax.fill_between(
    edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=1
)
tmp = axins.plot(edges, heights, "C1", label="STM Map", zorder=5, linewidth=1.5)
col = tmp[0].get_color()
axins.fill_between(
    edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=1
)
print("STM MSE", mse_stm)
print("STM LL", log_like_stm)

# Ground truth
x = np.linspace(0, 1, 1000)
y = env_func(x)
ax.plot(x, y, "--C2", alpha=1, lw=1.5, zorder=6, label="Ground truth")
axins.plot(x, y, "--C2", alpha=1, lw=1.5, zorder=6, label="Ground truth")

# ratio = log_like_stm - log_like_elev
# print("LL ratio", ratio)
# log_ratio_test.append(ratio)
# mse_stm_batch.append(mse_stm)
# mse_elev_batch.append(mse_elev)
# print("MSE's", mse_elev, mse_stm)

ax.set_ylim(y_min, y_max)
ax.set_xlim(0, 1)

# ax.legend(loc="lower center", ncol=3).set_zorder(8)
ax.legend(loc="upper right", ncol=1).set_zorder(8)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
sns.despine(ax=ax, bottom=True, left=True)
# ax.set_ylabel(r"$\gamma$")
# ax.set_xlabel(r"$\alpha$")

# xbounds = [0, 0]
# ybounds = [0, 0]
xbounds = [0.36496209215851083, 0.45246665643979583]
ybounds = [0.5847677015311482, 0.8139092985815604]

axins.set_xlim(*xbounds)
axins.set_ylim(*ybounds)

# def on_press(event):
#     if event.inaxes == ax:
#         xbounds[0] = event.xdata
#         ybounds[0] = event.ydata


# def on_release(event):
#     if event.inaxes == ax:
#         xbounds[1] = event.xdata
#         ybounds[1] = event.ydata
#         axins.set_xlim(*xbounds)
#         axins.set_ylim(*ybounds)
#         print(xbounds)
#         print(ybounds)

#         plt.draw()


# cid = fig.canvas.mpl_connect("button_press_event", on_press)
# cid = fig.canvas.mpl_connect("button_release_event", on_release)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
scale = 0.98
fs = (scale * plt_cfg.tex_textwidth, scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)

fig.savefig(directory / "model.pdf")

fig = plt.figure(figsize=(9, 3), constrained_layout=True)
ax = plt.gca()
# Ground truth
x = np.linspace(0, 1, 1000)
y = env_func(x)
ax.plot(x, y, "--C2", alpha=1, lw=1.5, zorder=6, label="Ground truth")

# ratio = log_like_stm - log_like_elev
# print("LL ratio", ratio)
# log_ratio_test.append(ratio)
# mse_stm_batch.append(mse_stm)
# mse_elev_batch.append(mse_elev)
# print("MSE's", mse_elev, mse_stm)

# Raw points
N_disp = 200
skip = int(N / N_disp)
ax.scatter(m_raw[::skip, 0, :], m_raw[::skip, 1, :], color="C3", alpha=0.9, linewidth="0", s=10)
# N_grid = 75
# hex = ax.hexbin(
#     *m_raw[:, :, 0].T,
#     cmap="Greys",
#     linewidths=0,
#     extent=[0, 1, y_min, y_max],
#     antialiased=True,
#     gridsize=(N_grid, int(N_grid / ratio))
# )

ax.set_ylim(y_min, y_max)
ax.set_xlim(0, 1)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
sns.despine(ax=ax, bottom=True, left=True)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
scale = 0.48
fs = (scale * plt_cfg.tex_textwidth, scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)

fig.savefig(directory / "ground_truth.pdf")

plt.show()
