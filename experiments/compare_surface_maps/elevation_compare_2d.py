import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import rc
from noise import pnoise1
from tqdm import tqdm

from stmmap import RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_2d, gen_meas_2d
from utils.svg_extract import extract_surface_from_svg

from .elevation_maps import ElevationMap2D

sns.set(style="ticks")

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Set noise seed
seed = (int)(np.random.rand() * (2.0 ** 30))
seed = 540398136
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# env_type = "perlin"
env_type = "simonsberg"
env_func = gen_env_fn_2d(env_type)

# XXX: Flags
n_depths = 10  # Repeat for different grid depths
divisions = 2
N = int(10 * divisions ** (n_depths))  # Number of measurements
nstds = 1  # Number of standard deviations to plot for height estimates

# XXX Test points
N_test = 10 * divisions ** n_depths
test_pts = np.zeros((N_test, 2, 1))
test_pts[:, 0, 0] = np.linspace(0, 1, N_test)
test_pts[:, 1, 0] = env_func(test_pts[:, 0, 0])

m_gen, C_gen = gen_meas_2d(env_func, N, scale=0.5)
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

# fig, ax = plt.subplots(
#     1,
#     n_depths+1,
#     sharex=True,
#     sharey=True,
#     figsize=(3 * n_depths+1, 3),
#     dpi=150,
#     constrained_layout=True)

ax = []
for i in range(n_depths + 1):
    figs = []
    fig = plt.figure(figsize=(9, 3), constrained_layout=True)
    figs.append(fig)
    ax.append(plt.gca())

log_ratio_test = []
mse_stm_batch = []
mse_elev_batch = []

for depth in range(n_depths + 1):
    print("Depth:", depth)
    sub_div = divisions ** (depth)
    m_z = np.copy(m_gen)
    C_z = np.copy(C_gen)

    # Calculate factor beliefs
    elev_map = ElevationMap2D(depth)
    stm_map = RelativeSubmap(depth, dim=2, tolerance=1e-5, max_iterations=100)

    print("Updating models")
    elev_map.update(1 * m_z, 1 * C_z)
    # NOTE Make sure the heights match
    stm_map.insert_measurements(1 * m_z, 1 * C_z)
    stm_map.update()

    # NOTE: Luckily this is only a chain
    bin_nums = np.arange(sub_div, dtype=np.uint32)
    bin_vars = np.vstack((bin_nums, bin_nums + 1)).T

    # XXX: Analysis
    # Elevation
    edges = (bin_vars / sub_div).flatten()
    heights = np.zeros_like(edges)
    stds = np.zeros_like(edges)
    log_like_elev = elev_map.loglike(test_pts)
    mse_elev = elev_map.mean_squared_error(test_pts)
    for i, values in enumerate(elev_map):
        height, variance = values
        heights[2 * i : 2 * (i + 1)] = 1 * height
        stds[2 * i : 2 * (i + 1)] = nstds * (variance) ** 0.5

    tmp = ax[depth].plot(edges, heights, label="Elevation Map", zorder=5)
    col = tmp[0].get_color()
    ax[depth].fill_between(
        edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=0
    )

    # Surfel Calibrated
    edges = (bin_vars / sub_div).flatten()
    heights = np.zeros_like(edges)
    stds = np.zeros_like(edges)
    log_like_stm = stm_map.loglike(test_pts)
    mse_stm = stm_map.mean_squared_error(test_pts)
    for i, surfel in enumerate(stm_map):
        heights[2 * i : 2 * (i + 1)] = surfel.bel_h_m.reshape(-1)
        stds[2 * i : 2 * (i + 1)] = nstds * (surfel.bel_v_b / surfel.bel_v_a) ** 0.5

    tmp = ax[depth].plot(edges, heights, label="STM Map", zorder=5)
    col = tmp[0].get_color()
    ax[depth].fill_between(
        edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=0
    )

    # Ground truth
    # tmp = plt.plot([0, 1], h_truths, label="Ground truth")
    # col = tmp[0].get_color()
    # std_truth = np.sqrt(var_truth)
    # print(std_truth, h_truths[0] + std_truth, h_truths[1] - std_truth)
    # plt.fill_between(
    #     np.array([0, 1]),
    #     h_truths + std_truth,
    #     h_truths - std_truth,
    #     alpha=0.5,
    #     color=col)
    x = np.linspace(0, 1, 1000)
    y = env_func(x)
    ax[depth].plot(x, y, "--", alpha=1, lw=1, zorder=6, label="Ground truth")

    ratio = log_like_stm - log_like_elev
    print("LL ratio", ratio)
    log_ratio_test.append(ratio)
    mse_stm_batch.append(mse_stm)
    mse_elev_batch.append(mse_elev)
    print("MSE's", mse_elev, mse_stm)

    # # Raw points
    # N_disp = 1000
    # skip = int(N / N_disp)
    # skip = 1
    # ax[depth].scatter(m_raw[::skip, 0, :], m_raw[::skip, 1, :], alpha=0.4, linewidth="0")
    # # hex = ax[depth].hexbin(
    # #     *m_raw[:, :, 0].T,
    # #     cmap="Greys",
    # #     linewidths=0,
    # #     extent=[0, 1, y_min, y_max],
    # #     antialiased=True)

    ax[depth].set_xlabel(r"$\alpha$")
    ax[depth].set_ylim(y_min, y_max)
    ax[depth].set_xlim(0, 1)
    ax[depth].set_ylabel(r"$\gamma$")

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "elev_comp_2d")
if not os.path.exists(directory):
    os.makedirs(directory)
directory = os.path.join(directory, env_type)
if not os.path.exists(directory):
    os.makedirs(directory)
directory = os.path.join(directory, str(seed))
if not os.path.exists(directory):
    os.makedirs(directory)

for i in plt.get_fignums():
    fig = plt.figure(i)
    sns.despine(fig)

    cur_size = fig.get_size_inches()
    ratio = cur_size[1] / cur_size[0]
    fig_scale = 0.48
    fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
    fig.set_size_inches(*fs, forward=True)

    # plt.legend()
    fig.savefig(os.path.join(directory, f"models_{i-1}.pdf"))

# Log-likelihoods
ratio = 4.8 / 6.4
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig = plt.figure(figsize=fs, constrained_layout=True)
depths = np.arange(0, n_depths + 1)
sns.barplot(depths, log_ratio_test, color="C3", linewidth=0)
plt.yscale("log")
plt.xlabel(r"Grid division depth")
plt.ylabel(r"Log-likelihood ratio")
plt.xticks(depths)
sns.despine(fig)

fig.savefig(os.path.join(directory, "LL.pdf"))

# MSE
ratio = 4.8 / 6.4
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig = plt.figure(figsize=fs, constrained_layout=True)
depths = np.arange(0, n_depths + 1)
plt.plot(depths, mse_elev_batch, marker="s", label="Elevation map", markersize=4)
plt.plot(depths, mse_stm_batch, marker="o", label="STM map", markersize=4)
plt.yscale("log")
plt.legend()
plt.xlabel(r"Grid division depth")
plt.ylabel(r"MSE")
plt.xticks(depths)
sns.despine(fig)

fig.savefig(os.path.join(directory, "MSE.pdf"))

plt.show()
