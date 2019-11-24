import os
import random

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm import trange

from stmmap import RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_3d, gen_meas_3d

from .elevation_maps import ElevationMap3D

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Set noise seed
seed = (int)(np.random.rand() * (2.0 ** 30))
seed = 443839078
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# XXX: Flags
n_depths = 8  # Repeat for different grid depths
N = int(10 * 4 ** (n_depths))  # Number of measurements
print("Max number of measurements", N)
nstds = 1  # Number of standard deviations to plot for height estimates
n_batches = 10
env_type = "perlin"

# XXX Test points
N_test_row = 5 * 2 ** n_depths
xy = np.mgrid[0 : 1 : N_test_row * 1j, 0 : 1 : N_test_row * 1j].T.reshape(N_test_row ** 2, 2)
b = np.sum(xy, axis=1) <= 1
xy = xy[b, :]
N_test = xy.shape[0]
print("Number of test points", N_test)

log_ratio_test_batch = np.zeros((n_batches, n_depths))
mse_stm_batch = np.zeros((n_batches, n_depths))
mse_elev_batch = np.zeros((n_batches, n_depths))

load = True

if not load:
    for batch in trange(n_batches, desc="Batch"):  # TODO Add this back
        env_func = gen_env_fn_3d(env_type, seed=seed)

        test_pts = np.zeros((N_test, 3, 1))
        test_pts[:, :2, 0] = xy
        test_pts[:, -1] = env_func(test_pts[:, 0], test_pts[:, 1])

        m_gen, C_gen = gen_meas_3d(env_func, N, scale=0.5)
        for depth in trange(n_depths, desc="Depth"):
            m_z = np.copy(m_gen)
            C_z = np.copy(C_gen)

            # print("Updating models")
            # print("Update elev")
            elev_map = ElevationMap3D(depth)
            elev_map.update(1 * m_z, 1 * C_z)

            # print("Update STM")
            stm_map = RelativeSubmap(depth, dim=3, tolerance=1e-5, max_iterations=100)
            stm_map.insert_measurements(1 * m_z, 1 * C_z)
            stm_map.update()

            # XXX: Analysis
            # print("Analysis")
            # Elevation
            log_like_elev = elev_map.loglike(1 * test_pts)
            mse_elev = elev_map.mean_squared_error(1 * test_pts)

            # STM
            log_like_stm = stm_map.loglike(1 * test_pts)
            mse_stm = stm_map.mean_squared_error(1 * test_pts)

            ratio = log_like_stm - log_like_elev
            # print("LL ratio", ratio)
            log_ratio_test_batch[batch, depth] = ratio
            mse_stm_batch[batch, depth] = mse_stm
            mse_elev_batch[batch, depth] = mse_elev
            # print("MSE's (elev, stm)", mse_elev, mse_stm)

    directory = Path(__file__).resolve().parent / "results" / "elev_comp_3d" / env_type / str(seed)
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savez(
        open(os.path.join(directory, "data.npz"), "wb"),
        log_ratio_test_batch,
        mse_stm_batch,
        mse_elev_batch,
    )
else:
    directory = Path(__file__).resolve().parent / "results" / "elev_comp_3d" / env_type / str(seed)
    if not os.path.exists(directory):
        raise FileNotFoundError("Cannot find save folder")
    data = np.load(open(directory / "data.npz", "rb"))

    log_ratio_test_batch = data["arr_0"]
    mse_stm_batch = data["arr_1"]
    mse_elev_batch = data["arr_2"]

# Log-likelihoods
fig = plt.figure(constrained_layout=True)
depths = np.arange(0, n_depths)
sns.barplot(
    data=log_ratio_test_batch,
    color="C3",
    estimator=np.mean,
    ci="sd",
    capsize=0.2,
    errwidth=1.5,
    linewidth=0.0,
)
plt.yscale("log")
plt.xlabel(r"Grid division depth")
plt.ylabel(r"Log-likelihood ratio")
plt.xticks(depths)
sns.despine(fig)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)
fig.savefig(os.path.join(directory, "LL.pdf"))

# MSE
fig = plt.figure(constrained_layout=True)
depths = np.arange(0, n_depths)
mse_elev_batch_mean = np.mean(mse_elev_batch, axis=0)
mse_elev_batch_std = np.std(mse_elev_batch, axis=0)
plt.errorbar(
    depths,
    mse_elev_batch_mean,
    yerr=mse_elev_batch_std,
    marker="s",
    label="Elevation map",
    capsize=5,
    markersize=4,
)

mse_stm_batch_mean = np.mean(mse_stm_batch, axis=0)
mse_stm_batch_std = np.std(mse_stm_batch, axis=0)
plt.errorbar(
    depths,
    mse_stm_batch_mean,
    yerr=mse_stm_batch_std,
    marker="o",
    label="STM map",
    capsize=5,
    markersize=4,
)
plt.yscale("log")
plt.legend()
plt.xlabel(r"Grid division depth")
plt.ylabel(r"MSE")
plt.xticks(depths)
sns.despine(fig)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)
fig.savefig(os.path.join(directory, "MSE.pdf"))

# plt.show()
