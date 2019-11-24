import argparse
import logging
import os
import random

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import trange

from stmmap import RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_3d, gen_meas_3d

from .gp_maps import GPMap

sns.set(style="ticks")


# -------- Handle command line args -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scale",
    help="Measurement uncertainty scaling. 0.05 (Accurate) or 1.5 (Inaccurate)",
    type=float,
    default=0.05,
)
parser.add_argument("--seed", help="Manually specify the noise seed.", type=int, default=None)
args = parser.parse_args()

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Set noise seed
if args.seed is not None:
    seed = args.seed
else:
    seed = (int)(np.random.rand() * (2.0 ** 30))
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# scale = 0.05 (Accurate) or 1.5 (Inaccurate)
scale = args.scale

directory = Path(__file__).resolve().parent / "results" / "gp_comp_3d" / str(seed) / str(scale)
if not os.path.exists(directory):
    os.makedirs(directory)

# XXX Setup logging
level = logging.DEBUG
output_file = directory / "output.log"
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

loggers = [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]

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
n_depths = 4  # Repeat for different grid depths. default=4
N = int(10 * 4 ** (n_depths))  # Number of measurements
loggers += [logging.getLogger("GPy")]
print("Max number of measurements", N)
nstds = 1  # Number of standard deviations to plot for height estimates
n_batches = 10  # default=10

# XXX Test points
N_test_row = 5 * 2 ** n_depths
xy = np.mgrid[0 : 1 : N_test_row * 1j, 0 : 1 : N_test_row * 1j].T.reshape(N_test_row ** 2, 2)
b = np.sum(xy, axis=1) <= 1
xy = xy[b, :]
N_test = xy.shape[0]
print("Number of test points", N_test)

log_like_stm_batch = np.zeros((n_batches))
log_like_gp_batch = np.zeros((n_batches))
mse_stm_batch = np.zeros((n_batches))
mse_gp_batch = np.zeros((n_batches))

load = True

if not load:
    for batch in trange(n_batches, desc="Batch"):
        env_type = "perlin"
        env_func = gen_env_fn_3d(env_type, seed=seed)

        test_pts = np.zeros((N_test, 3, 1))
        test_pts[:, :2, 0] = xy
        test_pts[:, -1] = env_func(test_pts[:, 0], test_pts[:, 1])

        m_gen, C_gen = gen_meas_3d(env_func, N, scale=scale)
        depth = 3
        m_z = np.copy(m_gen)
        C_z = np.copy(C_gen)

        # STM Map
        stm_map = RelativeSubmap(depth, dim=3, tolerance=1e-5, max_iterations=1000)
        stm_map.insert_measurements(1 * m_z, 1 * C_z)
        stm_map.update()
        log_like_stm = stm_map.loglike(1 * test_pts)
        mse_stm = stm_map.mean_squared_error(1 * test_pts)

        # GP Map
        gp_map = GPMap(dim=3)
        gp_map.update(1 * m_z, 1 * C_z)
        log_like_gp = gp_map.loglike(1 * test_pts)
        mse_gp = gp_map.mean_squared_error(1 * test_pts)

        log_like_stm_batch[batch] = log_like_stm
        log_like_gp_batch[batch] = log_like_gp
        mse_stm_batch[batch] = mse_stm
        mse_gp_batch[batch] = mse_gp
        print("MSE's (gp, stm)", mse_gp, mse_stm)
        print("LL's (gp, stm)", log_like_gp, log_like_stm)

    np.savez(
        open(os.path.join(directory, "data.npz"), "wb"),
        log_like_stm_batch,
        log_like_gp_batch,
        mse_stm_batch,
        mse_gp_batch,
    )
else:
    if not os.path.exists(directory):
        raise FileNotFoundError("Cannot find save folder")
    data = np.load(open(directory / "data.npz", "rb"))

    log_like_stm_batch = data["arr_0"]
    log_like_gp_batch = data["arr_1"]
    mse_stm_batch = data["arr_2"]
    mse_gp_batch = data["arr_3"]

# Log-likelihoods
data_ll = {"STM Map": log_like_stm_batch, "GP Map": log_like_gp_batch}
df_ll = pd.DataFrame(data_ll)
fig = plt.figure(constrained_layout=True)
ax = plt.gca()
sns.barplot(
    data=df_ll,
    palette=["C1", "C0"],
    estimator=np.mean,
    ci="sd",
    capsize=0.2,
    errwidth=1.5,
    linewidth=0.0,
)
# plt.yscale("log")
plt.ylabel(r"Log likelihood")
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

# sns.despine(fig)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
# golden = (1 + 5 ** 0.5) / 2
# ratio = golden
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)
fig.savefig(os.path.join(directory, "LL.pdf"))

# MSE
data_mse = {"STM Map": mse_stm_batch, "GP Map": mse_gp_batch}
df_mse = pd.DataFrame(data_mse)
fig = plt.figure(constrained_layout=True)
ax = plt.gca()
sns.barplot(
    data=df_mse,
    palette=["C1", "C0"],
    estimator=np.mean,
    ci="sd",
    capsize=0.2,
    errwidth=1.5,
    linewidth=0.0,
)
# plt.yscale("log")
plt.ylabel(r"MSE")
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
# sns.despine(fig)

cur_size = fig.get_size_inches()
ratio = cur_size[1] / cur_size[0]
# golden = (1 + 5 ** 0.5) / 2
# ratio = golden
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig.set_size_inches(*fs, forward=True)
fig.savefig(os.path.join(directory, "MSE.pdf"))

plt.show()
