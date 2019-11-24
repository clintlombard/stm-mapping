#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import logging
import os
import random
import sys
import time

from pathlib import Path
from typing import List

import dill
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import seaborn as sb
import sympy as sym

from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise1, pnoise2
from opensimplex import OpenSimplex
from scipy import stats
from scipy.linalg import block_diag
from scipy.special import gamma as gammafn
from scipy.stats import gamma, multivariate_normal
from tqdm import tqdm
from transformations import *

from stmmap import MessageCounter, RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_3d, gen_meas_3d

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# -------- Logging configuration -------------------------------------------------------
loggers: List[logging.Logger] = []
loggers += [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]

level = logging.DEBUG  # This could be controlled with --log=DEBUG (I think)

# create formatter and add it to the handlers
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

ch = logging.StreamHandler()
ch.setLevel(level)
ch.setFormatter(formatter)

for logger in loggers:
    logger.setLevel(level)
    logger.addHandler(ch)

root_logger = loggers[0]
root_logger.debug("Logging initialised")


# -------- Constants and Flags ----------------------------------------------------------
seed = (int)(np.random.rand() * (2.0 ** 30))
# seed =
# seed = 66490824
# seed = 307072117
# seed = 335220882
# seed = 48956876
seed = 861412477
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# Measurement constants
# N_z_gen = 50000
# N_updates = 30
# grid_depth = 5

N_z_gen = 10 * 4 ** 5
N_updates = 50
grid_depth = 5

env_type = "perlin"
func = gen_env_fn_3d(env_type, seed=seed)

directory = Path(__file__).parents[0] / "results"
if not os.path.exists(directory):
    os.makedirs(directory)
directory /= "incremental"
directory /= "reobserve"
if not os.path.exists(directory):
    os.makedirs(directory)

load_iterations = True

if load_iterations == False:
    root_logger.debug("Generating new measurements and updating")
    grid = RelativeSubmap(max_depth=grid_depth, tolerance=1e-5)
    normalised_iterations = []
    iterations = []

    # NOTE This is a pseudo update. There is something weird with just the first update. It add more messages than needed I think
    C = 10000 * np.eye(3).reshape(1, 3, 3)
    grid.insert_measurements(np.zeros((1, 3, 1)), C)
    update_iter = grid.update()

    for i in range(N_updates):
        print("Update number:", i)
        # Generate Measurements
        print(i, "------- Generating Measurements -------")
        m_z_stack, C_z_stack = gen_meas_3d(func, N_z_gen, scale=1)
        b = (
            (m_z_stack[:, 0, 0] + m_z_stack[:, 1, 0] <= 1)
            & (m_z_stack[:, 0, 0] > 0)
            & (m_z_stack[:, 0, 0] < 1)
            & (m_z_stack[:, 1, 0] > 0)
            & (m_z_stack[:, 1, 0] < 1)
        )
        m_z_stack = m_z_stack[b, :, :]
        C_z_stack = C_z_stack[b, :, :]
        N_z = m_z_stack.shape[0]
        print(i, "N: total =", N_z_gen, " remainder = ", N_z)

        # Calculate approximation
        print(i, "------- Fusing Measurements -------")
        grid.insert_measurements(m_z_stack, C_z_stack)
        t0 = time.time()
        update_iter: MessageCounter = grid.update()
        normalised_iterations.append(len(update_iter) / N_z)
        iterations.append(len(update_iter))
        t1 = time.time()
        tot = t1 - t0
        print(i, "Approximation runtime:", tot, "s")

    np.savez(open(directory / "iterations.npz", "wb"), iterations, normalised_iterations)
else:
    root_logger.debug("Loading previously calculated results")
    # Load iterations if you want to reuse
    iterations = np.load(open(directory / "iterations.npz", "rb"))["arr_0"]
    normalised_iterations = np.load(open(directory / "iterations.npz", "rb"))["arr_1"]

ratio = 7 / 16
fig_scale = 0.98
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig = plt.figure(figsize=fs, constrained_layout=True)
ax = plt.gca()
# ax_right = ax.twinx()

p1, = ax.plot(iterations, marker=".", color="C0", markersize=4)
# p1, = ax.plot(np.array(iterations) / 4**grid_depth, marker=".", color="C0")
# p2, = ax_right.plot(normalised_iterations, marker=".", color="C1")

ax.set_xlabel("Time step")
ax.set_ylabel("Number of messages passed")
# ax.set_ylabel("Average number of messages passed per surfel")
# ax_right.set_ylabel("Normalised number of messages passed")

ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

# tkw = dict(size=4, width=1.5)
# ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
# ax_right.tick_params(axis="y", colors=p2.get_color(), **tkw)

# base = 10
# ax.set_ylim(0, base * np.ceil(np.max(iterations) * 1.05 / base))
# ax_right.set_ylim(0, base * np.ceil(np.max(normalised_iterations) * 1.05 / base))

# N_surfels = 4**grid_depth
# ln = ax.axhline(N_surfels, color="r", linestyle="-.", label="Number of surfels")
# bgcol = ax.get_facecolor()
# ax.text(
#     0.2,
#     N_surfels,
#     "Number of surfels",
#     ha="center",
#     va="center",
#     rotation="horizontal",
#     backgroundcolor=bgcol,
#     transform=ln.get_transform(),
# )
# ax.legend()

# plt.xticks(fontsize=tex_fontsize - 1)
# plt.yticks(fontsize=tex_fontsize - 1)

filename = directory / f"{env_type}_{seed}.pdf"
plt.savefig(filename)

# plt.show()
