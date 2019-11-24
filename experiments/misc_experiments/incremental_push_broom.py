#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Experiment: Incremental updating in the global PGM.

This is an investigation on the effects of updating the global map with
incremental batches of measurements. Specifically we look at how the normalised_iterations
change.

Clint Lombard

"""
import argparse
import copy
import logging
import os
import random

from pathlib import Path
from typing import List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mayavi.mlab as mlab
import numpy as np
import seaborn as sns

from noise import pnoise2
from opensimplex import OpenSimplex
from tqdm import tqdm
from transformations import *

from stmmap import RelativeSubmap
from stmmap.utils.distances import DistanceMetric, compare_maps
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_3d, gen_meas_3d

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# -------- Handle command line args -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Noise seed.", type=int)
parser.add_argument(
    "--env",
    help="Environment type ('planar', 'perlin','simplex', 'step')",
    type=str,
    default="perlin",
)
args = parser.parse_args()

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
if args.seed is not None:
    seed = args.seed
np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

grid_depth = 5

env_type = args.env
func = gen_env_fn_3d(env_type)

directory = Path(__file__).parents[0] / "results"
if not os.path.exists(directory):
    os.makedirs(directory)
directory /= "incremental"
directory /= "push-broom"
if not os.path.exists(directory):
    os.makedirs(directory)

# -------- Experiment 1: Incremental push-broom style samples ---------------------------
print("------- Generating Measurements -------")
scans_indices = []
N = 1280
m_z_batch = np.array([])
C_z_batch = np.array([])
steps = 5 * 3 + 1
# print(np.linspace(0, 1, steps, endpoint=False))
for start in np.linspace(0, 1, steps, endpoint=False):
    root_logger.debug(f"Batch increment at {start}")
    depth = 1 / steps

    m_a = np.random.rand(N)
    # m_a = np.random.rand(N) * (1 - start)
    m_b = depth * np.random.rand(N) + start

    valid = (m_a + m_b) <= 1
    valid &= (m_a <= 1) & (m_a >= 0)
    valid &= (m_b <= 1) & (m_b >= 0)
    N_valid = np.sum(valid)
    m_a = m_a[valid]
    m_b = m_b[valid]

    m_g = func(m_a, m_b)
    m_z = np.array([m_a, m_b, m_g]).T.reshape(N_valid, 3, 1)

    scale = 0.05
    std_a = 0.01 * scale
    std_b = 0.01 * scale
    std_g = 0.02 * scale
    std_z = np.diag([std_a, std_b, std_g])

    alpha_rot = (np.pi * np.random.random_sample(N_valid) - np.pi / 2) * 1
    beta_rot = (np.pi * np.random.random_sample(N_valid) - np.pi / 2) * 1
    gamma_rot = (np.pi * np.random.random_sample(N_valid) - np.pi / 2) * 1
    xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    C_z = np.empty((N_valid, 3, 3), dtype=float)
    for j in tqdm(range(N_valid)):
        # Rotate Covariance Matrix
        Rx = rotation_matrix(alpha_rot[j], xaxis)
        Ry = rotation_matrix(beta_rot[j], yaxis)
        Rz = rotation_matrix(gamma_rot[j], zaxis)
        R = concatenate_matrices(Rx, Ry, Rz)[:-1, :-1]

        C = R.dot(std_z * std_z).dot(R.T)
        e = np.random.multivariate_normal(np.zeros(3), C, 1).T

        m_z[j, :, :] += e
        C_z[j] = C

    valid = (m_z[:, 0, 0] + m_z[:, 1, 0]) <= 1
    valid &= (m_z[:, 0, 0] <= 1) & (m_z[:, 0, 0] >= 0)
    valid &= (m_z[:, 1, 0] <= 1) & (m_z[:, 1, 0] >= 0)
    m_z = m_z[valid]
    C_z = C_z[valid]
    print("N_actual:", m_z.shape[0])
    if m_z_batch.size == 0:
        m_z_batch = np.copy(m_z)
        C_z_batch = np.copy(C_z)
    else:
        m_z_batch = np.vstack((m_z_batch, m_z))
        C_z_batch = np.vstack((C_z_batch, C_z))
    scans_indices.append(m_z.shape[0])

# # XXX Plot measurements in hexbin
# plt.figure(constrained_layout=True)
# plt.hexbin(*m_z_batch[:, :2, 0].T, gridsize=50)
# plt.xlim(0, 1)
# plt.ylim(0, 1)

grid_iterations = []
grid = RelativeSubmap(max_depth=grid_depth, tolerance=1e-5, prior_corr=0.5)
# grid.plot_map_indices()
# plt.show()
# exit()
grid_iterations.append(copy.deepcopy(grid.surfel_dict))
normalised_iterations = []
iterations = []

# NOTE This is a pseudo update. There is something weird with just the first update. It add more messages than needed I think
C = 10000 * np.eye(3).reshape(1, 3, 3)
grid.insert_measurements(np.zeros((1, 3, 1)), C)
update_iter = grid.update()

# XXX Uncomment if you want update the whole map once before push-broom
# N_z_gen = 10*4**5
# m_z_stack, C_z_stack = gen_meas_3d(func, N_z_gen, scale=1)
# b = (
#     (m_z_stack[:, 0, 0] + m_z_stack[:, 1, 0] <= 1)
#     & (m_z_stack[:, 0, 0] > 0)
#     & (m_z_stack[:, 0, 0] < 1)
#     & (m_z_stack[:, 1, 0] > 0)
#     & (m_z_stack[:, 1, 0] < 1)
# )
# m_z_stack = m_z_stack[b, :, :]
# C_z_stack = C_z_stack[b, :, :]
# N_z_new = m_z_stack.shape[0]
# print("N: total =", N_z_gen, " remainder = ", N_z_new)

# grid.insert_measurements(m_z_stack, C_z_stack)
# update_iter = grid.update()

# normalised_iterations.append(len(update_iter) / 1)
# iterations.append(len(update_iter))
# grid_iterations.append(copy.deepcopy(grid.surfel_dict))

start = 0
for i in range(len(scans_indices)):
    stop = start + scans_indices[i]
    N_z = stop - start
    grid.insert_measurements(m_z_batch[start:stop, :, :], C_z_batch[start:stop, :, :])
    update_iter = grid.update()
    start = stop

    normalised_iterations.append(len(update_iter) / N_z)
    iterations.append(len(update_iter))
    grid_iterations.append(copy.deepcopy(grid.surfel_dict))

ratio = 0.9
fig_scale = 0.48
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig = plt.figure(figsize=fs, constrained_layout=True)
ax = plt.gca()
ax_right = ax.twinx()

p1, = ax.plot(iterations, marker=".", color="C0")
p2, = ax_right.plot(normalised_iterations, marker=".", color="C1")

ax.set_xlabel("Time step")
ax.set_ylabel("Number of messages passed")
ax_right.set_ylabel("Normalised number of messages passed")

ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

# ax.yaxis.label.set_color(p1.get_color())
# ax_right.yaxis.label.set_color(p2.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
ax_right.tick_params(axis="y", colors=p2.get_color(), **tkw)

base = 10
ax.set_ylim(0, base * np.ceil(np.max(iterations) * 1.05 / base))
ax_right.set_ylim(0, base * np.ceil(np.max(normalised_iterations) * 1.05 / base))
# sns.despine()

filename = directory / f"{grid_depth}_{N}_{steps}_incremental.pdf"
fig.savefig(filename)

filename = directory / f"{grid_depth}_{N}_{steps}_incremental.png"
fig.savefig(filename)

# XXX Plot differences between incremental maps
ncols = 4
step = (len(grid_iterations) - 1) // (ncols - 1)
# ncols = (len(grid_iterations) - 1) // step + 1
ratio = 4.8 / (4.8 * ncols + 0.4)
fig_scale = 0.98
fs = (fig_scale * plt_cfg.tex_textwidth, fig_scale * plt_cfg.tex_textwidth * ratio)
fig, axes = plt.subplots(
    figsize=fs,
    squeeze=False,
    sharex=True,
    sharey=True,
    nrows=1,
    ncols=ncols,
    constrained_layout=True,
)
ax_iter = iter(axes.flat)
vmin, vmax = (np.inf, -np.inf)
for i in range(0, len(grid_iterations) - 1, step):
    grid0 = grid_iterations[i]
    grid1 = grid_iterations[i + 1]
    distances = compare_maps(grid0, grid1, metric=DistanceMetric.EXCLUSIVE_KLD)

    N = len(grid0)
    pts = np.empty((3 * N, 2), dtype=float)
    D = np.empty(N, dtype=float)
    for j, path in enumerate(grid.surfel_ids):
        surfel = grid0[path]
        corners = 1 * surfel.corners
        pts[3 * j : (3 * j + 3), :] = corners.T
        D[j] = distances[path]
    vmin = min(vmin, np.min(D))
    vmax = max(vmax, np.max(D))

print(vmin, vmax)

for i in range(0, len(grid_iterations) - 1, step):
    grid0 = grid_iterations[i]
    grid1 = grid_iterations[i + 1]
    distances = compare_maps(grid0, grid1, metric=DistanceMetric.EXCLUSIVE_KLD)

    N = len(grid0)
    pts = np.empty((3 * N, 2), dtype=float)
    D = np.empty(N, dtype=float)
    for j, path in enumerate(grid.surfel_ids):
        surfel = grid0[path]
        corners = 1 * surfel.corners
        pts[3 * j : (3 * j + 3), :] = corners.T
        D[j] = distances[path]

    triangles = np.arange(0, pts.shape[0]).reshape((int(pts.shape[0] / 3), 3))

    ax = next(ax_iter)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    norm = colors.SymLogNorm(linthresh=1e5, vmin=vmin, vmax=vmax)
    trip = ax.tripcolor(
        *pts.T,
        triangles,
        edgecolors="none",
        facecolors=D,
        norm=norm,
        linewidth=0.5,
        cmap="viridis",
    )
    ax.triplot(*pts[:, :2].T, triangles, color="k", linewidth=0.1)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_title(rf"$t_{ {i} }$")

    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.colorbar(trip, ax=axes.ravel().tolist(), label="Exclusive KL divergence")

filename = directory / f"{grid_depth}_{N}_{steps}_distances.pdf"
fig.savefig(filename, interpolation=None)


if __name__ == "__main__":
    plt.show()
