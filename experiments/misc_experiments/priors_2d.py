# -*- coding: utf-8 -*-
"""A 2-D simulation investigating the difference between using different
priors over the global heights

"""
import logging
import os
import random

from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from matplotlib import rc
from matplotlib.cm import ScalarMappable

from stmmap import RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig
from utils.generate_environments import gen_env_fn_2d, gen_meas_2d

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

# Set noise seed
seed = (int)(np.random.rand() * (2.0 ** 30))
seed = 654981589
np.random.seed(seed)
random.seed(seed)

# env_type = 'perlin'
env_type = "simonsberg"
env_func = gen_env_fn_2d(env_type)

directory = Path(__file__).parents[0] / "results"
if not os.path.exists(directory):
    os.makedirs(directory)
directory /= "priors"
if not os.path.exists(directory):
    os.makedirs(directory)

N = 10000  # Number of measurements
m_z, C_z = gen_meas_2d(env_func, N, scale=0.1)
depth = 6

fig_width = 0.98 * plt_cfg.tex_textwidth
ratio = 24 / 9
fig_height = fig_width / ratio
fig = plt.figure(figsize=(fig_width, fig_height), dpi=150, constrained_layout=True)
ax = plt.gca()
# norm=colors.SymLogNorm(linthresh=0.875, linscale=0.01, vmin=0.0, vmax=1.0)
# norm = colors.PowerNorm(10, vmin=0, vmax=1)
cmap = ScalarMappable(cmap="plasma_r")

# XXX: Occlusion is set here
# Hole occlusion
# center = 0.5
# occl_start = center - 2**-depth*3
# occl_stop = center + 2**-depth*4
# b = (m_raw[:, 0,:] > occl_start) & (m_raw[:, 0,:] < occl_stop)
# bin_alloc[b.flatten()] = -1

# # Half occlusion
# center = 0.5
# occl_start = center
# occl_stop = 1
# b = (m_raw[:, 0,:] > occl_start)
# bin_alloc[b.flatten()] = -1

# Hole occlusion
center = 0.5
occl_start = center - 2 ** -depth * 2
occl_stop = center + 2 ** -depth * 2
b = (m_z[:, 0, 0] > occl_start) & (m_z[:, 0, 0] < occl_stop)
# b = ~b
print(b.shape)
m_z = m_z[b, :, :]
C_z = C_z[b, :, :]

ax.axvspan(
    occl_start, occl_stop, color="g", linestyle=":", fill=True, alpha=0.3, label="Observed region"
)

N_corrs = 7
corr = 0
corr_range = [0]
for i_plot in range(1, N_corrs + 1):
    # Correlated prior
    print("Corr:", corr)
    submap = RelativeSubmap(
        max_depth=depth, dim=2, tolerance=1e-8, max_iterations=100, prior_corr=corr
    )
    submap.insert_measurements(m_z, C_z)
    submap.update()

    # Plot resulting map
    col = cmap.to_rgba(corr, norm=False)
    # col = cmap.to_rgba(corr)
    for i, surfel in enumerate(submap):
        edges = surfel.corners.reshape(2)
        heights = surfel.bel_h_m.reshape(2)
        variation = surfel.bel_v_b / surfel.bel_v_a
        stds = np.ones(2) * variation ** 0.5

        tmp = plt.plot(edges, heights, "-o", color=col, markersize=2)
        # col = tmp[0].get_color()
        # plt.fill_between(
        #     edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=0
        # )
    corr += 2 ** -i_plot
    corr_range += [corr]

A = np.linspace(0, 1, 5)
# corr_range += [1]
cmap.set_array(A)
cbar = plt.colorbar(cmap, format="%.1f", ticks=A)
cbar.set_label(r"Correlation coefficient ($\rho$)")
y_min, y_max = (1.1 * np.min(m_z[:, 1, 0]), 1.1 * np.max(m_z[:, 1, :]))
ax.set_xlabel(r"$\alpha$")
# ax.set_ylim(y_min, y_max)
ax.set_xlim(0, 1)
plt.xticks(fontsize=plt_cfg.tex_fontsize - 1)
plt.yticks(fontsize=plt_cfg.tex_fontsize - 1)
ax.set_ylabel(r"$\gamma$")

plt.legend()

filename = directory / f"priors.pdf"
fig.savefig(filename)

plt.show()
