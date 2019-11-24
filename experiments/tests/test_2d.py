import logging
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

from stmmap import RelativeSubmap
from utils.generate_environments import gen_env_fn_2d, gen_meas_2d

matplotlib.use("Qt5Agg")

# NOTE Uncomment this to use latex fonts
# plt.rc("text", usetex=True)
# font = {"family": "serif", "size": 10, "serif": ["Latin Modern Roman"]}
# plt.rc("font", **font)
# plt.rc("legend", **{"fontsize": 10})
# plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

# XXX Setup logging
level = logging.DEBUG
timestr = time.strftime("-%Y%m%d-%H%M%S")
output_file = os.path.basename(__file__)[:-3] + timestr + ".log"

logger = logging.getLogger("stmmap")
logger.setLevel(level)

# create formatter and add it to the handlers
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

# create file handler which logs even debug messages
# if output_file is not None:
#     fh = logging.FileHandler(output_file, mode="w")
#     fh.setLevel(level)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(level)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Set noise seed
seed = (int)(np.random.rand() * (2.0 ** 30))
seed = 654981589
np.random.seed(seed)
random.seed(seed)

# env_type = 'perlin'
env_type = "simonsberg"
env_func = gen_env_fn_2d(env_type)

# XXX: Flags
N = 1000  # Number of measurements
m_z, C_z = gen_meas_2d(env_func, N, scale=0.5)
depth = 4


# XXX Occlude some measurements
valid = m_z[:, 0, 0] > 0.5
m_z = m_z[valid]
C_z = C_z[valid]

submap = RelativeSubmap(max_depth=depth, dim=2, tolerance=1e-3, max_iterations=10)

submap.insert_measurements(m_z, C_z)
submap.update()

# XXX: Analysis
plt.figure(constrained_layout=True)
ax = plt.gca()

submap.plot_map_indices(ax)

# Plot resulting map
for i, surfel in enumerate(submap):
    edges = surfel.corners.reshape(2)
    heights = surfel.bel_h_m.reshape(2)
    variation = surfel.bel_v_b / surfel.bel_v_a
    stds = np.ones(2) * variation ** 0.5

    tmp = plt.plot(edges, heights, "C1")
    col = tmp[0].get_color()
    plt.fill_between(
        edges, heights + stds, heights - stds, alpha=0.5, color=col, zorder=5, linewidths=0
    )


# Plot ground truth
x = np.linspace(0, 1, 1000)
y = env_func(x)
ax.plot(x, y, "--C2", alpha=1, lw=1, zorder=6, label="Ground truth")

# Plot raw points
y_min, y_max = (1.1 * np.min(m_z[:, 1, 0]), 1.1 * np.max(m_z[:, 1, :]))
hex = ax.hexbin(
    *m_z[:, :, 0].T, cmap="Greys", linewidths=0, extent=[0, 1, y_min, y_max], antialiased=True
)

ax.set_xlabel(r"$\alpha$", fontsize=10)
ax.set_ylim(y_min, y_max)
ax.set_xlim(0, 1)

plt.legend()
plt.show()
