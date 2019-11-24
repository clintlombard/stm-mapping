import logging
import os
import random
import time

import mayavi.mlab as mlab
import numpy as np

from stmmap import RelativeSubmap
from utils.generate_environments import gen_env_fn_3d, gen_meas_3d

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
env_type = "perlin"
env_func = gen_env_fn_3d(env_type, seed)

depth = 4
submap = RelativeSubmap(max_depth=depth, dim=3, tolerance=1e-5, max_iterations=500)

N = 500  # Number of measurements
m_z, C_z = gen_meas_3d(env_func, N, scale=0.9)
submap.insert_measurements(m_z, C_z)
msg_counter = submap.update()
print(msg_counter)

# m_z, C_z = gen_meas_3d(env_func, N, scale=0.9)
# submap.insert_measurements(m_z, C_z)
# msg_counter = submap.update()
# print(msg_counter)

# XXX: Analysis
# Plot Samples
# pts = mlab.points3d(
#     *m_z[:, :, 0].T, m_z[:, 2, 0], scale_mode="none", colormap="viridis", scale_factor=0.005
# )

# Plot exact surface
step = 0.01
N_steps = int(1 / step) + 1
a_arr = []
b_arr = []
for i in range(N_steps):
    a_tmp = np.ones(N_steps - i) * (step * i)
    b_tmp = step * np.arange(0, N_steps - i)
    a_arr.append(a_tmp)
    b_arr.append(b_tmp)
a = np.hstack(a_arr)
b = np.hstack(b_arr)
g = env_func(a, b)

pts = mlab.points3d(a, b, g, g, scale_mode="none", scale_factor=0.0)
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.contour_surface(mesh, colormap="viridis", contours=50)

# Plot fitted model
n_surfels = submap
points3d = np.empty((3 * submap.n_surfels, 3))
variations = np.empty((3 * submap.n_surfels))
for i, surfel in enumerate(submap):
    points3d[(3 * i) : (3 * i + 3), :2] = surfel.corners.T
    points3d[(3 * i) : (3 * i + 3), 2] = surfel.bel_h_m.flat
    variations[(3 * i) : (3 * i + 3)] = surfel.bel_v_b / surfel.bel_v_a


triangulation = np.arange(0, points3d.shape[0]).reshape((int(points3d.shape[0] / 3), 3))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# tri_surf = ax.plot_trisurf(
#     points3d[0, :],
#     points3d[1, :],
#     points3d[2, :],
#     triangles=triangulation,
#     cmap="viridis",
#     alpha=1)
# fig.colorbar(tri_surf, shrink=0.5)
# plt.show()

# # mlab.points3d(*points3d)
tri_mesh = mlab.triangular_mesh(
    *points3d.T, triangulation, scalars=variations, colormap="viridis", opacity=1
)
mlab.colorbar(tri_mesh, orientation="vertical")
mlab.show()
