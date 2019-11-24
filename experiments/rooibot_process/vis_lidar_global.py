# -*- coding: utf-8 -*-
#!/usr/bin/python3
import argparse
import ast
import math
import multiprocessing as mp
import os
import pickle
import sys

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from matplotlib import colors, rc

from stmmap.preprocess_data import *
from stmmap.utils.read import *

try:
    os.environ["ETS_TOOLKIT"] = "wx"
    import mayavi.mlab as mlab
except Exception as e:
    import mayavi.mlab as mlab

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()

sns.set_style("whitegrid")
sns.set_palette("RdBu")
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
fontSize = 12
sns.set_context(
    "paper",
    font_scale=1.0,
    rc={
        "axes.linewidth": 0.75,
        "font.size": fontSize,
        "axes.labelsize": fontSize,
        "xtick.labelsize": fontSize,
        "ytick.labelsize": fontSize,
        "legend.fontsize": fontSize,
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
    },
)


parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("--nolight", help="Disable lighting", action="store_true")

args = parser.parse_args()
dirs = dict()
dirs["main"] = args.path
dirs["slam"] = dirs["main"] + "/slam/"
dirs["img"] = dirs["main"] + "/Cameras/"
dirs["disparity"] = dirs["main"] + "/Cameras/disparity/"
dirs["ldr"] = dirs["main"] + "/Lidar/"

slam_ts = load_slam(dirs)
# Select only lidar SLAM indices
df = pd.DataFrame(slam_ts)
slam_ts = df[df[0] == "lidar"].to_numpy()

# Load stereo_extrinsics
stereo_ts, stereo_extrinsics, Q, u_limits, v_limits = load_stereo(dirs)

ldr_ts, ldr_extrinsics = load_lidar(dirs)
if ldr_ts != []:
    R = ldr_extrinsics[0]
    t = ldr_extrinsics[1]

    t[0] = -0.08
    t[1] = -0.17430850463719597 / 2
    t[2] = -0.12

    # # 18May2019/wall_scan01/
    # t[2] -= 0.05
    # t[0] += 0.05
    # Transform from robot->cam (lidar_extrinsics go from cam->lidar)
    R_ext_stereo = stereo_extrinsics[0]
    t_ext_stereo = stereo_extrinsics[1]
    R_ext_ldr = R_ext_stereo.dot(R)
    t_ext_ldr = R_ext_stereo.dot(t) + t_ext_stereo
else:
    raise ValueError("No lidar measurements found...")


def para_fn(params):
    index = params
    t = slam_ts[index][3]
    if t < 358:  # 17Oct2019-18h33
        return

    z_arr = readLidar(dirs["ldr"] + ldr_ts[index][0], fov=40, min_range=1.1).T

    # Transform points to global coordinates
    # Spherical to cartesian
    xyz = np.zeros_like(z_arr)
    xyz[0, :] = z_arr[0, :] * np.cos(z_arr[2, :]) * np.cos(z_arr[1, :])
    xyz[1, :] = z_arr[0, :] * np.cos(z_arr[2, :]) * np.sin(z_arr[1, :])
    xyz[2, :] = z_arr[0, :] * np.sin(z_arr[2, :])

    # Sensor to robot
    pts_rbt = R_ext_ldr.dot(xyz) + t_ext_ldr

    # Robot to global
    filename_mean = "mean-" + slam_ts[index][1] + ".npy"
    slam_mean = np.load(dirs["slam"] + filename_mean)
    pose = slam_mean[:7]

    translate = pose[:3, 0].reshape(3, 1)
    q0 = pose[3:7].reshape(4, 1)
    R = np.array(
        [
            [
                q0[0, 0] ** 2 - q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                2 * (q0[0, 0] * q0[1, 0] - q0[2, 0] * q0[3, 0]),
                2 * (q0[0, 0] * q0[2, 0] + q0[1, 0] * q0[3, 0]),
            ],
            [
                2 * (q0[0, 0] * q0[1, 0] + q0[2, 0] * q0[3, 0]),
                -q0[0, 0] ** 2 + q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                2 * (q0[1, 0] * q0[2, 0] - q0[0, 0] * q0[3, 0]),
            ],
            [
                2 * (q0[0, 0] * q0[2, 0] - q0[1, 0] * q0[3, 0]),
                2 * (q0[1, 0] * q0[2, 0] + q0[0, 0] * q0[3, 0]),
                -q0[0, 0] ** 2 - q0[1, 0] ** 2 + q0[2, 0] ** 2 + q0[3, 0] ** 2,
            ],
        ]
    )
    pts_global = R.dot(pts_rbt) + translate

    return pts_global, t, translate


mlab.figure(size=(1920, 1080))

# Plot Landmarks
filename_mean = "mean-" + slam_ts[-1][1] + ".npy"
# filename_mean = "mean-" + slam_ts[5016][1] + ".npy"
slam_mean = np.load(dirs["slam"] + filename_mean)

lms = slam_mean[7:].reshape(-1, 3).T
print(lms.shape)
# mlab.points3d(*lms, colormap="viridis", scale_factor=0.1, scale_mode="vector")

indices = np.arange(0, len(slam_ts), 3).tolist()
pool = mp.Pool()
returns_list = pool.map(para_fn, indices)
points = np.array([])
time = np.array([])
rbt_path = np.array([])
for arr in returns_list:
    if arr is None:
        continue
    if points.size == 0:
        points = arr[0]
        time = arr[1] * np.ones(arr[0].shape[1])
        rbt_path = np.array([arr[1], *arr[2].flatten()])
    else:
        points = np.hstack((points, arr[0]))
        t = arr[1] * np.ones((arr[0].shape[1]))
        time = np.hstack((time, t))
        rbt = np.array([arr[1], *arr[2].flatten()])
        rbt_path = np.vstack((rbt_path, rbt))

rbt_time = rbt_path[:, 0]
sort_indices = np.argsort(rbt_time)
rbt_time = rbt_time[sort_indices]
rbt_path = rbt_path[sort_indices, 1:]
# mlab.plot3d(*rbt_path.T, rbt_time, tube_radius=0.025, colormap="viridis")
path_plt = mlab.plot3d(*rbt_path.T, tube_radius=0.025, color=colors.to_rgb("tab:blue"))

# points = np.vstack((points, points[2, :]))
pts_plt = mlab.points3d(
    *points[:, :], time, colormap="viridis", scale_factor=0.03, scale_mode="vector"
)
# mlab.points3d(*points[:, :], points[2, :], colormap="viridis", scale_factor=0.03, scale_mode="vector")

if args.nolight:
    path_plt.actor.property.lighting = False
    pts_plt.actor.property.lighting = False

scene = engine.scenes[0]
scene.scene.background = (1.0, 1.0, 1.0)  # white background

mlab.show()
