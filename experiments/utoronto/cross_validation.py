#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate STM mapping against elevation mapping on practical data using holdout and comparing the
resulting model log-likelihoods.

"""

import argparse
import logging
import os
import pickle
import random
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import List

import dill
import mayavi.mlab as mlab  # XXX Maybe only import this if not remote
import numpy as np
import pandas as pd
import yaml

from scipy.spatial import Delaunay

from compare_surface_maps.elevation_maps import ElevationMap3D
from stmmap import RelativeSubmap

from .utils import *

loggers: List[logging.Logger] = []
loggers += [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]
loggers += [logging.getLogger("compare_surface_maps.elevation_maps")]

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
# NOTE: maybe it is better to specify the desire resolution in meters?
parser.add_argument(
    "--hold", help="Percentage of samples to keep for testing.", type=float, default=0.2
)
parser.add_argument("--repeat", help="Number of times to repeat training.", type=int, default=10)
parser.add_argument("--grid_depth", help="Depth of each grid subdivision.", type=int, default=4)
parser.add_argument("--seed", help="Manually specify the noise seed.", type=int, default=None)
parser.add_argument("--vis", help="Enable visualising the submaps.", action="store_true")
parser.add_argument(
    "--remote",
    help="Enable remote mode, i.e. root_logger.debug to file instead of terminal.",
    action="store_true",
)
parser.add_argument(
    "--subsample",
    help="The amount of points to subsample by (0 does no subsampling).",
    type=int,
    default=0,
)

args = parser.parse_args()

main_path = Path(args.path).resolve()  # Get abs path

seed = None
if args.seed is not None:
    seed = args.seed
else:
    seed = (int)(np.random.rand() * (2.0 ** 30))
np.random.seed(seed)
random.seed(seed)

# Check pre-existing map results
save_path = Path(__file__).resolve().parent / "results" / "cross_validation" / str(seed)
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise FileExistsError("Folder (%s) already exits." % (save_path))

submap_path = save_path / "submaps"
if not os.path.exists(submap_path):
    os.makedirs(submap_path)

# XXX Setup logging
level = logging.DEBUG  # This could be controlled with --log=DEBUG (I think)
output_file = save_path / "output.log"

# create formatter and add it to the handlers
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

# create file handler which logs even debug messages
fh = logging.FileHandler(output_file, mode="w")
fh.setLevel(level)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(level)
ch.setFormatter(formatter)

for logger in loggers:
    logger.setLevel(level)
    logger.addHandler(fh)
    # If not remote the output to console as well
    if not args.remote:
        logger.addHandler(ch)

root_logger = loggers[0]
root_logger.debug("Logging initialised")

dataset_specific_flags = {}
if "p2at_met" in str(main_path):
    root_logger.debug("Found p2at_met DS")
    dataset_specific_flags["p2at_met"] = True
else:
    dataset_specific_flags["p2at_met"] = False
if "box_met" in str(main_path):
    root_logger.debug("Found box_met DS")
    dataset_specific_flags["box_met"] = True
else:
    dataset_specific_flags["box_met"] = False

# Some flags
grid_depth = args.grid_depth
vis = args.vis
subsample = args.subsample
holdout_ratio = args.hold
train_repetitions = args.repeat

# Measurement covariance
std_r = 0.01
std_azi = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
std_elev = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
cov = np.diag([std_r ** 2, std_azi ** 2, std_elev ** 2])

# Read in dataset
pts_sensor_list, pos_list, gt_list, N_total = read_all_data(main_path, subsample=subsample)
N = len(pts_sensor_list)  # Number of measurement poses

# Preallocate memory for batch
batch_pts = np.zeros((N_total, 3, 1), dtype=float)
batch_covs = np.zeros((N_total, 3, 3), dtype=float)
b_index = 0

# Extract Landmarks
if dataset_specific_flags["p2at_met"]:
    raise NotImplementedError("Can only handle 'box_met' dataset")
elif dataset_specific_flags["box_met"]:
    lms = np.array(
        [
            [-100.8191813, -5.98536893, -2.90530072],
            [-74.87969074, -8.822031, -2.84433425],
            [-71.73638351, 10.16799498, -3.06947789],
            [-99.64019653, 12.85220453, -1.79018201],
        ]
    ).reshape(-1, 3, 1)
    lms[:, -1, 0] = np.mean(lms[:, -1, 0])
    triangulation = Delaunay(lms[:, :2, 0])
else:
    raise NotImplementedError("Can only handle 'box_met' dataset")

# Save map setup
setup = {}
setup["lms"] = lms
setup["triangulation"] = triangulation
setup["grid_depth"] = grid_depth
setup["subsample"] = subsample
setup["seed"] = seed
pickle.dump(setup, open(save_path / "setup.p", "wb"))
yaml.dump(setup, open(save_path / "setup.yaml", "w"))

# Visualise divisions (if flag true)
if vis and not args.remote:
    root_logger.debug("Visualising map divisons.")
    # Plot submap divisions
    mlab.triangular_mesh(
        *lms[:, :].T, triangulation.simplices, colormap="viridis", representation="wireframe"
    )

    # Plot all robot positions
    pos_arr = np.hstack(pos_list)
    mlab.points3d(*pos_arr, pos_arr[2, :], colormap="viridis", scale_mode="none", scale_factor=1)

    mlab.show()

simplices = triangulation.simplices

root_logger.debug("Reading in measurements")
for index in range(N):
    root_logger.debug(f"Increment: {index}")
    # Convert from sensor to inertial IRF
    transform = gt_list[index]
    pts_sensor = pts_sensor_list[index]
    pts_world, covs = sensor_to_world(transform, pts_sensor, cov)

    # Remove outlier points. Height threshhold was exprimentally determined to
    # be sufficient. A smarter way would be to remove points far away from
    # other points (i.e. KNN), but this would be very expensive.
    if dataset_specific_flags["p2at_met"]:
        b = pts_world[:, 2, 0] <= 2
    elif dataset_specific_flags["box_met"]:
        b = pts_world[:, 2, 0] <= 1
    pts_world = pts_world[b, :]
    covs = covs[b, :]

    N = pts_world.shape[0]
    batch_pts[b_index : b_index + N, :, :] = pts_world
    batch_covs[b_index : b_index + N, :, :] = covs
    b_index += N

root_logger.debug("Processing batch measurements.")
# Remove unused memory (due to deleted measurements)
batch_pts = batch_pts[:b_index, :, :]
batch_covs = batch_covs[:b_index, :, :]

# Associate points
selections = find_approx_associations(triangulation, batch_pts)

log_like_stm_batch = np.zeros(train_repetitions, dtype=float)
log_like_elev_batch = np.zeros(train_repetitions, dtype=float)
for n_rep in range(train_repetitions):
    # Create submaps for each LTR
    root_logger.debug(f"Starting repetition {n_rep+1}")
    N_regions = simplices.shape[0]
    stm_submaps = np.empty(N_regions, dtype=RelativeSubmap)
    elev_submaps = np.empty(N_regions, dtype=ElevationMap3D)
    for i in range(N_regions):
        stm_submaps[i] = RelativeSubmap(grid_depth)
        elev_submaps[i] = ElevationMap3D(grid_depth)

    stm_submaps_dict_arr = np.empty(N_regions, dtype=dict)
    elev_submaps_dict_arr = np.empty(N_regions, dtype=dict)

    # Convert to relative IRF and store in ltr
    for i, b in selections.items():
        N_ltr = np.sum(b)
        if N_ltr > 0:
            lms_ltr = lms[simplices[i]]
            pts_ltr = batch_pts[b, :, :]
            covs_ltr = batch_covs[b, :, :]

            # Split into training and test datasets
            N_test = int(holdout_ratio * N_ltr)
            test_indices = random.sample(range(N_ltr), N_test)
            test_mask = np.zeros(N_ltr, dtype=bool)
            test_mask[test_indices] = True
            train_mask = ~test_mask
            pts_train = pts_ltr[train_mask]
            covs_train = covs_ltr[train_mask]

            pts_test = pts_ltr[test_mask]
            covs_test = covs_ltr[test_mask]
            pts_rel_test, _ = stm_submaps[i].process_utoronto(
                pts_test, covs_test, lms_ltr, test=True
            )

            t0 = time.time()
            stm_submaps[i].process_utoronto(1 * pts_train, 1 * covs_train, 1 * lms_ltr)
            stm_submaps[i].update()
            t1 = time.time()
            root_logger.debug(f"Time taken to update STM map region: {t1 - t0}s")
            elev_submaps[i].process_utoronto(1 * pts_train, 1 * covs_train, 1 * lms_ltr)
            t2 = time.time()
            root_logger.debug(f"Time taken to update elevation map region: {t2 - t1}s")

            root_logger.debug("Calculating log-likelihoods")
            stm_ll = stm_submaps[i].loglike(1 * pts_rel_test)
            log_like_stm_batch[n_rep] = stm_ll
            root_logger.debug(f"STM log-likelihood: {stm_ll}")
            elev_ll = elev_submaps[i].loglike(1 * pts_rel_test)
            log_like_elev_batch[n_rep] = elev_ll
            root_logger.debug(f"Elevation log-likelihood: {elev_ll}")

    # Save submaps for visualisation purposes
    root_logger.debug("Saving submaps")
    for i in range(N_regions):
        stm_submaps_dict_arr[i] = stm_submaps[i].surfel_dict
        elev_submaps_dict_arr[i] = elev_submaps[i].surfel_dict

    repetition_save_path = submap_path / f"{n_rep}"
    if not os.path.exists(repetition_save_path):
        os.makedirs(repetition_save_path)

    filename = repetition_save_path / f"stm.p"
    pickle.dump(stm_submaps_dict_arr, open(filename, "wb"))

    filename = repetition_save_path / f"elev.p"
    pickle.dump(elev_submaps_dict_arr, open(filename, "wb"))
    root_logger.debug("Saving successful")

data_ll = {"STM Map": log_like_stm_batch, "Elevation Map": log_like_elev_batch}
df_ll = pd.DataFrame(data_ll)
df_ll.to_pickle(save_path / "likelihoods.p")

log_like_ratios = log_like_stm_batch - log_like_elev_batch
log_like_ratio_mean = np.mean(log_like_ratios)
log_like_ratio_std = np.std(log_like_ratios)
root_logger.debug(f"Log likelihood-ratio mean: {log_like_ratio_mean}")
root_logger.debug(f"Log likelihood-ratio std: {log_like_ratio_std}")
