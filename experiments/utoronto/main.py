#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import yaml

from stmmap import RelativeSubmap

from .utils import *

loggers: List[logging.Logger] = []
loggers += [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("--batch", help="Enable batch processing of input data.", action="store_true")
# NOTE: maybe it is better to specify the desire resolution in meters?
parser.add_argument("--grid_depth", help="Depth of each grid subdivision.", type=int, default=4)
parser.add_argument("--seed", help="Manually specify the noise seed.", type=int, default=None)
parser.add_argument("--vis_div", help="Enable visualising the submaps.", action="store_true")
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

# Check pre-existing map results
now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
save_path = main_path / date_time
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

t_start = time.time()
root_logger.debug("Timing started")

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

seed = None
if args.seed is not None:
    seed = args.seed
else:
    seed = (int)(np.random.rand() * (2.0 ** 30))
np.random.seed(seed)
random.seed(seed)

# Some flags
grid_depth = args.grid_depth
batch = args.batch
vis_div = args.vis_div
subsample = args.subsample

# Measurement covariance
std_r = 0.01
std_azi = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
std_elev = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
cov = np.diag([std_r ** 2, std_azi ** 2, std_elev ** 2])

# Read in dataset
pts_sensor_list, pos_list, gt_list, N_total = read_all_data(main_path, subsample=subsample)
N = len(pts_sensor_list)  # Number of measurement poses

# Preallocate memory for batch
if batch:
    batch_pts = np.zeros((N_total, 3, 1), dtype=float)
    batch_covs = np.zeros((N_total, 3, 3), dtype=float)
    b_index = 0

# Extract Landmarks
if dataset_specific_flags["p2at_met"]:
    lms, triangulation = manually_extract_pseudo_lms(pos_list, keep=(36, 13, 28), div=1)
elif dataset_specific_flags["box_met"]:
    lms, triangulation = manually_extract_pseudo_lms(pos_list, keep=(30, 64, 96), div=1)
else:
    pseudo_lms, triangulation = extract_pseudo_lms(pos_list)

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
if vis_div and not args.remote:
    root_logger.debug("Visualising map divisons.")
    # Plot submap divisions
    mlab.triangular_mesh(
        *lms[:, :, 0].T, triangulation.simplices, colormap="viridis", representation="wireframe"
    )

    # Plot grid divisions

    # Plot all robot positions
    pos_arr = np.hstack(pos_list)
    mlab.points3d(*pos_arr, pos_arr[2, :], colormap="viridis", scale_mode="none", scale_factor=1)

    mlab.show()

simplices = triangulation.simplices

# Create submaps for each LTR
root_logger.debug("Initialising submaps")
N_regions = simplices.shape[0]
# N_store = N_total // N_regions # TODO re-add this...
submaps = np.empty(N_regions, dtype=RelativeSubmap)
for i in range(N_regions):
    submaps[i] = RelativeSubmap(grid_depth)

submaps_dict_arr = np.empty(N_regions, dtype=dict)
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

    if batch:
        N = pts_world.shape[0]
        batch_pts[b_index : b_index + N, :, :] = pts_world
        batch_covs[b_index : b_index + N, :, :] = covs
        b_index += N
        continue

    # Associate points
    selections = find_approx_associations(triangulation, pts_world)

    # Convert to relative IRF and store in ltr
    for i, b in selections.items():
        N_ltr = np.sum(b)
        if N_ltr > 0:
            pts_ltr = pts_world[b, :, :]
            covs_ltr = covs[b, :, :]
            lms_ltr = lms[simplices[i]]
            submaps[i].process_utoronto(pts_ltr, covs_ltr, lms_ltr)

    # Update local map
    root_logger.debug("Updating local factor of map")
    for i in range(N_regions):
        root_logger.debug(f"Updating region: {i}/{N_regions}")
        submaps[i].update()

    # Save submaps for visualisation purposes
    root_logger.debug("Saving submaps")
    for i in range(N_regions):
        submaps_dict_arr[i] = submaps[i].surfel_dict
    filename = submap_path / f"{index:03d}.p"
    pickle.dump(submaps_dict_arr, open(filename, "wb"))

if batch:
    root_logger.debug("Processing batch measurements.")
    # Remove unused memory (due to deleted measurements)
    batch_pts = batch_pts[:b_index, :, :]
    batch_covs = batch_covs[:b_index, :, :]

    # Associate points
    selections = find_approx_associations(triangulation, batch_pts)

    # Convert to relative IRF and store in ltr
    for i, b in selections.items():
        N_ltr = np.sum(b)
        if N_ltr > 0:
            pts_ltr = batch_pts[b, :, :]
            covs_ltr = batch_covs[b, :, :]
            lms_ltr = lms[simplices[i]]
            submaps[i].process_utoronto(pts_ltr, covs_ltr, lms_ltr)

    # Update local map
    root_logger.debug("Updating local factor of map")
    for i in range(N_regions):
        root_logger.debug(f"Updating region: {i}/{N_regions}")
        t0 = time.time()
        submaps[i].update()
        t1 = time.time()
        root_logger.debug(f"Time taken to update: {t1 - t0}s")

    # Save submaps for visualisation purposes
    root_logger.debug("Saving map")
    for i in range(N_regions):
        submaps_dict_arr[i] = submaps[i].surfel_dict
    filename = submap_path / f"{index:03d}.p"
    pickle.dump(submaps_dict_arr, open(filename, "wb"))
    root_logger.debug("Saving successful")

t_end = time.time()
root_logger.debug(f"Total time taken: {t_end - t_start}s")
