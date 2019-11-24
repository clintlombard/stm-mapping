#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse maps from different (stationary) view points.

NOTE This is limited to stereo data. (I think...)

Created on 28 May 2019 07:46

@author Clint Lombard

"""
import argparse
import ast
import logging
import os

from typing import List

import dill
import numpy as np

from stmmap import RelativeSubmap, Surfel

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("sensor_type", metavar="LIDAR/STEREO", type=str, default="STEREO")
parser.add_argument(
    "--remote",
    help="Enable remote mode, i.e. root_logger.debug to file instead of terminal.",
    action="store_true",
)

args = parser.parse_args()

main_path = args.path

sensor_type = args.sensor_type

# XXX Setup logging
loggers: List[logging.Logger] = []
loggers += [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]

level = logging.DEBUG  # This could be controlled with --log=DEBUG (I think)
output_file = os.path.join(main_path, "output.log")

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

# 1 Find all dataset folders
candidate_folders = [
    os.path.join(main_path, name)
    for name in os.listdir(main_path)
    if os.path.isdir(os.path.join(main_path, name))
]
root_logger.debug(candidate_folders)

# Double check they contain valid dataset
ds_folders = []
submap_folder = os.path.join("submap", sensor_type)
require = ["Cameras/", "slam/", submap_folder]
for i, f in enumerate(candidate_folders):
    add_dir = True
    for r in require:
        if not os.path.isdir(os.path.join(f, r)):
            add_dir = False
            break
    if add_dir:
        path = os.path.join(f, submap_folder)
        ds_folders.append(path)

root_logger.debug(ds_folders)
assert len(ds_folders) >= 2, "Need at least 2 maps to perform fusion."

# 2 Initialise maps with the first folders maps
f = ds_folders[0]
map_dirs = [name for name in os.listdir(f) if os.path.isdir(os.path.join(f, name))]
if len(map_dirs) <= 0:
    raise ValueError("No submaps folders found")
else:
    root_logger.debug(f"Found maps: {map_dirs}")

submaps = dict()
for m in map_dirs:
    map_dir = os.path.join(f, m)
    map_names = [name for name in os.listdir(map_dir) if name.endswith(".p")]
    if len(map_names) == 0:
        raise ValueError("No submaps found in folders")
    map_names.sort()
    root_logger.debug(map_names)

    # Read in latest submap
    filename = os.path.join(map_dir, map_names[-1])
    submap_surf_dict = dill.load(open(filename, "rb"))

    depth = np.log(len(submap_surf_dict)) / np.log(4)
    assert (depth >= 0) and ((depth % 1) == 0)
    submap = RelativeSubmap(depth)
    submap.surfel_dict = submap_surf_dict
    submap.surfel_ids = submap_surf_dict.keys()

    key = ast.literal_eval(m)
    submaps[key] = submap

# 3 Fuse the local models in each submap
root_logger.debug("Fusing submaps' surfels")
for f in ds_folders[1:]:
    for m in submaps.keys():
        map_dir = os.path.join(f, str(m))
        if not os.path.isdir(map_dir):
            root_logger.debug(f"No matching submap region found for {m} in {f}")
            continue

        map_names = [name for name in os.listdir(map_dir) if name.endswith(".p")]
        if len(map_names) > 0:
            map_names.sort()

            # Read in latest submap
            filename = os.path.join(map_dir, map_names[-1])
            submap_surf_dict = dill.load(open(filename, "rb"))

            depth = np.log(len(submap_surf_dict)) / np.log(4)
            assert (depth >= 0) and ((depth % 1) == 0)

            submaps[m].fuse_submaps(submap_surf_dict)
        else:
            root_logger.debug(f"Empty submap folder for {m} in {f}")

root_logger.debug("Updating fused submaps")
for key, submap in submaps.items():
    root_logger.debug(f"Updating submap {key}")
    submap.update()


# 4 Save result
root_logger.debug("Saving fused submaps")
for m in submaps.keys():
    root_logger.debug(f"Saving {m}")
    submap = submaps[m]

    save_path = os.path.join(main_path, "submap", sensor_type, str(m))
    root_logger.debug(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, "0.p")
    dill.dump(dict(submap.surfel_dict), open(filename, "wb"))
