# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Main mapping script

This exectutes my mapping algorithm. First measurements are decoupled and then
finally fused into the map.

Clint Lombard

"""
import argparse
import logging
import random
import time

from pathlib import Path
from typing import List

import dill
import numpy as np

from stmmap import MapBuilder, SensorType

loggers: List[logging.Logger] = []
loggers += [logging.getLogger(__name__)]
loggers += [logging.getLogger("stmmap")]

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("--check_tri", help="Check triangulation", action="store_true")
parser.add_argument("--depth", help="Grid depth", type=int, default=4)
parser.add_argument(
    "--fusion",
    help="Whether this is a stationary fusing dataset (keeps all measurements).",
    action="store_true",
)
parser.add_argument("--seed", help="Noise seed.", type=int)
parser.add_argument(
    "--remote",
    help="Enable remote mode, i.e. print to file instead of terminal.",
    action="store_true",
)
parser.add_argument(
    "--stereo_steps",
    help="Subsample the stereo measurements in time (-1 disables the sensor).",
    type=int,
    default=10,
)
parser.add_argument(
    "--lidar_steps",
    help="Subsample the lidar measurements in time (-1 disables the sensor).",
    type=int,
    default=-1,
)
parser.add_argument(
    "--stereo_subsample", help="Subsample stereo data in each measurement.", type=int, default=1
)
parser.add_argument(
    "--lidar_subsample", help="Subsample lidar data in each measurement.", type=int, default=1
)
parser.add_argument(
    "--stereo_update",
    help="When to perform map update (-1 to perform only after all data is acquired)",
    type=int,
    default=-1,
)
parser.add_argument(
    "--lidar_update",
    help="When to perform map update (-1 to perform only after all data is acquired)",
    type=int,
    default=-1,
)

args = parser.parse_args()
check_triangulation = args.check_tri

main_path = Path(args.path).resolve()  # Get abs path

# # Check pre-existing map results
# now = datetime.now()
# date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
# save_path = main_path / date_time
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# else:
#     raise FileExistsError("Folder (%s) already exits." % (save_path))

dirs = dict()
dirs["main"] = str(main_path)
dirs["slam"] = dirs["main"] + "/slam/"
dirs["img"] = dirs["main"] + "/Cameras/"
dirs["disparity"] = dirs["main"] + "/Cameras/disparity/"
dirs["sensor_calib"] = dirs["main"] + "/sensor-calib/"
dirs["submap"] = dirs["main"] + "/submap/"
dirs["ldr"] = dirs["main"] + "/Lidar/"

# XXX: Global variables
# std_u = std_v = 3
# std_d = 9
# std_r = 0.01
# std_angles = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)

# Set the noise seed for both random and numpy
seed = (int)(np.random.rand() * (2.0 ** 30))
if args.seed is not None:
    seed = args.seed
np.random.seed(seed)
random.seed(seed)

# XXX Setup logging
level = logging.DEBUG  # This could be controlled with --log=DEBUG (I think)
timestr = time.strftime("%Y-%m-%d-%H%M%S")
output_file = main_path / f"output-{timestr}.log"

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

if "17Oct2019" in str(main_path):
    manual_lms = [(0, 1, 2), (0, 2, 3)]  # 17Oct2019
else:
    manual_lms = [
        (0, 1, 2),
        (1, 2, 3),
    ]  # 18May2019 wall_scan_01, 26May2019 stationary_heap01, 08June2019 static_00, 04May2019-14h23
# manual_lms = [(0,1,2)] # 08June2019 static_01
depth = args.depth
window_size = -1 if args.fusion else 0
mb = MapBuilder(
    dirs, check_triangulation, manual_lms=manual_lms, grid_depth=depth, window_size=window_size
)
# mb = MapBuilder(dirs, check_triangulation, grid_depth=depth, debug=debug)
start_index = 0

# XXX Subsample the sensor sample frequency (Negative to disable)
# steps = {SensorType.STEREO: -1, SensorType.LIDAR: 1}  # All lidar, no images
# steps = {SensorType.STEREO: np.inf, SensorType.LIDAR: -1}; start_index = -1 # Only one image, no lidar
# steps = {SensorType.STEREO: 20, SensorType.LIDAR: -1} # Some images, no lidar
# steps = {SensorType.STEREO: 1, SensorType.LIDAR: -1} # All images, no lidar
steps = {SensorType.STEREO: int(args.stereo_steps), SensorType.LIDAR: int(args.lidar_steps)}

# XXX When to perform global update (-1 to perform only after all data is acquired)
# update_steps = {SensorType.STEREO: -1, SensorType.LIDAR: -1}
update_steps = {SensorType.STEREO: args.stereo_update, SensorType.LIDAR: args.lidar_update}

# XXX Subsample sensor data in each measurement
# subsamples = {SensorType.STEREO: 1, SensorType.LIDAR: 1}
# subsamples = {SensorType.STEREO: 50, SensorType.LIDAR: 1}
subsamples = {SensorType.STEREO: args.stereo_subsample, SensorType.LIDAR: args.lidar_subsample}

mb.run(start_index, steps, update_steps, subsamples)
