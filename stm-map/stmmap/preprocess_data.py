# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Preprocessing for sensor data

Author: Clint Lombard
"""

import logging

from operator import itemgetter

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Need this to read OpenCV yaml files...
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix)


def load_slam(dirs):
    slam_ts = []
    with open(dirs["slam"] + "timestamps.txt", "r") as stream:
        for line in stream.readlines():
            split = line.split()
            # NOTE: format = [stereo/lidar slam_index meas_index timestamp]
            slam_ts.append([split[0], split[1], int(split[2]), float(split[3])])
    return slam_ts


def load_stereo(dirs):
    stereo_ts = []
    try:
        with open(dirs["disparity"] + "timestamps.txt", "r") as stream:
            for line in stream.readlines():
                split = line.split()
                stereo_ts.append(split[0])  # Only take filename
    except Exception as e:
        logger.exception("Error reading in disparity data, probably missing.")
    try:
        # Extrinsic parameters from Q matrix
        with open(dirs["img"] + "extrinsics.yml", "r") as stream:
            # NOTE: Remove first line from file because opencv's yaml standard is outdated...
            lines = stream.readlines()[1:]
            long_string = ""
            for line in lines:
                long_string += line

            data = yaml.load(long_string, Loader=yaml.Loader)
            Q = data["Q"]

            valid_roi1 = data["Valid Roi 1"]
            valid_roi2 = data["Valid Roi 2"]

            # NOTE: Setting v_min to help prevent speckles from the disparities
            # Value chosen experimentally
            # v_min = max(valid_roi1[1], valid_roi2[1])
            v_min = 200
            v_max = min(valid_roi1[1] + valid_roi1[3], valid_roi2[1] + valid_roi1[3])

            u_min = max(valid_roi1[0], valid_roi2[0])
            u_max = min(valid_roi1[0] + valid_roi1[2], valid_roi2[0] + valid_roi2[2])
            u_limits = [u_min, u_max]
            v_limits = [v_min, v_max]

        with open(dirs["main"] + "/cam_rbt_extrinsics.yml", "r") as stream:
            data = yaml.load(stream, Loader=yaml.Loader)
            R_ext_stereo = np.array(data["R"])

        # angles = np.deg2rad([-0.28666603863, 13.7945079522, 0])

        # R1 = transformations.rot_x(angles[0])
        # R2 = transformations.rot_y(angles[1])
        # R3 = transformations.rot_z(angles[2])
        # R_ext_stereo = (R1.dot(R2).dot(R3))
        t_ext_stereo = np.array([[0.0, 0.0, 0.0]]).T

    except yaml.YAMLError as e:
        logger.error(f"Error parsing calibration files.")
        raise e
    except Exception as e:
        logger.error("Error reading in stereo data, probably missing.")
        raise e

    return stereo_ts, (R_ext_stereo, t_ext_stereo), Q, u_limits, v_limits


def load_lidar(dirs):
    ldr_ts = []
    try:
        with open(dirs["ldr"] + "timestamps.txt", "r") as stream:
            for line in stream.readlines():
                split = line.split()
                ldr_ts.append([split[0], float(split[1])])
            ldr_ts = sorted(ldr_ts, key=itemgetter(1))
        with open(dirs["ldr"] + "lidar_extrinsics.yml") as stream:
            RT_ext_ldr = np.array(yaml.load(stream, Loader=yaml.Loader)["RT"])

        # Convert to cam -> lidar transform
        R = RT_ext_ldr[:3, :3]
        t = RT_ext_ldr[:3, 3].reshape((3, 1))

        t[0] = -0.08
        t[1] = -0.17430850463719597 / 2
        t[2] = -0.10

        # 18May2019
        # t[2] -= 0.05
        # t[0] += 0.05

        return ldr_ts, (R, t)
    except Exception as e:
        logger.exception("Error reading in lidar data, probably missing.")
        return [], ()
