import ast
import csv
import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_yaml_opencv():
    def opencv_matrix(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat

    yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix)


def readOdo(filename):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # Linear velocity in mm/s
            lin_vel = float(data["lin_vel"])
            lin_vel /= 1000  # m/s
            # Rotational velocity in deg/s
            rot_vel = float(data["rot_vel"])
            rot_vel *= np.pi / 180  # rad/s
            ut = np.array([lin_vel, 0, 0, rot_vel])
            return ut
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading odometry from file: {filename}")
            return []


def readLandmarks(filename, max_reading=15, ignore_right_cam=True, ignore_lm_list=[17]):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            ct = []
            zt = []
            if data is None:
                return zt, ct
            for meas in data:
                # Camera identification
                if "14191992" in filename:
                    sensor_id = 0  # Left camera
                else:
                    sensor_id = 1  # Right camera
                    if ignore_right_cam:
                        continue
                lm_id = meas[0]
                if lm_id in ignore_lm_list:
                    continue
                ct.append([lm_id, sensor_id])  # [lm_id, sensor_id]
                # Convert to correct axes
                # tmp_meas = [meas[1][2], meas[1][0], meas[1][1]]
                tmp_meas = [meas[1][2], -meas[1][0], -meas[1][1]]
                p = np.linalg.norm(tmp_meas)
                if p > max_reading:
                    logger.warn(f"Measurement distance exceeded: {p} > {max_reading}")
                    return [], []
                # Calculate polar coordinates
                azi = np.arctan2(tmp_meas[1], tmp_meas[0])
                elev = np.arcsin(tmp_meas[2] / p)
                zt.append([p, azi, elev])
            return zt, ct
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading landmarks from file: {filename}")
            return [], []


def readLidar(filename, fov=135, min_range=0.4):
    """Read lidar data

    output: (rho, azi, elev)
    """
    ldr_stream = open(filename, "r")
    ldr_csv_reader = csv.reader(ldr_stream, delimiter=",")
    ldr_data = []
    for row in ldr_csv_reader:
        row = [float(row[0]), float(row[1])]
        if row[1] > min_range and row[1] < 19 and row[0] < fov and row[0] > -fov:
            tmp = [row[1], np.deg2rad(row[0]), 0.0]
            ldr_data.append(tmp)
    data_arr = np.array(ldr_data)
    return data_arr


def readDecoupledTimestamps(filename):
    ts = []
    regions = set()  # Find all the unique trianglulated regions
    with open(filename, "r") as stream:
        for line in stream.readlines():
            split = line.split()
            tri = ast.literal_eval(", ".join(split[3:]))
            ts.append([int(split[0]), split[1], split[2], tuple(tri)])
            regions.add(tuple(tri))
    sort_crit = lambda x: x[0]
    ts.sort(key=sort_crit)
    return ts, list(regions)


def readCVCalib(filename):
    load_yaml_opencv()
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            R = data["R"].T
            T_tmp = -R.dot(data["T"].reshape(3, 1))
            T = np.array([T_tmp[2], T_tmp[0], T_tmp[1]])

            RT = np.eye(4, 4)
            RT[:3, :3] = R
            RT[:3, 3] = T[:, 0]
            return RT
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading OpenCV camera calibration from file: {filename}")
            return np.array([])


def readLidarCalib(filename):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            RT = np.asarray(data["RT"])
            return RT
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading lidar-camera calibration from file: {filename}")
            return np.array([])


def readRobotCalib(filename):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            RT = np.asarray(data["RT"])
            return RT
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading camera-robot calibration from file: {filename}")
            return np.array([])


def readIMUGravVector(filename, accl_scale=0.333e-3):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            grav_vec = np.zeros(3, dtype=float)
            grav_vec[0] = data["xaccl"] * accl_scale * 9.80655
            grav_vec[1] = data["yaccl"] * accl_scale * 9.80655
            grav_vec[2] = data["zaccl"] * accl_scale * 9.80655
            return grav_vec
        except yaml.YAMLError as exc:
            logger.exception(f"Error reading imu gravity vector from file: {filename}")
            return np.array([])


def readIMUMeas(imu_meas_raw, timestamp, gyro_scale=0.0125, accl_scale=0.333e-3):
    meas = np.zeros(6, dtype=float)
    if imu_meas_raw[0] != timestamp[0]:
        logger.warn("Timestamp mismatch. Look at sorting timestamps and data")
        return None

    meas[0] = float(imu_meas_raw[4]) * accl_scale * 9.80655
    meas[1] = float(imu_meas_raw[5]) * accl_scale * 9.80655
    meas[2] = float(imu_meas_raw[6]) * accl_scale * 9.80655
    meas[3] = np.deg2rad(float(imu_meas_raw[1]) * gyro_scale)
    meas[4] = np.deg2rad(float(imu_meas_raw[2]) * gyro_scale)
    meas[5] = np.deg2rad(float(imu_meas_raw[3]) * gyro_scale)
    return meas


# def readIMUMeas(imu_data, timestamp, gyro_scale=0.05, accl_scale=0.333e-3):
#     meas_yaml = imu_data[timestamp]
#     meas = np.zeros(6, dtype=float)
#     meas[0] = meas_yaml['xaccl'] * accl_scale * 9.80655
#     meas[1] = meas_yaml['yaccl'] * accl_scale * 9.80655
#     meas[2] = meas_yaml['zaccl'] * accl_scale * 9.80655
#     meas[3] = np.deg2rad(meas_yaml['xgyro'] * gyro_scale)
#     meas[4] = np.deg2rad(meas_yaml['ygyro'] * gyro_scale)
#     meas[5] = np.deg2rad(meas_yaml['zgyro'] * gyro_scale)
#     return meas
