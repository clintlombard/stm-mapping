import logging
import os

import cv2
import dill
import numpy as np

from tqdm import tqdm

from stmmap.decouple import Decouple
from stmmap.ltr_map import LTRMap
from stmmap.preprocess_data import *
from stmmap.relative_submap import RelativeSubmap, SensorType
from stmmap.utils.read import readLidar

logger = logging.getLogger(__name__)


class MapBuilder:
    def __init__(
        self, dirs, check_triangulation=False, manual_lms=[], grid_depth=4, window_size=0
    ):
        # Directories dictionary
        self.dirs = dirs

        # Load SLAM and sensor timestamps
        self.sensor_extrinsics = {}
        self.timestamps = {}

        self.timestamps["slam"] = load_slam(self.dirs)
        self.timestamps["slam"] = self.timestamps["slam"]
        stereo_ts, stereo_extrinsics, Q, u_limits, v_limits = load_stereo(self.dirs)
        if stereo_ts != []:
            self.timestamps["stereo"] = stereo_ts
            self.sensor_extrinsics[SensorType.STEREO] = stereo_extrinsics
            self.Q = Q
            self.u_limits = u_limits
            self.v_limits = v_limits
            self.f = 1 * Q[2, 3]
            self.B = 1 / Q[3, 2]
        ldr_ts, ldr_extrinsics = load_lidar(self.dirs)
        if ldr_ts != []:
            self.timestamps["lidar"] = ldr_ts
            R = ldr_extrinsics[0]
            t = ldr_extrinsics[1]
            # Transform from robot->cam (lidar_extrinsics go from lidar->cam)
            R_ext_stereo = stereo_extrinsics[0]
            t_ext_stereo = stereo_extrinsics[1]
            R_ext_ldr = R_ext_stereo.dot(R)
            # t_ext_ldr = t + t_ext_stereo
            t_ext_ldr = R_ext_stereo.dot(t) + t_ext_stereo

            self.sensor_extrinsics[SensorType.LIDAR] = (R_ext_ldr, t_ext_ldr)

        # Calculate LTR map
        self.ltr_map = LTRMap(self.timestamps["slam"], self.dirs, check_triangulation, manual_lms)

        # Create grids for each LTR
        self.submaps = {}
        for c in self.ltr_map.cliques:
            # A submap is stored for each sensor modularity
            sensor_subgrid_dict = {}
            for sensor_t in SensorType:
                sensor_subgrid_dict[sensor_t] = RelativeSubmap(
                    grid_depth, Q, window_size=window_size
                )
            self.submaps[c] = sensor_subgrid_dict

        # Dimensionality of robot and landmark states
        self.RBT_DIM = 7
        self.LM_DIM = 3
        self.MEAS_DIM = 3

        self.decouple = Decouple(self.RBT_DIM + self.LM_DIM * 3 + self.MEAS_DIM)

        # TODO: Read these in
        std_u = std_v = 1
        std_d = std_u * std_v
        std_r = 0.01
        std_angles = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
        self.max_stereo_dist = 15

        self.sensor_covs = {}
        cov_stereo = np.diag([std_u ** 2, std_v ** 2, std_d ** 2])
        self.sensor_covs[SensorType.STEREO] = cov_stereo
        cov_ldr = np.diag([std_r ** 2, std_angles ** 2, std_angles ** 2])
        self.sensor_covs[SensorType.LIDAR] = cov_ldr

    def run(self, start_index, steps, update_steps, subsamples):
        """Incrementally build map.

        steps: Skip increment between measurements. (-1 disables the sensor type.)
        update_steps: Increments between calculating the local measurement
        factors. If -1 then only update after all measurements have been added.

        """

        N_slam = len(self.timestamps["slam"])
        update_flag = False
        counts = {SensorType.STEREO: 0, SensorType.LIDAR: 0}
        update_counts = {SensorType.STEREO: 0, SensorType.LIDAR: 0}
        for index in tqdm(range(start_index, N_slam)):
            # logger.debug(f"Index: {index}")
            # Read in slam estimate
            filename_mean = "mean-" + self.timestamps["slam"][index][1] + ".npy"
            filename_cov = "cov-" + self.timestamps["slam"][index][1] + ".npy"
            if self.timestamps["slam"][index][0] == "stereo":
                sensor_type = SensorType.STEREO
            elif self.timestamps["slam"][index][0] == "lidar":
                sensor_type = SensorType.LIDAR
            else:
                logger.debug("Error: Invalid sensor parsed: {self.timestamps['slam'][index][0]}")
                exit()

            meas_index = self.timestamps["slam"][index][2]
            self.slam_mean = np.load(self.dirs["slam"] + filename_mean)
            self.slam_cov = np.load(self.dirs["slam"] + filename_cov)

            # Use the sum of the distances to each lm as a metric for choosing
            # the nearest LTR
            tri_dists = []
            rbt_pos = self.slam_mean[:3]
            for c in self.ltr_map.cliques:
                d = 0
                for lm_id in c:
                    tmp = self.RBT_DIM + self.LM_DIM * lm_id
                    lm = self.slam_mean[tmp : tmp + self.LM_DIM, :]
                    # Check that the ltr has actually been observed
                    if (lm == 0).all():
                        d = np.inf
                        continue
                    else:
                        d += np.linalg.norm(rbt_pos - lm)
                tri_dists.append(d)
            grid_index = np.argmin(tri_dists)
            tri = tuple(self.ltr_map.cliques[grid_index])

            if (sensor_type == SensorType.STEREO) and (steps[SensorType.STEREO] > 0):
                if (counts[SensorType.STEREO] % steps[SensorType.STEREO]) == 0:
                    update_flag = True
                    counts[SensorType.STEREO] = 1
                else:
                    counts[SensorType.STEREO] += 1
                    continue

                img = cv2.imread(
                    self.dirs["disparity"] + self.timestamps["stereo"][meas_index], cv2.CV_16UC1
                )

                rows = np.arange(0, img.shape[0], dtype=float)
                cols = np.arange(0, img.shape[1], dtype=float)

                u, v = np.meshgrid(cols, rows)
                u = u.flatten()
                v = v.flatten()
                d = np.array(img.flatten(), dtype=float)
                # Divide by 16 because SGBM returns int disparities scaled by 16
                d /= 16
                # Remove disparities which would result in a z-distance greater than a max
                # Or any negative disparities
                b = (u > self.u_limits[0]) & (u < self.u_limits[1])
                b &= (v > self.v_limits[0]) & (v < self.v_limits[1])
                b &= (d > 0) & (d > (self.f * self.B) / self.max_stereo_dist)

                tmp_u = u.flatten()[b]
                uvd_homo = np.ones((4, tmp_u.size), dtype=float)
                uvd_homo[0, :] = tmp_u
                uvd_homo[1, :] = v.flatten()[b]
                uvd_homo[2, :] = d.flatten()[b]

                pts_homo = self.Q.dot(uvd_homo)
                pts = (pts_homo[:3, :] / pts_homo[3, :]).T

                # Order points according to robot coordinates
                # axes x,y,z (cam) -> -y,-z,x (cam robot-orientation)
                reorder = [2, 0, 1]
                pts = pts[:, reorder]
                pts[:, 1] *= -1
                pts[:, 2] *= -1

                raw = np.array([u.flatten()[b], v.flatten()[b], d[b]]).T

            if (sensor_type == SensorType.LIDAR) and (steps[SensorType.LIDAR] > 0):
                if (counts[SensorType.LIDAR] % steps[SensorType.LIDAR]) == 0:
                    update_flag = True
                    counts[SensorType.LIDAR] = 1
                else:
                    counts[SensorType.LIDAR] += 1
                    continue

                raw = readLidar(
                    self.dirs["ldr"] + self.timestamps["lidar"][meas_index][0],
                    fov=50,
                    min_range=1.1,
                )  # Spherical data

                if raw.shape[1] != 3:
                    logger.error(f"Lidar point dimensions incorrect: {raw.shape}")
                    exit()
                # Order points according to robot coordinates
                pts = np.zeros_like(raw, dtype=float)

                pts[:, 0] = raw[:, 0] * np.cos(raw[:, 2]) * np.cos(raw[:, 1])
                pts[:, 1] = raw[:, 0] * np.cos(raw[:, 2]) * np.sin(raw[:, 1])
                pts[:, 2] = raw[:, 0] * np.sin(raw[:, 2])

                # pts[:, 0] = raw[:, 0] * np.cos(raw[:, 1])
                # pts[:, 1] = raw[:, 0] * np.sin(raw[:, 1])
                # pts[:, 2] = 0

            if update_flag:
                update_flag = False

                sub = subsamples[sensor_type]
                pts = pts[::sub, :]
                raw = raw[::sub, :]

                self.update(pts, raw, tri, sensor_type)

                if update_counts[sensor_type] == update_steps[sensor_type]:
                    update_counts[sensor_type] = 0
                    for tri in self.ltr_map.cliques:
                        submap = self.submaps[tri][sensor_type]
                        if submap.needs_update:
                            logger.debug(f"Updating submap: {tri}")
                            submap.update()

                        # NOTE: Saving even if it hasn't changed
                        # Create output folders
                        save_dir = self.dirs["submap"] + "/" + sensor_type.name + "/" + str(tri)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        # Save submap for visualisation purposes
                        filename = save_dir + "/%07d.p" % (index)
                        submap = self.submaps[tri][sensor_type]
                        dill.dump(dict(submap.surfel_dict), open(filename, "wb"))
                else:
                    update_counts[sensor_type] += 1

        for sensor_type in SensorType:
            if update_steps[sensor_type] == -1:
                update_counts[sensor_type] = 0
                for tri in self.ltr_map.cliques:
                    submap = self.submaps[tri][sensor_type]

                    if submap.needs_update:
                        logger.debug(f"Updating submap: {tri}")
                        submap.update()

                    # NOTE: Saving even if it hasn't changed
                    # Create output folders
                    save_dir = self.dirs["submap"] + "/" + sensor_type.name + "/" + str(tri)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # Save submap for visualisation purposes
                    filename = save_dir + "/%07d.p" % (index)
                    submap = self.submaps[tri][sensor_type]
                    dill.dump(dict(submap.surfel_dict), open(filename, "wb"))

    def rigid_transform(self, pose):
        translate = pose[:3].reshape(3, 1)
        if pose[3:].size != 4:
            logger.debug("Orientation must be a quaternion!")
            exit()
        q0 = pose[3:].reshape(4, 1)
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
        return (R, translate)

    def update(self, pts, raw, tri, sensor_type, original_tri=()):
        # Slice relevant part of the slam estimates
        indices = [j for j in range(self.RBT_DIM)]
        for t in tri:
            for k in range(3):
                indices.append(self.RBT_DIM + 3 * t + k)
        indices.sort()

        slam_mean_slice = self.slam_mean[indices, :]
        slam_cov_slice = self.slam_cov[indices, :][:, indices]

        lms = slam_mean_slice[self.RBT_DIM :].reshape((3, 3))

        if (lms == 0).any():
            logger.debug(f"Trying to update an unobserved LTR {tri}")
            return None

        R_rbt, t_rbt = self.rigid_transform(slam_mean_slice[: self.RBT_DIM])
        R_ext = self.sensor_extrinsics[sensor_type][0]
        t_ext = self.sensor_extrinsics[sensor_type][1]

        # Transform from robot to sensor coordinates (applied to landmarks)
        R = R_ext.T.dot(R_rbt.T)
        t = R_ext.T.dot(R_rbt.T.dot(-t_rbt) - t_ext)

        transform = (R, t)

        pts_rel, a0, a1, a2, a3 = self.decouple.test_points(pts, lms, transform)

        if np.count_nonzero(a0) > 0:
            raw_valid = raw[a0]

            cov_z = self.sensor_covs[sensor_type]
            self.submaps[tri][sensor_type].process_raw(
                raw_valid,
                slam_mean_slice,
                slam_cov_slice,
                cov_z,
                sensor_type,
                self.sensor_extrinsics,
            )

        # Don't try update neighbours if there is only one ltr
        if self.ltr_map.N_LTRS == 1:
            return

        neighbours = self.ltr_map.neighbours[tri]
        # a1
        if np.count_nonzero(a1) > 0:
            neigh_tri = neighbours[1]
            if neigh_tri != () and neigh_tri != original_tri:
                self.update(pts[a1], raw[a1], neigh_tri, sensor_type, tri)
        # a2
        if np.count_nonzero(a2) > 0:
            neigh_tri = neighbours[0]
            if neigh_tri != () and neigh_tri != original_tri:
                self.update(pts[a2], raw[a2], neigh_tri, sensor_type, tri)
        # a3
        if np.count_nonzero(a3) > 0:
            neigh_tri = neighbours[2]
            if neigh_tri != () and neigh_tri != original_tri:
                self.update(pts[a3], raw[a3], neigh_tri, sensor_type, tri)
