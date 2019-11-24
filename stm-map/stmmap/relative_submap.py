from __future__ import annotations

import copy
import logging

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pylab as plt
import numpy as np
import pyemdw

from matplotlib.patches import Circle
from scipy import linalg
from scipy.stats import multivariate_normal
from tqdm import tqdm

import stmmap.utils.triangle_transforms as tt

from stmmap.message_queue import MessagePriority, MessageQueue
from stmmap.projections import ProjectionMethod, VMPFactorised, VMPStructured
from stmmap.surfel import Surfel
from stmmap.unscented import Unscented
from stmmap.utils.distances import DistanceMetric, distance_gauss, distance_invgamma

logger = logging.getLogger(__name__)


@unique
class SensorType(Enum):
    # NOTE:  define __order__ to loop through SensorType
    __order__ = "STEREO LIDAR"

    STEREO = 0
    LIDAR = 1


@dataclass
class MessageCounter:
    init_msgs: int = 0
    local_msgs: int = 0
    global_msgs: int = 0

    def __sub__(self, msg):
        init_msgs = self.init_msgs - msg.init_msgs
        local_msgs = self.local_msgs - msg.local_msgs
        global_msgs = self.global_msgs - msg.global_msgs
        return MessageCounter(init_msgs, local_msgs, global_msgs)

    def __add__(self, msg):
        init_msgs = self.init_msgs + msg.init_msgs
        local_msgs = self.local_msgs + msg.local_msgs
        global_msgs = self.global_msgs + msg.global_msgs
        return MessageCounter(init_msgs, local_msgs, global_msgs)

    def __len__(self):
        return self.init_msgs + self.local_msgs + self.global_msgs


class RelativeSubmap:
    """A triangular tree defined in relative coordinates."""

    def __init__(
        self,
        max_depth: int = 4,
        Q: Optional[np.ndarray] = None,
        tolerance: float = 1e-2,
        dim: int = 3,
        max_iterations: int = 500,
        projection_method: Callable[..., ProjectionMethod] = VMPStructured,
        window_size: int = 0,
        prior_corr: float = 0.5,
    ):
        logger.debug("Initialising submap")

        self.dim = dim
        if self.dim not in [2, 3]:
            raise ValueError("Can only handle 2 or 3 dimensions")

        self.max_depth = max_depth

        self.surfel_dict: Dict[Tuple[int, ...], Surfel] = dict()
        self.surfel_ids = self.surfel_dict.keys()

        self.base = 1.0 / 2 ** max_depth
        if self.dim == 3:
            self.n_surfels = 4 ** max_depth  # number of surfels
        else:
            self.n_surfels = 2 ** max_depth  # number of surfels

        self.prior_corr = prior_corr

        self.unscented = Unscented(19)

        # Ids of only heights
        self.h_ids: List[int] = []

        # Approximation convergence parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Add reordering from image to sensor IRF (0,1,2) -> (2,-0,-1)
        if Q is not None:
            reorder = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
            )
            self.Q = reorder.dot(Q)
        else:
            self.Q = None

        self.needs_update = False

        self.projection_method = projection_method
        self.window_size = window_size

        logger.debug("Populating surfels")
        self._counter = 0
        self.id_mapping: Dict[int, Tuple[int, ...]] = {}
        if self.dim == 3:
            self._corner_ids_dict: Dict[Tuple[int, ...], np.ndarray] = dict()
            self._rv_id_counter = 0
            self._generate_3d_grid()
        else:
            self._generate_2d_grid()

        # Calibrate initial graph
        logger.debug("Calibrating initialisation")
        damping = 0.0 if self.dim == 2 else 1 / 3
        self.stmm_bp = pyemdw.STMMBP(damping)
        for surfel_id in self.surfel_ids:
            surfel = self[surfel_id]
            self.stmm_bp.update_graph(
                surfel.pot_h_m.reshape(self.dim), surfel.pot_h_C, surfel.rv_ids
            )
        n_msg_init = self.stmm_bp.calibrate()

        self.message_counter = MessageCounter()
        self.message_counter.init_msgs = n_msg_init
        logger.debug(f"Initial message counter {self.message_counter}")

        for surfel_id in self.surfel_ids:
            surfel = self[surfel_id]

            inmsg_h_h, inmsg_h_K = self.stmm_bp.get_incoming_message(surfel.rv_ids)

            surfel.update_incoming_messages(inmsg_h_h, inmsg_h_K)

        # Construct graph to match EMDW cluster graph
        self.graph: Dict[int, Tuple[int, ...]] = self.stmm_bp.get_graph()

        logger.debug("Submap successfully initialised")

    def __iter__(self):
        for surfel_id in self.surfel_ids:
            yield self[surfel_id]

    def __len__(self):
        return len(self.surfel_dict)

    def __getitem__(self, key: Tuple[int, ...]) -> Surfel:
        return self.surfel_dict[key]

    def __setitem__(self, key: Tuple[int, ...], value: Surfel):
        self.surfel_dict[key] = value

    def _generate_3d_grid(self, corners=None, depth=1):
        """
        A recursive function which populates the tree with triangle objects.
        """
        if corners is None:
            # NOTE: corners = [alphas, betas, 1 - (alpha + betas)]^T
            corners = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)

        if depth <= self.max_depth:
            # NOTE: numbering is arbitrary and doesn't correspond to paths
            for i in range(4):
                # Surfel 0
                if i == 0:
                    corners_ = tt.tri0_trans_inv(corners)
                    self._generate_3d_grid(corners_, depth + 1)
                # Surfel 1
                elif i == 1:
                    corners_ = tt.tri1_trans_inv(corners)
                    self._generate_3d_grid(corners_, depth + 1)
                # Surfel 2
                elif i == 2:
                    corners_ = tt.tri2_trans_inv(corners)
                    self._generate_3d_grid(corners_, depth + 1)
                # Surfel 3
                else:
                    corners_ = tt.tri3_trans_inv(corners)
                    self._generate_3d_grid(corners_, depth + 1)
        else:
            corners_ = corners[:2, :]
            corners_tup = [tuple(i) for i in corners_.T]
            rv_ids = []
            # Add rv_ids for heights
            for i in corners_tup:
                if i not in self._corner_ids_dict:
                    self._corner_ids_dict[i] = self._rv_id_counter
                    rv_ids.append(self._rv_id_counter)
                    self.h_ids.append(self._rv_id_counter)
                    self._rv_id_counter += 1
                else:
                    rv_ids.append(self._corner_ids_dict[i])

            # Calculate the incenter of the triangle and use it as a point to
            # determine the surfel_id.
            tmp = np.array([[2 ** 0.5, 1, 1]])
            incenter = np.sum(tmp * corners_, axis=1) / np.sum(tmp)
            incenter = np.hstack((incenter, np.sum(incenter)))
            surfel_id = tuple(np.floor(incenter / self.base).astype(int))

            graph_id = self._counter

            self[surfel_id] = Surfel(
                corners=corners_,
                uid=surfel_id,
                rv_ids=tuple(rv_ids),
                graph_id=self._counter,
                dim=self.dim,
                projection_method=self.projection_method,
                window_size=self.window_size,
                prior_corr=self.prior_corr,
            )

            self.id_mapping[graph_id] = surfel_id

            self._counter += 1

    def _generate_2d_grid(self, corners=None, depth=1):
        self.h_ids = [i for i in range(self.n_surfels + 1)]
        corners = np.array(self.h_ids) * self.base
        assert corners[0] == 0.0
        assert corners[-1] == 1.0

        for s in range(self.n_surfels):
            surfel_id = (s,)
            ids = self.h_ids[s : (s + 2)]
            corners_ = corners[s : (s + 2)].reshape(1, 2)
            graph_id = self._counter
            self[surfel_id] = Surfel(
                corners=corners_,
                uid=surfel_id,
                rv_ids=tuple(ids),
                graph_id=graph_id,
                dim=self.dim,
                projection_method=self.projection_method,
                window_size=self.window_size,
                prior_corr=self.prior_corr,
            )

            self.id_mapping[graph_id] = surfel_id

            self._counter += 1

    def insert_measurements(self, z_means, z_covs):
        """
        Insert already decoupled measurements into the tree.

        Parameters
          * z_means - (N, self.dim, 1)
          * z_covs - (N, self.dim, self.dim)
        """
        N = z_means.shape[0]
        if N == 0:
            return None

        logger.info(f"Inserting {N} measurements into surfels")
        assert z_means.shape == (N, self.dim, 1)
        assert z_covs.shape == (N, self.dim, self.dim)

        if self.dim == 3:
            sum_ab = np.sum(z_means[:, :2, 0], axis=1)
            pts_aug = np.insert(z_means[:, :2, 0], 2, sum_ab, axis=1)  # (N, self.dim)
            paths = np.floor(pts_aug / self.base).astype(int)
        else:
            pts_aug = z_means[:, 0, 0]
            paths = np.floor(pts_aug / self.base).astype(int).reshape(-1, 1)

        unique_paths = np.unique(paths, axis=0)

        for path in unique_paths:
            surfel_id = tuple(path)
            if surfel_id in self.surfel_ids:
                surfel = self[surfel_id]
            else:
                continue

            if self.dim == 3:
                b = (paths[:, 0] == path[0]) & (paths[:, 1] == path[1]) & (paths[:, 2] == path[2])
            else:
                b = paths[:, 0] == path[0]

            n_additional = np.sum(b)
            if n_additional > 0:
                z_new_means = z_means[b]
                z_new_covs = z_covs[b]

                surfel.insert_measurements(z_new_means, z_new_covs)

    def process_raw(self, pts_raw, slam_mean, slam_cov, cov_z, sensor_type, sensor_extrinsics):
        """
        Decouple raw measurements and insert into tree

        Parameters:
          pts_raw (N, 3)
            Raw measurements to decouple and fuse
          slam_mean (16, 1)
            SLAM mean (7 robot + 9 landmarks)
          slam_cov (16, 16)
            SLAM covariance
          cov_z (3, 3)
            Measurement covariance in the sensor IRF
          sensor_type (SensorType)
            The sensor type
          sensor_extrinsics (list)
            Rotation and translation.
        """
        if self.dim == 2:
            raise NotImplementedError()

        N = pts_raw.shape[0]
        assert pts_raw.shape == (N, 3)
        logger.info(f"Processing {N} measurements")

        # NOTE: The sigma points can be calculated before. Instead of for each
        # measurement. Just add the measurement to the last 3 elements of each
        # sigma point. Cov remains the same.
        mean = np.vstack((slam_mean, np.zeros((3, 1))))
        cov = linalg.block_diag(slam_cov, cov_z)

        sig = self.unscented.calcSigma(mean, cov)  # (D=19, N_sig=39)
        N_sig = sig.shape[1]

        # Transform LM sigma pts to sensor frame
        l0 = sig[7:10, :].T
        la = sig[10:13, :].T
        lb = sig[13:16, :].T

        for i, s in enumerate(sig.T):
            t = s[:3].reshape(3, 1)
            q0 = s[3:7].reshape(4, 1)
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

            l0[i, :] = R.T.dot(l0[i, :].reshape(3, 1) - t).flat
            la[i, :] = R.T.dot(la[i, :].reshape(3, 1) - t).flat
            lb[i, :] = R.T.dot(lb[i, :].reshape(3, 1) - t).flat

        R_sens = sensor_extrinsics[sensor_type][0]
        t_sens = sensor_extrinsics[sensor_type][1]

        # Transform landmarks to sensor frame
        l0 = R_sens.T.dot(l0.reshape(N_sig, 3).T - t_sens).T
        la = R_sens.T.dot(la.reshape(N_sig, 3).T - t_sens).T
        lb = R_sens.T.dot(lb.reshape(N_sig, 3).T - t_sens).T

        da = la - l0
        db = lb - l0

        n = np.cross(da, db).T
        n /= np.linalg.norm(n, axis=0)

        # Create combined sigma point array for all measurements (N, 3, N_sig)
        sig_pts = np.zeros((N, 3, N_sig), dtype=float)
        sig_pts += sig[-3:, :]
        sig_pts += np.repeat(pts_raw.reshape(N, 3, 1), N_sig, axis=-1)

        sig_sensor = np.zeros((N, 3, N_sig), dtype=float)
        if sensor_type == SensorType.STEREO:
            # Transform from image to cartesian coordinates
            uvd = sig_pts
            uvd_homo = np.insert(uvd, 3, 1.0, axis=1)
            xyz_homo = np.einsum("ijk,mj->imk", uvd_homo, self.Q)
            sig_sensor = xyz_homo[:, :3, :]
            sig_sensor /= xyz_homo[:, 3, :].reshape(N, 1, N_sig)
        elif sensor_type == SensorType.LIDAR:
            # Transform from spherical to cartesian coordinates
            rho = sig_pts[:, 0, :]
            azi = sig_pts[:, 1, :]
            elev = sig_pts[:, 2, :]

            sig_sensor[:, 0, :] = rho * np.cos(elev) * np.cos(azi)
            sig_sensor[:, 1, :] = rho * np.cos(elev) * np.sin(azi)
            sig_sensor[:, 2, :] = rho * np.sin(elev)
        else:
            raise ValueError("Invalid sensor type specified")

        # Transform to relative coordinates
        sig_sensor -= l0.T  # Shift origin to l0

        h = np.einsum("ijk,jk->ik", sig_sensor, n)
        h_diff = np.einsum("ij,kj->ikj", h, n)

        sig_proj = sig_sensor - h_diff
        alpha = sig_proj[:, 1, :] * db[:, 0] - sig_proj[:, 0, :] * db[:, 1]
        beta = sig_proj[:, 0, :] * da[:, 1] - sig_proj[:, 1, :] * da[:, 0]

        k = da[:, 1] * db[:, 0] - da[:, 0] * db[:, 1]
        alpha /= k
        beta /= k

        sig_rel = np.empty_like(sig_sensor)
        sig_rel[:, 0, :] = alpha
        sig_rel[:, 1, :] = beta
        sig_rel[:, 2, :] = h

        # Calculate moments
        w_m = np.copy(self.unscented.w_m)
        w_c = np.copy(self.unscented.w_c)

        pts_rel = np.sum(w_m * sig_rel, axis=-1).reshape(N, 3, 1)
        diff = sig_rel - pts_rel
        covs_rel = np.einsum("ijk,k,imk->ijm", diff, w_c, diff)

        # Double check point is in the correct region
        a0 = pts_rel[:, 0, 0] > 0
        a0 &= pts_rel[:, 1, 0] > 0
        a0 &= (pts_rel[:, 0, 0] + pts_rel[:, 1, 0]) < 1

        pts_rel = pts_rel[a0, :, :]
        covs_rel = covs_rel[a0, :, :]
        N = np.sum(a0)

        if N > 0:
            self.needs_update = True
            self.insert_measurements(pts_rel, covs_rel)

    def process_utoronto(self, pts_world, covs, lms, test=False):
        """Decouple measurements and insert into tree for the UTorono dataset.

        Decouple measurements when the landmarks are known perfectly, in
        otherwords the map should be in the world IRF.

        Parameters:

        pts_world : ndarray, (N, 3, 1).
            Measurements in the world IRF (x, y, z).
        covs : ndarray, (N, 3, 3).
            Associated measurement covariances.
        lms : ndarray, (3, 3, 1).
           LTR landmarks.
        """
        if self.dim == 2:
            raise NotImplementedError()

        N = pts_world.shape[0]
        logger.info(f"Processing {N} measurements")

        l0 = lms[0, :]
        la = lms[1, :]
        lb = lms[2, :]

        da = (la - l0).flatten()
        db = (lb - l0).flatten()

        # n = [0, 0, 1]^T in this special case
        n = np.cross(da, db).reshape(3, 1)
        n /= np.linalg.norm(n)

        # Transform to relative coordinates
        # Shift origin to l0
        pts_shift = pts_world - l0  # (N, 3, 1)

        # The  transform matrix from relative to world
        T_rel_world = np.hstack((da.reshape(3, 1), db.reshape(3, 1), n))
        T_world_rel = np.linalg.inv(T_rel_world)
        pts_rel = np.einsum("ij,ajk->aik", T_world_rel, pts_shift)

        # Double check point is in the correct region
        valid = pts_rel[:, 0, 0] >= 0
        valid &= pts_rel[:, 1, 0] >= 0
        valid &= (pts_rel[:, 0, 0] + pts_rel[:, 1, 0]) <= 1

        pts_rel = pts_rel[valid, :, :]
        covs = covs[valid, :, :]
        N = np.sum(valid)

        # Transform covariances using Jaccobian (exact in this case)
        # Jaccobian of transformation
        J = T_world_rel

        # Calculate J*covs*J^T
        # J*covs
        covs_rel = np.einsum("ij,ajk->aik", J, covs)
        # (J*covs)*J^T
        covs_rel = np.einsum("aij,jk->aik", covs_rel, J.T)

        if test:
            return pts_rel, covs_rel

        if N > 0:
            self.needs_update = True
            self.insert_measurements(pts_rel, covs_rel)

    def reset_message_counter(self):
        self.message_counter = MessageCounter()

    def update(self) -> MessageCounter:
        """Update the map belief for the given measurements."""
        logger.debug("Starting map update")

        update_queue_current = MessageQueue()
        update_queue_next = MessageQueue()

        for surfel in self:
            # Setup the initial queue based on the number of measurements in the surfel
            if surfel.n_window > 0:
                msg_priority = MessagePriority(surfel.uid, -surfel.n_window)
                update_queue_current.put(msg_priority)

        start_message_counter = copy.deepcopy(self.message_counter)
        prev_message_counter = copy.deepcopy(self.message_counter)

        iterations = 0
        while not update_queue_current.empty() and iterations < self.max_iterations:
            msg_priority = update_queue_current.get()
            surfel_id = msg_priority.msg_id
            surfel = self[surfel_id]

            global_convergence = False

            # Collect incoming messages
            inmsg_h_h, inmsg_h_K = self.stmm_bp.get_incoming_message(surfel.rv_ids)

            global_distance = surfel.update_incoming_messages(inmsg_h_h, inmsg_h_K)
            # if global_distance < 1e-8:  # TODO this should be global convergence, or should it...
            if global_distance < self.tolerance:
                global_convergence = True

            # Perform local projection
            if not surfel.converged:  # XXX Maybe add "or not global_convergence"
                n_msg_local, local_distance = surfel.local_projection()
                self.message_counter.local_msgs += n_msg_local

                if local_distance < self.tolerance:
                    surfel.converged = True
                    surfel.window_measurements()

            # Send outgoing messages
            n_msg_global = self.stmm_bp.update_graph(
                surfel.pot_h_m.reshape(self.dim), surfel.pot_h_C, surfel.rv_ids
            )
            self.message_counter.global_msgs += n_msg_global

            # NOTE I think this is because of weird EMDW magic...
            m, C, h, K = self.stmm_bp.get_cluster_potential(surfel.rv_ids, surfel.graph_id)
            if not np.allclose(h, surfel.pot_h_h.flat) or not np.allclose(K, surfel.pot_h_K):
                logger.debug("Potentials are wack...")

            if not (surfel.converged and global_convergence):
                distance = local_distance + global_distance
                # Use the negative distance as the priority queue will pop the smallest
                msg_priority = MessagePriority(surfel.uid, -distance)
                update_queue_next.put(msg_priority)

                for link in self.graph[surfel.graph_id]:
                    link_uid = self.id_mapping[link]
                    # NOTE for now just use the distance of the current surfel as a proxy for the
                    # messages from the neighbours.
                    msg_priority = MessagePriority(link_uid, -distance)
                    update_queue_next.put(msg_priority)

            if update_queue_current.empty():
                logger.debug(
                    f"Message counter delta: {self.message_counter - prev_message_counter}"
                )
                if update_queue_next.empty():
                    logger.debug(f"Successfully updated submap in {iterations + 1} iterations")
                    break
                update_queue_current = update_queue_next
                update_queue_next = MessageQueue()
                prev_message_counter = copy.deepcopy(self.message_counter)

                iterations += 1
        else:
            if iterations != 0:
                logger.debug(
                    f"Update halted prematurely as iterations {iterations} has reached max {self.max_iterations}"
                )

        # XXX Last little calibration
        # n_msg_global = self.stmm_bp.calibrate()
        # self.message_counter.global_msgs += n_msg_global
        # for surfel in self:
        #     inmsg_h_h, inmsg_h_K = self.stmm_bp.get_incoming_message(surfel.rv_ids)
        #     surfel.update_incoming_messages(inmsg_h_h, inmsg_h_K)
        # self.message_counter.global_msgs += n_msg_global
        # logger.debug(
        #     f"Message counter delta: {self.message_counter - prev_message_counter}"
        # )

        self.needs_update = False
        logger.debug("Completed map update")

        return self.message_counter - start_message_counter

    def get_surfel(self, path):
        # Try cast to tuple first
        if type(path) != tuple:
            tuple_path = tuple(path)
        else:
            tuple_path = path
        if tuple_path in self.surfel_dict.keys():
            return self[tuple_path]
        else:
            return None

    def fuse_submaps(self, surfel_dict):
        """Fuse another submap's surfels into the current submap."""
        assert len(self.surfel_dict) == len(surfel_dict), "Submaps must be of equal size"

        for surfel_id, surfel in surfel_dict.items():
            self[surfel_id].combine(surfel)

    def plot_map_indices(self, ax: Optional[plt.Axes] = None):
        """Plot the ids of the heights in the map."""

        if ax is None:
            plt.figure(constrained_layout=True)
            ax = plt.gca()

        ids = {}
        r = self.base / 4

        for path in self.surfel_ids:
            surfel = self.get_surfel(path)
            for i, id in enumerate(surfel.rv_ids):
                if id not in ids:
                    corner = np.zeros(2)
                    corner[: (self.dim - 1)] = surfel.corners[:, i]
                    ids[id] = corner
                    circle = Circle(corner, r, facecolor="w", edgecolor="k", linewidth=1)
                    ax.add_patch(circle)
                    ax.annotate(str(id), xy=corner, ha="center", va="center")
        # ax.set_aspect("equal")
        # ax.autoscale_view()
        return ids

    def loglike(self, test_pts: np.ndarray) -> float:
        """Calculate the log-likelihood for the model given some test points."""
        test_pts = np.copy(test_pts)
        if self.dim == 3:
            sum_ab = np.sum(test_pts[:, :2, 0], axis=1)
            pts_aug = np.insert(test_pts[:, :2, 0], 2, sum_ab, axis=1)  # (N, self.dim)
            paths = np.floor(pts_aug / self.base).astype(int)
        else:
            pts_aug = test_pts[:, 0, 0]
            paths = np.floor(pts_aug / self.base).astype(int).reshape(-1, 1)

        unique_paths = np.unique(paths, axis=0)

        log_like = 0.0
        for path in unique_paths:
            surfel_id = tuple(path)
            if surfel_id in self.surfel_ids:
                surfel = self[surfel_id]
            else:
                continue

            if self.dim == 3:
                b = (paths[:, 0] == path[0]) & (paths[:, 1] == path[1]) & (paths[:, 2] == path[2])
            else:
                b = paths[:, 0] == path[0]

            n_additional = np.sum(b)
            if n_additional > 0:
                trans_test_pts = surfel.transform_means(test_pts[b])

                m = surfel.bel_h_m
                C = surfel.bel_h_C
                for pt in trans_test_pts:
                    alpha_t = pt[0, 0]
                    h_t = pt[-1, 0]
                    if self.dim == 3:
                        beta_t = pt[1, 0]
                        trans = np.array([[1 - alpha_t - beta_t, alpha_t, beta_t]])
                    else:
                        trans = np.array([[1 - alpha_t, alpha_t]])
                    m_t = trans.dot(m)
                    v = surfel.bel_v_b / (surfel.bel_v_a + 1)
                    C_t = trans.dot(C).dot(trans.T) + v
                    log_like += multivariate_normal.logpdf(h_t, m_t, C_t)

        return log_like

    def mean_squared_error(self, test_pts: np.ndarray) -> float:
        """Calculate the mean squared error for the model given some test points."""
        test_pts = np.copy(test_pts)
        if self.dim == 3:
            sum_ab = np.sum(test_pts[:, :2, 0], axis=1)
            pts_aug = np.insert(test_pts[:, :2, 0], 2, sum_ab, axis=1)  # (N, self.dim)
            paths = np.floor(pts_aug / self.base).astype(int)
        else:
            pts_aug = test_pts[:, 0, 0]
            paths = np.floor(pts_aug / self.base).astype(int).reshape(-1, 1)

        N_test = test_pts.shape[0]
        squared_error = 0.0
        for surfel_id in self.surfel_ids:
            if surfel_id in self.surfel_ids:
                surfel = self[surfel_id]
            else:
                continue

            if self.dim == 3:
                b = (
                    (paths[:, 0] == surfel_id[0])
                    & (paths[:, 1] == surfel_id[1])
                    & (paths[:, 2] == surfel_id[2])
                )
            else:
                b = paths[:, 0] == surfel_id[0]

            n_additional = np.sum(b)
            if n_additional > 0:
                trans_test_pts = surfel.transform_means(test_pts[b])
                m = surfel.bel_h_m.reshape(-1)
                if self.dim == 3:
                    est_line = (
                        (m[1] - m[0]) * trans_test_pts[:, 0, 0]
                        + (m[2] - m[0]) * trans_test_pts[:, 1, 0]
                        + m[0]
                    )
                else:
                    est_line = (m[1] - m[0]) * trans_test_pts[:, 0, 0] + m[0]
                diff = (est_line - trans_test_pts[:, -1, 0]) ** 2
                squared_error += np.sum(diff)
            else:
                logger.warn(f"Could not find test points in surfel {surfel.uid}")

        mse = squared_error / N_test
        return mse
