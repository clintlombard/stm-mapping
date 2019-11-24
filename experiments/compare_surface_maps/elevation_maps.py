# -*- coding: utf-8 -*-
import logging

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import sympy as sym

from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

import stmmap.utils.triangle_transforms as tt

from stmmap.unscented import Unscented

logger = logging.getLogger(__name__)


class ElevationMap2D:
    def __init__(self, depth=4):
        self.sub_div = 2 ** depth  # number of surfels
        self.bins = np.linspace(0, 1, self.sub_div + 1)
        self.heights_m = np.zeros(self.sub_div)
        self.heights_v = 100 * np.ones(self.sub_div)
        self.heights_h = np.zeros(self.sub_div)
        self.heights_K = np.ones(self.sub_div) / 100

    def __iter__(self):
        for i in range(self.sub_div):
            yield self.heights_m[i], self.heights_v[i]

    def update(self, m_z, C_z):
        m_z = np.copy(m_z)
        C_z = np.copy(C_z)

        # Transform measurements
        bin_alloc = (np.digitize(m_z[:, 0, :], self.bins) - 1).reshape(-1)
        offsets = self.bins[bin_alloc]
        scaling = 1 / self.sub_div
        m_z[:, 0, :] -= offsets[None].T
        m_z[:, 0, :] /= scaling
        C_z[:, 0, 0] /= scaling ** 2
        C_z[:, 1, 0] /= scaling
        C_z[:, 0, 1] /= scaling

        for i in range(self.sub_div):
            m_z_sel = m_z[bin_alloc == i, :, :]
            # print(m_z_sel)
            # print(m_z_sel.shape)
            C_z_sel = C_z[bin_alloc == i, :, :]
            if m_z_sel.shape[0] > 0:
                K_z = 1 / C_z_sel[:, -1, -1]
                h_z = K_z * m_z_sel[:, 1, 0]
                self.heights_K[i] += np.sum(K_z)
                self.heights_h[i] += np.sum(h_z)
                self.heights_v[i] = 1 / self.heights_K[i]
                self.heights_m[i] = self.heights_v[i] * self.heights_h[i]

    def loglike(self, test_pts: np.ndarray) -> float:
        """Calculate the log-likelihood for the model given some test points."""
        test_pts = np.copy(test_pts)
        bin_alloc = (np.digitize(test_pts[:, 0, :], self.bins) - 1).reshape(-1)
        log_like = 0.0
        for i in range(self.sub_div):
            sel_pts = test_pts[bin_alloc == i, :, :]
            m = self.heights_m[i].reshape(-1)
            C = self.heights_v[i]
            h_t = sel_pts[:, 1, 0]
            try:
                log_like += np.sum(multivariate_normal.logpdf(h_t, m, C))
            except Exception:
                print(h_t)
                print(C)
                print(m)
                exit()
        return log_like

    def mean_squared_error(self, test_pts: np.ndarray) -> float:
        test_pts = np.copy(test_pts)
        bin_alloc = (np.digitize(test_pts[:, 0, :], self.bins) - 1).reshape(-1)
        N_test = test_pts.shape[0]
        squared_error = 0.0
        for i in range(self.sub_div):
            sel_pts = test_pts[bin_alloc == i, :, :]
            m = self.heights_m[i].reshape(-1)
            C = self.heights_v[i]
            h_test = sel_pts[:, 1, 0]
            diff = (m - h_test) ** 2
            squared_error += np.sum(diff)

        mse = squared_error / N_test
        return mse


class ElevationSurfel(object):
    """Elevation Surfel object.
    """

    def __init__(self, corners: np.ndarray, uid: Tuple[int, ...], dim: int = 3):
        self.uid = uid  # surfel unique id
        self.corners = np.copy(corners)
        self.dim = dim

        if self.dim not in [2, 3]:
            raise ValueError("Can only handle 2 or 3 dimensions")

        if self.dim != (corners.shape[0] + 1):  # +1 because shape is (dim-1, N)
            raise ValueError(
                f"Corner dimensions, {corners.shape}, mismatch the surfel dimensions, {self.dim}."
            )

        self._init_transfrom()

        # Init belief distributions
        self.bel_h_m = 0
        self.bel_h_C = 1e9
        self.bel_h_h = 0
        self.bel_h_K = 1e-9

    def _init_transfrom(self):
        # Calculate the transformation for measurement mean and covariance from the unit surfel
        # coordinates to the sub-surfel coordinates. NOTE: This is a simple linear transformation
        # therefore the Gaussian assumption is still valid.
        alpha_axis = self.corners[:, 1] - self.corners[:, 0]
        alpha_axis /= np.linalg.norm(alpha_axis) ** 2
        self.transform = np.eye(self.dim)
        self.transform[0, : (self.dim - 1)] = alpha_axis
        if self.dim == 3:
            beta_axis = self.corners[:, 2] - self.corners[:, 0]
            beta_axis /= np.linalg.norm(beta_axis) ** 2
            self.transform[1, : (self.dim - 1)] = beta_axis
        self.offset = np.vstack((self.corners[:, 0][None].T, 0))

    def update(self, z_new_means: np.ndarray, z_new_covs: np.ndarray):
        z_means = self.transform_means(z_new_means)[:, -1, 0]
        z_covs = self.transform_covs(z_new_covs)[:, -1, -1]

        z_K = 1 / z_covs
        z_h = z_K * z_means

        self.bel_h_h += np.sum(z_h)
        self.bel_h_K += np.sum(z_K)
        self.bel_h_C = 1 / self.bel_h_K
        self.bel_h_m = self.bel_h_C * self.bel_h_h

    def transform_means(self, z_means: np.ndarray) -> np.ndarray:
        J = self.transform
        z_trans_means = z_means - self.offset
        z_trans_means = np.einsum("ij,ajk->aik", J, z_trans_means)

        return z_trans_means

    def transform_covs(self, z_covs: np.ndarray) -> np.ndarray:
        J = self.transform
        z_trans_covs = np.einsum("ij,ajk->aik", J, z_covs)
        z_trans_covs = np.einsum("aij,kj->aik", z_trans_covs, J)

        return z_trans_covs


class ElevationMap3D:
    def __init__(self, max_depth: int = 4, dim: int = 3):
        logger.debug("Initialising elevation map")
        self.dim = dim
        if self.dim not in [2, 3]:
            raise ValueError("Can only handle 2 or 3 dimensions")

        self.max_depth = max_depth

        self.surfel_dict: Dict[Tuple[int, ...], ElevationSurfel] = dict()
        self.surfel_ids = self.surfel_dict.keys()

        self.base = 1.0 / 2 ** max_depth
        if self.dim == 3:
            self.n_surfels = 4 ** max_depth  # number of surfels
        else:
            self.n_surfels = 2 ** max_depth  # number of surfels

        if self.dim == 3:
            self._generate_3d_grid()
        else:
            self._generate_2d_grid()

        logger.debug("Elevation map successfully initialised")

    def __iter__(self):
        for surfel_id in self.surfel_ids:
            yield self[surfel_id]

    def __len__(self):
        return len(self.surfel_dict)

    def __getitem__(self, key: Tuple[int, ...]) -> ElevationSurfel:
        return self.surfel_dict[key]

    def __setitem__(self, key: Tuple[int, ...], value: ElevationSurfel):
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

            # Calculate the incenter of the triangle and use it as a point to
            # determine the surfel_id.
            tmp = np.array([[2 ** 0.5, 1, 1]])
            incenter = np.sum(tmp * corners_, axis=1) / np.sum(tmp)
            incenter = np.hstack((incenter, np.sum(incenter)))
            surfel_id = tuple(np.floor(incenter / self.base).astype(int))

            self[surfel_id] = ElevationSurfel(corners=corners_, uid=surfel_id, dim=self.dim)

    def _generate_2d_grid(self, corners=None, depth=1):
        raise NotImplementedError("No 2D here")
        # self.h_ids = [i for i in range(self.n_surfels + 1)]
        # corners = np.array(self.h_ids) * self.base
        # assert corners[0] == 0.0
        # assert corners[-1] == 1.0

        # for s in range(self.n_surfels):
        #     surfel_id = (s,)
        #     ids = self.h_ids[s : (s + 2)]
        #     corners_ = corners[s : (s + 2)].reshape(1, 2)
        #     graph_id = self._counter
        #     self[surfel_id] = Surfel(
        #         corners=corners_,
        #         uid=surfel_id,
        #         rv_ids=tuple(ids),
        #         graph_id=graph_id,
        #         dim=self.dim,
        #         projection_method=self.projection_method,
        #         window_size=self.window_size,
        #         prior_corr=self.prior_corr,
        #     )

    def update(self, z_means, z_covs):
        """
        Insert already decoupled measurements into the tree.

        Parameters
          * z_means - (N, self.dim, 1)
          * z_covs - (N, self.dim, self.dim)
        """
        N = z_means.shape[0]
        if N == 0:
            return None

        logger.info(f"Updating elevation map with {N} measurements")
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

                surfel.update(z_new_means, z_new_covs)

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
                log_like += np.sum(multivariate_normal.logpdf(trans_test_pts[:, -1, 0], m, C))

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
                m = surfel.bel_h_m
                diff = (m - trans_test_pts[:, -1, 0]) ** 2
                squared_error += np.sum(diff)
            else:
                print(f"Could not find test points in surfel {surfel.uid}")

        mse = squared_error / N_test
        return mse

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
            self.update(pts_rel, covs_rel)
