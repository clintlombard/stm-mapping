from __future__ import annotations

import logging
import random

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from stmmap.projections import ProjectionMethod, VMPFactorised, VMPStructured
from stmmap.utils.distances import DistanceMetric, distance_gauss

logger = logging.getLogger(__name__)


class Surfel(object):
    """Surfel object.

    Attributes:
        corners (ndarray): relative coordinates of the surfel corners
            with respect to the root surfel.
         [[alpha_0 alpha_1 alpha_2],
          [beta_0 beta_1 beta_2]]
    """

    def __init__(
        self,
        corners: np.ndarray,
        uid: Tuple[int, ...],
        rv_ids: Tuple[int, ...],
        graph_id: int,
        projection_method: Callable[..., ProjectionMethod] = VMPStructured,
        dim: int = 3,
        init_capacity: int = 5,
        capacity_growth_factor: float = 1.5,
        window_size: int = 0,
        prior_corr: float = 0.5,
    ):
        self.uid = uid  # surfel unique id
        self.rv_ids = rv_ids  # height rv ids
        self.graph_id = graph_id
        self.corners = np.copy(corners)
        self.dim = dim
        self.window_size = window_size  # number of measurements to keep in the window
        self.capacity_growth_factor = capacity_growth_factor
        self.capacity = init_capacity
        self.prior_corr = prior_corr

        if self.dim not in [2, 3]:
            raise ValueError("Can only handle 2 or 3 dimensions")

        if self.dim != (corners.shape[0] + 1):  # +1 because shape is (dim-1, N)
            raise ValueError(
                f"Corner dimensions, {corners.shape}, mismatch the surfel dimensions, {self.dim}."
            )

        # Some counters
        self.n_frozen = 0  # number of 'frozen' measurements (outside the window)
        self.n_window = 0  # number of measurements in the window

        self._init_transfrom()
        self._init_prior_cluster()

        # Init measurement potentials
        self.z_m = np.empty((self.capacity, self.dim, 1), dtype=float)
        # self.z_C = np.empty((self.capacity, self.dim, self.dim), dtype=float)
        self.z_h = np.empty((self.capacity, self.dim, 1), dtype=float)
        self.z_K = np.empty((self.capacity, self.dim, self.dim), dtype=float)

        # Init messages from each likelihood cluster
        self.like_outmsg_h_h = np.empty((self.capacity, self.dim, 1), dtype=float)
        self.like_outmsg_h_K = np.empty((self.capacity, self.dim, self.dim), dtype=float)
        self.like_outmsg_v_a = np.empty((self.capacity, 1), dtype=float)
        self.like_outmsg_v_b = np.empty((self.capacity, 1), dtype=float)

        # Init belief distributions to priors
        self.bel_h_m = np.copy(self.prior_h_m)
        self.bel_h_C = np.copy(self.prior_h_C)
        self.bel_h_h = np.copy(self.prior_h_h)
        self.bel_h_K = np.copy(self.prior_h_K)
        self.bel_v_a = np.copy(self.prior_v_a)
        self.bel_v_b = np.copy(self.prior_v_b)

        # Init incoming message from the rest of the map
        self.inmsg_h_h = np.zeros((self.dim, 1), dtype=float)
        self.inmsg_h_K = np.zeros((self.dim, self.dim), dtype=float)

        # Init surfel potentials (this is only needed for heights)
        self.pot_h_m = np.copy(self.prior_h_m)
        self.pot_h_C = np.copy(self.prior_h_C)
        self.pot_h_h = np.copy(self.prior_h_h)
        self.pot_h_K = np.copy(self.prior_h_K)

        self.converged = False

        # NOTE This must be last
        self.projection_method = projection_method(surf=self)

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

    def _init_prior_cluster(self):
        # Mean plane heights
        # Correlated prior
        off_diag = np.where(~np.eye(self.dim, dtype=bool))
        corr = np.eye(self.dim, dtype=float)
        corr[off_diag] = self.prior_corr
        std = 100
        C_prior = np.full((self.dim, self.dim), std, dtype=float)
        C_prior *= C_prior.T
        C_prior *= corr

        # Moments form
        self.prior_h_C = 1 * C_prior
        self.prior_h_m = 0.01 * np.ones((self.dim, 1), dtype=float)  # mean vector
        # Canonical form
        self.prior_h_K = np.linalg.inv(C_prior)  # info. mat
        self.prior_h_h = self.prior_h_K.dot(self.prior_h_m)

        # Planar deviation
        self.prior_v_a = np.array([2.0])
        self.prior_v_b = np.array([1e-3])
        # self.prior_v_a = 2
        # self.prior_v_b = 2

    def update_incoming_messages(
        self, inmsg_new_h_h: np.ndarray, inmsg_new_h_K: np.ndarray
    ) -> float:
        inmsg_h_h_old = np.copy(self.inmsg_h_h)
        inmsg_h_K_old = np.copy(self.inmsg_h_K)

        self.inmsg_h_h = np.copy(inmsg_new_h_h).reshape(self.dim, 1)
        self.inmsg_h_K = np.copy(inmsg_new_h_K).reshape(self.dim, self.dim)

        self.bel_h_h = self.pot_h_h + self.inmsg_h_h
        self.bel_h_K = self.pot_h_K + self.inmsg_h_K
        self.bel_h_C = np.linalg.inv(self.bel_h_K)
        self.bel_h_m = self.bel_h_C.dot(self.bel_h_h)

        # NOTE calculating the distance between messages is problematic because the scope is
        # extended to be the same as the full CP scope, which means dimensions of the message could
        # be vacuous.
        # XXX A fix is to assume the vacuous dimensions can be ignored as this most likely means
        # that there is no connection with neighbours.
        diag = np.diag(inmsg_h_K_old)
        valid = diag > 1e-8

        old_vacuous = False
        try:
            old_inmsg_h_C = np.linalg.inv(inmsg_h_K_old[valid, :][:, valid])
            old_inmsg_h_m = old_inmsg_h_C.dot(inmsg_h_h_old[valid, :])
        except np.linalg.LinAlgError:
            old_vacuous = True

        new_vacuous = False
        try:
            new_inmsg_h_C = np.linalg.inv(self.inmsg_h_K[valid, :][:, valid])
            new_inmsg_h_m = new_inmsg_h_C.dot(self.inmsg_h_h[valid, :])
        except np.linalg.LinAlgError:
            new_vacuous = True

        if new_vacuous and old_vacuous:
            return 0.0
        elif new_vacuous or old_vacuous:
            return 1e5

        distance = distance_gauss(old_inmsg_h_m, old_inmsg_h_C, new_inmsg_h_m, new_inmsg_h_C)

        return distance

    def local_projection(self) -> Tuple[int, float]:
        """Perform a local projection iteration.

        Note: the incoming messages might need to be updated before calling this.

        """
        distance = self.projection_method.iteration()

        return self.n_window, distance

    def insert_measurements(self, z_new_means: np.ndarray, z_new_covs: np.ndarray):
        n_additional = z_new_means.shape[0]
        n_required = n_additional + self.n_window
        # logger.debug(
        #     f"Surfel {self.uid}: Inserting {n_additional} measurements. "
        #     f"Growing window size to {n_required}"
        # )

        # Allocate more storage if required
        if self.capacity < n_required:
            self.capacity = int(n_required * self.capacity_growth_factor)

            z_m_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            # z_C_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)
            z_h_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            z_K_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)

            like_outmsg_h_h_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            like_outmsg_h_K_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)
            like_outmsg_v_a_tmp = np.empty((self.capacity, 1), dtype=float)
            like_outmsg_v_b_tmp = np.empty((self.capacity, 1), dtype=float)

            z_m_tmp[: self.n_window] = self.z_m[: self.n_window]
            # z_C_tmp[: self.n_window] = self.z_C[: self.n_window]
            z_h_tmp[: self.n_window] = self.z_h[: self.n_window]
            z_K_tmp[: self.n_window] = self.z_K[: self.n_window]

            like_outmsg_h_h_tmp[: self.n_window] = self.like_outmsg_h_h[: self.n_window]
            like_outmsg_h_K_tmp[: self.n_window] = self.like_outmsg_h_K[: self.n_window]
            like_outmsg_v_a_tmp[: self.n_window] = self.like_outmsg_v_a[: self.n_window]
            like_outmsg_v_b_tmp[: self.n_window] = self.like_outmsg_v_b[: self.n_window]

            self.z_m = z_m_tmp
            # self.z_C = z_C_tmp
            self.z_h = z_h_tmp
            self.z_K = z_K_tmp

            self.like_outmsg_h_h = like_outmsg_h_h_tmp
            self.like_outmsg_h_K = like_outmsg_h_K_tmp
            self.like_outmsg_v_a = like_outmsg_v_a_tmp
            self.like_outmsg_v_b = like_outmsg_v_b_tmp

        z_new_means = self.transform_means(z_new_means)
        z_new_covs = self.transform_covs(z_new_covs)

        self.z_m[self.n_window : n_required] = z_new_means
        # self.z_C[self.n_window : n_required] = z_new_covs
        self.z_K[self.n_window : n_required] = np.linalg.inv(z_new_covs)
        self.z_h[self.n_window : n_required] = np.einsum(
            "aij,ajk->aik", self.z_K[self.n_window : n_required], z_new_means
        )

        gammas = z_new_means[:, -1, 0].reshape(n_additional, 1, 1)
        self.like_outmsg_h_h[self.n_window : n_required] = np.tile(
            0.001 * gammas, (1, self.dim, 1)
        )
        self.like_outmsg_h_K[self.n_window : n_required] = np.tile(
            0.001 * np.eye(self.dim, dtype=float), (n_additional, 1, 1)
        )
        self.like_outmsg_v_a[self.n_window : n_required] = 0.5
        self.like_outmsg_v_b[self.n_window : n_required] = np.var(gammas) * 0.5

        self.pot_h_h += n_additional * self.like_outmsg_h_h[n_required - 1]
        self.pot_h_K += n_additional * self.like_outmsg_h_K[n_required - 1]
        self.pot_h_C = np.linalg.inv(self.pot_h_K)
        self.pot_h_m = self.pot_h_C.dot(self.pot_h_h)

        self.bel_h_h = self.pot_h_h + self.inmsg_h_h
        self.bel_h_K = self.pot_h_K + self.inmsg_h_K
        self.bel_h_C = np.linalg.inv(self.bel_h_K)
        self.bel_h_m = self.bel_h_C.dot(self.bel_h_h)

        self.bel_v_a += n_additional * self.like_outmsg_v_a[n_required - 1]
        self.bel_v_b += n_additional * self.like_outmsg_v_b[n_required - 1]

        self.n_window = n_required
        self.converged = False

    def window_measurements(self) -> None:
        selected_indices: Union[Tuple[int], List[int]]

        # logger.debug(
        #     "Applying windowing: "
        #     f"window_size = {self.window_size}, "
        #     f"n_frozen = {self.n_frozen} "
        #     f"n_window = {self.n_window} "
        # )

        # NOTE This could be performed on the fly when updating each like_outmsg. Windowing would
        # then simply be to discard the like_outmsg's in this function.
        if self.window_size == 0:
            self.prior_h_h += np.sum(self.like_outmsg_h_h[: self.n_window], axis=0)
            self.prior_h_K += np.sum(self.like_outmsg_h_K[: self.n_window], axis=0)
            self.prior_v_a += np.sum(self.like_outmsg_v_a[: self.n_window], axis=0)
            self.prior_v_b += np.sum(self.like_outmsg_v_b[: self.n_window], axis=0)

            # Init measurement potentials
            self.z_m = np.empty((0, self.dim, 1), dtype=float)
            # self.z_C = np.empty((0, self.dim, self.dim), dtype=float)
            self.z_h = np.empty((0, self.dim, 1), dtype=float)
            self.z_K = np.empty((0, self.dim, self.dim), dtype=float)

            # Init messages from each likelihood cluster
            self.like_outmsg_h_h = np.empty((0, self.dim, 1), dtype=float)
            self.like_outmsg_h_K = np.empty((0, self.dim, self.dim), dtype=float)
            self.like_outmsg_v_a = np.empty((0, 1), dtype=float)
            self.like_outmsg_v_b = np.empty((0, 1), dtype=float)
        elif self.n_window > self.window_size > 0:
            # Remove some measurements
            selected_indices = random.sample(range(self.n_window), self.window_size)

            delete_like_outmsg_h_h = np.delete(self.like_outmsg_h_h, selected_indices, axis=0)
            delete_like_outmsg_h_K = np.delete(self.like_outmsg_h_K, selected_indices, axis=0)
            delete_like_outmsg_v_a = np.delete(self.like_outmsg_v_a, selected_indices, axis=0)
            delete_like_outmsg_v_b = np.delete(self.like_outmsg_v_b, selected_indices, axis=0)

            self.like_outmsg_h_h = self.like_outmsg_h_h[selected_indices]
            self.like_outmsg_h_K = self.like_outmsg_h_K[selected_indices]
            self.like_outmsg_v_a = self.like_outmsg_v_a[selected_indices]
            self.like_outmsg_v_b = self.like_outmsg_v_b[selected_indices]

            self.prior_h_h += np.sum(delete_like_outmsg_h_h, axis=0)
            self.prior_h_K += np.sum(delete_like_outmsg_h_K, axis=0)
            self.prior_v_a += np.sum(delete_like_outmsg_v_a, axis=0)
            self.prior_v_b += np.sum(delete_like_outmsg_v_b, axis=0)

            self.z_m = self.z_m[selected_indices]
            # self.z_C = self.z_C[selected_indices]
            self.z_h = self.z_h[selected_indices]
            self.z_K = self.z_K[selected_indices]
        else:
            return None

        self.capacity = self.window_size
        self.n_frozen += self.n_window - self.window_size
        self.n_window = self.window_size

    def combine(self, other: Surfel):
        # raise NotImplementedError("Some things seem dodgy in here...")
        n_additional = other.n_window
        if n_additional == 0:
            return None

        n_required = n_additional + self.n_window

        # Allocate more storage if required
        if self.capacity < n_required:
            self.capacity = n_required

            z_m_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            # z_C_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)
            z_h_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            z_K_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)

            like_outmsg_h_h_tmp = np.empty((self.capacity, self.dim, 1), dtype=float)
            like_outmsg_h_K_tmp = np.empty((self.capacity, self.dim, self.dim), dtype=float)
            like_outmsg_v_a_tmp = np.empty((self.capacity, 1), dtype=float)
            like_outmsg_v_b_tmp = np.empty((self.capacity, 1), dtype=float)

            z_m_tmp[: self.n_window] = self.z_m[: self.n_window]
            # z_C_tmp[: self.n_window] = self.z_C[: self.n_window]
            z_h_tmp[: self.n_window] = self.z_h[: self.n_window]
            z_K_tmp[: self.n_window] = self.z_K[: self.n_window]

            like_outmsg_h_h_tmp[: self.n_window] = self.like_outmsg_h_h[: self.n_window]
            like_outmsg_h_K_tmp[: self.n_window] = self.like_outmsg_h_K[: self.n_window]
            like_outmsg_v_a_tmp[: self.n_window] = self.like_outmsg_v_a[: self.n_window]
            like_outmsg_v_b_tmp[: self.n_window] = self.like_outmsg_v_b[: self.n_window]

            self.z_m = z_m_tmp
            # self.z_C = z_C_tmp
            self.z_h = z_h_tmp
            self.z_K = z_K_tmp

            self.like_outmsg_h_h = like_outmsg_h_h_tmp
            self.like_outmsg_h_K = like_outmsg_h_K_tmp
            self.like_outmsg_v_a = like_outmsg_v_a_tmp
            self.like_outmsg_v_b = like_outmsg_v_b_tmp

        self.z_m[self.n_window : n_required] = other.z_m[: other.n_window]
        # self.z_C[self.n_window : n_required] = other.z_C[: other.n_window]
        self.z_h[self.n_window : n_required] = other.z_h[: other.n_window]
        self.z_K[self.n_window : n_required] = other.z_K[: other.n_window]

        self.like_outmsg_h_h[self.n_window : n_required] = other.like_outmsg_h_h[: other.n_window]
        self.like_outmsg_h_K[self.n_window : n_required] = other.like_outmsg_h_K[: other.n_window]
        self.like_outmsg_v_a[self.n_window : n_required] = other.like_outmsg_v_a[: other.n_window]
        self.like_outmsg_v_b[self.n_window : n_required] = other.like_outmsg_v_b[: other.n_window]

        # This will exclude any frozen potentials...
        self.bel_h_h += np.sum(other.like_outmsg_h_h[: other.n_window], axis=0)
        self.bel_h_K += np.sum(other.like_outmsg_h_K[: other.n_window], axis=0)
        self.bel_h_C = np.linalg.inv(self.bel_h_K)
        self.bel_h_m = self.bel_h_C.dot(self.bel_h_h)
        self.bel_v_a += np.sum(other.like_outmsg_v_a[: other.n_window], axis=0)
        self.bel_v_b += np.sum(other.like_outmsg_v_b[: other.n_window], axis=0)

        self.pot_h_h += np.sum(other.like_outmsg_h_h[: other.n_window], axis=0)
        self.pot_h_K += np.sum(other.like_outmsg_h_K[: other.n_window], axis=0)
        self.pot_h_C = np.linalg.inv(self.pot_h_K)
        self.pot_h_m = self.pot_h_C.dot(self.pot_h_h)

        self.n_window = n_required

        self.converged = False

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
