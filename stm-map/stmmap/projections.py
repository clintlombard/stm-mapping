from __future__ import annotations

import logging

from enum import Enum, unique
from typing import TYPE_CHECKING, List

import numpy as np

from scipy.linalg import block_diag
from scipy.special import digamma, gammaln

from stmmap.utils.distances import DistanceMetric, distance_gauss, distance_invgamma

if TYPE_CHECKING:
    from stmmap.surfel import Surfel


logger = logging.getLogger(__name__)


class ProjectionMethod:
    def __init__(self, surf: Surfel):
        self.surfel = surf
        self.dim = surf.dim

        self.iterations = 0

        self._load()

    def _load(self):
        pass

    def iteration(self) -> float:
        """Iterate once through all the measurements."""
        pot_old_h_C = np.copy(self.surfel.pot_h_C)
        pot_old_h_m = np.copy(self.surfel.pot_h_m)

        bel_old_v_a = np.copy(self.surfel.bel_v_a)
        bel_old_v_b = np.copy(self.surfel.bel_v_b)

        batch_size = self.surfel.n_window

        if batch_size == 0:
            return 0.0

        order = np.random.choice(range(batch_size), batch_size, replace=False)
        for i in order:
            self.surfel.pot_h_h -= self.surfel.like_outmsg_h_h[i]
            self.surfel.pot_h_K -= self.surfel.like_outmsg_h_K[i]

            self.update_likelihood_cluster_potential(i)

            self.surfel.pot_h_h += self.surfel.like_outmsg_h_h[i]
            self.surfel.pot_h_K += self.surfel.like_outmsg_h_K[i]

        # self.surfel.pot_h_h = self.surfel.bel_h_h - self.surfel.inmsg_h_h
        # self.surfel.pot_h_K = self.surfel.bel_h_K - self.surfel.inmsg_h_K
        # # self.surfel.pot_h_h = self.surfel.prior_h_h + np.sum(
        # #     self.surfel.like_outmsg_h_h[: self.surfel.n_window], axis=0
        # # )
        # # self.surfel.pot_h_K = self.surfel.prior_h_K + np.sum(
        # #     self.surfel.like_outmsg_h_K[: self.surfel.n_window], axis=0
        # # )
        self.surfel.pot_h_C = np.linalg.inv(self.surfel.pot_h_K)
        self.surfel.pot_h_m = self.surfel.pot_h_C.dot(self.surfel.pot_h_h)

        # self.surfel.bel_h_h = self.surfel.pot_h_h + self.surfel.inmsg_h_h
        # self.surfel.bel_h_K = self.surfel.pot_h_K + self.surfel.inmsg_h_K
        # self.surfel.bel_h_C = np.linalg.inv(self.surfel.bel_h_K)
        # self.surfel.bel_h_m = self.surfel.bel_h_C.dot(self.surfel.bel_h_h)
        # distance = distance_gauss(
        #     bel_old_h_m, bel_old_h_C, self.surfel.bel_h_m, self.surfel.bel_h_C
        # )

        distance = distance_gauss(
            pot_old_h_m, pot_old_h_C, self.surfel.pot_h_m, self.surfel.pot_h_C
        )
        distance += distance_invgamma(
            bel_old_v_a, bel_old_v_b, self.surfel.bel_v_a, self.surfel.bel_v_b
        )

        return distance

    def update_likelihood_cluster_potential(self, i: int) -> None:
        """Update the likelihood cluster potential for the i-th measurement.

        Note this should also update the surfel belief.

        """
        raise NotImplementedError()


class VMPStructured(ProjectionMethod):
    def _load(self):
        # Predefine some matrices for the EIF
        taylor_dims = 2 * self.dim - 1
        self.G = np.zeros((2 * self.dim, taylor_dims), dtype=float)
        self.G[:taylor_dims, :] = np.eye(taylor_dims, dtype=float)

        self.C_stack = 100 * np.eye(taylor_dims, dtype=float)  # h, a, b

        # Selection matix
        self.S = np.zeros((self.dim, 2 * self.dim), dtype=float)
        self.S[:, self.dim :] = np.eye(self.dim, dtype=float)

    def update_likelihood_cluster_potential(self, i: int) -> None:
        logging_pre = f"z[{i}] : "

        m_z = self.surfel.z_m[i, :, 0]

        # Update heights
        # [ Calculate cavity distribution
        self.surfel.bel_h_K -= self.surfel.like_outmsg_h_K[i]
        self.surfel.bel_h_h -= self.surfel.like_outmsg_h_h[i]

        h_cav_K = np.copy(self.surfel.bel_h_K)
        h_cav_h = np.copy(self.surfel.bel_h_h)
        h_cav_C = np.linalg.inv(h_cav_K)
        h_cav_m = h_cav_C.dot(h_cav_h)
        # ]

        # EIF Prediction Update
        # [ Taylor
        m_pred = np.zeros((2 * self.dim, 1), dtype=float)
        m_pred[: self.dim] = h_cav_m
        a_prior = m_z[0]
        if self.dim == 3:
            b_prior = m_z[1]
            A = np.array((1 - a_prior - b_prior, a_prior, b_prior)).reshape(1, self.dim)
            m_pred[self.dim + 1] = b_prior
        else:  # 2-D
            A = np.array((1 - a_prior, a_prior)).reshape(1, self.dim)

        g_pred = A.dot(h_cav_m)
        m_pred[self.dim] = a_prior
        m_pred[-1] = g_pred

        self.G[-1, : self.dim] = A.flat
        self.G[-1, self.dim :] = h_cav_m[1:, 0] - h_cav_m[0, 0]

        self.C_stack[: self.dim, : self.dim] = h_cav_C
        C_pred = self.G.dot(self.C_stack).dot(self.G.T)
        C_pred[-1, -1] += self.surfel.bel_v_b / self.surfel.bel_v_a

        # Numerical Symmetry
        C_pred += C_pred.T
        C_pred /= 2

        if (np.diag(C_pred) < 0).any():
            msg = (
                logging_pre + "Weird prediction covariance matrix\n"
                f"Rank: {np.linalg.matrix_rank(C_pred)}\n"
                f"Diagonal: {np.diag(C_pred)}"
            )
            logger.warn(msg)

        # ]

        # EIF Measurement Update
        K_pred = np.linalg.inv(C_pred)
        h_pred = K_pred.dot(m_pred)
        z = m_z.reshape(self.dim, 1)
        K_tmp = K_pred + self.S.T.dot(self.surfel.z_K[i]).dot(self.S)
        h_tmp = h_pred + self.S.T.dot(self.surfel.z_K[i]).dot(z)

        # Numerical Symmetry
        K_tmp += K_tmp.T
        K_tmp /= 2

        C_tmp = np.linalg.inv(K_tmp)
        m_tmp = C_tmp.dot(h_tmp)

        if (np.diag(C_tmp) < 0).any():
            msg = (
                logging_pre + "Weird height belief covariance matrix\n"
                f"Rank: {np.linalg.matrix_rank(C_tmp)}\n"
                f"Diagonal: {np.diag(C_tmp)}"
            )
            logger.warn(msg)

        # Calculate measurement factor
        m_C_tmp = np.linalg.inv(K_tmp[self.dim :, self.dim :])
        self.surfel.like_outmsg_h_K[i] = (
            K_tmp[: self.dim, : self.dim]
            - self.surfel.bel_h_K
            - K_tmp[: self.dim, self.dim :].dot(m_C_tmp).dot(K_tmp[self.dim :, : self.dim])
        )
        self.surfel.like_outmsg_h_h[i] = (
            h_tmp[: self.dim, :]
            - self.surfel.bel_h_h
            - K_tmp[: self.dim, self.dim :].dot(m_C_tmp).dot(h_tmp[self.dim :, :])
        )

        # Update posterior
        self.surfel.bel_h_K += self.surfel.like_outmsg_h_K[i]
        self.surfel.bel_h_h += self.surfel.like_outmsg_h_h[i]
        self.surfel.bel_h_C = np.linalg.inv(self.surfel.bel_h_K)
        self.surfel.bel_h_m = self.surfel.bel_h_C.dot(self.surfel.bel_h_h)

        # Update variation
        # [ Calculate cavity distribution
        self.surfel.bel_v_a -= self.surfel.like_outmsg_v_a[i]
        self.surfel.bel_v_b -= self.surfel.like_outmsg_v_b[i]

        variation = self.surfel.bel_v_b / self.surfel.bel_v_a
        if variation < 0:
            msg = (
                logging_pre
                + f"The deviation cavity distribution expected value is negative: {variation}"
            )
            logger.warn(msg)
        # ]

        # expectation = self._exact_expectation(m_tmp, C_tmp)
        expectation = self._taylor_expectation(m_tmp, C_tmp)  # This is faster

        self.surfel.like_outmsg_v_b[i] = expectation / 2.0
        self.surfel.like_outmsg_v_a[i] = 0.5

        self.surfel.bel_v_a += self.surfel.like_outmsg_v_a[i]
        self.surfel.bel_v_b += self.surfel.like_outmsg_v_b[i]
        variation = self.surfel.bel_v_b / self.surfel.bel_v_a
        if variation < 0:
            msg = logging_pre + f"The deviation belief expected value is negative: {variation}"
            logger.warn(msg)

    def _exact_expectation(self, m, C):
        """Exact expectation calculation.

        This was derived using sympy.

        E_g(h,m)[(gamma - f(alpha, h))^2]

        """

        def second_order(m, C):
            return (m[0,] * m[1,] + C[0, 1]).item()

        def forth_order(m, C):
            E = 0.0
            E += second_order(m[(0, 1),], C[(0, 1),][:, (0, 1)]) * second_order(
                m[(2, 3),], C[(2, 3),][:, (2, 3)]
            )
            E += second_order(m[(0, 2),], C[(0, 2),][:, (0, 2)]) * second_order(
                m[(1, 3),], C[(1, 3),][:, (1, 3)]
            )
            E += second_order(m[(0, 3),], C[(0, 3),][:, (0, 3)]) * second_order(
                m[(1, 2),], C[(1, 2),][:, (1, 2)]
            )
            E += -2 * m[0,] * m[1,] * m[2,] * m[3,]

            return E.item()

        def third_order(m, C):
            E = 0.0
            E += m[0,] * m[1,] * m[2,]
            E += m[0,] * C[1, 2]
            E += m[1,] * C[0, 2]
            E += m[2,] * C[0, 1]

            return E.item()

        if self.dim == 3:
            # h0 ha hb a  b  g
            # 0  1  2  3  4  5
            E = 0.0
            # XXX 4th order
            # a**2*h0**2     | +1 | 0 0 3 3
            indices = (0, 0, 3, 3)
            E += forth_order(m[indices,], C[indices,][:, indices])
            # - 2*a**2*h0*ha | -2 | 0 1 3 3
            indices = (0, 1, 3, 3)
            E += -2 * forth_order(m[indices,], C[indices,][:, indices])
            # + a**2*ha**2   | +1 | 1 1 3 3
            indices = (1, 1, 3, 3)
            E += forth_order(m[indices,], C[indices,][:, indices])
            # + 2*a*b*h0**2  | +2 | 0 0 3 4
            indices = (0, 0, 3, 4)
            E += 2 * forth_order(m[indices,], C[indices,][:, indices])
            # - 2*a*b*h0*ha  | -2 | 0 1 3 4
            indices = (0, 1, 3, 4)
            E += -2 * forth_order(m[indices,], C[indices,][:, indices])
            # - 2*a*b*h0*hb  | -2 | 0 2 3 4
            indices = (0, 2, 3, 4)
            E += -2 * forth_order(m[indices,], C[indices,][:, indices])
            # + 2*a*b*ha*hb  | +2 | 1 2 3 4
            indices = (1, 2, 3, 4)
            E += 2 * forth_order(m[indices,], C[indices,][:, indices])
            # + b**2*h0**2   | +1 | 0 0 4 4
            indices = (0, 0, 4, 4)
            E += forth_order(m[indices,], C[indices,][:, indices])
            # - 2*b**2*h0*hb | -2 | 0 2 4 4
            indices = (0, 2, 4, 4)
            E += -2 * forth_order(m[indices,], C[indices,][:, indices])
            # + b**2*hb**2   | +1 | 2 2 4 4
            indices = (2, 2, 4, 4)
            E += forth_order(m[indices,], C[indices,][:, indices])

            # XXX 3rd order
            # + 2*a*y*h0     | +2 | 0 3 5
            indices = (0, 3, 5)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*a*y*ha     | -2 | 1 3 5
            indices = (1, 3, 5)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*a*h0**2    | -2 | 0 0 3
            indices = (0, 0, 3)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # + 2*a*h0*ha    | +2 | 0 1 3
            indices = (0, 1, 3)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])
            # + 2*b*y*h0     | +2 | 0 4 5
            indices = (0, 4, 5)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*b*y*hb     | -2 | 2 4 5
            indices = (2, 4, 5)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*b*h0**2    | -2 | 0 0 4
            indices = (0, 0, 4)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # + 2*b*h0*hb    | +2 | 0 2 4
            indices = (0, 2, 4)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])

            # XXX 2nd order
            # + y**2         | +1 | 5 5
            indices = (5, 5)
            E += second_order(m[indices,], C[indices,][:, indices])
            # - 2*y*h0       | -2 | 0 5
            indices = (0, 5)
            E += -2 * second_order(m[indices,], C[indices,][:, indices])
            # + h0**2        | +1 | 0 0
            indices = (0, 0)
            E += second_order(m[indices,], C[indices,][:, indices])
        else:  # 2-D
            # h0 ha a g
            # 0  1  2 3
            E = 0.0
            # XXX 4th order
            # a**2*h0**2     | +1 | 0 0 2 2
            indices = (0, 0, 2, 2)
            E += forth_order(m[indices,], C[indices,][:, indices])
            # - 2*a**2*h0*ha | -2 | 0 1 2 2
            indices = (0, 1, 2, 2)
            E += -2 * forth_order(m[indices,], C[indices,][:, indices])
            # + a**2*ha**2   | +1 | 1 1 2 2
            indices = (1, 1, 2, 2)
            E += forth_order(m[indices,], C[indices,][:, indices])

            # XXX 3rd order
            # + 2*a*g*h0     | +2 | 0 2 3
            indices = (0, 2, 3)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*a*g*ha     | -2 | 1 2 3
            indices = (1, 2, 3)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # - 2*a*h0**2    | -2 | 0 0 2
            indices = (0, 0, 2)
            E += -2 * third_order(m[indices,], C[indices,][:, indices])
            # + 2*a*h0*ha    | +2 | 0 1 2
            indices = (0, 1, 2)
            E += 2 * third_order(m[indices,], C[indices,][:, indices])

            # XXX 2nd order
            # + g**2         | +1 | 3 3
            indices = (3, 3)
            E += second_order(m[indices,], C[indices,][:, indices])
            # - 2*g*h0       | -2 | 0 3
            indices = (0, 3)
            E += -2 * second_order(m[indices,], C[indices,][:, indices])
            # + h0**2        | +1 | 0 0
            indices = (0, 0)
            E += second_order(m[indices,], C[indices,][:, indices])

        return E

    def _monte_expectation(self, m, C, N_samples=10000000):
        x = np.random.multivariate_normal(m[:, 0], C, N_samples).T

        h0 = x[0, :]
        ha = x[1, :]
        alphas = x[2, :]
        gammas = x[3, :]

        E = np.mean((gammas - ((ha - h0) * alphas + h0)) ** 2)

        return E

    def _taylor_expectation(self, m, C):
        if self.dim == 3:
            h_m = m[:3, :]
            m_m = m[3:, :]
            H = np.array(
                (
                    1 - m_m[0, 0] - m_m[1, 0],
                    m_m[0, 0],
                    m_m[1, 0],
                    h_m[1, 0] - h_m[0, 0],
                    h_m[2, 0] - h_m[0, 0],
                    1,
                )
            ).reshape(1, 6)
            g_pred = (
                (1 - m_m[0, 0] - m_m[1, 0]) * h_m[0, 0]
                + m_m[0, 0] * h_m[1, 0]
                + m_m[1, 0] * h_m[2, 0]
            )
            E_taylor = H.dot(C).dot(H.T) + (m_m[2] - g_pred) ** 2
        else:  # 2-D
            h_m = m[:2, :]
            m_m = m[2:, :]
            H = np.array((1 - m_m[0, 0], m_m[0, 0], h_m[1, 0] - h_m[0, 0], 1)).reshape(1, 4)
            g_pred = (1 - m_m[0, 0]) * h_m[0, 0] + m_m[0, 0] * h_m[1, 0]
            E_taylor = H.dot(C).dot(H.T) + (m_m[1] - g_pred) ** 2

        return E_taylor.item()


class VMPFactorised(ProjectionMethod):
    def update_likelihood_cluster_potential(self, i: int) -> None:
        # [ NOTE these are views not copies!
        z_m = self.surfel.z_m[i, :, :]
        z_h = self.surfel.z_h[i, :, :]
        z_K = self.surfel.z_K[i, :, :]
        # ]

        # Update q(m_i) ------------------------------------------------
        a_v = 100
        m_h_prior = np.zeros((self.dim, 1), dtype=float)
        m_h_prior[:-1] = z_m[:-1] / a_v
        m_K_prior = np.eye(self.dim, dtype=float) / a_v
        m_K_prior[-1, -1] = 0
        if self.dim == 3:
            B_m = np.array([[-1, -1, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            b_m = np.array([[1], [0], [0]], dtype=float)
            a_h = np.array(
                [
                    [self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[1, 0]],
                    [self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[2, 0]],
                    [1],
                ],
                dtype=float,
            )
        else:
            B_m = np.array([[-1, 0], [1, 0]], dtype=float)
            b_m = np.array([[1], [0]], dtype=float)
            a_h = np.array(
                [[self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[1, 0]], [1]], dtype=float
            )
        m_K_like = B_m.T.dot(self.surfel.bel_h_C).dot(B_m)
        m_K_like += a_h.dot(a_h.T)
        m_h_like = -B_m.T.dot(self.surfel.bel_h_C).dot(b_m)
        tmp = np.zeros((self.dim, 1), dtype=float)
        tmp[-1, 0] = self.surfel.bel_h_m[0, 0]
        m_h_like += a_h.dot(a_h.T).dot(tmp)

        variation = self.surfel.bel_v_b / self.surfel.bel_v_a
        m_K_like /= variation
        m_h_like /= variation

        m_h = z_h + m_h_like + m_h_prior
        m_K = z_K + m_K_like + m_K_prior

        m_C = np.linalg.inv(m_K)
        m_m = m_C.dot(m_h)

        # Update q(h) --------------------------------------------------
        if self.dim == 3:
            B_h = np.array([[1, -1, 0], [1, 0, -1], [0, 0, 0]], dtype=float)
            b_h = np.array([[0], [0], [1]], dtype=float)
            a_m = np.array([[1 - m_m[0, 0] - m_m[1, 0]], [m_m[0, 0]], [m_m[1, 0]]], dtype=float)
        else:
            B_h = np.array([[1, -1], [0, 0]], dtype=float)
            b_h = np.array([[0], [1]], dtype=float)
            a_m = np.array([[1 - m_m[0, 0]], [m_m[0, 0]]], dtype=float)
        h_K_like = B_h.T.dot(m_C).dot(B_h)
        h_K_like += a_m.dot(a_m.T)
        h_h_like = -B_h.T.dot(m_C).dot(b_h)

        tmp = np.zeros((self.dim, 1), dtype=float)
        tmp[0, 0] = m_m[-1, 0] / (1 - np.sum(m_m[:-1, 0]))
        h_h_like += a_m.dot(a_m.T).dot(tmp)

        h_K_like /= variation
        h_h_like /= variation

        self.surfel.bel_h_K -= self.surfel.like_outmsg_h_K[i]
        self.surfel.bel_h_h -= self.surfel.like_outmsg_h_h[i]
        self.surfel.like_outmsg_h_K[i] = h_K_like
        self.surfel.like_outmsg_h_h[i] = h_h_like
        self.surfel.bel_h_K += self.surfel.like_outmsg_h_K[i]
        self.surfel.bel_h_h += self.surfel.like_outmsg_h_h[i]
        self.surfel.bel_h_C = np.linalg.inv(self.surfel.bel_h_K)
        self.surfel.bel_h_m = self.surfel.bel_h_C.dot(self.surfel.bel_h_h)

        # Update q(v) --------------------------------------------------
        if self.dim == 3:
            a_h = np.array(
                [
                    [self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[1, 0]],
                    [self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[2, 0]],
                    [1],
                ],
                dtype=float,
            )
        else:
            a_h = np.array(
                [[self.surfel.bel_h_m[0, 0] - self.surfel.bel_h_m[1, 0]], [1]], dtype=float
            )

        u_h = np.zeros((self.dim, 1), dtype=float)
        u_h[-1, 0] = self.surfel.bel_h_m[0, 0]
        A_h = a_h.dot(a_h.T)

        E = np.trace(np.linalg.multi_dot((B_m.T, self.surfel.bel_h_C, B_m, m_C)))
        E += np.linalg.multi_dot((m_m.T, B_m.T, self.surfel.bel_h_C, B_m, m_m))
        E += 2 * np.linalg.multi_dot((b_m.T, self.surfel.bel_h_C, B_m, m_m))
        E += np.linalg.multi_dot((b_m.T, self.surfel.bel_h_C, b_m))
        E += np.trace(A_h.dot(m_C))
        E += np.linalg.multi_dot((m_m.T, A_h, m_m))
        E += -2 * np.linalg.multi_dot((u_h.T, A_h, m_m))
        E += np.linalg.multi_dot((u_h.T, A_h, u_h))
        E *= 0.5

        self.surfel.bel_v_a -= self.surfel.like_outmsg_v_a[i]
        self.surfel.bel_v_b -= self.surfel.like_outmsg_v_b[i]

        self.surfel.like_outmsg_v_b[i] = E.item()
        self.surfel.like_outmsg_v_a[i] = 0.5

        self.surfel.bel_v_a += self.surfel.like_outmsg_v_a[i]
        self.surfel.bel_v_b += self.surfel.like_outmsg_v_b[i]
