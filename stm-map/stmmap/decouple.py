# -*- coding: utf-8 -*-
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Decouple:
    def __init__(self, dim=None):
        if dim is not None:
            self.preconfig = True

            # UT tuning parameters
            n = dim
            self.beta = 2.0  # optimal for Gaussian
            self.kappa = 3 - n  # 3-n (for higher dimensional magic *cough* *cough*) or 0
            self.spread = 1  # number of std's
            self.alpha = np.sqrt(self.spread ** 2 / (n + self.kappa))  # lam=1-n
            lam = self.spread ** 2 - n  # alpha**2 * (n + kappa) - n # lam = 1 - n
            c = n + lam

            # Mean weights
            self.w_m = np.zeros((2 * n + 1), dtype=float)
            self.w_m[0] = lam / c
            self.w_m[1:] = 0.5 / c

            # Covariance weights
            self.w_c = np.copy(self.w_m)
            self.w_c[0] += 1 - self.alpha ** 2 + self.beta
            # self.w_c[1:] = np.copy(self.w_m[1:]) # NOTE: useless??
        else:
            self.preconfig = False

    def test_points(self, pts, lms, transform):
        """
        Parameters:
          * Points - (N, 3)
          * landmark poses - (M, 3)
          * transform - (R, t) tuple
        Returns:
          * Allocation matrix
            (which points belong and which neighbours to look in if they dont)
        """

        # Transform the landmarks to the sensor frame
        R = transform[0]
        t = transform[1]
        lms_sensor = (R.dot(lms.T) + t).T

        pts_rel = self.to_relative(pts, lms_sensor)

        a0, a1, a2, a3 = self.split_allocation(pts_rel)  # This is faster

        return pts_rel, a0, a1, a2, a3

    def split_allocation(self, pts_rel):
        """
        Allocation scheme:
        0 - valid
        1 - alpha < 0
        2 - beta < 0
        3 - alpha + beta > 1
        """
        # x1 = pts_rel[:, 0] < 0
        # x2 = pts_rel[:, 1] < 0
        # x3 = (pts_rel[:, 0] + pts_rel[:, 1]) > 1
        # a1 = x1
        # a2 = x2 & ~a1
        # a3 = x3 & ~a2 & ~a1
        # a0 = (~x3) & (~x2) & (~x1)

        a1 = pts_rel[:, 0] < 0
        a2 = (pts_rel[:, 1] < 0) & ~a1
        a3 = ((pts_rel[:, 0] + pts_rel[:, 1]) > 1) & ~a2 & ~a1
        a0 = (~a3) & (~a2) & (~a1)

        return a0, a1, a2, a3

    def to_relative(self, pts, lms):
        """
        Inputs:
          * Points - (N, 3) array
          * lms - (M, 3) array (M=3 always)
        Returns:
         * Relative Points - (N, 3) array
        """
        # la = alpha axis landmark (previously l1)
        # lb = beta axis landmark (previously l2)
        l0, la, lb = lms

        n = np.cross(la - l0, lb - l0)[None]
        n /= np.linalg.norm(n)

        # Shift origin to l0
        pts_l0 = pts - l0

        da = (la - l0).flatten()
        db = (lb - l0).flatten()
        tmp = da[1] * db[0] - da[0] * db[1]

        h = pts_l0.dot(n.T)
        pts_proj = pts_l0 - (h * n)
        alpha = pts_proj[:, 1] * db[0] - pts_proj[:, 0] * db[1]
        beta = pts_proj[:, 0] * da[1] - pts_proj[:, 1] * da[0]
        alpha /= tmp
        beta /= tmp

        pts_rel = np.empty_like(pts)
        pts_rel[:, 0] = alpha
        pts_rel[:, 1] = beta
        pts_rel[:, 2] = h.T

        return pts_rel

    def from_relative(self, pts_rel, lms):
        # la = alpha axis landmark (previously l1)
        # lb = beta axis landmark (previously l2)
        l0, la, lb = lms

        n = np.cross(la - l0, lb - l0)[None]
        n /= np.linalg.norm(n)

        da = la - l0
        db = lb - l0

        alpha = pts_rel[:, 0][None].T
        beta = pts_rel[:, 1][None].T
        h = pts_rel[:, 2][None].T

        pts = alpha * da + beta * db + l0
        pts += h * n

        return pts


# XXX: Test code
# tmp = DecoupleEfficient(18)
# pts = np.array([[0.1, 0.5, 0], [0, 0.4, 0.1]])
# N = int(5e6)
# pts = np.random.rand(N, 3)
# lms = np.array([[0.1, 0.1, 0], [0.9, 0, 0], [0, 0.9, 0]], dtype=float)
# R = np.eye(3)
# t = np.zeros((3, 1))
# transfrom = (R, t)
# tmp.test_points(pts, lms, transfrom)

# rel = tmp.to_relative(pts, lms)
# res = tmp.from_relative(rel, lms)
# print(pts)
# print(rel)
# print(res)
# print(np.allclose(pts, res))
