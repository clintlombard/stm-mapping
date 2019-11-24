import logging

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)


class Unscented:
    def __init__(self, dim=None, spread=1):
        if dim is not None:
            self.preconfig = True
            self.__config__(dim, spread)
        else:
            self.preconfig = False

    def __config__(self, dim, spread=1):
        # UT tuning parameters
        self.n = dim
        self.beta = 2.0  # optimal for Gaussian
        self.kappa = 3 - self.n  # 3-n (for higher dimensional magic *cough* *cough*) or 0
        self.spread = spread  # number of std's
        self.alpha = np.sqrt(self.spread ** 2 / (self.n + self.kappa))  # lam=1-n
        lam = self.spread ** 2 - self.n  # alpha**2 * (n + kappa) - n # lam = 1 - n
        self.c = self.n + lam

        # Mean weights
        self.w_m = np.zeros((2 * self.n + 1), dtype=float)
        self.w_m[0] = lam / self.c
        self.w_m[1:] = 0.5 / self.c

        # Covariance weights
        self.w_c = np.copy(self.w_m)
        self.w_c[0] += 1 - self.alpha ** 2 + self.beta

    def calcSigma(self, mean, cov, spread=1):
        n = mean.size
        if n != self.n or not self.preconfig:
            logger.info("No preconfig for unscented transform")
            self.__config__(n, spread)
            self.preconfig = True

        try:
            cov_sqrt = np.linalg.cholesky(cov)  # Cholesky
        except Exception as e:
            logger.warn("Using the SVD instead of Cholesky")
            # Can be unstable, but always works. It's also slower.
            U, S, V = np.linalg.svd(cov)  # SVD - Safe for pos. semi-definite
            cov_sqrt = U.dot(np.diag(np.sqrt(S)))

        sig = np.tile(mean, (1, 2 * n + 1))
        # Note: This matrix is a batch calculation, and thus does not need to
        #   calculate for each column.
        sig[:, 1 : (n + 1)] += cov_sqrt * np.sqrt(self.c)
        sig[:, (n + 1) :] -= cov_sqrt * np.sqrt(self.c)

        return sig

    def calcCov(self, sig_t, mean):
        cov = (self.w_c * (sig_t - mean)).dot((sig_t - mean).T)

        return cov

    def calcMoments(self, sig_t):
        mean = np.sum(self.w_m * sig_t, axis=-1)[None].T
        cov = (self.w_c * (sig_t - mean)).dot((sig_t - mean).T)

        return mean, cov
