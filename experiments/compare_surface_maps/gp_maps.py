# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Optional, Set, Tuple

import GPy as gpy
import numpy as np

from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

import stmmap.utils.triangle_transforms as tt


class GPMap:
    def __init__(self, dim=2, kernel_func=gpy.kern.Matern52):
        self.kernel = kernel_func(dim - 1, ARD=True)
        self.gp_model: gpy.models.GPRegression

    def update(self, m_z, C_z) -> None:
        m_z = np.copy(m_z)
        C_z = np.copy(C_z)  # Not used for anything

        X = m_z[:, :-1, 0]
        Y = m_z[:, -1]

        noise_var = np.mean(C_z[:, -1, -1])
        self.gp_model = gpy.models.GPRegression(X, Y, self.kernel, noise_var=noise_var)
        # self.gp_model = gpy.models.SparseGPRegression(X, Y, self.kernel)

        self.gp_model.optimize(messages=True, max_f_eval=1000)

    def loglike(self, test_pts: np.ndarray) -> float:
        """Calculate the log-likelihood for the model given some test points."""
        test_pts = np.copy(test_pts)

        x_test = test_pts[:, :-1, 0]
        y_test = test_pts[:, -1]

        # NOTE I don't think this will work...
        log_like = self.gp_model.log_predictive_density(x_test, y_test)
        return np.sum(log_like)

    def mean_squared_error(self, test_pts: np.ndarray) -> float:
        test_pts = np.copy(test_pts)

        N_test = test_pts.shape[0]
        x_test = test_pts[:, :-1, 0]
        y_test = test_pts[:, -1]
        y_pred = self.gp_model.predict(x_test)[0]

        squared_error = np.sum((y_pred - y_test) ** 2)

        mse = squared_error / N_test
        return mse


if __name__ == "__main__":
    from utils.generate_environments import gen_env_fn_2d, gen_meas_2d
    import random

    # Set noise seed
    seed = (int)(np.random.rand() * (2.0 ** 30))
    seed = 540398136
    np.random.seed(seed)
    random.seed(seed)
    print("Seed", seed)

    env_type = "simonsberg"
    env_func = gen_env_fn_2d(env_type)

    # XXX: Flags
    n_depths = 7  # Repeat for different grid depths
    divisions = 2
    N = int(10 * divisions ** (n_depths))  # Number of measurements
    nstds = 1  # Number of standard deviations to plot for height estimates

    # XXX Test points
    N_test = 10 * divisions ** n_depths
    test_pts = np.zeros((N_test, 2, 1))
    test_pts[:, 0, 0] = np.linspace(0, 1, N_test)
    test_pts[:, 1, 0] = env_func(test_pts[:, 0, 0])

    m_gen, C_gen = gen_meas_2d(env_func, N, scale=0.5)
    invalid = np.logical_or(m_gen[:, 0, 0] < 0, m_gen[:, 0, 0] > 1)
    keep = ~invalid
    m_gen = m_gen[keep, :, :]
    C_gen = C_gen[keep, :, :]
    m_raw = 1 * m_gen

    m_z = np.copy(m_gen)
    C_z = np.copy(C_gen)

    gp_map = GPMap(dim=2)
    gp_map.update(m_z, C_z)

    gpy.plotting.change_plotting_library("matplotlib")
    fig = plt.figure(constrained_layout=True)
    ax = plt.gca()
    gp_map.gp_model.plot(ax=ax, plot_data=False, plot_limits=[0, 1], resolution=1000)

    x = np.linspace(0, 1, 1000)
    y = env_func(x)
    ax.plot(x, y, "--", alpha=1, lw=1, zorder=6, label="Ground truth")
    ax.set_xlim(0, 1)

    plt.show()
