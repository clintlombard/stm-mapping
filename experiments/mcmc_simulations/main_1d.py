# -*- coding: utf-8 -*-
#!/usr/bin/python3

import itertools
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sympy as sym

from matplotlib import rc, rcParams
from mayavi import mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag
from scipy.special import gamma as gammafn
from scipy.stats import gamma, multivariate_normal, norm
from tqdm import tqdm

from plot_utils import plotEllipse

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 10, "serif": ["Latin Modern Roman"]}
plt.rc("font", **font)
plt.rc("legend", **{"fontsize": 10})
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

# sb.set(color_codes=True)
# sb.set(style="ticks")


def metrohastapprox(m_z, C_z, N_z, B=1000, N=10000):
    """
    Metropolis Hastings MCMC sampling approximation of the posterior

    B: Burn-in period
    N: Num. samples to be generated

    """

    def f(x, m_z, C_z):
        N = m_z.size
        C = C_z + x[1]
        w = np.exp(-0.5 * np.sum((x[0] - m_z) ** 2) / C) * C ** (-N / 2)
        # w = -0.5 * np.sum((x[0] - m_z)**2) / C - (N / 2) * np.log(C)
        return w

    # Proposal distribution
    C_p = np.diag([0.1, 0.01])
    g = lambda x: np.random.multivariate_normal(x.flatten(), C_p)

    samples = []
    x_old = np.array([[0], [0.1]])
    like_old = f(x_old, m_z, C_z)
    for i in range(B):
        x_prop = g(x_old)
        while x_prop[1] < 0:
            x_prop = g(x_old)

        like_prop = f(x_prop, m_z, C_z)
        ratio = like_prop / like_old
        ratio *= norm.cdf(x_prop[1], 0, np.sqrt(0.01)) / norm.cdf(x_old[1], 0, np.sqrt(0.01))
        thresh = np.random.rand()
        if ratio >= thresh:
            x_old = x_prop
            like_old = like_prop

    pbar = tqdm(total=N)
    i = 0
    while i < N:
        x_prop = g(x_old)
        while x_prop[1] < 0:
            x_prop = g(x_old)

        like_prop = f(x_prop, m_z, C_z)
        ratio = like_prop / like_old
        ratio *= norm.cdf(x_prop[1], 0, np.sqrt(0.01)) / norm.cdf(x_old[1], 0, np.sqrt(0.01))
        thresh = np.random.rand()
        if ratio >= thresh:
            pbar.update(1)
            i += 1
            x_old = x_prop
            like_old = like_prop
            samples.append(np.copy(x_prop))
    return np.array(samples).T


# -------- Constants and Flags ----------------------------------------------------------
seed = (int)(np.random.rand() * (2.0 ** 30))
seed = 89334069
np.random.seed(seed)
print("Seed", seed)

# Measurement consts
N_z = 10
C_z = 0.1

h_truth = 0
var_truth = 0.5

# ------------------------------------------------------------------------------

# Generate Measurements

print("------- Generating Measurements -------")
print("N =", N_z)
m_z = np.random.randn(N_z) * np.sqrt(var_truth) + h_truth
m_z += np.random.randn(N_z) * np.sqrt(C_z)

print("Metropolis Hastings")
sig_h = metrohastapprox(m_z, C_z, N_z)
print("Correlation Coefficients")
print(np.corrcoef(sig_h))

# Plotting results
print("------- Plotting Results -------")
sb.jointplot(*sig_h)

plt.show()
