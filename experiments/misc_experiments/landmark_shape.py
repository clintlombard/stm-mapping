#!/usr/bin/env python3
"""Visualise the landmark shapes under different correlations."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from scipy.linalg import block_diag

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 10, "serif": ["Latin Modern Roman"]}
plt.rc("font", **font)
plt.rc("legend", **{"fontsize": 10})
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]


def plotEllipse(mean, cov, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance`
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        mean : The location of the center of the ellipse. Expects a
            2-element sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are passed on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)

    return ellip


def cov_corrcoef(c):
    stds = np.sqrt(np.diag(c))
    corr_coeffs = np.ones_like(c)
    n = c.shape[0]
    for i in range(n):
        for j in range(n):
            corr_coeffs[i, j] = c[i, j] / (stds[i] * stds[j])
    return corr_coeffs


def gen_cov(corr_coeff: float, std: float, std_ratio: float) -> np.ndarray:
    C_block = np.diag([std, std_ratio * std])
    # NOTE Rotations completely break the correlation effect
    angles = np.deg2rad([30, -30]) * 0
    R0 = np.array(
        [[np.cos(angles[0]), -np.sin(angles[0])], [np.sin(angles[0]), np.cos(angles[0])]]
    )
    C0 = R0.dot(C_block).dot(R0.T)

    R1 = np.array(
        [[np.cos(angles[1]), -np.sin(angles[1])], [np.sin(angles[1]), np.cos(angles[1])]]
    )
    C1 = R1.dot(C_block).dot(R1.T)
    C2 = C_block
    C = block_diag(C0, C1, C2)
    stds = np.sqrt(np.diag(C)).reshape(6, 1)

    sel = np.zeros_like(C)
    sel[2:, :4] += np.eye(4)
    sel[:4, 2:] += np.eye(4)
    sel[:2, 4:] += np.eye(2)
    sel[4:, :2] += np.eye(2)

    C += corr_coeff * sel * stds.dot(stds.T)
    return C


L = np.array([[-1, 0], [1, 0], [0, 2]])

std = 0.05
std_ratio = 3
# for corr in [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
for corr in [0.5, 0.8, 0.95]:
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = plt.gca()
    # ax.set_title(corr)
    plt.scatter(*L.T, color="C0")
    C = gen_cov(corr, std, std_ratio)
    print(cov_corrcoef(C))
    for i in range(7):
        x = np.random.multivariate_normal(L.flat, C).reshape(3, 2)
        plt.scatter(*x.T, color="C1")
        tri = plt.Polygon(x, ec="C1", fill=None, alpha=0.8)
        ax.add_patch(tri)

    for i, l in enumerate(L):
        plotEllipse(
            l,
            C[(2 * i) : (2 * i + 2), (2 * i) : (2 * i + 2)],
            nstd=1,
            fill=None,
            ec="C0",
            linewidth=2,
        )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    plt.savefig(f"corr_{corr}.pdf")
plt.show()
