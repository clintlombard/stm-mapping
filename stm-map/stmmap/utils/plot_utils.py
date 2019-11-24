# -*- coding: utf-8 -*-
#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


class PlotConfig:
    def __init__(self, tex_fontsize=10, tex_textwidth=6.29707):
        self.tex_fontsize = tex_fontsize
        self.tex_textwidth = tex_textwidth

        sns.set(
            rc={
                "font.size": self.tex_fontsize,
                "axes.titlesize": self.tex_fontsize,
                "axes.labelsize": self.tex_fontsize,
                "legend.fontsize": self.tex_fontsize,
                "xtick.labelsize": self.tex_fontsize,
                "ytick.labelsize": self.tex_fontsize - 2,
                "patch.facecolor": "black",
                "patch.edgecolor": "black",
            },
            style="ticks",
            color_codes=True,
        )

        plt.rc("text", usetex=True)
        plt.rc("legend", **{"fontsize": tex_fontsize})
        plt.rcParams["text.latex.preamble"] = [
            r"\usepackage{amsmath}",
            r"\usepackage{bm}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
        ]
        plt.rcParams["svg.fonttype"] = "none"  # Don't save text as path
        plt.rcParams["font.family"] = ["Latin Modern Roman"]
        plt.rcParams["font.size"] = tex_fontsize


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def plotAxes(rotation, translation, c1="k", c2="r", ax=None, scale=3):
    """Plots the calibration results.

    Parameters
    ----------
    rotation : numpy array
        3x3 rotation array as defined by 3-2-1 Euler rotations.
    translation : numpy array
        Vector describing the sensor axis offset from the reference.
    points_ref : numpy array
        3xN array of the measurements in the reference frame.
    points_sensor : numpy array
        3xN array of the measurements in the sensor frame.
    ax : 3d matplotlib axis
        Axis to plot on.
    scale : float
        Scale of axes arrows.
    """
    # rotation = rotation[0:3, 0:3]
    # t = translation.reshape(3, 1)
    t = translation.flatten()

    assert len(t) == 3

    if ax == None:
        ax = plt.gca()

    bounds_plus = scale * np.eye(3)
    ax.scatter(*bounds_plus, alpha=0)
    bounds_minus = -scale * np.eye(3)
    ax.scatter(*bounds_minus, alpha=0)
    # add reference axis
    ax.scatter(0, 0, 0, alpha=0)  # dummy point at reference origin
    refx = Arrow3D([0, scale], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color=c1)
    refy = Arrow3D([0, 0], [0, scale], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color=c1)
    refz = Arrow3D([0, 0], [0, 0], [0, scale], mutation_scale=20, lw=1, arrowstyle="-|>", color=c1)
    ax.add_artist(refx)
    ax.add_artist(refy)
    ax.add_artist(refz)
    # ax.text(scale, 0, 0, r"$x_{r}$", color='black')
    # ax.text(0, scale, 0, r"$y_{r}$", color='black')
    # ax.text(0, 0, scale, r"$z_{r}$", color='black')

    # add sensor axis
    # ax.scatter(t[0], t[1], t[2], alpha=0)
    # # NOTE: scale vectors by the current size of the axes
    # extents = np.array(
    #     [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    # scale_vec = extents[:, 1] - extents[:, 0]
    # scale_vec = 1 / scale_vec
    # scale_vec /= np.linalg.norm(scale_vec)
    sx, sy, sz = (rotation.dot(np.eye(3))).T + t
    sensorx = Arrow3D(
        [t[0], sx[0]],
        [t[1], sx[1]],
        [t[2], sx[2]],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="r",
    )
    sensory = Arrow3D(
        [t[0], sy[0]],
        [t[1], sy[1]],
        [t[2], sy[2]],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="g",
    )
    sensorz = Arrow3D(
        [t[0], sz[0]],
        [t[1], sz[1]],
        [t[2], sz[2]],
        mutation_scale=20,
        lw=1,
        arrowstyle="-|>",
        color="b",
    )
    ax.add_artist(sensorx)
    ax.add_artist(sensory)
    ax.add_artist(sensorz)
    # ax.text(sx[0], sx[1], sx[2], r"$x$", color='black')
    # ax.text(sy[0], sy[1], sy[2], r"$y$", color='black')
    # ax.text(sz[0], sz[1], sz[2], r"$z$", color='black')

    ax.set_xlabel(r"x (m)")
    ax.set_ylabel(r"y (m)")
    ax.set_zlabel(r"z (m)")
    axisEqual3D(ax)


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
    # ax.add_patch(ellip)

    return ellip


class HandlerEllipse(HandlerPatch):
    """
    Use this to add ellipse to legend
    ax.legend([ellip], ["Name"], handler_map={mpatches.Ellipse: HandlerEllipse()})
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
