# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import itertools
import logging
import os
import pickle
import random

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sb

from matplotlib import cm
from noise import pnoise1
from scipy.linalg import block_diag
from scipy.stats import invgamma, multivariate_normal, norm
from tqdm import tqdm

from stmmap import RelativeSubmap
from stmmap.projections import VMPFactorised, VMPStructured
from stmmap.utils.plot_utils import PlotConfig

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

root_logger = logging.getLogger(__name__)


def cov_corrcoef(C):
    stds = np.sqrt(np.diag(C))
    corr_coeffs = np.ones_like(C)
    N = C.shape[0]
    for i in range(N):
        for j in range(N):
            corr_coeffs[i, j] = C[i, j] / (stds[i] * stds[j])
    return corr_coeffs


def genMeas(func, N, std, ratio):
    """ Generate a set of measurements according to a height function."""
    m_a = np.random.random_sample(N)
    m_g = func(m_a)
    m_z = np.array([m_a, m_g]).T.reshape(N, 2, 1)

    std_a = std
    std_g = ratio * std
    std_z = np.diag([std_a, std_g])
    J = np.array([[std, 0.0], [0.0, 1.0]])

    C_z = np.empty((N, 2, 2), dtype=float)
    for i in tqdm(range(N)):
        # Rotate Covariance Matrix
        theta = np.deg2rad(5) * np.random.random_sample() - np.deg2rad(
            2.5
        )  # TODO add non-zero offset?
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        C = R.dot(std_z * std_z).dot(R.T)

        e = np.random.multivariate_normal(np.zeros(2), C, 1).T

        m_z[i, :, :] += e
        C_z[i] = C

    return m_z, C_z


def metrohastapprox(x_init, C_p_init, m_z, C_z, N_z, B=1000, N=10000, skip=1):
    """Metropolis Hastings MCMC sampling approximation of the posterior

    x_init: Initial starting point
    C_p_init: Initial proposal covariance
    B: Burn-in period
    N: Num. samples to be generated
    skip: N-th sample to keep

    Note the optimal acceptance rate (AR) should be around 23%.

    # Useful for calulating the correction factor when forcing samples of the variation > 0
    https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/

    """

    def log_like(x, m_z_stack, C_z_stack):
        """
        Calculate the log-likelihood for a set of parameters x.
        """
        A = m_z_stack[:, 0].flat
        G = m_z_stack[:, 1].flat
        h = x[:2].reshape(2)

        v = x[2]
        a_var = 100

        J = np.eye(2, dtype=float)
        J[1, 0] = h[1] - h[0]
        C_1 = J.dot(np.array([[a_var, 0.0], [0.0, v]], dtype=float)).dot(J.T)
        C = C_1 + C_z_stack
        C_inv = np.linalg.inv(C)

        m1 = (h[1] - h[0]) * A + h[0]

        w = -0.5 * (m1 - G) ** 2 * C_inv[:, 1, 1]
        sign, logdet = np.linalg.slogdet(C)
        w += -0.5 * sign * logdet

        w = np.sum(w)

        # # Priors
        v_a_p = 0.01
        v_b_p = 0.01
        w += invgamma.logpdf(v, a=v_a_p, scale=v_b_p)

        # Correlated prior
        off_diag = np.where(~np.eye(2, dtype=bool))
        corr = np.eye(2, dtype=float)
        corr[off_diag] = 0.5
        std = 100
        C_prior = np.full((2, 2), std, dtype=float)
        C_prior *= C_prior.T
        C_prior *= corr
        h_m_p = np.zeros(2, dtype=float)
        h_cov_p = C_prior

        w += multivariate_normal.logpdf(h, mean=h_m_p, cov=h_cov_p)

        if w == np.inf:
            root_logger.debug("Likelihood too big...")
            exit()
        return w

    assert skip >= 1, "skip must be >= 1"

    # Proposal distribution
    C_p = C_p_init
    prop_sample = lambda x, scale=1.0: np.random.multivariate_normal(x.flatten(), C_p * scale)

    samples = np.empty((N, 3))
    x_old = x_init
    log_like_old = log_like(x_old, m_z, C_z)
    if log_like_old == 0:
        root_logger.debug("Initial condition poor (likelihood = 0)")
        return

    # Optimise for the acceptance ratio
    root_logger.debug("Optimising Acceptance Ratio (20 < AR < 25):")
    scale = 1.1
    ar = 0  # acceptance ratio
    while 1:
        N_reject = 0
        N_accept = 0
        while (N_accept + N_reject) < B:
            x_prop = prop_sample(x_old, scale)

            log_like_prop = log_like(x_prop, m_z, C_z)
            log_ratio = log_like_prop - log_like_old

            thresh = np.random.rand()
            log_thresh = np.log(thresh)
            if log_ratio >= log_thresh:
                N_accept += 1
                x_old = x_prop
                log_like_old = log_like_prop
            else:
                N_reject += 1
            if N_accept == 0:
                ar = 0
            else:
                ar = N_accept / (N_accept + N_reject) * 100
        print(f"AR = {ar:6.2f}%", sep="", end="\r")

        if ar < 20:
            scale *= 1 - np.random.rand() * 0.5
        elif ar > 25:
            scale *= 1 + np.random.rand() * 0.5
        else:
            break
    print()
    root_logger.debug(f"Acceptance ratio = {ar:6.2f} %%")
    root_logger.debug("Proposal Covariance:")
    root_logger.debug(C_p_init * scale)
    root_logger.debug(f"Starting point: {x_old.flatten()}")

    pbar = tqdm(dynamic_ncols=True, total=N)
    N_accept = 0.0
    N_reject = 0.0
    ar = 0.0
    i_skip = 0
    i = 0
    while i < N:
        x_prop = prop_sample(x_old, scale)

        log_like_prop = log_like(x_prop, m_z, C_z)
        log_ratio = log_like_prop - log_like_old

        thresh = np.random.rand()
        log_thresh = np.log(thresh)
        if log_ratio >= log_thresh:
            N_accept += 1
            x_old = x_prop
            log_like_old = log_like_prop
        else:
            N_reject += 1

        if N_accept == 0:
            ar = 0
        else:
            ar = N_accept / (N_accept + N_reject) * 100
        if i_skip == (skip - 1):
            samples[i] = x_old
            i += 1
            i_skip = 0

            # Update progress bar
            post_dict = {"AR": "%3.f%%" % ar}
            pbar.update()
            pbar.set_postfix(post_dict)
        else:
            i_skip += 1

    pbar.close()
    root_logger.debug(f"Final acceptance ratio = {ar:6.2f} %%")
    return samples.T


def gen_env_fn(choice="planar"):
    """
    choice (str): 'planar', 'perlin', 'step'
    """
    if choice == "planar":
        ma = np.tan(np.random.rand() * 90)
        c = np.random.rand() * 10 - 5
        variation = np.random.rand() * 2
        root_logger.debug("Planar parameters: {ma} {c} {variation}")
        func = lambda x: ma * x + c + np.random.normal(0, np.sqrt(variation))
    elif choice == "perlin":
        # Old params 3, 1.5, 5
        octaves = int(np.random.rand() * 10) + 1  # Affects terrain roughness
        speed = np.random.rand() * 10  # affects rate of variation
        offset = np.random.rand() * 500  # A randomness based on shift
        scale = np.random.rand() * 5 + 1  # A randomness based on shift
        root_logger.debug("Perlin parameters: {octaves} {speed} {offset} {scale}")
        func = lambda x: scale * pnoise1(speed * x + offset, octaves)  # Perlin noise
    elif choice == "step":
        # Step
        c = np.random.random_sample()
        root_logger.debug("Step parameter: {c}")
        func = lambda x: 0 if x < c else 1  # Step
    else:
        raise ValueError("Invalid choice of environment.")

    f = np.vectorize(func)

    return f


# -------- Handle command line args -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--N_mcmc", help="Number of MCMC samples", type=int, default=1000)
parser.add_argument("--N_z", help="Number of measurements.", type=int, default=10)
parser.add_argument("--N_contours", help="Number of contours to plot", type=int, default=4)
parser.add_argument("--kde", help="Enable KDE.", action="store_true")
parser.add_argument("--seed", help="Noise seed.", type=int)
parser.add_argument(
    "--env", help="Environement type ('planar', 'perlin', 'step')", type=str, default="planar"
)
parser.add_argument("--debug", help="Enable debugging.", action="store_true")
parser.add_argument(
    "--std",
    help="The standard deviation of the measurements uncertainty in alpha.",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--ratio",
    help="The ratio between measurement uncertainty in alpha and gamma (gamma / alpha).",
    type=float,
    default=2,
)

args = parser.parse_args()

# -------- Constants and Flags ----------------------------------------------------------
seed = (int)(np.random.rand() * (2.0 ** 30))
if args.seed is not None:
    seed = args.seed
else:
    args.seed = seed
np.random.seed(seed)
random.seed(seed)
scipy.random.seed(seed)

debug = args.debug
kde = args.kde

N_contours = args.N_contours

np.set_printoptions(precision=5, linewidth=150)

# Measurement consts
N_z = int(args.N_z)

env_type = args.env
func = gen_env_fn(env_type)

# Create save save_path
if not debug:
    file_path = Path(__file__).resolve().parent
    session_dir = str(args.N_z) + "_" + str(seed)
    save_path = file_path / "results" / env_type / session_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the parsed arguments (for reproducibility)
    pickle.dump(args, open(save_path / "setup_args.p", "wb"))
    with open(save_path / "setup_args.txt", "w") as f:
        f.write(str(args))

# -------- Logging --------------------------------------------------------------------
level = logging.DEBUG  # This could be controlled with --log=DEBUG (I think)
output_file = save_path / "output.log"

loggers: List[logging.Logger] = []
loggers += [root_logger]
loggers += [logging.getLogger("stmmap")]

# create formatter and add it to the handlers
FORMAT = "%(asctime)s - %(name)s::%(funcName)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(FORMAT)

# create file handler which logs even debug messages
fh = logging.FileHandler(output_file, mode="w")
fh.setLevel(level)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(level)
ch.setFormatter(formatter)

for logger in loggers:
    logger.setLevel(level)
    logger.addHandler(fh)
    # If not remote the output to console as well
    if not args.debug:
        logger.addHandler(ch)

root_logger = loggers[0]
root_logger.debug("Logging initialised")

root_logger.debug(f"Seed {seed}")

# Generate Measurements
root_logger.debug("Generating Measurements")
root_logger.debug(f"N = {N_z}")
z_m, z_C = genMeas(func, N_z, args.std, args.ratio)
invalid = np.logical_or(z_m[:, 0, 0] < 0, z_m[:, 0, 0] > 1)
keep = ~invalid
z_m = z_m[keep, :, :]
z_C = z_C[keep, :, :]
N_z = z_m.shape[0]
root_logger.debug(f"N_actual {N_z}")

# Approximate
root_logger.debug("Calculating approximations")
m_approx = []
C_approx = []
v_a_approx = []
v_b_approx = []
corr_approx = []
approx_labels = []

# root_logger.debug("VMP Factorised")
# approx_labels.append("VMP factorised")
# corners = np.array([[0, 1]], dtype=float)
# uid = (0,)
# rv_ids = (0, 1)
# graph_id = 0
# submap = RelativeSubmap(0, tolerance=1e-8, dim=2, projection_method=VMPFactorised)
# submap.insert_measurements(1 * z_m, 1 * z_C)
# submap.update()
# surfel = [ s for s in submap ][0]
# h_m = surfel.bel_h_m
# h_C = surfel.bel_h_C
# v_a = surfel.bel_v_a.item()
# v_b = surfel.bel_v_b.item()
# # Gaussian projection
# v_m = v_b / v_a
# v_var = v_b ** 2 / ((v_a - 1) ** 2 * (v_a - 2))
# m_joint = np.vstack((h_m, v_m))
# C_joint = block_diag(h_C, v_var)
# corr_joint = cov_corrcoef(C_joint)
# root_logger.debug(m_joint.flatten())
# root_logger.debug(C_joint)
# root_logger.debug(corr_joint)
# m_approx.append(m_joint)
# C_approx.append(C_joint)
# v_a_approx.append(v_a)
# v_b_approx.append(v_b)
# corr_approx.append(corr_joint)

root_logger.debug("VMP Structured")
approx_labels.append("VMP structured")
submap = RelativeSubmap(0, tolerance=1e-8, dim=2, projection_method=VMPStructured)
submap.insert_measurements(1 * z_m, 1 * z_C)
submap.update()
surfel = [s for s in submap][0]
h_m = surfel.bel_h_m
h_C = surfel.bel_h_C
v_a = surfel.bel_v_a.item()
v_b = surfel.bel_v_b.item()
# Gaussian projection
v_m = v_b / v_a
v_var = v_b ** 2 / ((v_a - 1) ** 2 * (v_a - 2))

m_joint = np.vstack((h_m, v_m))
C_joint = block_diag(h_C, v_var)
corr_joint = cov_corrcoef(C_joint)
root_logger.debug(m_joint.flatten())
root_logger.debug(C_joint)
root_logger.debug(corr_joint)
m_approx.append(m_joint)
C_approx.append(C_joint)
v_a_approx.append(v_a)
v_b_approx.append(v_b)
corr_approx.append(corr_joint)

root_logger.debug("Metropolis Hastings")
root_logger.debug(f"N_mcmc = {args.N_mcmc}")
x_init = np.array([[h_m[0, 0]], [h_m[1, 0]], [v_b / v_a]])
C_p_init = np.diag([h_C[0, 0], h_C[1, 1], v_var])
mh_samples = metrohastapprox(x_init, C_p_init, 1 * z_m, 1 * z_C, N_z, N=args.N_mcmc, skip=1)
m_exact = np.mean(mh_samples, axis=1)
C_exact = np.cov(mh_samples)
corr_exact = np.corrcoef(mh_samples)
root_logger.debug(m_exact)
root_logger.debug(C_exact)
root_logger.debug(corr_exact)

# Plotting results
root_logger.debug("Plotting Results")
approx_colors = ["C1", "C2"]
ellipses = []

pal = sb.cubehelix_palette(rot=-0.25, light=1, gamma=1.2, as_cmap=True)
labels = [r"$h_0$", r"$h_\alpha$", r"$\nu$"]
for c in itertools.combinations(np.array([0, 1, 2]), 2):
    indices = np.array(c, dtype=int)
    stds = np.sqrt(np.diag(C_exact[np.ix_(indices, indices)]))
    scale = 3
    y_std = np.sqrt(C_exact[indices[1], indices[1]])
    x_std = np.sqrt(C_exact[indices[0], indices[0]])
    y_lims = tuple(m_exact[indices[1]] + scale * np.array([-y_std, y_std]))
    x_lims = tuple(m_exact[indices[0]] + scale * np.array([-x_std, x_std]))
    if 2 in c:
        y_lims = (max(0, y_lims[0]), y_lims[1])

    fig_height = 0.3 * plt_cfg.tex_textwidth
    if kde:
        g = sb.jointplot(
            *mh_samples[indices, :],
            kind="kde",
            cmap="Blues",
            height=fig_height,
            n_levels=N_contours,
            shade_lowest=False,
            space=0,
        )
    else:
        joint_kws = dict(gridsize=50, linewidths=0.2, extent=x_lims + y_lims)
        g = sb.jointplot(
            *mh_samples[indices, :],
            kind="hex",
            height=fig_height,
            marginal_kws={"norm_hist": True, "bins": "auto"},
            cmap="Blues",
            joint_kws=joint_kws,
            xlim=x_lims,
            ylim=y_lims,
            space=0,
        )
    ax = g.ax_joint
    light_col = cm.get_cmap("Blues")(0)
    ax.set_facecolor(light_col)
    ax.set_xlabel(labels[c[0]])
    ax.set_ylabel(labels[c[1]])

    # Scale axis extents
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)

    for i in range(len(m_approx)):
        ax = g.ax_marg_x
        x = np.linspace(*ax.get_xlim(), 1000)
        pdf_x = norm(
            loc=m_approx[i][indices[0]], scale=np.sqrt(C_approx[i][indices[0], indices[0]])
        )
        ax.plot(x, pdf_x.pdf(x), color=approx_colors[i])

        ax = g.ax_marg_y
        y = np.linspace(*ax.get_ylim(), 1000)
        if 2 in indices:
            pdf_y = invgamma(a=v_a_approx[i], scale=v_b_approx[i])
        else:
            pdf_y = norm(
                loc=m_approx[i][indices[1]], scale=np.sqrt(C_approx[i][indices[1], indices[1]])
            )
        ax.plot(pdf_y.pdf(y), y, color=approx_colors[i])

        # NOTE Uncomment this to check the IG fit to the MH samples
        # if 2 in indices:
        #     params = invgamma.fit(mh_samples[2, :])
        #     ig = invgamma(*params)
        #     pdf = ig.pdf(y)
        #     ax.plot(pdf, y, color='red')

        # Calculate approximation contours
        # Credit: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
        # NOTE This might only work nicely for a single approximation
        ax = g.ax_joint
        X, Y = np.mgrid[x_lims[0] : x_lims[1] : 100j, y_lims[0] : y_lims[1] : 100j]
        if 2 in indices:
            Z = pdf_x.pdf(X) * pdf_y.pdf(Y)
        else:
            pdf = multivariate_normal(
                mean=m_approx[i][indices].flat, cov=C_approx[i][np.ix_(indices, indices)]
            )
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            Z = pdf.pdf(pos)
        Z /= np.sum(Z)

        # n = 1000
        # t = np.linspace(0, Z.max(), n)
        # integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1,2))

        # from scipy import interpolate
        # f = interpolate.interp1d(integral, t)
        # t_contours = f(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
        # tmp = r"%s ($\rho$ = %.3f)" % (
        #     approx_labels[i], corr_approx[i][indices[0], indices[1]].item())

        # t_contours = np.linspace(np.min(Z), np.max(Z), N_contours + 2)
        t_contours = np.linspace(np.min(Z), np.max(Z), N_contours + 2)[1:-1]
        ax.contour(X, Y, Z, t_contours, colors=approx_colors[i])

        # ax.tick_params(axis="both", which="both", labelsize=plt_cfg.tex_fontsize - 2)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.tight_layout()

    if not debug:
        base_filename = str(args.N_z) + "_" + str(c[0]) + str(c[1])
        g.savefig(os.path.join(save_path, base_filename + ".pdf"))
        g.savefig(os.path.join(save_path, base_filename + ".png"))

if debug:
    plt.show()
