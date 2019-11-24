import logging
import os

import numpy as np

from noise import pnoise1, pnoise2
from opensimplex import OpenSimplex
from tqdm import trange
from transformations import concatenate_matrices, rotation_matrix

from .svg_extract import extract_surface_from_svg

logger = logging.getLogger(__name__)


def gen_meas_3d(func, N, scale=1):
    """ Generate a set of measurements according to a height function.
    """
    # Use 3-D dirichlet distribution to sample 3-simplex then project to 2-d
    # Gives a nicely uniform spread of points
    m_a, m_b = np.random.dirichlet((1, 1, 1), N)[:, :2].T
    m_g = func(m_a, m_b)
    m_z = np.array([m_a, m_b, m_g]).T.reshape(N, 3, 1)

    std_a = 0.01 * scale
    std_b = 0.01 * scale
    std_g = 0.02 * scale
    std_z = np.diag([std_a, std_b, std_g])

    alpha_rot = np.pi * np.random.random_sample(N) - np.pi / 2
    beta_rot = np.pi * np.random.random_sample(N) - np.pi / 2
    gamma_rot = np.pi * np.random.random_sample(N) - np.pi / 2
    xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    C_z = np.empty((N, 3, 3), dtype=float)

    for i in trange(N):
        # Rotate Covariance Matrix
        Rx = rotation_matrix(alpha_rot[i], xaxis)
        Ry = rotation_matrix(beta_rot[i], yaxis)
        Rz = rotation_matrix(gamma_rot[i], zaxis)
        R = concatenate_matrices(Rx, Ry, Rz)[:-1, :-1]

        C = R.dot(std_z * std_z).dot(R.T)
        e = np.random.multivariate_normal(np.zeros(3), C, 1).T

        m_z[i, :, :] += e
        C_z[i] = C

    return m_z, C_z


def gen_env_fn_3d(choice="planar", seed=None):
    """
    choice (str): 'planar', 'perlin', 'simplex', 'step'
    """
    if choice == "planar":
        h_truths = [1, 10, 1]
        h0 = h_truths[0]
        ha = h_truths[1]
        hb = h_truths[2]
        ma = ha - h0
        mb = hb - h0
        variation = 0.5
        func = lambda x, y: ma * x + mb * y + h0 + np.random.normal(0, variation)
    elif choice == "perlin":
        # offset = [1, -1]
        octaves = int(np.random.rand() * 12) + 4  # Affects terrain roughness
        # speed = [np.random.rand() * 3, np.random.rand() * 3]  # affects rate of variation
        speed = [1.8, 1.2]
        offset = [np.random.rand() * 500, np.random.rand() * 500]  # A randomness based on shift
        print("Perlin parameters:", octaves, speed, offset)
        func = lambda x, y: pnoise2(
            speed[0] * x + offset[0], speed[1] * y + offset[1], octaves
        )  # Perlin noise
    elif choice == "simplex":
        if seed is None:
            logger.warn("No noise see passed, using random seed.")
            simplex = OpenSimplex()
        else:
            simplex = OpenSimplex(seed=seed)
        # speed = [2, 5]
        # offset = [0, 0]
        speed = [np.random.rand() * 5, np.random.rand() * 5]  # affects rate of variation
        offset = [np.random.rand() * 500, np.random.rand() * 500]  # A randomness based on shift
        # print("Perlin parameters:", speed, offset)
        func = lambda x, y: simplex.noise2d(x=speed[0] * x + offset[0], y=speed[1] * y + offset[1])
    elif choice == "step":
        m = np.random.random_sample()
        c = np.random.random_sample()
        func = lambda x, y: 0 if y < m * x + c else 1  # Step
    else:
        raise ValueError("Invalid choice of environment.")

    func = np.vectorize(func)

    return func


def gen_meas_2d(func, N, scale=1):
    """ Generate a set of measurements according to a height function.
    """
    m_a = np.random.random_sample(N)
    m_g = func(m_a)
    m_z = np.array([m_a, m_g]).T.reshape(N, 2, 1)

    std_a = 0.01 * scale
    std_g = 0.05 * scale
    std_z = np.diag([std_a, std_g])

    C_z = np.empty((N, 2, 2), dtype=float)
    for i in trange(N):
        # Rotate Covariance Matrix
        theta = np.pi * np.random.random_sample() - np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        C = R.dot(std_z * std_z).dot(R.T)
        e = np.random.multivariate_normal(np.zeros(2), C, 1).T

        m_z[i, :, :] += e
        C_z[i] = C

    return m_z, C_z


def gen_env_fn_2d(choice="planar", path=None):
    """
    choice (str): 'planar', 'perlin', 'step'
    """
    if choice == "planar":
        ma = np.tan(np.random.rand() * 90)
        c = np.random.rand() * 10 - 5
        variation = np.random.rand() * 2
        print("Planar parameter:", ma, c, variation)
        func = lambda x: ma * x + c + np.random.normal(0, np.sqrt(variation))
    elif choice == "perlin":
        # Old params 3, 1.5, 5
        octaves = int(np.random.rand() * 10) + 1  # Affects terrain roughness
        speed = np.random.rand() * 10  # affects rate of variation
        offset = np.random.rand() * 500  # A randomness based on shift
        scale = np.random.rand() * 5 + 1  # A randomness based on shift
        print("Perlin parameters:", octaves, speed, offset, scale)
        func = lambda x: scale * pnoise1(speed * x + offset, octaves)  # Perlin noise
    elif choice == "step":
        # Step
        c = np.random.random_sample()
        func = lambda x: 0 if x < c else 2  # Step
    elif choice == "simonsberg":
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simonsberg.svg")
        func, _, _ = extract_surface_from_svg(path)
    else:
        raise ValueError("Invalid choice of environment.")

    f = np.vectorize(func)

    return f


if __name__ == "__main__":
    import mayavi.mlab as mlab

    # Set noise seed
    seed = (int)(np.random.rand() * (2.0 ** 30))
    print("Seed", seed)

    env_type = "perlin"
    # env_type = "simplex"
    env_func = gen_env_fn_3d(env_type, seed=seed)

    N_test_row = 100
    xy = np.mgrid[0 : 1 : N_test_row * 1j, 0 : 1 : N_test_row * 1j].T.reshape(N_test_row ** 2, 2)
    b = np.sum(xy, axis=1) <= 1
    xy = xy[b, :]
    N_test = xy.shape[0]

    test_pts = np.zeros((N_test, 3))
    test_pts[:, :2] = xy
    z = env_func(test_pts[:, 0], test_pts[:, 1])
    test_pts[:, -1] = z

    pts = mlab.points3d(*test_pts.T, z, scale_mode="none", scale_factor=0.0)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.contour_surface(mesh, colormap="viridis", contours=50)

    meas, _ = gen_meas_3d(env_func, N=5000, scale=1.5)
    meas = meas.reshape(-1, 3)
    mlab.points3d(*meas.T, scale_factor=0.01)

    mlab.show()
