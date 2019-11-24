# -*- coding: utf-8 -*-
import ast
import itertools
import logging
import math
import os
import sys

import matplotlib.pylab as plt
import matplotlib.tri as mtri
import mayavi.mlab as mlab
import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

import pyemdw as pyemdw

logger = logging.getLogger(__name__)


def read_iteration(directory, folder_name, subsample=0, fov=[-85, 45]):
    """ Read an iterations data

    directory
        Root directory of the dataset.

    folder_name : str.
        Name of the folder. Of the form:
            some_folder_name_<zero padded measurment number>

    subsample : int, optional.
       The amount of points to subsample by (0 does no subsampling).
       NOTE This is random subsampling.
    """
    if type(subsample) != int:
        raise TypeError("subsample must be of type int.")
    if subsample < 0:
        raise ValueError("subsample must be positive.")

    f = folder_name
    # Read inclination
    # NOTE only needed really to align the initial pose with gravity vec
    R_inc = np.genfromtxt(directory / f / f"{f}.inc")
    T_inc = np.eye(4)
    T_inc[:3, :3] = R_inc

    # Ground Truth .gt
    gt = np.genfromtxt(directory / f / f"{f}.gt")

    # Lidar angular data .fcl (azi, elev, r)
    df = pd.read_csv(directory / f / f"{f}.fcl", skiprows=11, delimiter=" ", header=None)
    pts_sensor = df.to_numpy(dtype=float)

    #
    b = (pts_sensor[:, 1] > fov[0]) & (pts_sensor[:, 1] < fov[1])
    pts_sensor = pts_sensor[b]

    # Calculate subsampled data
    N = pts_sensor.shape[0]
    N_subsampled = int(N / (subsample + 1))
    indices = np.random.choice(N, N_subsampled, replace=False)
    pts_sensor = pts_sensor[indices, :]

    pts_sensor[:, :2] = np.deg2rad(pts_sensor[:, :2])
    valid = (pts_sensor[:, 2] < 20) & (pts_sensor[:, 2] > 0)
    valid_pts_sensor = pts_sensor[valid, :]
    reorder = (2, 0, 1)  # (azi, elev, rho) -> (rho, azi, elev)
    valid_pts_sensor = valid_pts_sensor[:, reorder]
    valid_pts_sensor[:, 1] *= -1  # Azimuth from lidar is in the opposite order

    return T_inc, gt, valid_pts_sensor


def read_all_data(directory, subsample=0):
    """Read in data from Utoronto dataset

    directory
        Root directory of the dataset

    subsample : int, optional.
       The amount of points to subsample by (0 does no subsampling).
       NOTE This is random subsampling.
    """
    if type(subsample) != int:
        raise TypeError("subsample must be of type int.")
    if subsample < 0:
        raise ValueError("subsample must be positive.")

    # Determine folders
    folders = []
    for f in os.listdir(directory):
        folders.append(f)
    folders.sort()

    N_total = 0
    pts_sensor_list = []
    pos_list = []
    gt_list = []
    for f in tqdm(folders):
        try:
            i = int(f[-3:])
        except:
            logger.debug(f"Probably invalid folder: {f}")
            continue

        T_inc, gt, pts_sensor = read_iteration(directory, f, subsample)

        if i == 0:
            T_init = T_inc

        # Adjust the gt base on the initial inclination
        gt_adj = T_init.dot(gt)

        # Extract positions to help demarcate the mapping area
        pos = gt_adj[:3, 3].reshape(3, 1)

        N_total += pts_sensor.shape[0]

        gt_list.append(gt_adj)
        pos_list.append(pos)
        pts_sensor_list.append(pts_sensor)

    return pts_sensor_list, pos_list, gt_list, N_total


def read_all_data_test(directory, subsample=0):
    """Read in data from Utoronto dataset

    directory
        Root directory of the dataset

    subsample : int, optional.
       The amount of points to subsample by (0 does no subsampling).
       NOTE This is random subsampling.
    """
    if type(subsample) != int:
        raise TypeError("subsample must be of type int.")
    if subsample < 0:
        raise ValueError("subsample must be positive.")

    # Determine folders
    folders = []
    for f in os.listdir(directory):
        folders.append(f)
    folders.sort()

    N_total = 0
    pts_sensor_list = []
    pts_world_list = []
    pos_list = []
    gt_list = []
    for f in tqdm(folders):
        try:
            i = int(f[-3:])
        except:
            logger.debug(f"Probably invalid folder: {f}")
            continue

        T_inc, gt, pts_sensor = read_iteration(directory, f, subsample)

        if i == 0:
            T_init = T_inc

        # Adjust the gt base on the initial inclination
        gt_adj = T_init.dot(gt)

        # Extract positions to help demarcate the mapping area
        pos = gt_adj[:3, 3].reshape(3, 1)

        # Calculate points in the world IRF
        rho = pts_sensor[:, 0].flatten()
        azi = pts_sensor[:, 1].flatten()
        elev = pts_sensor[:, 2].flatten()

        xyz = np.zeros_like(pts_sensor, dtype=float)
        xyz[:, 0] = rho * np.cos(elev) * np.cos(azi)
        xyz[:, 1] = rho * np.cos(elev) * np.sin(azi)
        xyz[:, 2] = rho * np.sin(elev)

        # Convert from sensor irf to world irf
        homo_sensor_irf = np.insert(xyz, 3, 1, axis=1).T
        homo_world_irf = gt_adj.dot(homo_sensor_irf)
        pts_world = homo_world_irf[:3, :]

        gt_list.append(gt_adj)
        pos_list.append(pos)
        pts_sensor_list.append(pts_sensor)
        pts_world_list.append(pts_world)

    return pts_sensor_list, pts_world_list, pos_list, gt_list


def manually_extract_pseudo_lms(pos_list, keep, div):
    """Determine pseudo-landmarks

    Calculate the landmarks using the positions of the robot as anchor points.
    NOTE For now this is only a nice approach for square datasets.

    pos_list : list.
        List of robot positions.

    """
    if len(keep) != 3:
        raise ValueError("keep tuple must contain at least 3 elements.")
    if div < 1 or type(div) != int:
        raise ValueError("div must be >= 1 and an integer.")

    pos_arr = np.hstack(pos_list).T

    if len(keep) != 3:
        raise ValueError("Can only keep 3 landmarks")
    lms = pos_arr[keep, :].reshape(3, 3, 1)  # (N, d, 1)
    # Enforce all lms in the same plane
    lms[1:, 2] = lms[0, 2]

    # Axis vectors
    vec_x = lms[1, :] - lms[0, :]
    axis_x = vec_x / np.linalg.norm(vec_x)
    axis_z = np.array([[0, 0, 1]]).T
    axis_y = np.cross(axis_x.flatten(), axis_z.flatten()).reshape(3, 1)
    vec_2 = lms[2, :] - lms[0, :]
    vec_y = axis_y.T.dot(vec_2) * axis_y

    vec_x = vec_x.flatten()
    vec_y = vec_y.flatten()

    N_lms = (div + 1) ** 2
    spacing = np.linspace(0, 1, div + 1)
    x, y = np.meshgrid(spacing, spacing)
    coords = np.hstack(
        (x.reshape(N_lms, 1), y.reshape(N_lms, 1), np.zeros((N_lms, 1)))
    )  # (N_lms, 3)

    pseudo_lms = np.zeros((N_lms, 3, 1), dtype=float)
    pseudo_lms[:, :, 0] = coords[:, 0, None] * vec_x + coords[:, 1, None] * vec_y
    pseudo_lms += lms[0, :]

    triangulation = Delaunay(pseudo_lms[:, :2, 0])

    return pseudo_lms, triangulation


def extract_pseudo_lms(pos_list):
    """Determine pseudo-landmarks

    Calculate the landmarks using the positions of the robot as anchor points.
    NOTE For now this is only a nice approach for square datasets.

    pos_list : list.
        List of robot positions.

    """
    pos_arr = np.hstack(pos_list).T

    fig = plt.figure()
    plt.ion()
    plt.axis("equal")

    hull = ConvexHull(pos_arr[:, :2])
    indices = hull.vertices
    indices = np.hstack((indices, indices[0]))
    # Visual check for triangulation
    plt.gca().clear()
    plt.scatter(*pos_arr[:, :2].T)
    plt.plot(*pos_arr[indices, :2].T)
    for i in indices:
        plt.text(*pos_arr[i, :2].T, s=str(i))
    plt.draw()
    plt.pause(0.01)
    while 1:
        keep_str = input(
            "Enter the IDs of landmarks to be keep (comma seperated, blank continues, order matters!): "
        )
        try:
            keep_new = ast.literal_eval(keep_str)

            # XXX incase something wrong is entered, just restart loop
            if type(keep_new) is int:
                continue
            elif len(keep_new) != 3:
                continue
            else:
                keep = keep_new
        except Exception:
            logger.exception("Error understanding input")
            break

        plt.gca().clear()
        plt.scatter(*pos_arr[:, :2].T)
        plt.plot(*pos_arr[keep[:2], :2].T)
        plt.plot(*pos_arr[keep[::2], :2].T)
        for i in indices:
            plt.text(*pos_arr[i.T, :2], s=str(i))
        plt.draw()
        plt.pause(0.01)

    if len(keep) != 3:
        raise ValueError("Can only keep 3 landmarks")
    lms = pos_arr[keep, :].reshape(3, 3, 1)  # (N, d, 1)
    # Enforce all lms in the same plane
    lms[1:, 2] = 1 * lms[0, 2]

    # Axis vectors
    vec_x = lms[1, :] - lms[0, :]
    axis_x = vec_x / np.linalg.norm(vec_x)
    axis_z = np.array([[0, 0, 1]]).T
    axis_y = np.cross(axis_x.flatten(), axis_z.flatten()).reshape(3, 1)
    vec_2 = lms[2, :] - lms[0, :]
    vec_y = axis_y.T.dot(vec_2) * axis_y

    vec_x = vec_x.flatten()
    vec_y = vec_y.flatten()

    tri = Delaunay(lms[:, :2, 0])
    simplices = tri.simplices

    # Visual check for triangulation
    plt.gca().clear()
    plt.triplot(*lms[:, :2, 0].T, simplices)
    plt.scatter(*pos_arr[:, :2].T)
    plt.draw()
    plt.pause(0.01)

    indices = np.arange(0, lms.shape[1])
    valid_indices = np.arange(0, lms.shape[1])
    while 1:
        div_str = input("Enter the number of divisions (>= 1): ")
        try:
            div = ast.literal_eval(div_str)
            if (type(div) is not int) or (div < 0):
                raise ValueError("Divisons must be integer and >=1.")

        except Exception:
            logger.exception("Error understanding input")
            div = 0
            break

        N_lms = (div + 1) ** 2
        spacing = np.linspace(0, 1, div + 1)
        x, y = np.meshgrid(spacing, spacing)
        coords = np.hstack(
            (x.reshape(N_lms, 1), y.reshape(N_lms, 1), np.zeros((N_lms, 1)))
        )  # (N_lms, 3)

        pseudo_lms = np.zeros((N_lms, 3, 1), dtype=float)
        pseudo_lms[:, :, 0] = coords[:, 0, None] * vec_x + coords[:, 1, None] * vec_y
        pseudo_lms += lms[0, :]

        triangulation = Delaunay(pseudo_lms[:, :2, 0])

        # Check number of regions from triangulation matches the divisions
        N = triangulation.simplices.shape[0]

        # This shouldn't be an issue anymore
        # assert (N == 2*div**2) , "Number of triangulation regions is
        # inconsistent with number of divisions."

        simplices = triangulation.simplices

        # Visual check for triangulation
        plt.gca().clear()
        plt.triplot(*pseudo_lms[:, :2, 0].T, simplices)
        plt.scatter(*pos_arr[:, :2].T)
        plt.draw()
        plt.pause(0.01)

    plt.close(fig)
    plt.ioff()

    return pseudo_lms, triangulation


def find_approx_associations(triangulation, pts):
    # XXX Associate points with LTRs
    associations = triangulation.find_simplex(pts[:, :2, 0])
    selections = {}
    N_regions = triangulation.simplices.shape[0]
    for i in range(N_regions):
        b = associations == i
        selections[i] = b

    return selections


def sensor_to_world(transform, pts_sensor, cov):
    """Convert sensor IRF -> world IRF

    Because of memory constraints on such a large dataset I'm going to rather
    use Taylor series linearisation, as opposed to the unscented transform:

    500000 points * 2*(n=3)+1 sigma pts = 3.5M points

    Parameters:

    transform : ndarray, (4, 4).
        Homogenous rotation and translation matrix representation of the
        transformation from the sensor IRF to the world IRF

    cov : ndarray, (3, 3).
        Uncertainty covariance in the sensor IRF

    pts_sensor : ndarray, (N_meas, 3).
        Measurements (rho, azi, elev)


    Returns:
    pts_world : ndarray, (N_meas, 3).
        Transformed measurements (x, y, z)

    covs : ndarray (N_meas, 3, 3).
        The transformed covariances for each measurement.

    """
    # Transform from the spherical to cartesian
    rho = pts_sensor[:, 0]
    azi = pts_sensor[:, 1]
    elev = pts_sensor[:, 2]

    # Calculate cartesian homogenous coordinates
    N = pts_sensor.shape[0]
    xyz_homo = np.zeros((N, 4), dtype=float)
    xyz_homo[:, 0] = rho * np.cos(elev) * np.cos(azi)
    xyz_homo[:, 1] = rho * np.cos(elev) * np.sin(azi)
    xyz_homo[:, 2] = rho * np.sin(elev)
    xyz_homo[:, 3] = 1.0

    # Convert from sensor irf to world irf
    pts_world_homo = transform.dot(xyz_homo.T).T
    pts_world = pts_world_homo[:, :3] / pts_world_homo[:, 3, None]
    pts_world = pts_world.reshape(N, 3, 1)

    # Calculate the N Jaccobians (N, 3, 3)
    # NOTE Jaccobian matches the one on Wolfram Alpha
    J = np.zeros((N, 3, 3), dtype=float)
    J[:, 0, 0] = np.cos(elev) * np.cos(azi)
    J[:, 1, 0] = np.cos(elev) * np.sin(azi)
    J[:, 2, 0] = np.sin(elev)
    J[:, 0, 1] = -rho * np.cos(elev) * np.sin(azi)
    J[:, 1, 1] = rho * np.cos(elev) * np.cos(azi)
    J[:, 2, 1] = 0
    J[:, 0, 2] = -rho * np.sin(elev) * np.cos(azi)
    J[:, 1, 2] = -rho * np.sin(elev) * np.sin(azi)
    J[:, 2, 2] = rho * np.cos(elev)

    # Calculate J*cov*J^T
    # J*cov
    covs = np.einsum("aij,jk->aik", J, cov)
    # (J*cov)*J^T
    covs = np.einsum("aij,akj->aik", covs, J)

    return pts_world, covs


def fuse_regions(grid):
    # Remove old factors from c++ pgm before adding more
    pyemdw.clear()

    for path in grid.get_all_paths():
        surfel = grid.get_surfel(path)

        h, K, m, C = surfel.get_pgm_factor()
        vars = surfel.corner_ids[:3]
        pyemdw.update_graph(m, C, vars)

    pyemdw.calibrate(0, 1e-11, 1000000)

    for path in grid.get_all_paths():
        surfel = grid.get_surfel(path)
        vars = surfel.corner_ids[:3]
        m_calib, C_calib, h_calib, K_calib = pyemdw.get_cluster_belief(vars)
        # Remove priors before updating
        # h_calib -= h_prior
        # K_calib -= K_prior
        surfel.update_belief(h_calib, K_calib)
