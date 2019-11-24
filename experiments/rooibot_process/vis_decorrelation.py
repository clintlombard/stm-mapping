# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Clint Lombard
"""

import ast
import itertools
import random
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import scipy
import seaborn as sns
import yaml

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy.spatial import Delaunay
from tqdm import *

from stmmap.decouple import Decouple
from stmmap.relative_submap import SensorType
from stmmap.unscented import Unscented
from stmmap.utils.helpers import cov_corrcoef
from stmmap.utils.plot_utils import HandlerEllipse, PlotConfig, plotEllipse
from stmmap.utils.read import readLidar

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Need this to read OpenCV yaml files...
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)


def cov_corrcoef(C):
    stds = np.sqrt(np.diag(C))
    corr_coeffs = np.ones_like(C)
    N = C.shape[0]
    for i in range(N):
        for j in range(N):
            corr_coeffs[i, j] = C[i, j] / (stds[i] * stds[j])
    return corr_coeffs


usage = "python3 decouple_main.py /path/to/dataset/ [-no_check -h/--help]"

check_triangulation = True
for i in range(1, len(sys.argv)):
    if sys.argv[i] == "-no_check":
        check_triangulation = False
    elif sys.argv[i] == "-h" or sys.argv[i] == "--help":
        print(usage)
        exit()

main_dir = str(sys.argv[1])
slam_dir = main_dir + "/slam/"
img_dir = main_dir + "/Cameras/"
disp_dir = main_dir + "/Cameras/disparity/"
ldr_dir = main_dir + "/Lidar/"

# XXX: Global variables
# Robot state vector size (7 for quaternions, 6 for euler)
dof = 7
# TODO: Tune these
sensor_covs = {}
std_u = std_v = 1
std_d = 2
cov_stereo = np.diag([std_u ** 2, std_v ** 2, std_d ** 2])
sensor_covs[SensorType.STEREO] = cov_stereo
std_r = 0.01
std_angles = 15e-3  # Lidar beam divergence = 15 mrad (from datasheet)
cov_ldr = np.diag([std_r ** 2, std_angles ** 2, std_angles ** 2])
sensor_covs[SensorType.LIDAR] = cov_ldr

# Maximum distance (metres) to use from stereo measurements
max_dist = 100
grid_depth = 5

# Set the noise seed for both random and numpy
seed = (int)(np.random.rand() * (2.0 ** 30))
# seed = 647180245  # 18h39 N_slam*0.4
seed = 758094008  # N_slam*0.4, 26May2019/stationart_heap01/26May2019-15h58

np.random.seed(seed)
random.seed(seed)
print("Seed", seed)

# XXX: Triangulate the landmarks to define the different regions.
# * Read in the last SLAM state and use the final estimates of the landmarks to
#   determine the triangulations.
# * Add the option to exclude some landmarks.
# * Also check if the estimates are valid, i.e. the estimate mean isn't
#   [0, 0, 0].
print("Triangulating landmarks")
slam_ts = []
with open(slam_dir + "timestamps.txt", "r") as stream:
    for line in stream.readlines():
        split = line.split()
        # stereo/lidar slam_index meas_index timestamp
        slam_ts.append([split[0], split[1], int(split[2]), float(split[3])])

# Isolate the landmarks from mean, and remove invalid estimates
# NOTE: It is assumed that any values equal to exactly 0 are invalid,
# and there is no check to make sure that there are 3 sequential 0s before
# removing a landmark. But this should be fine.
last_estimate = slam_ts[-1]
filename = "mean-" + last_estimate[1] + ".npy"
last_mean = np.load(slam_dir + filename)
last_lms = last_mean[dof:]
N_lm = int((last_mean.size - dof) / 3)
lm_ids = np.arange(N_lm)

# Find only the landmarks which were seen
valid_lms = []
for i in range(N_lm):
    if last_lms[3 * i] != 0:
        valid_lms.append(i)

valid_lms_arr = np.array(valid_lms)
print("Valid landmarks:", valid_lms)

last_lms = last_lms.reshape((int(last_lms.size / 3), 3))
last_lms = last_lms[:, :2]
notHappy = True

# plt.ion()
# plt.gca().invert_yaxis()
# plt.axis('equal')
while notHappy:
    last_lms_valid = last_lms[valid_lms, :]  # Remove invalids
    if (last_lms_valid.size / 2) < 3:
        print("Too little triangles to (< 3).")
        exit()
    if (last_lms_valid.size / 2) == 3:
        # If there are only 3 landmarks don't need user input
        simplices = np.array([valid_lms])
        notHappy = False
        break
    else:
        triangles = Delaunay(last_lms_valid, incremental=True)
        simplices = triangles.simplices  # Indices of the points in each triangulation
        # Remap simplices to valid landmark ids
        remap = lambda x: valid_lms_arr[x]
        simplices = np.apply_along_axis(remap, 0, simplices)

    # Visual check for triangulation
    plt.gca().clear()
    plt.triplot(last_lms[:, 0], last_lms[:, 1], simplices.copy())
    for i in valid_lms:
        plt.text(*last_lms[i, :], s=str(i))
    plt.draw()
    plt.pause(0.01)

    if check_triangulation:
        remove_str = input("Enter the IDs of landmarks to be removed: ")
        try:
            remove = ast.literal_eval(remove_str)
        except Exception as e:
            print("Error understanding input:", e)
            remove = ()
            notHappy = False

        # If only one number entered
        if type(remove) is int:
            remove = (remove,)
        new_valid = sorted(list(set(valid_lms) - set(remove)))
        valid_lms = new_valid
        valid_lms_arr = np.array(valid_lms)
    else:
        break
plt.close("all")
plt.ioff()

# Calculate neighbouring triangle dictionary
G = nx.Graph()

for s in simplices:
    G.add_nodes_from(s)
    G.add_cycle(s)

# Must make a copy into a list
cliques = nx.find_cliques(G)
clique_list = []
for c in cliques:
    clique_list.append(tuple(sorted(c)))
N_ltrs = len(clique_list)

neighbour_dict = dict()
for c in clique_list:
    neigh_cliques = []
    for edge in itertools.combinations(c, 2):
        neigh = nx.common_neighbors(G, *edge)
        for i in neigh:
            if i not in c:
                # TODO : find which clique it is in
                for c_tmp in clique_list:
                    if i in c_tmp and (len(set(c) & set(c_tmp)) == 2):
                        neigh_cliques.append(c_tmp)
    if len(neigh_cliques) > 0:
        region_a = tuple()
        region_b = tuple()
        region_ab = tuple()
        for c_tmp in neigh_cliques:
            # check subsets
            set_c_tmp = set(c_tmp)
            if set(c[:2]) <= set_c_tmp:
                region_a = tuple(c_tmp)
            if set(c[::2]) <= set_c_tmp:
                region_b = tuple(c_tmp)
            if set(c[1:]) <= set_c_tmp:
                region_ab = tuple(c_tmp)
        neighbour_dict[tuple(c)] = [region_a, region_b, region_ab]
single_ltr = neighbour_dict == {}


def to_xyz(u, v, d, params):
    f, B, u0, v0 = params
    x = B * (u - u0) / d
    y = B * (v - v0) / d
    z = f * B / d
    return np.array([x, y, z])


def to_xyz_Q(uvd_homo, Q):
    xyz_homo = Q.dot(uvd_homo)
    xyz = xyz_homo[:3, :] / xyz_homo[3, :]
    return xyz


def rigid_transform(pose):
    translate = pose[:3].reshape(3, 1)
    if pose[3:].size != 4:
        print("Orientation must be a quaternion!")
        exit()
    q0 = pose[3:].reshape(4, 1)
    R = np.array(
        [
            [
                q0[0, 0] ** 2 - q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                2 * (q0[0, 0] * q0[1, 0] - q0[2, 0] * q0[3, 0]),
                2 * (q0[0, 0] * q0[2, 0] + q0[1, 0] * q0[3, 0]),
            ],
            [
                2 * (q0[0, 0] * q0[1, 0] + q0[2, 0] * q0[3, 0]),
                -q0[0, 0] ** 2 + q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                2 * (q0[1, 0] * q0[2, 0] - q0[0, 0] * q0[3, 0]),
            ],
            [
                2 * (q0[0, 0] * q0[2, 0] - q0[1, 0] * q0[3, 0]),
                2 * (q0[1, 0] * q0[2, 0] + q0[0, 0] * q0[3, 0]),
                -q0[0, 0] ** 2 - q0[1, 0] ** 2 + q0[2, 0] ** 2 + q0[3, 0] ** 2,
            ],
        ]
    )
    return (R, translate)


disp_ts = []
sensor_extrinsics = {}
try:
    with open(disp_dir + "timestamps.txt", "r") as stream:
        for line in stream.readlines():
            split = line.split()
            disp_ts.append(split[0])  # Only take filename

    # Extrinsic parameters from Q matrix
    with open(img_dir + "extrinsics.yml", "r") as stream:
        # NOTE: Remove first line from file because opencv's yaml standard is outdated...
        lines = stream.readlines()[1:]
        long_string = ""
        for line in lines:
            long_string += line
        try:
            data = yaml.load(long_string, Loader=yaml.Loader)
            Q = data["Q"]

            f = 1 * Q[2, 3]
            B = 1 / Q[3, 2]
            # u0 = -1 * Q[0, 3]
            # v0 = -1 * Q[1, 3]
            # d_off = Q[3, 3] / Q[3, 2]
            # params = (f, B, u0, v0)

            valid_roi1 = data["Valid Roi 1"]
            valid_roi2 = data["Valid Roi 2"]

            # NOTE: Setting v_min to help prevent speckles from the disparities
            # Value chosen experimentally
            # v_min = max(valid_roi1[1], valid_roi2[1])
            v_min = 200
            v_max = min(valid_roi1[1] + valid_roi1[3], valid_roi2[1] + valid_roi1[3])

            u_min = max(valid_roi1[0], valid_roi2[0])
            u_max = min(valid_roi1[0] + valid_roi1[2], valid_roi2[0] + valid_roi2[2])
        except yaml.YAMLError as exc:
            print("Error reading OpenCV camera calibration from file: ", filename)
            print(exc)
            exit()

    with open(main_dir + "/cam_rbt_extrinsics.yml", "r") as stream:
        data = yaml.load(long_string, Loader=yaml.Loader)
        R_ext_stereo = np.array(data["R"])
    t_ext_stereo = np.array([[0.0, 0.0, 0.0]]).T

    sensor_extrinsics[SensorType.STEREO] = (R_ext_stereo, t_ext_stereo)
except Exception as e:
    print("Error reading in stereo data, probably missing.")
    print(e)
    exit()

ldr_ts = []
try:
    with open(ldr_dir + "timestamps.txt", "r") as stream:
        for line in stream.readlines():
            split = line.split()
            ldr_ts.append([split[0], float(split[1])])
    with open(ldr_dir + "lidar_extrinsics.yml") as stream:
        RT_ext_ldr = np.array(yaml.load(stream)["RT"])

    # Convert to cam -> lidar transform
    R = RT_ext_ldr[:3, :3]
    t = RT_ext_ldr[:3, 3].reshape((3, 1))
    # R = R.T
    # t = R.dot(-t)

    R_ext_ldr = R_ext_stereo.dot(R)
    t_ext_ldr = R_ext_stereo.dot(t) + t_ext_stereo

    sensor_extrinsics[SensorType.LIDAR] = (R_ext_ldr, t_ext_ldr)
except:
    print("Error reading in lidar data, probably missing.")

decouple = Decouple(19)

plot_skip = 0
step_ldr = -10
step_stereo = 1
step_update_stereo = 15
step_update_ldr = 250
count_update_ldr = 0
count_update_stereo = 0
count_ldr = 0
count_stereo = 0
update_flag = True
N_slam = len(slam_ts)
print("N_slam", N_slam)
for index in tqdm(range(int(N_slam * 0.4), N_slam)):
    # for index in tqdm(range(70, N_slam)):
    # Read in slam estimate
    filename_mean = "mean-" + slam_ts[index][1] + ".npy"
    filename_cov = "cov-" + slam_ts[index][1] + ".npy"
    if slam_ts[index][0] == "stereo":
        sensor_type = SensorType.STEREO
    elif slam_ts[index][0] == "lidar":
        sensor_type = SensorType.LIDAR
    else:
        print("Error: Invalid sensor parsed: ", slam_ts[index][0])
        exit()

    meas_index = slam_ts[index][2]
    slam_mean = np.load(slam_dir + filename_mean)
    slam_cov = np.load(slam_dir + filename_cov)

    # Use the sum of the distances to each lm as a metric for choosing the
    # nearest LTR
    tri_dists = []
    rbt_pos = slam_mean[:3]
    for c in clique_list:
        d = 0
        for lm_id in c:
            tmp = dof + 3 * lm_id
            lm = slam_mean[tmp : tmp + 3, :]
            # Check that the ltr has actually been observed
            if (lm == 0).all():
                d = np.inf
                continue
            else:
                d += np.linalg.norm(rbt_pos - lm)
        tri_dists.append(d)
    grid_index = np.argmin(tri_dists)
    tri = tuple(clique_list[grid_index])
    # print("Closest tri:", tri)

    if sensor_type == SensorType.STEREO:
        if count_stereo == step_stereo:
            update_flag = True
            count_stereo = 0
        else:
            count_stereo += 1
            continue
        img = cv2.imread(disp_dir + disp_ts[meas_index], cv2.CV_16UC1)

        rows = np.arange(0, img.shape[0], dtype=float)
        cols = np.arange(0, img.shape[1], dtype=float)

        u, v = np.meshgrid(cols, rows)
        u = u.flatten()
        v = v.flatten()
        d = np.array(img.flatten(), dtype=float)
        # Divide by 16 because SGBM returns int disparities scaled by 16
        d /= 16
        # Remove disparities which would result in a z-distance greater than a max
        # Or any negative disparities
        b = (u > u_min) & (u < u_max)
        b &= (v > v_min) & (v < v_max)
        b &= (d > 0) & (d > (f * B) / max_dist)

        tmp_u = u.flatten()[b]
        uvd_homo = np.ones((4, tmp_u.size), dtype=float)
        uvd_homo[0, :] = tmp_u
        uvd_homo[1, :] = v.flatten()[b]
        uvd_homo[2, :] = d.flatten()[b]

        pts = to_xyz_Q(uvd_homo, Q).T

        # Order points according to robot coordinates
        # axes x,y,z (cam) -> -y,-z,x (cam robot-orientation)
        reorder = [2, 0, 1]
        pts = pts[:, reorder]
        pts[:, 1] *= -1
        pts[:, 2] *= -1

        raw = np.array([u.flatten()[b], v.flatten()[b], d[b]]).T

    if sensor_type == SensorType.LIDAR:
        if count_ldr == step_ldr:
            update_flag = True
            count_ldr = 0
        else:
            count_ldr += 1
            continue
        raw = readLidar(ldr_dir + ldr_ts[meas_index][0], fov=80)  # Spherical data
        if raw.shape[1] != 3:
            print("Lidar point dimensions incorrect:", raw.shape)
            exit()
        # Order points according to robot coordinates
        pts = np.zeros_like(raw)
        pts[:, 0] = raw[:, 0] * np.cos(raw[:, 1])
        pts[:, 1] = raw[:, 0] * np.sin(raw[:, 1])

    if update_flag:
        update_flag = False

        # Slice relevant part of the slam estimates
        indices = [j for j in range(dof)]
        for t in tri:
            for k in range(3):
                indices.append(dof + 3 * t + k)
        indices.sort()

        slam_mean_slice = slam_mean[indices, :]
        slam_cov_slice = slam_cov[indices, :][:, indices]
        # slam_cov_slice[7:,7:] = np.diag(np.diag(slam_cov_slice[7:,7:]))

        corr = cov_corrcoef(slam_cov_slice)
        cov_no_corr = np.divide(slam_cov_slice, corr)  # Remove correleations
        scale = np.ones_like(cov_no_corr)
        # scale *= (8)**2
        # corr[:7, 7:] *= 0.5
        # corr[7:, :7] *= 0.5
        # corr[7:, 7:] *= 0.5
        corr[np.diag_indices_from(corr)] = 1

        scaled_cov = np.multiply(cov_no_corr, scale)
        slam_cov_slice = np.multiply(scaled_cov, corr)

        # corr = cov_corrcoef(slam_cov_slice)
        # sns.heatmap(
        #     corr,
        #     square=True,
        #     annot=False,
        #     annot_kws={'fontsize': 4},
        #     fmt=".2f",
        #     xticklabels=False,
        #     yticklabels=False,
        #     vmin=-1,
        #     vmax=1,
        #     cmap="RdBu",
        #     cbar=True)
        lms = slam_mean_slice[dof:].reshape((3, 3))

        if (lms == 0).any():
            print("Trying to update an unobserved LTR %s" % (str(tri)))
            exit()

        R_rbt, t_rbt = rigid_transform(slam_mean_slice[:dof])
        R_ext = sensor_extrinsics[sensor_type][0]
        t_ext = sensor_extrinsics[sensor_type][1]

        # Transform from robot to sensor coordinates (applied to landmarks)
        R_sens = R_ext.T.dot(R_rbt.T)
        t_sens = R_ext.T.dot(R_rbt.T.dot(-t_rbt) - t_ext)

        transform = (R_sens, t_sens)

        pts_rel, a0, a1, a2, a3 = decouple.test_points(pts, lms, transform)
        M = np.sum(a0)
        n_meas = 5

        if sensor_type == SensorType.STEREO:
            print("stereo")
            cov_z = cov_stereo
        else:
            print("ldr")
            cov_z = cov_ldr

        if M > n_meas:
            choices = random.sample(range(M), n_meas)

            measurements = raw[a0][choices, :].reshape(3 * n_meas, 1)
            mean = np.vstack((slam_mean_slice, measurements))
            cov = scipy.linalg.block_diag(slam_cov_slice, cov_z)
            for i in range(n_meas - 1):
                cov = scipy.linalg.block_diag(cov, cov_z)

            UT = Unscented(7 + 9 + 3 * n_meas)
            sig = UT.calcSigma(mean, cov)
            print("sigma points", sig.shape)
            n_sig = sig.shape[1]

            # Transform LM sigma pts to sensor frame
            l0 = sig[7:10, :].T
            la = sig[10:13, :].T
            lb = sig[13:16, :].T

            da = la - l0
            db = lb - l0
            tmp = da[:, 1] * db[:, 0] - da[:, 0] * db[:, 1]

            n = np.cross(da, db).T
            n /= np.linalg.norm(n, axis=0)
            # Only using means of landmarks
            l0_mean = mean[7:10, :].T
            la_mean = mean[10:13, :].T
            lb_mean = mean[13:16, :].T

            da_mean = la_mean - l0_mean
            db_mean = lb_mean - l0_mean
            tmp_mean = da_mean[:, 1] * db_mean[:, 0]
            tmp_mean -= da_mean[:, 0] * db_mean[:, 1]

            n_mean = np.cross(da_mean, db_mean).T
            n_mean /= np.linalg.norm(n_mean, axis=0)

            sig_inert = np.copy(sig)
            sig_rel = np.copy(sig)
            for i in range(n_meas):
                start_index = 7 + 9 + 3 * i
                stop_index = start_index + 3
                if sensor_type == SensorType.STEREO:
                    # Transform from image to cartesian coordinates (in the sensor frame)
                    uvd = sig[start_index:stop_index, :]
                    uvd_homo = np.insert(uvd, 3, 1.0, axis=0)
                    xyz_homo = Q.dot(uvd_homo)
                    xyz = xyz_homo[:3, :] / xyz_homo[3, :]

                    reorder = [2, 0, 1]
                    sig_sensor = xyz[reorder, :]
                    sig_sensor[1, :] *= -1
                    sig_sensor[2, :] *= -1
                elif sensor_type == SensorType.LIDAR:
                    spherical_pts = sig[start_index:stop_index, :]
                    sig_sensor = np.zeros_like(spherical_pts)
                    sig_sensor[0, :] = spherical_pts[0, :] * np.cos(spherical_pts[1, :])
                    sig_sensor[1, :] = spherical_pts[0, :] * np.sin(spherical_pts[1, :])
                else:
                    raise ValueError("Invalid sensor type specified")

                sig_robot = R_ext.dot(sig_sensor) + t_ext
                # Transform from sensor to world
                sig_world = np.copy(sig_sensor)
                for j, s in enumerate(sig.T):
                    t = s[:3]
                    q0 = s[3:7].reshape(4, 1)
                    R = np.array(
                        [
                            [
                                q0[0, 0] ** 2 - q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                                2 * (q0[0, 0] * q0[1, 0] - q0[2, 0] * q0[3, 0]),
                                2 * (q0[0, 0] * q0[2, 0] + q0[1, 0] * q0[3, 0]),
                            ],
                            [
                                2 * (q0[0, 0] * q0[1, 0] + q0[2, 0] * q0[3, 0]),
                                -q0[0, 0] ** 2 + q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                                2 * (q0[1, 0] * q0[2, 0] - q0[0, 0] * q0[3, 0]),
                            ],
                            [
                                2 * (q0[0, 0] * q0[2, 0] - q0[1, 0] * q0[3, 0]),
                                2 * (q0[1, 0] * q0[2, 0] + q0[0, 0] * q0[3, 0]),
                                -q0[0, 0] ** 2 - q0[1, 0] ** 2 + q0[2, 0] ** 2 + q0[3, 0] ** 2,
                            ],
                        ]
                    )
                    sig_world[:, j] = R.dot(sig_robot[:, j]) + t
                # Transform to relative coordinates
                sig_l0 = sig_world - l0.T  # Shift origin to l0

                # Fancy way of doing dot product using Einstein Summation
                h = np.einsum("ij,ij->i", sig_l0.T, n.T)

                sig_proj = (sig_l0 - (h * n)).T
                alpha = sig_proj[:, 1] * db[:, 0] - sig_proj[:, 0] * db[:, 1]
                beta = sig_proj[:, 0] * da[:, 1] - sig_proj[:, 1] * da[:, 0]
                alpha /= tmp
                beta /= tmp

                sig_rel_only = np.empty_like(sig_l0)
                sig_rel_only[0, :] = alpha
                sig_rel_only[1, :] = beta
                sig_rel_only[2, :] = h.T
                sig_rel[start_index:stop_index, :] = np.copy(sig_rel_only)
                sig_inert[start_index:stop_index, :] = np.copy(sig_world)

            mean_inert, cov_inert = UT.calcMoments(sig_inert)
            mean_rel, cov_rel = UT.calcMoments(sig_rel)

            # Calculate the measurement uncertainty in relative coordinates but
            # ignoring the landmark uncertainties. This is the only way to
            # compare inertial and relative in a similar IRF.

            mean = np.vstack((slam_mean_slice[:7], measurements))
            cov = scipy.linalg.block_diag(slam_cov_slice[:7, :7], cov_z)
            for i in range(n_meas - 1):
                cov = scipy.linalg.block_diag(cov, cov_z)

            UT = Unscented(7 + 3 * n_meas)
            # mean = measurements
            # cov = cov_z
            # for i in range(n_meas - 1):
            #     cov = scipy.linalg.block_diag(cov, cov_z)
            # UT = unscented(3 * n_meas)
            sig = UT.calcSigma(mean, cov)
            print("sigma points", sig.shape)
            n_sig = sig.shape[1]

            # Only using means of landmarks
            l0 = slam_mean_slice[7:10, :].T
            la = slam_mean_slice[10:13, :].T
            lb = slam_mean_slice[13:16, :].T

            da = la - l0
            db = lb - l0
            tmp = da[:, 1] * db[:, 0] - da[:, 0] * db[:, 1]

            n = np.cross(da, db).T
            n /= np.linalg.norm(n, axis=0)

            sig_inert_rel = np.copy(sig)
            for i in range(n_meas):
                start_index = 7 + 3 * i
                stop_index = start_index + 3
                if sensor_type == SensorType.STEREO:
                    # Transform from image to cartesian coordinates (in the sensor frame)
                    uvd = sig[start_index:stop_index, :]
                    uvd_homo = np.insert(uvd, 3, 1.0, axis=0)
                    xyz_homo = Q.dot(uvd_homo)
                    xyz = xyz_homo[:3, :] / xyz_homo[3, :]

                    reorder = [2, 0, 1]
                    sig_sensor = xyz[reorder, :]
                    sig_sensor[1, :] *= -1
                    sig_sensor[2, :] *= -1
                elif sensor_type == SensorType.LIDAR:
                    spherical_pts = sig[start_index:stop_index, :]
                    sig_sensor = np.zeros_like(spherical_pts)
                    sig_sensor[0, :] = spherical_pts[0, :] * np.cos(spherical_pts[1, :])
                    sig_sensor[1, :] = spherical_pts[0, :] * np.sin(spherical_pts[1, :])
                else:
                    raise ValueError("Invalid sensor type specified")

                sig_robot = R_ext.dot(sig_sensor) + t_ext
                # Transform from sensor to world
                sig_world = np.copy(sig_sensor)
                # t = slam_mean_slice[:3]
                # q0 = slam_mean_slice[3:7].reshape(4, 1)
                for j, s in enumerate(sig.T):
                    t = s[:3]
                    q0 = s[3:7].reshape(4, 1)
                    R = np.array(
                        [
                            [
                                q0[0, 0] ** 2 - q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                                2 * (q0[0, 0] * q0[1, 0] - q0[2, 0] * q0[3, 0]),
                                2 * (q0[0, 0] * q0[2, 0] + q0[1, 0] * q0[3, 0]),
                            ],
                            [
                                2 * (q0[0, 0] * q0[1, 0] + q0[2, 0] * q0[3, 0]),
                                -q0[0, 0] ** 2 + q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                                2 * (q0[1, 0] * q0[2, 0] - q0[0, 0] * q0[3, 0]),
                            ],
                            [
                                2 * (q0[0, 0] * q0[2, 0] - q0[1, 0] * q0[3, 0]),
                                2 * (q0[1, 0] * q0[2, 0] + q0[0, 0] * q0[3, 0]),
                                -q0[0, 0] ** 2 - q0[1, 0] ** 2 + q0[2, 0] ** 2 + q0[3, 0] ** 2,
                            ],
                        ]
                    )
                    sig_world[:, j] = R.dot(sig_robot[:, j]) + t
                # Transform to relative coordinates
                sig_l0 = sig_world - l0.T  # Shift origin to l0

                # Fancy way of doing dot product using Einstein Summation
                h = np.einsum("ij,ij->i", sig_l0.T, n.T)

                sig_proj = (sig_l0 - (h * n)).T
                alpha = sig_proj[:, 1] * db[:, 0] - sig_proj[:, 0] * db[:, 1]
                beta = sig_proj[:, 0] * da[:, 1] - sig_proj[:, 1] * da[:, 0]
                alpha /= tmp
                beta /= tmp

                sig_rel_only = np.empty_like(sig_l0)
                sig_rel_only[0, :] = alpha
                sig_rel_only[1, :] = beta
                sig_rel_only[2, :] = h.T
                sig_inert_rel[start_index:stop_index, :] = np.copy(sig_rel_only)

            mean_inert_rel, cov_inert_rel = UT.calcMoments(sig_inert_rel)

            # Ignoring both robot and lm uncertainty ---------------------------

            mean = measurements
            cov = 1 * cov_z
            for i in range(n_meas - 1):
                cov = scipy.linalg.block_diag(cov, cov_z)

            UT = Unscented(3 * n_meas)
            # mean = measurements
            # cov = cov_z
            # for i in range(n_meas - 1):
            #     cov = scipy.linalg.block_diag(cov, cov_z)
            # UT = unscented(3 * n_meas)
            sig = UT.calcSigma(mean, cov)
            print("sigma points", sig.shape)
            n_sig = sig.shape[1]

            # Only using means of landmarks
            l0 = slam_mean_slice[7:10, :].T
            la = slam_mean_slice[10:13, :].T
            lb = slam_mean_slice[13:16, :].T

            da = la - l0
            db = lb - l0
            tmp = da[:, 1] * db[:, 0] - da[:, 0] * db[:, 1]

            n = np.cross(da, db).T
            n /= np.linalg.norm(n, axis=0)

            sig_nouncert_rel = np.copy(sig)
            sig_nouncert_inert = np.copy(sig)
            for i in range(n_meas):
                start_index = 3 * i
                stop_index = start_index + 3
                if sensor_type == SensorType.STEREO:
                    # Transform from image to cartesian coordinates (in the sensor frame)
                    uvd = sig[start_index:stop_index, :]
                    uvd_homo = np.insert(uvd, 3, 1.0, axis=0)
                    xyz_homo = Q.dot(uvd_homo)
                    xyz = xyz_homo[:3, :] / xyz_homo[3, :]

                    reorder = [2, 0, 1]
                    sig_sensor = xyz[reorder, :]
                    sig_sensor[1, :] *= -1
                    sig_sensor[2, :] *= -1
                elif sensor_type == SensorType.LIDAR:
                    spherical_pts = sig[start_index:stop_index, :]
                    sig_sensor = np.zeros_like(spherical_pts)
                    sig_sensor[0, :] = spherical_pts[0, :] * np.cos(spherical_pts[1, :])
                    sig_sensor[1, :] = spherical_pts[0, :] * np.sin(spherical_pts[1, :])
                else:
                    raise ValueError("Invalid sensor type specified")

                sig_robot = R_ext.dot(sig_sensor) + t_ext
                # Transform from sensor to world
                sig_world = np.copy(sig_sensor)
                t = slam_mean_slice[:3]
                q0 = slam_mean_slice[3:7].reshape(4, 1)
                R = np.array(
                    [
                        [
                            q0[0, 0] ** 2 - q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                            2 * (q0[0, 0] * q0[1, 0] - q0[2, 0] * q0[3, 0]),
                            2 * (q0[0, 0] * q0[2, 0] + q0[1, 0] * q0[3, 0]),
                        ],
                        [
                            2 * (q0[0, 0] * q0[1, 0] + q0[2, 0] * q0[3, 0]),
                            -q0[0, 0] ** 2 + q0[1, 0] ** 2 - q0[2, 0] ** 2 + q0[3, 0] ** 2,
                            2 * (q0[1, 0] * q0[2, 0] - q0[0, 0] * q0[3, 0]),
                        ],
                        [
                            2 * (q0[0, 0] * q0[2, 0] - q0[1, 0] * q0[3, 0]),
                            2 * (q0[1, 0] * q0[2, 0] + q0[0, 0] * q0[3, 0]),
                            -q0[0, 0] ** 2 - q0[1, 0] ** 2 + q0[2, 0] ** 2 + q0[3, 0] ** 2,
                        ],
                    ]
                )
                sig_world = R.dot(sig_robot) + t
                # Transform to relative coordinates
                sig_l0 = sig_world - l0.T  # Shift origin to l0

                # Fancy way of doing dot product using Einstein Summation
                h = np.einsum("ij,ij->i", sig_l0.T, n.T)

                sig_proj = (sig_l0 - (h * n)).T
                alpha = sig_proj[:, 1] * db[:, 0] - sig_proj[:, 0] * db[:, 1]
                beta = sig_proj[:, 0] * da[:, 1] - sig_proj[:, 1] * da[:, 0]
                alpha /= tmp
                beta /= tmp

                sig_nouncert_rel_only = np.empty_like(sig_l0)
                sig_nouncert_rel_only[0, :] = alpha
                sig_nouncert_rel_only[1, :] = beta
                sig_nouncert_rel_only[2, :] = h.T
                sig_nouncert_rel[start_index:stop_index, :] = np.copy(sig_nouncert_rel_only)
                sig_nouncert_inert[start_index:stop_index, :] = np.copy(sig_world)

            mean_nouncert_rel, cov_nouncert_rel = UT.calcMoments(sig_nouncert_rel)
            mean_nouncert_inert, cov_nouncert_inert = UT.calcMoments(sig_nouncert_inert)

            # Relative coordinates vs inertial visualisation -------------------

            # fig, ax = plt.subplots(
            #     1,
            #     1,
            #     # sharex=True,
            #     # sharey=True,
            #     figsize=(4, 3),
            #     dpi=150,
            #     constrained_layout=True)

            # mean = slam_mean_slice[:2].flatten()
            # cov = slam_cov_slice[:2, :2]
            # ax.scatter(*mean, color='k', marker='*', s=1.5)
            # plotEllipse(mean, cov, nstd=1, ax=ax, ec='k', fc='none')

            # for i in range(3):
            #     start = 7 + 3 * i
            #     stop = start + 2
            #     mean = mean_inert[start:stop].flatten()
            #     cov = cov_inert[start:stop, start:stop]
            #     ax.scatter(*mean, color='k', marker='*', s=1.5)
            #     plotEllipse(mean, cov, nstd=1, ax=ax, ec='k', fc='none')

            # for i in range(n_meas):
            #     start = 7 + 3 * 3 + 3 * i
            #     stop = start + 2
            #     mean = mean_inert[start:stop].flatten()
            #     cov = cov_inert[start:stop, start:stop]
            #     ax.scatter(*mean, color='r', marker='.', s=1.5)
            #     plotEllipse(mean, cov, nstd=1, ax=ax, ec='r', fc='none')

            #     start = 3 * i
            #     stop = start + 2
            #     mean = mean_nouncert_inert[start:stop].flatten()
            #     cov = cov_nouncert_inert[start:stop, start:stop]
            #     ax.scatter(*mean, color='b', marker='.', s=1.5)
            #     plotEllipse(mean, cov, nstd=1, ax=ax, ec='b', fc='none')

            #     ax[1].scatter(*mean, color='r', marker='.', s=1.5)
            #     plotEllipse(
            #         mean,
            #         cov,
            #         nstd=1,
            #         ax=ax[1],
            #         ec='r',
            #         fc='none',
            #         label="Exact")

            scale = 0.32
            ratio = 1.0 / 1.0
            fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size, constrained_layout=True)
            ax.tick_params(axis="both", labelsize=8)

            # palette = sns.color_palette('tab10', 3)
            palette = ["C1", "C2", "C0"]

            axins = zoomed_inset_axes(ax, 4, loc=1)
            axins.set_visible(True)
            axins.set_yticklabels([])
            axins.set_xticklabels([])
            axins.tick_params(direction="in")
            axins.set_aspect("equal")
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

            # full uncertainty
            mean = mean_rel[16:18].flatten()
            cov = cov_rel[16:18, 16:18]
            ax.scatter(*mean, color=palette[0], marker=".", s=1.5)
            e1 = plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[0], fc="none")

            axins.scatter(*mean, color=palette[0], marker=".", s=1.5)
            plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[0], fc="none")
            for i in range(1, n_meas):
                start = 7 + 3 * 3 + 3 * i
                stop = start + 2
                mean = mean_rel[start:stop].flatten()
                cov = cov_rel[start:stop, start:stop]
                ax.scatter(*mean, color=palette[0], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[0], fc="none")

                axins.scatter(*mean, color=palette[0], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[0], fc="none")

            # perfectly know landmarks
            mean = mean_inert_rel[7:9].flatten()
            cov = cov_inert_rel[7:9, 7:9]
            ax.scatter(*mean, color=palette[1], marker=".", s=1.5)
            e2 = plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[1], fc="none")
            axins.scatter(*mean, color=palette[1], marker=".", s=1.5)
            plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[1], fc="none")
            for i in range(1, n_meas):
                start = 7 + 3 * i
                stop = start + 2
                mean = mean_inert_rel[start:stop].flatten()
                cov = cov_inert_rel[start:stop, start:stop]
                ax.scatter(*mean, color=palette[1], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[1], fc="none")

                axins.scatter(*mean, color=palette[1], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[1], fc="none")

            # perfectly know landmarks and pose
            mean = mean_nouncert_rel[:2].flatten()
            cov = cov_nouncert_rel[:2, :2]
            ax.scatter(*mean, color=palette[2], marker=".", s=1.5)
            e3 = plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[2], fc="none")
            axins.scatter(*mean, color=palette[2], marker=".", s=1.5)
            plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[2], fc="none")
            for i in range(1, n_meas):
                start = 3 * i
                stop = start + 2
                mean = mean_nouncert_rel[start:stop].flatten()
                cov = cov_nouncert_rel[start:stop, start:stop]
                ax.scatter(*mean, color=palette[2], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=ax, ec=palette[2], fc="none")

                axins.scatter(*mean, color=palette[2], marker=".", s=1.5)
                plotEllipse(mean, cov, nstd=1, ax=axins, ec=palette[2], fc="none")

            xbounds = [0, 0]
            ybounds = [0, 0]

            ax.legend(
                [e1, e2, e3],
                [
                    r"$\mathcal{B}(\mathit{M}^R)$",
                    r"$\mathcal{B}_{\text{global}}(\mathit{M}^R)$",
                    r"$\mathcal{B}_{\text{ideal}}(\mathit{M}^R)$",
                ],
                handler_map={mpatches.Ellipse: HandlerEllipse()},
                loc="lower right",
                ncol=1,
                fontsize=10,
            )

            def on_press(event):
                if event.inaxes == ax:
                    xbounds[0] = event.xdata
                    ybounds[0] = event.ydata

            def on_release(event):
                if event.inaxes == ax:
                    xbounds[1] = event.xdata
                    ybounds[1] = event.ydata
                    axins.set_xlim(*xbounds)
                    axins.set_ylim(*ybounds)

                    plt.draw()

            cid = fig.canvas.mpl_connect("button_press_event", on_press)
            cid = fig.canvas.mpl_connect("button_release_event", on_release)
            # ax.legend(loc=2, ncol=1, fontsize=10)

            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\beta$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

            # plt.pause(-1)
            plt.show()
            # plt.close('all')
            # exit()

            # Correlation heatmap ----------------------------------------------
            # fig, ax = plt.subplots(
            #     1,
            #     2,
            #     sharex=True,
            #     sharey=True,
            #     figsize=(7, 3),
            #     dpi=150,
            #     constrained_layout=True)
            ax = []
            scale = 0.32
            ratio = 1.0 / 1.0
            fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
            print(fig_size)
            fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=fig_size, constrained_layout=True)
            ax.append(ax1)
            print(fig_size)
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=fig_size, constrained_layout=True)
            ax.append(ax2)

            # Calculate moments
            corr = cov_corrcoef(cov_inert)
            corr = np.abs(corr)
            # c_map = "Greys"
            # c_map = "viridis_r"
            # c_map = "magma_r"
            c_map = "Blues"
            v_min = 0
            v_max = 1
            sns.heatmap(
                corr,
                square=True,
                annot=False,
                annot_kws={"fontsize": 4},
                fmt=".2f",
                xticklabels=False,
                yticklabels=False,
                vmin=v_min,
                vmax=v_max,
                cmap=c_map,
                cbar=False,
                ax=ax[0],
            )
            # np.savetxt("inert.csv", corr, delimiter=",")

            # Calculate moments
            corr = cov_corrcoef(cov_rel)
            corr = np.abs(corr)
            heat = sns.heatmap(
                corr,
                square=True,
                annot=False,
                annot_kws={"fontsize": 4},
                fmt=".2f",
                xticklabels=False,
                yticklabels=False,
                vmin=v_min,
                vmax=v_max,
                cmap=c_map,
                cbar=False,
                cbar_kws={"shrink": 0.95},
                ax=ax[1],
            )

            # cbar = ax[1].collections[0].colorbar
            # ticks = np.linspace(-1, 1, 5)
            # cbar.set_ticks(ticks)
            # cbar.ax.tick_params(labelsize=8)
            # cbar.set_label("Correlation Coefficient", size=8)
            # cbar.ax.tick_params(labelsize=8)

            def annotate_group_x(name, span, ax=None):
                """Annotates a span of the x-axis"""

                def annotate(ax, name, left, right, y, pad):
                    arrow = ax.annotate(
                        name,
                        xy=(left, y),
                        xycoords="data",
                        xytext=(right, y - pad),
                        textcoords="data",
                        annotation_clip=False,
                        verticalalignment="center",
                        horizontalalignment="center",
                        linespacing=2.0,
                        arrowprops=dict(
                            arrowstyle="-",
                            shrinkA=1,
                            shrinkB=3,
                            connectionstyle="angle,angleA=180,angleB=90,rad=5",
                        ),
                    )
                    return arrow

                if ax is None:
                    ax = plt.gca()
                ymin = 0  # ax.get_ylim()[0]
                ypad = 0.045 * np.ptp(ax.get_ylim())
                xcenter = np.mean(span)
                left_arrow = annotate(ax, name, span[0], xcenter, ymin, ypad)
                right_arrow = annotate(ax, name, span[1], xcenter, ymin, ypad)
                return left_arrow, right_arrow

            def annotate_group_y(name, span, ax=None):
                """Annotates a span of the x-axis"""

                def annotate(ax, name, left, right, y, pad):
                    arrow = ax.annotate(
                        name,
                        xy=(y, left),
                        xycoords="data",
                        xytext=(y - pad, right),
                        textcoords="data",
                        annotation_clip=False,
                        verticalalignment="center",
                        horizontalalignment="center",
                        linespacing=2.0,
                        arrowprops=dict(
                            arrowstyle="-",
                            shrinkA=3,
                            shrinkB=3,
                            connectionstyle="angle,angleA=90,angleB=0,rad=5",
                        ),
                    )
                    return arrow

                if ax is None:
                    ax = plt.gca()
                ymin = 0  # ax.get_ylim()[0]
                ypad = 0.045 * np.ptp(ax.get_ylim())
                xcenter = np.mean(span)
                left_arrow = annotate(ax, name, span[0], xcenter, ymin, ypad)
                right_arrow = annotate(ax, name, span[1], xcenter, ymin, ypad)
                return left_arrow, right_arrow

            # fig.set_constrained_layout_pads(
            #     w_pad=0.05, bottom=0.5, h_pad=0., hspace=0., wspace=0.)
            # plt.subplots_adjust(
            #     left=0.03,
            #     bottom=0,
            #     right=0.85,
            #     top=0.92,
            #     wspace=0.08,
            #     hspace=0)
            delta = 0.5
            groups = [
                (r"$\bm{x}^{}$", (0 + delta, 7 - delta)),
                (r"$\mathit{L}^{}$", (7 + delta, 7 + 9 - delta)),
                (r"$\mathit{M}^{}$", (7 + delta + 9, 7 + 9 + 3 * n_meas - delta)),
            ]
            for name, span in groups:
                annotate_group_x(name, span, ax=ax[0])
                annotate_group_y(name, span, ax=ax[0])

            groups = [
                (r"$\bm{x}^{}$", (0 + delta, 7 - delta)),
                (r"$\mathit{L}^{}$", (7 + delta, 7 + 9 - delta)),
                (r"$\mathit{M}^R$", (7 + delta + 9, 7 + 9 + 3 * n_meas - delta)),
            ]
            for name, span in groups:
                annotate_group_x(name, span, ax=ax[1])
                annotate_group_y(name, span, ax=ax[1])
            # np.savetxt("rel.csv", corr, delimiter=",")

            fig1.subplots_adjust(left=0.08, bottom=0, right=1, top=0.92, wspace=0, hspace=0)
            fig2.subplots_adjust(left=0.08, bottom=0, right=1, top=0.92, wspace=0, hspace=0)
            fig1.savefig("corr_abs.pdf")
            fig2.savefig("corr_rel.pdf")
            plt.pause(-1)
            plt.show()

            exit()
