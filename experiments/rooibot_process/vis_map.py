# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""Map Visualisation

Visualise the full map in 3-D as well as project it onto images.

Author: Clint Lombard

"""

import argparse
import ast
import os

import cv2
import dill
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sb
import transformations
import yaml

from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from stmmap import Decouple, RelativeSubmap
from stmmap.utils.plot_utils import PlotConfig

try:
    os.environ["ETS_TOOLKIT"] = "wx"
    import mayavi.mlab as mlab
except Exception as e:
    import mayavi.mlab as mlab


try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()


# Need this to read OpenCV yaml files...
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix)

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()


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


def find_slam_from_img(img_index, slam_ts):
    for index, val in enumerate(slam_ts):
        if (val[0] == "stereo") and (val[2] == img_index):
            return index
    print("Error could not find slam from img")
    exit()


def find_img_from_submap(submap_index, slam_ts):
    N = len(slam_ts)
    for i in range(submap_index, N):
        sensor = slam_ts[i][0]
        if sensor == "stereo":
            return i

    print("Couldn't find a stereo measurement in after submap, doing backwards search")

    for i in reversed(range(0, submap_index)):
        sensor = slam_ts[i][0]
        if sensor == "stereo":
            return i
    print("Error could not find img from submap")
    exit()


parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("sensor_type", metavar="STEREO/LIDAR", type=str, default="STEREO")
parser.add_argument("--local", help="Visualise local factors.", action="store_true")
parser.add_argument("--nolight", help="Disable lighting", action="store_true")
parser.add_argument(
    "--points",
    help="Plot measurement points. Note that this will only work if --fusion is usedwhen creating the map.",
    action="store_true",
)

args = parser.parse_args()

main_path = args.path
save_path = os.path.join(main_path, "scenes")
if not os.path.exists(save_path):
    os.makedirs(save_path)

sensor_type = args.sensor_type

# Used global or local model
local_model = args.local

slam_path = main_path + "/slam/"
img_path = main_path + "/Cameras/"
submap_path = ""
fused = True

if sensor_type == "STEREO":
    submap_path = main_path + "/submap/STEREO/"
elif sensor_type == "LIDAR":
    submap_path = main_path + "/submap/LIDAR/"
if submap_path == "":
    print("No sensor selected using STEREO")
    submap_path = main_path + "/submap/STEREO/"

# fused = False

with open(img_path + "intrinsics_14191992.yml", "r") as stream:
    # NOTE: Remove first line from file because opencv's yaml standard is outdated...
    lines = stream.readlines()[1:]
    long_string = ""
    for line in lines:
        long_string += line
    try:
        data = yaml.load(long_string, Loader=yaml.Loader)
        M_l = data["M"]
        D_l = data["D"]
    except yaml.YAMLError as exc:
        print("Error reading OpenCV camera calibration from file: ", filename)
        print(exc)
        exit()

with open(img_path + "intrinsics_14191994.yml", "r") as stream:
    # NOTE: Remove first line from file because opencv's yaml standard is outdated...
    lines = stream.readlines()[1:]
    long_string = ""
    for line in lines:
        long_string += line
    try:
        data = yaml.load(long_string, Loader=yaml.Loader)
        M_r = data["M"]
        D_r = data["D"]
    except yaml.YAMLError as exc:
        print("Error reading OpenCV camera calibration from file: ", filename)
        print(exc)
        exit()

with open(img_path + "extrinsics.yml", "r") as stream:
    # NOTE: Remove first line from file because opencv's yaml standard is outdated...
    lines = stream.readlines()[1:]
    long_string = ""
    for line in lines:
        long_string += line
    try:
        data = yaml.load(long_string, Loader=yaml.Loader)
        R_cam1cam2 = data["R"]
        t_cam1cam2 = data["T"]
        Q = data["Q"]
    except yaml.YAMLError as exc:
        print("Error reading OpenCV camera calibration from file: ", filename)
        print(exc)
        exit()

with open(main_path + "/cam_rbt_extrinsics.yml", "r") as stream:
    data = yaml.load(stream, Loader=yaml.Loader)
    R_cam1rbt = np.array(data["R"])

    # angles = np.deg2rad([-0.28666603863, 13.7945079522, 0])
    # alpha, beta, gamma = angles
    # origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # Rx = transformations.rotation_matrix(alpha, xaxis)
    # Ry = transformations.rotation_matrix(beta, yaxis)
    # Rz = transformations.rotation_matrix(gamma, zaxis)
    # R = transformations.concatenate_matrices(Rx, Ry, Rz)
    # R_cam1rbt = R[:3, :3]

    t_cam1rbt = np.array([[0.0, 0.0, 0.0]]).T

img_ts = []
with open(img_path + "timestamps.txt", "r") as stream:
    for line in stream.readlines():
        split = line.split()
        img_ts.append([split[0], float(split[1])])

slam_ts = []
with open(slam_path + "timestamps.txt", "r") as stream:
    for line in stream.readlines():
        split = line.split()
        slam_ts.append([split[0], split[1], float(split[2])])

folders = [f for f in os.listdir(submap_path) if os.path.isdir(submap_path + f)]

N_folders = len(folders)
if N_folders == 0:
    print("No folders found. I'm out...")
    exit()

if N_folders > 1:
    print("Regions found: ", folders)

submaps = []
ltrs = []
lm_set = set()
for i in range(N_folders):
    folder = submap_path + folders[i] + "/"
    ltr = ast.literal_eval(folders[i])
    print("LTR:", ltr)
    submap_files = []
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith(".p"):
                submap_files.append(file)

    N_submap_files = len(submap_files)
    if N_submap_files == 0:
        print("No submap files found. I'm out...")
        exit()

    submap_files = sorted(submap_files)

    if i == 0:
        if N_submap_files > 1:
            # list_index = N_submap_files - 1
            print("Available: ", dict(zip(range(N_submap_files), submap_files)))
            try:
                list_index = int(input("Choose region from 0 to %d: " % (N_submap_files - 1)))
            except:
                print("Invalid choice. Using region 0.")
                list_index = 0
        else:
            list_index = 0

        end = submap_files[list_index].find(".p")
        index = int(submap_files[list_index][:end])
        print("Index:", index)
        # Load submap
        submap = dill.load(open(folder + submap_files[list_index], "rb"))
        submaps.append(submap)

        ltrs.append(ltr)
        for l in ltr:
            lm_set.add(l)
    else:
        # Find index in other ltrs otherwise ignore
        for j in range(N_submap_files):
            end = submap_files[j].find(".p")
            index_search = int(submap_files[j][:end])
            if index_search == index:
                print("Index:", index_search)
                # Load submap
                submap = dill.load(open(folder + submap_files[j], "rb"))
                submaps.append(submap)

                ltrs.append(ltr)
                for l in ltr:
                    lm_set.add(l)

try:
    img_index = int(input("Which image to load [0, %d]: " % (len(img_ts) / 2 - 1)))
except:
    print("Invalid choice. Using region same as submap.")
    img_index = find_img_from_submap(index, slam_ts)
print(img_index)

slam_index = find_slam_from_img(img_index, slam_ts)

# Read images
if "14191992" in img_ts[img_index][0]:
    img_l = cv2.imread(img_path + img_ts[2 * img_index][0], 0)
    img_r = cv2.imread(img_path + img_ts[2 * img_index + 1][0], 0)
else:
    img_l = cv2.imread(img_path + img_ts[2 * img_index + 1][0], 0)
    img_r = cv2.imread(img_path + img_ts[2 * img_index][0], 0)

col_img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGBA)
col_img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGBA)

# Read mean slam estimate
print("Slam index:", slam_index)
filename_mean = "mean-" + slam_ts[slam_index][1] + ".npy"
slam_mean = np.load(slam_path + filename_mean)

# Sanity check
assert slam_ts[slam_index][2] == img_index

indices = [j for j in range(7)]
for l in lm_set:
    for k in range(3):
        indices.append(7 + 3 * l + k)
indices.sort()
slam_mean = slam_mean[indices, :]
if (slam_mean[7:] == 0).any():
    print("Error: One or more landmarks in the LTR are unobserved.")
    exit()

# Transform lms to sensor frame
N_lms = len(lm_set)
lms = slam_mean[7:].reshape((N_lms, 3))

R_rbt, t_rbt = rigid_transform(slam_mean[:7])

lms_all_sensor = (R_cam1rbt.T.dot(R_rbt.T).dot(lms.T - t_rbt) - R_cam1rbt.T.dot(t_cam1rbt)).T
reorder = [1, 2, 0]

rvec = np.zeros(3)
tvec = np.zeros(3)

# mesh_img_overlay = np.zeros_like(col_img_l)
de = Decouple(19)

# Find min and max for colour mapping
global_max = -np.inf
global_min = np.inf
for i, submap in enumerate(submaps):
    ltr = ltrs[i]
    lms_sensor = lms_all_sensor[ltr, :]
    # for path in submap.get_all_paths():
    for path in submap.keys():
        tri = submap[path]

        m = tri.bel_h_m
        C = tri.bel_h_C

        corners = tri.corners
        # Get the inverse transform
        T_inv = np.linalg.inv(tri.transform)
        offset = tri.offset
        corners_3d = np.vstack((corners, m.T))
        mesh_rel = corners_3d

        mesh_sensor = de.from_relative(mesh_rel.T, lms_sensor)
        dists = np.linalg.norm(mesh_sensor, axis=1)
        # dists = mesh_sensor[:, 2]
        # dists = mesh_sensor[:, 2] * 0 + np.sqrt(tri.global_v)
        local_max = np.max(dists)
        local_min = np.min(dists)

        global_max = np.max([local_max, global_max])
        global_min = np.min([local_min, global_min])
# If using distance not variation uncomment these
tmp_max = global_max
tmp_min = global_min
# tmp_max = global_max
# tmp_min = global_min
global_max = max(tmp_max, tmp_min)
global_min = min(tmp_max, tmp_min)
print(global_min, global_max)

# XXX Project mesh vertices onto images
col_map = cm.ScalarMappable(cmap="viridis")
col_map.set_clim(global_min, global_max)
for i, submap in enumerate(submaps):
    ltr = ltrs[i]
    lms_sensor = lms_all_sensor[ltr, :]
    # for path in submap.get_all_paths():
    for path in submap.keys():
        tri = submap[path]

        m = tri.bel_h_m
        C = tri.bel_h_C

        corners = tri.corners
        corners_3d = np.vstack((corners, m.T))
        mesh_rel = corners_3d
        mesh_sensor_l = de.from_relative(mesh_rel.T, lms_sensor)
        mesh_sensor_l = mesh_sensor_l[:, reorder]
        b = mesh_sensor_l[:, 2] > 1.2
        mesh_sensor_l = mesh_sensor_l[b, :]
        if mesh_sensor_l.size == 0:
            continue

        mesh_sensor_l[:, 0] *= -1
        mesh_sensor_l[:, 1] *= -1
        mesh_sensor_r = (R_cam1cam2.dot(mesh_sensor_l.T) + t_cam1cam2).T

        b = mesh_sensor_r[:, 2] > 1.2
        mesh_sensor_r = mesh_sensor_r[b, :]
        if mesh_sensor_r.size == 0:
            continue

        dists = np.linalg.norm(mesh_sensor_l, axis=1)
        # dists = mesh_sensor_l[:, 1]
        # dists = mesh_sensor_l[:, 1] * 0 + np.sqrt(tri.global_v)

        mesh_img_l, _ = cv2.projectPoints(mesh_sensor_l, rvec, tvec, M_l, D_l)
        mesh_img_r, _ = cv2.projectPoints(mesh_sensor_r, rvec, tvec, M_r, D_r)

        avg_col = np.zeros(4)
        alpha = 1
        for i, pt in enumerate(mesh_img_l):
            pt_tup = tuple(pt.astype(int).flatten())
            col = col_map.to_rgba(dists[i])
            col = tuple([col[2], col[1], col[0], alpha])
            avg_col += np.array(col)
            col_scaled = tuple(255 * i for i in col)
            cv2.circle(col_img_l, pt_tup, 4, col_scaled, -1)
        avg_col /= mesh_img_l.shape[0]
        avg_col *= 255
        col = (avg_col, avg_col, avg_col)
        tmp = np.zeros_like(col_img_l)
        # cv2.fillConvexPoly(tmp, mesh_img_l.astype(int), tuple(avg_col))
        col_img_l = cv2.add(col_img_l, tmp)
        cv2.polylines(col_img_l, [mesh_img_l.astype(int)], True, tuple(avg_col), 2)

        tmp = np.zeros_like(col_img_r)
        # cv2.fillConvexPoly(tmp, mesh_img_r.astype(int), tuple(avg_col))
        col_img_r = cv2.add(col_img_r, tmp)
        cv2.polylines(col_img_r, [mesh_img_r.astype(int)], True, tuple(avg_col), 2)

# XXX Project lms onto the images
for ltr in ltrs:
    lms_sensor = lms_all_sensor[ltr, :]
    tmp = lms_sensor[:, reorder]
    tmp[:, 0] *= -1
    tmp[:, 1] *= -1
    b = tmp[:, 2] > 0
    tmp = tmp[b, :]
    lms_img_r, _ = cv2.projectPoints((R_cam1cam2.dot(tmp.T) + t_cam1cam2).T, rvec, tvec, M_r, D_r)
    lms_img_l, _ = cv2.projectPoints(tmp, rvec, tvec, M_l, D_l)

    # Mark landmarks with dots
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    for lm in lms_img_r:
        lm = np.array(lm, dtype=int)
        lm_tup = tuple(lm.flatten())
        cv2.circle(col_img_r, lm_tup, 10, (0, 255, 0, 255), -1)
        # cv2.putText(col_img_r,
        #             str(ltr[count]), lm_tup, font, 0.8, (255, 0, 0, 255), 2,
        #             cv2.LINE_AA)
        count += 1

    count = 0
    for lm in lms_img_l:
        lm = np.array(lm, dtype=int)
        lm_tup = tuple(lm.flatten())
        cv2.circle(col_img_l, lm_tup, 10, (0, 255, 0, 255), -1)
        # cv2.putText(col_img_l,
        #             str(ltr[count]), lm_tup, font, 0.8, (255, 0, 0, 255), 2,
        #             cv2.LINE_AA)
        count += 1
cv2.imwrite("./Image_orig_l.png", img_l)
cv2.imwrite("./Image_orig_r.png", img_r)
cv2.imwrite("./Image_l.png", col_img_l)
cv2.imwrite("./Image_r.png", col_img_r)

# XXX 3D visualisation
col_options = ["nf", "v", "h"]
N_regions = len(submaps)
N_surf = len(submaps[0])  # Assume all grids have the same depth
points3d = np.zeros((3 * N_surf * N_regions, 3), dtype=float)
color_metric = np.zeros((3 * N_surf * N_regions, len(col_options) - 1), dtype=float)
triangulations = np.array([])
dense_meas = np.zeros((1000000, 3), dtype=float)
n_curr = 0

for i, submap in enumerate(submaps):
    ltr = ltrs[i]
    lms_ltr = lms_all_sensor[ltr, :]
    lms_ltr[:, 2] *= 0

    # [ Precalculate transformation from relative to world IRF
    l0 = lms_ltr[0, :].reshape(3, 1)
    la = lms_ltr[1, :].reshape(3, 1)
    lb = lms_ltr[2, :].reshape(3, 1)

    da = (la - l0).flatten()
    db = (lb - l0).flatten()
    k = da[1] * db[0] - da[0] * db[1]

    n = np.cross(da, db).reshape(3, 1)
    n /= np.linalg.norm(n)

    T_rel_world = np.hstack((da.reshape(3, 1), db.reshape(3, 1), n))
    # ]
    for j, surfel in enumerate(submap.values()):
        corners = surfel.corners

        m = surfel.bel_h_m
        C = surfel.bel_h_C

        # Get the inverse transform
        T_inv = np.linalg.inv(surfel.transform)
        offset = surfel.offset
        corners_3d = np.vstack((corners, m.T))
        mesh_rel = corners_3d.T.reshape(-1, 3, 1)  # (N, d, 1)

        mesh_world = np.einsum("ij,ajk->aik", T_rel_world, mesh_rel) + l0
        mesh_world = mesh_world.reshape(-1, 3)

        start = 3 * j + 3 * N_surf * i
        end = start + 3
        points3d[start:end, :] = mesh_world
        color_metric[start:end, 0] = surfel.n_window + surfel.n_frozen
        color_metric[start:end, 1] = np.sqrt(surfel.bel_v_b / surfel.bel_v_a)

        if args.points:
            z_m = surfel.z_m[: surfel.n_window, :, 0].T
            z_sensor = T_inv.dot(z_m) + offset
            z_world = T_rel_world.dot(z_sensor) + l0
            dense_meas[n_curr : (n_curr + surfel.n_window), :] = z_world.T
            n_curr += surfel.n_window

    triangulation = np.arange(0, points3d.shape[0]).reshape((int(points3d.shape[0] / 3), 3))
    if triangulations.size == 0:
        triangulations = triangulation
    else:
        triangulations = np.vstack((triangulations, triangulation))


# mlab.figure(size=(1920, 1080))

# meshes = {}
# lut_managers = {}
# for col_type in col_options:
#     print(col_type)
#     if col_type == "h":
#         mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap="plasma", name=col_type)

#         lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
#         lut_manager.use_default_range = False
#         lut_manager.data_range = (-0.27, 0)
#     elif col_type == "v":
#         col_sel = col_options.index(col_type)
#         scalars = color_metric[:, col_sel]
#         # b = scalars <= 0
#         # scalars[b] = 1
#         # scalars = np.log(scalars)
#         # scalars[b] = 0
#         mesh = mlab.triangular_mesh(
#             *points3d.T,
#             triangulations,
#             scalars=scalars.flatten(),
#             colormap="viridis",
#             name=col_type,
#         )

#         lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
#         # lut_manager.use_default_range = False
#         # lut_manager.data_range = (0, 1)
#     elif col_type == "nf":
#         col_sel = col_options.index(col_type)
#         scalars = color_metric[:, col_sel]
#         b = scalars <= 0
#         scalars[b] = 1
#         scalars = np.log(scalars)
#         scalars[b] = 0
#         mesh = mlab.triangular_mesh(
#             *points3d.T,
#             triangulations,
#             scalars=scalars.flatten(),
#             colormap="inferno",
#             name=col_type,
#         )

#         lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)

#     lut_manager.use_default_name = False
#     lut_manager.data_name = ""
#     lut_manager.show_scalar_bar = False
#     lut_manager.show_legend = False
#     # lut_managers[col_type] = lut_manager
#     mesh.visible = False
#     if args.nolight:
#         mesh.actor.property.lighting = False
#     # mesh.actor.property.edge_visibility = True
#     # mesh.actor.property.line_width = 1.5
#     mesh.actor.property.opacity = 1.0
#     meshes[col_type] = mesh

# if args.points:
#     pts = mlab.points3d(
#         *dense_meas[:n_curr, :].T,
#         dense_meas[:n_curr, 2],
#         colormap="plasma",
#         scale_factor=0.02,
#         scale_mode="vector",
#     )
#     pts.visible = False
#     if args.nolight:
#         pts.actor.property.lighting = False

#     lut_manager = mlab.scalarbar(pts, title="", orientation="horizontal", nb_labels=0)
#     lut_manager.use_default_range = False
#     lut_manager.data_range = (-0.27, 0)

#     lut_manager.use_default_name = False
#     lut_manager.data_name = ""
#     lut_manager.show_scalar_bar = False
#     lut_manager.show_legend = False

# scene = engine.scenes[0]
# scene.scene.background = (1.0, 1.0, 1.0)  # white background

# # scene.scene.camera.position = [4.094029876552614, -1.979535756032068, 1.1689308834649381]
# # scene.scene.camera.focal_point = [5.543235227038305, -0.2524400731839973, -0.42528611017000884]
# # scene.scene.camera.view_angle = 30.0
# # scene.scene.camera.view_up = [0.0, 0.0, 1.0]
# # scene.scene.camera.clipping_range = [0.45357283783197166, 5.672833934150267]
# # scene.scene.camera.compute_view_plane_normal()
# # scene.scene.render()
# # WX settings
# scene.scene.camera.position = [4.080690282532662, -1.9443231737003708, 1.2845514556036606]
# scene.scene.camera.focal_point = [5.557045358970688, -0.18487170741065545, -0.33953194341768705]
# scene.scene.camera.view_angle = 30.0
# scene.scene.camera.view_up = [0.0, 0.0, 1.0]
# scene.scene.camera.clipping_range = [0.4111023123325719, 5.6351827182815395]
# scene.scene.camera.compute_view_plane_normal()
# scene.scene.render()

# for col_type in col_options:
#     meshes[col_type].visible = True
#     if args.nolight:
#         meshes[col_type].actor.property.edge_visibility = True
#         meshes[col_type].actor.property.line_width = 1.5
#     filename = f"3d_{col_type}.png"
#     filename = os.path.join(save_path, filename)
#     mlab.savefig(filename, magnification=1)
#     meshes[col_type].visible = False
#     meshes[col_type].actor.property.edge_visibility = False

# pts.visible = True
# # if args.nolight:
# #     pts.actor.property.edge_visibility = True
# #     pts.actor.property.line_width = 1.5
# filename = "3d_pts.png"
# filename = os.path.join(save_path, filename)
# mlab.savefig(filename, magnification=1)
# pts.visible = False
# mlab.show()


print("Plotting in Matplotlib")
cmaps = {"h": "plasma", "v": "viridis", "nf": "inferno"}
cbar_labels = {
    "h": r"Height (m)",
    "v": r"Planar deviation (m\textsuperscript{2})",
    "nf": "Number of measurements",
}

points3d = points3d[:, [1, 0, 2]]


for col_type in col_options:
    ratio = (np.max(points3d[:, 1]) - np.min(points3d[:, 1])) * 1.01
    ratio /= np.max(points3d[:, 0]) - np.min(points3d[:, 0])
    scale = 0.48
    fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    ax = plt.gca()
    ncont = 25

    if col_type == "h":
        scalars = points3d[:, 2]
        ax.tricontour(
            *points3d[:, :2].T, scalars, levels=ncont, linewidths=0.5, colors="k", alpha=0.5
        )
        # vmin = np.min(scalars)
        # vmax = np.max(scalars)
        vmin = -0.27
        vmax = 0
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cntr = ax.tricontourf(
            *points3d[:, :2].T, scalars, levels=ncont, cmap=cmaps[col_type], norm=norm
        )
        ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
    elif col_type == "v":
        col_sel = col_options.index(col_type)
        scalars = color_metric[:, col_sel]
        vmin = np.min(scalars)
        vmax = np.max(scalars)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
        cntr = ax.tripcolor(
            *points3d[:, :2].T,
            triangulation,
            scalars,
            linewidth=0.5,
            cmap=cmaps[col_type],
            norm=norm,
        )
    elif col_type == "nf":
        col_sel = col_options.index(col_type)
        scalars = color_metric[:, col_sel]
        ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
        vmin = np.min(scalars)
        vmax = np.max(scalars)
        norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, clip=True)
        cntr = ax.tripcolor(
            *points3d[:, :2].T,
            triangulation,
            scalars,
            linewidth=0.5,
            cmap=cmaps[col_type],
            norm=norm,
        )

    print(col_type, vmin, vmax)
    ax.set_aspect("equal", "box")

    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # NOTE axes swapped around
    plt.xlabel("y (m)")
    plt.ylabel("x (m)")
    cb = plt.colorbar(cntr, label=cbar_labels[col_type], orientation="horizontal", pad=0.0)
    if col_type != "nf":
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
    cb.solids.set_edgecolor("face")
    plt.draw()

    ax.set_xlim(np.min(points3d[:, 0]), np.max(points3d[:, 0]))
    ax.set_ylim(np.min(points3d[:, 1]), np.max(points3d[:, 1]))

    filename = f"2d_{col_type}.pdf"
    filename = os.path.join(save_path, filename)
    fig.savefig(filename)

# Plot height and variation together
col_options = ["h", "v"]

plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True
ratio = (np.max(points3d[:, 1]) - np.min(points3d[:, 1])) * 1.1
ratio /= np.max(points3d[:, 0]) - np.min(points3d[:, 0]) * 2.5
scale = 0.98
fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
fig, axes = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=fig_size, tight_layout=True
)
print(axes.shape)
ncont = 25
axes[0].set_ylabel("x (m)")
for i, col_type in enumerate(col_options):
    ax = axes[i]
    if col_type == "h":
        scalars = points3d[:, 2]
        ax.tricontour(
            *points3d[:, :2].T, scalars, levels=ncont, linewidths=0.5, colors="k", alpha=0.5
        )
        # vmin = np.min(scalars)
        # vmax = np.max(scalars)
        vmin = -0.27
        vmax = 0
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cntr = ax.tricontourf(
            *points3d[:, :2].T, scalars, levels=ncont, cmap=cmaps[col_type], norm=norm
        )
        ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
    elif col_type == "v":
        col_sel = col_options.index(col_type)
        scalars = color_metric[:, col_sel]
        vmin = np.min(scalars)
        vmax = np.max(scalars)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
        cntr = ax.tripcolor(
            *points3d[:, :2].T,
            triangulation,
            scalars,
            linewidth=0.5,
            cmap=cmaps[col_type],
            norm=norm,
        )
    ax.set_aspect("equal", "box")

    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # NOTE axes swapped around
    ax.set_xlabel("y (m)")
    ax.xaxis.set_label_position("top")
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cb = plt.colorbar(
        cntr, label=cbar_labels[col_type], orientation="horizontal", pad=0.0, cax=cax
    )
    if col_type != "nf":
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(labelsize=plt_cfg.tex_fontsize)
    plt.draw()

    ax.set_xlim(np.min(points3d[:, 0]), np.max(points3d[:, 0]))
    ax.set_ylim(np.min(points3d[:, 1]), np.max(points3d[:, 1]))

plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=1.0, hspace=0.0, wspace=0.00)

filename = f"combined.pdf"
filename = os.path.join(save_path, filename)
fig.savefig(filename)

plt.show()
