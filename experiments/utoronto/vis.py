# -*- coding: utf-8 -*-
#!/usr/bin/python3

import argparse
import os
import pickle
import random
import re
import textwrap as _textwrap

import dill
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import mayavi.mlab as mlab
import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull, Delaunay

from stmmap import RelativeSubmap

from .utils import *

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()
# -------------------------------------------

# Allow bullets in args help
class PreserveWhiteSpaceWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __add_whitespace(self, idx, iWSpace, text):
        if idx is 0:
            return text
        return (" " * iWSpace) + text

    def _split_lines(self, text, width):
        textRows = text.splitlines()
        for idx, line in enumerate(textRows):
            search = re.search("\s*[0-9\-]{0,}\.?\s*", line)
            if line.strip() is "":
                textRows[idx] = " "
            elif search:
                lWSpace = search.end()
                lines = [
                    self.__add_whitespace(i, lWSpace, x)
                    for i, x in enumerate(_textwrap.wrap(line, width))
                ]
                textRows[idx] = lines

        return [item for sublist in textRows for item in sublist]


parser = argparse.ArgumentParser(formatter_class=PreserveWhiteSpaceWrapRawTextHelpFormatter)
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("--save", help="Save the view points.", action="store_true")
parser.add_argument("--seed", help="Noise seed.", type=int)
parser.add_argument("--nolight", help="Disable lighting", action="store_true")

args = parser.parse_args()

main_path = args.path
if main_path[-1] != "/":
    main_path += "/"
submaps_path = main_path + "submaps/"

dataset_specific_flags = {}
if "p2at_met" in main_path:
    print("Found p2at_met DS")
    dataset_specific_flags["p2at_met"] = True
else:
    dataset_specific_flags["p2at_met"] = False
if "box_met" in main_path:
    print("Found box_met DS")
    dataset_specific_flags["box_met"] = True
else:
    dataset_specific_flags["box_met"] = False

seed = None
if args.seed is not None:
    seed = args.seed
np.random.seed(seed)
random.seed(seed)

col_options = ["nf", "v", "h"]
# col_type = args.col
# assert col_options[-1] == 'h'  # h must always be last
# if not (col_type in col_options):
#     raise ValueError("Invalid col parameter: " + col_type)

setup = pickle.load(open(os.path.join(main_path, "setup.p"), "rb"))
lms = setup["lms"]

# Rotate lms to be inline with the axes
hull = ConvexHull(lms[:, :2, 0])
vert = hull.vertices
corner_lms = lms[vert]
a = corner_lms[0] - corner_lms[1]
a /= np.linalg.norm(a)
rot_angle = np.arccos(a[1])
K = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * K.dot(K)
lms = np.einsum("ij,kjl->kil", R, lms)

triangulation = setup["triangulation"]
grid_depth = setup["grid_depth"]

grid_files = [f for f in os.listdir(submaps_path) if os.path.isfile(submaps_path + f)]
grid_files.sort()
N_grids = len(grid_files)
print("Number of grids found: ", N_grids)

if N_grids == 0:
    print("No files found. I'm out...")
    exit()
elif N_grids == 1:
    index = 0
else:
    while 1:
        index_str = input("Choose index from 0 to %d: " % (N_grids - 1))
        if index_str == "":
            index = N_grids - 1
            break
        try:
            index = int(index_str)
            if (index >= 0) and (index < N_grids):
                break
            else:
                print("Index is not within range.")
        except:
            print("Must be an integer from 0 to %d" % (N_grids - 1))

print("Loading file:", submaps_path + grid_files[index])
grids = pickle.load(open(submaps_path + grid_files[index], "rb"))

mlab.figure(size=(1920, 1080))

simplices = triangulation.simplices
N_regions = simplices.shape[0]

mesh = mlab.triangular_mesh(
    *lms[:, :, 0].T,
    triangulation.simplices,
    colormap="viridis",
    representation="wireframe",
    name="lm mesh"
)
mesh.visible = False

# 3D visualisation
points3d = np.array([])
points3d_var = np.array([])
color_metric = np.array([])
triangulations = np.array([])

dense_pts = np.array([])
print("Reading in values")
# This is very slow...
N_surf = len(grids[0])  # Assume all grids have the same depth
points3d = np.zeros((3 * N_surf * N_regions, 3), dtype=float)
color_metric = np.zeros((3 * N_surf * N_regions, len(col_options) - 1), dtype=float)
for i in range(N_regions):
    grid = grids[i]
    ltr = simplices[i]
    lms_ltr = lms[ltr, :]

    # [ Precalculate transformation from relative to world IRF
    l0 = lms_ltr[0, :]
    la = lms_ltr[1, :]
    lb = lms_ltr[2, :]

    da = (la - l0).flatten()
    db = (lb - l0).flatten()
    k = da[1] * db[0] - da[0] * db[1]

    n = np.cross(da, db).reshape(3, 1)
    n /= np.linalg.norm(n)

    T_rel_world = np.hstack((da.reshape(3, 1), db.reshape(3, 1), n))
    # ]

    for j, surfel in enumerate(grid.values()):
        corners = surfel.corners

        m = surfel.bel_h_m
        C = surfel.bel_h_C

        # Get the inverse transform
        T_inv = np.linalg.inv(surfel.transform)
        offset = surfel.offset
        corners_3d = np.vstack((corners, m.T))
        mesh_rel = corners_3d.T.reshape(-1, 3, 1)  # (N, d, 1)

        # XXX Use this to visualise raw measurements in the map
        # if surfel.pt_archive.size != 0:
        #     print(surfel.pt_archive.size, surfel.tri_id)
        #     pts_rel = (T_inv.dot(surfel.pt_archive[:, :, 0].T).T + offset[:, 0])
        #     pts_rel = pts_rel.reshape(-1,3,1)
        #     pts = np.einsum('ij,ajk->aik', T_rel_world, pts_rel) + l0
        #     pts = pts.reshape(-1, 3)
        #     if dense_pts.size == 0:
        #         dense_pts = pts
        #     else:
        #         dense_pts = np.append(dense_pts, pts, axis=0)
        # elif len(surfel.unfused_means) != 0:
        #     pts = np.array(surfel.unfused_means)
        #     pts_rel = (T_inv.dot(pts[:, :, 0].T).T + offset[:, 0])
        #     pts_rel = pts_rel.reshape(-1,3,1)
        #     pts = np.einsum('ij,ajk->aik', T_rel_world, pts_rel) + l0
        #     pts = pts.reshape(-1, 3)
        #     if dense_pts.size == 0:
        #         dense_pts = pts
        #     else:
        #         dense_pts = np.append(dense_pts, pts, axis=0)

        mesh_world = np.einsum("ij,ajk->aik", T_rel_world, mesh_rel) + l0
        mesh_world = mesh_world.reshape(-1, 3)

        start = 3 * j + 3 * N_surf * i
        end = start + 3
        points3d[start:end, :] = mesh_world
        color_metric[start:end, 0] = surfel.n_window + surfel.n_frozen
        color_metric[start:end, 1] = np.sqrt(surfel.bel_v_b / surfel.bel_v_a)

    triangulation = np.arange(0, points3d.shape[0]).reshape((int(points3d.shape[0] / 3), 3))
    if triangulations.size == 0:
        triangulations = triangulation
    else:
        triangulations = np.vstack((triangulations, triangulation))

print("Displaying results")
print("Points:", points3d.shape)
# if dense_pts.size != 0:
#     mlab.points3d(*dense_pts.T, scale_factor=0.1)

# Plot unfused points
# mlab.points3d(
#     *points3d.T,
#     points3d[:, 2],
#     scale_factor=1,
#     scale_mode="none",
#     colormap="viridis",
#     opacity=0.9)
meshes = {}
lut_managers = {}
for col_type in col_options:
    print(col_type)
    if col_type == "h":
        mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap="plasma", name=col_type)

        lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
        lut_manager.use_default_range = False
        lut_manager.data_range = (-3.5, 1)
    elif col_type == "v":
        col_sel = col_options.index(col_type)
        scalars = color_metric[:, col_sel]
        # b = scalars <= 0
        # scalars[b] = 1
        # scalars = np.log(scalars)
        # scalars[b] = 0
        mesh = mlab.triangular_mesh(
            *points3d.T,
            triangulations,
            scalars=scalars.flatten(),
            colormap="viridis",
            name=col_type
        )

        lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
        lut_manager.use_default_range = False
        lut_manager.data_range = (0, 1)
    elif col_type == "nf":
        col_sel = col_options.index(col_type)
        scalars = color_metric[:, col_sel]
        b = scalars <= 0
        scalars[b] = 1
        scalars = np.log(scalars)
        scalars[b] = 0
        mesh = mlab.triangular_mesh(
            *points3d.T,
            triangulations,
            scalars=scalars.flatten(),
            colormap="inferno",
            name=col_type
        )

        lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)

    lut_manager.use_default_name = False
    lut_manager.data_name = ""
    lut_manager.show_scalar_bar = False
    lut_manager.show_legend = False
    mesh.visible = False
    if args.nolight:
        mesh.actor.property.lighting = False
    # mesh.actor.property.edge_visibility = True
    # mesh.actor.property.line_width = 1.5
    mesh.actor.property.opacity = 1.0
    meshes[col_type] = mesh
    lut_managers[col_type] = lut_manager

if not args.save:
    mlab.show()
    exit()

# XXX Save specific view points
scene = engine.scenes[0]
scene.scene.background = (1.0, 1.0, 1.0)  # white background

save_path = os.path.join(main_path, "scenes/")
print(save_path, main_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Isometric view
for col_type in col_options:
    meshes[col_type].visible = True
    scene.scene.camera.position = [42.217336325569896, -73.78638477744869, 69.5555264984863]
    scene.scene.camera.focal_point = [-40.698279343987856, 9.129230892109124, -13.360089171071504]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.scene.camera.clipping_range = [38.333634437416315, 276.4868285350922]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 0.1
    filename = "iso_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=3)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False

# Top View
scene.scene.camera.position = [-51.41366611536666, 10.591947526943116, 117.49309591847023]
scene.scene.camera.focal_point = [-51.41366611536666, 10.591947526943116, -1.1971726213649483]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.0, 1.0, 0.0]
scene.scene.camera.clipping_range = [111.05759529244679, 128.58451968239754]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
# scene.scene.parallel_projection = True
# scene.scene.camera.parallel_scale = 38*1.5
for col_type in col_options:
    meshes[col_type].visible = True

    # lut_managers[col_type].show_scalar_bar = True
    # lut_managers[col_type].show_legend = True

    # if args.nolight:
    #     meshes[col_type].actor.property.edge_visibility = True
    #     meshes[col_type].actor.property.line_width = 0.1
    filename = "top_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=3)
    lut_managers[col_type].show_scalar_bar = False
    lut_managers[col_type].show_legend = False
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False
scene.scene.parallel_projection = False

# Middle heap 1
for col_type in col_options:
    if dataset_specific_flags["p2at_met"]:
        scene.scene.camera.position = [16.221040987451286, -29.666093122533486, 11.673166927492149]
        scene.scene.camera.focal_point = [
            5.8114457738690675,
            -40.50361041870613,
            -0.39050476101674425,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.scene.camera.clipping_range = [0.14771409672427943, 147.71409672427941]
    elif dataset_specific_flags["box_met"]:
        scene.scene.camera.position = [-34.21092141150307, -17.276974544865286, 9.01571999723853]
        scene.scene.camera.focal_point = [
            -42.07013498616955,
            -6.062988429209429,
            -1.580882797064015,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.2775467576102802, 0.5498887222941172, 0.787775469555738]
        scene.scene.camera.clipping_range = [0.13927715562142295, 139.27715562142296]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = "heap_1_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False

# Middle heap 2
for col_type in col_options:
    if dataset_specific_flags["p2at_met"]:
        scene.scene.camera.position = [-32.18275574399024, -71.31261009623663, 3.355334561348498]
        scene.scene.camera.focal_point = [
            -26.550959521827565,
            -60.97128470627609,
            -0.9661247864271556,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.15350111965726185, 0.3086713449240815, 0.9387009146084422]
        scene.scene.camera.clipping_range = [0.15964566031721372, 159.64566031721372]
    elif dataset_specific_flags["box_met"]:
        scene.scene.camera.position = [-71.36705682461039, 36.66178337919497, 3.9945895967028906]
        scene.scene.camera.focal_point = [
            -54.92784043303266,
            24.10071595053957,
            -5.326113201350141,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.3185433686992782, -0.25959520485052684, 0.9116690473391805]
        scene.scene.camera.clipping_range = [0.1598268517231999, 159.8268517231999]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = "heap_2_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False

# Cliff
for col_type in col_options:
    if dataset_specific_flags["p2at_met"]:
        scene.scene.camera.position = [22.17128119626667, -78.83702658131341, 6.3210189630520315]
        scene.scene.camera.focal_point = [
            13.573920292320809,
            -83.02276179817655,
            0.6820427166811363,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [
            -0.4501761617692164,
            -0.23566148600699502,
            0.8612810733945747,
        ]
        scene.scene.camera.clipping_range = [0.11228165651393682, 112.28165651393681]
    elif dataset_specific_flags["box_met"]:
        scene.scene.camera.position = [-74.8061039248119, -19.01704109900529, 5.940112781802376]
        scene.scene.camera.focal_point = [
            -82.75773701471383,
            -12.578872179878106,
            -0.45998845603942173,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.3943515050779973, 0.36665785726829714, 0.8426439972765126]
        scene.scene.camera.clipping_range = [0.1051863829037232, 105.1863829037232]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = "cliff_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False

# Pathway up
for col_type in col_options:
    if dataset_specific_flags["p2at_met"]:
        scene.scene.camera.position = [-13.145536160846707, -60.33720579769111, 6.015630819042983]
        scene.scene.camera.focal_point = [
            -14.216176137814989,
            -77.67309157636734,
            -1.8383146208833456,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [
            -0.015217158541222766,
            -0.41184263383019143,
            0.9111279180475936,
        ]
        scene.scene.camera.clipping_range = [0.087740985347917, 87.74098534791699]
    elif dataset_specific_flags["box_met"]:
        scene.scene.camera.position = [-61.270195620179976, 14.832140534898889, 4.9454089144093585]
        scene.scene.camera.focal_point = [
            -78.86028202593761,
            14.334762619506158,
            -3.0160802959566846,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [
            -0.41108305854889726,
            -0.03976945203930041,
            0.9107299872402235,
        ]
        scene.scene.camera.clipping_range = [0.08624314050691477, 86.24314050691477]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = "pathway_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False

# Cave heaps
for col_type in col_options:
    if dataset_specific_flags["p2at_met"]:
        print("Camera angles not found for this scene")
        break
    elif dataset_specific_flags["box_met"]:
        scene.scene.camera.position = [-2.450368835702743, 31.957154177942904, 10.708592564982546]
        scene.scene.camera.focal_point = [
            -4.730236307626639,
            18.977055219163777,
            -1.3406058790781625,
        ]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [
            -0.12890616956601822,
            -0.6624050338261784,
            0.7379720662799889,
        ]
        scene.scene.camera.clipping_range = [0.10571248160985322, 105.71248160985321]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = "caves_" + col_type + ".png"
    filename = os.path.join(save_path, filename)
    mlab.savefig(filename, magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False
