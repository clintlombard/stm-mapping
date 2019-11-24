# -*- coding: utf-8 -*-
#!/usr/bin/python3

import argparse
import os
import pickle
import random
import re
import textwrap as _textwrap

from pathlib import Path

import dill
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull, Delaunay

from compare_surface_maps.elevation_maps import ElevationMap3D
from stmmap import RelativeSubmap

try:
    os.environ["ETS_TOOLKIT"] = "wx"
    import mayavi.mlab as mlab
    from .utils import *
except Exception as e:
    import mayavi.mlab as mlab
    from .utils import *


try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()
# -------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("--save", help="Save the view points.", action="store_true")
parser.add_argument("--seed", help="Noise seed.", type=int)
parser.add_argument("--nolight", help="Disable lighting", action="store_true")

args = parser.parse_args()

seed = args.seed
main_path = Path(args.path).resolve()
submaps_path = main_path / "submaps"

# NOTE for now assume that the box_met dataset is being used
dataset_specific_flags = dict()
dataset_specific_flags["box_met"] = True

seed = None
if args.seed is not None:
    seed = args.seed
np.random.seed(seed)
random.seed(seed)

col_options = ["nf", "v", "h"]

setup = pickle.load(open(os.path.join(main_path, "setup.p"), "rb"))
lms = setup["lms"]

# TODO is this necessary
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

grid_files = [f for f in os.listdir(submaps_path)]
grid_files.sort()
N_grids = len(grid_files)
print("Number of stm_grids found: ", N_grids)

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

print("Loading files in:", submaps_path / grid_files[index])

mlab.figure(size=(1920, 1080))

simplices = triangulation.simplices
N_regions = simplices.shape[0]

mesh = mlab.triangular_mesh(
    *lms[:, :, 0].T,
    triangulation.simplices,
    colormap="viridis",
    representation="wireframe",
    name="lm mesh",
)
mesh.visible = False


# Plot STM Map
print("Plotting STM Map")
stm_grids = pickle.load(open(submaps_path / grid_files[index] / "stm.p", "rb"))

# 3D visualisation
points3d = np.array([])
color_metric = np.array([])
triangulations = np.array([])

dense_pts = np.array([])
# This is very slow...
N_surf = len(stm_grids[0])  # Assume all stm_grids have the same depth
points3d = np.zeros((3 * N_surf * N_regions, 3), dtype=float)
color_metric = np.zeros((3 * N_surf * N_regions, len(col_options) - 1), dtype=float)
for i in range(N_regions):
    grid = stm_grids[i]
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

meshes = {}
lut_managers = {}
for col_type in col_options:
    print(col_type)
    if col_type == "h":
        mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap="plasma", name=col_type)

        lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
        lut_manager.use_default_range = False
        lut_manager.data_range = (-3.1, 0.2)
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
            name=col_type,
        )

        lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
        # lut_manager.use_default_range = False
        # lut_manager.data_range = (0, 1)
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
            name=col_type,
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

# Plot elevation map
print("Plotting Elevation Map")
elev_grids = pickle.load(open(submaps_path / grid_files[index] / "elev.p", "rb"))

# 3D visualisation
points3d = np.array([])
triangulations = np.array([])

dense_pts = np.array([])
# This is very slow...
N_surf = len(elev_grids[0])  # Assume all elev_grids have the same depth
points3d = np.zeros((3 * N_surf * N_regions, 3), dtype=float)
color_metric = np.zeros((3 * N_surf * N_regions, len(col_options) - 1), dtype=float)
for i in range(N_regions):
    grid = elev_grids[i]
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
        corners_3d = np.zeros((3, 3))
        corners_3d[:2, :] = surfel.corners

        m = surfel.bel_h_m
        C = surfel.bel_h_C

        # Get the inverse transform
        T_inv = np.linalg.inv(surfel.transform)
        offset = surfel.offset
        corners_3d[2, :] = m
        mesh_rel = corners_3d.T.reshape(-1, 3, 1)  # (N, d, 1)

        mesh_world = np.einsum("ij,ajk->aik", T_rel_world, mesh_rel) + l0
        mesh_world = mesh_world.reshape(-1, 3)

        start = 3 * j + 3 * N_surf * i
        end = start + 3
        points3d[start:end, :] = mesh_world

    triangulation = np.arange(0, points3d.shape[0]).reshape((int(points3d.shape[0] / 3), 3))
    if triangulations.size == 0:
        triangulations = triangulation
    else:
        triangulations = np.vstack((triangulations, triangulation))

mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap="plasma", name="elevation")
# triangulations = Delaunay(points3d[:, :2]).simplices
# mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap="plasma", name="elevation")

lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=0)
lut_manager.use_default_range = False
lut_manager.data_range = (-3.1, 0.2)

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
meshes["elevation"] = mesh
lut_managers["elevation"] = lut_manager

if not args.save:
    mlab.show()
    exit()

# XXX Save specific view points
save_path = main_path / "scenes"
if not os.path.exists(save_path):
    os.makedirs(save_path)

scene = engine.scenes[0]
scene.scene.background = (1.0, 1.0, 1.0)  # white background

scene.scene.camera.position = [-62.711753438810604, -53.97895685490163, 35.505462783605786]
scene.scene.camera.focal_point = [-85.78755652469613, -4.492737440904958, -3.103982951529997]
scene.scene.camera.view_angle = 15.36
scene.scene.camera.view_up = [-0.24399876718044772, 0.5232570448142633, 0.8164965809277254]
scene.scene.camera.clipping_range = [39.987551578962204, 100.87872165042502]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()

col_options += ["elevation"]
for col_type in col_options:
    meshes[col_type].visible = True
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    meshes[col_type].visible = True
    if args.nolight:
        meshes[col_type].actor.property.edge_visibility = True
        meshes[col_type].actor.property.line_width = 1.5
    filename = save_path / f"{col_type}.png"
    mlab.savefig(str(filename), magnification=1)
    meshes[col_type].visible = False
    meshes[col_type].actor.property.edge_visibility = False
