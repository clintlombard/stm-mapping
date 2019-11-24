#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualisation of a station dataset fused using stationary_fuse.py

Recursively visualise all datasets in the passed directory. Under the assumption that the datasets
are all of the same scene.

Created on 29 May 2019 19:13

@author Clint Lombard

"""

import argparse
import ast
import os

import dill

# try: # This needs to be before all other matplotlib imports
#     import matplotlib
#     matplotlib.use("WX")
# except:
#     pass
import matplotlib.colors as colors
import matplotlib.pylab as plt
import matplotlib.tri as mtri
import mayavi.mlab as mlab
import numpy as np
import seaborn as sb

from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stmmap import RelativeSubmap, Surfel
from stmmap.utils.plot_utils import PlotConfig

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()

# -------- Plot configurations ---------------------------------------------------------
plt_cfg = PlotConfig()

# Need this to load dill-pickled sympy lambdified functions
dill.settings["recurse"] = True

parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="path/to/dataset", type=str)
parser.add_argument("sensor_type", metavar="STEREO/LIDAR", type=str, default="STEREO")
parser.add_argument("--nolight", help="Disable lighting", action="store_true")
parser.add_argument("--save", help="Save the view points.", action="store_true")

args = parser.parse_args()

main_path = args.path

sensor_type = args.sensor_type

# 1 Find all dataset folders used in fusing
candidate_folders = [
    os.path.join(main_path, name)
    for name in os.listdir(main_path)
    if os.path.isdir(os.path.join(main_path, name))
]
print("Candidate folders:\n", candidate_folders)

# Double check they contain valid dataset
ds_folders = []
submap_folder = os.path.join("submap", sensor_type)
require = ["Cameras/", "slam/", submap_folder]
for i, f in enumerate(candidate_folders):
    add_dir = True
    for r in require:
        if not os.path.isdir(os.path.join(f, r)):
            add_dir = False
            break
    if add_dir:
        ds_folders.append(f)

print("Valid DS folders:\n", ds_folders)

# 2 Read in fused submaps
print("Reading in fused submaps")
# fused_path = os.path.join(main_path, "fused_submaps", sensor_type)
fused_path = os.path.join(main_path, "submap", sensor_type)
if not os.path.exists(fused_path):
    # raise RuntimeWarning("Could not find fused data in the root path.")
    pass
else:
    ds_folders.append(main_path)

col_options = ["nf", "v", "h"]
cbar_labels = {
    "h": r"Height (m)",
    "v": r"Planar deviation (m\textsuperscript{2})",
    "nf": "Number of measurements",
}
cmaps = {"h": "plasma", "v": "viridis", "nf": "inferno"}

for path_count in range(len(ds_folders)):
    submap_dicts = dict()
    curr_path = os.path.join(ds_folders[path_count], "submap", sensor_type)

    unique_lms = set()
    for subdir in os.listdir(curr_path):
        key = ast.literal_eval(subdir)
        assert type(key) == tuple
        subpath = os.path.join(curr_path, subdir)
        for name in os.listdir(subpath):
            if name.endswith(".p"):
                filename = os.path.join(subpath, name)
                submap_surf_dict = dill.load(open(filename, "rb"))

                if key in submap_dicts:
                    raise KeyError(
                        "Already inserted this submap. You can only have one pickled map per submap."
                    )

                unique_lms |= set([i for i in key])
                submap_dicts[key] = submap_surf_dict
    print("Unique landmarks:", unique_lms)

    # 3 Read in slam landmark positions from the first dataset
    # slam_path = os.path.join(main_path, ds_folders[2], "slam")
    # TODO Maybe do something like this? It might make all the maps have wonky alignment.
    if path_count == len(ds_folders) - 1:
        # Use the first DS folder for the fused data
        slam_path = os.path.join(main_path, ds_folders[2], "slam")
    else:
        # slam_path = os.path.join(main_path, ds_folders[path_count], "slam")
        slam_path = os.path.join(main_path, ds_folders[0], "slam")

    slam_ts = []
    ts_filename = os.path.join(slam_path, "timestamps.txt")
    with open(ts_filename, "r") as stream:
        for line in stream.readlines():
            split = line.split()
            slam_ts.append([split[0], split[1], float(split[2])])

    mean_filename = os.path.join(slam_path, "mean-" + slam_ts[-1][1] + ".npy")
    slam_mean = np.load(mean_filename)

    # Remove unobserved LMs
    indices = [j for j in range(7)]
    for l in unique_lms:
        for k in range(3):
            indices.append(7 + 3 * l + k)
    indices.sort()
    slam_mean = slam_mean[indices, :]
    if (slam_mean[7:] == 0).any():
        print("Error: LTR landmarks partially observed.")
        exit()

    # Transform lms to sensor frame
    N_lms = len(unique_lms)
    lms = slam_mean[7:].reshape((N_lms, 3))
    # lms = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],dtype=float)

    print("Reading in values")
    # XXX 3D visualisation
    points3d = np.array([])
    points3d_var = np.array([])
    color_metric = np.array([])
    triangulations = np.array([])

    dense_pts = np.array([])
    N_regions = len(submap_dicts)
    N_surf = len(submap_dicts[key])  # Assume all grids have the same depth
    points3d = np.zeros((3 * N_surf * N_regions, 3), dtype=float)
    color_metric = np.zeros((3 * N_surf * N_regions, len(col_options) - 1), dtype=float)
    triangulations = np.array([])

    dense_pts = np.array([])
    for i, lm_ids in enumerate(submap_dicts.keys()):
        submap = submap_dicts[lm_ids]
        lms_ltr = lms[lm_ids, :]

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

        triangulation = np.arange(0, points3d.shape[0]).reshape((int(points3d.shape[0] / 3), 3))
        if triangulations.size == 0:
            triangulations = triangulation
        else:
            triangulations = np.vstack((triangulations, triangulation))

    # # Shift z-axis origin to lowest point
    # z_min = np.min(points3d[:, 2])
    points3d[:, 2] -= -1.575

    # print("Plotting in Mayavi")
    # mlab.figure(size=(1920, 1080))
    # meshes = {}
    # lut_managers = {}
    # for col_type in col_options:
    #     print(col_type)
    #     if col_type == "h":
    #         mesh = mlab.triangular_mesh(*points3d.T, triangulations, colormap=cmaps[col_type], name=col_type)
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
    #             colormap=cmaps[col_type],
    #             name=col_type
    #         )

    #     elif col_type == "nf":
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
    #             colormap=cmaps[col_type],
    #             name=col_type
    #         )

    #     lut_manager = mlab.scalarbar(mesh, title="", orientation="horizontal", nb_labels=5)
    #     lut_manager.use_default_name = False
    #     lut_manager.data_name = ""
    #     lut_manager.show_scalar_bar = False
    #     lut_manager.show_legend = False
    #     lut_managers[col_type] = lut_manager
    #     mesh.visible = False
    #     if args.nolight:
    #         mesh.actor.property.lighting = False
    #         mesh.actor.property.edge_visibility = True
    #         mesh.actor.property.line_width = 1.5
    #     mesh.actor.property.opacity = 1.0
    #     meshes[col_type] = mesh

    # if args.save:
    #     # XXX Save specific view points
    #     scene = engine.scenes[0]
    #     scene.scene.background = (1.0, 1.0, 1.0)  # white background

    #     for col_type in col_options:
    #         scene.scene.camera.position = [6.205246580390905, 0.07039329286808771, 10.111725430415998]
    #         scene.scene.camera.focal_point = [
    #             6.205246580390905,
    #             0.07039329286808771,
    #             0.29846364739939313,
    #         ]
    #         scene.scene.camera.view_angle = 15.36
    #         scene.scene.camera.view_up = [-0.03903656348160365, 0.9992377828682955, 0.0]
    #         scene.scene.camera.clipping_range = [9.121186506861648, 10.708112146497333]
    #         scene.scene.camera.compute_view_plane_normal()
    #         scene.scene.render()
    #         meshes[col_type].visible = True
    #         lut_managers[col_type].show_scalar_bar = True
    #         lut_managers[col_type].show_legend = True
    #         if args.nolight:
    #             meshes[col_type].actor.property.edge_visibility = True
    #             meshes[col_type].actor.property.line_width = 1.5
    #         filename = "top_" + col_type + ".png"
    #         save_path = os.path.join(ds_folders[path_count], "vis")
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         filename = col_type + ".pdf"
    #         filename = os.path.join(save_path, filename)
    #         mlab.savefig(filename, magnification=1)
    #         lut_managers[col_type].show_scalar_bar = False
    #         lut_managers[col_type].show_legend = False
    #         meshes[col_type].visible = False
    #         meshes[col_type].actor.property.edge_visibility = False

    print("Plotting in Matplotlib")

    ncont = 25
    if path_count == len(ds_folders) - 1:
        plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
        plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True
        scale = 0.98
        ratio = (np.max(points3d[:, 1]) - np.min(points3d[:, 1])) * 1.28
        ratio /= (np.max(points3d[:, 0]) - np.min(points3d[:, 0])) * 3

        fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
        fig, axes = plt.subplots(
            nrows=1, ncols=3, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True
        )
        axes[0].set_ylabel("y (m)")

        for i, col_type in enumerate(col_options):
            axes[i].set_xlabel("x (m)")
            axes[i].xaxis.set_label_position("top")
            if col_type == "h":
                scalars = points3d[:, 2]
                axes[i].tricontour(
                    *points3d[:, :2].T,
                    scalars,
                    levels=ncont,
                    linewidths=0.5,
                    colors="k",
                    alpha=0.5,
                )
                vmin = 0
                vmax = 0.6
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cntr = axes[i].tricontourf(
                    *points3d[:, :2].T, scalars, levels=ncont, cmap=cmaps[col_type], norm=norm
                )
                axes[i].triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
            elif col_type == "v":
                col_sel = col_options.index(col_type)
                scalars = color_metric[:, col_sel]
                vmin = 0
                vmax = 0.25
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                axes[i].triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
                cntr = axes[i].tripcolor(
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
                axes[i].triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
                vmin = 0
                vmax = 1e4
                norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, clip=True)
                cntr = axes[i].tripcolor(
                    *points3d[:, :2].T,
                    triangulation,
                    scalars,
                    linewidth=0.5,
                    cmap=cmaps[col_type],
                    norm=norm,
                )
            axes[i].set_aspect("equal", "box")
            axes[i].set_xlim(np.min(points3d[:, 0]), np.max(points3d[:, 0]))
            axes[i].set_ylim(np.min(points3d[:, 1]), np.max(points3d[:, 1]))

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
            cb.ax.tick_params(labelsize=plt_cfg.tex_fontsize - 2)
            cax.set_xticklabels([f"{t:e}" for t in cb.get_ticks()], rotation="vertical")
            cb.update_ticks()
            plt.draw()
        # plt.subplots_adjust(
        #     top=1.0, bottom=0.0, left=0.00, right=1.0, hspace=0.0, wspace=0.0
        # )
        # fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, hspace=0.0, wspace=0.0, right=0.95)
        fig.set_constrained_layout_pads(w_pad=2.0 / 72.0, h_pad=0.0 / 72.0, hspace=0.0, wspace=0.0)
        if args.save:
            save_path = os.path.join(ds_folders[path_count], "vis")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = "fused.pdf"
            filename = os.path.join(save_path, filename)
            fig.savefig(filename)
        plt.show()
    else:
        for col_type in col_options:
            plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = True
            plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = False
            ratio = np.max(points3d[:, 1]) - np.min(points3d[:, 1])
            ratio /= np.max(points3d[:, 0]) - np.min(points3d[:, 0])
            scale = 0.25
            fig_size = (scale * plt_cfg.tex_textwidth, scale * ratio * plt_cfg.tex_textwidth)
            fig = plt.figure(figsize=fig_size)
            ax = plt.gca()
            if col_type == "h":
                scalars = points3d[:, 2]
                ax.tricontour(
                    *points3d[:, :2].T,
                    scalars,
                    levels=ncont,
                    linewidths=0.5,
                    colors="k",
                    alpha=0.5,
                )
                vmin = 0
                vmax = 0.6
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cntr = ax.tricontourf(
                    *points3d[:, :2].T, scalars, levels=ncont, cmap=cmaps[col_type], norm=norm
                )
                ax.triplot(*points3d[:, :2].T, triangulation, color="k", linewidth=0.5)
            elif col_type == "v":
                col_sel = col_options.index(col_type)
                scalars = color_metric[:, col_sel]
                vmin = 0
                vmax = 0.25
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
                vmin = 0
                vmax = 1e4
                norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, clip=True)
                cntr = ax.tripcolor(
                    *points3d[:, :2].T,
                    triangulation,
                    scalars,
                    linewidth=0.5,
                    cmap=cmaps[col_type],
                    norm=norm,
                )
            ax.set_aspect("equal", "box")
            ax.set_xlim(np.min(points3d[:, 0]), np.max(points3d[:, 0]))
            ax.set_ylim(np.min(points3d[:, 1]), np.max(points3d[:, 1]))

            sb.despine(left=True, bottom=True, trim=True)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0.01, hspace=0, wspace=0)
            plt.margins(0, 0)

            if args.save:
                save_path = os.path.join(ds_folders[path_count], "vis")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = col_type + ".pdf"
                filename = os.path.join(save_path, filename)
                fig.savefig(filename)

if not args.save:
    # mlab.show()
    plt.show()
