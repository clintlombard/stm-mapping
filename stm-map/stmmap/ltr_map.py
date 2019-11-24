import ast
import itertools
import logging

import matplotlib.pylab as plt
import networkx as nx
import numpy as np

from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)

try:
    fig = plt.figure()
    plt.close(fig)
except:
    import matplotlib

    matplotlib.use("Agg")


class LTRMap:
    def __init__(self, slam_ts, dirs, check_triangulation=False, manual_lms=[]):
        self._triangulate_ltrs(slam_ts, dirs, check_triangulation, manual_lms)
        self._get_neighbours()
        self.N_LTRS = len(self.cliques)

    def _triangulate_ltrs(self, slam_ts, dirs, check_triangulation, manual_lms):
        """
         Triangulate the landmarks to define the LTRs.
        """

        # Isolate the landmarks from mean.
        last_estimate = slam_ts[-1]
        filename = "mean-" + last_estimate[1] + ".npy"
        last_mean = np.load(dirs["slam"] + filename)
        dof = 7  # robot state dof
        last_lms = last_mean[dof:]
        N_lm = int((last_mean.size - dof) / 3)
        lm_ids = np.arange(N_lm)

        # Find only the landmarks which were observed
        valid_lms = []
        for i in range(N_lm):
            if last_lms[3 * i] != 0:
                valid_lms.append(i)

        valid_lms_arr = np.array(valid_lms)

        last_lms = last_lms.reshape((int(last_lms.size / 3), 3))
        last_lms = last_lms[:, :2]
        not_happy = True

        if check_triangulation:
            print("Valid landmarks:", valid_lms)
            plt.ion()
            not_happy = True
        else:
            plt.ioff()
        fig = plt.figure()
        plt.axis("equal")
        while not_happy:
            last_lms_valid = last_lms[valid_lms, :]  # Remove invalids
            if (last_lms_valid.size / 2) < 3:
                logger.error(f"Too few landmarks in SLAM estimates: {last_lms_valid.size / 2} < 3")
                exit()

            if manual_lms != []:
                for tri in manual_lms:
                    if len(tri) != 3:
                        raise ValueError(
                            "Invalid manual landmarks. Incorrect size: %s" % str(manual_lms)
                        )
                valid_lms_set = set([i for i in valid_lms])
                manual_lms_set = set([j for i in manual_lms for j in i])
                if not (manual_lms_set <= valid_lms_set):
                    raise ValueError(
                        "Invalid manual landmarks. Not subset of valid landmark set. %s !<= %s"
                        % (str(manual_lms_set), str(valids_lms_set))
                    )

                self.simplices = manual_lms
                not_happy = False
            elif (last_lms_valid.size / 2) == 3:
                # If there are only 3 landmarks don't need user input
                self.simplices = np.array([valid_lms])
                not_happy = False
            else:
                triangles = Delaunay(last_lms_valid, incremental=True)
                self.simplices = triangles.simplices  # Indices of the points in each triangulation
                # Remap simplices to valid landmark ids
                remap = lambda x: valid_lms_arr[x]
                self.simplices = np.apply_along_axis(remap, 0, self.simplices)

            # Visual check for triangulation
            plt.gca().clear()
            plt.triplot(last_lms[:, 0], last_lms[:, 1], self.simplices.copy())
            for i in valid_lms:
                plt.text(*last_lms[i, :], s=str(i))

            if check_triangulation and not_happy:
                plt.draw()
                plt.pause(0.01)
                remove_str = input("Enter the IDs of landmarks to be removed (comma seperated): ")
                try:
                    remove = ast.literal_eval(remove_str)
                except Exception:
                    logger.exception("Error understanding input")
                    remove = ()
                    not_happy = False

                # If only one number entered
                if type(remove) is int:
                    remove = (remove,)
                new_valid = sorted(list(set(valid_lms) - set(remove)))
                valid_lms = new_valid
                valid_lms_arr = np.array(valid_lms)
            else:
                break
        plt.savefig(dirs["main"] + "/triangulation.pdf")
        plt.close(fig)
        plt.ioff()

    def _get_neighbours(self):
        """
        Find neighbouring LTRs.
        """
        # Calculate neighbouring triangle dictionary
        G = nx.Graph()

        for s in self.simplices:
            G.add_nodes_from(s)
            G.add_cycle(s)

        # Must make a copy into a list
        clique_graph = nx.find_cliques(G)
        self.cliques = []
        for c in clique_graph:
            self.cliques.append(tuple(sorted(c)))

        self.neighbours = dict()
        for c in self.cliques:
            neigh_cliques = []
            for edge in itertools.combinations(c, 2):
                neigh = nx.common_neighbors(G, *edge)
                for i in neigh:
                    if i not in c:
                        # find which clique it is in
                        for c_tmp in self.cliques:
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
                self.neighbours[tuple(c)] = [region_a, region_b, region_ab]
