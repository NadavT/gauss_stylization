import igl
import numpy as np
from time import time


class Precompute:
    """
    Class storing precomputed data.
    """

    def __init__(self, v, f):
        """
        Precompute data for a given mesh.
        """
        self.precompute(v, f)

    def precompute(self, v, f):
        """
        Precompute data for a given mesh.
        """
        self.v = v
        self.f = f
        self.ev, self.fe, self.ef = igl.edge_topology(v, f)
        self.L = igl.cotmatrix(v, f)
        adj_f_list, NI = igl.vertex_triangle_adjacency(f, v.shape[0])
        adj_f_list = [adj_f_list[NI[i]:NI[i+1]] for i in range(v.shape[0])]

        # ARAP precompute.
        self.arap_rhs = igl.arap_rhs(
            v, f, v.shape[1], igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

        adj_f_edges = [self.fe[i] for i in adj_f_list]
        adj_f_vertices = [self.ev[i] for i in adj_f_edges]

        # Precompute for each vertex each adjacent face's vertices and weights.
        self.adj_f_vertices_flat = [
            adj_f_vertices[i].reshape(-1, 2) for i in range(v.shape[0])]

        indexes_x = [f[:, :, 0].reshape(1, -1) for f in adj_f_vertices]
        indexes_y = [f[:, :, 1].reshape(1, -1) for f in adj_f_vertices]
        all_values = self.L[(np.concatenate(indexes_x, axis=1),
                             np.concatenate(indexes_y, axis=1))].todense().transpose()
        self.vertices_w_list = []
        index = 0
        for f in adj_f_vertices:
            self.vertices_w_list.append(
                np.array(all_values[index:index+(f.shape[0] * f.shape[1])].reshape(-1, 1)))
            index += f.shape[0] * f.shape[1]

        # Precompute for each vertex each adjacent face's edges deltas.
        self.adj_edges_deltas_list = [(v[self.adj_f_vertices_flat[i][:, 1]] -
                                       v[self.adj_f_vertices_flat[i][:, 0]]).reshape(-1, 3).transpose() for i in range(v.shape[0])]
