import igl
import numpy as np


class Precompute:
    def __init__(self, v, f):
        self.precompute(v, f)

    def precompute(self, v, f):
        self.v = v
        self.f = f
        self.ev, self.fe, self.ef = igl.edge_topology(v, f)
        self.L = igl.cotmatrix(v, f)
        adj_f_list, NI = igl.vertex_triangle_adjacency(f, v.shape[0])
        adj_f_list = [adj_f_list[NI[i]:NI[i+1]] for i in range(v.shape[0])]

        self.arap_rhs = igl.arap_rhs(
            v, f, v.shape[1], igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

        adj_f_edges = [self.fe[i] for i in adj_f_list]
        adj_f_vertices = [self.ev[i] for i in adj_f_edges]

        self.adj_f_vertices_flat = [
            adj_f_vertices[i].reshape(-1, 2) for i in range(v.shape[0])]

        self.vertices_w_list = [np.array(
            [[self.L[i, j] for i, j in e] for e in f]).reshape(-1, 1) for f in adj_f_vertices]

        self.adj_edges_deltas_list = [(v[self.adj_f_vertices_flat[i][:, 1]] -
                                       v[self.adj_f_vertices_flat[i][:, 0]]).reshape(-1, 3).transpose() for i in range(v.shape[0])]
