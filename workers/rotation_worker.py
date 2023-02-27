from multiprocessing import shared_memory, Process, Queue
import numpy as np
from g_function import g_Function
from precompute import Precompute


class RotationWorker(Process):
    """
    The rotation worker calculates the rotation matrix for each vertex.
    """

    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue, random_seed: str):
        """
        Initialize the rotation worker.
        """
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_U = shared_memory.SharedMemory(name='U' + random_seed)
        self.shm_rotations = shared_memory.SharedMemory(
            name='rotations' + random_seed)
        super().__init__()

    def run(self):
        """
        Run the rotation worker (waiting on queue).
        """
        items = self.work_queue.get()
        while type(items) != type(None):
            for i in items:
                self.calculate_rotation(i)
            self.complete_queue.put(0)
            items = self.work_queue.get()
        self.shm_U.close()
        self.shm_rotations.close()

    def calculate_rotation(self, v: int):
        """
        Calculate the rotation matrix for a single vertex.
        """
        U = np.ndarray(
            (self.precomputed.v.shape[0], 3), dtype=np.float64, buffer=self.shm_U.buf)
        rotations = np.ndarray(
            (self.precomputed.v.shape[0], 3, 3), dtype=np.float64, buffer=self.shm_rotations.buf)

        edge_starts = U[self.precomputed.adj_f_vertices_flat[v][:, 0]]
        edge_ends = U[self.precomputed.adj_f_vertices_flat[v][:, 1]]

        vertex_rotation_from_original = (self.precomputed.adj_edges_deltas_list[v].dot(
            np.diag(self.precomputed.vertices_w_list[v].flatten()))).dot(edge_ends - edge_starts)

        u, _, vh = np.linalg.svd(vertex_rotation_from_original)
        if np.linalg.det(u.dot(vh)) < 0:
            vh[2, :] *= -1
        rotations[v] = u.dot(vh).transpose()
