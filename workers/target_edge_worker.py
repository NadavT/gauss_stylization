from multiprocessing import shared_memory, Process, Queue
import numpy as np
from g_function import g_Function
from precompute import Precompute


class TargetEdgeWorker(Process):
    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue):
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
        self.shm_E_target_edges_rhs = shared_memory.SharedMemory(
            name='E_target_edges_rhs')
        super().__init__()

    def run(self):
        items = self.work_queue.get()
        while type(items) != type(None):
            for i in items:
                self.calculate_target_edges(i)
            self.complete_queue.put(0)
            items = self.work_queue.get()
        self.shm_e_ij_stars.close()
        self.shm_E_target_edges_rhs.close()

    def calculate_target_edges(self, e: int):
        e_ij_stars = np.ndarray(
            (self.precomputed.ev.shape[0], 3), dtype=np.float64, buffer=self.shm_e_ij_stars.buf)
        E_target_edges_rhs = np.ndarray(
            (self.precomputed.v.shape[0], 3), dtype=np.float64, buffer=self.shm_E_target_edges_rhs.buf)

        v1 = self.precomputed.ev[e, 0]
        v2 = self.precomputed.ev[e, 1]
        w_ij = self.precomputed.L[v1, v2]
        E_target_edges_rhs[v1, :] -= w_ij * e_ij_stars[e]
        E_target_edges_rhs[v2, :] += w_ij * e_ij_stars[e]
