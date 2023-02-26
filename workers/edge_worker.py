from multiprocessing import shared_memory, Process, Queue
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute


class EdgeWorker(Process):
    """
    The edge worker calculates e*.
    """

    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue):
        """
        Initialize the edge worker.
        """
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
        self.shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
        self.shm_u = shared_memory.SharedMemory(name='u')
        self.shm_U = shared_memory.SharedMemory(name='U')
        super().__init__()

    def run(self):
        """
        Run the edge worker (waiting on queue).
        """
        items = self.work_queue.get()
        while type(items) != type(None):
            for i in items:
                self.calculate_edge(i)
            self.complete_queue.put(0)
            items = self.work_queue.get()
        self.shm_e_ij_stars.close()
        self.shm_nf_stars.close()
        self.shm_u.close()
        self.shm_U.close()

    def calculate_edge(self, e: int):
        """
        Calculate e* for a single edge.
        """
        precomputed = self.precomputed
        g = self.g
        v, ev, fe, ef, mu = precomputed.v, precomputed.ev, precomputed.fe, precomputed.ef, g.lambda_value

        e_ij_stars = np.ndarray(
            (precomputed.ev.shape[0], 3), dtype=np.float64, buffer=self.shm_e_ij_stars.buf)
        nf_stars = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_nf_stars.buf)
        u = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_u.buf)
        U = np.ndarray(
            (precomputed.v.shape[0], 3), dtype=np.float64, buffer=self.shm_U.buf)

        face_1 = ef[e, 0]
        face_2 = ef[e, 1]

        A = np.eye(3) + mu * (sum(nf_stars[f, :].reshape(-1, 1)
                                  * nf_stars[f, :].reshape(1, -1) for f in ef[e]))
        b = U[ev[e, 1], :] - U[ev[e, 0], :] - mu * sum(nf_stars[face_1 if f == 0 else face_2, :] *
                                                       u[face_1 if f == 0 else face_2, i] for f, i in zip(*np.where(fe[(face_1, face_2), :] == e)))

        e_ij_stars[e] = sp.linalg.lstsq(A, b)[0]
