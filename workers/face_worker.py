from multiprocessing import shared_memory, Process, Queue
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute


class FaceWorker(Process):
    """
    The face worker calculates n_f*.
    """

    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue, random_seed: str):
        """
        Initialize the face worker.
        """
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars' + random_seed)
        self.shm_nf_stars = shared_memory.SharedMemory(name='nf_stars' + random_seed)
        self.shm_u = shared_memory.SharedMemory(name='u' + random_seed)

        super().__init__()

    def run(self):
        """
        Run the face worker (waiting on queue).
        """
        items = self.work_queue.get()
        while type(items) != type(None):
            for i in items:
                self.calculate_face(i)
            self.complete_queue.put(0)
            items = self.work_queue.get()
        self.shm_e_ij_stars.close()
        self.shm_nf_stars.close()
        self.shm_u.close()

    def calculate_face(self, f: int):
        """
        Calculate n_f* for a single face.
        """
        precomputed = self.precomputed
        g = self.g
        ev, fe, L, lambda_value = precomputed.ev, precomputed.fe, precomputed.L, g.lambda_value
        e_ij_stars = np.ndarray(
            (precomputed.ev.shape[0], 3), dtype=np.float64, buffer=self.shm_e_ij_stars.buf)
        nf_stars = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_nf_stars.buf)
        u = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_u.buf)

        g_grad = g.gradient(nf_stars[f, :])
        Hg = g.hessian(nf_stars[f, :])

        r_grad = np.zeros(3)
        Hr = np.zeros([3, 3])
        for j in range(3):
            w_ij = L[ev[fe[f, j], 0], ev[fe[f, j], 1]]
            e_ij_star = e_ij_stars[fe[f, j]]
            r_grad += (lambda_value * w_ij *
                       (e_ij_star.dot(nf_stars[f, :]) + u[f, j]) * e_ij_star)
            Hr += lambda_value * w_ij * \
                (e_ij_star.reshape(-1, 1) * e_ij_star.reshape(1, -1))

        # Newton step
        overall_grad = (r_grad - g_grad)
        overall_grad_newton = sp.linalg.lstsq(Hr - Hg, -overall_grad)[0]

        # project gradient step
        projection = (
            np.eye(3) - nf_stars[f, :].reshape(-1, 1) * nf_stars[f, :].reshape(1, -1))
        step = projection.dot(overall_grad_newton)
        if step.dot(overall_grad) > 0:
            step = -0.1 * projection.dot(overall_grad)

        # Update normal
        nf_stars[f, :] += step
        nf_stars[f, :] /= np.linalg.norm(nf_stars[f, :])
