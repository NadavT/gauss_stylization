from multiprocessing import shared_memory, Queue, cpu_count
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute
import igl
from time import time
from random import randint

from workers import FaceWorker, EdgeWorker, RotationWorker


class Calculate:
    """
    Calculates the deformation of a mesh.
    """

    def __init__(self, V: np.array, F: np.array, precomputed: Precompute, g: g_Function):
        """
        Initializes the calculation with a given mesh and precomputed data.
        """
        self.V = V
        self.F = F
        self.precomputed = precomputed
        self.g = g
        self.random_seed = str(randint(0, 10000))

        # Create shared memory for calculations.
        e_ij_stars = np.array([V[self.precomputed.ev[i, 1]] - V[self.precomputed.ev[i, 0]]
                               for i in range(self.precomputed.ev.shape[0])])
        nf_stars = igl.per_face_normals(V, self.F, np.array([1.0, 0.0, 0.0]))
        u = np.zeros([self.precomputed.ev.shape[0], 3])
        rotations = np.zeros([V.shape[0], 3, 3])
        while True:
            try:
                self.shm_e_ij_stars = shared_memory.SharedMemory(
                    create=True, size=e_ij_stars.nbytes, name="e_ij_stars" + self.random_seed)
                break
            except FileExistsError:
                self.random_seed = str(randint(0, 10000))
        self.e_ij_stars_shared = np.ndarray(
            e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=self.shm_e_ij_stars.buf)
        self.shm_nf_stars = shared_memory.SharedMemory(
            create=True, size=nf_stars.nbytes, name="nf_stars" + self.random_seed)
        self.nf_stars_shared = np.ndarray(
            nf_stars.shape, dtype=nf_stars.dtype, buffer=self.shm_nf_stars.buf)
        self.shm_u = shared_memory.SharedMemory(
            create=True, size=u.nbytes, name="u" + self.random_seed)
        self.u_shared = np.ndarray(
            u.shape, dtype=u.dtype, buffer=self.shm_u.buf)
        self.shm_U = shared_memory.SharedMemory(
            create=True, size=V.nbytes, name="U" + self.random_seed)
        self.U_shared = np.ndarray(
            V.shape, dtype=V.dtype, buffer=self.shm_U.buf)
        self.shm_rotations = shared_memory.SharedMemory(
            create=True, size=rotations.nbytes, name="rotations" + self.random_seed)
        self.rotations_shared = np.ndarray(
            rotations.shape, dtype=rotations.dtype, buffer=self.shm_rotations.buf)

        # Create queues for workers.
        self.face_work_queue = Queue()
        self.face_complete_queue = Queue()
        self.edge_work_queue = Queue()
        self.edge_complete_queue = Queue()
        self.rotation_work_queue = Queue()
        self.rotation_complete_queue = Queue()

        # Start workers.
        self.face_workers = [FaceWorker(
            self.precomputed, self.g, self.face_work_queue, self.face_complete_queue, self.random_seed) for _ in range(cpu_count())]
        for worker in self.face_workers:
            worker.start()
        self.edge_workers = [EdgeWorker(
            self.precomputed, self.g, self.edge_work_queue, self.edge_complete_queue, self.random_seed) for _ in range(cpu_count())]
        for worker in self.edge_workers:
            worker.start()
        self.rotation_workers = [RotationWorker(
            self.precomputed, self.g, self.rotation_work_queue, self.rotation_complete_queue, self.random_seed) for _ in range(cpu_count())]
        for worker in self.rotation_workers:
            worker.start()

    def terminate(self):
        """
        Terminates the calculation and cleans up shared memory.
        """
        for _ in self.face_workers:
            self.face_work_queue.put(None)
        for _ in self.edge_workers:
            self.edge_work_queue.put(None)
        for _ in self.rotation_workers:
            self.rotation_work_queue.put(None)

        for worker in self.face_workers:
            worker.join()
        for worker in self.edge_workers:
            worker.join()
        for worker in self.rotation_workers:
            worker.join()

        self.shm_nf_stars.close()
        self.shm_nf_stars.unlink()
        self.shm_e_ij_stars.close()
        self.shm_e_ij_stars.unlink()
        self.shm_u.close()
        self.shm_u.unlink()
        self.shm_U.close()
        self.shm_U.unlink()
        self.shm_rotations.close()
        self.shm_rotations.unlink()

    def single_iteration(self, U: np.array, iterations: int):
        """
        Performs a single iteration.
        """
        # Initialization
        e_ij_stars = np.array([U[self.precomputed.ev[i, 1]] - U[self.precomputed.ev[i, 0]]
                               for i in range(self.precomputed.ev.shape[0])])
        nf_stars = igl.per_face_normals(U, self.F, np.array([1.0, 0.0, 0.0]))
        u = np.zeros([self.precomputed.ev.shape[0], 3])
        self.e_ij_stars_shared[:] = e_ij_stars[:]
        self.nf_stars_shared[:] = nf_stars[:]
        self.u_shared[:] = u[:]
        self.U_shared[:] = U[:]

        # ADMM optimization
        for i in range(iterations):
            # Update nf_stars
            start = time()
            for items in np.array_split(range(self.F.shape[0]), len(self.face_workers)):
                self.face_work_queue.put(items)
            for _ in range(len(self.face_workers)):
                self.face_complete_queue.get()
            print("Face time: ", time() - start)
            start = time()
            for items in np.array_split(range(self.precomputed.ev.shape[0]), len(self.edge_workers)):
                self.edge_work_queue.put(items)
            for _ in range(len(self.edge_workers)):
                self.edge_complete_queue.get()
            print("Edge time: ", time() - start)

            self.e_ij_stars_shared = np.ndarray(
                e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=self.shm_e_ij_stars.buf)

            # Update u
            for f in range(self.F.shape[0]):
                for i in range(self.precomputed.fe[f].shape[0]):
                    e = self.precomputed.fe[f, i]
                    self.u_shared[f, i] += self.e_ij_stars_shared[e].dot(
                        self.nf_stars_shared[f])

        # Calculate part of right hand side of the global step.
        start = time()
        E_target_edges_rhs = np.zeros([self.V.shape[0], 3])
        for e in range(self.precomputed.ev.shape[0]):
            v1 = self.precomputed.ev[e, 0]
            v2 = self.precomputed.ev[e, 1]
            w_ij = self.precomputed.L[v1, v2]
            E_target_edges_rhs[v1, :] -= w_ij * self.e_ij_stars_shared[e]
            E_target_edges_rhs[v2, :] += w_ij * self.e_ij_stars_shared[e]
        print("E_target_edges_rhs time: ", time() - start)

        # ARAP local step
        start = time()
        for items in np.array_split(range(U.shape[0]), len(self.rotation_workers)):
            self.rotation_work_queue.put(items)
        for _ in range(len(self.rotation_workers)):
            self.rotation_complete_queue.get()
        print("Rotation time: ", time() - start)

        self.rotations_shared = np.ndarray(
            self.rotations_shared.shape, dtype=self.rotations_shared.dtype, buffer=self.shm_rotations.buf)

        # ARAP global step
        start = time()
        rotations_as_column = np.array([rot[j, i] for i in range(
            3) for j in range(3) for rot in self.rotations_shared]).reshape(-1, 1)
        arap_B_prod = self.precomputed.arap_rhs.dot(rotations_as_column)
        known = np.array([0], dtype=int)
        known_positions = igl.snap_points(
            np.array([U[self.F[0, 0]]]), self.V)[2]
        for dim in range(self.V.shape[1]):
            B = (arap_B_prod[dim*self.V.shape[0]:(dim+1)*self.V.shape[0]] + self.g.lambda_value *
                 E_target_edges_rhs[:, dim].reshape(-1, 1)) / (1 + self.g.lambda_value)

            new_U = igl.min_quad_with_fixed(
                self.precomputed.L, B, known, np.array([known_positions[dim]]), sp.sparse.csr_matrix((0, 0)), np.array([]), False)
            U[:, dim] = new_U[1]
        print("ARAP time: ", time() - start)
