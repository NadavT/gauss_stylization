from multiprocessing import shared_memory, Queue, cpu_count
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute
import igl

from workers import FaceWorker, EdgeWorker, RotationWorker, TargetEdgeWorker


class Calculate:
    def __init__(self, V: np.array, F: np.array, precomputed: Precompute, g: g_Function):
        self.V = V
        self.F = F
        self.precomputed = precomputed
        self.g = g

        e_ij_stars = np.array([V[self.precomputed.ev[i, 1]] - V[self.precomputed.ev[i, 0]]
                               for i in range(self.precomputed.ev.shape[0])])
        nf_stars = igl.per_face_normals(V, self.F, np.array([1.0, 0.0, 0.0]))
        u = np.zeros([self.precomputed.ev.shape[0], 3])
        E_target_edges_rhs = np.zeros([self.V.shape[0], 3])
        rotations = np.zeros([V.shape[0], 3, 3])
        try:
            self.shm_e_ij_stars = shared_memory.SharedMemory(
                create=True, size=e_ij_stars.nbytes, name="e_ij_stars")
        except FileExistsError:
            self.shm_e_ij_stars = shared_memory.SharedMemory(name="e_ij_stars")
            self.shm_e_ij_stars.unlink()
            self.shm_e_ij_stars = shared_memory.SharedMemory(
                create=True, size=e_ij_stars.nbytes, name="e_ij_stars")
        self.e_ij_stars_shared = np.ndarray(
            e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=self.shm_e_ij_stars.buf)
        try:
            self.shm_nf_stars = shared_memory.SharedMemory(
                create=True, size=nf_stars.nbytes, name="nf_stars")
        except FileExistsError:
            self.shm_nf_stars = shared_memory.SharedMemory(name="nf_stars")
            self.shm_nf_stars.unlink()
            self.shm_nf_stars = shared_memory.SharedMemory(
                create=True, size=nf_stars.nbytes, name="nf_stars")
        self.nf_stars_shared = np.ndarray(
            nf_stars.shape, dtype=nf_stars.dtype, buffer=self.shm_nf_stars.buf)
        try:
            self.shm_u = shared_memory.SharedMemory(
                create=True, size=u.nbytes, name="u")
        except FileExistsError:
            self.shm_u = shared_memory.SharedMemory(name="u")
            self.shm_u.unlink()
            self.shm_u = shared_memory.SharedMemory(
                create=True, size=u.nbytes, name="u")
        self.u_shared = np.ndarray(
            u.shape, dtype=u.dtype, buffer=self.shm_u.buf)
        try:
            self.shm_U = shared_memory.SharedMemory(
                create=True, size=V.nbytes, name="U")
        except FileExistsError:
            self.shm_U = shared_memory.SharedMemory(name="U")
            self.shm_U.unlink()
            self.shm_U = shared_memory.SharedMemory(
                create=True, size=V.nbytes, name="U")
        self.U_shared = np.ndarray(
            V.shape, dtype=V.dtype, buffer=self.shm_U.buf)
        try:
            self.shm_E_target_edges_rhs = shared_memory.SharedMemory(
                create=True, size=E_target_edges_rhs.nbytes, name="E_target_edges_rhs")
        except FileExistsError:
            self.shm_E_target_edges_rhs = shared_memory.SharedMemory(
                name="E_target_edges_rhs")
            self.shm_E_target_edges_rhs.unlink()
            self.shm_E_target_edges_rhs = shared_memory.SharedMemory(
                create=True, size=V.nbytes, name="E_target_edges_rhs")
        self.E_target_edges_rhs_shared = np.ndarray(
            E_target_edges_rhs.shape, dtype=E_target_edges_rhs.dtype, buffer=self.shm_U.buf)
        try:
            self.shm_rotations = shared_memory.SharedMemory(
                create=True, size=rotations.nbytes, name="rotations")
        except FileExistsError:
            self.shm_rotations = shared_memory.SharedMemory(name="rotations")
            self.shm_rotations.unlink()
            self.shm_rotations = shared_memory.SharedMemory(
                create=True, size=rotations.nbytes, name="rotations")
        self.rotations_shared = np.ndarray(
            rotations.shape, dtype=rotations.dtype, buffer=self.shm_rotations.buf)

        self.face_work_queue = Queue()
        self.face_complete_queue = Queue()
        self.edge_work_queue = Queue()
        self.edge_complete_queue = Queue()
        self.target_edges_rhs_work_queue = Queue()
        self.target_edges_rhs_complete_queue = Queue()
        self.rotation_work_queue = Queue()
        self.rotation_complete_queue = Queue()

        self.face_workers = [FaceWorker(
            self.precomputed, self.g, self.face_work_queue, self.face_complete_queue) for _ in range(cpu_count())]
        for worker in self.face_workers:
            worker.start()
        self.edge_workers = [EdgeWorker(
            self.precomputed, self.g, self.edge_work_queue, self.edge_complete_queue) for _ in range(cpu_count())]
        for worker in self.edge_workers:
            worker.start()
        self.target_edges_rhs_workers = [TargetEdgeWorker(
            self.precomputed, self.g, self.target_edges_rhs_work_queue, self.target_edges_rhs_complete_queue) for _ in range(cpu_count())]
        for worker in self.target_edges_rhs_workers:
            worker.start()
        self.rotation_workers = [RotationWorker(
            self.precomputed, self.g, self.rotation_work_queue, self.rotation_complete_queue) for _ in range(cpu_count())]
        for worker in self.rotation_workers:
            worker.start()

    def terminate(self):
        for _ in self.face_workers:
            self.face_work_queue.put(None)
        for _ in self.edge_workers:
            self.edge_work_queue.put(None)
        for _ in self.target_edges_rhs_workers:
            self.target_edges_rhs_work_queue.put(None)
        for _ in self.rotation_workers:
            self.rotation_work_queue.put(None)

        for worker in self.face_workers:
            worker.join()
        for worker in self.edge_workers:
            worker.join()
        for worker in self.target_edges_rhs_workers:
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
        self.shm_E_target_edges_rhs.close()
        self.shm_E_target_edges_rhs.unlink()
        self.shm_rotations.close()
        self.shm_rotations.unlink()

    def single_iteration(self, U: np.array, iterations: int):
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
            for items in np.array_split(range(self.F.shape[0]), len(self.face_workers)):
                self.face_work_queue.put(items)
            for _ in range(len(self.face_workers)):
                self.face_complete_queue.get()
            for items in np.array_split(range(self.precomputed.ev.shape[0]), len(self.edge_workers)):
                self.edge_work_queue.put(items)
            for _ in range(len(self.edge_workers)):
                self.edge_complete_queue.get()

            self.e_ij_stars_shared = np.ndarray(
                e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=self.shm_e_ij_stars.buf)

            # Update u
            for f in range(self.F.shape[0]):
                for i in range(self.precomputed.fe[f].shape[0]):
                    e = self.precomputed.fe[f, i]
                    self.u_shared[f,
                                  i] += self.e_ij_stars_shared[e].dot(self.nf_stars_shared[f])

        self.E_target_edges_rhs_shared = np.zeros([self.V.shape[0], 3])
        for items in np.array_split(range(self.precomputed.ev.shape[0]), len(self.target_edges_rhs_workers)):
            self.target_edges_rhs_work_queue.put(items)
        for _ in range(len(self.target_edges_rhs_workers)):
            self.target_edges_rhs_complete_queue.get()

        self.E_target_edges_rhs_shared = np.ndarray(
            self.E_target_edges_rhs_shared.shape, dtype=self.E_target_edges_rhs_shared.dtype, buffer=self.shm_E_target_edges_rhs.buf)

        # ARAP local step
        for items in np.array_split(range(U.shape[0]), len(self.rotation_workers)):
            self.rotation_work_queue.put(items)
        for _ in range(len(self.rotation_workers)):
            self.rotation_complete_queue.get()

        self.rotations_shared = np.ndarray(
            self.rotations_shared.shape, dtype=self.rotations_shared.dtype, buffer=self.shm_rotations.buf)
        # for i in range(U.shape[0]):
        #     edge_starts = U[self.precomputed.adj_f_vertices_flat[i][:, 0]]
        #     edge_ends = U[self.precomputed.adj_f_vertices_flat[i][:, 1]]

        #     vertex_rotation_from_original = (self.precomputed.adj_edges_deltas_list[i].dot(
        #         np.diag(self.precomputed.vertices_w_list[i].flatten()))).dot(edge_ends - edge_starts)

        #     u, _, vh = np.linalg.svd(vertex_rotation_from_original)
        #     if np.linalg.det(u.dot(vh)) < 0:
        #         vh[2, :] *= -1
        #     self.rotations_shared[i] = u.dot(vh).transpose()

        # ARAP global step
        rotations_as_column = np.array([rot[j, i] for i in range(
            3) for j in range(3) for rot in self.rotations_shared]).reshape(-1, 1)
        arap_B_prod = self.precomputed.arap_rhs.dot(rotations_as_column)
        for dim in range(self.V.shape[1]):
            B = (arap_B_prod[dim*self.V.shape[0]:(dim+1)*self.V.shape[0]] + self.g.lambda_value *
                 self.E_target_edges_rhs_shared[:, dim].reshape(-1, 1)) / (1 + self.g.lambda_value)

            known = np.array([0], dtype=int)
            known_positions = np.array([self.V[self.F[0, 0], dim]])

            new_U = igl.min_quad_with_fixed(
                self.precomputed.L, B, known, known_positions, sp.sparse.csr_matrix((0, 0)), np.array([]), False)
            U[:, dim] = new_U[1]
