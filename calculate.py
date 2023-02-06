from multiprocessing import Pool, shared_memory, Process, Queue, cpu_count
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute
import igl
from sys import platform


class FaceWorker(Process):
    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue):
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
        self.e_ij_stars = np.ndarray(
            (precomputed.ev.shape[0], 3), dtype=np.float64, buffer=self.shm_e_ij_stars.buf)
        self.shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
        self.nf_stars = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_nf_stars.buf)
        self.shm_u = shared_memory.SharedMemory(name='u')
        self.u = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_u.buf)

        super().__init__()

    def run(self):
        for i in iter(self.work_queue.get, None):
            self.calculate_face(i)
            self.complete_queue.put(i)
        self.shm_e_ij_stars.close()
        self.shm_nf_stars.close()
        self.shm_u.close()

    def calculate_face(self, f: int):
        precomputed = self.precomputed
        g = self.g
        ev, fe, L, lambda_value = precomputed.ev, precomputed.fe, precomputed.L, g.lambda_value
        e_ij_stars = self.e_ij_stars
        nf_stars = self.nf_stars
        u = self.u

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


class EdgeWorker(Process):
    def __init__(self, precomputed: Precompute, g: g_Function, work_queue: Queue, complete_queue: Queue):
        self.precomputed = precomputed
        self.g = g
        self.work_queue = work_queue
        self.complete_queue = complete_queue

        self.shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
        self.e_ij_stars = np.ndarray(
            (precomputed.ev.shape[0], 3), dtype=np.float64, buffer=self.shm_e_ij_stars.buf)
        self.shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
        self.nf_stars = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_nf_stars.buf)
        self.shm_u = shared_memory.SharedMemory(name='u')
        self.u = np.ndarray(
            (precomputed.fe.shape[0], 3), dtype=np.float64, buffer=self.shm_u.buf)
        self.shm_U = shared_memory.SharedMemory(name='U')
        self.U = np.ndarray(
            (precomputed.v.shape[0], 3), dtype=np.float64, buffer=self.shm_U.buf)

        super().__init__()

    def run(self):
        for i in iter(self.work_queue.get, None):
            self.calculate_edge(i)
            self.complete_queue.put(i)
        self.shm_e_ij_stars.close()
        self.shm_nf_stars.close()
        self.shm_u.close()
        self.shm_U.close()

    def calculate_edge(self, e: int):
        precomputed = self.precomputed
        g = self.g
        v, ev, fe, ef, mu = precomputed.v, precomputed.ev, precomputed.fe, precomputed.ef, g.lambda_value

        nf_stars = self.nf_stars
        e_ij_stars = self.e_ij_stars
        u = self.u
        U = self.U

        face_1 = ef[e, 0]
        face_2 = ef[e, 1]

        A = np.eye(3) + mu * (sum(nf_stars[f, :].reshape(-1, 1)
                                  * nf_stars[f, :].reshape(1, -1) for f in ef[e]))
        b = U[ev[e, 1], :] - U[ev[e, 0], :] - mu * sum(nf_stars[face_1 if f == 0 else face_2, :] *
                                                       u[face_1 if f == 0 else face_2, i] for f, i in zip(*np.where(fe[(face_1, face_2), :] == e)))

        e_ij_stars[e] = sp.linalg.lstsq(A, b)[0]


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
        try:
            self.shm_e_ij_stars = shared_memory.SharedMemory(
                create=True, size=e_ij_stars.nbytes, name="e_ij_stars")
        except FileExistsError:
            self.shm_e_ij_stars = shared_memory.SharedMemory(name="e_ij_stars")
        self.e_ij_stars_shared = np.ndarray(
            e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=self.shm_e_ij_stars.buf)
        try:
            self.shm_nf_stars = shared_memory.SharedMemory(
                create=True, size=nf_stars.nbytes, name="nf_stars")
        except FileExistsError:
            self.shm_nf_stars = shared_memory.SharedMemory(name="nf_stars")
        self.nf_stars_shared = np.ndarray(
            nf_stars.shape, dtype=nf_stars.dtype, buffer=self.shm_nf_stars.buf)
        try:
            self.shm_u = shared_memory.SharedMemory(
                create=True, size=u.nbytes, name="u")
        except FileExistsError:
            self.shm_u = shared_memory.SharedMemory(name="u")
        self.u_shared = np.ndarray(
            u.shape, dtype=u.dtype, buffer=self.shm_u.buf)
        try:
            self.shm_U = shared_memory.SharedMemory(
                create=True, size=V.nbytes, name="U")
        except FileExistsError:
            self.shm_U = shared_memory.SharedMemory(name="U")
        self.U_shared = np.ndarray(
            V.shape, dtype=V.dtype, buffer=self.shm_U.buf)

        self.face_work_queue = Queue()
        self.face_complete_queue = Queue()
        self.edge_work_queue = Queue()
        self.edge_complete_queue = Queue()
        self.face_workers = [FaceWorker(
            self.precomputed, self.g, self.face_work_queue, self.face_complete_queue) for _ in range(cpu_count())]
        for worker in self.face_workers:
            worker.start()
        self.edge_workers = [EdgeWorker(
            self.precomputed, self.g, self.edge_work_queue, self.edge_complete_queue) for _ in range(cpu_count())]
        for worker in self.edge_workers:
            worker.start()

    def terminate(self):
        for _ in self.face_workers:
            self.face_work_queue.put(None)
        for _ in self.edge_workers:
            self.edge_work_queue.put(None)

        for worker in self.face_workers:
            worker.terminate()
            worker.join()
        for worker in self.edge_workers:
            worker.terminate()
            worker.join()

        self.shm_nf_stars.close()
        self.shm_nf_stars.unlink()
        self.shm_e_ij_stars.close()
        self.shm_e_ij_stars.unlink()
        self.shm_u.close()
        self.shm_u.unlink()
        self.shm_U.close()
        self.shm_U.unlink()

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
            for i in range(self.F.shape[0]):
                self.face_work_queue.put(i)
            for _ in range(self.F.shape[0]):
                self.face_complete_queue.get()
            for i in range(self.precomputed.ev.shape[0]):
                self.edge_work_queue.put(i)
            for _ in range(self.precomputed.ev.shape[0]):
                self.edge_complete_queue.get()

            # Update u
            for f in range(self.F.shape[0]):
                for i in range(self.precomputed.fe[f].shape[0]):
                    e = self.precomputed.fe[f, i]
                    self.u_shared[f,
                                  i] += self.e_ij_stars_shared[e].dot(self.nf_stars_shared[f])

        E_target_edges_rhs = np.zeros([self.V.shape[0], 3])
        for e in range(self.precomputed.ev.shape[0]):
            v1 = self.precomputed.ev[e, 0]
            v2 = self.precomputed.ev[e, 1]
            w_ij = self.precomputed.L[v1, v2]
            E_target_edges_rhs[v1, :] -= w_ij * self.e_ij_stars_shared[e]
            E_target_edges_rhs[v2, :] += w_ij * self.e_ij_stars_shared[e]

        # ARAP local step
        rotations = []
        for i in range(U.shape[0]):
            edge_starts = U[self.precomputed.adj_f_vertices_flat[i][:, 0]]
            edge_ends = U[self.precomputed.adj_f_vertices_flat[i][:, 1]]

            vertex_rotation_from_original = (self.precomputed.adj_edges_deltas_list[i].dot(
                np.diag(self.precomputed.vertices_w_list[i].flatten()))).dot(edge_ends - edge_starts)

            u, _, vh = np.linalg.svd(vertex_rotation_from_original)
            if np.linalg.det(u.dot(vh)) < 0:
                vh[2, :] *= -1
            rotations.append(u.dot(vh).transpose())

        # ARAP global step
        rotations_as_column = np.array([rot[j, i] for i in range(
            3) for j in range(3) for rot in rotations]).reshape(-1, 1)
        arap_B_prod = self.precomputed.arap_rhs.dot(rotations_as_column)
        for dim in range(self.V.shape[1]):
            B = (arap_B_prod[dim*self.V.shape[0]:(dim+1)*self.V.shape[0]] + self.g.lambda_value *
                 E_target_edges_rhs[:, dim].reshape(-1, 1)) / (1 + self.g.lambda_value)

            known = np.array([0], dtype=int)
            known_positions = np.array([self.V[self.F[0, 0], dim]])

            new_U = igl.min_quad_with_fixed(
                self.precomputed.L, B, known, known_positions, sp.sparse.csr_matrix((0, 0)), np.array([]), False)
            U[:, dim] = new_U[1]
