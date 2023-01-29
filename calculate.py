from multiprocessing import Pool, shared_memory
import numpy as np
import scipy as sp
from g_function import g_Function
from precompute import Precompute
import igl
from functools import partial


precomputed = None
g = None

def calculate_face(f: int):
    ev, fe, L, lambda_value = precomputed.ev, precomputed.fe, precomputed.L, g.lambda_value

    shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
    e_ij_stars = np.ndarray(
        (ev.shape[0], 3), dtype=np.float64, buffer=shm_e_ij_stars.buf)
    shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
    nf_stars = np.ndarray(
        (fe.shape[0], 3), dtype=np.float64, buffer=shm_nf_stars.buf)
    shm_u = shared_memory.SharedMemory(name='u')
    u = np.ndarray((precomputed.fe.shape[0], 3), dtype=np.float64, buffer=shm_u.buf)

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

    shm_e_ij_stars.close()
    shm_nf_stars.close()
    shm_u.close()

def calculate_edge(e: int):
    v, ev, fe, ef, mu = precomputed.v, precomputed.ev, precomputed.fe, precomputed.ef, g.lambda_value

    shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
    e_ij_stars = np.ndarray(
        (ev.shape[0], 3), dtype=np.float64, buffer=shm_e_ij_stars.buf)
    shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
    nf_stars = np.ndarray(
        (fe.shape[0], 3), dtype=np.float64, buffer=shm_nf_stars.buf)
    shm_u = shared_memory.SharedMemory(name='u')
    u = np.ndarray((fe.shape[0], 3), dtype=np.float64, buffer=shm_u.buf)
    shm_U = shared_memory.SharedMemory(name='U')
    U = np.ndarray((v.shape[0], 3), dtype=np.float64, buffer=shm_U.buf)

    face_1 = ef[e, 0]
    face_2 = ef[e, 1]
    # w_ij = L[v1,v2]

    A = np.eye(3)
    rhs = U[ev[e, 1], :] - U[ev[e, 0], :]

    if ef[e, 0] != -1:
        A += mu * nf_stars[face_1,
                           :].reshape(-1, 1) * nf_stars[face_1, :].reshape(1, -1)
        # for k in range(3):
        # 	if fe[f1, k] == e:
        # 		rhs -= mu * nf_stars[f1,:] * u[f1, k]
    if face_2 != -1:
        A += mu * nf_stars[face_2,
                           :].reshape(-1, 1) * nf_stars[face_2, :].reshape(1, -1)
        # for k in range(3):
        # 	if fe[f2, k] == e:
        # 		rhs -= mu * nf_stars[f2,:] * u[f2, k]

    e_ij_stars[e] = sp.linalg.lstsq(A, rhs)[0]

    shm_e_ij_stars.close()
    shm_nf_stars.close()
    shm_u.close()
    shm_U.close()

def single_iteration(V: np.array, U: np.array, F: np.array, iterations: int, precomputed_: Precompute, g_: g_Function):
    # t = time()
    global precomputed
    global g
    precomputed = precomputed_
    g = g_

    # Initialization
    e_ij_stars = np.array([U[precomputed.ev[i, 1]] - U[precomputed.ev[i, 0]]
                          for i in range(precomputed.ev.shape[0])])
    nf_stars = igl.per_face_normals(U, F, np.array([1.0, 0.0, 0.0]))
    u = np.zeros([precomputed.ev.shape[0], 3])

    shm_e_ij_stars = shared_memory.SharedMemory(
        create=True, size=e_ij_stars.nbytes, name="e_ij_stars")
    e_ij_stars_shared = np.ndarray(
        e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=shm_e_ij_stars.buf)
    e_ij_stars_shared[:] = e_ij_stars[:]
    shm_nf_stars = shared_memory.SharedMemory(
        create=True, size=nf_stars.nbytes, name="nf_stars")
    nf_stars_shared = np.ndarray(
        nf_stars.shape, dtype=nf_stars.dtype, buffer=shm_nf_stars.buf)
    nf_stars_shared[:] = nf_stars[:]
    shm_u = shared_memory.SharedMemory(create=True, size=u.nbytes, name="u")
    u_shared = np.ndarray(u.shape, dtype=u.dtype, buffer=shm_u.buf)
    u_shared[:] = u[:]
    shm_U = shared_memory.SharedMemory(create=True, size=U.nbytes, name="U")
    U_shared = np.ndarray(U.shape, dtype=U.dtype, buffer=shm_U.buf)
    U_shared[:] = U[:]

    # ADMM optimization
    for i in range(iterations):
        with Pool() as p:
            # func = partial(calculate_face, precomputed=precomputed, g=g)
            p.map(calculate_face, range(F.shape[0]))
            p.map(calculate_edge, range(precomputed.ev.shape[0]))

        # Update u
        # TODO

    E_target_edges_rhs = np.zeros([V.shape[0], 3])
    for e in range(precomputed.ev.shape[0]):
        v1 = precomputed.ev[e, 0]
        v2 = precomputed.ev[e, 1]
        w_ij = precomputed.L[v1, v2]
        E_target_edges_rhs[v1, :] -= w_ij * e_ij_stars_shared[e]
        E_target_edges_rhs[v2, :] += w_ij * e_ij_stars_shared[e]

    # ARAP local step
    rotations = []
    for i in range(U.shape[0]):
        edge_starts = U[precomputed.adj_f_vertices_flat[i][:, 0]]
        edge_ends = U[precomputed.adj_f_vertices_flat[i][:, 1]]

        vertex_rotation_from_original = (precomputed.adj_edges_deltas_list[i].dot(
            np.diag(precomputed.vertices_w_list[i].flatten()))).dot(edge_ends - edge_starts)

        u, _, vh = np.linalg.svd(vertex_rotation_from_original)
        if np.linalg.det(u.dot(vh)) < 0:
            vh[2, :] *= -1
        rotations.append(u.dot(vh).transpose())

    # print(f"Update local rotations: {time() - t}")
    # t = time()

    # ARAP global step
    rotations_as_column = np.array([rot[j, i] for i in range(
        3) for j in range(3) for rot in rotations]).reshape(-1, 1)
    arap_B_prod = precomputed.arap_rhs.dot(rotations_as_column)
    for dim in range(V.shape[1]):
        B = (arap_B_prod[dim*V.shape[0]:(dim+1)*V.shape[0]] + g.lambda_value *
             E_target_edges_rhs[:, dim].reshape(-1, 1)) / (1 + g.lambda_value)

        known = np.array([0], dtype=int)
        known_positions = np.array([V[F[0, 0], dim]])

        new_U = igl.min_quad_with_fixed(
            precomputed.L, B, known, known_positions, sp.sparse.csr_matrix((0, 0)), np.array([]), False)
        U[:, dim] = new_U[1]

    # print(f"Update global rotations: {time() - t}")
    shm_nf_stars.close()
    shm_nf_stars.unlink()
    shm_e_ij_stars.close()
    shm_e_ij_stars.unlink()
    shm_u.close()
    shm_u.unlink()
    shm_U.close()
    shm_U.unlink()
