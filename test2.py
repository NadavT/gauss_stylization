import igl
import scipy as sp
import numpy as np
import meshplot as mp
from math import exp

import os
root_folder = os.getcwd()

def normalize_unitbox(v):
	v = v - v.min()
	v = v / v.max()
	return v

## Load a mesh in OFF format
sphere_v, sphere_f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "sphere_s3.off"))
model_v, model_f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "cat_s3.off"))
# model_v, model_f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "sphere.obj"))

model_v = normalize_unitbox(model_v)

# Get topology and cot laplacian
ev, fe, ef = igl.edge_topology(model_v, model_f)
L = igl.cotmatrix(model_v, model_f)
adj_f_list, NI = igl.vertex_triangle_adjacency(model_f, model_v.shape[0])
adj_f_list = [adj_f_list[NI[i]:NI[i+1]] for i in range(model_v.shape[0])]

arap_rhs = igl.arap_rhs(model_v, model_f, model_v.shape[1], igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

adj_f_edges = [fe[i] for i in adj_f_list]
adj_f_vertices = [ev[i] for i in adj_f_edges]

adj_f_vertices_flat = [adj_f_vertices[i].reshape(-1, 2) for i in range(model_v.shape[0])]

vertices_w_list = [np.array([[L[i, j] for i, j in e] for e in f]).reshape(-1,1) for f in adj_f_vertices]

adj_edges_deltas_list = [(model_v[adj_f_vertices_flat[i][:,1]] - model_v[adj_f_vertices_flat[i][:,0]]).reshape(-1,3).transpose() for i in range(model_v.shape[0])]

def normalize_g_weights(N: list[np.array], sigma: float):
	mat = np.zeros((len(N), len(N)))
	mat = np.array([[exp(sigma * N[i][:].dot(N[j][:])) for j in range(len(N))] for i in range(len(N))])
	return sp.linalg.solve(mat, np.ones(len(N)))

def g_value(v, N, w, sigma):
	return sum(w[i] * exp(sigma * v.dot(N[i])) for i in range(len(N)))

def g_gradient(v, N, w, sigma):
	return sigma * sum(w[i] * N[i] * exp(sigma * v.dot(N[i])) for i in range(len(N)))

def g_hessian(v, N, w, sigma):
	return (sigma**2) * sum(w[i] * exp(sigma * v.dot(N[i])) * N[i].reshape((-1,1)) * N[i] for i in range(len(N)))

N = [
		np.array([1,0,0]), np.array([-1,0,0]),
		np.array([0,1,0]), np.array([0,-1,0]),
		np.array([0,0,1]), np.array([0,0,-1]),
	]
sigma = 8
mu = 1
lambda_value = 4
w = normalize_g_weights(N, sigma)

from multiprocessing import Pool, shared_memory
from time import time
from functools import partial

def calculate_face(f, a):
    shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
    e_ij_stars = np.ndarray((ev.shape[0], 3), dtype=np.float64, buffer=shm_e_ij_stars.buf)
    shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
    nf_stars = np.ndarray((model_f.shape[0], 3), dtype=np.float64, buffer=shm_nf_stars.buf)
    shm_u = shared_memory.SharedMemory(name='u')
    u = np.ndarray((model_f.shape[0], 3), dtype=np.float64, buffer=shm_u.buf)

    g_grad = g_gradient(nf_stars[f,:], N, w, sigma)
    Hg = g_hessian(nf_stars[f,:], N, w, sigma)

    r_grad = np.zeros(3)
    Hr = np.zeros([3,3])
    for j in range(3):
        w_ij = L[ev[fe[f, j], 0],ev[fe[f, j], 1]]
        e_ij_star = e_ij_stars[fe[f,j]]
        r_grad += (lambda_value * w_ij * (e_ij_star.dot(nf_stars[f,:]) + u[f, j]) * e_ij_star)
        Hr += lambda_value * w_ij * (e_ij_star.reshape(-1,1) * e_ij_star.reshape(1,-1))

    # Newton step
    overall_grad = (r_grad - g_grad)
    overall_grad_newton = sp.linalg.lstsq(Hr - Hg, -overall_grad)[0]

    # project gradient step
    projection = (np.eye(3) - nf_stars[f,:].reshape(-1,1) * nf_stars[f,:].reshape(1,-1))
    step = projection.dot(overall_grad_newton)
    if step.dot(overall_grad) > 0:
        step = -0.1 * projection.dot(overall_grad)
        
    # Update normal
    nf_stars[f,:] += step
    nf_stars[f,:] /= np.linalg.norm(nf_stars[f,:])

    shm_e_ij_stars.close()
    shm_nf_stars.close()
    shm_u.close()


def calculate_edge(e, a):
    shm_e_ij_stars = shared_memory.SharedMemory(name='e_ij_stars')
    e_ij_stars = np.ndarray((ev.shape[0], 3), dtype=np.float64, buffer=shm_e_ij_stars.buf)
    shm_nf_stars = shared_memory.SharedMemory(name='nf_stars')
    nf_stars = np.ndarray((model_f.shape[0], 3), dtype=np.float64, buffer=shm_nf_stars.buf)
    shm_u = shared_memory.SharedMemory(name='u')
    u = np.ndarray((model_f.shape[0], 3), dtype=np.float64, buffer=shm_u.buf)
    shm_U = shared_memory.SharedMemory(name='U')
    U = np.ndarray((model_v.shape[0], 3), dtype=np.float64, buffer=shm_U.buf)

    face_1 = ef[e, 0]
    face_2 = ef[e, 1]
    # w_ij = L[v1,v2]

    A = np.eye(3)
    rhs = U[ev[e,1],:] - U[ev[e,0],:]

    if ef[e,0] != -1:
        A += mu * nf_stars[face_1,:].reshape(-1,1) * nf_stars[face_1,:].reshape(1,-1)
        # for k in range(3):
        # 	if fe[f1, k] == e:
        # 		rhs -= mu * nf_stars[f1,:] * u[f1, k]
    if face_2 != -1:
        A += mu * nf_stars[face_2,:].reshape(-1,1) * nf_stars[face_2,:].reshape(1,-1)
        # for k in range(3):
        # 	if fe[f2, k] == e:
        # 		rhs -= mu * nf_stars[f2,:] * u[f2, k]

    e_ij_stars[e] = sp.linalg.lstsq(A, rhs)[0]


    shm_e_ij_stars.close()
    shm_nf_stars.close()
    shm_u.close()
    shm_U.close()

def single_iteration(V, U, F, iterations):
    # t = time()

    # Initialization
    e_ij_stars = np.array([U[ev[i,1]] - U[ev[i,0]] for i in range(ev.shape[0])])
    nf_stars = igl.per_face_normals(U, F, np.array([1.0,0.0,0.0]))
    u = np.zeros([ev.shape[0], 3])

    shm_e_ij_stars = shared_memory.SharedMemory(create=True, size=e_ij_stars.nbytes, name="e_ij_stars")
    e_ij_stars_shared = np.ndarray(e_ij_stars.shape, dtype=e_ij_stars.dtype, buffer=shm_e_ij_stars.buf)
    e_ij_stars_shared[:] = e_ij_stars[:]
    shm_nf_stars = shared_memory.SharedMemory(create=True, size=nf_stars.nbytes, name="nf_stars")
    nf_stars_shared = np.ndarray(nf_stars.shape, dtype=nf_stars.dtype, buffer=shm_nf_stars.buf)
    nf_stars_shared[:] = nf_stars[:]
    shm_u = shared_memory.SharedMemory(create=True, size=u.nbytes, name="u")
    u_shared = np.ndarray(u.shape, dtype=u.dtype, buffer=shm_u.buf)
    u_shared[:] = u[:]
    shm_U = shared_memory.SharedMemory(create=True, size=U.nbytes, name="U")
    U_shared = np.ndarray(U.shape, dtype=U.dtype, buffer=shm_U.buf)
    U_shared[:] = U[:]
    
    # print(f"Initialization time: {time() - t}")
    # t = time()

    # ADMM optimization
    for i in range(iterations):
        with Pool() as p:
            func = partial(calculate_face, a=nf_stars)
            p.map(func, range(F.shape[0]))
        
        # print(f"Update face normals stars: {time() - t}")
        # t = time()
            func = partial(calculate_edge, a=nf_stars)
            p.map(func, range(ev.shape[0]))

        # print(f"Update edge stars: {time() - t}")
        # t = time()

        # Update u
        # TODO

    E_target_edges_rhs = np.zeros([V.shape[0], 3])
    for e in range(ev.shape[0]):
        v1 = ev[e, 0]
        v2 = ev[e, 1]
        w_ij = L[v1,v2]
        E_target_edges_rhs[v1,:] -= w_ij * e_ij_stars_shared[e]
        E_target_edges_rhs[v2,:] += w_ij * e_ij_stars_shared[e]

    # print(f"Update target edges: {time() - t}")
    # t = time()

    # ARAP local step
    rotations = []
    for i in range(U.shape[0]):
        edge_starts = U[adj_f_vertices_flat[i][:,0]]
        edge_ends = U[adj_f_vertices_flat[i][:,1]]

        vertex_rotation_from_original = (adj_edges_deltas_list[i].dot(np.diag(vertices_w_list[i].flatten()))).dot(edge_ends - edge_starts)

        u, s, vh = np.linalg.svd(vertex_rotation_from_original)
        if np.linalg.det(u.dot(vh)) < 0:
            vh[2,:] *= -1
        rotations.append(u.dot(vh).transpose())

    # print(f"Update local rotations: {time() - t}")
    # t = time()

    # ARAP global step
    rotations_as_column = np.array([rot[j,i] for i in range(3) for j in range(3) for rot in rotations]).reshape(-1,1)
    arap_B_prod = arap_rhs.dot(rotations_as_column)
    for dim in range(V.shape[1]):
        B = (arap_B_prod[dim*V.shape[0]:(dim+1)*V.shape[0]] + lambda_value * E_target_edges_rhs[:,dim].reshape(-1,1)) / (1 + lambda_value)

        known = np.array([0], dtype=int)
        known_positions = np.array([V[F[0,0],dim]])

        new_U = igl.min_quad_with_fixed(L, B, known, known_positions, sp.sparse.csr_matrix((0,0)), np.array([]), False)
        U[:,dim] = new_U[1]
    
    # print(f"Update global rotations: {time() - t}")
    shm_nf_stars.close()
    shm_nf_stars.unlink()
    shm_e_ij_stars.close()
    shm_e_ij_stars.unlink()
    shm_u.close()
    shm_u.unlink()
    shm_U.close()
    shm_U.unlink()

import copy
U = copy.deepcopy(model_v)
single_iteration(model_v, U, model_f, 1)