import numpy as np
import scipy as sp
from math import exp


class g_Function:
    def __init__(self, N: list[np.array], R: list[int], R_axis: np.array, sigma: float, mu: float, lambda_value: float, caxiscontrib: float):
        self.N = np.array(N)
        self.R = np.array(R)
        self.R_axis = R_axis
        self.sigma = sigma
        self.mu = mu
        self.lambda_value = lambda_value
        self.caxiscontrib = caxiscontrib if len(R) > 0 else 1
        self._normalize_g_weights()

    def _normalize_g_weights(self):
        if len(self.N) > 0:
            mat = np.array([[exp(self.sigma * self.N[i][:].dot(self.N[j][:]))
                             for j in range(len(self.N))] for i in range(len(self.N))])
            self.N_w = sp.linalg.solve(mat, np.ones(len(self.N)))
        if len(self.R) > 0:
            mat = np.array([[exp(self.sigma * (1 - (self.R[i] - self.R[j])**2))
                             for j in range(len(self.R))] for i in range(len(self.R))])
            self.R_w = sp.linalg.solve(mat, np.ones(len(self.R)))

    def value(self, v: np.array):
        N_value = np.sum(self.N_w * np.exp(self.sigma *
                         v.dot(self.N.transpose())))
        R_value = np.sum(self.R_w * np.exp(self.sigma * (1 -
                         (v.dot(self.R_axis) - self.R)**2))) if len(self.R) > 0 else 0
        return (self.caxiscontrib * N_value + R_value)

    def gradient(self, v: np.array):
        N_grad = np.sum(self.N_w.reshape(-1, 1) * self.N *
                        np.exp(self.sigma * v.dot(self.N.transpose())).reshape((-1, 1)), axis=0)
        R_grad = np.sum(self.R_w * np.repeat(self.R_axis.reshape(-1, 1), 2, axis=1) * -2 * (v.dot(self.R_axis) - self.R)
                        * np.exp(self.sigma * (1 - np.power(v.dot(self.R_axis) - self.R, 2))), axis=1) if len(self.R) > 0 else 0
        return self.sigma * (self.caxiscontrib * N_grad + R_grad)

    def hessian(self, v: np.array):
        N_hess = (self.sigma**2) * np.sum(self.N_w.reshape(-1, 1) * np.exp(self.sigma *
                                                                           v.dot(self.N.transpose())).reshape((-1, 1))) * self.N.transpose().dot(self.N)
        R_hess = np.sum(self.R_w * 2 * self.sigma * (2 * self.sigma * np.power((v.dot(self.R_axis) - self.R), 2) - 1) * np.exp(
            self.sigma * (1 - np.power((v.dot(self.R_axis) - self.R), 2)))) * self.R_axis.reshape((-1, 1)) * self.R_axis
        return (self.caxiscontrib * N_hess + R_hess)
