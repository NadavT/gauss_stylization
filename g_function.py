import numpy as np
import scipy as sp
from math import exp


class g_Function:
    def __init__(self, N: list[np.array], R: list[int], R_axis: np.array, sigma: float, mu: float, lambda_value: float, caxiscontrib: float):
        self.N = N
        self.R = R
        self.R_axis = R_axis
        self.sigma = sigma
        self.mu = mu
        self.lambda_value = lambda_value
        self.caxiscontrib = caxiscontrib
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
        return self.caxiscontrib * sum(self.N_w[i] * exp(self.sigma * v.dot(self.N[i])) for i in range(len(self.N))) + \
            sum(self.R_w[i] * exp(self.sigma * (1 - (v.dot(self.R_axis) - self.R[i])**2))
                for i in range(len(self.R)))

    def gradient(self, v: np.array):
        return self.sigma * (self.caxiscontrib * sum(self.N_w[i] * self.N[i] * exp(self.sigma * v.dot(self.N[i])) for i in range(len(self.N))) +
                             sum(self.R_w[i] * self.R_axis * -2 * (v.dot(self.R_axis) - self.R[i]) * exp(self.sigma * (1 - (v.dot(self.R_axis) - self.R[i])**2)) for i in range(len(self.R))))

    def hessian(self, v: np.array):
        return (
            (self.sigma**2) * sum(
                self.N_w[i] * exp(self.sigma * v.dot(self.N[i])) *
                self.N[i].reshape((-1, 1)) * self.N[i]
                for i in range(len(self.N))) +
            sum(
                self.R_w[i] * 2 * self.sigma * (2 * self.sigma * (v.dot(self.R_axis) - self.R[i])**2 - 1) * exp(
                    self.sigma * (1 - (v.dot(self.R_axis) - self.R[i])**2)) * self.R_axis.reshape((-1, 1)) * self.R_axis
                for i in range(len(self.R))
            ))
