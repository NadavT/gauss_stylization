import numpy as np
import scipy as sp
from math import exp


class g_Function:
    def __init__(self, N: list[np.array], sigma: float, mu: float, lambda_value: float):
        self.N = N
        self.sigma = sigma
        self.mu = mu
        self.lambda_value = lambda_value
        self._normalize_g_weights()

    def _normalize_g_weights(self):
        mat = np.zeros((len(self.N), len(self.N)))
        mat = np.array([[exp(self.sigma * self.N[i][:].dot(self.N[j][:]))
                       for j in range(len(self.N))] for i in range(len(self.N))])
        self.w = sp.linalg.solve(mat, np.ones(len(self.N)))

    def value(self, v: np.array):
        return sum(self.w[i] * exp(self.sigma * v.dot(self.N[i])) for i in range(len(self.N)))

    def gradient(self, v: np.array):
        return self.sigma * sum(self.w[i] * self.N[i] * exp(self.sigma * v.dot(self.N[i])) for i in range(len(self.N)))

    def hessian(self, v: np.array):
        return (self.sigma**2) * \
            sum(
            self.w[i] * exp(self.sigma * v.dot(self.N[i])) *
            self.N[i].reshape((-1, 1)) * self.N[i]
            for i in range(len(self.N)))
