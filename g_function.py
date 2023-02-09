import numpy as np
import scipy as sp
from math import exp


class g_Function:
    def __init__(self, N: list[np.array], R: list[int], R_axis: list[np.array], sigma: float, mu: float, lambda_value: float, caxiscontrib: float):
        self.N = np.array(N).transpose()
        self.R = np.array(R)
        self.R_axis = np.array(R_axis).transpose()
        self.sigma = sigma
        self.mu = mu
        self.lambda_value = lambda_value
        self.caxiscontrib = caxiscontrib if len(R) > 0 else 1
        self._normalize_g_weights()

    def _normalize_g_weights(self):
        if len(self.N) > 0:
            mat = np.exp(self.sigma * self.N.transpose().dot(self.N))
            self.N_w = sp.linalg.solve(mat, np.ones(self.N.shape[1]))
        if len(self.R) > 0:
            mat = np.exp(
                self.sigma * (1 - (self.R - self.R.reshape(-1, 1))**2))
            self.R_w = sp.linalg.solve(mat, np.ones(len(self.R)))

    def value(self, v: np.array):
        N_value = np.sum(self.N_w * np.exp(self.sigma *
                         v.dot(self.N)))
        R_value = np.sum(self.R_w * np.exp(self.sigma * (1 -
                         (v.dot(self.R_axis) - self.R)**2))) if len(self.R) > 0 else 0
        return (self.caxiscontrib * N_value + R_value)

    def gradient(self, v: np.array):
        N_grad = np.sum(self.N_w.reshape(-1, 1) * self.N.transpose() *
                        np.exp(self.sigma * v.dot(self.N)).reshape((-1, 1)), axis=0)
        R_grad = np.sum(self.R_w * self.R_axis * -2 * (v.dot(self.R_axis) - self.R)
                        * np.exp(self.sigma * (1 - np.power(v.dot(self.R_axis) - self.R, 2))), axis=1) if len(self.R) > 0 else 0
        return self.sigma * (self.caxiscontrib * N_grad + R_grad)

    def hessian(self, v: np.array):
        modified_N = np.sqrt(
            self.N_w * np.exp(self.sigma * v.dot(self.N))) * self.N
        modified_R = np.sqrt(self.R_w *
                             np.exp(self.sigma * (1 - (v.dot(self.R_axis) - self.R)**2))) * self.R_axis if len(self.R) > 0 else np.array([0])
        return (self.sigma**2) * (self.caxiscontrib * modified_N.dot(modified_N.transpose()) + modified_R.dot(modified_R.transpose()))
