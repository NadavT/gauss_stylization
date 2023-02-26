import numpy as np
import scipy as sp
from math import exp


class g_Function:
    """
    Class representing g function.
    """

    def __init__(self, N: list[np.array], R: list[int], R_axis: list[np.array], sigma: float, mu: float, lambda_value: float, caxiscontrib: float):
        """
        Creates g_Function object from given parameters.
        """
        self.N = np.array(N).transpose()
        self.R = np.array(R)
        self.R_axis = np.array(R_axis).transpose()
        self.sigma = sigma
        self.mu = mu
        self.lambda_value = lambda_value
        self.caxiscontrib = caxiscontrib if len(R) > 0 else 1
        self._normalize_g_weights()

    def _normalize_g_weights(self):
        """
        Normalizes weights of g function such that all normals will have a value of 1.
        """
        if len(self.N) > 0:
            mat = np.exp(self.sigma * self.N.transpose().dot(self.N))
            try:
                self.N_w = np.linalg.solve(mat, np.ones(self.N.shape[1]))
            except np.linalg.LinAlgError:
                self.N_w = np.linalg.lstsq(mat, np.ones(self.N.shape[1]))[0]
        if len(self.R) > 0:
            mat = np.exp(
                self.sigma * (1 - (self.R - self.R.reshape(-1, 1))**2))
            try:
                self.R_w = np.linalg.solve(mat, np.ones(len(self.R)))
            except np.linalg.LinAlgError:
                self.R_w = np.linalg.lstsq(mat, np.ones(len(self.R)))[0]

    def value(self, v: np.array):
        """
        Returns value of g function for given vector.
        """
        N_value = np.sum(self.N_w * np.exp(self.sigma *
                         v.dot(self.N))) if len(self.N) > 0 else 0
        R_value = np.max(self.R_w * np.exp(self.sigma * (1 -
                         (v.dot(self.R_axis) - self.R)**2))) if len(self.R) > 0 else 0
        return (self.caxiscontrib * N_value + R_value)

    def gradient(self, v: np.array):
        """
        Returns gradient of g function for given vector.
        """
        # Calculate gradient of discrete normals.
        N_grad = np.sum(self.N_w.reshape(-1, 1) * self.N.transpose() *
                        np.exp(self.sigma * v.dot(self.N)).reshape((-1, 1)), axis=0) if len(self.N) > 0 else 0
        # Calculate gradient of semi-discrete normals (calculating with argmax).
        R_real_axis_index = np.argmax(self.R_w * np.exp(self.sigma * (1 -
                                                                      (v.dot(self.R_axis) - self.R)**2))) if len(self.R) > 0 else 0
        R_real_axis = self.R_axis[:, R_real_axis_index] if len(
            self.R) > 0 else 0
        R_grad = self.R_w[R_real_axis_index] * R_real_axis * -2 * (v.dot(R_real_axis) - self.R[R_real_axis_index]) * np.exp(
            self.sigma * (1 - np.power(v.dot(R_real_axis) - self.R[R_real_axis_index], 2))) if len(self.R) > 0 else 0
        return self.sigma * (self.caxiscontrib * N_grad + R_grad)

    def hessian(self, v: np.array):
        """
        Returns hessian of g function for given vector.
        """
        # Calculate hessian of discrete normals (using modified normals).
        modified_N = np.sqrt(
            self.N_w * np.exp(self.sigma * v.dot(self.N))) * self.N if len(self.N) > 0 else np.array([0])
        # Calculate hessian of semi-discrete normals (using modified normals and argmax).
        R_real_axis_index = np.argmax(self.R_w * np.exp(self.sigma * (1 -
                                                                      (v.dot(self.R_axis) - self.R)**2))) if len(self.R) > 0 else 0
        R_real_axis = self.R_axis[:, R_real_axis_index] if len(
            self.R) > 0 else 0
        modified_R = (np.sqrt(self.R_w[R_real_axis_index] *
                              np.exp(self.sigma * (1 - (v.dot(R_real_axis) - self.R[R_real_axis_index])**2))) * R_real_axis).reshape(-1, 1) if len(self.R) > 0 else np.array([0])
        return (self.sigma**2) * (self.caxiscontrib * modified_N.dot(modified_N.transpose()) + modified_R.dot(modified_R.transpose()))
