from g_function import g_Function
import numpy as np


def create_function(function_description: dict, sigma, mu, lambda_value, caxiscontrib):
    N = function_description["discrete_normals"]
    N = [n / np.linalg.norm(n) for n in N]
    R = function_description["semi_discrete_normals"]
    R_axes = function_description["semi_discrete_normals_axes"]
    R_axes = [R_axis / np.linalg.norm(R_axis) for R_axis in R_axes]
    return g_Function(N, R, R_axes, sigma, mu, lambda_value, caxiscontrib)
