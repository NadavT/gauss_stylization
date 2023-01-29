from g_function import g_Function
import numpy as np

def cube(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([1,0,0]), np.array([-1,0,0]),
			np.array([0,1,0]), np.array([0,-1,0]),
			np.array([0,0,1]), np.array([0,0,-1]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	return g_Function(N, [], [], sigma, mu, lambda_value, caxiscontrib)

def pyramid_x(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([-1,  0,  0]),
			np.array([ 2,  1,  1]),
			np.array([ 2,  1, -1]),
			np.array([ 2, -1,  1]),
			np.array([ 2, -1, -1]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	return g_Function(N, [], [], sigma, mu, lambda_value, caxiscontrib)

def pyramid_y(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([ 0, -1,  0]),
			np.array([ 1,  2,  1]),
			np.array([ 1,  2, -1]),
			np.array([-1,  2,  1]),
			np.array([-1,  2, -1]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	return g_Function(N, [], [], sigma, mu, lambda_value, caxiscontrib)

def pyramid_z(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([ 0,  0, -1]),
			np.array([ 1,  1,  2]),
			np.array([ 1, -1,  2]),
			np.array([-1,  1,  2]),
			np.array([-1, -1,  2]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	return g_Function(N, [], [], sigma, mu, lambda_value, caxiscontrib)

def cylinder_x(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([ 1,  0,  0]),
			np.array([-1,  0,  0]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	R = [0]
	R_axis = np.array([1, 0, 0])
	return g_Function(N, R, R_axis, sigma, mu, lambda_value, caxiscontrib)

def cylinder_y(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([ 0,  1,  0]),
			np.array([ 0, -1,  0]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	R = [0]
	R_axis = np.array([0, 1, 0])
	return g_Function(N, R, R_axis, sigma, mu, lambda_value, caxiscontrib)

def cylinder_z(sigma, mu, lambda_value, caxiscontrib):
	N = [
			np.array([ 0,  0,  1]),
			np.array([ 0,  0, -1]),
		]
	N = [n / np.linalg.norm(n) for n in N]
	R = [0]
	R_axis = np.array([0, 0, 1])
	return g_Function(N, R, R_axis, sigma, mu, lambda_value, caxiscontrib)
