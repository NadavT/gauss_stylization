import pyvista as pv
import igl
import numpy as np
import os
from load_model import load
from precompute import Precompute
from g_function import g_Function
from calculate import single_iteration
import copy
from time import time
from math import sqrt

root_folder = os.getcwd()

sphere_v, sphere_f = load(os.path.join(root_folder, "data", "sphere_s3.off"), normalize=False)
# model_v, model_f = load(os.path.join(root_folder, "data", "sphere_s2.off"))
model_v, model_f = load(os.path.join(root_folder, "data", "cat_s3.off"))

p = pv.Plotter(shape=(1,2))

precompute = Precompute(model_v, model_f)

# N = [
# 		np.array([1,0,0]), np.array([-1,0,0]),
# 		np.array([0,1,0]), np.array([0,-1,0]),
# 		np.array([0,0,1]), np.array([0,0,-1]),
# 	]

N = [
		np.array([-1,0,0]),
		np.array([2, 1, 1]),
		np.array([2, 1, -1]),
		np.array([2, -1, 1]),
		np.array([2, -1, -1]),
	]

N = [
		np.array([ 0, -1,  0]),
		np.array([ 1,  2,  1]),
		np.array([ 1,  2, -1]),
		np.array([-1,  2,  1]),
		np.array([-1,  2, -1]),
	]
N = [n / np.linalg.norm(n) for n in N]
sigma = 8
mu = 1
lambda_value = 4
g = g_Function(N, sigma, mu, lambda_value)

modified_v = copy.deepcopy(model_v)
# single_iteration(model_v, modified_v, model_f, 1, precompute, g)

deformed_sphere = np.array([v * g.value(v) for v in sphere_v[:]])

# Create the polydata object
sphere_mesh = pv.PolyData(deformed_sphere, np.concatenate(([[3]] * sphere_f.shape[0], sphere_f), axis=1))
model_mesh = pv.PolyData(modified_v, np.concatenate(([[3]] * model_f.shape[0], model_f), axis=1))

p.subplot(0,0)
actor = p.add_mesh(model_mesh)
p.reset_camera()

# def select_model(selection):
# 	global actor
# 	p.subplot(0,0)
# 	if selection == "model":
# 		p.remove_actor(actor)
# 		actor = p.add_mesh(model_mesh)
# 	else:
# 		p.remove_actor(actor)
# 		actor = p.add_mesh(sphere_mesh)

# _ = p.add_text_slider_widget(callback=select_model, data=["model", "sphere"], value=0)

iterations_amount = 1
def change_iterations_amount(iterations):
	global iterations_amount
	iterations_amount = int(iterations)
_ = p.add_text_slider_widget(callback=change_iterations_amount, data=list(map(str, range(1,101))), value=0)

def iterate():
	global actor
	global modified_v
	global model_mesh
	for _ in range(iterations_amount):
		print("start")
		s = time()
		single_iteration(model_v, modified_v, model_f, 1, precompute, g)
		print(f"end: {time() - s}")
		model_mesh = pv.PolyData(modified_v, np.concatenate(([[3]] * model_f.shape[0], model_f), axis=1))
		p.subplot(0,0)
		p.remove_actor(actor)
		actor = p.add_mesh(model_mesh)
		p.reset_camera()
_ = p.add_key_event("space", iterate)

def reset():
	global actor
	global modified_v
	global model_mesh
	modified_v = copy.deepcopy(model_v)
	model_mesh = pv.PolyData(modified_v, np.concatenate(([[3]] * model_f.shape[0], model_f), axis=1))
	p.subplot(0,0)
	p.remove_actor(actor)
	actor = p.add_mesh(model_mesh)
	p.reset_camera()
_ = p.add_key_event("z", reset)


p.subplot(0,1)
p.add_mesh(sphere_mesh)

p.show()
