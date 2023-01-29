import pyvista as pv
import igl
import scipy as sp
import numpy as np
from math import exp
import copy
import os
from multiprocessing import Pool, shared_memory
from tqdm import tqdm

root_folder = os.getcwd()

def normalize_unitbox(V):
	V = V - V.min()
	V = V / V.max()
	return V

## Load a mesh in OFF format
sphere_v, sphere_f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "sphere_s3.off"))
model_v, model_f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "cat_s3.off"))
sphere_v = normalize_unitbox(sphere_v)
model_v = normalize_unitbox(model_v)

vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])
faces = np.hstack(
    [
        [4, 0, 1, 2, 3],  # square
        [3, 0, 1, 4],  # triangle
        [3, 1, 2, 4],  # triangle
    ]
)

p = pv.Plotter(shape=(1,2))

# Create the polydata object
sphere_mesh = pv.PolyData(sphere_v, np.concatenate(([[3]] * sphere_f.shape[0], sphere_f), axis=1))
model_mesh = pv.PolyData(model_v, np.concatenate(([[3]] * model_f.shape[0], model_f), axis=1))

p.subplot(0,0)
actor = p.add_mesh(model_mesh)
def toggle(flag):
	actor.SetVisibility(flag)
_ = p.add_checkbox_button_widget(toggle, value=True)

def select_model(selection):
	global actor
	p.subplot(0,0)
	if selection == "model":
		p.remove_actor(actor)
		actor = p.add_mesh(model_mesh)
	else:
		p.remove_actor(actor)
		actor = p.add_mesh(sphere_mesh)

_ = p.add_text_slider_widget(callback=select_model, data=["model", "sphere"], value=0)

p.subplot(0,1)
p.add_mesh(sphere_mesh)

p.show()
