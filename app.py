import pyvista as pv
import igl
import numpy as np
import os
from load_model import load
from precompute import Precompute
from calculate import Calculate
import copy
from time import time
import argparse
from functions import create_function
import json

if __name__ == "__main__":
    # Parse arguments.
    root_folder = os.getcwd()
    parser = argparse.ArgumentParser(
        prog='gauss_stylization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,
                        default='cat_s3.off', help='model to stylize')
    parser.add_argument('--sigma', type=float, default=8,
                        help='sigma value for function')
    parser.add_argument('--mu', type=float, default=1,
                        help='mu parameter')
    parser.add_argument('--lambda_value', type=float,
                        default=4, help='lambda parameter')
    parser.add_argument('--caxiscontrib', type=float,
                        default=0.5, help='Axis contribution in semi-discrete normals (discrete normals contribution when using semi-discrete normals)')
    parser.add_argument('--admm_iterations', type=int,
                        default=1, help="admm iterations to do per gauss stylization update")

    # Load model and sphere (for function representation).
    sphere_v, sphere_f = load(os.path.join(
        root_folder, "data", "sphere_s3.off"), normalize=False)
    if os.path.exists(parser.parse_args().model):
        model_v, model_f = load(parser.parse_args().model)
    else:
        model_v, model_f = load(os.path.join(
            root_folder, "data", parser.parse_args().model))

    # Load functions descriptions.
    with open("functions.json", "r") as f:
        functions_descriptions = {
            function["name"]: function for function in json.loads(f.read())}
        assert "cube" in functions_descriptions

    # Precompute model data.
    precomputed = Precompute(model_v, model_f)

    # Create plotter.
    p = pv.Plotter(shape=(1, 2))

    # Create the cube function.
    modified_v = copy.deepcopy(model_v)
    g = create_function(functions_descriptions["cube"], parser.parse_args().sigma, parser.parse_args(
    ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
    deformed_sphere = np.array([v * g.value(v) for v in sphere_v[:]])

    # Create the polydata object
    sphere_mesh = pv.PolyData(deformed_sphere, np.concatenate(
        ([[3]] * sphere_f.shape[0], sphere_f), axis=1))
    model_mesh = pv.PolyData(modified_v, np.concatenate(
        ([[3]] * model_f.shape[0], model_f), axis=1))

    # Add the mesh to the plotter and render it.
    p.subplot(0, 0)
    actor = p.add_mesh(model_mesh)
    p.camera.up = (0, 1, 0)
    p.camera.roll = 0
    p.camera.position = (0, 0, 10)
    p.reset_camera()

    # Generate a calculation for this model and function.
    calc = Calculate(model_v, model_f, precomputed, g)

    # Iterations amount slider.
    iterations_amount = 1

    def change_iterations_amount(iterations):
        global iterations_amount
        iterations_amount = int(iterations)
    _ = p.add_text_slider_widget(
        callback=change_iterations_amount, data=list(map(str, range(1, 101))), value=0)

    # Iteration event.
    def iterate():
        global actor
        global modified_v
        global model_mesh
        for _ in range(iterations_amount):
            print("start")
            s = time()
            calc.single_iteration(modified_v, parser.parse_args(
            ).admm_iterations)
            print(f"end: {time() - s}")
            model_mesh = pv.PolyData(modified_v, np.concatenate(
                ([[3]] * model_f.shape[0], model_f), axis=1))
            p.subplot(0, 0)
            p.remove_actor(actor)
            actor = p.add_mesh(model_mesh)
            p.reset_camera()
    _ = p.add_key_event("space", iterate)

    # Reset event.
    def reset():
        global actor
        global modified_v
        global model_mesh
        modified_v = copy.deepcopy(model_v)
        model_mesh = pv.PolyData(modified_v, np.concatenate(
            ([[3]] * model_f.shape[0], model_f), axis=1))

        p.subplot(0, 0)
        p.remove_actor(actor)
        actor = p.add_mesh(model_mesh)
        p.reset_camera()
    _ = p.add_key_event("z", reset)

    # Add function representation panel.
    p.subplot(0, 1)
    p.camera.up = (0, 1, 0)
    p.camera.roll = 0
    p.camera.position = (0, 0, 10)
    p.reset_camera()
    function_actor = p.add_mesh(sphere_mesh)

    # Add function selection slider.
    def switch_function(function_name: str):
        global sphere_mesh
        global calc
        global function_actor
        g = create_function(functions_descriptions[function_name], parser.parse_args().sigma, parser.parse_args(
        ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        calc.terminate()
        calc = Calculate(model_v, model_f, precomputed, g)
        deformed_sphere = np.array([v * g.value(v) for v in sphere_v[:]])
        sphere_mesh = pv.PolyData(deformed_sphere, np.concatenate(
            ([[3]] * sphere_f.shape[0], sphere_f), axis=1))

        p.subplot(0, 1)
        p.remove_actor(function_actor)
        function_actor = p.add_mesh(sphere_mesh)
        p.reset_camera()
    _ = p.add_text_slider_widget(
        callback=switch_function, data=list(functions_descriptions.keys()), value=0)

    p.show()
    calc.terminate()
