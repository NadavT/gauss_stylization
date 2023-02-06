import pyvista as pv
import igl
import numpy as np
import os
from load_model import load
from precompute import Precompute
from g_function import g_Function
from calculate import Calculate
import copy
from time import time
import argparse
import functions

if __name__ == "__main__":
    root_folder = os.getcwd()
    parser = argparse.ArgumentParser(
        prog='gauss_stylization')
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

    sphere_v, sphere_f = load(os.path.join(
        root_folder, "data", "sphere_s3.off"), normalize=False)
    model_v, model_f = load(os.path.join(
        root_folder, "data", parser.parse_args().model))

    precomputed = Precompute(model_v, model_f)

    p = pv.Plotter(shape=(1, 2))

    modified_v = copy.deepcopy(model_v)
    g = functions.cube(parser.parse_args().sigma, parser.parse_args(
    ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
    deformed_sphere = np.array([v * g.value(v) for v in sphere_v[:]])

    # Create the polydata object
    sphere_mesh = pv.PolyData(deformed_sphere, np.concatenate(
        ([[3]] * sphere_f.shape[0], sphere_f), axis=1))
    model_mesh = pv.PolyData(modified_v, np.concatenate(
        ([[3]] * model_f.shape[0], model_f), axis=1))

    p.subplot(0, 0)
    actor = p.add_mesh(model_mesh)
    p.reset_camera()

    calc = Calculate(model_v, model_f, precomputed, g)

    iterations_amount = 1

    def change_iterations_amount(iterations):
        global iterations_amount
        iterations_amount = int(iterations)
    _ = p.add_text_slider_widget(
        callback=change_iterations_amount, data=list(map(str, range(1, 101))), value=0)

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

    p.subplot(0, 1)
    function_actor = p.add_mesh(sphere_mesh)

    def switch_function(function: str):
        global sphere_mesh
        global calc
        global function_actor
        if function == "cube":
            g = functions.cube(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "pyramid x":
            g = functions.pyramid_x(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "pyramid y":
            g = functions.pyramid_y(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "pyramid z":
            g = functions.pyramid_z(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cylinder x":
            g = functions.cylinder_x(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cylinder y":
            g = functions.cylinder_y(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cylinder z":
            g = functions.cylinder_z(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cone x":
            g = functions.cone_x(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cone y":
            g = functions.cone_y(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        elif function == "cone z":
            g = functions.cone_z(parser.parse_args().sigma, parser.parse_args(
            ).mu, parser.parse_args().lambda_value, parser.parse_args().caxiscontrib)
        else:
            raise RuntimeError(f"No function called {function}")
        calc.terminate()
        calc = Calculate(model_v, model_f, precomputed, g)
        deformed_sphere = np.array([v * g.value(v) for v in sphere_v[:]])
        sphere_mesh = pv.PolyData(deformed_sphere, np.concatenate(
            ([[3]] * sphere_f.shape[0], sphere_f), axis=1))

        p.subplot(0, 1)
        p.remove_actor(function_actor)
        function_actor = p.add_mesh(sphere_mesh)
        p.reset_camera()
    _ = p.add_text_slider_widget(callback=switch_function, data=[
        "cube", "pyramid x", "pyramid y", "pyramid z", "cylinder x", "cylinder y", "cylinder z", "cone x", "cone y", "cone z"], value=0)

    p.show()
    calc.terminate()
