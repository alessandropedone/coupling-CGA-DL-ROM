import os
import numpy as np
import csv


def save_force_on_segment(segment_coords, force_values, filename, folder = "results") :


    #grad = ufl.grad(solution)
    #values = np.zeros((len(segment_coords), solution.function_space().mesh.geometry.dim), dtype=solution.dtype)
    #solution.eval_gradient(values, segment_coords, solution.function_space().mesh)
    #force_values = -values


    with open(filename, mode='a', newline='') as file: # -a to append
        writer = csv.writer(file)
        for point, E in zip(segment_coords, force_values):
            row = list(point) + list(E)
            writer.writerow(row)

