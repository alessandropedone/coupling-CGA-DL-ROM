import os
import gmsh
import numpy as np
import csv
import ufl

def save_force_on_segment(segment_tag, solution, filename, folder = "results") :
    """
    Saves the values of the derivative of the (linear) solution at all mesh nodes along a specified segment
    that in our case is the top/bottom plate of the capacitor.

    Parameters:
    - segment_tag: int, the tag of the segment (1D entity) in the Gmsh model.
    - solution: callable, a function that takes a point (x, y, z) and returns a value.
    - filename: str, the name of the output file.
    - folder: str, the directory where the output file will be saved.
    """   
    filepath = os.path.join(folder, filename)

    # Retrieve the nodes (points) associated with the segment
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=1, tag=segment_tag, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)  # Reshape to (N, 3) array

    # The electric field is the derivative of the solution
    function = -ufl.grad(solution)

    # Compute the force values at the nodes
    force_values = np.array([function(x, y, z) for x, y, z in node_coords])
    force_values = force_values.reshape(-1, 3)

    with open(filename, mode='a', newline='') as file: # -a to append
        writer = csv.writer(file)
        for point, E in zip(node_coords, force_values):
            row = list(point) + list(E)
            writer.writerow(row)

