
## @package mesh_generation
# @brief Generate meshes from .geo files using gmsh in parallel.

import gmsh
import os
from pathlib import Path

##
# @param geo_path (str): Path to the .geo file.
def generate_mesh_from_geo(geo_path: str):
    """Generate a mesh from a .geo file using gmsh."""
    # Initialize gmsh
    gmsh.initialize()

    # Load the .geo file
    gmsh.open(geo_path)

    # Generate 2D or 3D mesh depending on your .geo setup
    gmsh.model.mesh.generate(2)

      
    # Create mesh folder if it doesn't exist
    msh_output_folder = Path("data/msh")
    msh_output_folder.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(geo_path))[0]
    msh_path = os.path.join(msh_output_folder, base_name + ".msh")

    # Write the mesh to a .msh file 
    gmsh.write(msh_path)

    # Finalize gmsh
    gmsh.finalize()

import multiprocessing

##
# @param i (int): Index of the geometry.
def generate_mesh(i: int):
    """
    Generate a mesh for a given geometry index.
    """
    # Generate the mesh for each geometry
    geo_path = f"data/geo/{i}.geo"
    generate_mesh_from_geo(geo_path)

def generate_meshes():
    """
    Generate meshes for all geometries in the range from 1 to 1000.
    This function uses multiprocessing to speed up the process.
    """
    # Create a pool of workers
    # and map the generate_mesh function to the range of geometries
    # Here, we assume you have 1000 geometries numbered from 1 to 1000
    r = range(1, 1001)
    with multiprocessing.Pool() as pool:
        pool.map(generate_mesh, r)
