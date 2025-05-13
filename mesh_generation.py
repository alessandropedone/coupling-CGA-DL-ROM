
import gmsh
import os
from pathlib import Path

def generate_mesh_from_geo(geo_path):

    # Initialize Gmsh
    gmsh.initialize()

    # Optional: Hide Gmsh messages
    gmsh.option.setNumber("General.Terminal", 1)

    # Load the .geo file
    gmsh.open(geo_path)

    # Generate 2D or 3D mesh depending on your .geo setup
    gmsh.model.mesh.generate(2)

    # Write the mesh to a .msh file    
    # Create meshes folder if it doesn't exist
    msh_output_folder = Path("data/msh")
    msh_output_folder.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(geo_path))[0]
    msh_path = os.path.join(msh_output_folder, base_name + ".msh")

    gmsh.write(msh_path)

    # Finalize Gmsh
    gmsh.finalize()

import multiprocessing

def generate_mesh(i):
    # Generate the mesh for each geometry
    geo_path = f"data/geo/{i}.geo"
    generate_mesh_from_geo(geo_path)

def generate_meshes():
    r = range(1, 1001)
    with multiprocessing.Pool() as pool:
        pool.map(generate_mesh, r)
