## @file convergence.py
#  @brief Solves a PDE using FEniCSx and plots the gradient of the solution on the upper plate.
#  @details This script generates a mesh, solves a Laplace equation, computes the gradient of the solution,
#           and plots the y-component of the gradient on the lower edge of the upper plate, refining progressively the mesh.

import matplotlib.pyplot as plt
import numpy as np

# Initialize the plot figure outside the function
plt.figure(figsize=(12, 8))

## 
# @param path (str): path to the directory containing the .geo file.
# @param name (str): name of the .geo file without extension.
# @param plot_style (dict): dictionary containing color, linestyle, and marker for the plot.
def solve_and_plot_grad(path: str, name: str, plot_style: dict):
    """Solve the PDE and plot the gradient of the solution on the upper plate."""
    from mpi4py import MPI
    from dolfinx.io import gmshio
    from mesh_generation import generate_mesh_from_geo
    import os
    import subprocess

    # Generate the mesh from path/name.geo
    with open(os.devnull, 'w') as devnull:
        subprocess.run(["gmsh", "-2", str(path)+str(name)+".geo"], stdout=devnull, stderr=devnull)

    # Read the mesh from the generated .msh file
    domain, cell_tags, facet_tags = gmshio.read_from_msh(str(path)+str(name)+".msh", MPI.COMM_WORLD, 0, gdim=2)

    # Define finite element function space
    from dolfinx.fem import functionspace
    V = functionspace(domain, ("Lagrange", 1))

    # Identify the boundary (create facet to cell connectivity required to determine boundary facets)
    from dolfinx import default_scalar_type
    from dolfinx.fem import (Constant, dirichletbc, locate_dofs_topological)
    from dolfinx.fem.petsc import LinearProblem
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # Find facets marked with 10, 11, 12 (the two plates)
    facets_rect1 = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
    facets_rect2 = facet_tags.find(12)

    # Locate degrees of freedom
    dofs_rect1 = locate_dofs_topological(V, fdim, facets_rect1)
    dofs_rect2 = locate_dofs_topological(V, fdim, facets_rect2)

    # Define different Dirichlet boundary conditions for the two plates
    u_rect1 = Constant(domain, 0.0)
    u_rect2 = Constant(domain, 1.0)

    # Create BCs
    bc1 = dirichletbc(u_rect1, dofs_rect1, V)
    bc2 = dirichletbc(u_rect2, dofs_rect2, V)

    bcs = [bc1, bc2]

    # Trial and test functions
    import ufl
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Source term
    from dolfinx import default_scalar_type
    from dolfinx import fem
    f = fem.Constant(domain, default_scalar_type(0.0))

    # Variational problem
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Assemble the system
    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Approximate the gradient of the solution
    import ufl

    # Define the vector function space for the gradient
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

    # Define the trial and test functions for the vector space
    u_vec = ufl.TrialFunction(V_vec)
    v_vec = ufl.TestFunction(V_vec)

    # Define the gradient of the solution
    grad_u = ufl.grad(uh)

    # Define the bilinear and linear forms
    a_grad = ufl.inner(u_vec, v_vec) * ufl.dx
    L_grad = ufl.inner(grad_u, v_vec) * ufl.dx

    # Assemble the system
    from dolfinx.fem.petsc import LinearProblem
    problem_grad = LinearProblem(a_grad, L_grad, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    grad_uh = problem_grad.solve()

    # PLOTTING THE GRADIENT OF THE SOLUTION ON THE UPPER PLATE

    # Step 1: Find facets with tag 10
    facets10 = facet_tags.find(10)
    dofs10 = locate_dofs_topological(V, fdim, facets10)

    # Step 2: Extract the x-coordinates and the y-coordinates of the DOFs
    x_dofs = V.tabulate_dof_coordinates()[dofs10]
    x_coords = x_dofs[:, 0]
    y_coords = x_dofs[:, 1]

    # Step 3: Evaluate grad_uh at those DOFs
    dim = domain.geometry.dim
    grad_x_uh_values = grad_uh.x.array[0::dim]
    grad_y_uh_values = grad_uh.x.array[1::dim]
    grad_x_uh_plate = grad_x_uh_values[dofs10]
    grad_y_uh_plate = grad_y_uh_values[dofs10]

    # Step 4: Add plot to the existing figure
    center_y = 1.5/2
    center_x = 0.0
    coords = np.sign(x_coords) * np.sqrt((x_coords-center_x)**2 + (y_coords-center_y)**2)
    sorted_indices = np.argsort(coords)
    coords = coords[sorted_indices]
    grad_y_uh_plate = grad_y_uh_plate[sorted_indices]
    grad_x_uh_plate = grad_x_uh_plate[sorted_indices]
    
    # Plot with specified style
    plt.plot(coords, grad_y_uh_plate, 
             label=f"{name} (grad_y_uh_plate)", 
             color=plot_style['color'], 
             linestyle=plot_style['linestyle'],
             marker=plot_style['marker'],
             markersize=4,
             markevery=len(coords)//20,  # Show markers every 20th point to avoid clutter
             linewidth=2)

# Define different styles for each refinement level
plot_styles = {
    "r1": {"color": "blue", "linestyle": "-", "marker": "o"},
    "r3": {"color": "red", "linestyle": "--", "marker": "s"},
    "r5": {"color": "green", "linestyle": "-.", "marker": "^"},
    "r7": {"color": "purple", "linestyle": ":", "marker": "D"}
}

# Solve and plot for each refinement level
solve_and_plot_grad("convergence/", "r1", plot_styles["r1"])
solve_and_plot_grad("convergence/", "r3", plot_styles["r3"])
solve_and_plot_grad("convergence/", "r5", plot_styles["r5"])
solve_and_plot_grad("convergence/", "r7", plot_styles["r7"])

# Finalize the combined plot
plt.xlim(-50, 50)
plt.xlabel("Coordinates", fontsize=12)
plt.ylabel("grad_y_uh_plate", fontsize=12)
plt.title("Convergence Study: grad_y_uh_plate vs coordinates for different mesh refinements", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# Save the combined plot
import os
output_path = os.path.join("convergence/", "combined_grad_y_uh_plate_convergence.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved as {output_path}")
plt.show()

from remove import remove_msh_files
remove_msh_files("convergence/")