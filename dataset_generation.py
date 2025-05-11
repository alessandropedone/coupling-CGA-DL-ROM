# LAPLACE EQUATION WITH DIRICHLET BOUNDARY CONDITIONS

# read a mesh from file
from mesh_generation import generate_mesh_from_geo

# generate the mesh from the geometry file
from pathlib import Path

""" geo_folder_path = Path('data/meshes')
for geofile in geo_folder_path.iterdir():
    if geofile.is_file() and geofile.suffix == ".geo":
        generate_mesh_from_geo(geofile)
"""
# Now solve the problem for each mesh
from mpi4py import MPI
from dolfinx.io import gmshio

mesh_folder_path = Path('data/mshfiles')
for mesh in mesh_folder_path.iterdir():
    if mesh.is_file() and mesh.suffix == ".msh":
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh, MPI.COMM_WORLD, 0, gdim=2)

        # define finite element function space
        from dolfinx.fem import functionspace
        import numpy as np
        V = functionspace(domain, ("Lagrange", 1))

        # identify the boundary (create facet to cell connectivity required to determine boundary facets)
        from dolfinx import default_scalar_type
        from dolfinx.fem import (Constant, dirichletbc, locate_dofs_topological)
        from dolfinx.fem.petsc import LinearProblem
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # Find facets marked with 2 and 3 (the two rectangles)
        facets_rect1 = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
        facets_rect2 = facet_tags.find(12)

        # Locate degrees of freedom
        dofs_rect1 = locate_dofs_topological(V, fdim, facets_rect1)
        dofs_rect2 = locate_dofs_topological(V, fdim, facets_rect2)

        # Define different Dirichlet values
        u_rect1 = Constant(domain, 0.0)
        u_rect2 = Constant(domain, 1.0)

        # Create BCs
        bc1 = dirichletbc(u_rect1, dofs_rect1, V)
        bc2 = dirichletbc(u_rect2, dofs_rect2, V)

        bcs = [bc1, bc2]

        # trial and test functions
        import ufl
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # source term
        from dolfinx import default_scalar_type
        from dolfinx import fem
        f = fem.Constant(domain, default_scalar_type(0.0))

        # variational problem
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # assemble the system
        from dolfinx.fem.petsc import LinearProblem
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # APPROXIMATION OF THE GRADIENT OF THE SOLUTION

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


        #SAVE RESULTS

        from dolfinx import io
        from pathlib import Path
        import os
        
        results_folder = Path("data/results")
        results_folder.mkdir(exist_ok=True, parents=True)
        
        # Save the mesh and solution in VTK format
        filename = results_folder / "fundamentals"
        """ with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
            vtx.write(0.0) """
        with io.VTKFile(domain.comm, filename.with_suffix(".pvd"), "w") as vtk:
            vtk.write_mesh(domain)
            vtk.write_function(uh)
        with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)

        #Save solution and electric field in h5 file
        import h5py

        # Step 1: Find facets with tag 30(domain)
        cells30 = cell_tags.find(30)
        dofs30 = locate_dofs_topological(V, fdim, cells30)
        #print(dofs30)
        # Step 2: Extract the x-coordinates and the y-coordinates of the DOFs
        dofs = V.tabulate_dof_coordinates()[dofs30]

        x_coords = dofs[:, 0]
        print(x_coords.size)
        #print(x_coords)
        y_coords = dofs[:, 1]
        print(y_coords.size)
        #print(y_coords)

        # Step 3: Evaluate the function at those DOFs
        dim = domain.geometry.dim
        fval = np.array(uh.x.array[dofs30])
        fval_x = grad_uh.x.array[0::dim]
        fval_y = grad_uh.x.array[1::dim]
        fval_x_plate = np.array(fval_x[dofs30])
        fval_y_plate = np.array(fval_y[dofs30])
    
        base_name = os.path.splitext(os.path.basename(mesh))[0]
        filename = results_folder / f"{base_name}_solution.h5"
        with h5py.File(filename, "w") as file:
            file.create_dataset("coordinates", data=x_dofs)
            file.create_dataset("potential_value", data=fval)
            file.create_dataset("field_value_x", data=fval_x_plate)
            file.create_dataset("field_value_y", data=fval_y_plate)
                
