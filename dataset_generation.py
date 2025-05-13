
def empty_results_folder():
    from pathlib import Path
    # Ensure the results folder is empty before running the script
    results_folder = Path("data/results")
    if results_folder.exists():
        for file in results_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        results_folder.mkdir(parents=True)

def set_up_environment():
    # Set up the environment
    empty_results_folder()

    import shutil
    # Duplicate the data/parameters.csv file
    shutil.copy("data/parameters.csv", "data/dataset.csv")

    from mpi4py import MPI
    from dolfinx.io import gmshio
    domain, cell_tags, facet_tags = gmshio.read_from_msh("data/msh/1.msh", MPI.COMM_WORLD, 0, gdim=2)

    # Define finite element function space
    from dolfinx.fem import functionspace
    V = functionspace(domain, ("Lagrange", 1))

    from dolfinx.fem import locate_dofs_topological
    # Identify the boundary (create facet to cell connectivity required to determine boundary facets)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # Step 1: Find facets with tag 10
    facets10 = facet_tags.find(10)
    dofs10 = locate_dofs_topological(V, fdim, facets10)

    # Step 2: Extract the x-coordinates and the y-coordinates of the DOFs
    x_dofs = V.tabulate_dof_coordinates()[dofs10]
    x_coords = x_dofs[:, 0]
    y_coords = x_dofs[:, 1]

    import pandas as pd
    import numpy as np
    # Step 4: Sort the coordinates
    parameters = pd.read_csv("data/parameters.csv")
    center_y = parameters.iloc[0, 2] / 2
    center_x = 0.0
    coords = np.sign(x_coords) * np.sqrt((x_coords-center_x)**2 + (y_coords-center_y)**2)
    sorted_indices = np.argsort(coords)
    coords = coords[sorted_indices]

    # Step 5: Save the sorted coordinates values to a CSV file
    # Load the existing CSV file into a pandas DataFrame
    df = pd.read_csv("data/dataset.csv")
    # Ensure the DataFrame has enough columns to accommodate the new data
    if len(df.columns) < 4 + len(coords):
        additional_columns = 4 + len(coords) - len(df.columns)
        for i in range(additional_columns):
            df[f"{coords[i]}"] = 0.0
    # Save the modified DataFrame back to the CSV file
    df.to_csv("data/dataset.csv", index=False)

    from pathlib import Path
    # Create a temporary folder to store intermediate CSV files
    temp_folder = Path("data/temp")
    if temp_folder.exists():
        for file in temp_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        temp_folder.mkdir(parents=True, exist_ok=True)

def process_mesh(mesh):
    if mesh.is_file() and mesh.suffix == ".msh":
        # read a mesh from file
        from mpi4py import MPI
        from dolfinx.io import gmshio
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

        # Step 4: Sort the coordinates and the corresponding grad_uh values, then save grad_y into dataset_{j}.csv
        import pandas as pd
        j = int(mesh.stem)
        df = pd.read_csv("data/dataset.csv")
        center_y = df.iloc[j-1, 2] / 2
        print("mesh number: ", j, " center_y: ", center_y)
        center_x = 0.0
        coords = np.sign(x_coords) * np.sqrt((x_coords-center_x)**2 + (y_coords-center_y)**2)
        sorted_indices = np.argsort(coords)
        coords = coords[sorted_indices]
        grad_y_uh_plate = grad_y_uh_plate[sorted_indices]
        grad_x_uh_plate = grad_x_uh_plate[sorted_indices]
        start_col = 4
        needed_cols = start_col + len(coords)
        df.iloc[j-1, start_col:needed_cols] = grad_y_uh_plate
        # Copy the jth line of dataset.csv into dataset_{j}.csv
        df.iloc[[j-1]].to_csv(f"data/temp/dataset_{j}.csv", index=False)             

        # SAVE RESULTS
        from pathlib import Path
        results_folder = Path("data/results")
        results_folder.mkdir(exist_ok=True, parents=True)

        # Save solution and electric field in h5 file

        # Step 1: Find all DOFs in the function space
        dofs = np.arange(V.dofmap.index_map.size_local)
        # Step 2: Extract the x-coordinates and the y-coordinates of the DOFs
        dofs_c = V.tabulate_dof_coordinates()[dofs]
        x_coords = np.array(dofs_c[:, 0])
        y_coords = np.array(dofs_c[:, 1])
        # Step 3: Evaluate the function at those DOFs
        dim = domain.geometry.dim
        pval = np.array(uh.x.array[dofs])
        fval_x = grad_uh.x.array[0::dim]
        fval_y = grad_uh.x.array[1::dim]
        fval_x = np.array(fval_x[dofs])
        fval_y = np.array(fval_y[dofs])
        
        import os
        import h5py
        base_name = os.path.splitext(os.path.basename(mesh))[0]
        filename = results_folder / f"{base_name}_solution.h5"
        with h5py.File(filename, "w") as file:
            file.create_dataset("coordinates_x", data=x_coords)
            file.create_dataset("coordinates_y", data=y_coords)
            file.create_dataset("potential_value", data=pval)
            file.create_dataset("field_value_x", data=fval_x)
            file.create_dataset("field_value_y", data=fval_y)

def generate_datasets():
    # Set up the environment
    set_up_environment()

    from pathlib import Path
    # Parallel processing
    mesh_folder_path = Path('data/msh')
    meshes = list(mesh_folder_path.iterdir())

    from multiprocessing import Pool, cpu_count
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_mesh, meshes)

def combine_temp_files():
    # Define paths
    from pathlib import Path
    temp_folder = Path("data/temp")
    dataset_file = Path("data/dataset.csv")

    import pandas as pd
    # Read the main dataset once
    dataset_df = pd.read_csv(dataset_file)

    # Collect updates from all temp files
    for temp_file in sorted(temp_folder.glob("*.csv"), key=lambda x: int(x.stem.split('_')[-1])):
        index = int(temp_file.stem.split('_')[-1]) - 1  # Adjust to 0-based index
        temp_df = pd.read_csv(temp_file)
        
        # Update the corresponding row
        if 0 <= index < len(dataset_df):
            dataset_df.iloc[index] = temp_df.iloc[0]
        print(f"Updated row {index + 1} from {temp_file.name}")

    # Write the dataset once after all updates
    dataset_df.to_csv(dataset_file, index=False)

    # Clean up temp files and folder
    for temp_file in temp_folder.glob("*.csv"):
        temp_file.unlink()
    temp_folder.rmdir()