## @file main.py
# @brief Main script to generate geometries, meshes, datasets, and plot results.

# Importing the necessary functions
from remove import remove_msh_files, reset_environment
from geometry_generation import generate_geometries
from mesh_generation import generate_meshes, generate_mesh_from_geo
from dataset_generation import generate_datasets, combine_temp_files

# Core of the code that produces the results
generate_geometries()
generate_meshes()
generate_datasets()
combine_temp_files("data/normal_derivate_potential.csv") 
combine_temp_files("data/normal_derivate_potential_temp.csv")
remove_msh_files()

# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset containing the field values on the upper plate
df = pd.read_csv('data/dataset.csv')

# Extract the relevant columns, i.e. the ones starting from the 5th column
column_names = df.columns[4:].astype(float)

# Plot the field values making the angle vary
for i in range(10):
    row_data = df.iloc[i, 4:].astype(float)
    plt.plot(column_names, row_data, label=f"Angle {df.iloc[i, 3]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("grad_y_uh_plate")
plt.title("Plot of grad_y_uh_plate vs coords (first 10 rows)")
plt.legend()
plt.grid(True)
plt.show()


# Plot the field values making the distance vary
for i in range(10):
    row_data = df.iloc[i*10, 4:].astype(float)
    plt.plot(column_names, row_data, label=f"Distance {df.iloc[i*10, 2]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("grad_y_uh_plate")
plt.title("Plot of grad_y_uh_plate vs coords (first 10 rows)")
plt.legend()
plt.grid(True)
plt.show()


# Plot the field values making the overetch vary
for i in range(10):
    row_data = df.iloc[i*100, 4:].astype(float)
    plt.plot(column_names, row_data, label=f"Overetch {df.iloc[i*100, 1]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("grad_y_uh_plate")
plt.title("Plot of grad_y_uh_plate vs coords (first 10 rows)")
plt.legend()
plt.grid(True)
plt.show()

# Import the dataset containing all the data/results corresponding to the first geometry
path = 'data/results/1_solution.h5'
import h5py
with h5py.File(path, 'r') as file:
    coordinates_x = file['coord_x'][:]
    coordinates_y = file['coord_y'][:]
    coordinates = file['coord'][:]
    potential = file['potential'][:]
    grad_x = file['grad_x'][:]
    grad_y = file['grad_y'][:]
    normal_derivate = file['normal_derivative_potential'][:]
    plt.scatter(coordinates_x, coordinates_y, c=normal_derivate, cmap='plasma', s=3, alpha=0.7)
    plt.xlim(-70, 70)
    plt.ylim(-70, 70)
    plt.colorbar(label='Potential Value')
    plt.xlabel('Coordinates X')
    plt.ylabel('Coordinates Y')
    plt.title('Potential Value Distribution')
    plt.grid(True)
    plt.show()