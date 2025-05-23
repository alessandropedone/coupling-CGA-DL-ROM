## @file main.py
# @brief Main script to generate geometries, meshes, datasets, and plot results.

# Importing the necessary functions
from remove import remove_msh_files, reset_environment
from geometry_generation import generate_geometries
from mesh_generation import generate_meshes, generate_mesh_from_geo
from dataset_generation import generate_datasets, combine_temp_files

# Core of the code that produces the results
#generate_geometries()
#generate_meshes()
#generate_datasets()
#combine_temp_files("data/coordinates.csv")
#combine_temp_files("data/normal_derivative_potential.csv") 
#remove_msh_files()

# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset containing the field values on the upper plate
normal_derivatives = pd.read_csv('data/normal_derivative_potential.csv')
coordinates = pd.read_csv('data/coordinates.csv')


# Plot the field values making the angle vary
for i in range(10):
    values = normal_derivatives.iloc[i, 4:].astype(float)
    coords = coordinates.iloc[i, 4:].astype(float)
    plt.plot(coords, values, label=f"Angle {normal_derivatives.iloc[i, 3]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("normal_derivative_potential")
plt.title("Angle variation")
plt.legend()
plt.grid(True)
plt.show()


# Plot the field values making the distance vary
for i in range(10):
    idx = i * 10
    values = normal_derivatives.iloc[idx, 4:].astype(float)
    coords = coordinates.iloc[idx, 4:].astype(float)
    plt.plot(coords, values, label=f"Distance {normal_derivatives.iloc[idx, 2]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("normal_derivative_potential")
plt.title("Distance variation")
plt.legend()
plt.grid(True)
plt.show()


# Plot the field values making the overetch vary
for i in range(10):
    idx = i * 100
    values = normal_derivatives.iloc[idx, 4:].astype(float)
    coords = coordinates.iloc[idx, 4:].astype(float)
    plt.plot(coords, values, label=f"Overetch {normal_derivatives.iloc[idx, 1]}")

plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("normal_derivative_potential")
plt.title("Overetch variation")
plt.legend()
plt.grid(True)
plt.show()

# Import the dataset containing all the data/results corresponding to the first geometry
path = 'data/results/1_solution.h5'
import h5py
with h5py.File(path, 'r') as file:
    coordinates_x = file['coord_x'][:]
    coordinates_y = file['coord_y'][:]
    potential = file['potential'][:]
    grad_x = file['grad_x'][:]
    grad_y = file['grad_y'][:]
    plt.scatter(coordinates_x, coordinates_y, c=potential, cmap='plasma', s=3, alpha=0.7)
    plt.xlim(-70, 70)
    plt.ylim(-70, 70)
    plt.colorbar(label='Potential Value')
    plt.xlabel('Coordinates X')
    plt.ylabel('Coordinates Y')
    plt.title('Potential Value Distribution')
    plt.grid(True)
    plt.show()
 