from remove import remove_msh_files, reset_environment
from geometry_generation import generate_geometries
from mesh_generation import generate_meshes, generate_mesh_from_geo
from dataset_generation import generate_datasets, combine_temp_files

#generate_geometries()
#generate_meshes()
#generate_datasets()
#combine_temp_files()
#remove_msh_files()

import pandas as pd
df = pd.read_csv('data/dataset.csv')
import matplotlib.pyplot as plt

# Extract the first row starting from column 4
# Extract the first row starting from column 4
column_names = df.columns[4:].astype(float)

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

path = 'data/results/1_solution.h5'
import h5py
with h5py.File(path, 'r') as file:
    coordinates_x = file['coordinates_x'][:]
    coordinates_y = file['coordinates_y'][:]
    potential_value = file['potential_value'][:]
    field_value_x = file['field_value_x'][:]
    field_value_y = file['field_value_y'][:]
    plt.scatter(coordinates_x, coordinates_y, c=potential_value, cmap='plasma', s=3, alpha=0.7)
    plt.xlim(-70, 70)
    plt.ylim(-70, 70)
    plt.colorbar(label='Potential Value')
    plt.xlabel('Coordinates X')
    plt.ylabel('Coordinates Y')
    plt.title('Potential Value Distribution')
    plt.grid(True)
    plt.show()