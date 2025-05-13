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
row_data = df.iloc[4, 4:].astype(float)
column_names = df.columns[4:].astype(float)

# Plot the data
plt.plot(column_names, row_data, label="grad_y_uh_plate", color="blue", linestyle="-")
plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("grad_y_uh_plate")
plt.title("Plot of grad_y_uh_plate vs coords")
plt.legend()
plt.grid(True)
plt.show()


row_data = df.iloc[890, 4:].astype(float)
column_names = df.columns[4:].astype(float)

# Plot the data
plt.plot(column_names, row_data, label="grad_y_uh_plate", color="blue", linestyle="-")
plt.xlim(-50, 50)
plt.xlabel("coords")
plt.ylabel("grad_y_uh_plate")
plt.title("Plot of grad_y_uh_plate vs coords")
plt.legend()
plt.grid(True)
plt.show()