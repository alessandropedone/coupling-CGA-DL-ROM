import os
import re

def modify_plates_distance(geometry, new_gap, name):
    # Open the geometry.geo file to read the lines
    with open(str(geometry), "r") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:

        if 'Point(1) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(new_gap / 2)
            # Recreate the line
            new_line = f"Point(1) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)

        elif 'Rotate' in line:
            # Extract the part after 'Rotate {{0, 0, 1},'
            rest = line.split("Rotate {{0, 0, 1},", 1)[1]

            # Find end of coordinate block
            coord_end = rest.find("}")
            coords_str = rest[:coord_end + 1].strip()  # includes closing brace

            # Get the angle part (right after coords)
            angle_str = rest[coord_end + 2:].split("}")[0].strip()

            # Parse and update y value (coords[1])
            coords = coords_str.strip("{}").split(',')
            coords = [c.strip() for c in coords]
            coords[1] = str(new_gap / 2)

            # Reconstruct the line
            new_line = f"Rotate {{ {{0, 0, 1}}, {{{', '.join(coords)}}}, {angle_str} }} {{\n"
            new_lines.append(new_line)
            
        elif 'Point(2) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(new_gap / 2)
            # Recreate the line
            new_line = f"Point(2) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(3) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(new_gap / 2 + 4)
            # Recreate the line
            new_line = f"Point(3) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(4) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(new_gap / 2 + 4)
            # Recreate the line
            new_line = f"Point(4) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(5) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(-new_gap / 2)
            # Recreate the line
            new_line = f"Point(5) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(6) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(-new_gap / 2)
            # Recreate the line
            new_line = f"Point(6) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(7) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(-new_gap / 2 - 4)
            # Recreate the line
            new_line = f"Point(7) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(8) =' in line:
            # Split the line by commas, remove curly braces, and strip whitespace
            parts = line.split('{')[1].split('}')[0].split(',')
            # Update the second value
            parts[1] = str(-new_gap / 2 - 4)
            # Recreate the line
            new_line = f"Point(8) = {{{','.join(parts)}}};\n"
            new_lines.append(new_line)
        else:
            # If the line doesn't match any of the points, keep it unchanged
            new_lines.append(line)
    
    # Define the directory and file name for saving the new geometry
    directory = "data/geo"
    file_name = str(name) + ".geo"
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define the full file path
    file_path = os.path.join(directory, file_name)
    
    # Write the modified lines to the new geometry file
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    print(f"Geometry updated with a gap of {new_gap}. Saved to {file_path}")


import os
import re

def modify_plates_overetch(geometry, overetch, name):
    # Open the geometry.geo file to read the lines
    with open(str(geometry), "r") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:

        # first rectangle
        if 'Point(1) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) + overetch}"
            new_line = f"Point(1) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(2) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) - overetch}"
            new_line = f"Point(2) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(3) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) - overetch}"
            parts[1] = f"{float(parts[1].strip()) - 2 * overetch}"
            new_line = f"Point(3) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(4) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) + overetch}"
            parts[1] = f"{float(parts[1].strip()) - 2 * overetch}"
            new_line = f"Point(4) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)

        # second rectangle
        elif 'Point(5) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) + overetch}"
            new_line = f"Point(5) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(6) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) - overetch}"
            new_line = f"Point(6) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(7) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) - overetch}"
            parts[1] = f"{float(parts[1].strip()) + 2 * overetch}"
            new_line = f"Point(7) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)
        elif 'Point(8) =' in line:
            parts = line.split('{')[1].split('}')[0].split(',')
            parts[0] = f"{float(parts[0].strip()) + overetch}"
            parts[1] = f"{float(parts[1].strip()) + 2 * overetch}"
            new_line = f"Point(8) = {{{', '.join(part.strip() for part in parts)}}};\n"
            new_lines.append(new_line)

        # ignore other lines
        else:
            new_lines.append(line)

    
    # Define the directory and file name for saving the new geometry
    directory = "data/geo"
    file_name = str(name) + ".geo"
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define the full file path
    file_path = os.path.join(directory, file_name)
    
    # Write the modified lines to the new geometry file
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    print(f"Geometry updated with an overetch of {overetch}. Saved to {file_path}")



import re
import os
from math import pi

def rotate_upper_plate(geometry, new_angle, name):
    with open(str(geometry), "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if 'Rotate' in line:
            # Extract text after 'Rotate {{0, 0, 1},'
            rest = line.split("Rotate { {0, 0, 1},", 1)[1]
            
            # Split at the last '}' to isolate the coordinates
            coord_end = rest.rfind('}')
            coords_str = rest[:coord_end+1].strip()  # includes closing }

            # Rebuild the line with the new angle in the correct position
            # Remove the last value (angle) from coords_str
            coords_str = ', '.join(coords_str.strip('{}').split(',')[:-1])
            new_line = f"Rotate {{ {{0, 0, 1}}, {{{coords_str}, {round(new_angle*pi/180, 3)} }} {{\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Define the directory and file name
    directory = "data/geo"
    file_name = str(name) + ".geo"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Create the file in the directory
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    print(f"Geometry updated with a rotation of {new_angle} degrees. Saved to {file_path}")


import numpy as np
import shutil
from remove import reset_environment

def reset_data():
    reset_environment()
    parameters_file = "data/parameters.csv"
    # Ensure the parameters.csv file is empty
    if os.path.exists(parameters_file):
        with open(parameters_file, "w") as csv_file:
            csv_file.write("ID,Overetch,Distance,Angle\n")
    else:
        os.makedirs("data", exist_ok=True)
        with open(parameters_file, "w") as csv_file:
            csv_file.write("ID,Overetch,Distance,Angle\n")

def generate_geometries():
    overetches = np.linspace(0.1, 0.6, 10)
    distances = np.linspace(1.5, 2.5, 10)
    angles = np.linspace(1, -1, 10)

    # Ensure the parameters.csv file is empty before writing
    with open("data/parameters.csv", "w") as csv_file:
        csv_file.write("ID,Overetch,Distance,Angle\n")
        csv_file.truncate()
    
    reset_data()

    j = 1
    for o in overetches:
        for d in distances:
            for a in angles:
                modify_plates_distance("geometry.geo", d, j)
                modify_plates_overetch("data/geo/" + str(j) + ".geo", o, j)
                rotate_upper_plate("data/geo/" + str(j) + ".geo", a, j)
                with open("data/parameters.csv", "a") as csv_file:
                    csv_file.write(f"{j},{o},{d},{a}\n")
                j += 1
