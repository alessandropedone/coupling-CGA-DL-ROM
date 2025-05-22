## @package remove
# @brief Functions to clean up the environment.

import os

## 
# @param directory (str): The directory to search for .msh files.
def remove_msh_files(directory: str):
    """
    Remove all .msh files in the specified directory and its subdirectories.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.msh'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

def reset_environment():
    """
    Reset the environment by removing all .msh files and deleting all contents inside the data folder.
    """
    # Remove all .msh files
    remove_msh_files("data/mshfiles")
    
    # Delete all contents inside data folder
    data_folder = "data"
    for root, dirs, files in os.walk(data_folder, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")