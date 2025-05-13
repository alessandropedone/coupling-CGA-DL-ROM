import os

def remove_msh_files(directory):
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
    # Remove all .msh files
    remove_msh_files("data/mshfiles")
    
    # Delete all contents of the data folder
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