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

if __name__ == "__main__":
    directory = "/home/ale/pacs-project"
    remove_msh_files(directory)