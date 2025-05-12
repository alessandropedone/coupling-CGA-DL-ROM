import pandas as pd
from pathlib import Path

# Define paths
temp_folder = Path("data/temp")
dataset_file = Path("data/dataset.csv")

# Read the main dataset once
dataset_df = pd.read_csv(dataset_file)

# Collect updates from all temp files
for temp_file in sorted(temp_folder.glob("*.csv"), key=lambda x: int(x.stem.split('_')[-1])):
    index = int(temp_file.stem.split('_')[-1]) - 1  # Adjust to 0-based index
    temp_df = pd.read_csv(temp_file)
    
    # Update the corresponding row
    if 0 <= index < len(dataset_df):
        dataset_df.iloc[index] = temp_df.iloc[index]
    print(f"Updated row {index + 1} from {temp_file.name}")

# Write the dataset once after all updates
dataset_df.to_csv(dataset_file, index=False)

# Clean up temp files and folder
for temp_file in temp_folder.glob("*.csv"):
    temp_file.unlink()
temp_folder.rmdir()
