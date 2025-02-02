import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

DIRECTORY_PATH = 'results/user_5/csv/'
regularization_range = [0.001, 0.01, 0.1, 1.0]

def get_files_with_substring(directory: str, substring: str) -> list[str]:
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        
        # Filter files that contain the specified substring
        filtered_files = [file for file in files if substring in file]
        
        return filtered_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def get_files_with_factor(files: list[str], factor: str) -> list[str]:
    try:
        # Filter files that contain the specified factor
        filtered_files = [file for file in files if f"factors_{factor}_" in file]
        
        return filtered_files
    except Exception as e:
        print(f"An error occurred while filtering files by factor: {e}")
        return []

# Example usage
substring = 'regularization_' + str(regularization_range[1])
filtered_files = get_files_with_substring(DIRECTORY_PATH, substring)

# Remove duplicates
filtered_files = list(set(filtered_files))

# Get files with a specific factor
specific_factor = "10"  # Change this to the desired factor
files_with_specific_factor = get_files_with_factor(filtered_files, specific_factor)
print(files_with_specific_factor)

# sort files by iterations

for file in files_with_specific_factor:
    print(file)



