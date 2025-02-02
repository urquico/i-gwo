# read all files in the results folder
import os
import csv
# Get all files in the results folder
results_folder = 'results/user_5/csv'

# Read all files in the results folder
files = os.listdir(results_folder)

scores = []

# Print all files
for file in files:
    total_score = 0.0
    count = 0
    
    # read the file
    with open(os.path.join(results_folder, file), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            total_score += float(row[2])
            count += 1
            
    # print the average score if count is not zero
    if count > 0:
        scores.append(total_score / count)
        print(f"{file}: {total_score / count}")
    else:
        print(f"No valid scores found in {file}")

# get the best score
best_score = max(scores)
print(f"Best score: {best_score}")

# get the file with the best score
best_file = files[scores.index(best_score)]
print(f"Best file: {best_file}")