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
        # print(f"{file}: {total_score / count}")
    else:
        print(f"No valid scores found in {file}")

# get the 10 best scores
best_scores = sorted(scores, reverse=True)[:10]

# get the files with the best scores
best_files = [files[scores.index(score)] for score in best_scores]

# print the best files
for file in best_files:
    print(f"{file}: {scores[files.index(file)]}")
    
# save the best files and scores to a csv file
with open('best_files.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['File', 'Score'])
    for file, score in zip(best_files, best_scores):
        writer.writerow([file, score])
