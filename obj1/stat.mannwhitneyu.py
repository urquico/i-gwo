import csv
import numpy as np
from scipy.stats import mannwhitneyu

def load_convergence_data(file_name):
    normal_convergence = []
    randomized_convergence = []

    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normal_convergence.append(float(row['Normal']))
            randomized_convergence.append(float(row['Randomized']))

    return np.array(normal_convergence), np.array(randomized_convergence)

def perform_mann_whitney_u_test(normal_data, randomized_data):
    u_stat, p_value = mannwhitneyu(randomized_data, normal_data, alternative='less')
    print(f"Mann-Whitney U Test: U-statistic = {u_stat}, p-value = {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Random initialization is statistically significantly better than normal initialization.")
    else:
        print("No statistically significant difference between random and normal initialization.")

def main():
    # File names for the CSV files
    file_names = ['obj1/convergence_data_eggholder.csv', 
                  'obj1/convergence_data_holder_table.csv', 
                  'obj1/convergence_data_mccormick.csv']

    for file_name in file_names:
        print(f"\nPerforming Mann-Whitney U test for {file_name}...")
        normal_data, randomized_data = load_convergence_data(file_name)
        perform_mann_whitney_u_test(normal_data, randomized_data)

if __name__ == "__main__":
    main()