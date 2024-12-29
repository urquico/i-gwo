import csv
import numpy as np
from scipy.stats import ttest_ind

def load_convergence_data(file_name):
	normal_convergence = []
	randomized_convergence = []

	with open(file_name, 'r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			normal_convergence.append(float(row['Normal']))
			randomized_convergence.append(float(row['Randomized']))

	return np.array(normal_convergence), np.array(randomized_convergence)

def perform_t_test(normal_data, randomized_data):
	t_stat, p_value = ttest_ind(normal_data, randomized_data)
	print(f"T-test: t-statistic = {t_stat}, p-value = {p_value}")

	alpha = 0.05
	if p_value < alpha:
		print("The difference in convergence iterations is statistically significant.")
	else:
		print("The difference in convergence iterations is not statistically significant.")

def main():
	# File names for the CSV files
	file_names = ['obj1/convergence_data_eggholder.csv', 
				  'obj1/convergence_data_holder_table.csv', 
				  'obj1/convergence_data_mccormick.csv']

	for file_name in file_names:
		print(f"\nPerforming t-test for {file_name}...")
		normal_data, randomized_data = load_convergence_data(file_name)
		perform_t_test(normal_data, randomized_data)

if __name__ == "__main__":
	main()
