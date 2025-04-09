# compute for the average in seconds for the generated data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_normal_randomized(target_function: str):
	"""
	Compare the normal and randomized methods for the given target function.
	
	Args:
		target_function (str): The name of the target function.
	"""
	# Placeholder for the actual implementation
	# This function should generate data and save it to a CSV file
	pass

	# the data is generated in convergence_data_{target_function}.csv
	# read the data
	data = pd.read_csv(f'obj1/convergence_data_{target_function}.csv')

	# data consists of 2 columns, Normal and Randomized, the content of the columns are the time in seconds

	# store the data in a list
	normal = data['Normal'].tolist()
	randomized = data['Randomized'].tolist()
 
	# calculate the average	
	average_normal = sum(normal) / len(normal)
	average_randomized = sum(randomized) / len(randomized)
	improvement = (average_normal - average_randomized) / average_normal * 100
 
	print(f'Average Normal: {average_normal:.2f} seconds')
	print(f'Average Randomized: {average_randomized:.2f} seconds')
	print(f'Improvement: {improvement:.2f}%')
 
if __name__ == "__main__":
	compare_normal_randomized("eggholder")
	compare_normal_randomized("holder_table")
	compare_normal_randomized("mccormick")
 
 