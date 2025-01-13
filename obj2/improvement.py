# compute the improvement of the new algorithm over the old algorithm

import matplotlib.pyplot as plt
import pandas as pd

# read data from old redundant_iterations.csv
old_data = pd.read_csv('obj2/old/redundant_iterations.csv')

# read data from new redundant_iterations.csv
new_data = pd.read_csv('obj2/new/redundant_iterations.csv')

# compute the percentage improvement then save it to a csv file 
improvement = (old_data['Redundant Iterations'] - new_data['Redundant Iterations']) / old_data['Redundant Iterations'] * 100
improvement.to_csv('obj2/improvement.csv', index=False)

