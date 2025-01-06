# get the data from the normal and parallel execution time
# plot the graph
# save the graph

import matplotlib.pyplot as plt
import pandas as pd

# read data from normal execution_time_10_20_1000.csv
normal_data_10_20_1000 = pd.read_csv('obj3/normal/execution_time_10_20_1000.csv')
normal_data_50_40_1000 = pd.read_csv('obj3/normal/execution_time_50_40_1000.csv')
normal_data_75_80_1000 = pd.read_csv('obj3/normal/execution_time_75_80_1000.csv')
normal_data_100_200_1000 = pd.read_csv('obj3/normal/execution_time_100_200_1000.csv')

# read data from parallel execution_time_10_20_1000.csv
parallel_data_10_20_1000 = pd.read_csv('obj3/parallel/execution_time_10_20_1000.csv')
parallel_data_50_40_1000 = pd.read_csv('obj3/parallel/execution_time_50_40_1000.csv')
parallel_data_75_80_1000 = pd.read_csv('obj3/parallel/execution_time_75_80_1000.csv')
parallel_data_100_200_1000 = pd.read_csv('obj3/parallel/execution_time_100_200_1000.csv')

# store the all data
normal_execution_times = [normal_data_10_20_1000, normal_data_50_40_1000, normal_data_75_80_1000, normal_data_100_200_1000]
parallel_execution_times = [parallel_data_10_20_1000, parallel_data_50_40_1000, parallel_data_75_80_1000, parallel_data_100_200_1000]

print(normal_execution_times)
print(parallel_execution_times)

# Extract the execution time column from each DataFrame
normal_execution_times = [
    normal_data_10_20_1000['Execution Time'],
    normal_data_50_40_1000['Execution Time'],
    normal_data_75_80_1000['Execution Time'],
    normal_data_100_200_1000['Execution Time']
]

parallel_execution_times = [
    parallel_data_10_20_1000['Execution Time'],
    parallel_data_50_40_1000['Execution Time'],
    parallel_data_75_80_1000['Execution Time'],
    parallel_data_100_200_1000['Execution Time']
]

# Flatten the lists of execution times
normal_execution_times_flat = [time for sublist in normal_execution_times for time in sublist]
parallel_execution_times_flat = [time for sublist in parallel_execution_times for time in sublist]

# Extract dimensions and pack size for annotations
dimensions = [
    normal_data_10_20_1000['Dimensions'],
    normal_data_50_40_1000['Dimensions'],
    normal_data_75_80_1000['Dimensions'],
    normal_data_100_200_1000['Dimensions']
]

pack_sizes = [
    normal_data_10_20_1000['Pack Size'],
    normal_data_50_40_1000['Pack Size'],
    normal_data_75_80_1000['Pack Size'],
    normal_data_100_200_1000['Pack Size']
]

# Flatten the lists of dimensions and pack sizes
dimensions_flat = [dim for sublist in dimensions for dim in sublist]
pack_sizes_flat = [size for sublist in pack_sizes for size in sublist]

# Plot the execution times
plt.figure(figsize=(10, 5))
plt.plot(normal_execution_times_flat, label='Normal Execution')  # 'o' for point marker
plt.plot(parallel_execution_times_flat, label='Parallel Execution')  # 'o' for point marker

# Annotate the plot with dimensions and pack sizes
for i, (dim, size) in enumerate(zip(dimensions_flat, pack_sizes_flat)):
    plt.annotate(f'Dim: {dim}, Size: {size}', (i, normal_execution_times_flat[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Index')
plt.ylabel('Execution Time')
plt.title('Normal vs Parallel Execution Times')
plt.legend()
plt.savefig('obj3/combined-graph.png')
# plt.show()