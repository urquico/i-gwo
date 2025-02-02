import matplotlib.pyplot as plt
import pandas as pd

def plot_and_save(data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Normal'], label='Normal', marker='o')
    plt.plot(data['Randomized'], label='Randomized', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Load the data
data1 = pd.read_csv('convergence_data_eggholder.csv')
data2 = pd.read_csv('convergence_data_holder_table.csv')
data3 = pd.read_csv('convergence_data_mccormick.csv')

# Plot and save the data
plot_and_save(data1, 'Normal vs Randomized Values (Eggholder)', 'plots/egg_holder_normal_vs_randomized.png')
plot_and_save(data2, 'Normal vs Randomized Values (Holder Table)', 'plots/holder_table_normal_vs_randomized.png')
plot_and_save(data3, 'Normal vs Randomized Values (McCormick)', 'plots/mccormick_normal_vs_randomized.png')