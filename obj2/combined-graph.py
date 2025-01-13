# get the data from the old and new redundant iterations
# plot the graph
# save the graph

import matplotlib.pyplot as plt
import pandas as pd

# read data from old redundant_iterations.csv
old_data = pd.read_csv('obj2/old/redundant_iterations.csv')

# read data from new redundant_iterations.csv
new_data = pd.read_csv('obj2/new/redundant_iterations.csv')

target_functions = {
    "ea": -1,
    "ack": 0.0,
    "aphe": 0.0,
    "be": 4.163,
    "bo1": 0.0,
    "bo2": 0.0,
    "bo3": 0.0,
    "boo": 3.122,
    "cit": -2.062,
    "dj1": 0.0,
    "dw": -1.0,
    "eh": -126.423,
    "g8": 0.0,
    "hb": 2.248,
    "ht": -1618728691.846,
    "mat": 0.0,
    "mcc": -49.037,
    "l13": 4.655,
    "ra": 0.0,
    "rv": 1.092,
    "sch2": 0.0,
    "sch6": 0.0,
    "shcb": -1.031,
    "st": -78.332,
    "thcb": 0.0,
    "zak": 0.0
}

# plot the graph
plt.plot(old_data['Redundant Iterations'], label='Old')
plt.plot(new_data['Redundant Iterations'], label='New')
plt.xlabel('Target Function')
plt.ylabel('Number of Redundant Iterations')
plt.xticks(range(len(target_functions)), list(target_functions.keys()), rotation=45)
plt.title('Redundant Iterations for All Target Functions')
plt.grid(True)
plt.tight_layout()
plt.savefig('obj2/combined-graph.png')