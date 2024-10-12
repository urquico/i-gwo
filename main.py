
from base_i_gwo import improved_grey_wolf_optimizer

# Import a Test Function. Available Test Functions: https://bit.ly/3KyluPp
from pyMetaheuristic.test_function import easom
import numpy as np

def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1)*np.cos(x2)*np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return func_value

# iGWO - Parameters
parameters = {
    'pack_size': 2,
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'iterations': 10,
	'verbose': True,
	'start_init': None,
	'target_value': None
}


gwo = improved_grey_wolf_optimizer(target_function = easom, **parameters)

# Print Solution
variables = gwo[:-1]
minimum   = gwo[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )
print(gwo)

# Plot Solution
# from pyMetaheuristic.utils import graphs
# plot_parameters = {
#     'min_values': (-5, -5),
#     'max_values': (5, 5),
#     'step': (0.1, 0.1),
#     'solution': [variables],
#     'proj_view': '3D',
#     'view': 'surface'
# }
# graphs.plot_single_function(target_function = easom, **plot_parameters)
