# Required Libraries
import numpy  as np
import matplotlib.pyplot as plt
from pyMetaheuristic.test_function import easom, ackley, axis_parallel_hyper_ellipsoid, beale, bohachevsky_1, bohachevsky_2, bohachevsky_3, booth, branin_rcos, bukin_6, cross_in_tray, de_jong_1, drop_wave, eggholder, griewangk_8, himmelblau, holder_table, matyas, mccormick, levi_13, rastrigin, rosenbrocks_valley, schaffer_2, schaffer_6, six_hump_camel_back, styblinski_tang, three_hump_camel_back, zakharov
import csv
############################################################################

# Problem: The algorithm faces inefficiency which means that it run longer than necessary even if the optimal solution is already achieved

############################################################################

# Function
def target_function():
    return

target_functions = {
    "ea": easom,
    "ack": ackley,
    "aphe": axis_parallel_hyper_ellipsoid,
    "be": beale,
    "bo1": bohachevsky_1,
    "bo2": bohachevsky_2,
    "bo3": bohachevsky_3,
    "boo": booth,
    "cit": cross_in_tray,
    "dj1": de_jong_1,
    "dw": drop_wave,
    "eh": eggholder,
    "g8": griewangk_8,
    "hb": himmelblau,
    "ht": holder_table,
    "mat": matyas,
    "mcc": mccormick,
    "l13": levi_13,
    "ra": rastrigin,
    "rv": rosenbrocks_valley,
    "sch2": schaffer_2,
    "sch6": schaffer_6,
    "shcb": six_hump_camel_back,
    "st": styblinski_tang,
    "thcb": three_hump_camel_back,
    "zak": zakharov
}

target_values = {
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

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

############################################################################

# Function: Initialize Alpha
def alpha_position(min_values, max_values, target_function):
    alpha       = np.zeros((1, len(min_values) + 1))
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

# Function: Initialize Beta
def beta_position(min_values, max_values, target_function):
    beta       = np.zeros((1, len(min_values) + 1))
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

# Function: Initialize Delta
def delta_position(min_values, max_values, target_function):
    delta       =  np.zeros((1, len(min_values) + 1))
    delta[0,-1] = target_function(np.clip(delta[0,0:delta.shape[1]-1], min_values, max_values))
    return delta[0,:]

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    idx   = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta  = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta

# Function: Update Position
def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function):
    dim                     = len(min_values)
    alpha_position          = np.copy(position)
    beta_position           = np.copy(position)
    delta_position          = np.copy(position)
    updated_position        = np.copy(position)
    r1                      = np.random.rand(position.shape[0], dim)
    r2                      = np.random.rand(position.shape[0], dim)
    a                       = 2 * a_linear_component * r1 - a_linear_component
    c                       = 2 * r2
    distance_alpha          = np.abs(c * alpha[:dim] - position[:, :dim])
    distance_beta           = np.abs(c * beta [:dim] - position[:, :dim])
    distance_delta          = np.abs(c * delta[:dim] - position[:, :dim])
    x1                      = alpha[:dim] - a * distance_alpha
    x2                      = beta [:dim] - a * distance_beta
    x3                      = delta[:dim] - a * distance_delta
    alpha_position[:,:-1]   = np.clip(x1, min_values, max_values)
    beta_position [:,:-1]   = np.clip(x2, min_values, max_values)
    delta_position[:,:-1]   = np.clip(x3, min_values, max_values)
    alpha_position[:, -1]   = np.apply_along_axis(target_function, 1, alpha_position[:, :-1])
    beta_position [:, -1]   = np.apply_along_axis(target_function, 1, beta_position [:, :-1])
    delta_position[:, -1]   = np.apply_along_axis(target_function, 1, delta_position[:, :-1])
    updated_position[:,:-1] = np.clip((alpha_position[:, :-1] + beta_position[:, :-1] + delta_position[:, :-1]) / 3, min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position        = np.vstack([position, updated_position, alpha_position, beta_position, delta_position])
    updated_position        = updated_position[updated_position[:, -1].argsort()]
    updated_position        = updated_position[:position.shape[0], :]
    return updated_position

############################################################################

# Function: Distance Calculations
def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Improve Position
def improve_position(position, updt_position, min_values, max_values, target_function):
    i_position  = np.copy(position)
    dist_matrix = build_distance_matrix(position[:, :-1])
    min_values  = np.array(min_values)
    max_values  = np.array(max_values)
    for i in range(position.shape[0]):
        dist = euclidean_distance(position[i, :-1], updt_position[i, :-1])
        idx  = np.where(dist_matrix[i, :] <= dist)[0]
        for j in range(len(min_values)):
            rand             = np.random.rand()
            ix_1             = np.random.choice(idx)
            ix_2             = np.random.choice(position.shape[0])
            i_position[i, j] = np.clip(i_position[i, j] + rand * (position[ix_1, j] - position[ix_2, j]), min_values[j], max_values[j])
        i_position[i, -1] = target_function(i_position[i, :-1])
        min_fitness       = min(position[i, -1], updt_position[i, -1], i_position[i, -1])
        if (updt_position[i, -1] == min_fitness):
            i_position[i, :] = updt_position[i, :]
        elif (position[i, -1] == min_fitness):
            i_position[i, :] = position[i, :]
    return i_position

############################################################################

# Function: iGWO
def improved_grey_wolf_optimizer(pack_size=25, min_values=[-100, -100], max_values=[100, 100], iterations=500, target_function=target_function, verbose=True, start_init=None, target_value=None):
    alpha = alpha_position(min_values, max_values, target_function)
    beta = beta_position(min_values, max_values, target_function)
    delta = delta_position(min_values, max_values, target_function)
    position = initial_variables(pack_size, min_values, max_values, target_function, start_init)
    
    # computation of the moving average
    w = 5
    threshold = 3
    fitness_history = []
    moving_average_list = []
    
    count = 0
    iteration_count = 0
    found_iteration = []
    convergence_curve = []  # To store the best fitness value at each iteration
    while count <= iterations:
        if verbose:
            print('Iteration =', count, 'f(x) =', alpha[0], alpha[1], ' = ', alpha[-1])
        
        convergence_curve.append(alpha[-1])  # Store the best fitness value
        fitness_history.append(alpha[-1])
        moving_average = 0
        
        if len(fitness_history) >= w:
            moving_average = sum(fitness_history[-w:]) / w * 2
            moving_average_list.append(moving_average)
            if verbose:
                print(f"Moving average: {moving_average}")
        
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)
        position = improve_position(position, updt_position, min_values, max_values, target_function)
        
        if target_value is not None and alpha[-1] <= target_value:
            found_iteration.append(iteration_count)
        
        # check if the moving_average is same from the previous one
        if moving_average_list and moving_average_list[-1] == moving_average:
            count += threshold
        else:
            count += 1
            
        iteration_count += 1
    

    
    if found_iteration:
        print('Optimum solution found at iteration:', found_iteration[0])
        # number of redundant iterations
        print('Number of redundant iterations:', len(found_iteration) - 1)
    else:
        print('Optimum solution not found within the given iterations.')
    
    return alpha, found_iteration

############################################################################

def main():
	# iGWO - Parameters
	parameters = {
		'pack_size': 25,
		'min_values': (-50, -50),
		'max_values': (50, 50),
		'iterations': 2000,
		'verbose': False,
		'start_init': None,
		'target_value': None
	}
 
	redundant_iterations = []
 
	# loop through all target functions
	for target_function_name, target_function in target_functions.items():
		parameters['target_value'] = target_values[target_function_name]
		print(target_function_name, parameters['target_value'])
		gwo, found_iteration = improved_grey_wolf_optimizer(target_function=target_function, **parameters)
		redundant_iterations.append(len(found_iteration) - 1)

	# Save redundant iterations to a CSV file
	with open('obj2/new/redundant_iterations.csv', mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Target Function', 'Redundant Iterations'])
		for target_function_name, redundant_iteration in zip(target_functions.keys(), redundant_iterations):
			writer.writerow([target_function_name, redundant_iteration])

	# Plot the results for all target functions
	plt.figure()
	plt.plot(redundant_iterations)
	plt.xlabel('Target Function')
	plt.ylabel('Number of Redundant Iterations')
	plt.xticks(range(len(target_functions)), list(target_functions.keys()), rotation=45)
	plt.title('Redundant Iterations for All Target Functions')
	plt.grid(True)
	plt.tight_layout()
	# plt.show()
	plt.savefig('obj2/new/all_redundant_iterations.png')

	# Print Solution
	variables = gwo[:-1]
	minimum   = gwo[ -1]
	print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )
	print(gwo)
    

    
	

if __name__ == "__main__":
    main()
