# Required Libraries
import numpy  as np
import matplotlib.pyplot as plt
from pyMetaheuristic.test_function import eggholder, holder_table, mccormick

############################################################################

# Problem: The existing algorithm's alpha, beta, and delta wolves are initialized to zero  which might introduce bias, especially if the search space is large or not centered around zero.
# Solution: Create an option to randomize the initial positions of the alpha, beta, and delta wolves.
# ! Egg Holder = [9.49, 10.0]
# ! Holder Table = [-8.05, 9.66]
# ! McCormick = [-9.61, -10]

# ? Tests how quickly the algorithm converges to a solution by tracking the number of iterations it takes to reach a certain fitness value.

############################################################################

# Function
def target_function():
    return

############################################################################

# Global Variables
target_value_holder_table = -19.2085
target_value_eggholder = -62.4327
target_value_mccormick =  -10.1221

target_position_holder_table = [-8.05, 9.66]
target_position_eggholder = [9.49, 10.0]
target_position_mccormick = [-9.61, -10]

# change this to test different target function
target_value = target_value_mccormick
target_position = target_position_mccormick
target_function = mccormick

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

# ! Solution: randomize wolves position

def random_alpha_position(min_values, max_values, target_function):
    dim = len(min_values)
    alpha = np.random.uniform(min_values, max_values, (1, dim))
    alpha = np.hstack((alpha, [[target_function(np.clip(alpha[0], min_values, max_values))]]))
    return alpha[0,:]

# Function: Initialize Beta
def random_beta_position(min_values, max_values, target_function):
    dim = len(min_values)
    beta = np.random.uniform(min_values, max_values, (1, dim))
    beta = np.hstack((beta, [[target_function(np.clip(beta[0], min_values, max_values))]]))
    return beta[0,:]

# Function: Initialize Delta
def random_delta_position(min_values, max_values, target_function):
    dim = len(min_values)
    delta = np.random.uniform(min_values, max_values, (1, dim))
    delta = np.hstack((delta, [[target_function(np.clip(delta[0], min_values, max_values))]]))
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
def plot_initial_positions(alpha, beta, delta):
    # Extract positions
    alpha_pos = alpha[:-1]  # Exclude the fitness value
    beta_pos = beta[:-1]
    delta_pos = delta[:-1]
    
    # ! plot target position
    target_pos = target_position

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(alpha_pos[0], alpha_pos[1], color='r', label='Alpha', s=100)
    plt.scatter(beta_pos[0], beta_pos[1], color='g', label='Beta', s=100)
    plt.scatter(delta_pos[0], delta_pos[1], color='b', label='Delta', s=100)

    # Plot target value as a horizontal line
    # if target_pos is not None:
    plt.axvline(x=target_pos[0], color='k', linestyle='--', label=f'Target X Value: {target_pos[0]}')
    plt.axhline(y=target_pos[1], color='k', linestyle='--', label=f'Target Y Value: {target_pos[1]}')

    plt.title('Initial Positions of Wolves')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function: iGWO
def improved_grey_wolf_optimizer(initialize_random=False, pack_size=25, min_values=[-100, -100], max_values=[100, 100], iterations=500, target_function=target_function, verbose=True, start_init=None, target_value=None):
    alpha, beta, delta = None, None, None
    if initialize_random:
        alpha = random_alpha_position(min_values, max_values, target_function)
        
        # Ensure beta is different from alpha
        while True:
            beta = random_beta_position(min_values, max_values, target_function)
            if not np.array_equal(beta[:-1], alpha[:-1]):
                break
        
        # Ensure delta is different from both alpha and beta
        while True:
            delta = random_delta_position(min_values, max_values, target_function)
            if (not np.array_equal(delta[:-1], alpha[:-1]) and 
                not np.array_equal(delta[:-1], beta[:-1])):
                break
    else:
        alpha = alpha_position(min_values, max_values, target_function)
        beta = beta_position(min_values, max_values, target_function)
        delta = delta_position(min_values, max_values, target_function)
    
    position = initial_variables(pack_size, min_values, max_values, target_function, start_init)
    
    # Initialize convergence speed tracking
    convergence_iteration = None
    
    count = 0
    while count <= iterations:
        if verbose:
            print('Iteration = ', count, alpha[0], alpha[1], ' = ', alpha[-1])
        
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)
        position = improve_position(position, updt_position, min_values, max_values, target_function)
        
        # Check for convergence
        if target_value is not None and alpha[-1] <= target_value:
            convergence_iteration = count
            break
        
        count += 1
    
    # Return the best solution and convergence iteration
    return alpha, convergence_iteration

############################################################################
def main():
	# iGWO - Parameters
	parameters = {
		'pack_size': 25,
		'min_values': (-10, -10),
		'max_values': (10, 10),
		'iterations': 1000,
		'verbose': False,
		'start_init': None,
		'target_value': target_value,
		'initialize_random': True # change this to simulate random initialization
	}

	# Run 1000 times and get the average convergence iteration
    # then save the 1000 convergence iterations to a file
	convergence_iterations = []
	for _ in range(1000):
		gwo, convergence_iteration = improved_grey_wolf_optimizer(target_function=target_function, **parameters)
		if convergence_iteration is not None:
			convergence_iterations.append(convergence_iteration)

	if convergence_iterations:
		average_convergence_iteration = np.mean(convergence_iterations)
		print('Average Convergence Iteration: ', average_convergence_iteration)
	else:
		print('No convergence achieved in any of the runs.')

	# Print Solution
	variables = gwo[:-1]
	minimum = gwo[-1]
	print('Variables: ', np.around(variables, 4), ' Minimum Value Found: ', round(minimum, 4))
	print('Average Convergence Iteration: ', average_convergence_iteration)

if __name__ == "__main__":
    main()
