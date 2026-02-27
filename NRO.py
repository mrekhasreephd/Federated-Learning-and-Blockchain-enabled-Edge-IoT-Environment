import time
import numpy as np
from mpmath import norm
def NRO(population,sphere_function,lower_limit,upper_limit,max_iterations):
    population_size,num_variables = population.shape[0],population.shape[1]
    alpha = 0.1
    beta = 0.5
    gamma = 0.1
    # Initialization
    population = lower_limit + (upper_limit - lower_limit) * np.random.rand(population_size,num_variables)
    convergence = np.zeros((max_iterations))
    # Main loop
    ct = time.time()
    for iter in np.arange(1,max_iterations+1).reshape(-1):
        # Evaluate objective function for each individual
        fitness = np.zeros((population_size,1))
        for i in np.arange(1,population_size+1).reshape(-1):
            fitness[i] = sphere_function(population[i,:])
        # Sort population based on fitness
        fitness,sorted_indices = np.sort(fitness)
        population = population[sorted_indices,:]
        # Update each individuals position
        for i in np.arange(1,population_size+1).reshape(-1):
            for j in np.arange(1,num_variables+1).reshape(-1):
                for k in np.arange(1,population_size+1).reshape(-1):
                    if k != i:
                        r = norm(population[k,:] - population[i,:])
                        direction = (population(k,j) - population(i,j)) / r
                        population[i,j] = population(i,j) + alpha * beta * direction + gamma * np.random.rand() * (upper_limit - lower_limit)
            # Ensure the new position is within bounds
            population[i,:] = np.amax(np.amin(population[i,:],upper_limit),lower_limit)
            # Find the best solution
        best_solution = population[1, :]
        best_fitness = sphere_function(best_solution)
        convergence[iter] = best_fitness
    ct = time.time()-ct
    return best_fitness,convergence,best_solution,ct



