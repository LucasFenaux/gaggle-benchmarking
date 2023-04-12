import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
from pyGAD import pygad
import numpy as np
import pickle
import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--dimension", dest="dimension", default=10, type=int)
    return parser

times = []

def fitness_func(solution, sol_idx):
    return -(args.dimension * len(solution) + np.sum(solution ** 2 - args.dimension * np.cos(2 * np.pi * solution)))

def mod_roulette_wheel_selection(fitness, num_parents, ga_instance):

        """
        Modified from PyGAD
        """
        min_fit = np.abs(np.min(fitness))
        fitness_temp = fitness + min_fit
        fitness_sum = np.sum(fitness_temp)
        if fitness_sum == 0:
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
        probs = fitness_temp / fitness_sum
        probs_start = np.zeros(probs.shape, dtype=np.float) # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape, dtype=np.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if ga_instance.gene_type_single == True:
            parents = np.empty((num_parents, ga_instance.population.shape[1]), dtype=ga_instance.gene_type[0])
        else:
            parents = np.empty((num_parents, ga_instance.population.shape[1]), dtype=object)
        
        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = ga_instance.population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, np.array(parents_indices)

def callback_generation(ga_instance):
    global last_gen_time, times
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    cur_time = time.time()
    time_taken = cur_time - last_gen_time
    print(f"Time taken = {time_taken}")
    last_gen_time = cur_time
    times.append(time_taken)

args = get_arg_parser().parse_args()

num_solutions = 200
num_generations = 100
num_parents_mating = 200
keep_parents = 0
keep_elitism = 0
crossover = "uniform"
mutation_type = "random"
mutation_probability = 0.01
random_mutation_min_val = -5.12
random_mutation_max_val = 5.12

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       parent_selection_type=mod_roulette_wheel_selection,
                       init_range_high=5.12,
                       init_range_low=-5.12,
                       sol_per_pop=200,
                       num_genes=args.dimension)

last_gen_time = time.time()

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


print(f"Times: {times}")

dir = 'Results/'
filename = 'pygad_dimension_{}.p'.format(args.dimension)
if not os.path.exists(dir):
    os.makedirs(dir)
with open(os.path.join(dir,filename), 'wb') as f:
    pickle.dump(times, f)
    
print(f"Total program time: {time.time() - beginning_time}")