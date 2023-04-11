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
    parser.add_argument("--dimension", dest="dimension", default=1000, type=int)
    return parser

times = []

def fitness_func(solution, sol_idx):
    return 1 / (args.dimension * len(solution) + np.sum(solution ** 2 - args.dimension * np.cos(2 * np.pi * solution)))



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
                       parent_selection_type="rws",
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