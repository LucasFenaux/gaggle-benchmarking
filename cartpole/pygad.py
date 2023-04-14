import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
from pyGAD import pygad
from TorchGA import torchga
import torch
import pickle
import numpy as np
import argparse
import gym
from models import DQN, LargeDQN


def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--population_size", default=100, type=int)
    parser.add_argument("--model_size", default="tiny", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    return parser



args = get_arg_parser().parse_args()
environment = gym.make('CartPole-v1')

device = args.device
num_inputs = 4
num_outputs = 2
if args.model_size == "tiny":
    hidden_size = 4
elif args.model_size == "small":
    hidden_size = 16
elif args.model_size == "medium":
    hidden_size = 64
elif args.model_size == "large":
    hidden_size = 128
else:
    hidden_size = 256

if args.model_size == "large" or args.model_size == "very_large":
    model = LargeDQN(num_inputs=num_inputs, num_outputs=num_outputs, hidden_size=hidden_size).to(device)
else:
    model = DQN(num_inputs=num_inputs, num_outputs=num_outputs, hidden_size=hidden_size).to(device)
times = []

steps = 10
runs = 5
gui = False
stop_on_done = False


def fitness_func(solution, sol_idx):
    global steps, runs, gui, stop_on_done, environment
    with torch.no_grad():

        observations = []
        rewards = []
        for r in range(runs):
            observation, _ = environment.reset()
            observation = torch.Tensor(observation).to(device)
            run_observations = [observation]
            run_rewards = []
            for t in range(steps):
                if gui:
                    environment.render()
                action = torch.argmax(torchga.predict(model=model,
                                    solution=solution,
                                    data=observation)).cpu().item()
                observation, reward, done, info, _ = environment.step(action)
                observation = torch.Tensor(observation).to(device)
                run_observations.append(observation)
                run_rewards.append(reward)
                if stop_on_done and done:
                    break
            observations.append(run_observations)
            rewards.append(run_rewards)

        sums = [sum(run) for run in rewards]
    return np.mean(sums).item()


def callback_generation(ga_instance):
    global last_gen_time, times
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    cur_time = time.time()
    time_taken = cur_time - last_gen_time
    print(f"Time taken = {time_taken}")
    last_gen_time = cur_time
    times.append(time_taken)



num_solutions = args.population_size
print(num_solutions)
num_generations = 100
num_parents_mating = args.population_size
keep_parents = 0
keep_elitism = 0
crossover = "uniform"
crossover_probability = 0.5
mutation_type = "random"
mutation_probability = 0.01
random_mutation_min_val = -1.
random_mutation_max_val = 1.


torch_ga = torchga.TorchGA(model=model,
                           num_solutions=num_solutions)
initial_population = torch_ga.population_weights  # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       parent_selection_type="rws",)
                       # parallel_processing=8)
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
filename = 'pygad_model_size_{}.p'.format(args.model_size)
if not os.path.exists(dir):
    os.makedirs(dir)
with open(os.path.join(dir,filename), 'wb') as f:
    pickle.dump(times, f)

print(f"Total program time: {time.time() - beginning_time}")