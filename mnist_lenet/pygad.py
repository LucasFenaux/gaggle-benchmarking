import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
from pyGAD import pygad
from TorchGA import torchga
import torch
from src.problem.dataset import MNIST
from src.base_nns.lenet import LeNet5
from src.arguments import ProblemArgs


def accuracy(y_pred, y) -> float:
    return (y_pred.argmax(1) == y).float().mean()


problem_args = ProblemArgs()

device = torch.device("cuda")

train_dataset = MNIST(problem_args, train=True)
train_data, train_transforms = train_dataset.get_data_and_transform()
data_input, data_target = train_data
data_input, data_target = data_input.to(device), data_target.to(device)

test_dataset = MNIST(problem_args, train=False)
test_data, test_transforms = test_dataset.get_data_and_transform()

model = LeNet5().to(device)

times = []


def fitness_func(solution, sol_idx):
    with torch.no_grad():
        transformed_data_input = train_transforms(data_input)
        predictions = torchga.predict(model=model,
                                      solution=solution,
                                      data=transformed_data_input)

        acc = accuracy(predictions, data_target).detach().cpu().item()*100.

    return acc


def callback_generation(ga_instance):
    global last_gen_time, times
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    cur_time = time.time()
    time_taken = cur_time - last_gen_time
    print(f"Time taken = {time_taken}")
    last_gen_time = cur_time
    times.append(time_taken)


num_solutions = 200
num_generations = 100
num_parents_mating = 200
keep_parents = 0
keep_elitism = 0
crossover = "uniform"
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

test_inputs, test_targets = test_data[0].to(device), test_data[1].to(device)
test_data_inputs = test_transforms(test_inputs)

# Make predictions based on the best solution.
predictions = torchga.predict(model=model,
                                    solution=solution,
                                    data=test_data_inputs)
print("Predictions : \n", predictions.cpu().detach().numpy())

abs_error = accuracy(predictions, test_targets)
print("Accuracy : ", abs_error.cpu().detach().numpy())

print(f"Times: {times}")

print(f"Total program time: {time.time() - beginning_time}")