"""An example of solving a reinforcement learning problem by using evolution to
tune the weights of a neural network."""
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP/leap_ec'))

import torch
from matplotlib import pyplot as plt
from typing import Dict
import numpy as np
import time

from src.data import MNIST
from src.base_nns.lenet import LeNet5
from src.arguments import DatasetArgs, OutdirArgs

from leap_ec import Individual, Representation, test_env_var, Decoder
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep import problems, executable, neural_network
from leap_ec.int_rep.ops import individual_mutate_randint
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec.global_vars import context
from leap_ec.ops import compute_expected_probability, iteriter_op
from toolz import curry
from collections.abc import Iterable
from typing import Iterator, List, Tuple, Union
import math

from TorchGA import torchga


def accuracy(y_pred, y) -> float:
    return (y_pred.argmax(1) == y).float().mean()


@curry
def genome_mutate_uniform(genome,
                           expected_num_mutations,
                           low=-1., high=1.):

    assert(expected_num_mutations is not None)

    if not isinstance(genome, np.ndarray):
        raise ValueError(("Expected genome to be a numpy array. "
                          f"Got {type(genome)}."))

    # compute actual probability of mutation based on expected number of
    # mutations and the genome length
    if expected_num_mutations == 'isotropic':
        # Default to isotropic Gaussian mutation
        p = 1.0
    else:
        p = compute_expected_probability(expected_num_mutations, genome)

    # select which indices to mutate at random
    selector = np.random.choice([0, 1], size=genome.shape, p=(1 - p, p))
    indices_to_mutate = np.nonzero(selector)[0]

    genome[indices_to_mutate] += np.random.uniform(low=low, high=high, size=indices_to_mutate.shape[0])

    return genome


@curry
@iteriter_op
def mutate_uniform(next_individual: Iterator,
                    expected_num_mutations: Union[int, str] = None, low=-1., high=1.) -> Iterator:

    if expected_num_mutations is None:
        raise ValueError("No value given for expected_num_mutations.  Must be either a float or the string 'isotropic'.")
    while True:
        individual = next(next_individual)

        individual.genome = genome_mutate_uniform(individual.genome, expected_num_mutations, low, high)
        # invalidate fitness since we have new genome
        individual.fitness = None

        yield individual


# what we want to define our problem: subclass a function problem
class ClassificationProblem(FunctionProblem):
    def __init__(self, fitness_function, train_data: torch.Tensor, train_targets: torch.Tensor,
                 train_transform, device, maximize: bool = True):  # maximize is true if fitness is accuracy and false if it's loss
        super(ClassificationProblem, self).__init__(fitness_function, maximize)
        self.train_data = train_data.to(device)
        self.train_targets = train_targets.to(device)
        self.train_transform = train_transform
        self.device = device

    @torch.no_grad()
    def evaluate(self, phenome: torch.nn.Module):
        phenome = phenome.to(self.device)
        predictions = phenome(self.train_transform(self.train_data))

        return accuracy(predictions, self.train_targets).detach().cpu().item()


class TimingProbe(ops.Operator):
    def __init__(self, context: Dict = context):
        self.context = context
        self.buffer = []
        self.timestamp = None
        self.start = True

    def __call__(self, population, *args, **kwargs):
        if self.start:
            self.timestamp = time.time()
            self.start = False
        else:
            interval = time.time() - self.timestamp
            print(interval)
            self.buffer.append(interval)
            self.start = True
        return population


class PytorchDecoder(Decoder):
    def __init__(self, model_fn, device, *args, **kwargs):
        self.model_fn = model_fn
        self.length = len(torchga.model_weights_as_vector(self.model_fn(*args, **kwargs)))
        self.device = device
        self.args = args
        self.kwargs = kwargs

    def decode(self, genome, *args, **kwargs):
        model = self.model_fn(*self.args, **self.kwargs)
        model.load_state_dict(torchga.model_weights_as_dict(model, genome))
        return model.to(self.device)


##############################
# Function build_probes()
##############################
def build_probes(genomes_file):
    """Set up probes for writings results to file and terminal and
    displaying live metric plots."""
    assert(genomes_file is not None)

    probes = []

    # Print fitness stats to stdout
    probes.append(probe.FitnessStatsCSVProbe(stream=sys.stdout))

    # Save genome of the best individual to a file
    probes.append(probe.AttributesCSVProbe(
                  stream=genomes_file,
                  best_only =True,
                  do_fitness=True,
                  do_genome=True))

    # Open a figure to plot a fitness curve to
    plt.figure()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.title("Best-of-Generation Fitness")
    probes.append(probe.FitnessPlotProbe(
                        ylim=(0, 1), xlim=(0, 1),
                        modulo=1, ax=plt.gca()))

    return probes


##############################
# Entry point
##############################
if __name__ == '__main__':
    dataset_args = DatasetArgs()

    device = torch.device("cpu")

    train_dataset = MNIST(dataset_args, train=True)
    train_data, train_transforms = train_dataset.get_data_and_transform()
    data_input, data_target = train_data
    data_input, data_target = data_input.to(device), data_target.to(device)

    test_dataset = MNIST(dataset_args, train=False)
    test_data, test_transforms = test_dataset.get_data_and_transform()

    model = LeNet5().to(device)

    # Parameters
    runs_per_fitness_eval = 1
    pop_size = 20
    gui = False  # Change to true to watch the cart-pole visualization
    low = -1.
    high = 1.

    generations = 10

    # Load the OpenAI Gym simulation

    # Representation

    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = PytorchDecoder(LeNet5, device)
    problem = ClassificationProblem(fitness_function=accuracy, train_data=data_input, train_targets=data_target,
                                    train_transform=train_transforms, device=device)
    timing_probe = TimingProbe()
    with open('./genomes.csv', 'w') as genomes_file:

        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                            # Solve a problem that executes agents in the
                            # environment and obtains fitness from it
                            problem=problem,

                            representation=Representation(
                                initialize=create_real_vector(bounds=([[-1, 1]]*decoder.length)),
                                decoder=decoder),

                            # The operator pipeline.
                            pipeline=[
                                timing_probe,
                                ops.proportional_selection,
                                ops.clone,
                                ops.uniform_crossover,
                                mutate_uniform(low=low, high=high, expected_num_mutations=600),
                                ops.evaluate,
                                ops.pool(size=pop_size),
                                timing_probe,  # we're nice we don't include all of their extra logging in the computation
                                *build_probes(genomes_file)  # Inserting all the probes at the end
                            ])
        list(ea)

    print(timing_probe.buffer)