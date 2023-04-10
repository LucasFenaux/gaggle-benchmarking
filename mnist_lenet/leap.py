"""An example of solving a reinforcement learning problem by using evolution to
tune the weights of a neural network."""
import time
beginning_time = time.time()
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP/leap_ec'))

import torch


from src.data import MNIST
from src.base_nns.lenet import LeNet5
from src.arguments import ProblemArgs, OutdirArgs

from leap_ec import Individual, Representation, test_env_var, Decoder
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea

from leap_ec.real_rep.initializers import create_real_vector
from new_leap_operators import PytorchDecoder, ClassificationProblem, accuracy, TimingProbe, mutate_uniform, build_probes
import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--population-size", dest="population_size", default=200, type=int)
    return parser

##############################
# Entry point
##############################
if __name__ == '__main__':
    problem_args = ProblemArgs()
    args = get_arg_parser().parse_args()

    device = torch.device("cuda")

    train_dataset = MNIST(ProblemArgs, train=True)
    train_data, train_transforms = train_dataset.get_data_and_transform()
    data_input, data_target = train_data
    data_input, data_target = data_input.to(device), data_target.to(device)

    test_dataset = MNIST(problem_args, train=False)
    test_data, test_transforms = test_dataset.get_data_and_transform()

    model = LeNet5().to(device)

    # Parameters
    runs_per_fitness_eval = 1
    pop_size = args.population_size
    print(pop_size)
    gui = False  # Change to true to watch the cart-pole visualization
    low = -1.
    high = 1.

    generations = 100

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
                                ops.uniform_crossover(p_xover=0.5),
                                mutate_uniform(low=low, high=high, expected_num_mutations=617),
                                ops.evaluate,
                                ops.pool(size=pop_size),
                                timing_probe,  # we're nice we don't include all of their extra logging in the computation
                                *build_probes(genomes_file)  # Inserting all the probes at the end
                            ])
        list(ea)

    print(timing_probe.buffer)

    print(f"Total program time: {time.time() - beginning_time}")
