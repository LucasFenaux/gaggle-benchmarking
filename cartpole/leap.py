"""An example of solving a reinforcement learning problem by using evolution to
tune the weights of a neural network."""
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP/leap_ec'))

import gym
from matplotlib import pyplot as plt
import torch
from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep import problems, executable, neural_network
from leap_ec.real_rep.initializers import create_real_vector
from new_leap_operators import TimingProbe, mutate_uniform, UpdatedEnvironmentProblem, UpdatedTorchEnvironmentProblem
import argparse
from src.base_nns.dqn import DQN
from new_leap_operators import PytorchDecoder, ClassificationProblem, accuracy, TimingProbe, mutate_uniform, build_probes


def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--population_size", dest="population_size", default=200, type=int)
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "small", "medium"])
    return parser

args = get_arg_parser().parse_args()

# Parameters
runs_per_fitness_eval = 5
simulation_steps = 10
pop_size = args.population_size

gui = False  # Change to true to watch the cart-pole visualization
low = -1.
high = 1.

generations = 100


##############################
# Function build_probes()
##############################
def build_tiny_probes(genomes_file):
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

    # Open a figure to plot the best-of-gen network graph to
    plt.figure()
    probes.append(neural_network.GraphPhenotypeProbe(
                        modulo=1, ax=plt.gca(),
                        weights=True, weight_multiplier=3.0))

    return probes


def run_tiny_network():
    num_hidden_nodes = 4
    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v1')

    # Representation
    num_inputs = 4
    num_actions = environment.action_space.n
    print(num_actions)
    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = executable.WrapperDecoder(
                wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(
                    shape=(num_inputs, num_hidden_nodes, num_actions)
                ),
                decorator=executable.ArgmaxExecutable)
    timing_probe = TimingProbe()

    with open('./genomes.csv', 'w') as genomes_file:

        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                            # Solve a problem that executes agents in the
                            # environment and obtains fitness from it
                            problem=UpdatedEnvironmentProblem(
                                runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
                                stop_on_done=False),

                            representation=Representation(
                                initialize=create_real_vector(bounds=([[-1, 1]]*decoder.wrapped_decoder.length)),
                                decoder=decoder),

                            # The operator pipeline.
                            pipeline=[
                                timing_probe,
                                ops.proportional_selection,
                                ops.clone,
                                ops.uniform_crossover,
                                mutate_uniform(low=low, high=high, expected_num_mutations=0.01*decoder.wrapped_decoder.length),
                                ops.evaluate,
                                ops.pool(size=pop_size),
                                timing_probe,
                                *build_tiny_probes(genomes_file)  # Inserting all the probes at the end
                            ])
        list(ea)


def run_small_network():
    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v1')
    # Representation
    num_actions = environment.action_space.n
    print(num_actions)
    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    device = torch.device("cuda")

    decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=16)

    timing_probe = TimingProbe()
    print(f"expected_num_mutations = {decoder.length*0.01}")

    with open('./genomes.csv', 'w') as genomes_file:
        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                             # Solve a problem that executes agents in the
                             # environment and obtains fitness from it
                             problem=UpdatedTorchEnvironmentProblem(
                                 runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
                                 stop_on_done=False),

                             representation=Representation(
                                 initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
                                 decoder=decoder),

                             # The operator pipeline.
                             pipeline=[
                                 timing_probe,
                                 ops.proportional_selection,
                                 ops.clone,
                                 ops.uniform_crossover,
                                 mutate_uniform(low=low, high=high, expected_num_mutations=0.01*decoder.length),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 timing_probe,
                                 *build_probes(genomes_file)  # Inserting all the probes at the end
                             ])
        list(ea)


def run_medium_network():
    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v1')
    # Representation
    num_actions = environment.action_space.n
    print(num_actions)
    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    device = torch.device("cuda")

    decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=64)

    timing_probe = TimingProbe()
    print(f"expected_num_mutations = {decoder.length * 0.01}")

    with open('./genomes.csv', 'w') as genomes_file:
        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                             # Solve a problem that executes agents in the
                             # environment and obtains fitness from it
                             problem=UpdatedTorchEnvironmentProblem(
                                 runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
                                 stop_on_done=False),

                             representation=Representation(
                                 initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
                                 decoder=decoder),

                             # The operator pipeline.
                             pipeline=[
                                 timing_probe,
                                 ops.proportional_selection,
                                 ops.clone,
                                 ops.uniform_crossover,
                                 mutate_uniform(low=low, high=high, expected_num_mutations=0.01 * decoder.length),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 timing_probe,
                                 *build_probes(genomes_file)  # Inserting all the probes at the end
                             ])
        list(ea)


##############################
# Entry point
##############################
if __name__ == '__main__':
    if args.model_size == "tiny":
        run_tiny_network()
    elif args.model_size == "small":
        run_small_network()
    else:
        run_medium_network()
