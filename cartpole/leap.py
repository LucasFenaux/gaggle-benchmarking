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

import gym
import pickle
from matplotlib import pyplot as plt
from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep import problems, executable, neural_network
from leap_ec.real_rep.initializers import create_real_vector
from new_leap_operators import UpdatedEnvironmentProblem, UpdatedTorchEnvironmentProblem
import argparse
from new_leap_operators import PytorchDecoder, TimingProbe, mutate_uniform, build_probes
from models import DQN, LargeDQN


def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--population_size", dest="population_size", default=100, type=int)
    parser.add_argument("--model_size", default="tiny")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser

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


def get_probes(model_size):
    if model_size == "tiny":
        return build_tiny_probes
    else:
        return build_probes


def get_problem(model_size):
    if model_size == "tiny":
        return UpdatedEnvironmentProblem
    else:
        return UpdatedTorchEnvironmentProblem


# def run_tiny_network():
#     num_hidden_nodes = 4
#     # Load the OpenAI Gym simulation
#     environment = gym.make('CartPole-v1')
#
#     # Representation
#     num_inputs = 4
#     num_actions = environment.action_space.n
#     print(num_actions)
#     # Decode genomes into a feed-forward neural network,
#     # but also wrap an argmax around the networks so their
#     # output is a single integer
#     decoder = executable.WrapperDecoder(
#                 wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(
#                     shape=(num_inputs, num_hidden_nodes, num_actions)
#                 ),
#                 decorator=executable.ArgmaxExecutable)
#     timing_probe = TimingProbe()
#     print(f"Num params:{decoder.wrapped_decoder.length}")
#     with open('./genomes.csv', 'w') as genomes_file:
#
#         ea = generational_ea(max_generations=generations, pop_size=pop_size,
#                             # Solve a problem that executes agents in the
#                             # environment and obtains fitness from it
#                             problem=UpdatedEnvironmentProblem(
#                                 runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
#                                 stop_on_done=False),
#
#                             representation=Representation(
#                                 initialize=create_real_vector(bounds=([[-1, 1]]*decoder.wrapped_decoder.length)),
#                                 decoder=decoder),
#
#                             # The operator pipeline.
#                             pipeline=[
#                                 timing_probe,
#                                 ops.proportional_selection,
#                                 ops.clone,
#                                 ops.uniform_crossover,
#                                 mutate_uniform(low=low, high=high, expected_num_mutations=0.01*decoder.wrapped_decoder.length),
#                                 ops.evaluate,
#                                 ops.pool(size=pop_size),
#                                 timing_probe,
#                                 *build_tiny_probes(genomes_file)  # Inserting all the probes at the end
#                             ])
#         list(ea)
#     times = timing_probe.buffer
#     dir = 'Results/'
#     filename = 'leap_model_size_{}.p'.format(args.model_size)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     with open(os.path.join(dir, filename), 'wb') as f:
#         pickle.dump(times, f)
#
#
# def run_small_network():
#     # Load the OpenAI Gym simulation
#     environment = gym.make('CartPole-v1')
#     # Representation
#     num_actions = environment.action_space.n
#     print(num_actions)
#     # Decode genomes into a feed-forward neural network,
#     # but also wrap an argmax around the networks so their
#     # output is a single integer
#     # device = torch.device("cuda")
#
#     decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=16)
#
#     timing_probe = TimingProbe()
#     print(f"expected_num_mutations = {decoder.length*0.01}")
#
#     with open('./genomes.csv', 'w') as genomes_file:
#         ea = generational_ea(max_generations=generations, pop_size=pop_size,
#                              # Solve a problem that executes agents in the
#                              # environment and obtains fitness from it
#                              problem=UpdatedTorchEnvironmentProblem(
#                                  runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
#                                  stop_on_done=False),
#
#                              representation=Representation(
#                                  initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
#                                  decoder=decoder),
#
#                              # The operator pipeline.
#                              pipeline=[
#                                  timing_probe,
#                                  ops.proportional_selection,
#                                  ops.clone,
#                                  ops.uniform_crossover,
#                                  mutate_uniform(low=low, high=high, expected_num_mutations=0.01*decoder.length),
#                                  ops.evaluate,
#                                  ops.pool(size=pop_size),
#                                  timing_probe,
#                                  *build_probes(genomes_file)  # Inserting all the probes at the end
#                              ])
#         list(ea)
#
#     times = timing_probe.buffer
#     dir = 'Results/'
#     filename = 'leap_model_size_{}.p'.format(args.model_size)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     with open(os.path.join(dir, filename), 'wb') as f:
#         pickle.dump(times, f)
#
#
# def run_medium_network():
#     # Load the OpenAI Gym simulation
#     environment = gym.make('CartPole-v1')
#     # Representation
#     num_actions = environment.action_space.n
#     print(num_actions)
#     # Decode genomes into a feed-forward neural network,
#     # but also wrap an argmax around the networks so their
#     # output is a single integer
#     # device = torch.device("cuda")
#
#     decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=64)
#
#     timing_probe = TimingProbe()
#     print(f"expected_num_mutations = {decoder.length * 0.01}")
#
#     with open('./genomes.csv', 'w') as genomes_file:
#         ea = generational_ea(max_generations=generations, pop_size=pop_size,
#                              # Solve a problem that executes agents in the
#                              # environment and obtains fitness from it
#                              problem=UpdatedTorchEnvironmentProblem(
#                                  runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
#                                  stop_on_done=False),
#
#                              representation=Representation(
#                                  initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
#                                  decoder=decoder),
#
#                              # The operator pipeline.
#                              pipeline=[
#                                  timing_probe,
#                                  ops.proportional_selection,
#                                  ops.clone,
#                                  ops.uniform_crossover,
#                                  mutate_uniform(low=low, high=high, expected_num_mutations=0.01 * decoder.length),
#                                  ops.evaluate,
#                                  ops.pool(size=pop_size),
#                                  timing_probe,
#                                  *build_probes(genomes_file)  # Inserting all the probes at the end
#                              ])
#         list(ea)
#
#     times = timing_probe.buffer
#     dir = 'Results/'
#     filename = 'leap_model_size_{}.p'.format(args.model_size)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     with open(os.path.join(dir, filename), 'wb') as f:
#         pickle.dump(times, f)
#
#
# def run_large_network():
#     # Load the OpenAI Gym simulation
#     environment = gym.make('CartPole-v1')
#     # Representation
#     num_actions = environment.action_space.n
#     print(num_actions)
#     # Decode genomes into a feed-forward neural network,
#     # but also wrap an argmax around the networks so their
#     # output is a single integer
#     # device = torch.device("cuda")
#
#     decoder = PytorchDecoder(LargeDQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=128)
#
#     timing_probe = TimingProbe()
#     print(f"expected_num_mutations = {decoder.length * 0.01}")
#
#     with open('./genomes.csv', 'w') as genomes_file:
#         ea = generational_ea(max_generations=generations, pop_size=pop_size,
#                              # Solve a problem that executes agents in the
#                              # environment and obtains fitness from it
#                              problem=UpdatedTorchEnvironmentProblem(
#                                  runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
#                                  stop_on_done=False),
#
#                              representation=Representation(
#                                  initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
#                                  decoder=decoder),
#
#                              # The operator pipeline.
#                              pipeline=[
#                                  timing_probe,
#                                  ops.proportional_selection,
#                                  ops.clone,
#                                  ops.uniform_crossover,
#                                  mutate_uniform(low=low, high=high, expected_num_mutations=0.01 * decoder.length),
#                                  ops.evaluate,
#                                  ops.pool(size=pop_size),
#                                  timing_probe,
#                                  *build_probes(genomes_file)  # Inserting all the probes at the end
#                              ])
#         list(ea)
#
#     times = timing_probe.buffer
#     dir = 'Results/'
#     filename = 'leap_model_size_{}.p'.format(args.model_size)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     with open(os.path.join(dir, filename), 'wb') as f:
#         pickle.dump(times, f)
#
#
# def run_very_large_network():
#     # Load the OpenAI Gym simulation
#     environment = gym.make('CartPole-v1')
#     # Representation
#     num_actions = environment.action_space.n
#     print(num_actions)
#     # Decode genomes into a feed-forward neural network,
#     # but also wrap an argmax around the networks so their
#     # output is a single integer
#     # device = torch.device("cuda")
#
#     decoder = PytorchDecoder(LargeDQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=256)
#
#     timing_probe = TimingProbe()
#     print(f"expected_num_mutations = {decoder.length * 0.01}")
#
#     with open('./genomes.csv', 'w') as genomes_file:
#         ea = generational_ea(max_generations=generations, pop_size=pop_size,
#                              # Solve a problem that executes agents in the
#                              # environment and obtains fitness from it
#                              problem=UpdatedTorchEnvironmentProblem(
#                                  runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
#                                  stop_on_done=False),
#
#                              representation=Representation(
#                                  initialize=create_real_vector(bounds=([[-1, 1]] * decoder.length)),
#                                  decoder=decoder),
#
#                              # The operator pipeline.
#                              pipeline=[
#                                  timing_probe,
#                                  ops.proportional_selection,
#                                  ops.clone,
#                                  ops.uniform_crossover,
#                                  mutate_uniform(low=low, high=high, expected_num_mutations=0.01 * decoder.length),
#                                  ops.evaluate,
#                                  ops.pool(size=pop_size),
#                                  timing_probe,
#                                  *build_probes(genomes_file)  # Inserting all the probes at the end
#                              ])
#         list(ea)
#
#     times = timing_probe.buffer
#     dir = 'Results/'
#     filename = 'leap_model_size_{}.p'.format(args.model_size)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     with open(os.path.join(dir, filename), 'wb') as f:
#         pickle.dump(times, f)


##############################
# Entry point
##############################
if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    device = args.device
    # Parameters
    runs_per_fitness_eval = 5
    simulation_steps = 10
    pop_size = args.population_size

    gui = False  # Change to true to watch the cart-pole visualization
    low = -1.
    high = 1.

    generations = 100

    num_hidden_nodes = 4
    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v1')

    # Representation
    num_inputs = 4
    num_actions = environment.action_space.n
    print(num_actions)

    timing_probe = TimingProbe()

    if args.model_size == "tiny":
        decoder = executable.WrapperDecoder(
            wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(
                shape=(num_inputs, num_hidden_nodes, num_actions)
            ),
            decorator=executable.ArgmaxExecutable)
        decoder_length = decoder.wrapped_decoder.length
    elif args.model_size == "small":
        decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=16)
        decoder_length = decoder.length
    elif args.model_size == "medium":
        decoder = PytorchDecoder(DQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=64)
        decoder_length = decoder.length
    elif args.model_size == "large":
        decoder = PytorchDecoder(LargeDQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=128)
        decoder_length = decoder.length
    else:
        decoder = PytorchDecoder(LargeDQN, device, num_inputs=4, num_outputs=num_actions, hidden_size=256)
        decoder_length = decoder.length

    probe_builder = get_probes(args.model_size)

    problem = get_problem(args.model_size)

    print(f"Num params:{decoder_length}")
    with open('./genomes.csv', 'w') as genomes_file:

        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                             # Solve a problem that executes agents in the
                             # environment and obtains fitness from it
                             problem=problem(
                                 runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui,
                                 stop_on_done=False),

                             representation=Representation(
                                 initialize=create_real_vector(bounds=([[-1, 1]] * decoder_length)),
                                 decoder=decoder),

                             # The operator pipeline.
                             pipeline=[
                                 timing_probe,
                                 ops.proportional_selection,
                                 ops.clone,
                                 ops.uniform_crossover,
                                 mutate_uniform(low=low, high=high, expected_num_mutations=0.01 * decoder_length),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 timing_probe,
                                 *probe_builder(genomes_file)  # Inserting all the probes at the end
                             ])
        list(ea)
    times = timing_probe.buffer
    dir = 'Results/'
    filename = 'leap_model_size_{}.p'.format(args.model_size)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(times, f)

    # if args.model_size == "tiny":
    #     run_tiny_network()
    # elif args.model_size == "small":
    #     run_small_network()
    # elif args.model_size == "medium":
    #     run_medium_network()
    # elif args.model_size == "large":
    #     run_large_network()
    # else:
    #     run_very_large_network()

    print(f"Total program time: {time.time() - beginning_time}")
