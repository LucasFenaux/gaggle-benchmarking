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

from leap_ec import Individual, Representation, test_env_var, Decoder
from leap_ec import probe, ops
from leap_ec.problem import FunctionProblem
from leap_ec.global_vars import context
from leap_ec.ops import compute_expected_probability, iteriter_op
from toolz import curry
from typing import Iterator, Union
from leap_ec.real_rep.problems import ScalarProblem
from TorchGA import torchga


class UpdatedEnvironmentProblem(ScalarProblem):
    """Defines a fitness function over :class:`~leap_ec.executable.phenotype.Executable` by
    evaluating them within a given environment.

    :param int runs: The number of independent runs to aggregate data over.
    :param int steps: The number of steps to run the simulation for within each run.
    :param environment: A simulation environment corresponding to the OpenAI Gym environment interface.
    :param behavior_fitness: A function
    """

    def __init__(self, runs: int, steps: int, environment, fitness_type: str,
                 gui: bool, stop_on_done=True, maximize=True):
        assert (runs > 0)
        assert (steps > 0)
        assert (environment is not None)
        assert (fitness_type is not None)
        super().__init__(maximize)
        self.runs = runs
        self.steps = steps
        self.environment = environment
        self.environment._max_episode_steps = steps  # This may not work with all environments.
        self.stop_on_done = stop_on_done
        self.gui = gui
        if fitness_type == 'reward':
            self.fitness = UpdatedEnvironmentProblem._reward_fitness
        elif fitness_type == 'survival':
            self.fitness = UpdatedEnvironmentProblem._survival_fitness
        else:
            raise ValueError(f"Unrecognized fitness type: '{fitness_type}'")

    @property
    def num_inputs(self):
        """Return the number of dimensions in the environment's input space."""
        self.space_dimensions(self.environment.observation_space)

    @property
    def num_outputs(self):
        """Return the number of dimensions in the environment's action space."""
        self.space_dimensions(self.environment.action_space)

    @classmethod
    def _reward_fitness(cls, observations, rewards):
        """Compute fitness by summing the rewards across all runs."""
        sums = [sum(run) for run in rewards]
        return np.mean(sums)

    @classmethod
    def _survival_fitness(cls, observations, rewards):
        """Compute fitness as the average length of the runs."""
        return np.mean([len(o) for o in observations])

    @staticmethod
    def space_dimensions(observation_space) -> int:
        """Helper to get the number of dimensions (variables) in an OpenAI Gym space.

        The point of this helper is that it works on simple spaces:

        >>> from gym import spaces
        >>> discrete = spaces.Discrete(8)
        >>> EnvironmentProblem.space_dimensions(discrete)
        1

        Box spaces:

        >>> box = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        >>> EnvironmentProblem.space_dimensions(box)
        12

        And Tuple spaces:

        >>> tup = spaces.Tuple([discrete, box])
        >>> EnvironmentProblem.space_dimensions(tup)
        13
        """
        if hasattr(observation_space, 'spaces'):
            # If we're a Tuple space, count the inputs across each space in the Tuple
            return sum([int(np.prod(s.shape)) for s in observation_space.spaces])
        else:
            # Otherwise just look at the shape of the space directly
            return int(np.prod(observation_space.shape))

    def evaluate(self, executable):
        """Run the environmental simulation using `executable` as a controller,
        and use the resulting observations & rewards to compute a fitness value."""
        observations = []
        rewards = []
        for r in range(self.runs):
            observation, _ = self.environment.reset()
            run_observations = [observation]
            run_rewards = []
            for t in range(self.steps):
                if self.gui:
                    self.environment.render()
                action = executable(observation)
                observation, reward, done, info, _ = self.environment.step(action)
                run_observations.append(observation)
                run_rewards.append(reward)
                if self.stop_on_done and done:
                    break
            observations.append(run_observations)
            rewards.append(run_rewards)
        return self.fitness(observations, rewards)


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


