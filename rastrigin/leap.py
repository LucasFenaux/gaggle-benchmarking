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

import pickle
import argparse
import numpy as np

def get_arg_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--dimension", dest="dimension", default=10, type=int)
    return parser




from leap_ec import Individual, Representation, test_env_var, Decoder
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea

from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.problems import ScalarProblem
from leap_ec.decoder import IdentityDecoder
from new_leap_operators import TimingProbe, mutate_uniform, build_probes
class RastriginProblem(ScalarProblem):
    """ Modified to include negative fittness (LEAP had a bug)
    """
    """ Standard bounds."""
    bounds = (-5.12, 5.12)
    #NOTE we changed maximize to true
    def __init__(self, a=10.0, maximize=True):
        super().__init__(maximize)
        self.a = a
    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [1.0/12, 0]
        >>> RastriginProblem().evaluate(phenome) # doctest: +ELLIPSIS
        0.1409190406...

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness
        """
        #NOTE: We made negative as it is wrong
        if isinstance(phenome, np.ndarray):
            return - (self.a * len(phenome) + \
                np.sum(phenome ** 2 - self.a * np.cos(2 * np.pi * phenome)))
        return self.a * \
            len(phenome) + sum([x ** 2 - self.a *
                                np.cos(2 * np.pi * x) for x in phenome])

    def worse_than(self, first_fitness, second_fitness):
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        return RastriginProblem.__name__
    
    
##############################
# Entry point
##############################
if __name__ == '__main__':

    args = get_arg_parser().parse_args()
    # Parameters
    runs_per_fitness_eval = 1
    pop_size = 100
    low = -5.12
    high = 5.12
    generations = 100
    # Load the OpenAI Gym simulation

    # Representation

    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = IdentityDecoder()
    problem = RastriginProblem(10., maximize=True)
    timing_probe = TimingProbe()
    with open('./genomes.csv', 'w') as genomes_file:

        ea = generational_ea(max_generations=generations, pop_size=pop_size,
                            # Solve a problem that executes agents in the
                            # environment and obtains fitness from it
                            problem=problem,

                            representation=Representation(
                                initialize=create_real_vector(bounds=([(-5.12, 5.12)]*args.dimension)),
                                decoder=decoder),

                            # The operator pipeline.
                            pipeline=[
                                timing_probe,
                                ops.proportional_selection(offset="pop-min"),
                                ops.clone,
                                ops.uniform_crossover(p_xover=0.5),
                                mutate_uniform(low=low, high=high, expected_num_mutations=args.dimension*0.01),
                                ops.evaluate,
                                ops.pool(size=pop_size),
                                timing_probe,  # we're nice we don't include all of their extra logging in the computation
                                *build_probes(genomes_file)  # Inserting all the probes at the end
                            ])
        list(ea)

    times = timing_probe.buffer
    dir = 'Results/'
    filename = 'leap_dimension_{}.p'.format(args.dimension)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)

    print(f"Total program time: {time.time() - beginning_time}")
