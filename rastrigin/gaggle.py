import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP'))
sys.path.insert(1, os.path.join(sys.path[0], '../LEAP/leap_ec'))

from src.arguments import ConfigArgs
from src.arguments.problem_args import ProblemArgs
from src.arguments.sys_args import SysArgs
from src.arguments.individual_args import IndividualArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.ga_args import GAArgs
from src.population.population_manager import PopulationManager
from src.utils.special_print import print_dict_highlighted
from src.problem.problem_factory import ProblemFactory
from src.problem.problem import Problem
from src.population.individual import Individual
from src.ga import GA
from src.ga.ga_factory import GAFactory
import transformers
import pickle
import numpy as np
from leap_ec.real_rep.problems import ScalarProblem
from dataclasses import dataclass, field


# class RastriginProblem(ScalarProblem):
#     """ Modified to include negative fittness (LEAP had a bug)
#     """
#     """ Standard bounds."""
#     bounds = (-5.12, 5.12)
#     #NOTE we changed maximize to true
#     def __init__(self, a=1.0, maximize=True):
#         super().__init__(maximize)
#         self.a = a

#     def evaluate(self, phenome):
#         """
#         Computes the function value from a real-valued list phenome:

#         >>> phenome = [1.0/12, 0]
#         >>> RastriginProblem().evaluate(phenome) # doctest: +ELLIPSIS
#         0.1409190406...

#         :param phenome: real-valued vector to be evaluated
#         :returns: its fitness
#         """
#         #NOTE: We made negative as it is wrong
#         if isinstance(phenome, np.ndarray):
#             return - (self.a * len(phenome) + \
#                 np.sum(phenome ** 2 - self.a * np.cos(2 * np.pi * phenome)))
#         return self.a * \
#             len(phenome) + sum([x ** 2 - self.a *
#                                 np.cos(2 * np.pi * x) for x in phenome])

#     def worse_than(self, first_fitness, second_fitness):
#         return super().worse_than(first_fitness, second_fitness)

#     def __str__(self):
#         return RastriginProblem.__name__


def parse_args():
    parser = transformers.HfArgumentParser((OutdirArgs, SysArgs, IndividualArgs, GAArgs, ProblemArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

     
def train(outdir_args: OutdirArgs,
          sys_args: SysArgs,
          individual_args: IndividualArgs,
          ga_args: GAArgs,
          problem_args: ProblemArgs,
          config_args: ConfigArgs):
    """ Train a model from scratch on a data. """
    if config_args.exists():
        dim = individual_args.np_individual_size
        outdir_args = config_args.get_outdir_args()
        sys_args = config_args.get_sys_args()
        individual_args = config_args.get_individual_args()
        problem_args = config_args.get_problem_args()
        ga_args = config_args.get_ga_args()
        individual_args.np_individual_size = dim
    print_dict_highlighted(vars(problem_args))

    class GaggleRastriginProblem(Problem):
        def evaluate(self, individual: Individual, *args, **kwargs) -> float:
            chromo = individual.forward()
            dimension = individual.individual_args.np_individual_size
            rastrigin = - (dimension * len(chromo) + \
                np.sum(chromo ** 2 - dimension * np.cos(2 * np.pi * chromo)))
            return rastrigin
        
    ProblemFactory.register_problem(problem_type='custom', problem_name='Rastrigin', problem=GaggleRastriginProblem)
    # ProblemFactory.convert_and_register_leap_problem(problem_name='Rastrigin', leap_problem=RastriginProblem,
    #                                                  a=individual_args.np_individual_size)

    
    population_manager: PopulationManager = PopulationManager(ga_args, individual_args, sys_args=sys_args)
    trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, ga_args=ga_args,
                                         problem_args=problem_args, sys_args=sys_args, outdir_args=outdir_args,
                                         individual_args=individual_args)
    trainer.train()
    times = trainer.saved_metrics['train_metrics']['time_taken']
    dir = 'Results/'
    filename = 'gaggle_dimension_{}.p'.format(individual_args.np_individual_size)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    train(*parse_args())

    print(f"Total program time: {time.time() - beginning_time}")
