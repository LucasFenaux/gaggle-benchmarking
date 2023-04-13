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
import torch
import numpy as np

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
        dim = individual_args.individual_size
        outdir_args = config_args.get_outdir_args()
        sys_args = config_args.get_sys_args()
        individual_args = config_args.get_individual_args()
        problem_args = config_args.get_problem_args()
        ga_args = config_args.get_ga_args()
        individual_args.individual_size = dim
    print_dict_highlighted(vars(problem_args))

    class GaggleRastriginProblem(Problem):
        @torch.no_grad()
        def evaluate(self, individual: Individual, *args, **kwargs) -> float:
            chromo = individual.forward()
            dimension = individual.individual_args.individual_size
            rastrigin = - (10 * dimension + \
                torch.sum(chromo ** 2 - 10 * torch.cos(2 * torch.pi * chromo)))
            return rastrigin.cpu().item()
        
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
    filename = 'gaggle_dimension_{}.p'.format(individual_args.individual_size)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    train(*parse_args())

    print(f"Total program time: {time.time() - beginning_time}")
