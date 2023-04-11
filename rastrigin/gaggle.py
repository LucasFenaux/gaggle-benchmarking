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
from src.ga import GA
from src.ga.ga_factory import GAFactory
import transformers
import pickle
from leap_ec.real_rep.problems import RastriginProblem
from dataclasses import dataclass, field
    
class ProblemArgsMod(ProblemArgs):
     dimension: int = field(default=10, metadata={
        "help": "dimension of rastrigin problem",
    })




def parse_args():
    parser = transformers.HfArgumentParser((OutdirArgs, SysArgs, IndividualArgs, GAArgs, ProblemArgsMod,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

     
def train(outdir_args: OutdirArgs,
          sys_args: SysArgs,
          individual_args: IndividualArgs,
          ga_args: GAArgs,
          problem_args: ProblemArgsMod,
          config_args: ConfigArgs):
    """ Train a model from scratch on a data. """
    if config_args.exists():
        dim = problem_args.dimension
        outdir_args = config_args.get_outdir_args()
        sys_args = config_args.get_sys_args()
        individual_args = config_args.get_individual_args()
        problem_args = config_args.get_problem_args()
        ga_args = config_args.get_ga_args()
        problem_args.dimension = dim
    print_dict_highlighted(vars(ga_args))

    ProblemFactory.convert_and_register_leap_problem(problem_name='Rastrigin', leap_problem=RastriginProblem, a=problem_args.dimension)

    
    population_manager: PopulationManager = PopulationManager(ga_args, individual_args, sys_args=sys_args)
    trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, ga_args=ga_args,
                                         problem_args=problem_args, sys_args=sys_args, outdir_args=outdir_args,
                                         individual_args=individual_args)
    trainer.train()
    times = trainer.saved_metrics['train_metrics']['time taken']
    dir = 'Results/'
    filename = 'gaggle_dimension_{}.p'.format(problem_args.dimension)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    train(*parse_args())

    print(f"Total program time: {time.time() - beginning_time}")
