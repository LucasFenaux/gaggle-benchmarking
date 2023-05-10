import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))

from gaggle.gaggle.arguments import ConfigArgs
from src.arguments.problem_args import ProblemArgs
from src.arguments.sys_args import SysArgs
from src.arguments.individual_args import IndividualArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.ga_args import GAArgs
from src.population.population_manager import PopulationManager
from src.utils.special_print import print_dict_highlighted
from src.ga import GA
from src.ga.ga_factory import GAFactory
import transformers
import pickle
from dataclasses import dataclass, field
from models import DQN, LargeDQN


@dataclass
class IndividualArgsMod(IndividualArgs):
    model_size: str = field(default="tiny")

    model_name: str = field(default="lenet", metadata={
        "help": "name of the model architecture. Note that not all (resolution, model architecture)"
                "combinations are implemented. Please see the 'get_base_model' method of this class.",
        "choices": ["resnet20", "resnet32", "resnet44", "lenet", "snet", "custom", "dqn", "large_dqn"]
    })


ConfigArgs.update(IndividualArgs.CONFIG_KEY, IndividualArgsMod)


IndividualArgsMod.update("dqn", DQN)


IndividualArgsMod.update("large_dqn", LargeDQN)


def parse_args():
    parser = transformers.HfArgumentParser((OutdirArgs, SysArgs, IndividualArgsMod, GAArgs, ProblemArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train(outdir_args: OutdirArgs,
          sys_args: SysArgs,
          individual_args: IndividualArgsMod,
          ga_args: GAArgs,
          problem_args: ProblemArgs,
          config_args: ConfigArgs):
    """ Train a model from scratch on a data. """
    if config_args.exists():
        pop_size = ga_args.population_size
        model_size = individual_args.model_size
        model_name = individual_args.model_name
        device = sys_args.device
        outdir_args = config_args.get_outdir_args()
        sys_args = config_args.get_sys_args()
        individual_args = config_args.get_individual_args()
        problem_args = config_args.get_problem_args()
        ga_args = config_args.get_ga_args()
        ga_args.population_size = pop_size
        ga_args.num_parents = pop_size
        individual_args.model_size = model_size
        sys_args.device = device
        individual_args.model_name = model_name

    print_dict_highlighted(vars(problem_args))
    print_dict_highlighted(vars(sys_args))

    num_inputs = 4
    num_outputs = 2
    if individual_args.model_size == "tiny":
        hidden_size = 4
    elif individual_args.model_size == "small":
        hidden_size = 16
    elif individual_args.model_size == "medium":
        hidden_size = 64
    elif individual_args.model_size == "large":
        hidden_size = 128
    else:
        hidden_size = 256

    if individual_args.model_name == "dqn" or individual_args.model_name == "large_dqn":
        kwargs = {"num_inputs": num_inputs,
                  "num_outputs": num_outputs,
                  "hidden_size": hidden_size}
    else:
        kwargs = {}

    population_manager: PopulationManager = PopulationManager(ga_args, individual_args, sys_args=sys_args, **kwargs)
    trainer: GA = GAFactory.from_ga_args(population_manager=population_manager, ga_args=ga_args,
                                         problem_args=problem_args, sys_args=sys_args, outdir_args=outdir_args,
                                         individual_args=individual_args)
    trainer.train()
    times = trainer.saved_metrics['train_metrics']['time_taken']
    dir = 'Results/'
    filename = 'gaggle_model_size_{}.p'.format(individual_args.model_size)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    train(*parse_args())

    print(f"Total program time: {time.time() - beginning_time}")
