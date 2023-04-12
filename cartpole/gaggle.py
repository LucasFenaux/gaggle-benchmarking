import time
beginning_time = time.time()
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))

from src.arguments import ConfigArgs
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

@dataclass
class IndividualArgsMod(IndividualArgs):
    model_size: str = field(default="tiny")

    model_name: str = field(default="lenet", metadata={
        "help": "name of the model architecture. Note that not all (resolution, model architecture)"
                "combinations are implemented. Please see the 'get_base_model' method of this class.",
        "choices": ["resnet20", "resnet32", "resnet44", "lenet", "snet", "custom", "dqn", "large_dqn"]
    })

ConfigArgs.update(IndividualArgs.CONFIG_KEY, IndividualArgsMod)

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=16):
        super(DQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively.
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers.

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        # The variable x denotes the input to the network.
        # The function returns the q value for the given input.

        x = x.view(-1, self.num_inputs)
        x = F.sigmoid(self.fc1(x))
        qvalue = F.sigmoid(self.fc2(x))  # wouldn't usually do a second sigmoid but leap does it so we have to
        return qvalue


IndividualArgsMod.update("dqn", DQN)


class LargeDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=16):
        super(LargeDQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively.
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers.

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        # The variable x denotes the input to the network.
        # The function returns the q value for the given input.

        x = x.view(-1, self.num_inputs)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


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

    print_dict_highlighted(vars(problem_args))
    print_dict_highlighted(vars(sys_args))

    num_inputs = 4
    num_outputs = 2
    if individual_args.model_size == "tiny":
        hidden_size = 4
    elif individual_args.model_size == "small":
        hidden_size = 16
    else:
        hidden_size = 64

    if individual_args.model_name == "dqn":
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
    filename = '{}_cartpole_gaggle_pop_size_{}.p'.format(individual_args.model_size, ga_args.population_size)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir,filename), 'wb') as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    train(*parse_args())

    print(f"Total program time: {time.time() - beginning_time}")
