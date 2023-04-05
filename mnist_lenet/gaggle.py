import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))

from src.arguments import ConfigArgs, DatasetArgs, EnvArgs, IndividualArgs, OutdirArgs, GAArgs
from src.data import Dataset, DatasetFactory
from src.population.population_manager import PopulationManager
from src.utils.special_print import print_dict_highlighted
from src.ga import SimpleGA, GAFactory
import transformers


def parse_args():
    parser = transformers.HfArgumentParser((OutdirArgs, EnvArgs, IndividualArgs, GAArgs, DatasetArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train(outdir_args: OutdirArgs,
          env_args: EnvArgs,
          individual_args: IndividualArgs,
          ga_args: GAArgs,
          dataset_args: DatasetArgs,
          config_args: ConfigArgs):
    """ Train a model from scratch on a dataset. """
    if config_args.exists():
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()
        individual_args = config_args.get_individual_args()
        dataset_args = config_args.get_dataset_args()
        ga_args = config_args.get_ga_args()

    print_dict_highlighted(vars(ga_args))

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    population_manager: PopulationManager = PopulationManager(ga_args, individual_args, env_args=env_args)
    trainer: SimpleGA = GAFactory.from_ga_args(ga_args, env_args=env_args)
    trainer.train(population_manager=population_manager, ds_train=ds_train, ds_test=ds_test, outdir_args=outdir_args)
    trainer.save_metrics(outdir_args, save_train=True, save_eval=True)


if __name__ == "__main__":
    train(*parse_args())
