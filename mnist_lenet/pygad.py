import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../gaggle'))

from pyGAD import pygad
from pyGAD import torch
import torch
from src.data.base_dataset.mnist import MNIST
from src.base_nns.lenet import LeNet5
from src.arguments import DatasetArgs

dataset_args = DatasetArgs()


train_dataset = MNIST(train=True)
test_dataset = MNIST(train=False)

model = LeNet5()

