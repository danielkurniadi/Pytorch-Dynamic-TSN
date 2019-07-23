import os
import time
import shutil

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import pretrainedmodels

from core.models import create_model
from core.dataset import create_dataset
from core.options.train_options import TrainOptions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "data/splits/")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints/")


def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


opts = TrainOptions().parse()
# dataset = create_dataset(opts)
# dataset_size = len(dataset)
# print("The number of training images: %d" % dataset_size)

model = create_model(opts)
model.prepare_model()
total_iters = 0


