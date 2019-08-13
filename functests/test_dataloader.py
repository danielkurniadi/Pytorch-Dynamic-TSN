import os
import time
import shutil
import argparse

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchsummary import summary

import pretrainedmodels

from core.dataset import create_dataset_loader
from core.options.train_options import TrainOptions


opts = TrainOptions().parse()   # get training options
test_loader = create_dataset_loader(opts, phase='train')    # create train dataset given opt.dataset_mode and other options

test_dataset_size = len(test_loader)    # get the number of images in the dataset.

print('The number of testing images = %d' % test_dataset_size)

# training options
batch_size = opts.batch_size


# printing to widget
print("----------------------------------------------------------------------------------------")
print("TEST DATA LOADER:")
print("Dataset: %s" % test_loader.dataset.__class__.__name__)
print("Modality: %s" % opts.modality)


# Iterate data loader
dataiter = iter(test_loader)
data, labels = next(dataiter)

print("=============================================")
print("Batch Dataset Shape: ", len(data), data[0].size())


