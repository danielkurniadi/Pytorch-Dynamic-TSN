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

from core.dataset import create_dataset
from core.models import create_model
from core.options.train_options import TrainOptions


opt = TrainOptions().parse()   # get training options
test_dataset = create_dataset(opt, opt.test_split_file)    # create train dataset given opt.dataset_mode and other options

test_dataset_size = len(test_dataset)    # get the number of images in the dataset.

print('The number of testing images = %d' % test_dataset)

# training options
batch_size = opt.batch_size
lr = opt.lr
momentum = opt.momentum

input_size = opt.input_size
input_channel = opt.input_nc

# model
model = create_model(opt)   # create a model given opt.model and other options
model.prepare_model()       # regular setup: load and print networks; create schedulers

if torch.cuda.is_available():
    model.cuda()


# printing to widget
print("----------------------------------------------------------------------------------------")
print("TRAINING SESSION:")
print("Model: %s" % model.__class__.__name__)
print("Dataset: %s" % test_dataset.__class__.__name__)

print("=======================================")
print("HYPERPARAMS: ")
print("Batch-size: %d" % batch_size)
print("Initial learning rate: %s" % lr)

print("========================================")
print("SUMMARY")
print("%s" % summary(model, (input_channel, input_size, input_size)))

print("BEGIN: %s" % time.time())

epoch_start_time = time.time()  # timer for entire epoch
iter_data_time = time.time()    # timer for data loading per iteration

running_acc = 0.0
running_loss = 0.0

model.eval()

for i, (data, labels) in enumerate(test_dataset):
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()

    with torch.no_grad():
        outputs = model(data)
        loss = model.criterion(outputs, labels)

        # evaluate model
        _, preds = torch.max(outputs.data, 1)
        acc = torch.mean((preds == labels.data).float())
        
        running_acc += acc
        running_loss += loss.data

# average acc and loss over 1 epoch
running_acc = running_acc / len(test_dataset) * batch_size
running_loss = running_loss / len(test_dataset) * batch_size

avg_log_tmpl = (
    ".. Test Avg Acc: {:2.4f} Test Avg Loss: {:2.4f} "
)

print(avg_log_tmpl.format(
    running_acc, running_loss,
))


print("----------------------------------------------------------------------------------------")
print("TESTING SESSION:")
print('*' * 100)
