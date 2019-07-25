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
train_dataset = create_dataset(opt, opt.train_split_file)   # create train dataset given opt.dataset_mode and other options
val_dataset = create_dataset(opt, opt.val_split_file)      # create val dataset given opt.dataset_mode and other options

train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
val_dataset_size = len(val_dataset)    # get the number of images in the dataset.

print('The number of training images = %d' % train_dataset_size)
print('The number of validation images = %d' % val_dataset_size)

# training options
epochs = opt.epochs
batch_size = opt.batch_size
lr = opt.lr
momentum = opt.momentum

input_size = opt.input_size
input_channel = opt.input_channels

# model
model = create_model(opt)      # create a model given opt.model and other options
model.prepare_model()       # regular setup: load and print networks; create schedulers

if torch.cuda.is_available():
    model.cuda()

# printing to widget
print("----------------------------------------------------------------------------------------")
print("TRAINING SESSION:")
print("Model: %s" % model.__class__.__name__)
print("Dataset: %s" % train_dataset.__class__.__name__)

print("=======================================")
print("HYPERPARAMS: ")
print("Epochs: %d" % epochs)
print("Batch-size: %d" % batch_size)
print("Initial learning rate: %s" % lr)

print("========================================")
print("SUMMARY")
print("%s" % summary(model, (input_channel, input_size, input_size)))

print("BEGIN: %s" % time.time())

for epoch in range(epochs):
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration

    running_acc = 0.0
    running_loss = 0.0

    model.train()
    for i, (data, labels) in enumerate(train_dataset):  # inner loop within one epoch
        iter_start_time = time.time()   # timer for computation per iteration

        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()

        outputs = model(data)   # calculate loss functions, get gradients, update network weights
        loss = model.criterion(outputs, labels)

        model.optimizer.zero_grad()     # clear previous gradient descent
        loss.backward()                 # backpropagation
        model.optimizer.step()          # optimize parameters

        # evaluate model
        _, preds = torch.max(outputs.data, 1)
        acc = torch.mean((preds == labels.data).float())
        
        running_acc += acc
        running_loss += loss.data

        iter_data_time = time.time()

        if i % opt.print_freq_iters == 0:
            log_tmpl = (
                ".. Epoch [{:03d}/{:03d}] Iter [{:04d}/{:04d}] "
                "Training Acc: {:.4f} Training Loss: {:.4f}\n"
                "\tElapsed: {}"
            )
            print(log_tmpl.format(
                epoch, epochs,
                i, len(train_dataset)//batch_size,
                acc, loss.data,
                (iter_data_time - iter_start_time)
            ))

    if epoch % opt.eval_freq_epoch == 0:
        model.eval()

        running_val_acc = 0.0
        running_val_loss = 0.0

        for i, (data, labels) in enumerate(val_dataset):
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                outputs = model(data)
                val_loss = model.criterion(outputs, labels)

                # evaluate model
                _, preds = torch.max(outputs.data, 1)
                val_acc = torch.mean((preds == labels.data).float())
                
                running_val_acc += val_acc
                running_val_loss += val_loss.data

        # average acc and loss over 1 epoch
        running_acc = running_acc / len(train_dataset) * batch_size
        running_loss = running_loss / len(train_dataset) * batch_size
        running_val_acc = running_val_acc / len(val_dataset) * batch_size
        running_val_loss = running_val_loss / len(val_dataset) * batch_size

        avg_log_tmpl = (
            ".. Epoch [{:03d}/{:03d}] "
            "Training Avg Acc: {:2.4f} Training Avg Loss: {:2.4f} "
            "Validation Avg Acc: {:2.4f} Validation Avg Loss: {:2.4f}"
        )

        print(avg_log_tmpl.format(
            epoch, epochs,
            running_acc, running_loss,
            running_val_acc, running_val_loss
        ))


print("----------------------------------------------------------------------------------------")
print("TESTING SESSION:")
print('*' * 100)

test_dataset = create_dataset(opt, opt.test_split_file)    # create train dataset given opt.dataset_mode and other options
test_dataset_size = len(test_dataset)    # get the number of images in the dataset.

print('The number of testing images = %d' % len(test_dataset))

running_acc = 0.0
running_loss = 0.0

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