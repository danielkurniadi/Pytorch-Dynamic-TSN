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

from core.dataset import AggressionDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "data/splits/")


def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


num_class = 2
model = pretrainedmodels.__dict__['resnext101_32x4d'](
    num_classes=1000, pretrained='imagenet'
)

# last fully-connected linear module settings
model.last_layer_name = 'last_linear'
num_feats = model.last_linear.in_features
model.last_linear = torch.nn.Linear(num_feats, num_class)

# last fully-connected linear module settings
model.last_layer_name = 'last_linear'
num_feats = model.last_linear.in_features
model.last_linear = torch.nn.Linear(num_feats, num_class)

# image settings
input_size = (3, 224, 224)
input_mean = (0.485, 0.456, 0.406)
input_std = (0.229, 0.224, 0.225)

# hyperparams settings
epochs = 20
batch_size = 32 # mini-batch-size
learning_rate = 0.01
momentum = 0.5
decay_factor = 0.35
decay_rate = 5 # in epochs
eval_freq = 3 # in epochs


# splits file
ftrain_split = os.path.join(SPLITS_DIR, "aggression_train_split_0.txt")
fval_split = os.path.join(SPLITS_DIR, "aggression_val_split_0.txt")
ftest_split = os.path.join(SPLITS_DIR, "aggression_test_split.txt")

# data generator settings: dataset and dataloader
train_dataset = AggressionDataset(ftrain_split, input_size,
    input_mean=input_mean, input_std=input_std)
val_dataset = AggressionDataset(fval_split, input_size,
    input_mean=input_mean, input_std=input_std)
test_dataset = AggressionDataset(ftest_split, input_size,
    input_mean=input_mean, input_std=input_std)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    model.cuda()

# Loss and backprop settings
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum
)

# printing to widget
print("----------------------------------------------------------------------------------------")
print("TRAINING SESSION:")
print("Model: %s" % model.__class__.__name__)
print("Dataset: %s" % AggressionDataset.__name__)

print("=======================================")
print("HYPERPARAMS: ")
print("Epochs: %d" % epochs)
print("Batch-size: %d" % batch_size)
print("Initial learning rate: %s" % learning_rate)

print("========================================")
print("SUMMARY")
print("%s" % summary(model, input_size))

print("BEGIN: %s" % time.time())


best_acc = 0.0

running_loss = 0.0
running_acc = 0

# trainload
dataiter = iter(train_loader)
X_train, labels = next(dataiter)

print("Input Train X shape: ", X_train.shape, labels.shape)

# training
if torch.cuda.is_available():
    X_train = torch.autograd.Variable(X_train.cuda())
    labels = torch.autograd.Variable(labels.cuda())
model.train()
outputs = model(X_train)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

_, preds = torch.max(outputs.data, 1)
acc = torch.mean((preds == labels.data).float())

learning_rate *= decay_factor
update_learning_rate(optimizer, learning_rate)

print("-- Training Once | Acc: {} Loss: {} ".format(acc, loss.data))


# valload
dataiter = iter(val_loader)
X_val, labels_val = next(dataiter)

print("Input Val X shape: ", X_val.shape, labels_val.shape)

# validation
if torch.cuda.is_available():
    X_val = torch.autograd.Variable(X_val.cuda())
    labels_val = torch.autograd.Variable(labels_val.cuda())

with torch.no_grad():
    model.eval()
    val_outputs = model(X_val)
    val_loss = criterion(outputs, labels_val)

    _, val_preds = torch.max(val_outputs.data, 1)
    val_acc = torch.mean((val_preds == labels_val.data).float())

    print("--Validation Once | Acc: {} Loss : {}".format(val_acc, val_loss.data))
