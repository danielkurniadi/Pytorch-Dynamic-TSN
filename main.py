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

# Important directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs/')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints/')
DATA_DIR = os.path.join(BASE_DIR, 'data/') 
SPLITS_DIR = os.path.join(DATA_DIR, 'splits/')

def main(ftrain_split, ftest_split, split):
    """Run model (resnext101) on aggression dataset
    """
    num_class = 2
    model = pretrainedmodels.__dict__['resnext101_32x4d'](
        num_classes=1000, pretrained='imagenet'
    )

    # last fully-connected linear module settings
    model.last_layer_name = 'last_linear'
    num_feats = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(num_feats, num_class)

    # image settings
    input_size = (3, 224, 224)
    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)

    # hyperparams settings
    epochs = 2
    batch_size = 32 # mini-batch-size
    learning_rate = 0.01
    momentum = 0.5
    decay_factor = 0.35
    decay_rate = 5 # in epochs
    eval_freq = 3 # in epochs

    # data generator settings: dataset and dataloader
    train_dataset = AggressionDataset(ftrain_split, input_size,
        input_mean=input_mean, input_std=input_std)
    val_dataset = AggressionDataset(ftest_split, input_size,
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
    # experiment session
    for e in range(epochs):
        # training
        train(model, train_loader, criterion, optimizer, e+1, split)

        if (e+1)%decay_rate == 0:
            learning_rate *= decay_factor
            update_learning_rate(optimizer, learning_rate)

        # validation
        if (e+1)%eval_freq == 0:
            acc = validate(model, val_loader, criterion, e+1, split)

        if (best_acc*1.02) < acc:
            best_acc = acc
            state = {
                'model_name': model.__class__.__name__,
                'split': split,
                'epoch': e+1,
                'state_dict': model.state_dict()
            }
            save_checkpoints(state)            

        print("----------------------------------------")

    print("END: %s" % time.time())
    print('*' * 100)


    print("----------------------------------------------------------------------------------------")
    print("TESTING SESSION:")
    test(model, test_loader, criterion, e+1, split)
    print('*' * 100)


def save_checkpoints(state):
    filename = "aggresion_{}_split{}_{}.pth".format(
        state['model_name'], state['split'], state['epoch']
    )
    savepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, savepath)


def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def train(model, dataloader, criterion, optimizer, e, k):
    model.train()
    
    running_loss = 0.0
    running_acc = 0
    start = time.time()
    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = X.cuda()
            labels = labels.cuda()

        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)
        running_loss += loss.data

        # backpropagation and weights update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # eval acc
        _, preds = torch.max(outputs.data, 1)
        acc = torch.mean((preds == labels.data).float())
        running_acc += acc

        if (i+1) % 25 == 0:
            log_loss(loss, acc, e, k, i, train=True)

        # print result
        elapsed = time.time() - start
        print("Split {} - Epoch: [{:03d}] : Iterations [{:03d}/{}] | Train loss: {: .4f} | Elapsed time: {: .4f}".format(
            k, e, i+1, len(dataloader), loss.data, elapsed
        ))

    # print 1 epoch result
    running_loss = running_loss/len(dataloader)
    running_acc = running_acc/len(dataloader)
    
    log_loss(running_loss, running_acc, e, k, train=True)
    print("Split {} - Epoch: [{:03d}] : Avg Train Loss: {: .4f} | Avg Train Acc: {: .4f}".format(
        k, e, running_loss, running_acc
    ))


def validate(model, dataloader, criterion, e, k):
    model.eval()
    
    running_loss = 0.0
    running_acc = 0

    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = torch.autograd.Variable(X.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        
        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data
        acc = torch.mean((preds == labels.data).float())
        running_acc += acc

    # print 1 epoch result
    running_loss = running_loss/len(dataloader)
    running_acc = running_acc/len(dataloader)
    
    log_loss(running_loss, running_acc, e, k, train=False)
    print("Split {} - Epoch: [{:03d}] : Avg Val Loss: {: .4f} | Avg Val Acc: {: .4f}".format(
        k, e, running_loss, running_acc
    ))

    return running_acc


def test(model, dataloader, criterion, e, k):
    model.eval()
    
    running_acc = 0.0
    for i, (X, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = torch.autograd.Variable(X.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        
        # feed forward & get prediction
        outputs = model(X)
        _, preds = torch.max(outputs.data, 1)
        acc = torch.mean((preds == labels.data).float())
        running_acc += acc
    
    running_acc = running_acc/len(dataloader)
    print("Split {} - Epoch: [{:03d}] - Avg Test Acc: {: .4f}".format(
        k, e, running_acc
    ))


def log_loss(loss, acc, epoch, k, i=None, train=True):
    train = 'train' if train else 'val'
    if not i:  # meaning average loss and acc passed
        filename = os.path.join(LOGS_DIR, 'aggresion_{}_split{}_avg.log'.format(train, k))
        with open(filename, 'a') as f:
            to_write = "epoch:{} avgloss:{} avgacc:{}\n".format(epoch, loss, acc)
            f.write(to_write)
    else:  # loss and acc per iteration
        filename = os.path.join(LOGS_DIR, 'aggresion_{}_split{}.log'.format(train, k))
        with open(filename, 'a') as f:
            to_write = "epoch:{} iter:{} loss:{}\n".format(epoch, i, loss)


if __name__ == '__main__':

    # prepare splits
    for k in range(0, 5):
        ftrain_split = os.path.join(SPLITS_DIR, 'aggression_train_split_{}.txt'.format(k))
        ftest_split = os.path.join(SPLITS_DIR, 'aggression_val_split_{}.txt'.format(k))
    
        main(ftrain_split, ftest_split, k)
