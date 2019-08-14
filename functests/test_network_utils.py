import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torchsummary

import pretrainedmodels
from pretrainedmodels import resnext101_32x4d

from core.networks import (
    init_net,
    init_weights,
    find_first_layer,
    reshape_input_nc
)


model = resnext101_32x4d(num_classes=1000, pretrained='imagenet')
model.cuda()
model = init_net(model, init_type='xavier', gpu_ids=[0,1,2])

torchsummary.summary(model, (3, 224, 224))


first_conv_layer, container = find_first_layer(model, nn.Conv2d)

print("First Conv2d Layer: ", first_conv_layer)
print("First Conv contained inside: ", type(container))

print("Reshaping Input NC -----------------------------------------")
model = reshape_input_nc(model, 15)
print("After reshaping model")
torchsummary.summary(model, (15, 224, 224))
first_conv_layer, container = find_first_layer(model, nn.Conv2d)

print("First Conv2d Layer after reshape: ", first_conv_layer)
print("Shape of first conv2d weight after reshape: ", first_conv_layer.weight)
