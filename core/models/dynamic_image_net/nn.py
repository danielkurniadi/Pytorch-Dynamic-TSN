import os

import torch
from torch.autograd import Variable

import pretrainedmodels

from .base import BaseModel


class DynamicImageNet(BaseModel):
    def __init__(self, n_class, hyperparams, 
        base_model="resnext101_32x4d", 
        checkpoint_path=None):
        super(DynamicImageNet, self).__init__(n_class)

        # Hyperparams related to model
        self.dropouts = hyperparams.dropouts
        self.learning_rate = hyperparams.learning_rate
        self.momentum = hyperparams.momentum

        # Special attributes
        self.input_space = None
        self.input_size = None
        self.input_range = None
        self.input_mean = None 
        self.input_std = None

        # Configure model
        self.base_model = base_model
        self.pretrained = pretrained

        # Construct model
        self.prepare_model()
        self.optimizer = torch.optim.SGD(
            self.base_model.parameters(),
            lr = self.learning_rate,
            momentum = self.momentum
            )

    def prepare_model(self):
        if "resnext101" in self.base_model.lower():
            self.base_model = pretrainedmodels.__dict__[self.base_model](
                num_classes = self.n_class,
                pretrained = 'imagenet'
            )
            self.input_space = 'RGB'
            self.input_size = (3, 224, 224)
            self.input_range = (0, 1)
            self.input_mean = (0.485, 0.456, 0.406)
            self.input_std = (0.229, 0.224, 0.225)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['state_dict']
            self.base_model.load_state_dict(state_dict)

    def forward(self, X):
        output = self.base_model(X)
        return output


    def optimize_parameters(self):
        pass
