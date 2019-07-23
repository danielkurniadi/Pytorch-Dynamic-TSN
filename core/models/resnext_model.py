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

from .base_model import BaseModel


class ResnextModel(BaseModel):
    """Resnext wrapper for testing framework
    """

    def __init__(self, opts):
        super(ResnextModel, self).__init__(opts)
        self.n_classes = opts.n_classes
        self.opts = opts
        
        self.pretrained = False

        self.prepare_model()

        # pretrained input image settings
        self.input_size = opts.input_size
        self.input_channels = opts.input_channels
        self.input_means = opts.input_means
        self.input_std = opts.input_std
        self.input_range = opts.input_range

        # hyperparams settings
        self.lr = opts.lr
        self.lr_policy = opts.lr_policy
        self.lr_decay_factor = opts.lr_decay_factor
        self.lr_decay_iters = opts.lr_decay_iters
        self.momentum = opts.momentum

        # criterion and optimizer settings
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr = self.lr,
            momentum = self.momentum
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            self.lr_decay_iters
        )

    #########################
	# MODEL
	#########################

    def prepare_model(self):
        """ Prepare necessary step.
		Configure model, layers, and hyperparams settings.
        """
        if self.pretrained:
            if cardinality=='32x4d':
                self.model = pretrainedmodels.__dict__['resnext101_32x4d'](
                    num_classes=1000, pretrained='imagenet'
                )
            elif cardinality=='64x4d':
                self.model = pretrainedmodels.__dict__['resnext101_64x4d'](
                    num_classes=1000, pretrained='imagenet'
                )
            else:
                raise ValueError("Invalid Cardinality %s" % cardinality)
            self.model.last_layer_name = 'last_linear'
        else:
            self.model = torchvision.models.resnet18(pretrained=False)
            self.model.last_layer_name = 'fc'
        # last fully-connected linear
        num_feats = getattr(self.model, self.model.last_layer_name).in_features
        setattr(self.model, self.model.last_layer_name, torch.nn.Linear(num_feats, self.opts.n_classes))

    @staticmethod
    def modify_cli_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        # pretrained input image settings
        parser.set_defaults(
            input_channels = 3,
            input_size = 224,
            input_range = [0.0, 1.0],
            input_means = [0.485, 0.456, 0.406],
            input_std = [0.229, 0.224, 0.225]
        )

        return parser

    #########################
	# CORE
	#########################

    def forward(self, X):
        """Run forward pass
        """
        return self.model(X)

    def optimize_parameters(self):
        """Calling forward pass, loss, optim steps, and backward prop.
        """
        outputs = self.forward(X)
        
        # calculate loss
        loss = self.criterion(outputs, labels)
        loss.backward()

        self.optimizer.step()

        return outputs, loss
