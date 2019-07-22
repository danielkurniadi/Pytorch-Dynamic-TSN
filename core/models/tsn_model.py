import os
import time
import shutil
import argparse

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import pretrainedmodels

from .base_model import BaseModel


class TemporalSegmentNetwork(BaseModel):
    """ Temporal Segmentation Network
    """
    def __init__(self, num_classes, n_segments,
                modality, base_model_name='bninception',
                new_length='auto', consensus_type='avg',
                partial_bn=True):
        super(TemporalSegmentNetwork, self).__init__()

        self.num_classes = num_classes
        self.n_segments = n_segments
        self.modality = modality
        self.new_length = new_length
        self.consensus_type = consensus_type
        self.partial_bn = partial_bn

        if self.new_length == 'auto':
            self.new_length = 1 if self.new_length == 'RGB' else 5

        # prepare and configure model
        self.base_model = self.prepare_model()
        self.consensus_module = ConsensusModule(consensus_type)
        
    def prepare_model(self):
        