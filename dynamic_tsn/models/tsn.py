from torch import nn
from collections import OrderedDict
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import pretrainedmodels


class TSN(nn.Module):
    def __init__(self, n_class, n_segments, base_model,
                modality='RBG', consensus='avg', partial_bn=True,
                pretrained='imagenet', dropout=0.8):
        super(TSN, self).__init__()
        self.n_class = n_class
        self.n_segments = n_segments
        self.modality = modality
        self.consensus = consensus
        self.partial_bn = partial_bn
        self.dropout = dropout
