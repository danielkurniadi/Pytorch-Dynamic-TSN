from torch import nn
from collections import OrderedDict
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

class BaseModel(nn.Module):
    pass