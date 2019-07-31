import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


#----------------------------
#  Net Configuration Helpers
#----------------------------

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


#----------------------------
#  Init Helpers
#----------------------------

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


#----------------------------
#  Modifier Helpers
#----------------------------

def find_first_layer(net, layer_type=nn.Conv2d, get_container=True):
    """ Find and return the first layer of specified type in the network.

    Parameters:
        net (network)           -- the network
        layer_type (class)      -- the layer class/type to find
        get_container (bool)    -- whether to return the container (nn.Module) which encapsulate that layer

    Return (first_layer, container) if get_container is True.
    If get_container is False or container is not found, return (first_layer, None).
    If specified layer is not found, return (None, None)
    """
    first_layer, container = None, None
    layer_index = -1
    net_modules = list(net.modules())
    for i, module in enumerate(net_modules):
        if isinstance(module, layer_type):
            first_layer = module
            layer_index = i
            break
    else:
        return None, None

    if get_container and layer_index > 0:
        container = net_modules[layer_index-1]
    
    return first_layer, container


def reshape_input_nc(net, input_nc, use_mean=True):
    """ Reshape input channel of CNN's input layer
    Parameters:
        net (network)       -- the network which input layer is to be modified
        input_nc (int)      -- new input channel length
        use_mean (bool)    -- whether to fill the new kernel with mean (over input_nc dim) of previous weight value

    Return modified network
    """
    first_conv2d_layer, container = find_first_layer(net, nn.Conv2d, get_container=True)
    
    if not (first_conv2d_layer):
        raise ValueError("Network doesn't has Conv2d (first_conv2d_layer: None)")

    kernel_size = first_conv2d_layer.kernel_size
    stride = first_conv2d_layer.stride
    padding = first_conv2d_layer.padding
    output_nc = first_conv2d_layer.output_channels
    
    reshaped_conv2d = nn.Conv2d(input_nc, output_nc, kernel_size,
                                stride=stride, padding=padding)
    if keep_data:
        weight, bias = first_conv2d_layer.parameters()
        weight, bias = weight.copy(), bias.copy()
        
        new_weight_data = weight.data.mean(dim=1).unsqueeze(dim=1) # mean over IN_C dimension, but keep the IN_C dim
        new_weight_data = new_weight_data.expand(-1, input_nc, -1, -1) # expand over IN_C dimension -> weight (OUT_C, IN_C, size, size)

        reshaped_conv2d.weight.data = new_weight_data
        reshaped_conv2d.bias = bias

    layer_name = next(iter(container.state_dict().keys())) # hackaround to get first layer name in network

    setattr(container, layer_name, reshaped_conv2d)
    return net


