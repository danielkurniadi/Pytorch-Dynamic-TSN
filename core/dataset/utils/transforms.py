"""This module implements common functions for basic transformations of input image. 
"""
import os
import random

import numpy as np
from numpy.random import randint

from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms


def get_params(opts, size):
    w, h = size
    new_h = h
    new_w = w
    if opts.preprocess == 'resize_and_crop':
        new_h = new_w = opts.load_size
    elif opts.preprocess == 'scale_width_and_crop':
        new_w = opts.load_size
        new_h = opts.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opts.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opts.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opts, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opts.preprocess:
        osize = [opts.load_size, opts.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opts.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opts.load_size, method)))

    if 'crop' in opts.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opts.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opts.crop_size)))

    if opts.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opts.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
