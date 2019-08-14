import random
import numpy as numpy
from collections.abc import Iterable
from PIL import Image, ImageOps

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale(img, size, interpolation=Image.BILINEAR):
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            # print("SCALE ", img.size)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            # print("SCALE ", img.size)
            return img.resize((ow, oh), interpolation)
    else:
        img = img.resize(size[::-1], interpolation)
        # print("SCALE ", img.size)
        return img


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        # print("CROP ", img.size)
        return img.crop((x1, y1, x1 + tw, y1 + th))
    # print("CROP ", img.size)
    return img


def __flip(img, invert):
    # print("FLIP ", img.size)
    out_img =  img.transpose(Image.FLIP_LEFT_RIGHT)
    if invert:
        out_img = ImageOps.invert(img)
    
    return out_img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
