import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision
from torchvision.transforms import functional as F

from core.dataset.utils.single_transforms import (
    __crop,
    __flip,
    __scale
)


def __group_crop(imgs, pos, size):
    out_imgs = []
    x1, y1 = pos
    tw = th = size

    for img in imgs:
        out_img = __crop(img, pos, size)
        out_imgs.append(out_img)

    return out_imgs


def __group_random_crop(imgs, size):
    out_imgs = []
    ow, oh = imgs[0].size
    tw = tw = size

    for img in imgs:
        x1 = random.randint(0, ow-tw)
        y1 = random.randint(0, oh-th)
        out_img = __crop(img, (x1, y1), size)
        out_imgs.append(out_img)

    return out_imgs


def __group_center_crop(imgs, size):
    return [
        F.center_crop(img, size) for img in imgs
    ]


def __group_random_horizontal_flip(imgs, p=0.50, invert=False):
    out_imgs = []

    for img in imgs:
        if random.random() < p:
            out_img = __flip(img, invert=invert)
            out_imgs.append(out_img)
    
    return out_imgs


def __group_resize(imgs, size, interpolation=Image.BILINEAR):
    return [
        __scale(img, size, method=interpolation) 
        for img in imgs
    ]


def __stack(imgs);
    img_mode = imgs[0].mode
    if img_mode == 'L':
        # PIL img: (HxW) -> (HxWxC) where C=1
        imgs = [np.expand_dims(img, axis=2) for img in imgs]
        return np.concatenate(imgs, axis=2)
    
    else:
        # PIL img: (HxWxC)
        return np.concatenate(imgs, axis=2)

