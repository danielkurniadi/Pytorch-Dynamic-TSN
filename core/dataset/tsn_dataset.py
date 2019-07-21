import os
import numpy as np
from numpy.random import randint

import core.dataset.base import BaseDataset

from PIL import Image


#-----------------------
# Utils
#-----------------------

def load_rgb_image(filepath):
    """
    """
    return Image(filepath).convert('RGB')


def load_flow_image(x_img_path, y_img_path):
    """
    """
    x_img = Image.open(x_img_path).convert('L')
    y_img = Image.open(y_img_path).convert('L')

    return x_img, y_img


def generate_sample_seg_indices(n_segments, n_frames, new_length):
    """
    """
    average_duration = (n_frames - new_length + 1) // n_segments
    if average_duration > 0:
        offsets = np.arange(n_segments) * average_duration + \
            np.random.randint(average_duration, size=n_segments)
    elif n_frames > n_segments:
        offsets = np.sort(np.random.randint(n_frames - new_length + 1, size=(n_segments)))
    else:
        offsets = np.zeros((n_segments,)).astype(int)
    return offsets + 1


def generate_median_seg_indices(n_segments, n_frames, new_length):
    """
    """
    average_duration = (n_frames - new_length + 1) / float(n_segments)
    if average_duration > 0:
        offsets = np.arange(n_segments) * (average_duration) + average_duration / 2.0
        offsets = offsets.astype(int)
    else:
        offsets = np.zeros((n_segments, ))
    return offsets + 1

