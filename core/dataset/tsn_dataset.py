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

#########################################################################################

class VideoFrameGenerator(object):
    """
    """
    def __init__(self, directory, n_frames,
                label, img_name_tmpl='img_{:05d}.jpg',
                random_shift=True):
        self.directory = directory
        self.n_frames = n_frames
        self.label = label
        self.img_name_tmpl = img_name_tmpl
        self.random_shift = random_shift

    #-----------------------
    # Generator methods
    #-----------------------

    def __iter__(self):
        """ Setup for iteration.

        Returns:
            VideoRecord: self
        """
        self._iter_count = 0
        self.vid_indices = self.generate_video_indices()
        self._max_iter = len(self.vid_indices)
        return self

    def __next__(self):
        """
        """
        if self._iter_count > self._max_iter:
            raise StopIteration

        videodata = self.load_video_frame(self._iter_count)
        self._iter_count += 1

        return videodata

    #-----------------------
    # Indices handlers
    #-----------------------

    def generate_video_indices(self):
        """ Generate indices of video frames from sampling/median

        eg: seg_indices = [0, 4, 9]; new_length = 3
            output  = [0, 0, 0, 4, 4, 4, 9, 9, 9] + [0, 1, 2, 0, 1, 2, 0, 1, 2]
                    = [0, 1, 2, 4, 5, 6, 9, 10, 11]
        """
        if self.random_shift:
            seg_indices = generate_sample_seg_indices(n_segments,
                n_frames, new_length)
        else:
            seg_indices = generate_median_seg_indices(n_segments,
                n_frames, new_length)

        repeats = np.repeat(np.arange(self.new_length), self.n_segments)
        full_indices = np.repeat(seg_indices, self.new_length) + repeats
        
        return full_indices

    #-----------------------
    # Image Loading
    #-----------------------

    def load_video_frame(self, idx):
        """
        """
        if self.modality in ['RGB', 'RGBDiff']:
            filepath = os.path.join(self.directory, self.img_name_tmpl.format(idx))
            return load_rgb_image(filepath)
        elif self.modality == 'Flow':
            x_img_path = os.path.join(self.directory, self.img_name_tmpl.format('x', idx))
            y_img_path = os.path.join(self.directory, self.img_name_tmpl.format('y', idx))
            
            return load_flow_image(x_img_path, y_img_path)

#########################################################################################

class TSNDataset(BaseDataset):
    """
    """
    def __init__(self, list_file,
                modality='RGB', new_length=1,
                img_name_tmpl='img_{:05d}.jpg',
                random_shift=True, randseed=42):
        self.list_file = list_file
        self.n_frames = n_frames
        self.label = label
        self.modality = modality
        self.new_length = new_length
        self.img_name_tmpl = img_name_tmpl
        self.random_shift = random_shift

        if self.modality == 'RGBDiff':
            new_length += 1  # Need 1 more frame to calculate diff

        self.parse_list_to_vids()

    #-----------------------
    # Frames generator
    #-----------------------

    def parse_list_to_vids(self):
        """
        """
        self.frame_generators = []
        
        f = open(self.list_file, 'w')
        for line in f.readlines():
            directory, n_frames, label = line.strip().split(' ')
            n_frames = int(n_frames)
            label = int(n_frames)
            
            self.frame_generators.append(VideoFrameGenerators(
                directory, n_frames, label,
                self.img_name_tmpl, self.random_shift
            ))
        
        return self.frame_generators

    #-----------------------
    # Dataset methods
    #-----------------------

    @staticmethod
    def modify_cli_options(parser, is_train=True):
        parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics']
            help='Choices of available datasets')
        parser.add_argument('--modality', type=str, default='Flow',
            help='Modality of dataset [RGB | RGBDiff| Flow | RankPool | OpenPose]')

        return parser

    def __getitem__(self, idx):
        frame_gen = self.frame_generators[idx]
        imgs = list(frame_gen)

        return imgs

    def __len__(self):
        return len(self.frame_generators)
