import os
import random
from abc import ABC, abstractmethod

import numpy as np

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from core.dataset.base_dataset import BaseDataset
from core.dataset.utils.videoframe import VideoFrameGenerator
from core.dataset.utils import (
	check_filepath,
	read_strip_split_lines
)


class TemporalDataset(BaseDataset):
	"""Temporal (action) dataset

    Temporal dataset from video frames which support loading data from pre-processed video frames.
    Temporal dataset has new channel length that comes from stacking images ordered chronogically.
    The channel axis represent time series.

	Temporal action frames initially take image of extension .jpg, .png, .jpeg, etc. 
	They represent frames taken from video.
	
	The metadata for this dataset, also called split file, contains three columns:
        ... folder path, num_frames, and labels
	"""

	def __init__(self, opts, phase='train'):
		super(BaseDataset, self).__init__(opts, phase)
		self.opts = opts

		# configure image file naming
		self.modality = opts.dataset_mode
		self.image_extension = opts.img_ext

        # configure input modality
		if self.modality in ['RGB', 'RGBDiff', 'ARP']:
			self.img_name_tmpl = self.modality + '_{:05d}_' + self.image_extension
			self.img_name_tmpl = self.img_name_tmpl.lower()

		elif self.modality == 'Flow':
			self.img_name_tmpl = self.modality + '_{}_' + '_{:05d}_' + self.image_extension

		else:
			raise NotImplementedError(
				'Unsupported Modality for Action Dataset. '
				'Please implement for specified modality: %s' % self.modality
			)

		# configure image property
		self.input_channels = opts.input_channels
		self.input_size = opts.input_size
		self.input_means = opts.input_means
		self.input_std = opts.input_std

		# configure transforms
		self.crop_size = opts.crop_size
		self.transforms = transforms.Compose([
			transforms.Resize(self.input_size),
			transforms.CenterCrop(self.crop_size),
			transforms.ToTensor(),
			transforms.Normalize(self.input_means, self.input_std)
		])

		# configure sampling
		self.random_frame_shift = self.opts.random_frame_shift
		
		# video generators
		self.frame_generators = self.create_frame_generators()

	@staticmethod
	def modify_cli_options(parser, is_train):
        parser.add_argument('--modality', type=str, default='RGB',
            help='Chooses modality for intended dataset. [RGB | RGBDiff | Flow | ARP]')
		parser.add_argument('--random_frame_shift', action='store_true',
			help='Whether to sample video frames at random shift or at the middle of each segments')

		return parser

	def __len__(self):
		return len(self.frame_generators)

	def __getitem__(self, index):
		frame_generator = self.frame_generators[index]
		imgs, label = list(frame_generator)		# iterate frame_generator to get frames and respective label

        # TODO: transforms
		return imgs, label

	#-----------------------
	# Frames loader
	#-----------------------

	def create_frame_generators(self):
		""" Create video frame generators to load frames for each video.

		Returns:
			.. vid generators (list): list of generators, each handle frame iteration 
				of one video.
		"""
		frame_generators = []

		for line in self.metadata:
			# parse metadata line by line
			directory, n_frames, label = line
			n_frames = int(n_frames)	# convert to int
			label = int(n_frames)		# convert to int

			# using information in metadata to
			# create frames generator for each video
			frame_generator = VideoFrameGenerator(
				directory,
				n_frames,
				label,
				self.img_name_tmpl,
				self.random_shift
			)

			frame_generators.append(frame_generator)

		return frame_generators
