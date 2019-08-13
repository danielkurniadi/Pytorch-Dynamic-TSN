import os
import random
from abc import ABC, abstractmethod

import numpy as np

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from core.dataset.base_dataset import BaseDataset
from core.dataset.functionals.frame_loaders import (
	VideoMetadata,
	generate_video_indices,
	load_video_frames
)
from core.dataset.functionals import (
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
		"""Initialize the class; save the options in the class

		Parameters:
			opts (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
			phase (str)-- specify if this dataset loader is used for [train | val | test]
		"""
		self.opts = opts
		self.phase = phase
		self.configure_dataset_settings(opts)
		self.setup_metadata(opts)

		# configure image property

		# configure transforms; TODO: seperate transforms config logic
		self.transforms = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def configure_dataset_settings(self, opts):
		# Configs for input video
		self.modality = opts.modality
		self.n_segments = opts.n_segments			# segment video to n_segment
		self.sample_length = opts.sample_length		# sample length per segment
		self.random_shift = opts.random_shift

		# Configs for image file naming
		self.image_extension = opts.img_ext

		if self.modality in ['RGB', 'RGBDiff', 'ARP']:
			self.img_name_tmpl = self.modality.lower() + '_{:05d}' + self.image_extension
		elif self.modality == 'Flow':
			self.img_name_tmpl = self.modality.lower() + '_{}_' + '_{:05d}' + self.image_extension
		else:
			raise NotImplementedError(
				'Unsupported Modality for Temporal Dataset. '
				'Please implement for specified modality: %s' % self.modality
			)

	def setup_metadata(self, opts):
		super(TemporalDataset, self).setup_metadata(opts)

		# Parse and map metadata to VideoMetadata (named_tuple)
		self.video_metadata_list = [
			VideoMetadata(directory, label, self.modality)
			for directory, label in self.metadata_list
		]
		return self.video_metadata_list

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser.add_argument('--modality', type=str, default='RGB',
			help='Chooses modality for intended dataset. [RGB | RGBDiff | Flow | ARP]')
		parser.add_argument('--n_segments', type=int, default=3,
			help='Number of video segments.')
		parser.add_argument('--sample_length', type=int, default=5,
			help='Number of frames to be sampled in each segment')
		parser.add_argument('--random_shift', action='store_true',
			help='Whether to sample video frames at random shift or at the middle of each segments')

		return parser

	def __getitem__(self, index):
		metadata = self.video_metadata_list[index]
		directory = metadata.directory		# directory of video frames (single dataset)
		n_frames = metadata.n_frames		# number of frames in directory having same modality
		label = metadata.label				# class label

		frame_indices = generate_video_indices(
			self.n_segments, n_frames,
			self.sample_length, self.random_shift)

		# iterate frame_generator to get frames and respective label
		imgs = list(
			load_video_frames(frame_indices, directory,
				self.img_name_tmpl, self.modality)
		)
		imgs = [self.transforms(img) for img in imgs]

		return imgs, label

