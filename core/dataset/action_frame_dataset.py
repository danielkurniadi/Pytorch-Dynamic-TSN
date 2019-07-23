import os
import random
from abc import ABC, abstractmethod

import numpy as np

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from core.dataset.base_dataset import SplitFileDataset
from core.dataset.utils.videoframe import VideoFrameGenerator
from core.dataset.utils import (
	check_filepath,
	read_strip_split_lines
)


class ActionFrameDataset(SplitFileDataset):
	"""Action Frame Dataset which support generating data from pre-processed video frames.
	
	Video frames can be of extension .jpg, .png, .jpeg, etc. They represent frames from video rather than static images.
	The splitfile here contains three columns: folder path, num_frames, and labels for video frames in a folder (each folder represent 1 video).
	"""

	def __init__(self, opts):
		super(ActionFrameDataset, self).__init__(opts)
		self.opts = opts
		self.modality = self.opts.dataset_mode
		self.new_length = self.opts.input_channels
		self.img_name_tmpl = self.opts.img_name_tmpl
		self.random_frame_shift = self.opts.random_frame_shift

		self.create_frame_generators()

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser = SplitFileDataset.modify_cli_options(parser, is_train)
		parser.add_argument('--img_name_tmpl', type=str, default='img_{:05d}.png',
			help='Image name template with (python curly braces format) for each frame in one video folder')
		parser.add_argument('--random_frame_shift', action='store_true',
			help='Whether to sample video frames at random shift or at the middle of each segments')

		return parser

	def __len__(self):
		return len(self.frame_generators)

	def __getitem__(self, index):
		frame_generator = self.frame_generators[index]
		imgs = list(frame_generator)

		return imgs

	#-----------------------
	# Frames generator
	#-----------------------

	def create_frame_generators(self):
		""" Create video frame generators for each video.
		
		Returns:
			list of video-frame generators, with size equals to number of video dataset
		"""

		self.frame_generators = []
		
		for line in self.lines:
			directory, n_frames, label = line
			n_frames = int(n_frames)
			label = int(n_frames)

			frame_generator = VideoFrameGenerator(
				directory, n_frames, label,
				self.img_name_tmpl, self.random_shift
			)	
			self.frame_generators.append(frame_generator)

		return self.frame_generators

#########################################################################################

class RGBDataset(ActionFrameDataset):
	"""Non-preprocessed frame dataset (RGB channels) which support generating data from pre-processed video frames.
	
	Video frames can be of extension .jpg, .png, .jpeg, etc. 
	They represent frames from video rather than static images.
	
	The splitfile here contains three columns: folder path, num_frames, and labels 
	for video frames in a folder (each folder represent 1 video).
	"""

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser = ActionFrameDataset.modify_cli_options(parser, is_train)
		parser.set_defaults(dataset_mode = 'RGB')

		return parser

#########################################################################################

class FlowDataset(ActionFrameDataset):
	"""Dense/Warped optical flow frame dataset (X,Y channels) which support generating data from pre-processed video frames.
	
	Video frames can be of extension .jpg, .png, .jpeg, etc. 
	They represent frames from video rather than static images.
	
	The splitfile here contains three columns: folder path, num_frames, and labels 
	for video frames in a folder (each folder represent 1 video).
	"""

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser = ActionFrameDataset.modify_cli_options(parser, is_train)
		parser.set_defaults(
			dataset_mode = 'Flow',
			img_name_tmpl = 'img_{}_{:05d}.png'
		)

		return parser

#########################################################################################

class RGBDiffDataset(ActionFrameDataset):
	"""Non-preprocessed frame dataset (RGB channels) which support generating data from pre-processed video frames.
	
	Video frames can be of extension .jpg, .png, .jpeg, etc. 
	They represent frames from video rather than static images.
	
	The splitfile here contains three columns: folder path, num_frames, and labels 
	for video frames in a folder (each folder represent 1 video)."""

	def __init__(self, opts):
		super(RGBDiffDataset, self).__init__(opts)
		self.new_length += 1

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser = ActionFrameDataset.modify_cli_options(parser, is_train)
		parser.set_defaults(dataset_mode = 'RGB')

		return parser
