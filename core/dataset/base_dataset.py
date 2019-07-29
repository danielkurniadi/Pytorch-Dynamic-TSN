"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
"""
import os
import random
from abc import ABC, abstractmethod

import numpy as np

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

# Dataset Utility
from core.dataset.utils.videoframe import VideoFrameGenerator
from core.dataset.utils import (
	check_filepath,
	read_strip_split_lines
)


class BaseDataset(data.Dataset, ABC):
	"""This class is an abstract base class (ABC) for datasets.

	To create a subclass, you need to implement the following four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point.
	-- <modify_cli_options>:            (optionally) add dataset-specific options and set default options.
	"""

	def __init__(self, opts, phase='train'):
		"""Initialize the class; save the options in the class

		Parameters:
			opts (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
			phase (str)-- specify if this dataset loader is used for [train | val | test]
		"""
		self.opts = opts

		# Warning! : Use your (previous) [--out_prefix] options in <split_cli>
		# to specify [--name] options when training
		self.split_idx = opts.split_idx
		if phase in ['train', 'val']:
			split_filename = "{}_{}_split_{}.txt".format(
				opts.name, phase, self.split_idx)
		else:
			split_filename = "{}_{}_split.txt"

		self.split_file = os.path.join(
			opts.split_dir, split_filename)

		# obtain metadata from split file
		self.metadata = read_strip_split_lines(self.split_file)

	@staticmethod
	def modify_cli_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		return parser

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return len(self.metadata)

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		pass
