"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import os
import random
from abc import ABC, abstractmethod

import numpy as np

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset, ABC):
	"""This class is an abstract base class (ABC) for datasets.

	To create a subclass, you need to implement the following four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point.
	-- <modify_cli_options>:            (optionally) add dataset-specific options and set default options.
	"""

	def __init__(self, opts):
		"""Initialize the class; save the options in the class

		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		self.opts = opts
		self.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')

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

	@abstractmethod
	def __len__(self):
		"""Return the total number of images in the dataset."""
		return 0

	@abstractmethod
	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		pass

#########################################################################################

class SplitFileDataset(BaseDataset):
	"""This class is an abstract datasets which uses splitfiles for pointing path to
	image/video data.
	"""

	def __init__(self, opts):
		super(SplitFileDataset, self).__init__(opts)
		self.split_file = opts.split_file
		self.lines = read_strip_split_lines(self.split_file)

	@staticmethod
	def modify_cli_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		parser.add_argument('--split_file', type=str, default='default_train_split_0.txt',
			help='Path to split textfile. See /docs/README.md#Dataset about split file')

		return parser

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return len(self.lines)

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		# parse self.lines here
		pass
