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
from core.dataset.functionals.frame_loaders import VideoMetadata
from core.dataset.functionals import (
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

	def configure_dataset_settings(self, opts):
		""" Configure settings of dataset instances """
		pass
		
	def setup_metadata(self, opts):
		""" Setup and prepare metadata before data loading
		
		Returns:
			metadata_list (list): the metadata list containing metadata stored in object/array-like
		"""
		# Warning! : Use your (previous) [--out_prefix] options in <split_cli>
		# to specify [--name] options when training
		self.split_idx = opts.split_idx
		if self.phase in ['train', 'val']:
			split_filename = "{}_{}_split_{}.txt".format(
				opts.name, self.phase, self.split_idx)
		else:
			split_filename = "{}_test_split.txt".format(opts.name)

		self.split_file = os.path.join(
			opts.split_dir, split_filename)
		# obtain metadata from split file
		self.metadata_list = read_strip_split_lines(self.split_file)

		# Guards for edge cases
		if not self.metadata_list:
			# Case 1: Metadata read return no value
			raise ValueError("Failed to read metadata from %s split file" % self.split_file)
		if len(self.metadata_list[0]) != 2:
			# Case 2: Metadata must have 2 values; datasetpath and label
			raise ValueError("Metadata rows should only contains 2 values: datasetpath and label.")
		
		return self.metadata_list


	def __len__(self):
		"""Return the total number of images in the dataset."""
		return len(self.metadata_list)

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		raise NotImplementedError
