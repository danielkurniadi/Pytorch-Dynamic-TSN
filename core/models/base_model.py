import os

import torch
from torch.autograd import Variable

from collections import OrderedDict
from abc import ABC, abstractmethod

from core import layers

class BaseModel(ABC):
	"""Base Model (Abstract)

	This is the abstract class for models. Model is a wrapper class that uses component design pattern.
	Typically components inside model consist of base neural network module(s), criterion, and optimizers.
	To create a model class you need to implement the following:

	Abstract Methods
	------------------
	.. __init__: (overload)
		Initialize new model instance.
		First, call <BaseModel.__init__(self, base_networks, n_class, opts)> before initializing the model.

	.. forward: (overide)
		Run forward pass for each base_network. This will be called by <optimize_parameters> and <test>
	
	.. optimize_parameters: (override)
		Run backward prop, calculate loss, gradients, and update network weights.
		Called for every iterations.

	.. backward: (optional)
		Implemented for modularity if backprop logic is complicated. Call this function in
		<optimize_parameters> if implemented.

	.. configure_model_options: (optional)
		Helper to configure <Options> instance, add model-specific options,
		and rewrite existing default options. Return modified <Options> instance.
		By default however, model-specific options should be written by 
	"""

	def __init__(self, n_class):
		"""Initialise models.

		Parameters
		------------------
		.. n_class: int
			Channel length of last outputs, number of classification classes.

		.. opts: Options (obj)
			Options for train/val/test (either one).

		"""
		self.model_names = self.__class__.__name__

		self.device = torch.device('cuda:%d' % self.gpu_ids[0]) if self.gpu_ids else torch.device('cpu')  # decide CUDA devices or CPU
		self.save_dir = os.path.join(opts.checkpoints_dir, opts.title)  # save checkpoint directory

		self.n_class = n_class
		self.optimizers = None
		self.criterions = None

	def forward(self, X):
		"""Forward pass.
		
		Setup input and calling <forward> for each model in base_models accordingly.
		Then reshape or add aditional processing to output of each model here if necessary.
		Called by <optimize_parameters>.

		Parameters:
		------------------
		.. X: torch-tensor | array-like
			Inputs of shape (batch_size, channel_size, height, width)

		Returns:
		------------------
		.. output: torch-tensor | array-like
			Outputs of shape (batch_size, n_class, ...)

		"""
		raise NotImplementedError
	
	def optimize_parameters(self):
		"""Run forward, calculate loss, gradients, and update network weights. 
		Called for each iterations in train/val/test.
		"""
		raise NotImplementedError

	def train(self, mode=True):
		"""Train mode setter and setup.
		Override default pytorch <train> for each modules.
		"""
		pass

	def eval(self):
		"""Wrapper function for train(mode=False).
		"""
		self.train(mode=False)
