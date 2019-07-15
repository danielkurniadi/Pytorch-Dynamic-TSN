import os

import torch
from torch.autograd import Variable

from collections import OrderedDict

from core import layers


class BaseModelWrapper:
	"""Base Model Wrapper (Abstract)

	This is the abstract class for models. Model is a wrapper class that uses composition design pattern.
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

	__abstract__ = True

	def __init__(self, n_classes):
		"""Initialise models.

		Parameters
		------------------
		.. n_class: int
			Channel length of last outputs, number of classification classes.

		"""
		self.model_name = self.__class__.__name__

		# Device configuration
		self.gpu_ids = gpu_ids
		self.device = torch.device('cuda:0') \
			if torch.cuda.is_available() else torch.device('cpu')

		# Objectives
		self.n_classes = n_classes

	#########################
	# CORE
	#########################

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

	#########################
	# INPUT
	#########################

	@property
	def input_size(self):
		# Convention: C x H x W
		if not hasattr(self, '__input_channel'):
			self.__input_size = (3, 244, 244)
		
		return self.__input_size

	@property
	def input_means(self):
		if not hasattr(self, '__input_mean'):
			self.__input_means = [0.5] * self.__input_size[0] # times channel length
		
		return self__input_means

	@property
	def input_std(self):
		if not hasattr(self, '__input_std'):
			self.__input_std = [0.5] * self.__input_size[0] # times channel length
		
		return self.__input_std

	@input_size.setter
	def input_size(self, input_size):
		if not isinstance(input_means, (list, tuple)):
			raise ValueError("Setting %s.input_means must be a colletion type"
				" matching the channel length. Trying to set input_means to type: %s" 
				% (self.model_name, type(input_means)))

		self.__input_size = input_size
	
	@input_mean.setter
	def input_means(self, input_means):
		if not isinstance(input_means, (list, tuple)):
			raise ValueError("Setting %s.input_means must be a colletion type"
				" matching the channel length. Trying to set input_means to type: %s" 
				% (self.model_name, type(input_means)))
		
		self.__input_means = input_means

	@input_std.setter
	def input_std(self, input_std):
		if not isinstance(input_std, (list, tuple)):
			raise ValueError("Setting %s.input_std must be a colletion type"
				" matching the channel length. Trying to set input_std to type: %s" 
				% (self.model_name, type(input_std)))

		self.__input_std = input_std


class SingleModelWrapper(BaseModelWrapper):
	"""Wrapper for single architecture model (Abstract)

	This is the abstract wrapper class for single-architecture model. Hence this class only
	control the flow of a single architecture from forward, backwards, optim policy.
	
	Wrapper class uses composition design pattern. Typically components 
	inside model consist of base neural network module(s), criterion, and optimizers.
	"""

	__abstract__ = True

	def __init__(self, base_model, n_classes):
		# Architecture
		self.base_model = base_model
		self.model_name = self.__class__.__name__

		# Device configuration
		self.gpu_ids = gpu_ids
		self.device = torch.device('cuda:0') \
			if torch.cuda.is_available() else torch.device('cpu')

		# Objectives
		self.n_classes = n_classes

	def prepare_model(self):
		"""Prepare necessary step.
		Configure model, layers, and hyperparams settings.
		"""
		raise NotImplementedError
