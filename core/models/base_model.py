import torch
from torch import nn
from torch.nn.init import normal, constant


class BaseModel(nn.Module):
	"""Base Model (Abstract)

	This is the abstract class for model. Model is a wrapper class that uses composition design pattern.
	Typically components inside model consist of base neural network module(s), criterion, and optimizers.

	The scope is to run the models, take care of learning process and progress.
	It needs to provide interface for Labs to run (train/eval), configure hyperparams + options, 
	and receive training progress reports (for logging purposes).

	To create a model class you need to implement the following:

	Abstract Methods
	------------------
	.. __init__: (overload)
		Initialize new model instance.
		
	.. forward: (overide)
		Run forward pass for each base_network. This will be called by <optimize_parameters> and <test>

	.. backward: (optional)
		Implemented for modularity if backprop logic requires modification. Call this function in
		<optimize_parameters> if implemented.

	.. modify_cli_options: (optional)
		Helper to configure <Options> instance, add model-specific options,
		and rewrite existing default options. Return modified <Options> instance.
		By default however, model-specific options should be written by 
	"""

	__abstract__ = True

	def __init__(self, opts):
		super(BaseModel, self).__init__()
		self.model_name = self.__class__.__name__
		self.opts = opts

	@staticmethod
	def modify_cli_options(parser, is_train=True):
		"""Add new dataset-specific options, and rewrite default values for existing options.
		Override this method.

		Parameters:
		------------------
		.. parser (ArgumentParser): original option parser
		.. is_train (bool): whether training phase or test phase

		Returns:
		------------------
		.. parser: the modified parser.
		"""
		return parser

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

	def prepare_model(self):
		"""Prepare necessary step.
		Configure model, layers, and hyperparams settings.
		"""
		raise NotImplementedError

	#########################
	# MODE
	#########################

	def train(self, mode=True):
		"""Train mode setter and setup.
		Override default pytorch <train> for each modules.
		"""
		super(BaseModel, self).train()

	def eval(self):
		"""Wrapper function for train(mode=False).
		"""
		self.train(mode=False)
