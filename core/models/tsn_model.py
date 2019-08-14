import os
import time
import shutil
import argparse

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import pretrainedmodels

from .base_model import BaseModel
from core.networks import (
	init_net,
	get_pretrainedmodels
)


class TSNModel(BaseModel):
	""" Temporal Segmentation Network

	This is the TSN model class. TSN model take temporal streams of the two-streams convnet 
	and add video length segmentation. The temporal streams are extended to include many modalities, including
	RGB, RGBDiff, Flow (Optical flow). 

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
	def __init__(self, opts):
		# Model configurations
		self.num_classes = opts.output_nc
		self.input_nc = opts.input_nc
		self.arch = opts.arch
		self.pretrained = opts.pretrained
		self.norm = opts.norm
		self.modality = opts.modality

		# Pretraining settings
		pretrained_settings = get_pretrained_settings(opts, opts.arch)

		# Prepare networks for training
		self.net = get_pretrained_networks(opts.arch)
		self.configure_networks(opts)
		self.configure_consensus_layers(opts)

		# Prepare optimizer for training
		self.optim_policies = self.configure_optim_oplicies()
		self.optimizers = self.configure_optimizer(opts)
		
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
		X = self.net(X)
		X

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
		