import torch
from torch import nn
from collections import OrderedDict
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import pretrainedmodels

from .base_model import BaseModel
from dynamic_tsn.utils.pretrainedmodels_support import get_last_feat_dim

class TSN(BaseModel):
	"""
	"""
	def __init__(self, n_class, base_network, opts,
					n_segments, modality='RGB', consensus='avg',
					partial_bn=True, pretrained='imagenet'):
		BaseModel.__init__(self, n_class, opts)

		# Setup TSN model
		self.n_segments = opts.n_segments
		self.modality = opts.modality
		self.partial_bn = opts.partial_bn
		self.dropout = opts.dropout

		# Prepare modules
		self.base_network = base_network  # redundant but since we only have one.
		self.consensus = ConsensusModule(consensus_type)
		#TODO: configure self.new_length
		#TODO: from utils.pretrained_support import InputConfig
		# self.Input = InputConfig.TSN
		# self.input_size, self.input_mean, self.input_std = self.Input("Flow|RGB|RGBDiff").transforms
		self.prepare_modules()

	def forward(self, X):
		"""Forward pass.

		The channel length C of input X determined by modality and type n_samples (number of samples in each segment).
		Suppose modality is (optical) Flow frame that has channel c = 2.
		Then channel length of input is C = c * n_samples; Hence X (batch_size, C, H, W).

		The forward method will pass input X to the spatio or temporal network depending on modality type.
		Next, the output X_ from spatio or temporal network will be passed to a Segmental Consensus Operation (paper section 3.1).
		Finally, output X__ of Segmental Consensus can of different shape for different consensus type:
			.. Identity Consensus: X__ (batch_size, C, n_class)
			.. Average Consensus:  X__ (batch_size, n_class); Backprop of consensus will sure take care of dimension

		Refer to docs: <TODO: Add Markdown/Docs regarding Temporal Segment Network Pipeline>.
		Refer to docs: <TODO: Add Markdown/Docs regarding modality>.
		
		Parameters:
		------------------
		.. X: torch-tensor | array-like
			Inputs of shape (batch_size, channel_size, height, width)

		Returns:
		------------------
		.. output: torch-tensor | array-like
			Outputs X__ of shape (batch_size, n_class, ...)
		"""
		if self.modality == self.Input.RGBDiff:
			X = self.get_RGB_diff(X)  # shape: (, , ,)
		
		X = self.base_network(X.view((-1, *self.input_size)))  # shape: (batch_size, C, H, W)
		X = X.view((-1, self.n_segments) + X.size()[1:]) # shape: (batch_size, n_segments, )


	def prepare_modules(self):
		# 1. TODO: check self.base_network is a string (module name)
		# or already an instance of neural net module

		# 2. if base_network is a module, TODO: add 1 more last layer of base network (the nn module)
		# How: a. get it's current last layer attr, b. get in_feature, 
		# c. finally replace last layer: nn.Linear(in_feature, numclass)

		# 3. TODO: get input size, mean and std
		pass

	def train(self, mode=True):
		"""Train mode setter and setup.
		Override default pytorch <train> to freeze BatchNorm parameters except the first one.
		"""
		self.base_network.train(mode)
		# partial bn mode off
		if self.partial_bn == False:
			return

		# partial bn mode on
		count = 0
		for module in self.base_network.modules():
			if isinstance(module, nn.BatchNorm2d):
				count += 1
				if count < 2:
					continue
				# shutdown module
				module.eval()
				module.weight.requires_grad = False
				module.bias.requires_grad = False

	def modify_optim_policies(self):
		"""Modify learning policies
		Choose which parameters to update/freeze and groups active parameters based on module/layer type.
		Different groups have different lr_mult and decay_mult values (so called optim policies).
		"""
		first_conv_weight = []
		first_conv_bias = []
		normal_weight = []
		normal_bias = []
		bn = []

		is_first_conv = True
		is_first_bn2d = True

		# groups params based on module type and 
		# choose which layers to update based on type
		for m in self.base_network.modules():
			# convolution kernels module
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
				ps = list(m.parameters())
				if is_first_conv:
					first_conv_weight.append(ps[0])
					if len(ps) == 2:
						first_conv_bias.append(ps[1])
					is_first_conv = False
					continue
				normal_weight.append(ps[0])
				if len(ps) == 2:
					normal_bias.append(ps[1])
			# linear module
			elif isinstance(m, torch.nn.Linear):
				ps = list(m.parameters())
				normal_weight.append(ps[0])
				if len(ps) == 2:
					normal_bias.append(ps[1])
			# batchnorm 1d module
			elif isinstance(m, torch.nn.BatchNorm1d):
				bn.extend(list(m.parameters()))
			# batchnorm 2d module
			elif isinstance(m, torch.nn.BatchNorm2d):
				# partial bn means only first bn is active
				# the rest of bn layers are frozen
				if (not self.partial_bn) or (is_first_bn2d):
					bn.extend(list(m.parameters()))
			
			elif len(m._modules) == 0 and len(list(m.parameters())) > 0:
				raise ValueError("New atomic module type: {} without learning policy.".format(type(m)))

		return [
			{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
				'name': "first_conv_weight"},
			{'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
				'name': "first_conv_bias"},
			{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
				'name': "normal_weight"},
			{'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
				'name': "normal_bias"},
			{'params': bn, 'lr_mult': 1, 'decay_mult': 0,
				'name': "BN scale/shift"},
		]

