import os
import time
import shutil
import argparse

from core.options.train_options import (
	BaseOptions,
	TrainOptions
)

from core import models, dataset
from core.models import find_model_using_name

from core.dataset import (
	find_dataset_using_name,
	get_option_setter
)

def print_basic_opts(opts):
	print("------------------- Runtime Configs -----------------")
	print(".. --name %s" %opts.name)
	print(".. --gpu_ids %s" %opts.gpu_ids)
	print(".. --serial_batches %s" %opts.serial_batches)
	print(".. --num_threads %s" %opts.num_threads)
	print(".. --checkpoints_dir %s" %opts.checkpoints_dir)
	
	print("------------------- Model Configs -------------------")
	print(".. --model %s" %opts.model)
	print(".. --n_classes %s" %opts.output_nc)
	print(".. --input_channels %s" %opts.input_nc)
	print(".. --norm %s" %opts.norm)
	print(".. --init_type %s" %opts.init_type)
	print(".. --init_gain %s" %opts.init_gain)

	print("------------------- Input Configs -------------------")
	print(".. --dataset_mode %s" %opts.dataset_mode)
	print(".. --batch_size %s" %opts.batch_size)
	print(".. --input_size %s" %opts.input_size)
	print(".. --input_range %s" %opts.input_means)
	print(".. --input_std %s" %opts.input_std)
	print(".. --preprocess %s" %opts.preprocess)
	print(".. --crop_size %s" %opts.crop_size)
	print(".. --no_flip %s" %opts.no_flip)

	print("------------------ Additional Params ----------------")
	print(".. --load_epoch %s" %opts.load_epoch)
	print(".. --verbose %s" %opts.verbose)
	print(".. --suffix %s" %opts.suffix)


def print_train_opts(opts):
	print("------------------- Runtime Configs -----------------")
	print(".. --save_freq_epoch %s" %opts.save_freq_epoch)
	print(".. --continue_last %s" %opts.continue_last)
	
	print("------------------- Learning Configs -----------------")
	print(".. --epochs %s" %opts.epochs)
	print(".. --momentum %s" %opts.momentum)
	print(".. --lr %s" %opts.lr)
	print(".. --lr_policy %s" %opts.lr_policy)

	print(".. --lr_decay_factor %s" %opts.lr_decay_factor)
	print(".. --lr_decay_iters %s" %opts.lr_decay_iters)
	print(".. --eval_freq_epoch %s" %opts.eval_freq_epoch)
	print(".. --print_freq_iters %s" %opts.print_freq_iters)


def options_initialise_default(options_obj, train=None):
	
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	) 

	print("- Initializing...")
	parser = options_obj.initialize(parser)
	
	opts, _ = parser.parse_known_args()
	print_basic_opts(opts)

	if train:
		print_train_opts(opts)


def test_base_option_initialise():
	print("-----------------------------------------------------")
	print("Test: test_base_option_initialise")
	base_options = BaseOptions()
	options_initialise_default(base_options)
	print("-----------------------------------------------------\n\n")


def test_train_option_initialise():
	print("-----------------------------------------------------")
	print("Test: test_train_option_initialise")
	train_options = TrainOptions()
	options_initialise_default(train_options)
	print("-----------------------------------------------------\n\n")


def test_find_dataset_using_name():
	print("-----------------------------------------------------")
	print("Test: test_find_dataset_using_name")
	
	dataset_name = 'RGB'
	dataset = find_dataset_using_name(dataset_name)
	dataset_option_setter = get_option_setter(dataset_name)
	print(dataset)

	dataset_name = 'RGBDiff'
	dataset = find_dataset_using_name(dataset_name)
	dataset_option_setter = get_option_setter(dataset_name)
	print(dataset)

	dataset_name = 'Flow'
	dataset = find_dataset_using_name(dataset_name)
	dataset_option_setter = get_option_setter(dataset_name)
	print(dataset)

	print("-----------------------------------------------------\n\n")

	
def test_train_options_dataset_specified():
	print("-----------------------------------------------------")
	print("Test: test_train_options_dataset_specified")
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	train_options = TrainOptions()
	parser = train_options.initialize(parser)
	opts, _ = parser.parse_known_args()

	dataset_name = opts.dataset_mode
	dataset_option_setter = get_option_setter(dataset_name)
	parser = dataset_option_setter(parser, is_train=train_options.isTrain)
	opts, _ = parser.parse_known_args()  # parse again with new defaults

	print_basic_opts(opts)
	print_train_opts(opts)
	
	dataset = find_dataset_using_name(dataset_name)
	print(dataset)

	print("------------------- Dataset Config -----------------")
	print(".. --dataset_mode %s" %opts.dataset_mode)
	print(".. --img_name_tmpl %s" %opts.img_name_tmpl)
	
	print(".. --split_dir %s" %opts.split_dir)

	print("-----------------------------------------------------\n\n")


def test_find_model_using_name():
	model_name = 'TSN'
	model = find_model_using_name(model_name)
	print(model)


def test_train_options_model_specified():
	print("-----------------------------------------------------")
	print("Test: test_train_options_model_specified")
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	train_options = TrainOptions()
	parser = train_options.initialize(parser)
	opts, _ = parser.parse_known_args()

	model_name = opts.model
	model_option_setter = models.get_option_setter(model_name)
	parser = model_option_setter(parser, is_train=train_options.isTrain)
	opts, _ = parser.parse_known_args()  # parse again with new defaults

	print_basic_opts(opts)
	print_train_opts(opts)
	
	model = find_model_using_name(model_name)
	print(model)

	print("------------------- Dataset Config -----------------")
	print(".. --input_means %s" %opts.input_means)
	print(".. --input_size %s" %opts.input_size)
	print(".. --input_range %s" %opts.input_range)
	print(".. --input_channels %s" %opts.input_nc)
	print(".. --input_std %s" %opts.input_std)

	print("-----------------------------------------------------\n\n")


def test_options_parse():
	print("-----------------------------------------------------")
	print("Test: test_train_options_model_specified")
	
	opts = TrainOptions().parse()
	print("-----------------------------------------------------")
	


if __name__ == '__main__':
	test_base_option_initialise()
	test_train_option_initialise()
	test_find_dataset_using_name()
	test_train_options_dataset_specified()
	test_find_model_using_name()
	test_train_options_model_specified()
	test_options_parse()
