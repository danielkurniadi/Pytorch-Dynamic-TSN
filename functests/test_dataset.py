import os
import time
import shutil
import argparse

from core.options.train_options import (
	BaseOptions,
	TrainOptions
)

from core.dataset import (
	find_dataset_using_name,
	get_option_setter,
	create_dataset_loader
)

def test_create_dataset():
	print("-----------------------------------------------------")
	print("Test: test_create_dataset")

	opts = TrainOptions().parse()
	print(".. Fetching data from paths specified in split_dir: %s" %opts.split_dir)

	dataset = create_dataset_loader(opts, phase='train')
	dataset_size = len(dataset)
	print('.. The number of training images = %d' % dataset_size)
	
	dataiter = iter(dataset)
	data, label = next(dataiter)
	print(".. Data Tensor")
	print(".. >> ", data)
	print("-----------------------------------------------------")


if __name__ == '__main__':
	test_create_dataset()
