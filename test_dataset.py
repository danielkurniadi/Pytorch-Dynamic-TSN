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
	create_dataset
)

def test_create_dataset():
	print("-----------------------------------------------------")
	print("Test: test_create_dataset")

	opts = TrainOptions().parse()
	print(".. Fetching data from paths specified in splitfile: %s" %opts.split_file)

	dataset = create_dataset(opts)
	dataset_size = len(dataset)
	print('.. The number of training images = %d' % dataset_size)
	
	dataiter = iter(dataset)
	data, label = next(dataiter)
	print(".. Data Tensor")
	print(".. >> ", data)
	print("-----------------------------------------------------")


if __name__ == '__main__':
	test_create_dataset()
