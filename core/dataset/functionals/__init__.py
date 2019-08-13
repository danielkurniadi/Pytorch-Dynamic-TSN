"""This module implements common functions for reading, loading, indicing image/frame dataset
"""
import os
import random

import numpy as np
from numpy.random import randint

from PIL import Image

from core.utils.file_system import (
	search_files_recursively,
)


#-----------------------
# Files
#-----------------------

def check_filepath(filepath):
	if not os.path.isfile(filepath):
		raise FileNotFoundError("File doesn't exists. "
			"Filepath: %s" % filepath)
	return True


def read_strip_split_lines(filepath, sep=' '):
	data_rows = []
	check_filepath(filepath)
	with open(filepath, 'r') as f:
		for line in f:
			data_row = line.strip().split(sep)
			if data_row:
				data_rows.append(data_row)
	return data_rows


def expand_split_folders_to_filepaths(
	split_list,
	label_idx = -1,
):
	"""
	"""
	filepaths, newlabels = [], []
	for metadata in split_list:
		folder_path = metadata[0]			# by convention, any path to dataset is put in first column
		label = int(metadata[label_idx])	# by convention, any label of dataset is put in last column

		files = search_files_recursively(folder_path)
		labels = [label] * len(files)
		
		filepaths.extend(files)
		newlabels.extend(labels)

	return list(zip(filepaths, newlabels))


#-----------------------
# Indices
#-----------------------

def generate_sample_seg_indices(n_segments, n_frames, new_length):
	"""
	"""
	average_duration = (n_frames - new_length + 1) // n_segments
	if average_duration > 0:
		offsets = np.arange(n_segments) * average_duration + \
			np.random.randint(average_duration, size=n_segments).astype(int)
	elif n_frames > n_segments:
		offsets = np.sort(np.random.randint(
			n_frames - new_length + 1, size=(n_segments))).astype(int)
	else:
		offsets = np.zeros((n_segments,)).astype(int)
	return offsets + 1


def generate_median_seg_indices(n_segments, n_frames, new_length):
	"""
	"""
	average_duration = (n_frames - new_length + 1) / float(n_segments)
	if average_duration > 0:
		offsets = np.arange(n_segments) * (average_duration) + average_duration / 2.0
		offsets = offsets.astype(int)
	else:
		offsets = np.zeros((n_segments,)).astype(int)
	return offsets + 1


#-----------------------
# Images
#-----------------------

def load_rgb_image(filepath):
	"""
	"""
	return Image.open(filepath).convert('RGB')


def load_flow_image(x_img_path, y_img_path):
	"""
	"""
	x_img = Image.open(x_img_path).convert('L')
	y_img = Image.open(y_img_path).convert('L')

	return x_img, y_img
