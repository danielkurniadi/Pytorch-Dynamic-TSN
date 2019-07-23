import os
import numpy as np
from numpy.random import randint

from PIL import Image


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


#-----------------------
# Indices
#-----------------------

def generate_sample_seg_indices(n_segments, n_frames, new_length):
	"""
	"""
	average_duration = (n_frames - new_length + 1) // n_segments
	if average_duration > 0:
		offsets = np.arange(n_segments) * average_duration + \
			np.random.randint(average_duration, size=n_segments)
	elif n_frames > n_segments:
		offsets = np.sort(np.random.randint(n_frames - new_length + 1, size=(n_segments)))
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
		offsets = np.zeros((n_segments, ))
	return offsets + 1


#-----------------------
# Images
#-----------------------

def load_rgb_image(filepath):
    """
    """
    return Image(filepath).convert('RGB')


def load_flow_image(x_img_path, y_img_path):
    """
    """
    x_img = Image.open(x_img_path).convert('L')
    y_img = Image.open(y_img_path).convert('L')

    return x_img, y_img


#-----------------------
# Videos
#-----------------------

class VideoFrameGenerator(object):
	"""
	"""
	def __init__(self, directory, n_frames,
				label, img_name_tmpl='img_{:05d}.jpg',
				random_shift=True):
		self.directory = directory
		self.n_frames = n_frames
		self.label = label
		self.img_name_tmpl = img_name_tmpl
		self.random_shift = random_shift

	#-----------------------
	# Generator methods
	#-----------------------

	def __iter__(self):
		""" Setup for iteration.

		Returns:
			VideoRecord: self
		"""
		self._iter_count = 0
		self.vid_indices = self.generate_video_indices()
		self._max_iter = len(self.vid_indices)
		return self

	def __next__(self):
		"""
		"""
		if self._iter_count > self._max_iter:
			raise StopIteration

		videodata = self.load_video_frame(self._iter_count)
		self._iter_count += 1

		return videodata

	#-----------------------
	# Indices handlers
	#-----------------------

	def generate_video_indices(self):
		""" Generate indices of video frames from sampling/median

		eg: seg_indices = [0, 4, 9]; new_length = 3
			output  = [0, 0, 0, 4, 4, 4, 9, 9, 9] + [0, 1, 2, 0, 1, 2, 0, 1, 2]
					= [0, 1, 2, 4, 5, 6, 9, 10, 11]
		"""
		if self.random_shift:
			seg_indices = generate_sample_seg_indices(n_segments,
				n_frames, new_length)
		else:
			seg_indices = generate_median_seg_indices(n_segments,
				n_frames, new_length)

		repeats = np.repeat(np.arange(self.new_length), self.n_segments)
		full_indices = np.repeat(seg_indices, self.new_length) + repeats
		
		return full_indices

	#-----------------------
	# Frame Loading
	#-----------------------

	def load_video_frame(self, idx):
		"""
		"""
		if self.modality in ['RGB', 'RGBDiff']:
			filepath = os.path.join(self.directory, self.img_name_tmpl.format(idx))
			return load_rgb_image(filepath)
		elif self.modality == 'Flow':
			x_img_path = os.path.join(self.directory, self.img_name_tmpl.format('x', idx))
			y_img_path = os.path.join(self.directory, self.img_name_tmpl.format('y', idx))
			
			return load_flow_image(x_img_path, y_img_path)
