from collections import namedtuple

import os
import numpy as np
from PIL import Image

from core.utils.file_system import abs_listdir
from core.dataset.functionals import (
	generate_sample_seg_indices,
	generate_median_seg_indices,
	load_rgb_image,
	load_flow_image,
)


#-----------------------
# Video Metadata
#-----------------------

"""
Base structure that contains metadata for video-frames input
"""

BaseVideoMetadata = namedtuple('VideoMetadata', 
	field_names = [
		'directory',	# directory of video frames (single dataset)
		'label',		# class label
		'modality'
	]
)


class VideoMetadata(BaseVideoMetadata):
	"""Structure that contains metadata for video-frames input
	"""
	def __new__(cls, directory, label, modality):
		self = super(VideoMetadata, cls).__new__(cls, directory, label, modality)
		
		# number of frames in directory having same modality
		self.n_frames = len([
			path for path in abs_listdir(directory)
			if path.startswith(modality)
		])

		return self


#-----------------------
# Indices handlers
#-----------------------

def generate_video_indices(
	n_segments,
	n_frames,
	sample_length,
	random_shift = True
):
	""" Generate indices of video frames by random sampling/median method

	1. First it generates and seg_indices, chosen indices for segment in video.
	2. Then it extend each index to the desired sample length. 
	For example: if seg_indices = [0, 4, 9]; sample_length = 3
		indices = [0, 0, 0, 4, 4, 4, 9, 9, 9] + [0, 1, 2, 0, 1, 2, 0, 1, 2]
				= [0, 1, 2, 4, 5, 6, 9, 10, 11]
	"""
	if random_shift:
		seg_indices = generate_sample_seg_indices(n_segments,
			n_frames, sample_length)
	else:
		seg_indices = generate_median_seg_indices(n_segments,
			n_frames, sample_length)

	repeats = np.repeat(np.arange(sample_length), n_segments)
	full_indices = np.repeat(seg_indices, sample_length) + repeats

	return full_indices


#-----------------------
# Frame Loading
#-----------------------

def load_video_frames(
	indices,
	directory,
	img_name_tmpl,
	modality
):
	"""
	Generator function that yields frames from a video given the frame indices.
	The indices are used to sample only some portion of the video frames
	"""
	_max_iter = len(indices)

	for idx in indices:
		if modality in ['RGB', 'RGBDiff', 'ARP']:
			filepath = os.path.join(directory, img_name_tmpl.format(idx))
			yield load_rgb_image(filepath)

		elif modality == 'Flow':
			x_img_path = os.path.join(directory, img_name_tmpl.format('x', idx))
			y_img_path = os.path.join(directory, img_name_tmpl.format('y', idx))
			yield load_flow_image(x_img_path, y_img_path)
