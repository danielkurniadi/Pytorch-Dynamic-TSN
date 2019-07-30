import os
import click
from click import echo

# Computation libs
import numpy as np
import cv2


class Buffer():
	def __init__(self, size):
		self.size = max(int(size), 1)
		self.container = []

	def enqueue(self, item):
		if len(self.container) < self.size:
			self.container.append(item)
		else:
			print('Buffer full')

	def dequeue(self):
		if not self.isempty():
			self.container.pop(0)
		else:
			print("Buffer empty")

	def clear(self):
		container = self.container
		self.container = []
		return np.array(container)

	def get(self):
		return np.array(self.container)

	def isempty(self):
		return len(self.container) == 0

	def isfull(self):
		return (len(self.container) == self.size)


def cvApproxRankPooling(imgs):
	T = len(imgs)
  
	harmonics = []
	harmonic = 0
	for t in range(0, T+1):
		harmonics.append(harmonic)
		harmonic += float(1)/(t+1)

	weights = []
	for t in range(1 ,T+1):
		weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
		weights.append(weight)
		
	feature_vectors = []
	for i in range(len(weights)):
		feature_vectors.append(imgs[i] * weights[i])

	feature_vectors = np.array(feature_vectors)

	rank_pooled = np.sum(feature_vectors, axis=0)
	rank_pooled = cv2.normalize(rank_pooled, None, alpha=0, beta=255, 
		norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	return rank_pooled


def run_video_appx_rank_pooling(
	video_path,
	outdir,
	img_ext='.jpg',
	buffer_size=10,
):
	"""Approximated Rank Pooling (ARP) runner for video input

	Outputs Rank pooled frames and RGB frames from a video.
	Number of rank pooled and RGB frames are kept equal. 
	"""
	arp_name_tmpl = 'arp_{:05d}' + img_ext
	rgb_name_tmpl = 'img_{:05d}' + img_ext
	safe_mkdir(outdir)  # create directory for each video data

	current = current_process()
	cap = cv2.VideoCapture(video_path)
	buffer = Buffer(buffer_size)
	success = True
	
	count = 0
	while success:
		success, frame = cap.read()

		if buffer.isfull():
			frames = buffer.get()
			buffer.dequeue()

			rgb_image = frames[buffer_size//2]  # take the middle frame
			rank_pooled = cvApproxRankPooling(frames)

			arp_name = arp_name_tmpl.format(count)
			rgb_name = rgb_name_tmpl.format(count)
			
			arp_outpath = os.path.join(outdir, arp_name)
			rgb_outpath = os.path.join(outdir, rgb_name)

			print(".. Writing to %s" % outpath)

			cv2.imwrite(rgb_outpath, rgb_image)
			cv2.imwrite(arp_outpath, rank_pooled)

		buffer.enqueue(frame)
		count += 1

	cap.release()


def run_img_appx_rank_pooling(
	directory,
	outpath,
	img_ext='.jpg',
):
	""" Approximated rank pooling for images input from a folder
	
	Outputs Rank pooled image from images input.
	Number of output is a rank pooled image.
	"""
	img_paths = search_files_recursively(
		directory,
		prefix = 'img_',
		by_extensions=['.png', '.jpg']
	)
	imgs = [
		cv.imread(img_path)
		for img_path in img_paths
	]
	outpath = os.path.join(
		dest,
		get_basename(class_folder),
		get_basename(subdir),
		'arp_00000' + img_ext
	)

	rank_pooled = cvApproxRankPooling(imgs)

	print(".. Writing to %s" % outpath)
	cv2.imwrite(rank_pooled, outpath)
