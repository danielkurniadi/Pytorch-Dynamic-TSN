import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision
from torchvision.transforms import functional as F

from core.dataset.functionals.transforms.single_transforms import (
	__crop,
	__flip,
	__scale
)


def __group_crop(imgs, pos, size):
	out_imgs = []
	x1, y1 = pos
	tw = th = size

	for img in imgs:
		out_img = __crop(img, pos, size)
		out_imgs.append(out_img)

	return out_imgs


def __group_random_crop(imgs, size):
	out_imgs = []
	ow, oh = imgs[0].size
	tw = th = size

	for img in imgs:
		x1 = random.randint(0, ow-tw)
		y1 = random.randint(0, oh-th)
		out_img = __crop(img, (x1, y1), size)
		out_imgs.append(out_img)
	
	# print("RAND CROP ", len(imgs))
	return out_imgs


def __group_center_crop(imgs, size):
	imgs = [
		F.center_crop(img, size) for img in imgs
	]
	# print("GCROP ", len(imgs))
	return imgs


def __group_random_horizontal_flip(imgs, p=0.50, invert=False):
	out_imgs = []

	for img in imgs:
		# print("GFLIP ", img.size)
		if random.random() < p:
			img = __flip(img, invert=invert)
		out_imgs.append(img)
	# print("RANDOM FLIP ", len(out_imgs))
	return out_imgs


def __group_resize(imgs, size, interpolation=Image.BILINEAR):
	# make (size, size) to indicate square resize
	if isinstance(size, int):
		size = (size, size)
	out_imgs = [
		__scale(img, size, interpolation=interpolation) 
		for img in imgs
	]
	# print("RESIZE ", len(out_imgs))
	return out_imgs


def __group_scale(imgs, size, interpolation=Image.BILINEAR):
	return [
		__scale(img, size, interpolation=interpolation) 
		for img in imgs
	]

def __group_normalize(tensor, mean, std)
	rep_mean = mean * (tensor.size()[0]//len(mean))
	rep_std = std * (tensor.size()[0]//len(std))

	for t, m, s in zip(tensor, rep_mean, rep_std):
		t.sub_(m).div_(s)

	return tensor


def __stack(imgs):
	img_mode = imgs[0].mode
	# print("STACK size: ", imgs[0].size, len(imgs), img_mode)
	if img_mode == 'L':
		# PIL img: (HxW) -> (HxWxC) where C=1
		imgs = [np.expand_dims(img, axis=2) for img in imgs]
		imgs = np.concatenate(imgs, axis=2)
		# print("STACK result: ", imgs.shape)
		return imgs
	
	else:
		# PIL img: (HxWxC)
		imgs = np.concatenate(imgs, axis=2)
		# print("STACK result: ", imgs.shape)
		return imgs

