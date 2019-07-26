import os
from PIL import Image

from torchvision import transforms

# Dataset utilities
from core.dataset.utils import (
	load_rgb_image,
	load_flow_image
)
from core.utils.file_system import(
	search_files_recursively,
)
from core.dataset.utils import (
	read_strip_split_lines,
	expand_split_folders_to_filepaths,
	load_rgb_image,
	load_flow_image
)
from core.dataset.base_dataset import BaseDataset


###################################################################################################

class FrameDataset(BaseDataset):
	"""Frame dataset can be treated as image with deterministic input channel length e.g. RGB has 3 channels
	Frame itself can be an image or preprocessed image representation of video. Frame has 3-dimensions 
	
	The metadata for frame dataset, also called split file, contains two columns:
		.. folder path and labels
	"""

	def __init__(self, opts, phase='train'):
		super(FrameDataset, self).__init__(opts, phase)

		# hackaround for converting metadata II to compatible metadata
		if opts.metadata_type in ['II', 'ii', 2]:
			self.metadata = expand_split_folders_to_filepaths(self.metadata)

		# configure image file naming
		self.image_extension = opts.img_ext
		self.img_name_tmpl = 'img_{:05d}_' + self.image_extension

		# configure image property
		self.input_channels = opts.input_channels
		self.input_size = opts.input_size
		self.input_means = opts.input_means
		self.input_std = opts.input_std

		# configure transforms; TODO: seperate transforms config logic
		self.crop_size = min(opts.crop_size, self.input_size)
		self.transforms = transforms.Compose([
			transforms.Resize(self.input_size),
			transforms.CenterCrop(self.crop_size),
			transforms.ToTensor(),
			transforms.Normalize(self.input_means, self.input_std)
		])

	@staticmethod
	def modify_cli_options(parser, is_train):
		parser.add_argument('--img_name_tmpl', type=str, default='img_{:05d}.png',
			help='Image name template with (python curly braces format) for each frame in one video folder')
		parser.add_argument('--random_frame_shift', action='store_true',
			help='Whether to sample video frames at random shift or at the middle of each segments')

		return parser

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, index):
		img_path, label = self.metadata[index]
		img = load_rgb_image(img_path)  #TODO: what if not RGB?
		img = self.transforms(img)

		return img, label
