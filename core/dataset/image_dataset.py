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
from core.dataset.base_dataset import SplitFileDataset


class ImageDataset(SplitFileDataset):
    """
    """
    def __init__(self, opts, split_file):
        super(ImageDataset, self).__init__(opts, split_file)

        self.metadata = expand_split_folders_to_filepaths(self.lines)
        self.transforms = transforms.Compose([
            transforms.Resize(self.opts.input_size),
            transforms.CenterCrop(self.opts.input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.opts.input_means, self.opts.input_std)
        ])

    @staticmethod
    def modify_cli_options(parser, is_train):
        parser = SplitFileDataset.modify_cli_options(parser, is_train)
        return parser

    def __getitem__(self, index):
        filepath, label = self.metadata[index]
        img = load_rgb_image(filepath)
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.metadata)
