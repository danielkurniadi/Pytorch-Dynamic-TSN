import os
from PIL import Image

from torchvision import transforms

from core.dataset.base_dataset import PathLabelPairDataset


def clean(path_label_pairs):
    cleaned = []
    for line in path_label_pairs:
        if len(line.strip().split()) != 2:
            continue
        cleaned.append(line)
    return cleaned


class AggressionDataset(PathLabelPairDataset):

    def __init__(self, opts):
        """Dataset wrapper for Approximated Rank Pool images.
        
        Arguments:
            - filepath (str): absolute path of txt file which contains paths_labels pairs
            - preprocessing (arr of fn): preprocessing functions in array 
            - input_mean (arr): input mean in array-like/tuples, must has same length as the input channels(e.g RGB: 3 channels)
            - input_std (arr): input standard dev in array-like/tuples, must has same length as the input channels (e.g RGB: 3 channels) 
        """
        super(AggressionDataset, self).__init__(opts)
        self.filepath = opts.pathlabel_pair_file
        AggressionDataset.check_path_label_pair_file(self.filepath)
        
        # read files and get file paths (dataset)
        with open(self.filepath, 'r') as fp:
            self.path_label_pairs = fp.readlines()
        
        self.path_label_pairs = clean(self.path_label_pairs)

        # prepare image preprocessings and transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self.opts.input_size),
            transforms.CenterCrop(self.opts.input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.opts.input_means, self.opts.input_std)
        ])

    def __getitem__(self, i):
        path, label = self.path_label_pairs[i].strip().split()
        img = Image.open(path)
        img = self.transforms(img)
        return img, int(label)

    def __len__(self):
        return len(self.path_label_pairs)
