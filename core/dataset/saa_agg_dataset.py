import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['AggressionDataset']


def clean(path_label_pairs):
    cleaned = []
    for line in path_label_pairs:
        if len(line.strip().split()) != 2:
            continue
        cleaned.append(line)
    return cleaned


class AggressionDataset(Dataset):

    def __init__(self, opts):
        """Dataset wrapper for Approximated Rank Pool images.
        
        Arguments:
            - filepath (str): absolute path of txt file which contains paths_labels pairs
            - preprocessing (arr of fn): preprocessing functions in array 
            - input_mean (arr): input mean in array-like/tuples, must has same length as the input channels(e.g RGB: 3 channels)
            - input_std (arr): input standard dev in array-like/tuples, must has same length as the input channels (e.g RGB: 3 channels) 
        """
        self.filepath = opts.filepath

        if not os.path.exists(os.path.abspath(self.filepath)):
            raise FileNotFoundError("path-label-pairs file (.txt) not found, %s" % filepath)

        # read files and get file paths (dataset)
        with open(self.filepath, 'r') as fp:
            self.path_label_pairs = fp.readlines()
        
        self.path_label_pairs = clean(self.path_label_pairs)

        # setup attributes
        self.preprocessing_filters = preprocessings
        C, H, W = input_size
        self.input_size = (H, W)
        self.input_mean = input_mean
        self.input_std = input_std

        # prepare image preprocessings and transforms
        self.transforms = transforms.Compose([
                                        transforms.Resize(self.input_size),
                                        transforms.CenterCrop(self.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.input_mean, self.input_std)
                                        ])

    def __getitem__(self, i):
        assert len(self.path_label_pairs[i].strip().split()) == 2 
        path, label = self.path_label_pairs[i].strip().split()
        img = Image.open(path)
        img = self.transforms(img)
        return img, int(label)

    def __len__(self):
        return len(self.path_label_pairs)
