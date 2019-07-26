"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_cli_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from core.dataset.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "core.dataset." + dataset_name.lower() + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            break

    if dataset is None:
        raise NotImplementedError("Cannot find Dataset with "
            "class name that matches %s in lowercase." % (target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_cli_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_cli_options


def create_dataset_loader(opts, phase):
    """Create a dataset given the option.

    This function wraps the class DatasetLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset_loader
        >>> dataset = create_dataset_loader(opt)
    """
    
    data_loader = DatasetLoader(opts, phase)
    dataset = data_loader.load_data()
    return dataset


class DatasetLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opts, phase='train'):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opts = opts
        dataset_class = find_dataset_using_name(opts.dataset_mode)
        self.dataset = dataset_class(opts, phase)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opts.batch_size,
            shuffle=(not opts.serial_batches),
            num_workers=int(opts.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for data in (self.dataloader):
            yield data
