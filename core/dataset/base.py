import argparse
import torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    """

    __abstract__ = True

    @staticmethod
	def modify_cli_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
		Override this method.

		Parameters:
		------------------
		.. parser (ArgumentParser): original option parser
		.. is_train (bool): whether training phase or test phase

		Returns:
		------------------
		.. parser: the modified parser.
		"""
        return parser

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
