import re
import sys
import argparse

from core import models
from core import dataset


class BaseOptions(object):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_cli_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    #-----------------------
    # Interface
    #-----------------------

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device.

        Returns:
            opts: Option objects representing gathered options
        """
        opts = self.gather_options()
        opts.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opts.suffix:
            opts.name = opts.name + '_' + opts.suffix

        self.print_options(opts)

        # set gpu ids
        # str_ids = opts.gpu_ids.split(',')
        # opts.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opts.gpu_ids.append(id)
        # if len(opts.gpu_ids) > 0:
        #     torch.cuda.set_device(opts.gpu_ids[0])

        self.opts = opts
        return self.opts

    #-----------------------
    # Setups
    #-----------------------

    def initialize(self, parser):
        """ Define all options by adding arguments to argparser
        """
        # ========================= Runtime Configs ==========================
        parser.add_argument('--name', type=str, default='', 
            help='Descriptive name of ongoing experiment. Used for name prefix when storing artifacts')
        parser.add_argument('--gpu_ids', type=str, default='0',
            help='GPU ids: e.g. 0,1,2; use -1 for CPU')
        parser.add_argument('--serial_batches', action='store_true',
            help='If true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int,
            help='# Threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
            help='Models are saved here')

        # ========================= Model Configs ==========================
        parser.add_argument('--model', type=str, default='tsn',
            help='Chooses which model to use. [tsn | resnext101]')
        parser.add_argument('--pretrained', type=str, default='imagenet',
            help='Chooses pretrained weights')
        parser.add_argument('--n_classes', type=int, default=10,
            help='# of output target classes')
        parser.add_argument('--input_channels', type=int, default=3,
            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--norm', type=str, default='instance',
            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
            help='Network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
            help='Scaling factor for normal, xavier and orthogonal.')

        # ========================= Data Configs ==========================
        parser.add_argument('--split_dir', type=str, required=True,
            help='Path to split directory where split files are')
        parser.add_argument('--split_idx', type=int, required=True,
            help='Split index, also known as "k" in KFold technique')
        parser.add_argument('--dataset_mode', type=str, default='Frame',
            help='Chooses how datasets are loaded. [Frame | Temporal]')
        parser.add_argument('--img_ext', type=str, default='.png',
            help='File extension of images. [.png | .jpeg | .jpg]')

        # ========================= Input Configs ==========================
        parser.add_argument('-b', '--batch_size', type=int, default=32,
            help='Input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--input_size', type=int, default=286,
            help='Scale images to this size')
        parser.add_argument('--input_means', action='append', nargs='+', type=float, default=[0.5, 0.5, 0.5],
            help='Input images means')
        parser.add_argument('--input_range', action='append', nargs='+', type=float, default=[0, 1.0],
            help='Input images range')
        parser.add_argument('--input_std', action='append', nargs='+', type=float, default=[0.25, 0.25, 0.25],
            help='Input images standard deviation')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
            help='Scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--crop_size', type=int, default=256,
            help='Then crop to this size')
        parser.add_argument('--no_flip', action='store_true',
            help='If specified, do not flip the images for data augmentation')
        
        # additional parameters
        parser.add_argument('--load_epoch', type=int, default='0',
            help='which epochs to load? if load_iter > 0, the code will load models by epoch_[load_epoch];')
        parser.add_argument('--verbose', action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
            help='customized suffix: opt.name = opt.name + suffix')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """ Initialize our parser with options (once). 
        Add additional model-specific and dataset-specific options by super this method.
        These options are defined in <model_options> function in model and dataset classes.

        Returns:
            opts: Option objects representing gathered options
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        if not self.initialized:
            parser = self.initialize(parser)
        
        # basic options
        opts, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opts.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, is_train=self.isTrain)
        opts, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opts.dataset_mode
        dataset_option_setter = dataset.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, is_train=self.isTrain)
        opts, _ = parser.parse_known_args()  # parse again with new defaults

        self.parser = parser
        self.opts = opts

        return opts

    #-----------------------
    # Utils
    #-----------------------

    def print_options(self, opts):
        """Print and save options
        It will print both current options and default values(if different).
        """
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opts).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '.. {:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

        # save options setting to logging dir
        if hasattr(opts, 'logging_dir'):
            if not os.path.isdir(opts.logging_dir):
                raise FileNotFoundError("Logging Directory %s is not found" % opts.logging_dir)
            
            opts_file = "{}_opts_cache.txt".format(opts.name)
            opts_save_path = os.path.join(opts.logging_dir, opts_file)
            with open(opts_save_path, 'w') as opts_file:
                opts_file.write(message)

