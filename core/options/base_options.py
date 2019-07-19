import re
import sys
import argparse

# from core.config import (dataconf, logconf, checkpointconf)


class BaseOptions(object):
    """ Centralised representation of options
    """

    def __init__(self, model, dataset, logger, name=None):
        self.name = self.__class__.__name__

        if name:
            self.name = name

        self.initialized = False
        self.model = model
        self.dataset = dataset
        self.logger = logger

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
        str_ids = opts.gpu_ids.split(',')
        opts.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opts.gpu_ids.append(id)
        if len(opts.gpu_ids) > 0:
            torch.cuda.set_device(opts.gpu_ids[0])

        self.opts = opts
        return self.opts

    #-----------------------
    # Setups
    #-----------------------

    def initialize(self, parser):
        """ Define all options by adding arguments to argparser
        """
        # basic parameters
        parser.add_argument('--name', type=str, default='<EXPERIMENT_NAME>', 
            help='Descriptive name of running experiment. Used for name prefix when storing artifacts')
        parser.add_argument('--gpu_ids', type=str, default='0',
            help='GPU ids: e.g. 0,1,2; use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
            help='models are saved here')
        # default model parameters
        parser.add_argument('--n_classes', type=int, default=10,
            help='# of output target classes'
        )
        parser.add_argument('--input_channels', type=int, default=3,
            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--norm', type=str, default='instance',
            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        # default dataset parameters
        parser.add_argument('--modality', type=str, default='Flow',
            help='Modality of dataset [RGB | RGBDiff| Flow | RankPool | OpenPose]')
        parser.add_argument('--num_threads', default=4, type=int,
            help='# Threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32,
            help='Input batch size')
        parser.add_argument('--input_size', type=int, default=286,
            help='Scale images to this size')
        parser.add_argument('--input_means', nargs='+', type=float, default=[0.5, 0.5, 0.5],
            help='Input images means')
        parser.add_argument('--input_range', nargs='+', type=float, default=[0, 1.0],
            help='Input images range')
        parser.add_argument('--input_std', nargs='+', type=float, default=[0.25, 0.25, 0.25],
            help='Input images standard deviation')
        parser.add_argument('--crop_size', type=int, default=256,
            help='Then crop to this size')
        parser.add_argument('--no_flip', action='store_true',
            help='If specified, do not flip the images for data augmentation')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='If specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')

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
        print(vars(opts))

        # modify model-related parser options
        parser = self.models.modify_cli_options(parser, is_train=True)
        opts, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        parser = self.dataset.modify_cli_options(parser, is_train=True)

        # modify logging-related parser options
        parser = self.logger.modify_cli_options(parser, is_train=True)

        return opts

    #-----------------------
    # Utils
    #-----------------------

    def print_options(self, opts):
        """Print and save options
        It will print both current options and default values(if different).
        """
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opts).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '.. {:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

        # save options setting to logging dir
        if hasattr(opts, logging_dir):
            if not os.path.isdir(opts.logging_dir):
                raise FileNotFoundError("Logging Directory %s is not found" % opts.logging_dir)
            
            opts_file = "{}_opts.txt".format(opts.name)
            opts_save_path = os.path.join(opts.logging_dir, opts_file)
            with open(opts_save_path, 'w') as opts_file:
                opts_file.write(message)

