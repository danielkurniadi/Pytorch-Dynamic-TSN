import os
import time
import shutil
from core.options.train_options import (
    BaseOptions,
    TrainOptions
)

import argparse

def test_options_initialise_default():
    print("-----------------------------------------------------")
    print("Test: test_base_options_default")
    parser1 = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ) 

    print("- Initializing...")
    base_options = BaseOptions()
    parser1 = base_options.initialize()
    
    opts, _ = parser1.parse_known_args()
    print("------------------Runtime Configs -------------------")
    print(".. --name %s" %opts.name)
    print(".. --gpu_ids %s" %opts.gpu_ids)
    print(".. --serial_batches %s" %opts.serial_batches)
    print(".. --num_threads %s" %opts.serial_batches)
    print(".. --checkpoints_dir %s" %opts.checkpoint_dir)
    
    print("------------------- Model Configs -------------------")
    print(".. --model %s" %opts.model)
    print(".. --n_classes %s" %opts.n_classes)
    print(".. --input_channels %s" %opts.input_channels)
    print(".. --norm %s" %opts.norm)
    print(".. --init_type %s" %opts.init_type)
    print(".. --init_gain %s" %opts.init_gain)

    print("------------------- Input Configs -------------------")
    print(".. --dataset_mode %s" %opts.dataset_mode)
    print(".. --batch_size %s" %opts.batch_size)
    print(".. --input_size %s" %opts.input_size)
    print(".. --input_range %s" %opts.input_means)
    print(".. --input_std %s" %opts.input_std)
    print(".. --preprocess %s" %opts.preprocess)
    print(".. --crop_size %s" %opts.crop_size)
    print(".. --no_flip %s" %opts.no_flip)

    print("------------------ Additional Params ----------------")
    print(".. --load_epoch %s" %opts.load_epoch)
    print(".. --verbose %s" %opts.verbose)
    print(".. --suffix %s" %opts.suffix)



if __name__ == '__main__':
    test_options_initialise_default