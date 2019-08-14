import os
import glob
from pathlib import Path

import click
from click import echo
from multiprocessing import Pool, current_process

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# File utilities
from core.utils.file_system import safe_mkdir

# Split util
from .utils import (
    generate_skf_split_files,
    get_path_label_list
)

__all__ = ['skf_split_metadata']

"""
METADATA SPLITTING

Apply stratified and shuffled splitting to dataset. Splitting assumes each datapoint
is represented by a folder. Each folder is seperated to class folder inside dataset `directory`.

Perfect use of this splitting is for video-frames dataset in action recognition classification where each folder represent one
video dataset. Inside the folder, are images/frames that has been preprocessed. 

It's a common practice to use img frames rather than video input to feed to neural net.

Example folder structure:
    |- dataset_dir/
        |- class_folder_A
            |- folder_of_video0001
            |- folder_of_video0002
                |- frame0001.png
                |- frame0002.png
                |- frame0003.png
            ...
        |- class_folder_B
            |- folder_of_video0004
            |- folder_of_video0005
            ...
"""
@click.command()
@click.argument(
    'dataset_dir',
    envvar = 'DATASET_DIR',
    type = click.Path(
        exists=True, dir_okay=True, file_okay=False
    )
)
@click.argument(
    'split_dir',
    envvar = 'SPLIT_DIR',
    type = click.Path(
        exists=False, dir_okay=True, file_okay=False
    )
)
@click.option(
    '--n_splits',
    default = 5,
    type = click.IntRange(3, 10, clamp=True)
)
@click.option(
    '--split_prefix',
    default = 'mysplit',
    type = click.STRING
)
@click.option(
    '--random_seed',
    default = 42,
    type = click.INT
)
def skf_split_metadata(
    dataset_dir,
    split_dir,
    n_splits,
    split_prefix,
    random_seed
):
    """Usage:
        > dataset_cli skf-split-metadata {YOUR_DATASET_DIR} \
            {YOUR_SPLIT_DIR} --n_splits {NUMBER_OF_SPLITS}  [--OPTIONS]
    """
    safe_mkdir(split_dir)
    paths, labels = get_path_label_list(dataset_dir)

    generate_skf_split_files(
        paths,
        labels,
        split_dir,
        include_test_split = True,
        split_prefix = split_prefix,
        n_splits = n_splits,
        random_seed = random_seed
    )
