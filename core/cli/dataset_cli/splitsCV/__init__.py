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

__all__ = ['skf_split_metadataII', 'skf_split_metadataI']


"""
METADATA type I SPLITTING

Apply stratified and shuffled splitting to dataset. Type I splitting assumes each datapoint
is represented by a file. Each file is seperated to class folder inside dataset `directory`.

Perfect use of this splitting is for image dataset in image classification. where each image file (.png, .jpg, etc) represent one
datapoint.

Example folder structure:
    |- dataset_dir/
        |- class_A
            |- img0001.png
            |- img0002.png
            |- img0003.png
            ...
        |- class_B
            |- img0001.png
            |- img0002.png
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
    default = '',
    type = click.STRING
)
def skf_split_metadataI(
    dataset_dir,
    split_dir,
    n_splits,
    split_prefix,
):
    """ Stratified KFold splitting for type I metadata
    """
    safe_mkdir(split_dir)
    paths, labels = get_path_label_list(dataset_dir)

    generate_skf_split_files(
        paths,
        labels,
        split_dir,
        include_test_split = True,
        split_type = 'I',
        split_prefix = split_prefix,
        n_splits = n_splits
    )


"""
METADATA type II SPLITTING

Apply stratified and shuffled splitting to dataset. Type II splitting assumes each datapoint
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
    '--data_prefix',
    default = '',
    type = click.STRING
)
@click.option(
    '--split_prefix',
    default = '',
    type = click.STRING
)
def skf_split_metadataII(
    dataset_dir,
    split_dir,
    n_splits,
    data_prefix,
    split_prefix,
):
    """Stratified KFold splitting for type II metadata
    """
    safe_mkdir(split_dir)
    paths, labels = get_path_label_list(dataset_dir)

    generate_skf_split_files(
        paths,
        labels,
        split_dir,
        include_test_split = True,
        split_type = 'II',
        split_prefix = split_prefix,
        n_splits = n_splits,
        data_prefix = data_prefix
    )
