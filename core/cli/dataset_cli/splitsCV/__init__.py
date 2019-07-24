import os
from pathlib import Path

import click
from click import echo
from multiprocessing import Pool, current_process

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# File utilities
from core.utils.file_system import (
    search_files_recursively,
    get_basename,
    clean_filename
)

# Split util
from .utils import generate_skf_split_files

__all__ = ['stratified_shuffle_split_folderset']


"""
FILES-DATA SPLITTING

Apply stratified and shuffled splitting to dataset. File-data splitting assumes each datapoint
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

def stratified_shuffle_split_fileset(dataset_dir):
    """File-data splitting
    """
    pass


"""
FOLDER-DATA SPLITTING

Apply stratified and shuffled splitting to dataset. Folder-data splitting assumes each datapoint
is represented by a folder. Each folder is seperated to class folder inside dataset `directory`.

Perfect use of this splitting is for video-frames dataset in action recognition classification where each folder represent one
video dataset. Inside the folder, are images/frames that has been preprocessed. 

It's a common practice to use img frames rather than video input to feed to neural net.

Example folder structure:
    |- dataset_dir/
        |- class_A
            |- folder_of_video0001
            |- folder_of_video0002
                |- frame0001.png
                |- frame0002.png
                |- frame0003.png
            ...
        |- class_B
            |- folder_of_video0004
            |- folder_of_video0005
            ...
"""
@click.command()
@click.argument(
    'dataset_dir',
    envvar = 'DATA_DIR',
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
    '--out_prefix',
    default = ''
    type = click.STRING
)
def stratified_shuffle_split_folderset(
    dataset_dir,
    split_dir,
    n_splits,
    out_prefix,
):
    """Folder-data splitting
    """
    listpaths, listlabels = [], []
    class_folders = filter(
        os.path.isdir,
        glob.glob(dataset_dir + '/*')
    )

    for label, class_folder in enumerate(class_folders):    # label encode using for-loop index
        data_paths = glob.glob(class_folder + '/*')
        data_labels = [label] * len(data_paths)
        
        listpaths.extend(data_paths)
        listlabels.extend(data_labels) 

    generate_skf_split_files(
        listpaths,
        listlabels,
        split_dir,
        include_test_split = True,
        split_type = 'folder',
        out_prefix = out_prefix,
        n_splits = n_splits
    )
