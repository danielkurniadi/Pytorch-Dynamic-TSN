import os
import glob
from pathlib import Path

import click
from click import echo
from multiprocessing import Pool, current_process

import cv2
import numpy as np

# File utilities
from core.utils.file_system import (
    clean_filename,
    get_basename,
    search_files_recursively
)

__all__ = ['format_index_of_filenames',]


"""
FORMAT DIGIT of FILENAMES

Apply index (digit) formatting to a filename for each file in dataset folder.
Using 5-digit format to re-index file in folder. This is useful for filename consistency

We assume the folder structure of type II:
    |- dataset_dir/
        |- class_folder_A
            |- folder_of_video0001
            |- folder_of_video0002
                |- frame0001.png  # indexed dataset
                |- frame0002.png
                |- frame0003.png
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
def format_index_of_filenames(
    dataset_dir
):
    """
    Usage:
        > preprocess_cli format-index-of-filenames {YOUR_VIDEO_DIR}
    """
    def split_digit_and_char_string(s):
        numeric = ''.join(filter(str.isdigit, s))
        return (
            int(numeric),
            s.replace(numeric, '')
        )

    formatted_filename_tmpl = '{}{:05d}' # assuming all digits placed after filename

    file_paths = search_files_recursively(dataset_dir)
    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        dirname = os.path.dirname(file_path)

        # assuming digit is used only for indexing the file
        index, basename = split_digit_and_char_string(get_basename(file_path))
        formatted_filename = formatted_filename_tmpl.format(basename, index) + ext

        os.rename(
            file_path,
            os.path.join(dirname, formatted_filename)
        )

