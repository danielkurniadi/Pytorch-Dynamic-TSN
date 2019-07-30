import os
from pathlib import Path

import click
from click import echo
from multiprocessing import Pool, current_process

import cv2
import numpy as np

# File utilities
from core.utils.file_system import (
    safe_mkdir,
    abs_listdir,
    search_files_recursively,
    get_basename,
    clean_filename
)
# Approximated Rank Pooling
from core.cli.preprocess_cli.appx_rank_pooling import (
    run_video_appx_rank_pooling,
    run_img_appx_rank_pooling
)
video_extensions = ['.avi', '.mp4', '.webm', '.mov', '.mkv']


"""
VIdeo APPROXIMATED RANK POOLING

Apply approximated rank pooling algorithm to convert video
into rank pooled frames representing the action in video
"""

@click.command()
@click.argument(
    'source',
    envvar = 'SRC',
    type = click.Path(exists=True, dir_okay=True)
)
@click.argument(
    'dest',
    envvar = 'SAVE_FOLDER',
    type = click.Path(exists=False, dir_okay=True)
)
@click.option(
    '-j',
    '--n_jobs',
    default = 8,
    type = int
)
@click.option(
    '-x',
    '--img_ext',
    default = '.jpg',
    type = str
)
def video_appxRankPooling(
    source,
    dest,
    n_jobs,
    img_ext
):
    """
    Usage: 
        > preprocess_cli video-appx-rank-pooling {YOUR_VIDEO_DIR} \ 
            {YOUR_SAVE_FOLDER} [--OPTIONS] 
    """
    print("Executing appx_rank_pool on video...")
    safe_mkdir(dest)
    
    for class_folder in os.listdir(source):     # run appx rank pool for each video in all class_folder
        video_files = search_files_recursively(
            os.path.join(source, class_folder),
            by_extensions = video_extensions
        )
        outfolder = os.path.join(dest, class_folder)

        safe_mkdir(outfolder)

        # take only the basename of each video url, clean name from dot and whitespace
        # and use this basename for output image name
        outpaths = [
            os.path.join(outfolder, clean_filename(get_basename(video)))
            for video in video_files
        ]
        img_exts = [img_ext]* len(outpaths)  # TODO: optimise this extension duplicating given every element is constant

        print(". Current class folder: %s, total:%d" %(class_folder, len(video_files)))

        run_args = list(zip(video_files, outpaths, img_exts))
        results = Pool(n_jobs).starmap(
            run_video_appx_rank_pooling, run_args
        )

        print(". Finished %s." % class_folder)


"""
images APPROXIMATED RANK POOLING

Apply approximated rank pooling algorithm to convert images/frames
into rank pooling representation.
"""

@click.command()
@click.argument(
    'source',
    envvar = 'SRC',
    type = click.Path(exists=True, dir_okay=True)
)
@click.argument(
    'dest',
    envvar = 'SAVE_FOLDER',
    type = click.Path(exists=False, dir_okay=True)
)
@click.option(
    '-j',
    '--n_jobs',
    default = 8,
    type = int
)
@click.option(
    '-x',
    '--img_ext',
    default = '.jpg',
    type = str
)
def imgs_appxRankPooling(
    source,
    dest,
    n_jobs,
    img_ext
):
    """
    Usage: 
        > preprocess_cli rgbs-appxrankpooling {YOUR_VIDEO_DIR} \ 
            {YOUR_SAVE_FOLDER} [--OPTIONS] 
    """
    print("Executing rgbs-appx-rank-pooling on video...")
    safe_mkdir(dest)
    
    source = os.path.abspath(source)
    for class_folder in abs_listdir(source):     # run appx rank pool for each video in all class_folder
        subdirs = abs_listdir(class_folder)
        outdir = os.join.path(dest, get_basename(class_folder))
        
        safe_mkdir(outdir)

        outpaths = [
            os.path.join(outdir, get_basename(subdir))
            for subdir in subdirs
        ]

        for outpath in outpaths:
            safe_mkdir(outpath)

        print(". Current class folder: %s, total:%d" %(class_folder, len(video_files)))

        run_args = list(zip(subdir, outpaths, img_exts))
        results = Pool(n_jobs).starmap(
            run_img_appx_rank_pooling, run_args
        )

        print(". Finished %s." % class_folder)
