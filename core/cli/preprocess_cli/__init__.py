import os
from pathlib import Path

import click
from click import echo

# Parallel and Subprocess library
import subprocess
import multiprocessing
from multiprocessing import Pool, current_process
from multiprocessing.pool import ThreadPool

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
    run_video_appx_rank_pooling
)

# Dense Optical Flow
from core.cli.preprocess_cli.dense_flow import run_video_dense_flow 

video_extensions = ['.avi', '.mp4', '.webm', '.mov', '.mkv']


"""
Video APPROXIMATED RANK POOLING

Apply approximated rank pooling algorithm to convert video
into rank pooled frames representing the action in video

Assuming source folder structure type I, output to a folder structure
that suits for metadata type II.
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
        > preprocess_cli video-appxrankpooling {YOUR_VIDEO_DIR} \ 
            {YOUR_SAVE_FOLDER} [--OPTIONS]
    """
    print(". Executing appx_rank_pool on video...")
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
        outdir = [
            os.path.join(outfolder, clean_filename(get_basename(video_file)))
            for video_file in video_files
        ]
        img_exts = [img_ext]* len(outdir)  # TODO: optimise this extension duplicating given every element is constant

        print(". Current class folder: %s, total:%d" %(class_folder, len(video_files)))

        run_args = list(zip(video_files, outdir, img_exts))
        results = Pool(n_jobs).starmap(
            run_video_appx_rank_pooling, run_args
        )

        print(". Finished %s." % class_folder)


"""
Video DENSE OPTICAL FLOW

Apply dense optical flow algorithm to convert video
into optical flow frames representing x and y motion flow in the video

Output frames is of .jpg format
"""

@click.command()
@click.argument(
    'source',
    envvar = 'SRC',
    type = click.Path(exists=True, dir_okay= True)
)
@click.argument(
    'dest',
    envvar = 'SAVE_FOLDER',
    type = click.Path(exists=False, dir_okay=True)
)
@click.option(
    '-j',
    '--n_jobs',
    default = 40,
    type = int
)
def video_denseOpticalFlow(
    source,
    dest,
    n_jobs
):
    """
    Usage:
        > preprocess_cli video-denseOpticalFlow {YOUR_VIDEO_DIR} \
            {YOUR_SAVE_FOLDER} [--OPTIONS]
    """
    print(". Executing dense_optical_flow on video...")
    safe_mkdir(dest)

    for class_folder in abs_listdir(source):
        video_files = search_files_recursively(
            class_folder,
            by_extensions = video_extensions
        )
        out_class_folder = os.path.join(dest, get_basename(class_folder))
        
        safe_mkdir(out_class_folder)

        # take only the base name of each video url
        outdir = [
            os.path.join(out_class_folder, clean_filename(get_basename(video_file)))
            for video_file in video_files
        ]
        
        print(". Current class folder: %s, total: %d" %(class_folder, len(video_files)))
        
        run_args = list(zip(video_files, outdir))
        pool = ThreadPool(min(n_jobs, multiprocessing.cpu_count()))

        results = []
        for run_arg in run_args:
            result = pool.apply_async(
                run_video_dense_flow, run_arg
            )
            results.append(result)

        # Close the pool and wait for each task to complete
        pool.close()
        pool.join()

        for result in results:
            out, err = result.get()
            # logging result
            with open("logs/denseflow.txt", "a+") as f:
                f.write("out %s, err %s\n" % (out, err))
