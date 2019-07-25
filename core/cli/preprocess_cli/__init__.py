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
    search_files_recursively,
    get_basename,
    clean_filename
)
# Approximated Rank Pooling
from core.cli.preprocess_cli.appx_rank_pooling import (
    cvApproxRankPooling, Buffer
)
video_extensions = ['.avi', '.mp4', '.webm', '.mov', '.mkv']


"""
APPROXIMATED RANK POOLING

Apply approximated rank pooling algorithm to convert video frames
into a video representation
"""

def run_appx_rank_pooling(
    video_path,
    outdir,
    buffer_size=25,
    img_name_tmpl='arp_rgb_{:05d}.png'
):
    """Run ARP function.
    """

    safe_mkdir(outdir)  # create directory for each video data

    current = current_process()
    cap = cv2.VideoCapture(video_path)
    buffer = Buffer(buffer_size)
    success = True

    while success:
        success, frame = cap.read()

        if buffer.isfull():
            frames = buffer.clear()
            rank_pooled = cvApproxRankPooling(frames)
            
            image_name = img_name_tmpl.format(buffer.batch_count)
            outpath = os.path.join(outdir, image_name)
            
            print(".. Writing to %s" % outpath)
            cv2.imwrite(outpath, rank_pooled)

        buffer.enqueue(frame)
    cap.release()


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
    '--img_name_tmpl',
    default = 'img_{:05d}.png',
    type = click.STRING
)
@click.option(
    '-j',
    '--n_jobs',
    default = 8,
    type = int
)
def appx_rank_pooling(
    source,
    dest,
    img_name_tmpl
    n_jobs
):
    """
    Usage: 
        > preprocess_cli appx_rank_pooling {YOUR_VIDEO_DIR} \ 
            {YOUR_SAVE_FOLDER} [--OPTIONS] 
    """
    print("Executing appx_rank_pool...")
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

        print(". Current class folder: %s, total:%d" %(class_folder, len(video_files)))

        run_args = list(zip(video_files, outpaths))
        Pool(n_jobs).starmap(
            run_appx_rank_pooling, run_args
        )

        print(". Finished %s." % class_folder)

