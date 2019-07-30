import os

import click
from click import echo

from core.cli.dataset_cli import (
    # Download dataset
    api_download,
    url_download,
    # Split dataset
    skf_split_metadataI,
    skf_split_metadataII
)
from core.cli.preprocess_cli import (
    # Approx rank pool preprocessing
    video_appxRankPooling,
    imgs_appxRankPooling
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def dataset_cli():
    echo('Command for dataset')

dataset_cli.add_command(api_download)
dataset_cli.add_command(url_download)
dataset_cli.add_command(skf_split_metadataII)
dataset_cli.add_command(skf_split_metadataI)


@click.group(context_settings=CONTEXT_SETTINGS)
def preprocess_cli():
    echo('Command for preprocessing')

preprocess_cli.add_command(video_appxRankPooling)
preprocess_cli.add_command(imgs_appxRankPooling)

