import os

import click
from click import echo

from core.cli.dataset_cli import (
    # Download dataset
    api_download,
    url_download,
    # Split dataset
    stratified_shuffle_split_folderset
)
from core.cli.preprocess_cli import (
    # Approx rank pool preprocessing
    appx_rank_pooling
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def dataset_cli():
    echo('Command for dataset')

dataset_cli.add_command(api_download)
dataset_cli.add_command(url_download)


@click.group(context_settings=CONTEXT_SETTINGS)
def splits_cli():
    echo('Command for data splitting')

splits_cli.add_command(stratified_shuffle_split_folderset)


@click.group(context_settings=CONTEXT_SETTINGS)
def preprocess_cli():
    echo('Command for preprocessing')

preprocess_cli.add_command(appx_rank_pooling)

