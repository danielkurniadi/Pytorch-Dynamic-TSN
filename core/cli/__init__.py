import os

import click
from click import echo

from core.cli.dataset_cli import (
    api_download,
    url_download
)
from core.cli.preprocess_cli import (
    appx_rank_pooling
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def dataset_cli():
    echo('Command for dataset')

dataset_cli.add_command(api_download)
dataset_cli.add_command(url_download)


@click.group(context_settings=CONTEXT_SETTINGS)
def preprocess_cli():
    echo('Command for preprocessing')

preprocess_cli.add_command(appx_rank_pooling)

