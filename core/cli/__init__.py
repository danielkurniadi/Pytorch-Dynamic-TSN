import os

import click
from click import echo

from .dataset import (
    api_download,
    url_download
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def dataset():
    echo('Command for dataset')


dataset.add_command(api_download)
dataset.add_command(url_download)
