"""
Checking the arguments of PARROT's command-line tools, and printing what a
command is about to do.

argparse checks the type of each argument and, where an option has a fixed set
of values, that the value is one of them (`choices=`). The checks here cover
what argparse cannot: that files and directories exist and that numbers are in
range.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math
import os
from collections.abc import Sequence
from typing import Any

from parrot.package_version import PARROT_VERSION

# Setting names are padded to this width when settings are printed, so the values line up
SETTING_NAME_WIDTH = 14

# Separator line printed under headings
RULE = '...................................'


# .................................................................
# Printing

def print_startup(program_name: str, silent: bool) -> None:
    """Print the name and version of the command being run (unless `silent`)

    Parameters
    ----------
    program_name : str
        Name of the command, e.g. 'parrot-predict'
    silent : bool
        If True, print nothing
    """
    if silent:
        return
    print(f'\n{program_name} (version {PARROT_VERSION})')
    print(RULE)
    print('Developed by Dan Griffith\n   Holehouse lab, WUSTL\n\n')


def print_settings(silent: bool, **settings: Any) -> None:
    """Print each setting as 'name : value' on its own line (unless `silent`)

    Parameters
    ----------
    silent : bool
        If True, print nothing
    **settings
        The settings to print, in order
    """
    if silent:
        return
    print('Settings:')
    print(RULE)
    for name, value in settings.items():
        print(f'{name:<{SETTING_NAME_WIDTH}}: {value}')


# .................................................................
# Files and directories

def check_file_exists(path: str, description: str) -> str:
    """Check that a file exists, returning its absolute path

    Parameters
    ----------
    path : str
        Path to the file
    description : str
        What the file is, for the error message (e.g. 'Datafile')

    Returns
    -------
    str
        The absolute path

    Raises
    ------
    FileNotFoundError
        If there is no such file
    """
    absolute_path = os.path.abspath(path)
    if not os.path.isfile(absolute_path):
        raise FileNotFoundError(f'{description} does not exist.')
    return absolute_path


def output_prefix_and_directory(path: str) -> tuple[str, str]:
    """Split the path of an output file into the prefix other output files are named with, and its directory

    Only the final extension is removed, so 'net.v2.pt' gives the prefix
    'net.v2' (output files are then named e.g. 'net.v2_predictions.tsv').

    Parameters
    ----------
    path : str
        Absolute path to a file (which need not exist yet)

    Returns
    -------
    str
        The path with its final extension removed, used as an output prefix
    str
        The directory containing the file, with a trailing path separator

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    """
    directory, filename = os.path.split(path)
    directory = os.path.join(directory, '')
    if not os.path.exists(directory):
        raise FileNotFoundError(f'{directory} does not exist.')
    prefix = os.path.join(directory, os.path.splitext(filename)[0])
    return prefix, directory


# .................................................................
# Numbers

def check_positive(value: int, description: str) -> None:
    """Check that a count (e.g. a number of epochs) is at least 1

    Parameters
    ----------
    value : int
    description : str
        What the value is, for the error message

    Raises
    ------
    ValueError
        If the value is less than 1
    """
    if value < 1:
        raise ValueError(f'{description} must be at least 1, got {value}.')


def check_fraction(value: float, description: str) -> None:
    """Check that a value is strictly between 0 and 1

    Parameters
    ----------
    value : float
    description : str
        What the value is, for the error message

    Raises
    ------
    ValueError
        If the value is 0 or less, or 1 or more
    """
    if not 0 < value < 1:
        raise ValueError(f'{description} must be between 0 and 1, got {value}.')


def check_set_fractions(fractions: Sequence[float]) -> None:
    """Check the fractions of the data in the training, validation and test sets

    Parameters
    ----------
    fractions : sequence of 3 floats
        Training, validation and test fractions

    Raises
    ------
    ValueError
        If any fraction is not between 0 and 1, or they do not sum to 1
    """
    for fraction in fractions:
        check_fraction(fraction, 'Set fractions')
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError(f'Set fractions must sum to 1, got {" + ".join(str(fraction) for fraction in fractions)}.')
