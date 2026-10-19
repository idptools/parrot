"""
Checking the arguments of PARROT's command-line tools, printing what a
command is about to do, and reporting mistakes in a command's input.

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

import argparse
import datetime
import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from parrot.package_version import PARROT_VERSION

# Separator line printed under headings
RULE = '...................................'

# The logo at the top of every command's header ('PARROT-2' in figlet's standard font, each character at full width).
# The second row keeps its trailing space, as a raw string cannot end in a backslash.
LOGO = (
    r' ____      _     ____   ____    ___   _____         ____',
    r'|  _ \    / \   |  _ \ |  _ \  / _ \ |_   _|       |___ \ ',
    r'| |_) |  / _ \  | |_) || |_) || | | |  | |   _____   __) |',
    r'|  __/  / ___ \ |  _ < |  _ < | |_| |  | |  |_____| / __/',
    r'|_|    /_/   \_\|_| \_\|_| \_\ \___/   |_|         |_____|',
)

DOCUMENTATION_URL = 'https://idptools-parrot.readthedocs.io'

# The option every command has for printing examples of its use
EXAMPLES_FLAG = '--show-examples'

# How the time a command started is shown in its header
HEADER_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Errors that mean something about a command's input is wrong: a file that does not exist, an invalid option or
# datafile, a package an option needs, a device that is not available. They are reported as a message, not a
# traceback. Any other error is a bug in PARROT, and keeps its traceback.
INPUT_ERRORS = (ValueError, OSError, ImportError, RuntimeError)

# Set this environment variable (to anything but an empty string) to see the full traceback of input errors as well
TRACEBACK_VARIABLE = 'PARROT_TRACEBACK'

# Exit codes: an error in the input, and a command stopped with Ctrl-C (the shell convention, 128 + SIGINT)
INPUT_ERROR_EXIT_CODE = 1
INTERRUPTED_EXIT_CODE = 130


# .................................................................
# Printing

def run_command(program_name: str, command: Callable[[], None]) -> None:
    """Run a command, reporting mistakes in its input as a one-line message rather than a traceback

    If the PARROT_TRACEBACK environment variable (TRACEBACK_VARIABLE) is set and not empty, input
    errors and Ctrl-C are raised as they are instead, with their traceback.

    Parameters
    ----------
    program_name : str
        Name of the command, e.g. 'parrot-train'
    command : function
        The command itself (it reads its own arguments)

    Raises
    ------
    SystemExit
        With INPUT_ERROR_EXIT_CODE after an input error (see INPUT_ERRORS), or
        INTERRUPTED_EXIT_CODE if the command was stopped with Ctrl-C
    """
    try:
        command()
    except INPUT_ERRORS as error:
        if os.environ.get(TRACEBACK_VARIABLE):
            raise
        print(f'{program_name}: error: {error}\n(For the full traceback, run again with {TRACEBACK_VARIABLE}=1.)',
              file=sys.stderr)
        sys.exit(INPUT_ERROR_EXIT_CODE)
    except KeyboardInterrupt:
        if os.environ.get(TRACEBACK_VARIABLE):
            raise
        print(f'\n{program_name}: stopped', file=sys.stderr)
        sys.exit(INTERRUPTED_EXIT_CODE)


class RetiredOption(argparse.Action):
    """An option PARROT no longer has: giving it stops the command with a message saying what replaced it

    The option is hidden from the help, and takes an optional value (so that,
    e.g., '-c 3' is caught whole).
    """

    def __init__(self, option_strings: Sequence[str], dest: str, replacement: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        option_strings, dest
            Passed on by argparse
        replacement : str
            What to use instead, e.g. 'use --device cpu'
        """
        super().__init__(option_strings, dest, nargs='?', default=argparse.SUPPRESS, help=argparse.SUPPRESS, **kwargs)
        self.replacement = replacement

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any,
                 option_string: str | None = None) -> None:
        """Stop the command with a usage error naming the replacement."""
        parser.error(f'{option_string} is no longer an option: {self.replacement}')


def header(program_name: str, started: datetime.datetime, examples_hint: bool = True) -> str:
    """The header a command prints when it starts: PARROT's logo, the command, PARROT's version and the time

    Parameters
    ----------
    program_name : str
        Name of the command, e.g. 'parrot-train'
    started : datetime.datetime
        When the command started
    examples_hint : bool, optional
        Also say how to see the command's examples (not needed when they are being shown)

    Returns
    -------
    str
        The header, ending in a newline
    """
    details = [f'{program_name}  |  PARROT {PARROT_VERSION}  |  {started.strftime(HEADER_TIME_FORMAT)}',
               f'Documentation: {DOCUMENTATION_URL}']
    if examples_hint:
        details.append(f'For examples of how to use {program_name}, run: {program_name} {EXAMPLES_FLAG}')

    rule = '=' * (max(len(line) for line in (*LOGO, *details)) + 1)
    return '\n'.join(['', *LOGO, '', rule, *(f' {line}' for line in details), rule]) + '\n'


def print_header(program_name: str, silent: bool) -> None:
    """Print a command's header (see header()), unless `silent`

    Parameters
    ----------
    program_name : str
        Name of the command, e.g. 'parrot-predict'
    silent : bool
        If True, print nothing
    """
    if not silent:
        print(header(program_name, datetime.datetime.now()))


def print_settings(silent: bool, **settings: Any) -> None:
    """Print each setting as 'name : value' on its own line, under a 'Settings:' heading (unless `silent`)

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
    name_width = max(len(name) for name in settings)
    for name, value in settings.items():
        print(f'{name:<{name_width}} : {value}')


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
        raise FileNotFoundError(f'{description} does not exist: {path}')
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
        raise FileNotFoundError(f'The output directory does not exist: {directory}')
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
