"""
Functions for validating user-provided arguments.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import os

# Ensure that provided file is valid
def check_file_exists(f, name):
    file = os.path.abspath(f)
    if not os.path.isfile(file):
        err_str = f"{name} does not exist."
        raise FileNotFoundError(err_str)
    else:
        return file

def check_directory(dir, name):
    if not os.path.exists(dir):
        raise FileNotFoundError(f'{name} does not exist.')

def split_file_and_directory(path):
    """Split an absolute file path into an output prefix and its directory

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
    check_directory(directory, directory)
    base_filename = os.path.join(directory, os.path.splitext(filename)[0])
    return base_filename, directory

def check_positive(arg, name):
    if arg < 1:
        raise ValueError(f'{name} must be a positive integer.')

def check_between_zero_and_one(val, name):
    if val >= 1 or val <= 0:
        raise ValueError(f'{name} must be between 0 and 1.')

def check_stop_condition(stop_cond, n_epochs):
    if stop_cond == 'auto':
        if n_epochs > 10:
            print("Warning: Stop condition is set to 'auto' and num_epochs > 10." +
                   " Network training may take a long time.\n")
    elif stop_cond != 'iter':
        raise ValueError('Invalid argument for `--stop` -- must be "auto" or "iter".')

