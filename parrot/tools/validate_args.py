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
from parrot import process_input_data as pid
from parrot import encode_sequence

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

# Returns the base filename prefix and the directory it is located in 
# from the absolute path
def split_file_and_directory(path):
    filename = path.split('/')[-1]
    directory = path[:-len(filename)]
    check_directory(directory, directory)
    base_filename = directory + filename.split('.')[0]
    return base_filename, directory

def set_encoding_scheme(encode_str):
    if encode_str == 'onehot':
        encoding_scheme = 'onehot'
        input_size = 20
        encoder = None
    elif encode_str == 'biophysics':
        encoding_scheme = 'biophysics'
        input_size = 9
        encoder = None
    else:
        encoding_scheme = 'user'
        encode_file = encode_str
        check_file_exists(encode_file, "Encoding file")
        encoder = encode_sequence.UserEncoder(encode_file)
        input_size = len(encoder)

    return encoding_scheme, encoder, input_size

def check_positive(arg, name):
    if arg < 1:
        raise ValueError(f'{name} must be a positive integer.')

def set_ml_task(n_classes, dtype):
    # Initialize network as classifier or regressor and set collate function
    if n_classes > 1:
        problem_type = 'classification'

        if dtype == 'sequence':
            collate_fnct = pid.seq_class_collate
        elif dtype == 'residues':
            collate_fnct = pid.res_class_collate
        else:
            raise ValueError('Invalid argument `--datatype`: must be "residues" or "sequence"')

    elif n_classes == 1:
        problem_type = 'regression'

        if dtype == 'sequence':
            collate_fnct = pid.seq_regress_collate
        elif dtype == 'residues':
            collate_fnct = pid.res_regress_collate
        else:
            raise ValueError('Invalid argument `--datatype`: must be "residues" or "sequence"')

    else:
        raise ValueError('Number of classes must be a positive integer.')

    return problem_type, collate_fnct

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

