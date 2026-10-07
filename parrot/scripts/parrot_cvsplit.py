#!/usr/bin/env python
"""
Usage: $ parrot-cvsplit data_file output_splitfiles <flags>

Driver script for creating PARROT-usable "split-files" for K-fold cross-validation
on a given dataset. Assumes datafile is already properly formatted for use in PARROT.
For more information on usage, use the '-h' flag.

Each fold's test set is one Kth of the data (every sequence is in exactly one
test set), and the rest of the data is divided at random between that fold's
training and validation sets.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import os

import numpy as np

from parrot import process_input_data as pid
from parrot.reproducibility import set_random_seed
from parrot.scripts import arguments

PROGRAM_NAME = 'parrot-cvsplit'

# Number of folds made by default
DEFAULT_FOLDS = 10

# Fraction of each fold's non-test data used for training by default (the rest is the validation set)
DEFAULT_TRAINING_FRACTION = 0.8


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-cvsplit

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME,
                                     description='Write one split-file per fold for k-fold cross-validation.')
    parser.add_argument('data_file', help="the datafile: '<ID> <sequence> <value(s)>' on each line")
    parser.add_argument('output_splitfiles',
                        help="path and name prefix of the split-files, written as '<output_splitfiles>_cv<fold>.txt'")
    parser.add_argument('-k', '--k-folds', default=DEFAULT_FOLDS, type=int, metavar='folds', dest='num_folds',
                        help=f'number of cross-validation folds (default {DEFAULT_FOLDS})')
    parser.add_argument('-t', '--training-fraction', default=DEFAULT_TRAINING_FRACTION, type=float,
                        dest='training_fraction', metavar='fraction',
                        help='fraction of each fold\'s non-test sequences used for training, the rest being for '
                             f'validation (default {DEFAULT_TRAINING_FRACTION})')
    parser.add_argument('--seed', type=int, default=None, metavar='seed',
                        help='random seed, for reproducible split-files (default: not seeded)')
    return parser


def main() -> None:
    """parrot-cvsplit's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Write one split-file per cross-validation fold."""
    args = build_parser().parse_args()
    arguments.print_startup(PROGRAM_NAME, silent=False)
    set_random_seed(args.seed)

    data_file = arguments.check_file_exists(args.data_file, 'Datafile')
    arguments.check_fraction(args.training_fraction, 'Training fraction')
    if args.num_folds <= 1:
        raise ValueError(f'--k-folds must be at least 2, got {args.num_folds}')
    prefix, _ = arguments.output_prefix_and_directory(os.path.abspath(args.output_splitfiles))

    # Blank and comment lines are skipped exactly as they are when the datafile is read for training,
    # so the indices line up
    num_sequences = len(pid.read_tsv_raw(data_file))
    splits = cross_validation_splits(num_sequences, args.num_folds, args.training_fraction)

    print('Creating split-files:')
    print('----------------------')
    for fold, (training, validation, test) in enumerate(splits):
        split_file = f'{prefix}_cv{fold}.txt'
        print(split_file)
        pid.write_split_file(split_file, training, validation, test)


def cross_validation_splits(num_sequences: int, num_folds: int,
                            training_fraction: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The training, validation and test sets of each cross-validation fold

    Parameters
    ----------
    num_sequences : int
        Number of sequences in the datafile
    num_folds : int
        Number of folds
    training_fraction : float
        Fraction of each fold's non-test sequences in its training set

    Returns
    -------
    list of (numpy array, numpy array, numpy array)
        For each fold, the sorted indices of its training, validation and test sets

    Raises
    ------
    ValueError
        If any set of any fold would be empty (checked before anything is written)
    """
    splits = []
    for fold, (non_test, test) in enumerate(pid.cross_validation_folds(np.arange(num_sequences), num_folds)):
        training, validation = pid.vector_split(non_test, training_fraction)
        pid.check_sets_not_empty(training, validation, test,
                                 f'in fold {fold} when {num_sequences} sequences are divided into {num_folds} folds '
                                 f'with a training fraction of {training_fraction}. Use fewer folds or more data.')
        splits.append((training, validation, test))
    return splits


if __name__ == '__main__':
    main()
