#!/usr/bin/env python
"""
Usage: $ parrot-cvsplit data_file output_splitfiles <flags>

Driver script for creating PARROT-usable "split-files" for K-fold cross-validation
on a given dataset. Assumes datafile is already properly formatted for use in PARROT.
For more information on usage, use the '-h' flag.

Each fold's test set is about one Kth of the data (every sequence is in exactly
one test set), and the rest of the data is divided at random between that
fold's training and validation sets. Identical sequences are always in the same set
of every fold. To keep similar sequences apart as well, make the split with
parrot-preprocess --split homology or kmer instead.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import math
import os

import numpy as np

from parrot import process_input_data as pid
from parrot.reproducibility import set_random_seed
from parrot.scripts import arguments
from parrot.scripts.examples import add_examples_argument

PROGRAM_NAME = 'parrot-cvsplit'

# Number of folds made by default
DEFAULT_FOLDS = 10

# Fraction of each fold's non-test data used for training by default (the rest is the validation set)
DEFAULT_TRAINING_FRACTION = 0.8

# The split-files' extension (a prefix given with it has it removed, so 'splits.txt' does not give 'splits.txt_cv0.txt')
SPLIT_FILE_EXTENSION = '.txt'


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-cvsplit

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME,
                                     description='Write one split-file per fold for k-fold cross-validation.')
    add_examples_argument(parser, PROGRAM_NAME)
    parser.add_argument('data_file', help="the datafile: '<ID> <sequence> <value(s)>' on each line (for matrix data, "
                                          "'<ID> <sequence>')")
    parser.add_argument('output_splitfiles',
                        help="path and name prefix of the split-files, written as '<output_splitfiles>_cv<fold>.txt'")
    parser.add_argument('-k', '--k-folds', default=DEFAULT_FOLDS, type=int, metavar='folds', dest='num_folds',
                        help=f'number of cross-validation folds (default {DEFAULT_FOLDS})')
    parser.add_argument('-t', '--training-fraction', default=DEFAULT_TRAINING_FRACTION, type=float,
                        dest='training_fraction', metavar='fraction',
                        help='fraction of each fold\'s non-test sequences used for training, the rest being for '
                             f'validation (default {DEFAULT_TRAINING_FRACTION})')
    parser.add_argument('--exclude-seq-id', dest='exclude_seq_id', action='store_true',
                        help='the datafile has no sequence IDs: each line starts with the sequence')
    parser.add_argument('--seed', type=int, default=None, metavar='seed',
                        help='random seed, for reproducible split-files (default: not seeded)')
    return parser


def main() -> None:
    """parrot-cvsplit's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Write one split-file per cross-validation fold."""
    args = build_parser().parse_args()
    arguments.print_header(PROGRAM_NAME, silent=False)
    set_random_seed(args.seed)

    data_file = arguments.check_file_exists(args.data_file, 'Datafile')
    arguments.check_fraction(args.training_fraction, 'Training fraction')
    if args.num_folds <= 1:
        raise ValueError(f'--k-folds must be at least 2, got {args.num_folds}')
    prefix = split_file_prefix(args.output_splitfiles)

    sequences = datafile_sequences(data_file, args.exclude_seq_id)
    splits = cross_validation_splits(sequences, args.num_folds, args.training_fraction)

    print('Creating split-files:')
    print('----------------------')
    for fold, (training, validation, test) in enumerate(splits):
        split_file = f'{prefix}_cv{fold}{SPLIT_FILE_EXTENSION}'
        print(split_file)
        pid.write_split_file(split_file, training, validation, test)


def datafile_sequences(data_file: str, exclude_seq_id: bool) -> list[str]:
    """The sequence on each data line of a datafile, in order

    Blank and comment lines are skipped exactly as they are when the datafile
    is read for training, so the indices line up with parrot-train's.

    Parameters
    ----------
    data_file : str
        The datafile
    exclude_seq_id : bool
        True if the lines have no sequence ID, so each starts with the sequence

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If a line has no sequence, or its sequence is a number (as it is for a
        datafile without IDs read as if it had them)
    """
    column = 0 if exclude_seq_id else 1
    sequences = []
    for number, line in enumerate(pid.read_tsv_raw(data_file), start=1):
        if len(line) <= column:
            raise ValueError(f'Data line {number} of {data_file} has no sequence: '
                             + ("expected '<sequence> <value(s)>'" if exclude_seq_id
                                else "expected '<ID> <sequence> <value(s)>'"))
        if is_finite_number(line[column]):
            hint = '' if exclude_seq_id else '; if the datafile has no sequence IDs, use --exclude-seq-id'
            raise ValueError(f"Data line {number} of {data_file} has the number '{line[column]}' where its sequence "
                             f'should be{hint}')
        sequences.append(line[column])
    return sequences


def is_finite_number(text: str) -> bool:
    """Whether text is a finite number (not 'NAN' or 'INF', which are also sequences: Asn-Ala-Asn, Ile-Asn-Phe)."""
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def split_file_prefix(output_splitfiles: str) -> str:
    """The prefix of the split-files' names: the argument as given (dots and all), less a final '.txt'

    Parameters
    ----------
    output_splitfiles : str
        The output_splitfiles argument

    Returns
    -------
    str
        An absolute path prefix; the split-files are '<prefix>_cv<fold>.txt'

    Raises
    ------
    FileNotFoundError
        If the directory it is in does not exist
    """
    prefix = os.path.abspath(output_splitfiles).removesuffix(SPLIT_FILE_EXTENSION)
    arguments.output_prefix_and_directory(prefix)  # checks the directory exists
    return prefix


def cross_validation_splits(sequences: list[str], num_folds: int,
                            training_fraction: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The training, validation and test sets of each cross-validation fold, keeping identical sequences together

    Parameters
    ----------
    sequences : list of str
        The sequence on each line of the datafile
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
        If there are fewer distinct sequences than folds, or any set of any fold would be empty (checked before
        anything is written)
    """
    num_sequences = len(sequences)
    groups = np.array(sequences, dtype=object)
    splits = []
    for fold, (non_test, test) in enumerate(pid.cross_validation_folds(np.arange(num_sequences), num_folds, groups)):
        part = pid.share_out_groups(groups[non_test], [training_fraction, 1 - training_fraction])
        training, validation = non_test[part == 0], non_test[part == 1]
        pid.check_sets_not_empty(training, validation, test,
                                 f'in fold {fold} when {num_sequences} sequences are divided into {num_folds} folds '
                                 f'with a training fraction of {training_fraction}. Use fewer folds or more data.')
        splits.append((training, validation, test))
    return splits


if __name__ == '__main__':
    main()
