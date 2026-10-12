"""
Reading matrix data, checking it, and padding it into batches.

A matrix dataset has two files:

- the datafile, with '<ID> <sequence>' on each line, exactly as other PARROT
  datafiles but without values (so split-files and the other tools that work
  with line numbers apply unchanged);
- a NumPy .npz file holding one array per sequence ID, of shape
  [sequence length X number of columns], written for example with
  ``numpy.savez_compressed('targets.npz', **{sequence_id: matrix, ...})``.

Each matrix has one row per residue, and either one column per residue (a
square map of every pair of residues; Task.width is None) or the same number of
columns N for every sequence (Task.width = N). Missing values are NaN and are
ignored in training and evaluation; for classification, the other values are
whole-number class labels.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections import Counter
from typing import Any

import numpy as np
import torch

from parrot.parrot_exceptions import IOExceptionParrot
from parrot.tasks import IGNORED_CLASS_LABEL, Task

# How many problem sequence IDs an error message lists before summarising the rest
MAX_IDS_LISTED = 5

# Two values of a symmetric map closer than this (relative, or absolute near zero) count as equal, allowing for
# rounding when the map was computed or written
SYMMETRY_RELATIVE_TOLERANCE = 1e-6
SYMMETRY_ABSOLUTE_TOLERANCE = 1e-9

# NumPy dtype kinds of the values a matrix may hold: booleans, signed and unsigned integers, and floats
REAL_NUMBER_KINDS = 'biuf'


# .................................................................
# Reading

def read_matrix_file(path: str) -> dict[str, np.ndarray]:
    """Read every array of a .npz file, by name (sequence ID)

    Parameters
    ----------
    path : str
        The .npz file

    Returns
    -------
    dict
        Sequence ID -> array

    Raises
    ------
    IOExceptionParrot
        If the file cannot be read as a .npz archive (a single array, as in a
        .npy file, is not one), or an array in it cannot be read (e.g. an
        array of Python objects, which would need unpickling)
    """
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise IOExceptionParrot(f'Cannot read the matrix file {path} as a NumPy .npz archive: {error}') from error
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise IOExceptionParrot(f'The matrix file {path} holds a single array, not a .npz archive with one matrix '
                                'per sequence ID; write it with numpy.savez(path, **matrices), where matrices maps '
                                'each sequence ID to its matrix')
    with loaded as archive:
        try:
            return {name: archive[name] for name in archive.files}
        except (OSError, ValueError) as error:
            raise IOExceptionParrot(f'Cannot read the matrices in the matrix file {path} ({error}); each must be an '
                                    'array of numbers, as numpy.savez writes from NumPy arrays') from error


def attach_matrices(lines: list[list[str]], matrices: dict[str, np.ndarray], task: Task) -> list[list[Any]]:
    """Pair each '<ID> <sequence>' datafile line with its matrix, checking the matrix fits the sequence

    Parameters
    ----------
    lines : list of list of str
        Datafile lines split into fields (see process_input_data.read_tsv_raw)
    matrices : dict
        Sequence ID -> matrix (see read_matrix_file)
    task : Task
        A matrix task

    Returns
    -------
    list
        One [ID, sequence, matrix] sample per line, each matrix a float64 array
        (NaN where a value is missing)

    Raises
    ------
    IOExceptionParrot
        If a line is not '<ID> <sequence>', an ID appears twice or has no
        matrix, or a matrix does not fit its sequence (see check_matrix)
    """
    for number, line in enumerate(lines, start=1):
        if len(line) != 2:
            raise IOExceptionParrot(f"Data line {number} should be '<ID> <sequence>' for matrix data (the values are "
                                    f"in the matrix file), but has {len(line)} column(s): {' '.join(line)}")

    ids = [line[0] for line in lines]
    repeated = sorted(sequence_id for sequence_id, count in Counter(ids).items() if count > 1)
    if repeated:
        raise IOExceptionParrot('Every sequence ID of matrix data must be unique (it names the sequence\'s matrix), '
                                f'but these appear more than once: {listed(repeated)}')

    missing = [sequence_id for sequence_id in ids if sequence_id not in matrices]
    if missing:
        raise IOExceptionParrot(f'{len(missing)} sequence(s) have no matrix in the matrix file: {listed(missing)}')

    return [[sequence_id, sequence, check_matrix(sequence_id, sequence, matrices[sequence_id], task)]
            for sequence_id, sequence in lines]


def check_matrix(sequence_id: str, sequence: str, matrix: np.ndarray, task: Task) -> np.ndarray:
    """Check that a sequence's matrix has the right shape and values, returning it as float64

    Parameters
    ----------
    sequence_id : str
        The sequence's ID (for error messages)
    sequence : str
    matrix : numpy array
        The sequence's matrix, as read from the matrix file
    task : Task
        A matrix task

    Returns
    -------
    numpy array
        float64 [len(sequence) X columns]

    Raises
    ------
    IOExceptionParrot
        If the matrix is not two-dimensional, does not have one row per residue
        and the task's number of columns, holds values that are not real
        numbers or are infinite, has no values at all (every value missing), or
        is not symmetric although the task is
    """
    matrix = np.asarray(matrix)
    expected_columns = len(sequence) if task.width is None else task.width
    if matrix.ndim != 2 or matrix.shape != (len(sequence), expected_columns):
        raise IOExceptionParrot(f'The matrix of {sequence_id} has shape {matrix.shape}, but should be '
                                f'({len(sequence)}, {expected_columns}): one row per residue and '
                                + ('one column per residue' if task.width is None else f'{task.width} columns'))

    # Real numbers only: converting complex numbers to float64 would silently drop their imaginary parts
    if matrix.dtype.kind not in REAL_NUMBER_KINDS:
        raise IOExceptionParrot(f'The matrix of {sequence_id} does not hold real numbers ({matrix.dtype})')
    values = matrix.astype(np.float64)
    if np.any(np.isinf(values)):
        raise IOExceptionParrot(f'The matrix of {sequence_id} holds infinite values; values must be finite numbers '
                                'or NaN (missing)')
    if np.all(np.isnan(values)):
        raise IOExceptionParrot(f'Every value of the matrix of {sequence_id} is missing (NaN)')
    if task.symmetric and not np.allclose(values, values.T, rtol=SYMMETRY_RELATIVE_TOLERANCE,
                                          atol=SYMMETRY_ABSOLUTE_TOLERANCE, equal_nan=True):
        raise IOExceptionParrot(f'The matrix of {sequence_id} is not symmetric, but the task is')
    return values


def matrix_class_labels(sequence_id: str, values: np.ndarray, task: Task) -> np.ndarray:
    """Convert a classification matrix to integer class labels, with missing values as IGNORED_CLASS_LABEL

    Parameters
    ----------
    sequence_id : str
        The sequence's ID (for error messages)
    values : numpy array
        float matrix (NaN where missing)
    task : Task
        A matrix classification task

    Returns
    -------
    numpy array
        int64 matrix of the same shape

    Raises
    ------
    IOExceptionParrot
        If a label is not a whole number in the range [0, num_classes - 1].
        Labels are never silently truncated (e.g. 1.5 is not turned into 1).
    """
    missing = np.isnan(values)
    present = values[~missing]
    if np.any(present != np.round(present)):
        raise IOExceptionParrot(f'The matrix of {sequence_id} holds class labels that are not whole numbers')
    if np.any(present < 0) or np.any(present >= task.num_classes):
        raise IOExceptionParrot(f'The matrix of {sequence_id} holds class labels outside 0 to {task.num_classes - 1}')
    return np.where(missing, IGNORED_CLASS_LABEL, values).astype(np.int64)


def listed(ids: list[str]) -> str:
    """The first MAX_IDS_LISTED IDs, and how many more there are, as text."""
    shown = ', '.join(ids[:MAX_IDS_LISTED])
    return shown if len(ids) <= MAX_IDS_LISTED else f'{shown} and {len(ids) - MAX_IDS_LISTED} more'


# .................................................................
# Batching

def matrix_width(task: Task, padded_length: int) -> int:
    """The number of columns of a batch's padded matrices: the padded length for square maps, otherwise the task's width

    Parameters
    ----------
    task : Task
        A matrix task
    padded_length : int
        The length the batch's sequences were padded to

    Returns
    -------
    int
    """
    return padded_length if task.width is None else task.width


def pad_matrix_targets(targets: list[np.ndarray], task: Task, padded_length: int) -> torch.Tensor:
    """Stack the matrices of a batch into one tensor, padded to the batch's padded length

    Padded rows (and, for square maps, padded columns) are marked like missing
    values, so the loss ignores them both: IGNORED_CLASS_LABEL for
    classification and NaN for regression. For a symmetric task, the cells
    below the diagonal are marked missing too, so each pair is counted once.

    Parameters
    ----------
    targets : list of numpy arrays
        Each sequence's matrix: int class labels (classification) or floats (regression)
    task : Task
        A matrix task
    padded_length : int
        The length the batch's sequences were padded to

    Returns
    -------
    torch.Tensor
        Classification: int64 [batch_dim X padded_length X width].
        Regression: float32 [batch_dim X padded_length X width X 1].
    """
    width = matrix_width(task, padded_length)
    if task.is_classification:
        padded = np.full((len(targets), padded_length, width), IGNORED_CLASS_LABEL, dtype=np.int64)
    else:
        padded = np.full((len(targets), padded_length, width), np.nan, dtype=np.float32)

    for index, target in enumerate(targets):
        rows, columns = np.shape(target)
        padded[index, :rows, :columns] = target

    if task.symmetric:
        below_diagonal = np.tril(np.ones((padded_length, width), dtype=bool), k=-1)
        padded[:, below_diagonal] = IGNORED_CLASS_LABEL if task.is_classification else np.nan

    if task.is_classification:
        return torch.from_numpy(padded)
    return torch.from_numpy(padded).unsqueeze(-1)


def unpad_matrix_outputs(outputs: np.ndarray, lengths: list[int], task: Task) -> list[np.ndarray]:
    """Cut each sequence's matrix out of a padded batch of network outputs

    Parameters
    ----------
    outputs : numpy array
        [batch_dim X padded_length X width X num_classes]
    lengths : list of int
        The true length of each sequence
    task : Task
        A matrix task

    Returns
    -------
    list of numpy arrays
        [length X length X num_classes] (square maps) or [length X width X num_classes] per sequence
    """
    return [outputs[index, :length, :length if task.width is None else task.width]
            for index, length in enumerate(lengths)]
