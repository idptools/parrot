"""
Checks of a dataset against common rules of thumb, printing a warning for each
one it breaks: duplicated sequences, imbalanced classes, and regression values
that are far from zero, widely spread or skewed. None of these stops PARROT
from training; they point out data that may train poorly or give misleading
test results.

The thresholds are rules of thumb rather than hard limits, chosen to flag
data worth a second look.

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
from scipy.stats import skew

from parrot.tasks import Task

# Classes count as balanced while the most common class has at most this many members per member of the least
# common class; parrot-preprocess --class-balance aims for the same ratio by default
BALANCED_CLASS_RATIO = 1.4

# Beyond this ratio, the imbalance is severe enough that classification is not recommended
SEVERE_CLASS_RATIO = 2.5

# Regression values whose mean is at least this far from zero, or whose standard deviation is larger than this
# (in the units of the data), train less efficiently than standardized values
LARGE_MEAN = 5.0
LARGE_STANDARD_DEVIATION = 5.0

# A skewness larger than this (in either direction) is conventionally "highly skewed" (Bulmer 1979,
# Principles of Statistics)
LARGE_SKEWNESS = 1.0

# Width of the line of '#' framing each warning
BANNER_WIDTH = 45


def warn_about_dataset(data: list[list[Any]], task: Task) -> None:
    """Run every check that applies to a dataset, printing a warning for each problem found

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, value(s)] samples, before class labels are
        converted to integers (so missing labels are NaN)
    task : Task
        The task the data is for
    """
    check_duplicate_sequences(data)
    if task.per_residue:
        report_missing_residue_labels(data)
    if task.is_classification:
        check_class_imbalance(data)
    else:
        check_regression_values(data)


def print_warning(heading: str, details: list[str]) -> None:
    """Print a warning framed by lines of '#' so it stands out from the rest of the output

    Parameters
    ----------
    heading : str
        One-line summary, e.g. 'WARNING: Dataset is skewed.'
    details : list of str
        Lines explaining the problem and what it may cause
    """
    print('\n' + '#' * BANNER_WIDTH)
    print(heading + '\n')
    for line in details:
        print(line)
    print('#' * BANNER_WIDTH + '\n')


def all_values(data: list[list[Any]]) -> np.ndarray:
    """Every value in a dataset (residue values pooled across sequences), leaving out missing (NaN) labels.

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, value(s)] samples

    Returns
    -------
    numpy array of float
    """
    values = np.hstack([np.asarray(sample[2], dtype=float) for sample in data])
    return values[~np.isnan(values)]


# .................................................................
# Every dataset

def check_duplicate_sequences(data: list[list[Any]]) -> None:
    """Warn about sequences that appear more than once

    A sequence in both the training and test sets makes test performance look
    better than it is.

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, value(s)] samples
    """
    counts = Counter(sample[1] for sample in data)
    duplicated = [sequence for sequence, count in counts.items() if count > 1]
    if not duplicated:
        return

    print_warning('WARNING: Duplicate sequences detected in dataset.',
                  ['Duplicated sequences:'] + [f'Duplicate found: {sequence}' for sequence in duplicated]
                  + ['', 'This may cause overfitting or overestimation of performance.', ''])


def report_missing_residue_labels(data: list[list[Any]]) -> None:
    """Report how many residue labels are missing (nan) in residue-mapped data

    Missing labels are ignored during training and evaluation.

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, values] samples, before class labels are
        converted to integers (so missing labels are NaN)
    """
    missing_per_sequence = [int(np.sum(np.isnan(np.asarray(sample[2], dtype=float)))) for sample in data]
    num_missing = sum(missing_per_sequence)
    if num_missing == 0:
        return

    num_labels = sum(len(sample[2]) for sample in data)
    sequences_with_missing = sum(count > 0 for count in missing_per_sequence)
    print_warning('NOTE: Some residues have no label (nan).',
                  [f'{num_missing} of {num_labels} residue labels ({np.around(100 * num_missing / num_labels, 1)}%) '
                   f'are missing, in {sequences_with_missing} of {len(data)} sequences.',
                   'These residues are ignored when training and evaluating.', ''])


# .................................................................
# Classification

def check_class_imbalance(data: list[list[Any]]) -> None:
    """Warn if some classes are much more common than others

    The imbalance is the ratio of the most to the least common class: up to
    BALANCED_CLASS_RATIO is balanced, up to SEVERE_CLASS_RATIO slightly
    imbalanced, and beyond that severely imbalanced.

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, label(s)] samples (missing labels as NaN)
    """
    counts = Counter(all_values(data))
    ratio = max(counts.values()) / min(counts.values())
    if ratio <= BALANCED_CLASS_RATIO:
        return

    total = sum(counts.values())
    frequencies = ['Class frequencies:'] + [f'  {int(label)} : {np.around(100 * count / total, 1)}%'
                                            for label, count in counts.items()] + ['']
    if ratio <= SEVERE_CLASS_RATIO:
        print_warning('WARNING: Slight class imbalance detected in dataset.',
                      frequencies + ['This may skew predictions towards overrepresented classes.', ''])
    else:
        print_warning('WARNING: Severe class imbalance detected in dataset.',
                      frequencies + ['Predictions will be strongly skewed towards overrepresented classes.',
                                     'Classification not recommended with current dataset.', ''])


# .................................................................
# Regression

def check_regression_values(data: list[list[Any]]) -> None:
    """Warn if regression values are far from zero, widely spread or skewed

    Parameters
    ----------
    data : list
        Parsed [seqID, sequence, value(s)] samples (missing labels as NaN)
    """
    values = all_values(data)
    check_mean(values)
    check_spread(values)
    check_skew(values)


def check_mean(values: np.ndarray) -> None:
    """Warn if the mean is LARGE_MEAN or more from zero.

    Parameters
    ----------
    values : numpy array
        Every regression value
    """
    mean = np.mean(values)
    if abs(mean) >= LARGE_MEAN:
        print_warning('WARNING: Dataset not centered around zero.',
                      [f'Data mean: {np.around(mean, 2)}', '', 'This may decrease training efficiency.', ''])


def check_spread(values: np.ndarray) -> None:
    """Warn if the standard deviation is larger than LARGE_STANDARD_DEVIATION.

    Parameters
    ----------
    values : numpy array
        Every regression value
    """
    standard_deviation = np.std(values)
    if standard_deviation > LARGE_STANDARD_DEVIATION:
        print_warning('WARNING: Dataset has large variance',
                      [f'Data standard deviation: {np.around(standard_deviation, 2)}', '',
                       'This may decrease training efficiency.', 'Consider standardizing your dataset.', ''])


def check_skew(values: np.ndarray) -> None:
    """Warn if the values are strongly skewed (a long tail of high or of low values).

    Parameters
    ----------
    values : numpy array
        Every regression value
    """
    skewness = skew(values)
    if abs(skewness) > LARGE_SKEWNESS:
        print_warning('WARNING: Dataset is skewed.',
                      [f'Skewness: {np.around(skewness, 2)}', '', 'This may cause model bias.', ''])
