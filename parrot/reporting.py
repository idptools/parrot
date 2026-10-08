"""
Writing what a trained network predicts, and how well it performs, to files:
prediction files for labeled (test-set) and unlabeled sequences, and the
performance-metric file. The metrics themselves are computed in metrics.py, and
the figures drawn in figures.py.

All functions here take plain numpy arrays (or the LabeledPredictions produced
by prediction.predict_labeled), so they work the same for every architecture.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from parrot import figures
from parrot.matrices import figures as matrix_figures
from parrot.matrices.output import name_matrices, write_matrix_predictions
from parrot.metrics import METRIC_DECIMALS, performance_metrics
from parrot.prediction import LabeledPredictions, output_to_prediction
from parrot.process_input_data import missing_label_mask
from parrot.tasks import Task

# Number of decimal places used when writing class probabilities
PROBABILITY_DECIMALS = 4


def report_test_results(labeled: LabeledPredictions, task: Task, output_prefix: str, probabilistic: bool = False,
                        include_figs: bool = False, write_stats: bool = True) -> tuple[dict[str, float], list[str]]:
    """Write everything describing performance on the test set

    Always writes the predictions: "<output_prefix>_predictions.tsv", or for
    matrix tasks "<output_prefix>_predictions.npz" (one predicted matrix per
    test sequence, named by its ID). Also writes
    "<output_prefix>_performance_stats.txt" if `write_stats`, and the figures
    appropriate to the task if `include_figs` (see figures.plot_test_results;
    for matrix tasks also "<output_prefix>_matrix_examples.png", heatmaps of a
    few test sequences).

    Parameters
    ----------
    labeled : LabeledPredictions
        Network outputs on the test set
    task : Task
        The task the network was trained for
    output_prefix : str
        Path and filename prefix for every output file
    probabilistic : bool, optional
        Report class probabilities rather than class labels (classification only)
    include_figs : bool, optional
        Whether to save figures
    write_stats : bool, optional
        Whether to write the performance-metrics file

    Returns
    -------
    dict
        The performance metrics (computed whether or not they are written)
    list of str
        The files written
    """

    predictions = [output_to_prediction(output, task, probabilistic) for output in labeled.outputs]
    decimals = PROBABILITY_DECIMALS if probabilistic else None
    written = []

    if task.is_matrix:
        written.append(write_matrix_predictions(output_prefix + '_predictions.npz',
                                                name_matrices(labeled.names, predictions)))
    else:
        written.append(output_prefix + '_predictions.tsv')
        write_predictions(labeled.names, labeled.sequences, labeled.true_values, predictions, task, written[-1],
                          decimals=decimals)

    metrics = performance_metrics(labeled.true_values, predictions, task, probabilistic)
    if write_stats:
        written.append(output_prefix + '_performance_stats.txt')
        write_performance_metrics(metrics, written[-1])

    if include_figs:
        written += figures.plot_test_results(labeled.true_values, predictions, task, output_prefix, probabilistic)
        if task.is_matrix:
            written.append(output_prefix + '_matrix_examples.png')
            matrix_figures.plot_matrix_examples(labeled.names, labeled.true_values, predictions, task, written[-1])

    return metrics, written


# .................................................................
# Formatting values

def format_values(values: np.ndarray, decimals: int | None = None) -> str:
    """Format a value, or an array of values, as whitespace-separated text

    A 2D array (several values for each residue, e.g. each residue's class
    probabilities) is written as one token per residue, its values joined by
    commas, so the tokens still line up with the residues of the sequence.

    Parameters
    ----------
    values : numpy array or scalar
        The value(s) to format
    decimals : int, optional
        If given, round to this many decimal places first

    Returns
    -------
    str
    """

    values = np.asarray(values)
    if decimals is not None:
        values = np.around(values, decimals=decimals)
    if values.ndim == 0:
        return str(values)
    if values.ndim == 2:
        return ' '.join(','.join(row.astype(str)) for row in values)
    return ' '.join(values.astype(str))


def format_true_values(true_values: np.ndarray, task: Task) -> str:
    """Format a sequence's true value(s) as text, writing missing labels as 'nan'

    Parameters
    ----------
    true_values : numpy array or scalar
        True value(s) as stored by parse_file()
    task : Task
        The task the values are for

    Returns
    -------
    str
    """

    missing = missing_label_mask(true_values, task)
    if not np.any(missing):
        return format_values(true_values)

    tokens = np.atleast_1d(np.asarray(true_values)).astype(str)
    tokens[np.atleast_1d(missing)] = 'nan'
    return ' '.join(tokens)


# .................................................................
# Files

def write_predictions(names: Sequence[str], sequences: Sequence[str], true_values: Sequence[np.ndarray],
                      predictions: Sequence[np.ndarray], task: Task, output_path: str,
                      decimals: int | None = None) -> None:
    """Write true and predicted values for labeled sequences

    Each sequence gets two lines:
        <name>_TRUE <sequence> <true value(s)>
        <name>_PRED <sequence> <predicted value(s)>
    Sequences without an ID (datafiles read with --exclude-seq-id) are named
    test0, test1, ... Missing true labels are written as 'nan'.

    Parameters
    ----------
    names, sequences : list of str
        Sequence IDs and sequences
    true_values, predictions : list of numpy arrays
        True and predicted value(s) for each sequence
    task : Task
        The task the network was trained for
    output_path : str
        File to write
    decimals : int, optional
        If given, round predictions to this many decimal places
    """

    with open(output_path, 'w') as fh:
        for i, (name, sequence, true_value, prediction) in enumerate(zip(names, sequences, true_values, predictions)):
            if name == '':
                name = f'test{i}'
            fh.write(f'{name}_TRUE {sequence} {format_true_values(true_value, task)}\n')
            fh.write(f'{name}_PRED {sequence} {format_values(prediction, decimals)}\n')


def write_unlabeled_predictions(names: Sequence[str] | None, sequences: Sequence[str],
                                outputs: Mapping[str, np.ndarray], task: Task, output_path: str,
                                probabilistic: bool = False) -> str:
    """Write predictions for unlabeled sequences (as parrot-predict does)

    One line per input sequence, in input order, so every sequence ID is kept
    even when the same sequence appears more than once:
        <name> <sequence> <prediction(s)>
    or, without names, <sequence> <prediction(s)>.

    For matrix tasks the predictions are instead written as a .npz archive
    with one matrix per input sequence, named by its sequence ID (or, without
    names, by its line number, counting from 1); '.npz' is added to
    `output_path` if it does not end with it.

    Parameters
    ----------
    names : list of str or None
        The ID of each sequence, or None to write sequences without IDs
    sequences : list of str
        The sequences, in input order
    outputs : dict
        Raw network output for each unique sequence (see prediction.predict_sequences)
    task : Task
        The task the network was trained for
    output_path : str
        File to write
    probabilistic : bool, optional
        For classification, write the probability of each class (to
        PROBABILITY_DECIMALS places) rather than the most likely class

    Returns
    -------
    str
        The path written
    """

    if task.is_matrix:
        matrix_names = list(names) if names is not None else [str(number) for number in range(1, len(sequences) + 1)]
        matrices = [output_to_prediction(outputs[sequence], task, probabilistic) for sequence in sequences]
        return write_matrix_predictions(output_path, name_matrices(matrix_names, matrices))

    decimals = PROBABILITY_DECIMALS if probabilistic else None
    with open(output_path, 'w') as fh:
        for index, sequence in enumerate(sequences):
            predicted = output_to_prediction(outputs[sequence], task, probabilistic)
            values = format_values(predicted, decimals)
            if names is None:
                fh.write(f'{sequence} {values}\n')
            else:
                fh.write(f'{names[index]} {sequence} {values}\n')
    return output_path


def write_performance_metrics(metrics: dict[str, float], output_path: str) -> None:
    """Write performance metrics as "<name> : <value>" lines

    Parameters
    ----------
    metrics : dict
        Metric names and values (see metrics.performance_metrics())
    output_path : str
        File to write
    """

    with open(output_path, 'w') as fh:
        for name, value in metrics.items():
            fh.write(f'{name} : {value:.{METRIC_DECIMALS}f}\n')
