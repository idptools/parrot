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

import csv
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from parrot import figures
from parrot.matrices import figures as matrix_figures
from parrot.matrices.output import name_matrices, write_matrix_predictions
from parrot.metrics import METRIC_DECIMALS, labelled_values, performance_metrics, uncertainty_metrics
from parrot.prediction import LabeledPredictions, output_to_prediction
from parrot.process_input_data import missing_label_mask
from parrot.tasks import Task
from parrot.uncertainties import MODEL_ERROR_SIGNIFICANT_FIGURES

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
    for residue tasks also "<output_prefix>_res_profiles.png", the true and
    predicted values along a few test sequences, and for matrix tasks
    "<output_prefix>_matrix_examples.png", heatmaps of a few test sequences).
    Warns if the network's output for any test sequence is not finite (see
    warn_about_non_finite_outputs()). With uncertainties, the metrics include
    how well the error model fits and the noise ceiling (see
    metrics.uncertainty_metrics), and the statistics file also gives the model
    error s.

    Parameters
    ----------
    labeled : LabeledPredictions
        Network outputs on the test set
    task : Task
        The task the network was trained for
    output_prefix : str
        Path and filename prefix for every output file
    probabilistic : bool, optional
        Report class probabilities rather than class labels (sequence and
        residue classification only)
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

    warn_about_non_finite_outputs(labeled.outputs)
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
    if task.uncertainties:
        assert labeled.uncertainties is not None
        metrics.update(uncertainty_metrics(labeled.true_values, labeled.uncertainties, predictions,
                                           labeled.model_error))
    if write_stats:
        written.append(output_prefix + '_performance_stats.txt')
        write_performance_metrics(metrics, written[-1],
                                  model_error=labeled.model_error if task.uncertainties else None)
        written += write_evaluation_tables(labeled, predictions, task, output_prefix, probabilistic)

    if include_figs:
        # Regression figures show the Pearson r of the metrics, so the two always agree
        written += figures.plot_test_results(labeled.true_values, predictions, task, output_prefix, probabilistic,
                                             pearson=metrics.get('Pearson R'))
        if task.per_residue:
            written += figures.plot_residue_profiles(labeled.names, labeled.true_values, predictions, task,
                                                     output_prefix + '_res_profiles.png')
        if task.is_matrix:
            written.append(output_prefix + '_matrix_examples.png')
            matrix_figures.plot_matrix_examples(labeled.names, labeled.true_values, predictions, task, written[-1])

    return metrics, written


def write_evaluation_tables(labeled: LabeledPredictions, predictions: Sequence[np.ndarray], task: Task,
                            output_prefix: str, probabilistic: bool = False) -> list[str]:
    """Per-protein and length-stratified test metrics with labelled counts.

    Fixed length bins refer to the input sequence, not its number of labels.
    Symmetric cells, missing values and probability outputs use the same masks
    as the pooled metrics. These tables never influence training or selection.
    """
    rows = []
    for name, sequence, true, predicted in zip(labeled.names, labeled.sequences, labeled.true_values, predictions):
        present, _ = labelled_values([true], [predicted], task, probabilistic)
        per_protein = {key: value for key, value in performance_metrics([true], [predicted], task, probabilistic).items()
                       if not key.startswith('Mean protein ')}
        rows.append({'ID': name, 'length': len(sequence), 'labelled_values': len(present), **per_protein})
    grouped = []
    edges = (0, 50, 100, 200, 400, 800)
    for lower, upper in zip(edges, (*edges[1:], None)):
        indices = [i for i, sequence in enumerate(labeled.sequences)
                   if len(sequence) >= lower and (upper is None or len(sequence) < upper)]
        if not indices:
            continue
        group_true = [labeled.true_values[i] for i in indices]
        group_predicted = [predictions[i] for i in indices]
        present, _ = labelled_values(group_true, group_predicted, task, probabilistic)
        grouped.append({'length_bin': f'{lower}+' if upper is None else f'{lower}-{upper - 1}',
                        'proteins': len(indices), 'labelled_values': len(present),
                        **performance_metrics(group_true, group_predicted, task, probabilistic)})
    paths = []
    for suffix, records in (('_per_protein_metrics.tsv', rows), ('_length_metrics.tsv', grouped)):
        if not records:
            continue
        path = output_prefix + suffix
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
        paths.append(path)
    return paths


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
        If given, write every value with exactly this many decimal places (as
        0.0001 and 0.0000, where rounding and printing gives 1e-04 and 0.0)

    Returns
    -------
    str
    """

    values = np.asarray(values)
    text = values.astype(str) if decimals is None else np.char.mod(f'%.{decimals}f', values)
    if values.ndim == 0:
        return str(text)
    if values.ndim == 2:
        return ' '.join(','.join(row) for row in text)
    return ' '.join(text)


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

    Each sequence gets two lines::

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


def warn_about_non_finite_outputs(outputs: Sequence[np.ndarray]) -> None:
    """Warn if the network's output for any test sequence is not finite (NaN or infinite), e.g. after diverging

    Such outputs are written out as they are, but the class predicted for them
    means nothing, and regression metrics involving them are NaN.

    Parameters
    ----------
    outputs : list of numpy arrays
        The network's raw output for each test sequence
    """
    affected = sum(not np.all(np.isfinite(output)) for output in outputs)
    if affected:
        warnings.warn(f"The network's outputs for {affected:,} of {len(outputs):,} test sequences are not finite (NaN "
                      'or infinite), probably because training diverged; predictions and metrics involving them are '
                      'not meaningful')


def matrix_prediction_names(names: Sequence[str] | None, num_sequences: int) -> list[str]:
    """The names predicted matrices are saved under: the sequence IDs, or 1, 2, ... without them

    Parameters
    ----------
    names : sequence of str or None
        Sequence IDs, or None if the sequences have none
    num_sequences : int
        Number of sequences (used only without IDs)

    Returns
    -------
    list of str
    """
    return list(names) if names is not None else [str(number) for number in range(1, num_sequences + 1)]


def write_unlabeled_predictions(names: Sequence[str] | None, sequences: Sequence[str],
                                outputs: Mapping[str, np.ndarray], task: Task, output_path: str,
                                probabilistic: bool = False) -> str:
    """Write predictions for unlabeled sequences (as parrot-predict does)

    One line per input sequence, in input order, so every sequence ID is kept
    even when the same sequence appears more than once::

        <name> <sequence> <prediction(s)>

    or, without names, <sequence> <prediction(s)>.

    For matrix tasks the predictions are instead written as a .npz archive
    with one matrix per input sequence, named by its sequence ID (or, without
    names, by its number among the sequences: 1, 2, ..., not counting blank
    or comment lines); '.npz' is added to
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

    Raises
    ------
    ValueError
        Matrix tasks: if a sequence ID is empty or appears more than once (see
        matrices.output.check_matrix_names)
    """

    if task.is_matrix:
        matrix_names = matrix_prediction_names(names, len(sequences))
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


def write_performance_metrics(metrics: dict[str, float], output_path: str, model_error: float | None = None) -> None:
    """Write performance metrics as "<name> : <value>" lines

    Parameters
    ----------
    metrics : dict
        Metric names and values (see metrics.performance_metrics() and, with
        uncertainties, metrics.uncertainty_metrics())
    output_path : str
        File to write
    model_error : float, optional
        With uncertainties, the network's model error s, written last (to
        MODEL_ERROR_SIGNIFICANT_FIGURES significant figures, as it is in the
        units of the values)
    """

    with open(output_path, 'w') as fh:
        for name, value in metrics.items():
            fh.write(f'{name} : {value:.{METRIC_DECIMALS}f}\n')
        if model_error is not None:
            fh.write(f'Model error s : {model_error:.{MODEL_ERROR_SIGNIFICANT_FIGURES}g}\n')
