"""
Measures of how well a trained network performs on labeled data (e.g. the
held-out test set).

Everything here takes plain numpy arrays: the true and predicted value(s) of
each sequence (see prediction.output_to_prediction), so it works the same for
every architecture. Writing the results to files is in reporting.py, and
plotting them in figures.py.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, auc, average_precision_score, f1_score, matthews_corrcoef, roc_curve

from parrot.process_input_data import missing_label_mask
from parrot.tasks import Task

# Number of decimal places performance metrics are rounded to
METRIC_DECIMALS = 3


def flatten(values: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-sequence values (scalars, per-residue arrays or matrices) into one 1D array.

    Parameters
    ----------
    values : list of numpy arrays or scalars
        One entry per sequence

    Returns
    -------
    numpy array
    """
    return np.concatenate([np.ravel(value) for value in values])


def upper_triangles(matrices: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The cells on and above the diagonal of each square matrix: each pair of residues once, for symmetric maps.

    Parameters
    ----------
    matrices : list of numpy arrays
        Square matrices

    Returns
    -------
    list of numpy arrays
        One 1D array per matrix
    """
    return [np.asarray(matrix)[np.triu_indices(len(matrix))] for matrix in matrices]


def labelled_values(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                    probabilistic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pool true and predicted values across sequences, keeping only labelled ones

    Residues (or matrix cells) with missing labels are dropped, so metrics and
    figures describe only values that are known. For symmetric matrix tasks
    each pair of residues is counted once.

    Parameters
    ----------
    true_values, predictions : list of numpy arrays
        True and predicted value(s) for each sequence
    task : Task
        The task the network was trained for
    probabilistic : bool, optional
        True if `predictions` are class probabilities (sequence or residue classification)

    Returns
    -------
    numpy array
        The labelled true values, pooled into one 1D array
    numpy array
        The matching predictions: a 1D array, or class probabilities if
        `probabilistic` ([num_labelled X num_classes], one row per sequence or
        labelled residue)
    """

    if probabilistic:
        # One row of class probabilities per sequence (sequence data) or per residue (residue data), in the same
        # order as the pooled true labels, so residues with a missing label are left out of both
        true_flat = flatten(true_values)
        probabilities = np.vstack(predictions)
        labelled = ~missing_label_mask(true_flat, task)
        return true_flat[labelled], probabilities[labelled]

    # A symmetric map holds each pair of residues twice: count it once, as training does
    if task.symmetric:
        true_values, predictions = upper_triangles(true_values), upper_triangles(predictions)

    true_flat = flatten(true_values)
    predicted_flat = flatten(predictions)
    labelled = ~missing_label_mask(true_flat, task)
    return true_flat[labelled], predicted_flat[labelled]


def one_hot_classes(classes: np.ndarray, num_classes: int) -> np.ndarray:
    """Turn class numbers into one-hot rows (one column per class), as the ROC and precision-recall functions need.

    Parameters
    ----------
    classes : numpy array
        Class number (0, 1, ...) of each sample
    num_classes : int
        Number of classes

    Returns
    -------
    numpy int array
        [num_samples X num_classes], 1 in each sample's class column and 0 elsewhere
    """
    one_hot: np.ndarray = np.eye(num_classes, dtype=int)[classes.astype(int)]
    return one_hot


def confusion_counts(true_classes: np.ndarray, predicted_classes: np.ndarray, num_classes: int) -> np.ndarray:
    """Count the samples with each combination of predicted and true class.

    Parameters
    ----------
    true_classes, predicted_classes : numpy int arrays
        Paired true and predicted classes
    num_classes : int
        Number of classes

    Returns
    -------
    numpy array
        [num_classes X num_classes] counts: rows are predicted classes, columns true classes
    """
    counts = np.zeros((num_classes, num_classes))
    np.add.at(counts, (predicted_classes.astype(int), true_classes.astype(int)), 1)
    return counts


def performance_metrics(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                        probabilistic: bool = False) -> dict[str, float]:
    """Compute summary statistics of performance

    Residue-mapped values are pooled across all sequences; residues with missing
    labels are left out.

    Parameters
    ----------
    true_values, predictions : list of numpy arrays
        True and predicted value(s) for each sequence
    task : Task
        The task the network was trained for
    probabilistic : bool, optional
        True if `predictions` are class probabilities

    Returns
    -------
    dict
        Regression: Pearson R and Spearman R. Classification: Matthews
        correlation coefficient, weighted F1 score and accuracy, plus (if
        probabilistic) the micro-averaged areas under the ROC and
        precision-recall curves. Values are rounded to METRIC_DECIMALS places.
    """

    true_flat, predicted_flat = labelled_values(true_values, predictions, task, probabilistic)
    metrics: dict[str, Any] = {}

    if not task.is_classification:
        metrics['Pearson R'] = pearsonr(true_flat, predicted_flat)[0]
        metrics['Spearman R'] = spearmanr(true_flat, predicted_flat)[0]

    else:
        if probabilistic:
            probabilities = predicted_flat
            true_one_hot = one_hot_classes(true_flat, task.num_classes)
            metrics['Area under Precision-Recall curve'] = average_precision_score(true_one_hot, probabilities,
                                                                                   average='micro')
            false_positive_rate, true_positive_rate, _ = roc_curve(true_one_hot.ravel(), probabilities.ravel())
            metrics['Area under ROC'] = auc(false_positive_rate, true_positive_rate)
            predicted_flat = np.argmax(probabilities, axis=1)

        metrics['Matthews Correlation Coef'] = matthews_corrcoef(true_flat, predicted_flat)
        metrics['F1 Score'] = f1_score(true_flat, predicted_flat, average='weighted')
        metrics['Accuracy'] = accuracy_score(true_flat, predicted_flat)

    return {name: round(float(value), METRIC_DECIMALS) for name, value in metrics.items()}
