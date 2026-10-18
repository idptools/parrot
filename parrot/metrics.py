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

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, auc, average_precision_score, f1_score, matthews_corrcoef, roc_curve

from parrot.process_input_data import missing_label_mask
from parrot.tasks import Task
from parrot.uncertainties import MIN_VARIANCE

# Number of decimal places performance metrics are rounded to
METRIC_DECIMALS = 3

# A correlation needs at least this many pairs of values (with fewer, it is reported as NaN)
MIN_CORRELATION_VALUES = 2

# With uncertainties, how many total SDs (sqrt(SD^2 + s^2)) from its prediction a value may be to count as within
# that many: 1 and 2 SDs hold 68% and 95% of a Gaussian
COVERAGE_SDS = (1, 2)


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

    Raises
    ------
    ValueError
        If `probabilistic` is True for matrix data (which has no class
        probabilities; see scripts/common.check_probabilistic_classification)
    """

    if probabilistic:
        if task.is_matrix:
            raise ValueError('Class probabilities apply to sequence and residue classification only, not to matrix '
                             'data')
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


def correlation(statistic: Callable[[np.ndarray, np.ndarray], Any], true_values: np.ndarray,
                predicted_values: np.ndarray) -> float:
    """A correlation coefficient (scipy's pearsonr or spearmanr), or NaN where it is undefined

    It is undefined for fewer than MIN_CORRELATION_VALUES pairs, for a set of
    values that are all the same, and for values that are not finite (e.g. the
    predictions of a network that diverged).

    Parameters
    ----------
    statistic : callable
        scipy.stats.pearsonr or scipy.stats.spearmanr
    true_values, predicted_values : numpy arrays
        Paired values

    Returns
    -------
    float
    """
    if len(true_values) < MIN_CORRELATION_VALUES:
        return math.nan
    if not (np.all(np.isfinite(true_values)) and np.all(np.isfinite(predicted_values))):
        return math.nan
    if np.ptp(true_values) == 0 or np.ptp(predicted_values) == 0:
        return math.nan
    return float(statistic(true_values, predicted_values).statistic)


def roc_area(true_one_hot: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the ROC curve: class 1's for two classes (the usual binary ROC, which class 0's equals), otherwise
    the micro-average over every class, as the ROC figure draws (see figures.plot_roc_curve)

    Parameters
    ----------
    true_one_hot : numpy int array
        [num_samples X num_classes] one-hot true classes
    probabilities : numpy array
        [num_samples X num_classes] predicted probabilities

    Returns
    -------
    float
    """
    if probabilities.shape[1] == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(true_one_hot[:, 1], probabilities[:, 1])
    else:
        false_positive_rate, true_positive_rate, _ = roc_curve(true_one_hot.ravel(), probabilities.ravel())
    return float(auc(false_positive_rate, true_positive_rate))


def performance_metrics(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                        probabilistic: bool = False) -> dict[str, float]:
    """Compute summary statistics of performance

    Residue and matrix values are pooled across all sequences; residues (or
    matrix cells) with missing labels are left out, and each pair of residues
    of a symmetric map is counted once (see labelled_values()).

    Parameters
    ----------
    true_values, predictions : list of numpy arrays
        True and predicted value(s) for each sequence
    task : Task
        The task the network was trained for
    probabilistic : bool, optional
        True if `predictions` are class probabilities (sequence or residue
        classification)

    Returns
    -------
    dict
        Regression: Pearson R and Spearman R (NaN where undefined; see
        correlation()). Classification: Matthews correlation coefficient,
        weighted F1 score and accuracy, plus (if probabilistic) the areas under
        the ROC curve (see roc_area()) and the micro-averaged precision-recall
        curve (NaN if any probability is not a finite number). Values are
        rounded to METRIC_DECIMALS places.
    """

    true_flat, predicted_flat = labelled_values(true_values, predictions, task, probabilistic)
    metrics: dict[str, Any] = {}

    if not task.is_classification:
        metrics['Pearson R'] = correlation(pearsonr, true_flat, predicted_flat)
        metrics['Spearman R'] = correlation(spearmanr, true_flat, predicted_flat)

    else:
        if probabilistic:
            probabilities = predicted_flat
            true_one_hot = one_hot_classes(true_flat, task.num_classes)
            if np.all(np.isfinite(probabilities)):
                metrics['Area under Precision-Recall curve'] = average_precision_score(true_one_hot, probabilities,
                                                                                       average='micro')
                metrics['Area under ROC'] = roc_area(true_one_hot, probabilities)
            else:
                metrics['Area under Precision-Recall curve'] = metrics['Area under ROC'] = math.nan
            predicted_flat = np.argmax(probabilities, axis=1)

        metrics['Matthews Correlation Coef'] = matthews_corrcoef(true_flat, predicted_flat)
        metrics['F1 Score'] = f1_score(true_flat, predicted_flat, average='weighted')
        metrics['Accuracy'] = accuracy_score(true_flat, predicted_flat)

    return {name: round(float(value), METRIC_DECIMALS) for name, value in metrics.items()}


def uncertainty_metrics(true_values: Sequence[np.ndarray], sds: Sequence[np.ndarray],
                        predictions: Sequence[np.ndarray], model_error: float) -> dict[str, float]:
    """How well the error model fits the test set, for regression with uncertainties

    Each value's total SD is sqrt(SD^2 + s^2), its own measurement SD combined
    with the model error s (see parrot/uncertainties.py). If the error model
    is right, the residuals divided by their total SDs are standard Gaussian.
    Residue values are pooled across all sequences, leaving out missing ones.

    Parameters
    ----------
    true_values, sds, predictions : list of numpy arrays
        The true value(s), their SDs and the predicted value(s) of each sequence
    model_error : float
        The network's model error s

    Returns
    -------
    dict
        'Reduced chi-squared': the mean squared residual in units of total SD
        (about 1 if the error model fits; above 1, the errors are larger than
        it says). 'Within 1 SD' and 'Within 2 SD': the fraction of values within
        1 and 2 total SDs of their prediction (about 0.68 and 0.95 if it fits).
        'Noise ceiling R': the highest Pearson R any model could reach on these
        values given their measurement errors, sqrt(1 - mean(SD^2) / var(values))
        (assuming the errors are independent of the true values; 0 if the
        measurement errors account for all the spread of the values, NaN for
        fewer than two values or values that are all the same). Values are
        rounded to METRIC_DECIMALS places, and are NaN if any prediction is not
        finite.
    """
    values, value_sds, predicted = (flatten(arrays) for arrays in (true_values, sds, predictions))
    labelled = ~np.isnan(values)
    values, value_sds, predicted = values[labelled], value_sds[labelled], predicted[labelled]

    metrics = {}
    if np.all(np.isfinite(predicted)):
        z_scores = (values - predicted) / np.sqrt(np.maximum(value_sds ** 2 + model_error ** 2, MIN_VARIANCE))
        metrics['Reduced chi-squared'] = float(np.mean(z_scores ** 2))
        for count in COVERAGE_SDS:
            metrics[f'Within {count} SD'] = float(np.mean(np.abs(z_scores) <= count))
    else:
        metrics['Reduced chi-squared'] = math.nan
        metrics.update({f'Within {count} SD': math.nan for count in COVERAGE_SDS})

    spread = np.var(values) if len(values) >= MIN_CORRELATION_VALUES else 0.0
    metrics['Noise ceiling R'] = (math.sqrt(max(0.0, 1 - float(np.mean(value_sds ** 2)) / spread)) if spread > 0
                                  else math.nan)
    return {name: round(value, METRIC_DECIMALS) for name, value in metrics.items()}
