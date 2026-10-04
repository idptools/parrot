"""
Reporting how a trained network performs on the held-out test set: prediction
files, performance-metric files and figures.

All functions here take plain numpy arrays (or the LabeledPredictions produced
by train_network.predict_labeled), so they work the same for every architecture.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, auc, average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve)

from parrot.process_input_data import missing_label_mask
from parrot.tasks import Task
from parrot.train_network import LabeledPredictions, output_to_prediction

# Number of decimal places used when writing class probabilities
PROBABILITY_DECIMALS = 4

# Number of decimal places used when writing performance metrics
METRIC_DECIMALS = 3


# .................................................................
# Top-level report

def report_test_results(labeled: LabeledPredictions, task: Task, output_prefix: str, probabilistic: bool = False,
                        include_figs: bool = False, write_stats: bool = True) -> None:
    """Write everything describing performance on the test set

    Always writes "<output_prefix>_predictions.tsv". Also writes
    "<output_prefix>_performance_stats.txt" if `write_stats`, and the figures
    appropriate to the task if `include_figs`.

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
    """

    predictions = [output_to_prediction(output, task, probabilistic) for output in labeled.outputs]
    decimals = PROBABILITY_DECIMALS if probabilistic else None

    write_predictions(labeled.names, labeled.sequences, labeled.true_values, predictions, task,
                      output_prefix + '_predictions.tsv', decimals=decimals)

    if write_stats:
        metrics = performance_metrics(labeled.true_values, predictions, task, probabilistic)
        write_performance_metrics(metrics, output_prefix + '_performance_stats.txt')

    if include_figs:
        plot_test_results(labeled.true_values, predictions, task, output_prefix, probabilistic)


def plot_test_results(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                      output_prefix: str, probabilistic: bool = False) -> None:
    """Save the figure(s) appropriate to the task

    Regression gives a scatterplot ("_seq_scatterplot.png" or
    "_res_scatterplot.png"), classification a confusion matrix ("_seq_CM.png"
    or "_res_CM.png"), and probabilistic classification ROC and
    precision-recall curves ("_ROC_curve.png" and "_PR_curve.png").

    Parameters
    ----------
    true_values : list of numpy arrays
        True value(s) for each test sequence
    predictions : list of numpy arrays
        Prediction(s) for each test sequence (see train_network.output_to_prediction)
    task : Task
        The task the network was trained for
    output_prefix : str
        Path and filename prefix for the figures
    probabilistic : bool, optional
        True if `predictions` are class probabilities
    """

    data_label = 'res' if task.per_residue else 'seq'
    true_flat, predicted_flat = labelled_values(true_values, predictions, task, probabilistic)

    if not task.is_classification:
        regression_scatterplot(true_flat, predicted_flat, f'{output_prefix}_{data_label}_scatterplot.png')
    elif probabilistic:
        plot_roc_curve(true_flat, predicted_flat, output_prefix + '_ROC_curve.png')
        plot_precision_recall_curve(true_flat, predicted_flat, output_prefix + '_PR_curve.png')
    else:
        plot_confusion_matrix(true_flat, predicted_flat, task.num_classes, f'{output_prefix}_{data_label}_CM.png')


def labelled_values(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                    probabilistic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pool true and predicted values across sequences, keeping only labelled ones

    Residues with missing labels are dropped, so metrics and figures describe
    only residues that have a true value.

    Parameters
    ----------
    true_values, predictions : list of numpy arrays
        True and predicted value(s) for each sequence
    task : Task
        The task the network was trained for
    probabilistic : bool, optional
        True if `predictions` are class probabilities (sequence classification only)

    Returns
    -------
    numpy array
        The labelled true values, pooled into one 1D array
    numpy array
        The matching predictions: a 1D array, or [num_sequences X num_classes]
        class probabilities if `probabilistic`
    """

    if probabilistic:
        return flatten(true_values), np.vstack(predictions)

    true_flat = flatten(true_values)
    predicted_flat = flatten(predictions)
    labelled = ~missing_label_mask(true_flat, task)
    return true_flat[labelled], predicted_flat[labelled]


def flatten(values: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-sequence values (scalars or per-residue arrays) into one 1D array."""
    return np.concatenate([np.atleast_1d(value) for value in values])


# .................................................................
# Prediction and metric files

def format_values(values: np.ndarray, decimals: int | None = None) -> str:
    """Format a value, or an array of values, as whitespace-separated text

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


def performance_metrics(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                        probabilistic: bool = False) -> dict[str, float]:
    """Compute summary statistics of test-set performance

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
        precision-recall curves. Values are rounded to 3 decimal places.
    """

    true_flat, predicted_flat = labelled_values(true_values, predictions, task, probabilistic)
    metrics: dict[str, Any] = {}

    if not task.is_classification:
        metrics['Pearson R'] = pearsonr(true_flat, predicted_flat)[0]
        metrics['Spearman R'] = spearmanr(true_flat, predicted_flat)[0]

    else:
        if probabilistic:
            probabilities = predicted_flat
            true_one_hot = np.eye(task.num_classes, dtype=int)[true_flat.astype(int)]
            metrics['Area under Precision-Recall curve'] = average_precision_score(true_one_hot, probabilities,
                                                                                   average='micro')
            fpr, tpr, _ = roc_curve(true_one_hot.ravel(), probabilities.ravel())
            metrics['Area under ROC'] = auc(fpr, tpr)
            predicted_flat = np.argmax(probabilities, axis=1)

        metrics['Matthews Correlation Coef'] = matthews_corrcoef(true_flat, predicted_flat)
        metrics['F1 Score'] = f1_score(true_flat, predicted_flat, average='weighted')
        metrics['Accuracy'] = accuracy_score(true_flat, predicted_flat)

    return {name: round(float(value), METRIC_DECIMALS) for name, value in metrics.items()}


def write_performance_metrics(metrics: dict[str, float], output_path: str) -> None:
    """Write performance metrics as "<name> : <value>" lines

    Parameters
    ----------
    metrics : dict
        Metric names and values (see performance_metrics())
    output_path : str
        File to write
    """

    with open(output_path, 'w') as fh:
        for name, value in metrics.items():
            fh.write('%s : %.3f\n' % (name, value))


# .................................................................
# Figures

def plot_training_loss(train_losses: Sequence[float], val_losses: Sequence[float], output_path: str) -> None:
    """Plot training and validation loss per epoch

    Parameters
    ----------
    train_losses, val_losses : list of float
        Average loss per sequence for each epoch
    output_path : str
        File to save the figure to
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    num_epochs = len(train_losses)
    epochs = np.arange(1, num_epochs + 1)
    training_line, = ax.plot(epochs, train_losses, label='Train')
    validation_line, = ax.plot(epochs, val_losses, label='Val')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg loss")
    ax.set_title("Training and testing loss per epoch")
    ax.legend(handles=[training_line, validation_line], fontsize=14, facecolor='gainsboro', edgecolor='slategray')

    if num_epochs < 21:
        ax.set_xticks(np.arange(2, num_epochs + 1, 2))
    elif num_epochs < 66:
        ax.set_xticks(np.arange(5, num_epochs + 1, 5))
    elif num_epochs < 151:
        ax.set_xticks(np.arange(10, num_epochs + 1, 10))
    else:
        ax.set_xticks(np.arange(50, num_epochs + 1, 50))

    fig.savefig(output_path)
    plt.close(fig)


def regression_scatterplot(true_values: np.ndarray, predicted_values: np.ndarray, output_path: str) -> None:
    """Scatterplot of predicted against true values, with R^2 in the title

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired true and predicted values (residue values pooled across sequences)
    output_path : str
        File to save the figure to
    """

    fig, ax = plt.subplots()
    ax.scatter(true_values, predicted_values, s=6)

    # Use matplotlib's autoscaled limits (which pad around the data correctly for
    # both positive and negative values) and make the axes square
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    edge_vals = (min(left, bottom), max(right, top))
    ax.set_xlim(edge_vals)
    ax.set_ylim(edge_vals)
    ax.plot(edge_vals, edge_vals, 'k--')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

    # A regression line needs more than one distinct true value
    r_value = linregress(true_values, predicted_values).rvalue if np.ptp(true_values) > 0 else np.nan
    ax.set_title('Testing accuracy: R^2=%.3f' % (r_value**2))
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(true_classes: np.ndarray, class_probabilities: np.ndarray, output_path: str) -> None:
    """ROC curve(s) for a probabilistic classifier

    For two classes a single curve is drawn (both classes give the same curve);
    for more classes, one curve per class plus the micro-average.

    Parameters
    ----------
    true_classes : numpy int array
        True class of each sample
    class_probabilities : numpy array
        [num_samples X num_classes] predicted probabilities
    output_path : str
        File to save the figure to
    """

    num_classes = class_probabilities.shape[1]
    true_one_hot = np.eye(num_classes, dtype=int)[true_classes.astype(int)]

    fig = plt.figure()
    if num_classes > 2:
        fpr, tpr, _ = roc_curve(true_one_hot.ravel(), class_probabilities.ravel())
        plt.plot(fpr, tpr, label='Average (area = {0:0.2f})'.format(auc(fpr, tpr)),
                 color='deeppink', linestyle=':', linewidth=4)

        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(true_one_hot[:, c], class_probabilities[:, c])
            plt.plot(fpr, tpr, lw=2, label='Class {0} (area = {1:0.2f})'.format(c, auc(fpr, tpr)))
    else:
        fpr, tpr, _ = roc_curve(true_one_hot[:, 1], class_probabilities[:, 1])
        plt.plot(fpr, tpr, lw=2, label='Binary class (area = {0:0.2f})'.format(auc(fpr, tpr)))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", fontsize=8)
    fig.savefig(output_path)
    plt.close(fig)


def plot_precision_recall_curve(true_classes: np.ndarray, class_probabilities: np.ndarray, output_path: str) -> None:
    """Precision-recall curves (one per class plus the micro-average) for a probabilistic classifier

    Parameters
    ----------
    true_classes : numpy int array
        True class of each sample
    class_probabilities : numpy array
        [num_samples X num_classes] predicted probabilities
    output_path : str
        File to save the figure to
    """

    num_classes = class_probabilities.shape[1]
    true_one_hot = np.eye(num_classes, dtype=int)[true_classes.astype(int)]

    fig = plt.figure()
    precision, recall, _ = precision_recall_curve(true_one_hot.ravel(), class_probabilities.ravel())
    average_precision = average_precision_score(true_one_hot, class_probabilities, average="micro")
    plt.plot(recall, precision, color='deeppink', linestyle=':', linewidth=4,
             label='Average (area = {0:0.2f})'.format(average_precision))

    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(true_one_hot[:, c], class_probabilities[:, c])
        average_precision = average_precision_score(true_one_hot[:, c], class_probabilities[:, c])
        plt.plot(recall, precision, lw=2, label='Class {0} (area = {1:0.2f})'.format(c, average_precision))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(true_classes: np.ndarray, predicted_classes: np.ndarray, num_classes: int,
                          output_path: str) -> None:
    """Confusion matrix heatmap: rows are predicted classes, columns true classes

    Parameters
    ----------
    true_classes, predicted_classes : numpy int arrays
        Paired true and predicted classes (residue classes pooled across sequences)
    num_classes : int
        Number of classes
    output_path : str
        File to save the figure to
    """

    counts = np.zeros((num_classes, num_classes))
    for predicted, true in zip(predicted_classes.astype(int), true_classes.astype(int)):
        counts[predicted, true] += 1

    df_cm = pd.DataFrame(counts, range(num_classes), range(num_classes))

    # Scope the seaborn font scaling to this figure so it does not change the
    # global matplotlib style for every plot made afterwards
    with sn.plotting_context(font_scale=1.4):
        fig, ax = plt.subplots()
        sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16}, ax=ax)
        ax.set_xlabel('True labels')
        ax.set_ylabel('Predicted labels')
        ax.set_title('Test set confusion matrix')
        fig.tight_layout()
        fig.savefig(output_path)
    plt.close(fig)
