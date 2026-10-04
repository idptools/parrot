"""
Figures describing training and test-set performance.

Every figure is drawn in one plain style: Arial text (saved so it stays
editable in vector-graphics editors such as Illustrator), solid lines, the
xmgrace colour order (black, red, blue, green, purple), no grid lines and no
titles. The style is applied to each figure as it is drawn, so it never changes
matplotlib's settings for other code.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import contextlib
from collections.abc import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from parrot.metrics import confusion_counts, labelled_values, one_hot_classes
from parrot.tasks import Task

# Line colours in order: the xmgrace order for the first five, then more distinct colours for figures
# with more lines (e.g. ROC curves of many classes)
LINE_COLOURS = ('black', 'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'olive')

# Resolution of saved (bitmap) figures, in dots per inch
FIGURE_DPI = 300

# Width and height of each figure, in inches
FIGURE_SIZE = (4.0, 4.0)

# Line widths, in points: a figure's main curve, and thinner reference lines (e.g. y = x)
MAIN_LINE_WIDTH = 1.5
REFERENCE_LINE_WIDTH = 0.75

# Area of each point in a scatterplot, in points^2
SCATTER_POINT_SIZE = 6

# matplotlib settings applied to every figure. Arial is used if it is installed, and otherwise the closest
# sans-serif font matplotlib has. Font type 42 (TrueType) keeps text as editable text in PDF and PostScript
# files, and 'none' keeps it as text in SVG files.
FIGURE_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.prop_cycle': cycler(color=LINE_COLOURS),
    'lines.linestyle': '-',
    'lines.linewidth': MAIN_LINE_WIDTH,
    'legend.frameon': False,
}


# .................................................................
# Drawing and saving

@contextlib.contextmanager
def figure_style() -> Iterator[None]:
    """Apply FIGURE_STYLE to the figures drawn inside a `with` block (and only to those)."""
    with plt.rc_context(FIGURE_STYLE):
        yield


def save_figure(fig: Figure, output_path: str) -> None:
    """Save a figure and close it, so figures never accumulate in memory.

    Parameters
    ----------
    fig : matplotlib Figure
    output_path : str
        File to save to; the extension sets the format (e.g. .png, .pdf)
    """
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


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
        Prediction(s) for each test sequence (see prediction.output_to_prediction)
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


# .................................................................
# The figures

def plot_training_loss(train_losses: Sequence[float], val_losses: Sequence[float], output_path: str) -> None:
    """Plot training (black) and validation (red) loss per epoch

    Parameters
    ----------
    train_losses, val_losses : list of float
        Average loss per sequence for each epoch
    output_path : str
        File to save the figure to
    """

    epochs = np.arange(1, len(train_losses) + 1)
    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(epochs, train_losses, label='Training')
        ax.plot(epochs, val_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (average per sequence)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        save_figure(fig, output_path)


def regression_scatterplot(true_values: np.ndarray, predicted_values: np.ndarray, output_path: str) -> None:
    """Scatterplot of predicted against true values, on square axes with the line y = x

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired true and predicted values (residue values pooled across sequences),
        in the units of the data
    output_path : str
        File to save the figure to
    """

    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.scatter(true_values, predicted_values, s=SCATTER_POINT_SIZE, color=LINE_COLOURS[0], linewidths=0)

        # Use matplotlib's autoscaled limits (which pad around the data correctly for both positive and
        # negative values) and make the axes square, so y = x is the diagonal
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        limits = (min(left, bottom), max(right, top))
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        ax.plot(limits, limits, color=LINE_COLOURS[0], linewidth=REFERENCE_LINE_WIDTH, zorder=0)

        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        save_figure(fig, output_path)


def plot_roc_curve(true_classes: np.ndarray, class_probabilities: np.ndarray, output_path: str) -> None:
    """ROC curve(s) for a probabilistic classifier

    For two classes a single curve is drawn (both classes give the same curve);
    for more classes, the micro-average (black) and one curve per class. The
    thin diagonal is a classifier that guesses at random.

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
    true_one_hot = one_hot_classes(true_classes, num_classes)

    # (label, true one-hot values, probabilities) of each curve
    if num_classes > 2:
        curves = [('Average', true_one_hot.ravel(), class_probabilities.ravel())]
        curves += [(f'Class {c}', true_one_hot[:, c], class_probabilities[:, c]) for c in range(num_classes)]
    else:
        curves = [('Class 1', true_one_hot[:, 1], class_probabilities[:, 1])]

    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for label, truth, probabilities in curves:
            false_positive_rate, true_positive_rate, _ = roc_curve(truth, probabilities)
            area = auc(false_positive_rate, true_positive_rate)
            ax.plot(false_positive_rate, true_positive_rate, label=f'{label} (area {area:.2f})')

        ax.plot([0, 1], [0, 1], color=LINE_COLOURS[0], linewidth=REFERENCE_LINE_WIDTH, zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(loc='lower right')
        save_figure(fig, output_path)


def plot_precision_recall_curve(true_classes: np.ndarray, class_probabilities: np.ndarray, output_path: str) -> None:
    """Precision-recall curves for a probabilistic classifier: the micro-average (black) and one per class

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
    true_one_hot = one_hot_classes(true_classes, num_classes)

    # (label, true one-hot values, probabilities) of each curve
    curves = [('Average', true_one_hot.ravel(), class_probabilities.ravel())]
    curves += [(f'Class {c}', true_one_hot[:, c], class_probabilities[:, c]) for c in range(num_classes)]

    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for label, truth, probabilities in curves:
            precision, recall, _ = precision_recall_curve(truth, probabilities)
            area = average_precision_score(truth, probabilities)
            ax.plot(recall, precision, label=f'{label} (area {area:.2f})')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower left')
        save_figure(fig, output_path)


def plot_confusion_matrix(true_classes: np.ndarray, predicted_classes: np.ndarray, num_classes: int,
                          output_path: str) -> None:
    """Confusion matrix: the number of samples with each predicted (row) and true (column) class

    Parameters
    ----------
    true_classes, predicted_classes : numpy int arrays
        Paired true and predicted classes (residue classes pooled across sequences)
    num_classes : int
        Number of classes
    output_path : str
        File to save the figure to
    """

    counts = confusion_counts(true_classes, predicted_classes, num_classes)

    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        image = ax.imshow(counts, cmap='Blues')
        fig.colorbar(image, ax=ax, label='Number of samples')

        # Write each count in its cell, in white on the darker half of the colour scale so it stays readable
        darkest = counts.max() if counts.max() > 0 else 1
        for row in range(num_classes):
            for column in range(num_classes):
                colour = 'white' if counts[row, column] > darkest / 2 else 'black'
                ax.text(column, row, f'{counts[row, column]:g}', ha='center', va='center', color=colour)

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xlabel('True class')
        ax.set_ylabel('Predicted class')
        save_figure(fig, output_path)
