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
import warnings
from collections.abc import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter
from scipy.stats import pearsonr
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from parrot.metrics import confusion_counts, labelled_values, one_hot_classes
from parrot.process_input_data import missing_label_mask
from parrot.tasks import IGNORED_CLASS_LABEL, Task

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

# Scatterplots of matrix data with more cells than this (a test set of 100 sequences of 300 residues has 9 million)
# show a random sample of this many, drawn with a fixed seed so the same results always give the same figure
MAX_SCATTER_POINTS = 100_000
SCATTER_SAMPLE_SEED = 0

# Margin around the values on each axis of a scatter or density plot, as a fraction of their range (as matplotlib
# pads axes by default), and, in the units of the data, when every value is the same, so the axes still have a range
PLOT_MARGIN = 0.05
EQUAL_VALUES_MARGIN = 0.5

# Colour map for regression values: density plots here, and matrix heatmaps (see matrices/figures.py)
REGRESSION_COLOUR_MAP = 'viridis'

# Density plots: bins along each axis of the 2D histogram. Counts are coloured on a log scale, because a few bins near
# y = x usually hold most of the values, unless no bin holds LOG_SCALE_MIN_COUNT values (a small test set), when a
# log scale has no whole decade to label
DENSITY_BINS = 100
LOG_SCALE_MIN_COUNT = 10

# What each value of a test-set plot is, for each kind of data (for the density plot's colour bar)
VALUE_NAMES = {'seq': 'Sequences', 'res': 'Residues', 'matrix': 'Cells'}

# Where in-plot labels (such as Pearson r) go, as a fraction of the axes from the bottom left: the top left
ANNOTATION_POSITION = (0.04, 0.96)

# Residue profiles: how many test sequences to show; the fewest labelled residues each should have (if enough test
# sequences have that many), so that each profile shows something; the size of each panel, in inches; and the size of
# the points marking labelled residues with no labelled neighbour (which a line cannot show), in points
NUM_PROFILE_EXAMPLES = 6
MIN_PROFILE_RESIDUES = 30
PROFILE_PANEL_SIZE = (7.0, 1.6)
ISOLATED_RESIDUE_MARKER_SIZE = 3.0

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
                      output_prefix: str, probabilistic: bool = False, pearson: float | None = None) -> list[str]:
    """Save the figure(s) appropriate to the task

    Regression gives a scatterplot and a density plot of predicted against
    true values, both labelled with Pearson's r over every value
    ("_seq_scatterplot.png" and "_seq_density.png", and likewise "_res_" and
    "_matrix_"); classification a confusion matrix ("_seq_CM.png", "_res_CM.png"
    or "_matrix_CM.png"); and probabilistic classification ROC and
    precision-recall curves ("_ROC_curve.png" and "_PR_curve.png"). Residue and
    matrix values are pooled over every sequence (matrix scatterplots show at
    most MAX_SCATTER_POINTS cells, see scatter_sample(); the density plot
    always shows every value). Predictions that are not finite (NaN or
    infinite, e.g. from a network whose training diverged) cannot be plotted,
    so they are left out of the regression figures, with a warning.

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
    pearson : float, optional
        Regression: Pearson's r to write on the plots, e.g. the one in the
        performance metrics, so that the figures and statistics agree
        (default: computed from the plotted values)

    Returns
    -------
    list of str
        The files written (none for regression, with a warning, if no
        prediction is finite)
    """

    data_label = 'matrix' if task.is_matrix else 'res' if task.per_residue else 'seq'
    true_flat, predicted_flat = labelled_values(true_values, predictions, task, probabilistic)

    if not task.is_classification:
        true_flat, predicted_flat = finite_pairs(true_flat, predicted_flat)
        if len(true_flat) == 0:
            warnings.warn('No test-set prediction is finite, so the regression figures are not drawn')
            return []
        # Both plots share axes and r, which covers every value (the scatterplot may show a sample)
        if pearson is None:
            pearson = pearson_r(true_flat, predicted_flat)
        limits = square_limits(true_flat, predicted_flat)
        scatter_true, scatter_predicted = (scatter_sample(true_flat, predicted_flat) if task.is_matrix
                                           else (true_flat, predicted_flat))
        scatterplot_path = f'{output_prefix}_{data_label}_scatterplot.png'
        density_path = f'{output_prefix}_{data_label}_density.png'
        regression_scatterplot(scatter_true, scatter_predicted, scatterplot_path, limits, pearson)
        regression_density_plot(true_flat, predicted_flat, density_path, VALUE_NAMES[data_label], limits, pearson)
        return [scatterplot_path, density_path]

    if probabilistic:
        roc_path, precision_recall_path = output_prefix + '_ROC_curve.png', output_prefix + '_PR_curve.png'
        plot_roc_curve(true_flat, predicted_flat, roc_path)
        plot_precision_recall_curve(true_flat, predicted_flat, precision_recall_path)
        return [roc_path, precision_recall_path]

    confusion_matrix_path = f'{output_prefix}_{data_label}_CM.png'
    plot_confusion_matrix(true_flat, predicted_flat, task.num_classes, confusion_matrix_path)
    return [confusion_matrix_path]


def scatter_sample(true_values: np.ndarray, predicted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The values a scatterplot shows: all of them, or (with a warning) a random sample of MAX_SCATTER_POINTS if there
    are more

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired values

    Returns
    -------
    (numpy array, numpy array)
    """
    if len(true_values) <= MAX_SCATTER_POINTS:
        return true_values, predicted_values
    warnings.warn(f'The scatterplot shows a random sample of {MAX_SCATTER_POINTS:,} of the {len(true_values):,} values; '
                  'the performance metrics use every value')
    chosen = np.random.default_rng(SCATTER_SAMPLE_SEED).choice(len(true_values), MAX_SCATTER_POINTS, replace=False)
    return true_values[chosen], predicted_values[chosen]


def finite_pairs(true_values: np.ndarray, predicted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of values that can be plotted (both finite), warning how many are left out if any are

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired values

    Returns
    -------
    (numpy array, numpy array)
        The pairs in which neither value is NaN or infinite
    """
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    if not finite.all():
        warnings.warn(f'{int((~finite).sum()):,} of {len(finite):,} test-set predictions are not finite (NaN or '
                      'infinite), so the regression figures leave them out; the performance metrics include them')
    return true_values[finite], predicted_values[finite]


def pearson_r(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Pearson's correlation coefficient r between paired values, NaN where it is undefined (as in the metrics)

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired values

    Returns
    -------
    float
        NaN if there are fewer than two values or either set is constant
    """
    if len(true_values) < 2 or np.ptp(true_values) == 0 or np.ptp(predicted_values) == 0:
        return np.nan
    return float(pearsonr(true_values, predicted_values).statistic)


def square_limits(true_values: np.ndarray, predicted_values: np.ndarray) -> tuple[float, float]:
    """Axis limits that hold every true and predicted value, the same for both axes so y = x is the diagonal

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Finite values (see finite_pairs()), at least one between them

    Returns
    -------
    (float, float)
        The lowest and highest value, each padded by PLOT_MARGIN of the range
        (by EQUAL_VALUES_MARGIN if all the values are equal)

    Raises
    ------
    ValueError
        If there are no values, or any is NaN or infinite
    """
    values = np.concatenate([np.ravel(true_values), np.ravel(predicted_values)])
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Axis limits need at least one value, and only finite ones')
    low, high = float(values.min()), float(values.max())
    margin = PLOT_MARGIN * (high - low) if high > low else EQUAL_VALUES_MARGIN
    return low - margin, high + margin


def annotate(ax: Axes, label: str) -> None:
    """Write a label in the top left of a plot, on a white background so it can be read over the data

    Parameters
    ----------
    ax : matplotlib Axes
        The plot to label
    label : str
        The text to write, e.g. 'Pearson r = 0.735'
    """
    ax.text(*ANNOTATION_POSITION, label, transform=ax.transAxes, ha='left', va='top',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2.0})


def square_axes(ax: Axes, limits: tuple[float, float], pearson: float | None, line_on_top: bool) -> None:
    """Finish a plot of predicted against true values: the line y = x, the same limits on both axes, labels and r

    Parameters
    ----------
    ax : matplotlib Axes
    limits : (float, float)
        Axis limits, the same for both axes
    pearson : float or None
        Pearson's r, written on the plot if given ('nan' if undefined, as in the performance metrics)
    line_on_top : bool
        Draw y = x over the data (over a density plot) rather than behind it (behind a scatterplot's points)
    """
    ax.plot(limits, limits, color=LINE_COLOURS[0], linewidth=REFERENCE_LINE_WIDTH, zorder=3 if line_on_top else 0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal')
    ax.set_xlabel('True value')
    ax.set_ylabel('Predicted value')
    if pearson is not None:
        annotate(ax, f'Pearson r = {pearson:.3f}')


def displayable(values: np.ndarray, task: Task) -> np.ndarray:
    """Values ready to plot: missing labels (NaN, or IGNORED_CLASS_LABEL for classes) become NaN, drawn as gaps

    Parameters
    ----------
    values : numpy array
        True values or classes, of any shape (a residue profile or a matrix)
    task : Task

    Returns
    -------
    numpy float array
    """
    values = np.asarray(values, dtype=float)
    if task.is_classification:
        values = np.where(values == IGNORED_CLASS_LABEL, np.nan, values)
    return values


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


def regression_scatterplot(true_values: np.ndarray, predicted_values: np.ndarray, output_path: str,
                           limits: tuple[float, float] | None = None, pearson: float | None = None) -> None:
    """Scatterplot of predicted against true values, on square axes with the line y = x

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired true and predicted values (residue values pooled across sequences),
        in the units of the data
    output_path : str
        File to save the figure to
    limits : (float, float), optional
        Axis limits, the same for both axes (default: square_limits() of the values)
    pearson : float, optional
        Pearson's r, written on the plot if given
    """

    if limits is None:
        limits = square_limits(true_values, predicted_values)
    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.scatter(true_values, predicted_values, s=SCATTER_POINT_SIZE, color=LINE_COLOURS[0], linewidths=0)
        square_axes(ax, limits, pearson, line_on_top=False)
        save_figure(fig, output_path)


def regression_density_plot(true_values: np.ndarray, predicted_values: np.ndarray, output_path: str,
                            value_name: str = 'Values', limits: tuple[float, float] | None = None,
                            pearson: float | None = None) -> None:
    """Density of predicted against true values: a 2D histogram on square axes, with the line y = x

    Where a scatterplot of many values is a solid block, this shows where
    they lie. Each bin is coloured by how many values it holds, on a log scale
    (or, if no bin holds LOG_SCALE_MIN_COUNT values, a linear one); empty bins
    are left white.

    Parameters
    ----------
    true_values, predicted_values : numpy arrays
        Paired true and predicted values, in the units of the data
    output_path : str
        File to save the figure to
    value_name : str, optional
        What each value is (e.g. 'Residues'), for the colour bar's label
    limits : (float, float), optional
        Axis limits, the same for both axes (default: square_limits() of the values)
    pearson : float, optional
        Pearson's r, written on the plot if given
    """

    if limits is None:
        limits = square_limits(true_values, predicted_values)
    counts, true_edges, predicted_edges = np.histogram2d(true_values, predicted_values, bins=DENSITY_BINS,
                                                         range=(limits, limits))
    log_scale = counts.max() >= LOG_SCALE_MIN_COUNT
    norm = LogNorm(vmin=1, vmax=counts.max()) if log_scale else Normalize(vmin=1, vmax=max(counts.max(), 2))
    with figure_style():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        # histogram2d counts true values along the first axis, so transpose to put them along x
        image = ax.pcolormesh(true_edges, predicted_edges, np.ma.masked_equal(counts.T, 0),
                              cmap=REGRESSION_COLOUR_MAP, norm=norm, rasterized=True)
        square_axes(ax, limits, pearson, line_on_top=True)
        colour_bar = fig.colorbar(image, ax=ax, shrink=0.8, label=f'{value_name} per bin')
        if log_scale:
            colour_bar.ax.yaxis.set_minor_formatter(NullFormatter())    # label powers of ten only
        else:
            colour_bar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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


# .................................................................
# Residue profiles

def predicted_labels(prediction: np.ndarray, task: Task) -> np.ndarray:
    """A residue prediction as one value per residue: for class probabilities, the most likely class

    Parameters
    ----------
    prediction : numpy array
        One sequence's prediction: a value (or class) per residue, or
        [sequence_length X num_classes] class probabilities
    task : Task

    Returns
    -------
    numpy array
        One value (or class) per residue
    """
    if task.is_classification and prediction.ndim == 2:
        return np.argmax(prediction, axis=-1)
    return prediction


def sequence_error(true: np.ndarray, predicted: np.ndarray, task: Task) -> float:
    """How wrong a prediction is over the labelled residues of one sequence

    Parameters
    ----------
    true : numpy array
        The true value (or class) of each residue
    predicted : numpy array
        The predicted value (or class) of each residue
    task : Task

    Returns
    -------
    float
        Mean absolute error (regression) or the fraction of residues wrongly
        classified (classification); inf if no residue has a label, and NaN
        if a prediction is NaN
    """
    labelled = ~missing_label_mask(true, task)
    if not labelled.any():
        return np.inf
    if task.is_classification:
        return float(np.mean(predicted[labelled] != true[labelled]))
    return float(np.mean(np.abs(predicted[labelled] - true[labelled])))


def profile_examples(true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray], task: Task,
                     num_examples: int | None = None) -> list[int]:
    """Which test sequences to show as profiles: a spread from the most to the least accurately predicted

    The sequences are ranked by sequence_error(), and those at evenly spaced
    quantiles (the middle of each of `num_examples` equal parts) are chosen, so
    the examples are typical of the range of accuracy rather than picked for
    being good or bad. Sequences with at least MIN_PROFILE_RESIDUES labelled
    residues are chosen from if there are enough of them, so that a sequence
    is never shown, and ranked, on the strength of a few residues; sequences
    with no labelled residue, or a prediction that is NaN, never are.

    Parameters
    ----------
    true_values, predictions : list of numpy arrays
        True and predicted value (or class) of each residue of each sequence
    task : Task
    num_examples : int, optional
        How many to choose (default NUM_PROFILE_EXAMPLES)

    Returns
    -------
    list of int
        Indices of the chosen sequences, from the most to the least accurately
        predicted (empty if no sequence can be shown)
    """
    if num_examples is None:
        num_examples = NUM_PROFILE_EXAMPLES
    errors = [sequence_error(true, predicted, task) for true, predicted in zip(true_values, predictions)]
    labelled = [index for index, error in enumerate(errors) if np.isfinite(error)]
    long_enough = [index for index in labelled
                   if np.count_nonzero(~missing_label_mask(true_values[index], task)) >= MIN_PROFILE_RESIDUES]
    candidates = sorted(long_enough if len(long_enough) >= num_examples else labelled, key=lambda i: errors[i])
    if len(candidates) <= num_examples:
        return candidates
    positions = ((np.arange(num_examples) + 0.5) / num_examples * len(candidates)).astype(int)
    return [candidates[position] for position in positions]


def plot_residue_profiles(names: Sequence[str], true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                          task: Task, output_path: str) -> list[str]:
    """True (black) and predicted (red) values along a few test sequences, one panel each

    The sequences are chosen by profile_examples(), and shown from the most to
    the least accurately predicted. Each panel's x-axis label names the
    sequence and gives, for regression, Pearson's r between the true and
    predicted values of its residues and their mean absolute error (MAE, in
    the units of the data), or, for classification, its accuracy (labels go
    outside the panels, which the profiles fill). Residues without a label are
    left as gaps in the true values, and a labelled residue with no labelled
    neighbour, which a line cannot show, is drawn as a point.

    Parameters
    ----------
    names : list of str
        Sequence IDs
    true_values : list of numpy arrays
        True value (or class) of each residue of each test sequence
    predictions : list of numpy arrays
        Predicted value of each residue, or class (or class probabilities) for classification
    task : Task
        A residue task
    output_path : str
        File to save the figure to

    Returns
    -------
    list of str
        The file written (none, with a warning, if no sequence can be shown)
    """
    predicted_values = [predicted_labels(prediction, task) for prediction in predictions]
    chosen = profile_examples(true_values, predicted_values, task)
    if not chosen:
        warnings.warn('No test sequence has a labelled residue and a finite prediction, so no profiles are drawn')
        return []

    with figure_style():
        fig, axes = plt.subplots(len(chosen), 1, squeeze=False, sharey=True,
                                 figsize=(PROFILE_PANEL_SIZE[0], PROFILE_PANEL_SIZE[1] * len(chosen)))
        for row, index in enumerate(chosen):
            ax = axes[row, 0]
            true = displayable(true_values[index], task)
            predicted = np.asarray(predicted_values[index], dtype=float)
            positions = np.arange(1, len(true) + 1)
            drawstyle = 'steps-mid' if task.is_classification else 'default'
            ax.plot(positions, true, color=LINE_COLOURS[0], drawstyle=drawstyle, label='True')
            ax.plot(positions, predicted, color=LINE_COLOURS[1], drawstyle=drawstyle, label='Predicted')
            isolated = isolated_residues(true)
            if isolated.any():
                ax.plot(positions[isolated], true[isolated], color=LINE_COLOURS[0], linestyle='none', marker='o',
                        markersize=ISOLATED_RESIDUE_MARKER_SIZE)

            ax.set_xlim(0.5, len(true) + 0.5)
            ax.set_xlabel(f'Residue of {names[index] or f"test{index}"}{profile_score(true, predicted, task)}')
            ax.set_ylabel('Class' if task.is_classification else 'Value')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if task.is_classification:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if row == 0:
                ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
        save_figure(fig, output_path)
    return [output_path]


def isolated_residues(values: np.ndarray) -> np.ndarray:
    """Which residues have a value but no neighbour with one, so a line through the values cannot show them

    Parameters
    ----------
    values : numpy float array
        One value per residue, NaN where there is none

    Returns
    -------
    numpy bool array
    """
    present = ~np.isnan(values)
    left = np.concatenate([[False], present[:-1]])
    right = np.concatenate([present[1:], [False]])
    isolated: np.ndarray = present & ~left & ~right
    return isolated


def profile_score(true: np.ndarray, predicted: np.ndarray, task: Task) -> str:
    """How well one sequence was predicted, for its profile's axis label

    Parameters
    ----------
    true : numpy float array
        True value (or class) of each residue, NaN where there is no label
    predicted : numpy float array
        Predicted value (or class) of each residue
    task : Task

    Returns
    -------
    str
        ' (Pearson r = ..., MAE = ...)' for regression (leaving out r if it is
        undefined, e.g. when the true values are all the same), or
        ' (accuracy ...)' for classification
    """
    labelled = ~np.isnan(true)
    if task.is_classification:
        return f' (accuracy {np.mean(predicted[labelled] == true[labelled]):.2f})'
    mae = f'MAE = {np.mean(np.abs(predicted[labelled] - true[labelled])):.3g}'
    pearson = pearson_r(true[labelled], predicted[labelled])
    return f' ({mae})' if np.isnan(pearson) else f' (Pearson r = {pearson:.3f}, {mae})'

