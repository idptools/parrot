"""
Figures for matrix predictions: heatmaps of the true and predicted matrices of
a few test sequences, in PARROT's figure style (see parrot/figures.py).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from parrot import figures
from parrot.tasks import Task

# How many test sequences the example heatmaps show
NUM_EXAMPLES = 3

# Size of each heatmap panel, in inches
PANEL_SIZE_INCHES = 3.0

# Colour map of classification heatmaps (distinct colours); regression heatmaps use figures.REGRESSION_COLOUR_MAP
CLASSIFICATION_COLOUR_MAP = 'tab10'


def example_indices(lengths: Sequence[int], num_examples: int = NUM_EXAMPLES) -> list[int]:
    """Which sequences to show: the shortest, the longest and one of middling length (or all, if there are few)

    Parameters
    ----------
    lengths : list of int
        Length of each test sequence
    num_examples : int, optional
        How many to choose (default NUM_EXAMPLES), spread evenly through the
        sequences in order of length, from the shortest to the longest

    Returns
    -------
    list of int
        Indices into `lengths`, in order of length
    """
    by_length = sorted(range(len(lengths)), key=lambda index: lengths[index])
    if len(by_length) <= num_examples:
        return by_length
    positions = np.linspace(0, len(by_length) - 1, num_examples).round().astype(int)
    return [by_length[position] for position in positions]


def colour_range(true: np.ndarray, predicted: np.ndarray, task: Task) -> tuple[float, float]:
    """The lowest and highest value one row of heatmaps shows: every class, or the range of both matrices.

    Parameters
    ----------
    true, predicted : numpy arrays
        Displayable matrices (see figures.displayable())
    task : Task

    Returns
    -------
    (float, float)
    """
    if task.is_classification:
        return 0.0, float(task.num_classes - 1)
    both = np.concatenate([true.ravel(), predicted.ravel()])
    return float(np.nanmin(both)), float(np.nanmax(both))


def plot_matrix_examples(names: Sequence[str], true_values: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                         task: Task, output_path: str) -> None:
    """Heatmaps of the true (left) and predicted (right) matrices of a few test sequences, one row each

    The two panels of a row share one colour scale, so they can be compared
    directly. The axis labels say which sequence (by ID) and which panel is
    which, as PARROT's figures have no titles.

    Parameters
    ----------
    names : list of str
        Sequence IDs
    true_values, predictions : list of numpy arrays
        True and predicted matrix of each sequence (predicted classes, for classification)
    task : Task
        A matrix task
    output_path : str
        File to save the figure to
    """
    chosen = example_indices([len(matrix) for matrix in true_values])
    colour_map = CLASSIFICATION_COLOUR_MAP if task.is_classification else figures.REGRESSION_COLOUR_MAP
    columns_name = 'residue' if task.width is None else 'column'

    with figures.figure_style():
        fig, axes = plt.subplots(len(chosen), 2, squeeze=False,
                                 figsize=(2 * PANEL_SIZE_INCHES, len(chosen) * PANEL_SIZE_INCHES))
        for row, index in enumerate(chosen):
            true = figures.displayable(true_values[index], task)
            predicted = figures.displayable(predictions[index], task)
            low, high = colour_range(true, predicted, task)
            name = names[index] or f'test{index}'

            for column, (values, which) in enumerate(((true, 'true'), (predicted, 'predicted'))):
                ax = axes[row, column]
                image = ax.imshow(values, cmap=colour_map, vmin=low, vmax=high, interpolation='nearest')
                ax.set_xlabel(f'{columns_name.capitalize()} ({which})')
                ax.set_ylabel(f'Residue of {name}')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.colorbar(image, ax=axes[row, 1])
        figures.save_figure(fig, output_path)
