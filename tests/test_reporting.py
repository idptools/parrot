"""
Tests for test-set reporting: figures (parrot.figures), performance metrics
(parrot.metrics) and prediction and metric files (parrot.reporting).
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from parrot import figures, metrics, reporting
from parrot import process_input_data as pid
from parrot.tasks import Task
from parrot.prediction import LabeledPredictions


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every figure reporting closes so its contents can be inspected afterwards."""
    closed_figures = []
    real_close = plt.close

    def capture_then_close(fig: Any = None) -> None:
        closed_figures.append(fig)
        real_close(fig)

    monkeypatch.setattr(figures.plt, 'close', capture_then_close)
    return closed_figures


def read_metrics(path: str) -> dict[str, float]:
    """Parse a '<name> : <value>' performance-stats file into a dictionary."""
    values = {}
    with open(path) as fh:
        for line in fh:
            name, value = line.rsplit(':', 1)
            values[name.strip()] = float(value)
    return values


def read_lines(path: str) -> list[str]:
    with open(path) as fh:
        return [line.rstrip('\n') for line in fh]


# ..........................................................................................
# Figures

def test_training_loss_plot(tmp_path):
    path = str(tmp_path / 'loss.png')
    figures.plot_training_loss([3.0, 2.0, 1.5, 1.2], [3.5, 2.5, 2.0, 1.9], path)

    assert os.path.isfile(path)
    assert plt.get_fignums() == []


def test_regression_scatterplot_covers_negative_values(tmp_path, captured_figures):
    """Axis limits must contain all points, including negative ones"""
    path = str(tmp_path / 'scatter.png')
    figures.regression_scatterplot(np.array([-5.0, -2.0, 1.0, 3.0]), np.array([-4.0, -2.5, 0.5, 2.0]), path)

    assert os.path.isfile(path)
    ax = captured_figures[-1].axes[0]
    low, high = ax.get_xlim()
    assert low < -5.0 and high > 3.0
    assert ax.get_xlim() == ax.get_ylim()


def test_regression_scatterplot_with_identical_true_values(tmp_path, captured_figures):
    """The figure is still made when every true value is the same"""
    path = str(tmp_path / 'scatter.png')
    figures.regression_scatterplot(np.full(4, 2.0), np.array([1.0, 2.0, 2.5, 3.0]), path)

    assert os.path.isfile(path)
    assert len(captured_figures[-1].axes[0].collections) == 1


def figure_texts(fig: Any) -> list[str]:
    """Every piece of text drawn in a figure's axes: labels and annotations."""
    texts = []
    for ax in fig.axes:
        texts += [ax.get_xlabel(), ax.get_ylabel(), *[text.get_text() for text in ax.texts]]
    return texts


@pytest.mark.parametrize('true, predicted, expected', [
    ([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], 1.0),
    ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0),
    ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], np.nan),   # undefined when either set is constant, as in the metrics
    ([1.0], [1.0], np.nan),
])
def test_pearson_r(true, predicted, expected):
    pearson = figures.pearson_r(np.array(true), np.array(predicted))
    assert np.isnan(pearson) if np.isnan(expected) else pearson == pytest.approx(expected, abs=1e-3)


def test_square_limits_hold_every_value_with_a_margin():
    assert figures.square_limits(np.array([0.0, 10.0]), np.array([-10.0, 5.0])) == (-11.0, 11.0)
    assert figures.square_limits(np.array([2.0, 2.0]), np.array([2.0])) == (1.5, 2.5)


@pytest.mark.parametrize('predicted', [[np.nan, 0.5], [np.inf, 0.5], []], ids=['nan', 'inf', 'empty'])
def test_square_limits_refuse_values_that_cannot_be_plotted(predicted):
    true = np.array([0.0, 1.0]) if predicted else np.array([])
    with pytest.raises(ValueError, match='only finite ones'):
        figures.square_limits(true, np.array(predicted))


@pytest.mark.parametrize('bad_value', [np.nan, np.inf], ids=['nan', 'inf'])
def test_predictions_that_are_not_finite_are_left_out_of_the_figures_with_a_warning(tmp_path, captured_figures,
                                                                                   bad_value):
    true = [np.array([0.0, 0.5, 1.0, 0.2])]
    predicted = [np.array([bad_value, 0.4, 1.1, 0.3])]
    with pytest.warns(UserWarning, match='1 of 4 test-set predictions are not finite'):
        written = figures.plot_test_results(true, predicted, Task('residues', 1), str(tmp_path / 'net'))

    assert len(written) == 2
    scatter, density = captured_figures
    assert density.axes[0].collections[0].get_array().sum() == 3
    low, high = scatter.axes[0].get_xlim()
    assert low < 0.2 and high > 1.1    # the axes hold every pair left (the first pair has the bad prediction)


def test_regression_figures_are_skipped_with_a_warning_when_no_prediction_is_finite(tmp_path):
    true, predicted = [np.array([0.2, 0.4])], [np.array([np.nan, np.nan])]
    with pytest.warns(UserWarning, match='regression figures are not drawn'):
        assert figures.plot_test_results(true, predicted, Task('residues', 1), str(tmp_path / 'net')) == []


def test_figures_show_the_pearson_r_of_the_performance_metrics(tmp_path, captured_figures):
    task = Task('sequence', 1)
    labeled = make_labeled(task)
    metrics, _ = reporting.report_test_results(labeled, task, str(tmp_path / 'net'), include_figs=True)
    for fig in captured_figures:
        assert f'Pearson r = {metrics["Pearson R"]:.3f}' in figure_texts(fig)


def test_scatter_and_density_plots_share_axes_and_show_pearson_r(tmp_path, captured_figures):
    rng = np.random.default_rng(0)
    true = [rng.random(20) for _ in range(5)]
    predicted = [values + rng.normal(0, 0.1, 20) for values in true]
    written = figures.plot_test_results(true, predicted, Task('residues', 1), str(tmp_path / 'net'))

    assert written == [str(tmp_path / 'net_res_scatterplot.png'), str(tmp_path / 'net_res_density.png')]
    scatter, density = captured_figures
    pearson = figures.pearson_r(np.concatenate(true), np.concatenate(predicted))
    for fig in (scatter, density):
        assert f'Pearson r = {pearson:.3f}' in figure_texts(fig)
    assert scatter.axes[0].get_xlim() == density.axes[0].get_xlim() == density.axes[0].get_ylim()
    assert density.axes[1].get_ylabel() == 'Residues per bin'


def test_density_plot_counts_every_value(tmp_path, captured_figures):
    true = np.array([0.1, 0.1, 0.1, 0.9])
    figures.regression_density_plot(true, true.copy(), str(tmp_path / 'density.png'), 'Sequences')
    counts = captured_figures[-1].axes[0].collections[0].get_array()
    assert counts.sum() == 4 and counts.max() == 3


@pytest.mark.parametrize('repeats, log_scale', [(3, False), (figures.LOG_SCALE_MIN_COUNT, True)])
def test_density_plots_use_a_log_scale_once_a_bin_holds_enough_values(tmp_path, captured_figures, repeats,
                                                                      log_scale):
    values = np.repeat([0.1, 0.9], repeats)
    figures.regression_density_plot(values, values.copy(), str(tmp_path / 'density.png'))
    norm = captured_figures[-1].axes[0].collections[0].norm
    assert isinstance(norm, LogNorm) == log_scale


def test_matrix_density_plots_show_every_cell_while_scatterplots_are_sampled(tmp_path, captured_figures,
                                                                             monkeypatch):
    monkeypatch.setattr(figures, 'MAX_SCATTER_POINTS', 10)
    values = [np.arange(25, dtype=float).reshape(5, 5)]
    with pytest.warns(UserWarning, match='random sample of 10'):
        figures.plot_test_results(values, values, Task('matrix', 1), str(tmp_path / 'net'))
    scatter, density = captured_figures
    assert len(scatter.axes[0].collections[0].get_offsets()) == 10
    assert density.axes[0].collections[0].get_array().sum() == 25


# ..........................................................................................
# Residue profiles

@pytest.mark.parametrize('true, predicted, task, expected', [
    ([0.0, 1.0, np.nan], [0.5, 1.0, 5.0], Task('residues', 1), 0.25),       # unlabelled residues are left out
    ([0, 1, pid.IGNORED_CLASS_LABEL, 2], [0, 2, 1, 2], Task('residues', 3), 1 / 3),
    ([np.nan, np.nan], [0.5, 0.5], Task('residues', 1), np.inf),
])
def test_sequence_error(true, predicted, task, expected):
    assert figures.sequence_error(np.array(true), np.array(predicted), task) == pytest.approx(expected)


def test_profile_examples_default_to_the_current_number_of_examples(monkeypatch):
    monkeypatch.setattr(figures, 'MIN_PROFILE_RESIDUES', 1)
    monkeypatch.setattr(figures, 'NUM_PROFILE_EXAMPLES', 2)
    true = [np.zeros(3) for _ in range(5)]
    predicted = [np.full(3, i / 10) for i in range(5)]
    assert len(figures.profile_examples(true, predicted, Task('residues', 1))) == 2


def test_profiles_need_enough_labelled_residues_not_just_long_sequences(monkeypatch):
    monkeypatch.setattr(figures, 'MIN_PROFILE_RESIDUES', 3)
    # The best-predicted sequence is long but has only 2 labelled residues, so the others are chosen from
    sparse = np.full(10, np.nan)
    sparse[[2, 7]] = 0.0
    true = [sparse, np.zeros(4), np.zeros(4)]
    predicted = [np.zeros(10), np.full(4, 0.1), np.full(4, 0.2)]
    assert figures.profile_examples(true, predicted, Task('residues', 1), num_examples=2) == [1, 2]


def test_isolated_labelled_residues_are_drawn_as_points(tmp_path, captured_figures):
    true = np.array([0.1, np.nan, 0.5, np.nan, 0.3, 0.4])    # residues 1 and 3 have no labelled neighbour
    figures.plot_residue_profiles(['a'], [true], [np.full(6, 0.3)], Task('residues', 1), str(tmp_path / 'p.png'))
    points = [line for line in captured_figures[-1].axes[0].lines if line.get_marker() == 'o']
    assert len(points) == 1 and list(points[0].get_xdata()) == [1, 3]
    np.testing.assert_array_equal(figures.isolated_residues(true), [True, False, True, False, False, False])


def test_profiles_are_skipped_with_a_warning_when_no_sequence_can_be_shown(tmp_path):
    with pytest.warns(UserWarning, match='no profiles are drawn'):
        written = figures.plot_residue_profiles(['a'], [np.full(4, np.nan)], [np.zeros(4)], Task('residues', 1),
                                                str(tmp_path / 'p.png'))
    assert written == [] and not (tmp_path / 'p.png').exists()


def test_profiles_span_the_range_of_accuracy(monkeypatch):
    monkeypatch.setattr(figures, 'MIN_PROFILE_RESIDUES', 3)
    # Sequence i is predicted with error i (12 sequences), except for one too short and one with no labels
    true = [np.zeros(4) for _ in range(12)] + [np.zeros(2), np.full(4, np.nan)]
    predicted = [np.full(4, np.sqrt(i)) for i in range(12)] + [np.zeros(2), np.zeros(4)]
    chosen = figures.profile_examples(true, predicted, Task('residues', 1), num_examples=6)
    assert chosen == [1, 3, 5, 7, 9, 11]    # the middle of each sixth, from the most to the least accurate


def test_profile_labels_give_the_mean_absolute_error_even_when_r_is_undefined():
    true, predicted = np.array([0.5, 0.5, np.nan]), np.array([0.4, 0.7, 0.0])
    assert figures.profile_score(true, predicted, Task('residues', 1)) == ' (MAE = 0.15)'


def test_short_sequences_are_shown_only_if_too_few_are_long_enough(monkeypatch):
    monkeypatch.setattr(figures, 'MIN_PROFILE_RESIDUES', 3)
    # Only one sequence has at least 3 residues, so all three are shown, from the most to the least accurate
    true = [np.zeros(2), np.zeros(5), np.zeros(2)]
    predicted = [np.full(2, 0.1), np.full(5, 0.3), np.full(2, 0.2)]
    assert figures.profile_examples(true, predicted, Task('residues', 1), num_examples=3) == [0, 2, 1]


@pytest.mark.parametrize('task, predicted, score', [
    (Task('residues', 1), [np.array([0.1, 0.5, 0.9, 0.4]), np.array([0.2, 0.3, 0.8])], 'Pearson r ='),
    (Task('residues', 1), [np.array([0.1, 0.5, 0.9, 0.4]), np.array([0.2, 0.3, 0.8])], ', MAE = '),
    (Task('residues', 3), [np.array([0, 1, 2, 2]), np.array([1, 1, 0])], 'accuracy'),
    (Task('residues', 3), [np.eye(3)[[0, 1, 2, 2]], np.eye(3)[[1, 1, 0]]], 'accuracy'),   # class probabilities
], ids=['regression-r', 'regression-mae', 'classes', 'probabilities'])
def test_residue_profiles_name_each_sequence_and_its_score(tmp_path, captured_figures, task, predicted, score):
    true = ([np.array([0.0, 0.5, 1.0, np.nan]), np.array([0.2, 0.4, 0.9])] if not task.is_classification
            else [np.array([0, 1, 2, pid.IGNORED_CLASS_LABEL]), np.array([1, 1, 1])])
    figures.plot_residue_profiles(['first', ''], true, predicted, task, str(tmp_path / 'profiles.png'))

    labels = [ax.get_xlabel() for ax in captured_figures[-1].axes]
    assert len(labels) == 2 and all(score in label for label in labels)
    assert {label.split(' (')[0] for label in labels} == {'Residue of first', 'Residue of test1'}
    assert [line.get_label() for line in captured_figures[-1].axes[0].lines] == ['True', 'Predicted']


def test_regression_scatterplot_does_not_accumulate(tmp_path, captured_figures):
    values = np.array([1.0, 2.0, 3.0])
    path = str(tmp_path / 'scatter.png')

    figures.regression_scatterplot(values, values, path)
    figures.regression_scatterplot(values, values, path)

    assert len(captured_figures) == 2
    assert len(captured_figures[1].axes[0].collections) == 1
    assert plt.get_fignums() == []


@pytest.mark.parametrize('num_classes', [2, 3])
def test_roc_and_precision_recall_curves(tmp_path, num_classes):
    rng = np.random.default_rng(0)
    true_classes = np.arange(30) % num_classes
    probabilities = rng.random((30, num_classes))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    figures.plot_roc_curve(true_classes, probabilities, str(tmp_path / 'roc.png'))
    figures.plot_precision_recall_curve(true_classes, probabilities, str(tmp_path / 'pr.png'))

    assert os.path.isfile(tmp_path / 'roc.png') and os.path.isfile(tmp_path / 'pr.png')
    assert plt.get_fignums() == []


def test_confusion_matrix(tmp_path, captured_figures):
    path = str(tmp_path / 'cm.png')

    figures.plot_confusion_matrix(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), 3, path)

    assert os.path.isfile(path)
    assert plt.get_fignums() == []

    # Rows are predicted labels and columns are true labels
    heatmap_values = captured_figures[-1].axes[0].images[0].get_array()
    np.testing.assert_array_equal(heatmap_values, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])


def test_confusion_counts():
    counts = metrics.confusion_counts(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), 3)
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])


def test_figures_use_the_lab_style_without_changing_global_settings(tmp_path, captured_figures):
    """Figures have no titles or grid lines, and drawing them leaves matplotlib's global settings alone"""
    settings_before = dict(plt.rcParams)
    figures.plot_training_loss([3.0, 2.0, 1.5], [3.5, 2.5, 2.0], str(tmp_path / 'loss.png'))

    ax = captured_figures[-1].axes[0]
    assert ax.get_title() == ''
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert [line.get_color() for line in ax.get_lines()] == ['black', 'red']
    assert all(line.get_linestyle() == '-' for line in ax.get_lines())
    assert dict(plt.rcParams) == settings_before


# ..........................................................................................
# Performance metrics

def test_metrics_sequence_classification():
    labels = [np.asarray(label) for label in [0, 1, 2, 0, 1, 2]]
    scores = metrics.performance_metrics(labels, labels, Task('sequence', 3))
    assert scores == {'Matthews Correlation Coef': 1.0, 'F1 Score': 1.0, 'Accuracy': 1.0}


def test_metrics_probabilistic_classification():
    labels = [np.asarray(label) for label in [0, 1, 1, 0]]
    probabilities = [np.array([0.9, 0.1]), np.array([0.2, 0.8]), np.array([0.4, 0.6]), np.array([0.7, 0.3])]

    scores = metrics.performance_metrics(labels, probabilities, Task('sequence', 2), probabilistic=True)

    assert scores['Area under ROC'] == 1.0
    assert scores['Area under Precision-Recall curve'] == 1.0
    assert scores['Accuracy'] == 1.0


def test_metrics_residue_classification():
    true_values = [np.array([0, 1, 1]), np.array([1, 0])]
    predictions = [np.array([0, 1, 0]), np.array([1, 0])]
    assert metrics.performance_metrics(true_values, predictions, Task('residues', 2))['Accuracy'] == 0.8


@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_metrics_regression(datatype):
    if datatype == 'sequence':
        true_values = [np.asarray(value) for value in [0.1, 0.5, 0.2, 0.9]]
    else:
        true_values = [np.array([0.1, 0.5]), np.array([0.2, 0.9])]
    predictions = [2 * value + 1 for value in true_values]

    scores = metrics.performance_metrics(true_values, predictions, Task(datatype, 1))

    assert scores['Pearson R'] == pytest.approx(1.0, abs=1e-3)
    assert scores['Spearman R'] == 1.0


def test_write_performance_metrics(tmp_path):
    path = str(tmp_path / 'stats.txt')
    reporting.write_performance_metrics({'Pearson R': 0.5, 'Spearman R': 0.25}, path)
    assert read_lines(path) == ['Pearson R : 0.500', 'Spearman R : 0.250']


# ..........................................................................................
# Prediction files

@pytest.mark.parametrize('values, decimals, expected', [
    (np.float32(1.25), None, '1.25'),
    (np.int64(2), None, '2'),
    (np.array([0, 2, 1]), None, '0 2 1'),
    (np.array([0.123456, 0.876544]), 4, '0.1235 0.8765'),
    # Several values per residue (class probabilities): one comma-separated token per residue
    (np.array([[0.91, 0.06, 0.03], [0.12, 0.8, 0.08]]), 4, '0.91,0.06,0.03 0.12,0.8,0.08'),
])
def test_format_values(values, decimals, expected):
    assert reporting.format_values(values, decimals) == expected


def test_probabilistic_metrics_leave_out_residues_with_missing_labels():
    task = Task('residues', 2)
    true_values = [np.array([0, pid.IGNORED_CLASS_LABEL, 1]), np.array([1])]
    probabilities = [np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]), np.array([[0.3, 0.7]])]
    true_flat, probability_rows = metrics.labelled_values(true_values, probabilities, task, probabilistic=True)
    np.testing.assert_array_equal(true_flat, [0, 1, 1])
    np.testing.assert_array_equal(probability_rows, [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])

    scores = metrics.performance_metrics(true_values, probabilities, task, probabilistic=True)
    assert scores['Accuracy'] == 1.0 and scores['Area under ROC'] == 1.0


def test_write_predictions(tmp_path):
    path = str(tmp_path / 'predictions.tsv')
    reporting.write_predictions(['protA', 'protB'], ['ACD', 'KK'], [np.asarray(1.5), np.asarray(-2.0)],
                                [np.float32(1.25), np.float32(-1.0)], Task('sequence', 1), path)

    assert read_lines(path) == ['protA_TRUE ACD 1.5', 'protA_PRED ACD 1.25',
                                'protB_TRUE KK -2.0', 'protB_PRED KK -1.0']


def test_write_predictions_without_ids(tmp_path):
    path = str(tmp_path / 'predictions.tsv')
    reporting.write_predictions(['', ''], ['ACD', 'KK'], [np.asarray(1), np.asarray(0)],
                                [np.int64(1), np.int64(2)], Task('sequence', 3), path)

    assert read_lines(path) == ['test0_TRUE ACD 1', 'test0_PRED ACD 1', 'test1_TRUE KK 0', 'test1_PRED KK 2']


def test_write_predictions_with_missing_labels(tmp_path):
    path = str(tmp_path / 'predictions.tsv')
    ignored = pid.IGNORED_CLASS_LABEL
    reporting.write_predictions(['a'], ['ACD'], [np.array([0, ignored, 2])], [np.array([0, 1, 2])],
                                Task('residues', 3), path)
    assert read_lines(path) == ['a_TRUE ACD 0 nan 2', 'a_PRED ACD 0 1 2']

    reporting.write_predictions(['a'], ['ACD'], [np.array([0.5, np.nan, 2.0])],
                                [np.array([0.25, 1.0, 2.0], dtype=np.float32)], Task('residues', 1), path)
    assert read_lines(path) == ['a_TRUE ACD 0.5 nan 2.0', 'a_PRED ACD 0.25 1.0 2.0']


def test_metrics_ignore_missing_labels():
    ignored = pid.IGNORED_CLASS_LABEL
    classification = metrics.performance_metrics([np.array([0, ignored, 1]), np.array([ignored, 0])],
                                                   [np.array([0, 1, 1]), np.array([1, 0])], Task('residues', 2))
    assert classification['Accuracy'] == 1.0

    regression = metrics.performance_metrics([np.array([0.1, np.nan, 0.5]), np.array([np.nan, 0.9])],
                                               [np.array([1.2, -50.0, 2.0]), np.array([99.0, 2.8])],
                                               Task('residues', 1))
    assert regression['Pearson R'] == pytest.approx(1.0, abs=1e-3)


# ..........................................................................................
# Full test-set report

def make_labeled(task: Task) -> LabeledPredictions:
    """Labeled predictions for three sequences, with outputs consistent with the task."""
    rng = np.random.default_rng(0)
    sequences = ['ACDE', 'FGH', 'KLMNP']
    if task.per_residue:
        true_values = [rng.integers(0, task.num_classes, len(seq)) if task.is_classification
                       else rng.random(len(seq)) for seq in sequences]
        outputs = [rng.random((len(seq), task.num_classes)) for seq in sequences]
    else:
        true_values = [np.asarray(i % task.num_classes) if task.is_classification else np.asarray(rng.random())
                       for i in range(len(sequences))]
        outputs = [rng.random(task.num_classes) for _ in sequences]
    return LabeledPredictions(names=['a', 'b', 'c'], sequences=sequences, true_values=true_values, outputs=outputs,
                              loss=0.5)


@pytest.mark.parametrize('task, probabilistic, expected_figures', [
    (Task('sequence', 1), False, ['_seq_scatterplot.png', '_seq_density.png']),
    (Task('residues', 1), False, ['_res_scatterplot.png', '_res_density.png', '_res_profiles.png']),
    (Task('sequence', 3), False, ['_seq_CM.png']),
    (Task('residues', 3), False, ['_res_CM.png', '_res_profiles.png']),
    (Task('sequence', 3), True, ['_ROC_curve.png', '_PR_curve.png']),
    (Task('residues', 3), True, ['_ROC_curve.png', '_PR_curve.png', '_res_profiles.png']),
], ids=['seq-regress', 'res-regress', 'seq-class', 'res-class', 'seq-prob', 'res-prob'])
def test_report_test_results(tmp_path, task, probabilistic, expected_figures):
    prefix = str(tmp_path / 'net')
    _, written = reporting.report_test_results(make_labeled(task), task, prefix, probabilistic=probabilistic,
                                               include_figs=True)

    assert len(read_lines(prefix + '_predictions.tsv')) == 6
    assert os.path.isfile(prefix + '_performance_stats.txt')
    assert [path for path in written if path.endswith('.png')] == [prefix + suffix for suffix in expected_figures]
    for path in written:
        assert os.path.isfile(path), path
    assert plt.get_fignums() == []


def test_report_test_results_optional_outputs(tmp_path):
    prefix = str(tmp_path / 'net')
    reporting.report_test_results(make_labeled(Task('sequence', 1)), Task('sequence', 1), prefix,
                                  include_figs=False, write_stats=False)

    assert os.listdir(tmp_path) == ['net_predictions.tsv']
