"""
Tests for the helper modules in parrot.tools: argument validation, CLI helpers,
dataset warnings, preprocessing and class balancing.
"""

import os
import random

import numpy as np
import pytest
import torch

import parrot
from parrot import process_input_data as pid
from parrot.parrot_exceptions import ParrotException
from parrot.tools import class_balancing, cli, dataset_warnings, preproc, validate_args

from tests.helpers import data_path


# ..........................................................................................
# validate_args

def test_check_file_exists(tmp_path):
    path = tmp_path / 'exists.txt'
    path.write_text('x')

    assert validate_args.check_file_exists(str(path), 'File') == str(path)
    with pytest.raises(FileNotFoundError, match='Datafile does not exist'):
        validate_args.check_file_exists(str(tmp_path / 'missing.txt'), 'Datafile')


def test_check_directory(tmp_path):
    validate_args.check_directory(str(tmp_path), 'Output directory')
    with pytest.raises(FileNotFoundError):
        validate_args.check_directory(str(tmp_path / 'missing'), 'Output directory')


@pytest.mark.parametrize('filename, expected_prefix', [
    ('network.pt', 'network'),
    ('network', 'network'),
    ('my.network.v2.pt', 'my.network.v2'),
])
def test_split_file_and_directory(tmp_path, filename, expected_prefix):
    prefix, directory = validate_args.split_file_and_directory(str(tmp_path / filename))

    assert directory == os.path.join(str(tmp_path), '')
    assert prefix == os.path.join(str(tmp_path), expected_prefix)


def test_split_file_and_directory_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_args.split_file_and_directory(str(tmp_path / 'missing' / 'network.pt'))


def test_check_positive():
    validate_args.check_positive(1, 'Value')
    for bad_value in [0, -3]:
        with pytest.raises(ValueError, match='positive'):
            validate_args.check_positive(bad_value, 'Value')


def test_check_between_zero_and_one():
    validate_args.check_between_zero_and_one(0.5, 'Value')
    for bad_value in [0, 1, -0.1, 1.5]:
        with pytest.raises(ValueError, match='between 0 and 1'):
            validate_args.check_between_zero_and_one(bad_value, 'Value')


def test_check_stop_condition(capsys):
    validate_args.check_stop_condition('iter', 100)
    validate_args.check_stop_condition('auto', 5)
    assert capsys.readouterr().out == ''

    validate_args.check_stop_condition('auto', 50)
    assert 'may take a long time' in capsys.readouterr().out

    with pytest.raises(ValueError):
        validate_args.check_stop_condition('never', 10)


# ..........................................................................................
# cli

def draw_random_numbers() -> tuple[float, float, float]:
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_set_random_seed_is_reproducible():
    cli.set_random_seed(123)
    first = draw_random_numbers()
    cli.set_random_seed(123)
    assert draw_random_numbers() == first

    cli.set_random_seed(124)
    assert draw_random_numbers() != first


def test_set_random_seed_none_does_not_reseed():
    cli.set_random_seed(123)
    first = draw_random_numbers()
    cli.set_random_seed(123)
    cli.set_random_seed(None)
    assert draw_random_numbers() == first


def test_set_random_seed_makes_cudnn_deterministic(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)
    cli.set_random_seed(123)
    assert torch.backends.cudnn.deterministic


def test_print_startup(capsys):
    cli.print_startup('parrot-train', silent=False)
    output = capsys.readouterr().out
    assert f'parrot-train (version {parrot.__version__})' in output
    assert 'parrot-predict' not in output

    cli.print_startup('parrot-predict', silent=False, print_parrot=True)
    assert 'Welcome to PARROT!' in capsys.readouterr().out

    cli.print_startup('parrot-train', silent=True)
    assert capsys.readouterr().out == ''


def test_print_settings(capsys):
    cli.print_settings(False, hidden_size=10, dtype='sequence')
    output = capsys.readouterr().out
    assert 'hidden_size   : 10' in output and 'dtype         : sequence' in output

    cli.print_settings(True, hidden_size=10)
    assert capsys.readouterr().out == ''


def test_cli_validate_args():
    cli.validate_args('residues', ['residues', 'sequence'])
    with pytest.raises(Exception, match='Invalid option passed'):
        cli.validate_args('residue', ['residues', 'sequence'], 'extra message')


# ..........................................................................................
# dataset_warnings

def test_duplicate_sequence_warning(capsys):
    dataset_warnings.check_duplicate_sequences([['a', 'ACD', 1], ['b', 'EFG', 1]])
    assert capsys.readouterr().out == ''

    dataset_warnings.check_duplicate_sequences([['a', 'ACD', 1], ['b', 'ACD', 2], ['c', 'EFG', 1]])
    output = capsys.readouterr().out
    assert 'Duplicate found: ACD' in output and 'EFG' not in output


@pytest.mark.parametrize('counts, expected', [
    ([10, 9], None),
    ([20, 10], 'Slight class imbalance'),
    ([30, 10], 'Severe class imbalance'),
])
def test_class_imbalance_warning(capsys, counts, expected):
    data = [['x', 'A', label] for label, count in enumerate(counts) for _ in range(count)]
    dataset_warnings.check_class_imbalance(data)
    output = capsys.readouterr().out

    if expected is None:
        assert output == ''
    else:
        assert expected in output


def test_regression_warnings(capsys):
    dataset_warnings.check_regression_imbalance([['x', 'A', value] for value in [-1.0, 0.0, 1.0, 0.5, -0.5]])
    assert capsys.readouterr().out == ''

    dataset_warnings.check_zero_centered(np.array([10.0, 11.0, 12.0]))
    assert 'not centered around zero' in capsys.readouterr().out

    dataset_warnings.check_low_std(np.array([-20.0, 0.0, 20.0]))
    assert 'large variance' in capsys.readouterr().out

    dataset_warnings.check_skew(np.array([0.0] * 20 + [10.0]))
    assert 'skewed' in capsys.readouterr().out

    # A long tail of low values is just as skewed
    dataset_warnings.check_skew(np.array([10.0] * 20 + [0.0]))
    assert 'skewed' in capsys.readouterr().out

    dataset_warnings.check_skew(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert capsys.readouterr().out == ''


def test_batch_size_warnings(capsys):
    dataset_warnings.eval_batch_size(32, 1000)
    assert capsys.readouterr().out == ''

    dataset_warnings.eval_batch_size(1024, 100000)
    assert 'very large batch size' in capsys.readouterr().out

    dataset_warnings.eval_batch_size(32, 200)
    assert 'large relative to the number of samples' in capsys.readouterr().out


# ..........................................................................................
# preproc

def test_recode_classes():
    lines = [['a', 'ACD', '5'], ['b', 'EFG', '9'], ['c', 'HIK', '7'], ['d', 'LMN', '5']]
    new_lines, mapping = preproc.recode_classes(lines)

    assert mapping == {'5': 0, '7': 1, '9': 2}
    assert [line[2] for line in new_lines] == ['0', '2', '1', '0']
    assert [line[:2] for line in new_lines] == [line[:2] for line in lines]


@pytest.mark.parametrize('label, missing', [('nan', True), ('NaN', True), ('NAN', True), ('0', False),
                                            ('1.5', False), ('inf', False), ('helix', False)])
def test_is_missing_label(label, missing):
    assert preproc.is_missing_label(label) == missing


def test_recode_classes_orders_numbers_by_value():
    # '1' and '1.0' are the same class, and 10 comes after 2
    new_lines, mapping = preproc.recode_classes([['a', 'A', '10'], ['b', 'C', '2'], ['c', 'D', '1'], ['d', 'E', '1.0']])
    assert mapping == {'1': 0, '1.0': 0, '2': 1, '10': 2}
    assert [line[2] for line in new_lines] == ['2', '1', '0', '0']


def test_recode_classes_orders_names_alphabetically():
    new_lines, mapping = preproc.recode_classes([['a', 'A', 'strand'], ['b', 'C', 'coil'], ['c', 'D', 'helix']])
    assert mapping == {'coil': 0, 'helix': 1, 'strand': 2}


def test_recode_classes_keeps_missing_labels():
    new_lines, mapping = preproc.recode_classes([['a', 'ACD', '2', 'nan', '5'], ['b', 'EF', 'NaN', '2']])
    assert mapping == {'2': 0, '5': 1}
    assert new_lines == [['a', 'ACD', '0', 'nan', '1'], ['b', 'EF', 'NaN', '0']]


def test_recode_classes_residues():
    new_lines, mapping = preproc.recode_classes([['a', 'ACD', '-1', '3', '3'], ['b', 'EF', '3', '-1']])
    assert mapping == {'-1': 0, '3': 1}
    assert new_lines == [['a', 'ACD', '0', '1', '1'], ['b', 'EF', '1', '0']]


@pytest.mark.parametrize('verbose', [False, True])
def test_remove_line_duplicates(verbose):
    lines = [['a', 'ACD', '1'], ['a', 'ACD', '1'], ['a', 'ACD', '0'], ['b', 'ACD', '1'], ['c', 'EFG', '1']]
    new_lines = preproc.remove_line_duplicates(lines, verbose=verbose)

    # Only the exact duplicate line is removed
    assert sorted(new_lines) == sorted([['a', 'ACD', '1'], ['a', 'ACD', '0'], ['b', 'ACD', '1'], ['c', 'EFG', '1']])


# ..........................................................................................
# Residue class balancing (SRCB)

def make_imbalanced_residue_lines(n_sequences: int = 200, length: int = 12,
                                  class_probabilities: tuple[float, ...] = (0.6, 0.3, 0.1),
                                  seed: int = 7) -> list[list[str]]:
    """Synthetic residue-classification lines whose class labels are drawn with the given probabilities."""
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n_sequences):
        labels = rng.choice(len(class_probabilities), size=length, p=class_probabilities)
        lines.append([f'seq{i}', 'A' * length] + [str(label) for label in labels])
    return lines


def imbalance_of(lines: list[list[str]]) -> float:
    """The imbalance (most / least common class) of residue-classification lines."""
    _, counts = class_balancing.class_count_matrix(lines)
    imbalance, _ = class_balancing.balance_scores(counts.sum(axis=0)[None, :])
    return float(imbalance[0])


def test_residue_class_balance_reduces_imbalance():
    lines = make_imbalanced_residue_lines()
    original_imbalance = imbalance_of(lines)
    max_fraction_removed = 0.25

    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), block_size=20,
                                                                    max_fraction_removed=max_fraction_removed,
                                                                    verbose=False)

    assert tracker[0] == pytest.approx(original_imbalance)
    assert imbalance_of(new_lines) < original_imbalance
    assert len(lines) - len(new_lines) <= int(len(lines) * max_fraction_removed)
    assert len(new_lines) == len(lines) - (len(tracker) - 1)
    assert all(line in lines for line in new_lines)


def test_residue_class_balance_tracker_reports_final_imbalance():
    """The last tracker entry must describe the returned data, not the state one removal earlier"""
    new_lines, tracker = class_balancing.run_class_balance_residues(make_imbalanced_residue_lines(), block_size=20,
                                                                    verbose=False)
    assert tracker[-1] == pytest.approx(imbalance_of(new_lines))


def test_residue_class_balance_stops_at_threshold():
    lines = make_imbalanced_residue_lines(class_probabilities=(0.45, 0.35, 0.2))
    new_lines, tracker = class_balancing.run_class_balance_residues(lines, balance_threshold=2.0, block_size=20,
                                                                    max_fraction_removed=0.9, verbose=False)

    assert tracker[-1] <= 2.0
    assert all(score > 2.0 for score in tracker[:-1])


def test_residue_class_balance_already_balanced():
    lines = make_imbalanced_residue_lines(class_probabilities=(0.34, 0.33, 0.33))
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), balance_threshold=1.4,
                                                                    block_size=20, verbose=False)
    assert len(tracker) == 1 and len(new_lines) == len(lines)


def test_residue_class_balance_keeps_input_order():
    lines = make_imbalanced_residue_lines()
    new_lines, _ = class_balancing.run_class_balance_residues(list(lines), block_size=20, verbose=False)

    original_positions = [lines.index(line) for line in new_lines]
    assert len(new_lines) < len(lines)
    assert original_positions == sorted(original_positions)


def test_residue_class_balance_never_makes_the_balance_worse():
    lines = make_imbalanced_residue_lines(n_sequences=300, class_probabilities=(0.5, 0.3, 0.2))
    _, tracker = class_balancing.run_class_balance_residues(list(lines), balance_threshold=1.0, block_size=5,
                                                            max_fraction_removed=0.9, verbose=False)
    assert all(later <= earlier + 1e-12 for earlier, later in zip(tracker, tracker[1:]))


def test_residue_class_balance_breaks_ties_between_the_most_common_classes():
    # Classes 0 and 1 tie as the most common, so no single removal lowers the imbalance on its own
    lines = ([[f'zero{i}', 'A', '0'] for i in range(10)] + [[f'one{i}', 'A', '1'] for i in range(10)]
             + [[f'two{i}', 'A', '2'] for i in range(5)])
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), balance_threshold=1.0,
                                                                    max_fraction_removed=0.5, verbose=False)
    assert tracker[-1] == 1.0
    assert len(new_lines) == 15


def test_residue_class_balance_stops_when_no_removal_helps():
    # Every line carries one residue of each class, so removing lines cannot change the balance
    lines = [[f'seq{i}', 'AAAA', '0', '0', '1', '2'] for i in range(40)]
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), max_fraction_removed=0.5,
                                                                    verbose=False)
    assert new_lines == lines and tracker == [2.0]


def test_residue_class_balance_never_removes_unlabelled_lines():
    lines = make_imbalanced_residue_lines() + [[f'unlabelled{i}', 'AAA', 'nan', 'nan', 'nan'] for i in range(20)]
    new_lines, _ = class_balancing.run_class_balance_residues(list(lines), balance_threshold=1.0, block_size=50,
                                                              max_fraction_removed=0.5, verbose=False)
    assert sum(line[0].startswith('unlabelled') for line in new_lines) == 20


def test_class_count_matrix_treats_equal_numbers_as_one_class():
    classes, counts = class_balancing.class_count_matrix([['a', 'AAA', '1', '1.0', 'nan'], ['b', 'A', '0']])
    assert classes == [0.0, 1.0]
    assert counts.tolist() == [[0, 2], [1, 0]]


def test_residue_class_balance_ignores_missing_labels():
    # The labelled residues are perfectly balanced; missing labels must not count as a third class
    lines = [[f'seq{i}', 'A' * 6] + ['0', '1', '0', '1', 'nan', 'nan'] for i in range(50)]
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), block_size=10, verbose=False)
    assert tracker == [1.0] and len(new_lines) == len(lines)


@pytest.mark.parametrize('n_sequences', [1, 2, 3, 4, 5])
def test_residue_class_balance_on_tiny_datasets(n_sequences):
    lines = make_imbalanced_residue_lines(n_sequences=n_sequences)
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), verbose=False)

    # max_fraction_removed (0.25) is a hard limit, even when it allows no removal at all
    assert len(lines) - len(new_lines) <= int(len(lines) * 0.25)


def test_residue_class_balance_never_removes_a_whole_class():
    # Removing either sequence would delete a class entirely, so nothing can be removed
    lines = [['a', 'A' * 10] + ['0'] * 10, ['b', 'A'] + ['1']]
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), max_fraction_removed=0.9,
                                                                    verbose=False)
    assert sorted(new_lines) == sorted(lines) and tracker == [10.0]


def test_residue_class_balance_does_not_modify_its_input():
    lines = make_imbalanced_residue_lines()
    original = [list(line) for line in lines]
    class_balancing.run_class_balance_residues(lines, block_size=20, verbose=False)
    assert lines == original


@pytest.mark.parametrize('kwargs', [
    {'block_size': 1},
    {'balance_threshold': 0.5},
    {'balance_threshold': 6},
    {'max_fraction_removed': 0},
    {'max_fraction_removed': 1.0},
])
def test_residue_class_balance_validates_input(kwargs):
    with pytest.raises(ParrotException):
        class_balancing.run_class_balance_residues(make_imbalanced_residue_lines(), verbose=False, **kwargs)


def test_residue_class_balance_on_example_data(capsys):
    lines = pid.read_tsv_raw(data_path('res_class_dataset.tsv'))
    new_lines, tracker = class_balancing.run_class_balance_residues(list(lines), balance_threshold=1.05,
                                                                    block_size=30, verbose=True)

    assert tracker[-1] <= tracker[0]
    assert tracker[-1] == pytest.approx(imbalance_of(new_lines))
    assert 'Performing automatic class balancing for residues' in capsys.readouterr().out


# ..........................................................................................
# Sequence class balancing

def make_sequence_lines(class_counts: list[int]) -> list[list[str]]:
    lines = []
    for label, count in enumerate(class_counts):
        lines.extend([[f'c{label}_{i}', 'ACD', str(label)] for i in range(count)])
    return lines


def class_counts_of(lines: list[list[str]]) -> dict[str, int]:
    counts = {}
    for line in lines:
        counts[line[2]] = counts.get(line[2], 0) + 1
    return counts


def test_sequence_class_balance_reaches_threshold():
    lines = make_sequence_lines([100, 30, 20])
    new_lines = class_balancing.run_class_balance_sequences(list(lines), balance_threshold=1.4,
                                                            max_fraction_removed=0.9, verbose=False)

    counts = class_counts_of(new_lines)
    assert max(counts.values()) / min(counts.values()) <= 1.4
    assert counts['2'] == 20  # the smallest class is never down-sampled
    assert all(line in lines for line in new_lines)


def test_sequence_class_balance_respects_max_fraction_removed():
    lines = make_sequence_lines([100, 30, 20])
    max_fraction_removed = 0.25
    new_lines = class_balancing.run_class_balance_sequences(list(lines), max_fraction_removed=max_fraction_removed,
                                                            verbose=False)

    assert len(lines) - len(new_lines) == int(len(lines) * max_fraction_removed)
    assert class_counts_of(new_lines) == {'0': 63, '1': 30, '2': 20}


def test_sequence_class_balance_gets_the_best_balance_the_budget_allows():
    # 7/6/3 with a budget of 4: capping both larger classes at 5 (3 removals) gives the best possible
    # imbalance, 5/3; a fourth removal could not lower it
    lines = make_sequence_lines([7, 6, 3])
    new_lines = class_balancing.run_class_balance_sequences(list(lines), max_fraction_removed=0.25, verbose=False)
    assert class_counts_of(new_lines) == {'0': 5, '1': 5, '2': 3}


def test_sequence_class_balance_removes_no_more_than_needed():
    # Reaching an imbalance of 1.4 needs class 0 cut to 140, not to 100
    lines = make_sequence_lines([1000, 100])
    new_lines = class_balancing.run_class_balance_sequences(list(lines), balance_threshold=1.4,
                                                            max_fraction_removed=0.9, verbose=False)
    assert class_counts_of(new_lines) == {'0': 140, '1': 100}


def test_sequence_class_balance_keeps_input_order():
    lines = make_sequence_lines([60, 20])
    new_lines = class_balancing.run_class_balance_sequences(list(lines), max_fraction_removed=0.5, verbose=False)
    positions = [lines.index(line) for line in new_lines]
    assert positions == sorted(positions) and len(new_lines) == 48


@pytest.mark.parametrize('counts, budget, threshold, cap', [
    ([10, 10], 5, 1.4, 10),     # already balanced
    ([100, 30, 20], 1000, 1.4, 28),
    ([100, 30, 20], 37, 1.4, 63),
    ([7, 6, 3], 4, 1.4, 5),
    ([5, 5], 3, 1.0, 5),
    ([50, 10], 100, 1.0, 10),   # never below the smallest class
])
def test_sequence_class_cap(counts, budget, threshold, cap):
    assert class_balancing.sequence_class_cap(counts, budget, threshold) == cap


def test_sequence_class_balance_keeps_missing_labels():
    lines = make_sequence_lines([100, 30]) + [[f'missing{i}', 'ACD', 'nan'] for i in range(5)]
    new_lines = class_balancing.run_class_balance_sequences(list(lines), max_fraction_removed=0.9, verbose=False)

    counts = class_counts_of(new_lines)
    assert counts['nan'] == 5  # never removed, and not balanced as a class
    assert counts['0'] / counts['1'] <= 1.4


def test_sequence_class_balance_does_not_modify_its_input():
    lines = make_sequence_lines([100, 30, 20])
    original = [list(line) for line in lines]
    class_balancing.run_class_balance_sequences(lines, verbose=False)
    assert lines == original


def test_sequence_class_balance_already_balanced():
    lines = make_sequence_lines([20, 19, 18])
    new_lines = class_balancing.run_class_balance_sequences(list(lines), verbose=False)
    assert sorted(new_lines) == sorted(lines)
