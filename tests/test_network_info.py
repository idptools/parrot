"""
Tests for the <network>_info.txt file (parrot/scripts/network_info.py): the
pieces that describe the data, the encoding and the training, and the file
parrot-optimize writes.
"""

import math

import numpy as np
import pytest
import torch

from parrot.scripts import network_info, parrot_optimize
from parrot.tasks import IGNORED_CLASS_LABEL, Task

from tests.helpers import data_path, training_command


@pytest.mark.parametrize('val_losses, expected', [
    ([3.0, 2.0, 2.5], 1),
    ([2.0, 1.0, 1.0], 1),                     # the first of equal losses, as training keeps it
    ([math.nan, 2.0, math.inf, 1.5], 3),      # losses that are not finite never count
    ([math.nan, math.inf], None),
])
def test_best_epoch(val_losses, expected):
    assert network_info.best_epoch(val_losses) == expected


@pytest.mark.parametrize('task, loss, over', [
    (Task('sequence', 1), 'L1 (absolute error)', 'over the sequences'),
    (Task('residues', 1), 'mean squared error', 'over every labelled residue'),
    (Task('sequence', 3), 'cross-entropy', 'over the sequences'),
    (Task('matrix', 2), 'cross-entropy', 'over every labelled cell'),
])
def test_describe_loss_matches_training(task, loss, over):
    description = network_info.describe_loss(task)
    assert description.startswith(loss) and over in description


def test_describe_values_summarizes_classes_and_counts_missing_labels():
    task = Task('residues', 3)
    values = [np.array([0, 1, 1, IGNORED_CLASS_LABEL]), np.array([2, 2])]
    name, description = network_info.describe_values(task, values)
    assert name == 'classes'
    assert description.startswith('per residue: 0: 1 (20.0%), 1: 2 (40.0%), 2: 2 (40.0%)')
    assert '1 missing' in description


def test_describe_values_summarizes_regression_values():
    name, description = network_info.describe_values(Task('sequence', 1), [1.0, 2.0, 3.0])
    assert name == 'values' and description == 'per sequence: mean 2, standard deviation 0.8165, range 1 to 3'


def test_long_settings_are_summarized_by_their_length():
    settings = {'name': 'standardized', 'base': {'name': 'biophysics'}, 'mean': list(range(20)), 'window': 4}
    assert network_info.flatten_settings(settings) == [('name', 'standardized'), ('base.name', 'biophysics'),
                                                       ('mean', '(20 values)'), ('window', '4')]


def test_optimizer_description_uses_pytorchs_settings():
    defaults = torch.optim.Adam([torch.zeros(1)]).defaults
    description = network_info.describe_optimizer(0.002)
    assert description.startswith('Adam, learning rate 0.002 (constant)')
    assert f'eps {defaults["eps"]:g}' in description


def test_write_network_info_lays_out_sections(tmp_path):
    sections = [network_info.Section('Data', [('input file', 'data.tsv'), ('sequences', '300')]),
                network_info.Section('Architecture', [('architecture', 'brnn')], text='BRNN(\n  (lstm): LSTM()\n)')]
    path = tmp_path / 'net_info.txt'
    network_info.write_network_info(str(path), str(tmp_path / 'net.pt'), sections)

    lines = path.read_text().splitlines()
    assert lines[:2] == ['PARROT network: net.pt', '======================']
    assert 'input file : data.tsv' in lines and 'sequences  : 300' in lines
    assert '    BRNN(' in lines and '      (lstm): LSTM()' in lines


def test_optimize_info_describes_the_search(run_cli, tmp_path, write_lines):
    lines = [line for line in open(data_path('seq_regress_dataset.tsv'))][:60]
    data_file = write_lines('data.tsv', [line.rstrip('\n') for line in lines])
    run_cli(parrot_optimize.main, training_command('parrot-optimize', 'cnn', data_file, tmp_path / 'opt', 'sequence', 1,
                                                   '-e', '3', '--trials', '3', '--channels', '8', '--seed', '1',
                                                   '--silent'))

    info = (tmp_path / 'opt_info.txt').read_text()
    assert '\nHyperparameter search\n' in info
    for fact in ('successive halving', 'searched learning_rate', 'fixed channels', 'chosen', 'trial',
                 'opt_hyperparameter_trials.tsv', 'opt_optimal_hyperparams.txt'):
        assert fact in info, fact
