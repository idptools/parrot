"""
Tests for checking command-line arguments (parrot.scripts.arguments) and for
seeding random number generators (parrot.reproducibility).
"""

import os
import random

import numpy as np
import pytest
import torch

import parrot
from parrot import reproducibility
from parrot.scripts import arguments


# ..........................................................................................
# Files and directories

def test_check_file_exists(tmp_path):
    path = tmp_path / 'exists.txt'
    path.write_text('x')

    assert arguments.check_file_exists(str(path), 'File') == str(path)
    with pytest.raises(FileNotFoundError, match='Datafile does not exist'):
        arguments.check_file_exists(str(tmp_path / 'missing.txt'), 'Datafile')


@pytest.mark.parametrize('filename, expected_prefix', [
    ('network.pt', 'network'),
    ('network', 'network'),
    ('my.network.v2.pt', 'my.network.v2'),
])
def test_output_prefix_and_directory(tmp_path, filename, expected_prefix):
    prefix, directory = arguments.output_prefix_and_directory(str(tmp_path / filename))

    assert directory == os.path.join(str(tmp_path), '')
    assert prefix == os.path.join(str(tmp_path), expected_prefix)


def test_output_prefix_and_directory_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        arguments.output_prefix_and_directory(str(tmp_path / 'missing' / 'network.pt'))


# ..........................................................................................
# Numbers

def test_check_positive():
    arguments.check_positive(1, 'Value')
    for bad_value in [0, -3]:
        with pytest.raises(ValueError, match='at least 1'):
            arguments.check_positive(bad_value, 'Value')


def test_check_fraction():
    arguments.check_fraction(0.5, 'Value')
    for bad_value in [0, 1, -0.1, 1.5]:
        with pytest.raises(ValueError, match='between 0 and 1'):
            arguments.check_fraction(bad_value, 'Value')


def test_check_set_fractions():
    arguments.check_set_fractions((0.7, 0.15, 0.15))
    with pytest.raises(ValueError, match='sum to 1'):
        arguments.check_set_fractions((0.8, 0.15, 0.15))
    with pytest.raises(ValueError, match='between 0 and 1'):
        arguments.check_set_fractions((1.0, 0.0, 0.0))


# ..........................................................................................
# Printing

def test_print_startup(capsys):
    arguments.print_startup('parrot-train', silent=False)
    output = capsys.readouterr().out
    assert f'parrot-train (version {parrot.__version__})' in output
    assert 'parrot-predict' not in output

    arguments.print_startup('parrot-train', silent=True)
    assert capsys.readouterr().out == ''


def test_print_settings(capsys):
    arguments.print_settings(False, hidden_size=10, dtype='sequence')
    output = capsys.readouterr().out
    assert 'hidden_size   : 10' in output and 'dtype         : sequence' in output

    arguments.print_settings(True, hidden_size=10)
    assert capsys.readouterr().out == ''


# ..........................................................................................
# Random seeds

def draw_random_numbers() -> tuple[float, float, float]:
    """One number from each of Python's, NumPy's and PyTorch's random number generators."""
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_set_random_seed_is_reproducible():
    reproducibility.set_random_seed(123)
    first = draw_random_numbers()
    reproducibility.set_random_seed(123)
    assert draw_random_numbers() == first

    reproducibility.set_random_seed(124)
    assert draw_random_numbers() != first


def test_set_random_seed_none_does_not_reseed():
    reproducibility.set_random_seed(123)
    first = draw_random_numbers()
    reproducibility.set_random_seed(123)
    reproducibility.set_random_seed(None)
    assert draw_random_numbers() == first


def test_set_random_seed_makes_cudnn_deterministic(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)
    reproducibility.set_random_seed(123)
    assert torch.backends.cudnn.deterministic


def test_random_generator_follows_the_seed():
    reproducibility.set_random_seed(7)
    first = reproducibility.random_generator().random()
    reproducibility.set_random_seed(7)
    assert reproducibility.random_generator().random() == first
