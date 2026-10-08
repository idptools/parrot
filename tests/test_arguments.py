"""
Tests for checking command-line arguments (parrot.scripts.arguments) and for
seeding random number generators (parrot.reproducibility).
"""

import datetime
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

def test_header_gives_the_command_version_and_time():
    started = datetime.datetime(2026, 10, 7, 14, 32, 5)
    text = arguments.header('parrot-train', started)
    lines = text.split('\n')

    assert all(line in lines for line in arguments.LOGO)
    assert f' parrot-train  |  PARROT {parrot.__version__}  |  2026-10-07 14:32:05' in lines
    assert f' Documentation: {arguments.DOCUMENTATION_URL}' in lines
    assert 'run: parrot-train --show-examples' in text
    assert '--show-examples' not in arguments.header('parrot-train', started, examples_hint=False)

    # Two rules, as wide as the widest line
    rules = [line for line in lines if line and set(line) == {'='}]
    assert len(rules) == 2 and len(rules[0]) == max(len(line) for line in lines)


def test_print_header(capsys):
    arguments.print_header('parrot-predict', silent=False)
    assert ' parrot-predict  |  PARROT ' in capsys.readouterr().out

    arguments.print_header('parrot-predict', silent=True)
    assert capsys.readouterr().out == ''


def test_print_settings(capsys):
    arguments.print_settings(False, hidden_size=10, datatype='sequence', loglength_feedforward=True)
    output = capsys.readouterr().out.splitlines()
    # Values line up after the longest name
    assert output[2:] == ['hidden_size           : 10', 'datatype              : sequence',
                          'loglength_feedforward : True']

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


# ..........................................................................................
# Reporting errors in a command's input

def failing_command(error: BaseException):
    def command() -> None:
        raise error
    return command


@pytest.mark.parametrize('error', [ValueError('Set -c to the number of classes'),
                                   FileNotFoundError('Datafile does not exist: data.tsv'),
                                   ImportError('NARDINI encodings need SPARROW'),
                                   RuntimeError('CUDA was requested but is not available')])
def test_input_errors_are_reported_as_a_message(monkeypatch, capsys, error):
    monkeypatch.delenv(arguments.TRACEBACK_VARIABLE, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        arguments.run_command('parrot-train', failing_command(error))

    assert exit_info.value.code == arguments.INPUT_ERROR_EXIT_CODE
    message = capsys.readouterr().err
    assert message.startswith(f'parrot-train: error: {error}\n')
    assert 'Traceback' not in message and f'{arguments.TRACEBACK_VARIABLE}=1' in message


def test_traceback_variable_shows_the_traceback(monkeypatch):
    monkeypatch.setenv(arguments.TRACEBACK_VARIABLE, '1')
    with pytest.raises(ValueError, match='bad input'):
        arguments.run_command('parrot-train', failing_command(ValueError('bad input')))


def test_unexpected_errors_keep_their_traceback(monkeypatch):
    """Anything but an input error is a bug, which needs its traceback to be fixed"""
    monkeypatch.delenv(arguments.TRACEBACK_VARIABLE, raising=False)
    with pytest.raises(KeyError):
        arguments.run_command('parrot-train', failing_command(KeyError('hidden_size')))


def test_stopping_a_command_with_ctrl_c(monkeypatch, capsys):
    monkeypatch.delenv(arguments.TRACEBACK_VARIABLE, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        arguments.run_command('parrot-predict', failing_command(KeyboardInterrupt()))
    assert exit_info.value.code == arguments.INTERRUPTED_EXIT_CODE
    assert 'parrot-predict: stopped' in capsys.readouterr().err


def test_missing_files_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match='Datafile does not exist: nowhere.tsv'):
        arguments.check_file_exists('nowhere.tsv', 'Datafile')
    with pytest.raises(FileNotFoundError, match='output directory does not exist'):
        arguments.output_prefix_and_directory(str(tmp_path / 'missing' / 'net.pt'))
