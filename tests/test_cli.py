"""
End-to-end tests for the PARROT command-line tools: parrot-train, parrot-optimize,
parrot-predict, parrot-cvsplit and parrot-preprocess.

Each test calls the tool's main() function exactly as the installed console
script would, using the example data in /data and writing output to a temporary
directory.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from parrot import process_input_data as pid
from parrot import prediction
from parrot.architectures import HyperParameter
from parrot.checkpoint import load_network, save_network
from parrot.encode_sequence import get_encoder
from parrot.scripts import common, parrot_cvsplit, parrot_optimize, parrot_predict, parrot_preprocess, parrot_train
from parrot.tasks import Task

from tests.helpers import EXAMPLE_TASKS, data_path, make_network

# Few epochs keep the end-to-end runs fast; these tests check plumbing, not accuracy
QUICK_EPOCHS = '2'


def read_lines(path: str | Path) -> list[str]:
    with open(path) as fh:
        return [line.rstrip('\n') for line in fh]


def write_subset(write_lines: Callable[[str, list[str]], str], filename: str, n_lines: int) -> str:
    """Write the first n_lines of an example dataset to tmp_path, for tests that train many networks."""
    lines = [' '.join(line) for line in pid.read_tsv_raw(data_path(filename))[:n_lines]]
    return write_lines(filename, lines)


def run_train(run_cli: Callable[..., Any], data_file: str, output_network: str | Path, datatype: str,
              num_classes: int, *extra_arguments: Any) -> None:
    arguments = ['parrot-train', data_file, output_network, '-d', datatype, '-c', num_classes,
                 '-e', QUICK_EPOCHS, *extra_arguments]
    run_cli(parrot_train.main, arguments)


# ..........................................................................................
# parrot-train

@pytest.mark.parametrize('task_name, figure_suffix', [
    ('seq_class', '_seq_CM.png'),
    ('seq_regress', '_seq_scatterplot.png'),
    ('res_class', '_res_CM.png'),
    ('res_regress', '_res_scatterplot.png'),
])
def test_train_all_tasks(run_cli, tmp_path, task_name, figure_suffix):
    filename, datatype, num_classes = EXAMPLE_TASKS[task_name]
    output_network = tmp_path / 'net.pt'

    run_train(run_cli, data_path(filename), output_network, datatype, num_classes,
              '--include-figs', '--save-splits', '--silent')

    prefix = str(tmp_path / 'net')
    for suffix in ['.pt', '_predictions.tsv', '_performance_stats.txt', '_train_val_loss.png', figure_suffix,
                   '_split_file.txt']:
        assert os.path.isfile(prefix + suffix), suffix

    # Two lines (true and predicted) per test-set sequence
    _, _, test_indices = pid.read_split_file(prefix + '_split_file.txt', num_samples=300)
    predictions = read_lines(prefix + '_predictions.tsv')
    assert len(predictions) == 2 * len(test_indices)

    # The saved network records everything needed to rebuild it
    network, encoder = load_network(str(output_network))
    assert network.name == 'brnn' and network.task == Task(datatype, num_classes)
    assert network.hyperparameter_values == {'hidden_size': 10, 'num_layers': 1, 'attention_pooling': True,
                                             'loglength_feedforward': True}
    assert encoder.name == 'onehot'


def test_train_hyperparameter_flags(run_cli, tmp_path):
    run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
              '-hs', '6', '--num-layers', '2', '--architecture', 'brnn', '--silent')

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.hyperparameter_values == {'hidden_size': 6, 'num_layers': 2, 'attention_pooling': True,
                                             'loglength_feedforward': True}


@pytest.mark.parametrize('architecture, flags, expected', [
    ('cnn', [], {'global_context': True, 'attention_pooling': True, 'loglength_feedforward': True}),
    ('cnn', ['--global-context-off', '--attention-pooling-off', '--loglength-feedforward-off'],
     {'global_context': False, 'attention_pooling': False, 'loglength_feedforward': False}),
    ('unet', ['--attention-pooling-off'], {'global_context': True, 'attention_pooling': False}),
    ('transformer', ['--loglength-feedforward-off'], {'attention_pooling': True, 'loglength_feedforward': False}),
    ('brnn', ['--loglength-feedforward-off'], {'attention_pooling': True, 'loglength_feedforward': False}),
    ('brnn', ['--attention-pooling-off'], {'attention_pooling': False, 'loglength_feedforward': True}),
])
def test_train_on_off_flags(run_cli, tmp_path, architecture, flags, expected):
    run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
              '--architecture', architecture, '--silent', *flags)

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert {name: network.hyperparameter_values[name] for name in expected} == expected


@pytest.mark.parametrize('architecture, flag', [('brnn', '--global-context-off'), ('transformer', '--global-context-off'),
                                                ('composition', '--loglength-feedforward-off')])
def test_train_rejects_on_off_flags_the_architecture_does_not_have(run_cli, tmp_path, architecture, flag):
    with pytest.raises(ValueError, match=f'{flag} is not a hyperparameter'):
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
                  '--architecture', architecture, '--silent', flag)


def test_train_other_architecture(run_cli, tmp_path, test_architecture):
    """A newly registered architecture is available to parrot-train and parrot-predict with no other changes"""
    run_train(run_cli, data_path('res_class_dataset.tsv'), tmp_path / 'net.pt', 'residues', 3,
              '--architecture', 'test_linear', '--hidden-units', '6', '--include-figs', '--silent')

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert isinstance(network, test_architecture)
    assert network.hyperparameter_values == {'hidden_units': 6}
    assert os.path.isfile(tmp_path / 'net_res_CM.png')

    output_file = tmp_path / 'predictions.txt'
    run_cli(parrot_predict.main, ['parrot-predict', data_path('seqfile.txt'), tmp_path / 'net.pt', output_file,
                                  '--silent'])
    for line in read_lines(output_file):
        seq_id, sequence, *labels = line.split()
        assert len(labels) == len(sequence)


@pytest.mark.parametrize('extra_arguments', [
    ['--hidden-units', '6'],                                   # belongs to test_linear, not brnn
    ['--architecture', 'test_linear', '--hidden-size', '6'],   # belongs to brnn, not test_linear
])
def test_train_rejects_hyperparameters_of_other_architectures(run_cli, tmp_path, test_architecture,
                                                              extra_arguments):
    with pytest.raises(ValueError, match='is not a hyperparameter'):
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
                  '--silent', *extra_arguments)


@pytest.mark.parametrize('architecture, flags', [
    ('composition', ['--num-hidden-layers', '0', '--window-radius', '3']),
    ('cnn', ['--channels', '16', '--num-blocks', '2', '--kernel-size', '3', '--pooling', 'max']),
    ('unet', ['--base-channels', '8', '--depth', '2', '--upsampling', 'nearest', '--dropout', '0.1']),
])
@pytest.mark.parametrize('task_name', ['seq_regress', 'res_class'])
def test_train_and_predict_with_each_architecture(run_cli, tmp_path, architecture, flags, task_name):
    filename, datatype, num_classes = EXAMPLE_TASKS[task_name]
    run_train(run_cli, data_path(filename), tmp_path / 'net.pt', datatype, num_classes,
              '--architecture', architecture, *flags, '--silent')

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.name == architecture
    given = dict(zip([flag.lstrip('-').replace('-', '_') for flag in flags[::2]], flags[1::2]))
    assert {name: str(network.hyperparameter_values[name]) for name in given} == given

    output_file = tmp_path / 'predictions.txt'
    run_cli(parrot_predict.main, ['parrot-predict', data_path('seqfile.txt'), tmp_path / 'net.pt', output_file,
                                  '--silent'])
    assert len(read_lines(output_file)) == len(pid.read_tsv_raw(data_path('seqfile.txt')))


@pytest.mark.parametrize('flags', [['--architecture', 'cnn', '--pooling', 'sum'],
                                   ['--architecture', 'cnn', '--kernel-size', '4'],
                                   ['--architecture', 'unet', '--upsampling', 'bilinear']])
def test_train_rejects_invalid_choices(run_cli, tmp_path, flags):
    with pytest.raises(SystemExit):
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1, *flags)


@pytest.mark.parametrize('second, problem', [
    (HyperParameter('num_layers', float, 2.0, 'help', cli_aliases=('-nl',)), 'different types'),
    (HyperParameter('num_layers', int, 2, 'help'), 'different command-line aliases'),
])
def test_shared_hyperparameters_must_be_able_to_share_a_flag(second, problem):
    first = HyperParameter('num_layers', int, 1, 'help', cli_aliases=('-nl',))
    with pytest.raises(ValueError, match=problem):
        common.check_shared_declarations('num_layers', [first, second])


def test_batch_size_warnings(capsys):
    common.warn_about_batch_size(32, 1000)
    assert capsys.readouterr().out == ''

    common.warn_about_batch_size(1024, 100000)
    assert 'very large batch size' in capsys.readouterr().out

    common.warn_about_batch_size(32, 200)
    assert 'large relative to the number of samples' in capsys.readouterr().out


def test_shared_on_off_hyperparameters_must_share_a_default():
    on, off = HyperParameter('extra', bool, True, 'help'), HyperParameter('extra', bool, False, 'help')
    with pytest.raises(ValueError, match='different defaults'):
        common.check_shared_declarations('extra', [on, off])


@pytest.mark.parametrize('flags', [['--architecture', 'cnn', '--dropout', '1.5'],
                                   ['--architecture', 'composition', '--num-hidden-layers', '-1'],
                                   ['--architecture', 'linear', '--alpha', '0']])
def test_train_rejects_out_of_range_values(run_cli, tmp_path, flags):
    with pytest.raises(ValueError):
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1, *flags)


@pytest.mark.parametrize('architecture, flags', [
    ('linear', ['--alpha', '0.1']),
    ('svm', ['--kernel', 'linear', '--svm-c', '10']),
    ('random_forest', ['--num-trees', '20', '--features', 'dipeptides']),
    ('knn', ['--num-neighbors', '3', '--neighbor-weights', 'distance']),
])
@pytest.mark.parametrize('task_name', ['seq_regress', 'seq_class'])
def test_train_and_predict_one_step_models(run_cli, tmp_path, architecture, flags, task_name, capsys):
    filename, datatype, num_classes = EXAMPLE_TASKS[task_name]
    run_train(run_cli, data_path(filename), tmp_path / 'net.pt', datatype, num_classes,
              '--architecture', architecture, *flags, '--include-figs', '--ignore-warnings')

    assert 'fitted in one step' in capsys.readouterr().out
    # There is no loss curve to plot for a model fitted in one step, but the test-set figure is made
    assert not os.path.isfile(tmp_path / 'net_train_val_loss.png')
    assert os.path.isfile(tmp_path / ('net_seq_CM.png' if num_classes > 1 else 'net_seq_scatterplot.png'))

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.name == architecture

    output_file = tmp_path / 'predictions.txt'
    probabilistic = ['--probabilistic-classification'] if num_classes > 1 else []
    run_cli(parrot_predict.main, ['parrot-predict', data_path('seqfile.txt'), tmp_path / 'net.pt', output_file,
                                  '--silent', *probabilistic])
    values_per_line = {len(line.split()) - 2 for line in read_lines(output_file)}
    assert values_per_line == {num_classes}


def test_one_step_models_reject_residue_data(run_cli, tmp_path):
    with pytest.raises(ValueError, match='only supports sequence-mapped data'):
        run_train(run_cli, data_path('res_regress_dataset.tsv'), tmp_path / 'net.pt', 'residues', 1,
                  '--architecture', 'svm', '--silent')


@pytest.mark.parametrize('num_classes', [1, 3])
def test_train_with_missing_residue_labels(run_cli, tmp_path, write_lines, num_classes, capsys):
    """Residues labelled nan are ignored in training, metrics and figures, and reported as nan"""
    rng = np.random.default_rng(0)
    lines = []
    for line in pid.read_tsv_raw(data_path('res_class_dataset.tsv' if num_classes > 1 else 'res_regress_dataset.tsv')):
        values = line[2:]
        for position in rng.choice(len(values), size=len(values) // 4, replace=False):
            values[position] = 'nan'
        lines.append(' '.join(line[:2] + values))
    data_file = write_lines('data_with_missing.tsv', lines)

    run_train(run_cli, data_file, tmp_path / 'net.pt', 'residues', num_classes, '--include-figs')

    assert 'are missing' in capsys.readouterr().out
    stats = (tmp_path / 'net_performance_stats.txt').read_text()
    assert 'nan' not in stats
    true_lines = [line for line in read_lines(tmp_path / 'net_predictions.tsv') if '_TRUE' in line]
    assert any(' nan' in line for line in true_lines)
    assert os.path.isfile(tmp_path / ('net_res_CM.png' if num_classes > 1 else 'net_res_scatterplot.png'))


def test_train_rejects_unknown_architecture(run_cli, tmp_path):
    with pytest.raises(SystemExit):
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
                  '--architecture', 'not_an_architecture')


def test_train_is_reproducible_with_seed(run_cli, tmp_path):
    for name in ['first', 'second']:
        run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / f'{name}.pt', 'sequence', 1,
                  '--seed', '42', '--silent')

    assert read_lines(tmp_path / 'first_predictions.tsv') == read_lines(tmp_path / 'second_predictions.tsv')
    first = torch.load(str(tmp_path / 'first.pt'))['state_dict']
    second = torch.load(str(tmp_path / 'second.pt'))['state_dict']
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_train_with_split_file(run_cli, tmp_path):
    run_train(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3,
              '--split', data_path('splitfile.txt'), '--silent')

    names = [line.split()[0] for line in read_lines(tmp_path / 'net_predictions.tsv')]
    expected = [line[0] for line in pid.read_tsv_raw(data_path('seq_class_dataset.tsv'))[200:300]]
    assert names[::2] == [name + '_TRUE' for name in expected]


def test_train_exclude_seq_id(run_cli, tmp_path, write_lines):
    lines = [' '.join(line[1:]) for line in pid.read_tsv_raw(data_path('seq_class_dataset.tsv'))]
    data_file = write_lines('noids.tsv', lines)

    run_train(run_cli, data_file, tmp_path / 'net.pt', 'sequence', 3, '--exclude-seq-id', '--silent')

    assert read_lines(tmp_path / 'net_predictions.tsv')[0].startswith('test0_TRUE ')


@pytest.mark.parametrize('encoding, data_file, datatype, num_classes', [
    ('biophysics', 'res_regress_dataset.tsv', 'residues', 1),
    ('encoding_example.txt', 'seq_class_dataset.tsv', 'sequence', 3),
    ('nucleotide_encoding.txt', 'nucleotide_dataset.tsv', 'sequence', 1),
])
def test_train_other_encodings(run_cli, tmp_path, encoding, data_file, datatype, num_classes):
    encode_argument = encoding if encoding == 'biophysics' else data_path(encoding)

    run_train(run_cli, data_path(data_file), tmp_path / 'net.pt', datatype, num_classes,
              '--encode', encode_argument, '--silent')

    # Predictions are written with the original sequences
    original_sequences = {line[1] for line in pid.read_tsv_raw(data_path(data_file))}
    predicted_sequences = {line.split()[1] for line in read_lines(tmp_path / 'net_predictions.tsv')}
    assert predicted_sequences <= original_sequences


def test_train_probabilistic_classification(run_cli, tmp_path):
    run_train(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3,
              '--probabilistic-classification', '--include-figs', '--silent')

    assert os.path.isfile(tmp_path / 'net_ROC_curve.png')
    assert os.path.isfile(tmp_path / 'net_PR_curve.png')
    assert 'Area under ROC' in (tmp_path / 'net_performance_stats.txt').read_text()

    predicted_line = read_lines(tmp_path / 'net_predictions.tsv')[1]
    probabilities = [float(value) for value in predicted_line.split()[2:]]
    assert len(probabilities) == 3 and sum(probabilities) == pytest.approx(1.0, abs=1e-3)


def test_train_no_stats(run_cli, tmp_path):
    run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 1,
              '--no-stats', '--silent')

    assert not os.path.isfile(tmp_path / 'net_performance_stats.txt')
    assert os.path.isfile(tmp_path / 'net_predictions.tsv')


def test_train_auto_stop(run_cli, tmp_path, write_lines):
    data_file = write_subset(write_lines, 'seq_regress_dataset.tsv', 40)
    run_cli(parrot_train.main, ['parrot-train', data_file, tmp_path / 'net.pt',
                                '-d', 'sequence', '-c', '1', '-e', '1', '--stop', 'auto', '-lr', '0.05',
                                '--silent'])
    assert os.path.isfile(tmp_path / 'net.pt')


def test_train_silent_prints_nothing(run_cli, tmp_path, capsys):
    run_train(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3, '--silent')
    assert capsys.readouterr().out == ''


def test_train_verbose_prints_settings(run_cli, tmp_path, capsys):
    run_train(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3, '--verbose',
              '--ignore-warnings')

    output = capsys.readouterr().out
    assert 'device' in output and 'hidden_size   : 10' in output and 'architecture  : brnn' in output
    assert 'Test Loss' in output


def test_train_dotted_output_name(run_cli, tmp_path):
    run_train(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'net.v2.pt', 'sequence', 1, '--silent')
    assert os.path.isfile(tmp_path / 'net.v2_predictions.tsv')


@pytest.mark.parametrize('extra_arguments, error', [
    (['-d', 'sequence', '-c', '3', '-lr', '1.5'], ValueError),
    (['-d', 'sequence', '-c', '3', '--stop', 'sometimes'], SystemExit),  # not one of argparse's choices
    (['-d', 'sequence', '-c', '3', '-b', '0'], ValueError),
    (['-d', 'sequence', '-c', '3', '-hs', '0'], ValueError),
    (['-d', 'sequence', '-c', '1', '--probabilistic-classification'], ValueError),
    (['-d', 'residues', '-c', '3', '--probabilistic-classification'], ValueError),
    (['-d', 'residues', '-c', '3'], Exception),  # sequence-mapped data passed as residues
    (['-d', 'sequence', '-c', '3', '--split', 'missing_split.txt'], FileNotFoundError),
    (['-d', 'sequence', '-c', '3', '--encode', 'missing_encoding.txt'], FileNotFoundError),
])
def test_train_invalid_arguments(run_cli, tmp_path, extra_arguments, error):
    with pytest.raises(error):
        run_cli(parrot_train.main, ['parrot-train', data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt',
                                    '-e', QUICK_EPOCHS, '--silent', *extra_arguments])


@pytest.mark.parametrize('encoding', ['onehot', 'biophysics'])
def test_train_rejects_invalid_residues_before_training(run_cli, tmp_path, write_lines, encoding):
    # Whichever set the bad sequence lands in, the error comes before any training (so no network is saved)
    lines = [f's{i} ACDEFGHIKLMNPQ {i % 3}' for i in range(30)] + ['bad ACDEFGHiKLMNPQ 1']
    with pytest.raises(ValueError, match='Invalid amino acid detected: i'):
        run_cli(parrot_train.main, ['parrot-train', write_lines('data.tsv', lines), tmp_path / 'net.pt',
                                    '-d', 'sequence', '-c', '3', '-e', QUICK_EPOCHS, '--encode', encoding,
                                    '--silent'])
    assert not (tmp_path / 'net.pt').exists()


def test_train_missing_data_file(run_cli, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_cli(parrot_train.main, ['parrot-train', tmp_path / 'missing.tsv', tmp_path / 'net.pt',
                                    '-d', 'sequence', '-c', '1'])


def test_train_missing_output_directory(run_cli, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_cli(parrot_train.main, ['parrot-train', data_path('seq_class_dataset.tsv'),
                                    tmp_path / 'missing' / 'net.pt', '-d', 'sequence', '-c', '3'])


# ..........................................................................................
# parrot-optimize (the hyperparameter search runs for real, on small data with few trials and epochs)

def run_optimize(run_cli: Callable[..., Any], data_file: str, output_network: str | Path, datatype: str,
                 num_classes: int, *extra_arguments: Any) -> None:
    run_cli(parrot_optimize.main, ['parrot-optimize', data_file, output_network, '-d', datatype, '-c', num_classes,
                                   '--seed', '1', '--silent', *extra_arguments])


def read_trials(path: str | Path) -> list[dict[str, str]]:
    """The rows of a _hyperparameter_trials.tsv file as {column: value} dictionaries."""
    lines = [line for line in read_lines(path) if not line.startswith('#')]
    header = lines[0].split('\t')
    return [dict(zip(header, line.split('\t'))) for line in lines[1:]]


def test_optimize_end_to_end(run_cli, tmp_path, write_lines):
    data_file = write_subset(write_lines, 'seq_regress_dataset.tsv', 80)
    run_optimize(run_cli, data_file, tmp_path / 'net.pt', 'sequence', 1, '-e', '4', '--trials', '6', '--include-figs')

    for suffix in ['.pt', '_predictions.tsv', '_performance_stats.txt', '_train_val_loss.png',
                   '_optimal_hyperparams.txt', '_hyperparameter_trials.tsv']:
        assert os.path.isfile(str(tmp_path / 'net') + suffix), suffix

    trials = read_trials(tmp_path / 'net_hyperparameter_trials.tsv')
    assert len(trials) == 6 and trials[0]['origin'] == 'default'
    assert {trial['status'] for trial in trials} <= {'completed', 'stopped'}

    # The saved network is the best trial, with the hyperparameters written to _optimal_hyperparams.txt
    best = min((trial for trial in trials if trial['status'] == 'completed'),
               key=lambda trial: float(trial['val_loss_4_epochs']))
    lines = read_lines(tmp_path / 'net_optimal_hyperparams.txt')
    assert lines[:4] == ['architecture: brnn', f"learning_rate: {best['learning_rate']}",
                         f"hidden_size: {best['hidden_size']}", f"num_layers: {best['num_layers']}"]
    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.hyperparameter_values == {'hidden_size': int(best['hidden_size']),
                                             'num_layers': int(best['num_layers']), 'attention_pooling': True,
                                             'loglength_feedforward': True}


def test_optimize_is_reproducible_with_seed(run_cli, tmp_path, write_lines):
    data_file = write_subset(write_lines, 'res_regress_dataset.tsv', 60)
    for name in ['first', 'second']:
        run_optimize(run_cli, data_file, tmp_path / f'{name}.pt', 'residues', 1, '-e', '3', '--trials', '4',
                     '--architecture', 'cnn', '--channels', '8')
    assert read_lines(tmp_path / 'first_hyperparameter_trials.tsv') == \
        read_lines(tmp_path / 'second_hyperparameter_trials.tsv')


def test_optimize_other_architecture(run_cli, tmp_path, write_lines, test_architecture):
    data_file = write_subset(write_lines, 'res_regress_dataset.tsv', 60)
    run_optimize(run_cli, data_file, tmp_path / 'net.pt', 'residues', 1, '-e', '2', '--trials', '3',
                 '--architecture', 'test_linear')

    assert read_lines(tmp_path / 'net_optimal_hyperparams.txt')[0] == 'architecture: test_linear'
    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert isinstance(network, test_architecture)


def test_optimize_with_fixed_hyperparameters(run_cli, tmp_path, write_lines):
    """Hyperparameters given to parrot-optimize are fixed rather than searched"""
    data_file = write_subset(write_lines, 'seq_regress_dataset.tsv', 60)
    run_optimize(run_cli, data_file, tmp_path / 'net.pt', 'sequence', 1, '-e', '2', '--trials', '3',
                 '--architecture', 'cnn', '--channels', '12', '--pooling', 'max', '--attention-pooling-off')

    # Only the learning rate, num_blocks, kernel_size and dropout were searched
    header = [line for line in read_lines(tmp_path / 'net_hyperparameter_trials.tsv') if not line.startswith('#')][0]
    assert header.split('\t')[-4:] == ['learning_rate', 'num_blocks', 'kernel_size', 'dropout']
    lines = read_lines(tmp_path / 'net_optimal_hyperparams.txt')
    assert lines[0] == 'architecture: cnn'
    assert 'channels: 12' in lines and 'pooling: max' in lines and 'attention_pooling: False' in lines

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.hyperparameter_values['channels'] == 12 and network.hyperparameter_values['pooling'] == 'max'


def test_optimize_with_cross_validation(run_cli, tmp_path, write_lines):
    data_file = write_subset(write_lines, 'seq_class_dataset.tsv', 60)
    run_optimize(run_cli, data_file, tmp_path / 'net.pt', 'sequence', 3, '-e', '2', '--trials', '3',
                 '--cv-folds', '3', '--architecture', 'composition')

    assert len(read_trials(tmp_path / 'net_hyperparameter_trials.tsv')) == 3
    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.name == 'composition'
    assert os.path.isfile(tmp_path / 'net_predictions.tsv')


def test_optimize_one_step_model(run_cli, tmp_path):
    run_optimize(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3,
                 '--architecture', 'random_forest', '--num-trees', '20', '--trials', '4')

    # No learning rate is searched or recorded for a model fitted in one step
    lines = read_lines(tmp_path / 'net_optimal_hyperparams.txt')
    assert lines[0] == 'architecture: random_forest' and 'num_trees: 20' in lines
    assert not any(line.startswith('learning_rate') for line in lines)
    trials = read_trials(tmp_path / 'net_hyperparameter_trials.tsv')
    assert len(trials) == 4 and set(trials[0]) >= {'val_loss', 'min_samples_leaf', 'features'}

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.name == 'random_forest'


def test_optimize_max_iter_is_an_older_name_for_trials(run_cli, tmp_path, write_lines):
    data_file = write_subset(write_lines, 'seq_regress_dataset.tsv', 40)
    run_optimize(run_cli, data_file, tmp_path / 'net.pt', 'sequence', 1, '-e', '1', '--max-iter', '2')
    assert len(read_trials(tmp_path / 'net_hyperparameter_trials.tsv')) == 2


@pytest.mark.parametrize('extra_arguments', [['--trials', '0'], ['--trials', '3', '--max-iter', '3'],
                                             ['--min-epochs', '0'], ['-e', '5', '--min-epochs', '6'],
                                             ['--reduction-factor', '1'], ['--cv-folds', '1']])
def test_optimize_invalid_search_arguments(run_cli, tmp_path, extra_arguments):
    with pytest.raises(ValueError):
        run_optimize(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt', 'sequence', 3, *extra_arguments)


def test_optimize_invalid_set_fractions(run_cli, tmp_path):
    with pytest.raises(ValueError, match='sum to 1'):
        run_cli(parrot_optimize.main, ['parrot-optimize', data_path('seq_class_dataset.tsv'), tmp_path / 'net.pt',
                                       '-d', 'sequence', '-c', '3', '--set-fractions', '0.5', '0.1', '0.1'])


# ..........................................................................................
# parrot-predict

def save_test_network(tmp_path: Path, datatype: str, num_classes: int, legacy: bool = False,
                      encoding: str = 'onehot') -> tuple[torch.nn.Module, str]:
    """Save a small random BRNN, either as a checkpoint or (legacy=True) as bare weights like older PARROT versions."""
    encoder = get_encoder(encoding)
    network = make_network(datatype, num_classes, input_size=len(encoder), hidden_size=6, num_layers=2)
    path = str(tmp_path / 'trained.pt')
    if legacy:
        torch.save(network.state_dict(), path)
    else:
        save_network(path, network, encoder)
    return network, path


def run_predict(run_cli: Callable[..., Any], seq_file: str, network_file: str | Path, output_file: str | Path,
                *extra_arguments: Any) -> None:
    run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, output_file, '--silent',
                                  *extra_arguments])


@pytest.mark.parametrize('legacy', [False, True], ids=['checkpoint', 'legacy-weights'])
@pytest.mark.parametrize('datatype, num_classes', [('sequence', 1), ('sequence', 3), ('residues', 1), ('residues', 3)])
def test_predict_output_format(run_cli, tmp_path, datatype, num_classes, legacy):
    network, network_file = save_test_network(tmp_path, datatype, num_classes, legacy=legacy)
    output_file = tmp_path / 'predictions.txt'

    # Networks saved by older versions of PARROT need the old flags; new networks need none
    legacy_flags = ['-d', datatype, '-c', num_classes, '-hs', '6', '-nl', '2'] if legacy else []
    run_predict(run_cli, data_path('seqfile.txt'), network_file, output_file, *legacy_flags)

    input_lines = pid.read_tsv_raw(data_path('seqfile.txt'))
    output_lines = [line.split() for line in read_lines(output_file)]
    assert [line[:2] for line in output_lines] == input_lines

    raw_outputs = prediction.predict_sequences(network, [line[1] for line in input_lines], get_encoder('onehot'),
                                                  'cpu')
    for seq_id, sequence, *values in output_lines:
        raw = raw_outputs[sequence]
        if datatype == 'sequence' and num_classes == 1:
            np.testing.assert_allclose(float(values[0]), raw[0], rtol=1e-5)
        elif datatype == 'sequence':
            assert values == [str(np.argmax(raw))]
        elif num_classes == 1:
            np.testing.assert_allclose([float(v) for v in values], raw.flatten(), rtol=1e-5)
        else:
            assert values == [str(label) for label in np.argmax(raw, axis=1)]


def test_predict_uses_stored_encoding(run_cli, tmp_path):
    _, network_file = save_test_network(tmp_path, 'sequence', 1, encoding='biophysics')
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, data_path('seqfile.txt'), network_file, output_file)

    assert len(read_lines(output_file)) == len(pid.read_tsv_raw(data_path('seqfile.txt')))


def test_predict_legacy_network_needs_datatype(run_cli, tmp_path):
    _, network_file = save_test_network(tmp_path, 'sequence', 1, legacy=True)
    with pytest.raises(ValueError, match='datatype'):
        run_predict(run_cli, data_path('seqfile.txt'), network_file, tmp_path / 'predictions.txt')


@pytest.mark.parametrize('extra_arguments', [['-d', 'residues'], ['-c', '3'], ['--encode', 'biophysics']])
def test_predict_rejects_arguments_that_contradict_the_network(run_cli, tmp_path, extra_arguments):
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    with pytest.raises(ValueError, match='does not match'):
        run_predict(run_cli, data_path('seqfile.txt'), network_file, tmp_path / 'predictions.txt', *extra_arguments)


def test_predict_probabilistic_classification(run_cli, tmp_path):
    _, network_file = save_test_network(tmp_path, 'sequence', 3)
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, data_path('seqfile.txt'), network_file, output_file, '--probabilistic-classification')

    for line in read_lines(output_file):
        probabilities = [float(value) for value in line.split()[2:]]
        assert len(probabilities) == 3 and sum(probabilities) == pytest.approx(1.0, abs=1e-3)


def test_predict_without_ids(run_cli, tmp_path):
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, data_path('seqfile_noIDs.txt'), network_file, output_file, '--exclude-seq-id')

    sequences = [line[0] for line in pid.read_tsv_raw(data_path('seqfile_noIDs.txt'))]
    assert [line.split()[0] for line in read_lines(output_file)] == sequences


def test_predict_keeps_every_line_for_duplicate_sequences(run_cli, tmp_path, write_lines):
    """Duplicate sequences under different IDs must each get their own output line"""
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    seq_file = write_lines('seqs.txt', ['first ACDEFG', 'second KLMNP', 'third ACDEFG'])
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, seq_file, network_file, output_file)

    output_lines = [line.split() for line in read_lines(output_file)]
    assert [line[:2] for line in output_lines] == [['first', 'ACDEFG'], ['second', 'KLMNP'], ['third', 'ACDEFG']]
    assert output_lines[0][2] == output_lines[2][2]


def test_predict_skips_blank_and_comment_lines(run_cli, tmp_path, write_lines):
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    seq_file = write_lines('seqs.txt', ['# my sequences', 'a ACDEFG', '', 'b KLMNP', '', ''])
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, seq_file, network_file, output_file)

    assert [line.split()[0] for line in read_lines(output_file)] == ['a', 'b']


def test_predict_rejects_wrong_column_count(run_cli, tmp_path, write_lines):
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    seq_file = write_lines('seqs.txt', ['ACDEFG', 'KLMNP'])

    with pytest.raises(ValueError, match='--exclude-seq-id'):
        run_predict(run_cli, seq_file, network_file, tmp_path / 'predictions.txt')


def test_predict_rejects_probabilistic_regression(run_cli, tmp_path):
    _, network_file = save_test_network(tmp_path, 'sequence', 1)
    with pytest.raises(ValueError, match='Probabilistic classification'):
        run_predict(run_cli, data_path('seqfile.txt'), network_file, tmp_path / 'predictions.txt',
                    '--probabilistic-classification')


def test_predict_on_network_from_train(run_cli, tmp_path):
    """A network written by parrot-train can be used directly by parrot-predict"""
    run_train(run_cli, data_path('res_class_dataset.tsv'), tmp_path / 'net.pt', 'residues', 3,
              '-hs', '6', '-nl', '2', '--silent')
    output_file = tmp_path / 'predictions.txt'

    run_predict(run_cli, data_path('seqfile.txt'), tmp_path / 'net.pt', output_file)

    for line in read_lines(output_file):
        seq_id, sequence, *labels = line.split()
        assert len(labels) == len(sequence) and set(labels) <= {'0', '1', '2'}


# ..........................................................................................
# parrot-cvsplit

def read_split_files(prefix: str, k: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [pid.read_split_file(f'{prefix}_cv{i}.txt') for i in range(k)]


def test_cvsplit(run_cli, tmp_path):
    k = 5
    run_cli(parrot_cvsplit.main, ['parrot-cvsplit', data_path('seq_class_dataset.tsv'), tmp_path / 'splits',
                                  '-k', k, '-t', '0.75'])

    splits = read_split_files(str(tmp_path / 'splits'), k)
    all_test = []
    for train, val, test in splits:
        assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(300))
        assert len(train) == pytest.approx(0.75 * (len(train) + len(val)), abs=1)
        all_test.extend(test.tolist())

    # The K test sets partition the dataset
    assert sorted(all_test) == list(range(300))


def test_cvsplit_ignores_blank_and_comment_lines(run_cli, tmp_path, write_lines):
    """Indices must refer to data lines exactly as parrot-train reads them"""
    data_file = write_lines('data.tsv', ['# header comment', 'a ACD 1', '', 'b EFG 0', 'c HIK 1', '', 'd KLM 0'])
    run_cli(parrot_cvsplit.main, ['parrot-cvsplit', data_file, tmp_path / 'splits', '-k', '2', '-t', '0.5'])

    for train, val, test in read_split_files(str(tmp_path / 'splits'), 2):
        assert sorted(np.concatenate([train, val, test]).tolist()) == [0, 1, 2, 3]


def test_cvsplit_is_reproducible_with_seed(run_cli, tmp_path):
    for name in ['first', 'second']:
        run_cli(parrot_cvsplit.main, ['parrot-cvsplit', data_path('seq_class_dataset.tsv'), tmp_path / name,
                                      '-k', '3', '--seed', '9'])

    for i in range(3):
        assert read_lines(tmp_path / f'first_cv{i}.txt') == read_lines(tmp_path / f'second_cv{i}.txt')


def test_cvsplit_rejects_too_few_sequences(run_cli, tmp_path, write_lines):
    # 5 sequences in 5 folds leave 4 per fold for training and validation, and -t 0.8 gives all 4 to training
    data_file = write_lines('data.tsv', [f's{i} ACDEF {i}.0' for i in range(5)])
    with pytest.raises(ValueError, match='The validation set is empty'):
        run_cli(parrot_cvsplit.main, ['parrot-cvsplit', data_file, tmp_path / 'splits', '-k', '5'])
    assert not list(tmp_path.glob('splits_cv*.txt'))


@pytest.mark.parametrize('extra_arguments', [['-k', '1'], ['-t', '1.0'], ['-t', '0']])
def test_cvsplit_invalid_arguments(run_cli, tmp_path, extra_arguments):
    with pytest.raises(ValueError):
        run_cli(parrot_cvsplit.main, ['parrot-cvsplit', data_path('seq_class_dataset.tsv'), tmp_path / 'splits',
                                      *extra_arguments])


# ..........................................................................................
# parrot-preprocess

def run_preprocess(run_cli: Callable[..., Any], input_file: str, output_file: str | Path, datatype: str, mode: str,
                   *extra_arguments: Any) -> None:
    run_cli(parrot_preprocess.main, ['parrot-preprocess', input_file, output_file, '-d', datatype, '-m', mode,
                                     *extra_arguments])


def test_preprocess_passthrough(run_cli, tmp_path):
    output_file = tmp_path / 'out.tsv'
    run_preprocess(run_cli, data_path('seq_class_dataset.tsv'), output_file, 'sequence', 'classification')
    assert pid.read_tsv_raw(str(output_file)) == pid.read_tsv_raw(data_path('seq_class_dataset.tsv'))


def test_preprocess_recode_classes(run_cli, tmp_path, write_lines, monkeypatch):
    monkeypatch.chdir(tmp_path / '..')  # nothing should be written to the working directory
    input_file = write_lines('in.tsv', ['a ACD 10', 'b EFG 30', 'c HIK 20', 'd KLM 10.0'])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'classification', '--recode-classes')

    assert [line[2] for line in pid.read_tsv_raw(str(output_file))] == ['0', '2', '1', '0']
    mapping = (tmp_path / 'out_class_mapping.txt').read_text()
    assert '10 = 0' in mapping and '10.0 = 0' in mapping and '20 = 1' in mapping and '30 = 2' in mapping
    assert 'Number of classes: 3' in mapping
    assert not (tmp_path / '..' / 'class_mapping.txt').exists()


def test_preprocess_recode_classes_keeps_missing_residue_labels(run_cli, tmp_path, write_lines):
    input_file = write_lines('in.tsv', ['a ACD 2 nan 5', 'b EF nan 2'])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'residues', 'classification', '--recode-classes')

    assert pid.read_tsv_raw(str(output_file)) == [['a', 'ACD', '0', 'nan', '1'], ['b', 'EF', 'nan', '0']]


def test_preprocess_remove_line_duplicates(run_cli, tmp_path, write_lines):
    input_file = write_lines('in.tsv', ['a ACD 1', 'a ACD 1', 'b ACD 0', 'c EFG 1'])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'classification', '--remove-line-duplicates')

    assert sorted(pid.read_tsv_raw(str(output_file))) == [['a', 'ACD', '1'], ['b', 'ACD', '0'], ['c', 'EFG', '1']]


def test_preprocess_remove_sequence_duplicates_verbose(run_cli, tmp_path, write_lines, capsys):
    input_file = write_lines('in.tsv', ['a ACD 1', 'b ACD 0', 'c EFG 1'])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'classification',
                   '--remove-sequence-duplicates', '--verbose')

    assert pid.read_tsv_raw(str(output_file)) == [['c', 'EFG', '1']]
    assert '1 sequences appear more than once' in capsys.readouterr().out


def test_preprocess_sequence_class_balance(run_cli, tmp_path, write_lines):
    lines = [f'a{i} ACD 0' for i in range(60)] + [f'b{i} EFG 1' for i in range(20)]
    input_file = write_lines('in.tsv', lines)
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'classification', '--class-balance',
                   '--cb-max-fraction-removed', '0.6', '--seed', '3')

    labels = [line[2] for line in pid.read_tsv_raw(str(output_file))]
    assert labels.count('1') == 20
    assert labels.count('0') / labels.count('1') <= 1.4


def write_imbalanced_residue_file(write_lines: Callable[[str, list[str]], str], n_sequences: int = 100) -> str:
    rng = np.random.default_rng(1)
    lines = []
    for i in range(n_sequences):
        labels = rng.choice(3, size=10, p=[0.6, 0.3, 0.1])
        lines.append(f'seq{i} AAAAAAAAAA ' + ' '.join(str(label) for label in labels))
    return write_lines('in.tsv', lines)


def output_positions(input_file: str, output_file: str | Path) -> list[int]:
    """Return the position in the input file of every line in the output file."""
    input_ids = [line[0] for line in pid.read_tsv_raw(input_file)]
    return [input_ids.index(line[0]) for line in pid.read_tsv_raw(str(output_file))]


def test_preprocess_residue_class_balance_keeps_input_order(run_cli, tmp_path, write_lines):
    input_file = write_imbalanced_residue_file(write_lines)
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'residues', 'classification', '--class-balance',
                   '--cb-block-size', '10', '--seed', '3')

    positions = output_positions(input_file, output_file)
    assert positions == sorted(positions)
    assert len(positions) < 100
    assert os.path.isfile(tmp_path / 'out_class_balance_trace.csv')


def test_preprocess_residue_class_balance_shuffle_off_keeps_order(run_cli, tmp_path, write_lines):
    input_file = write_imbalanced_residue_file(write_lines)
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'residues', 'classification', '--class-balance',
                   '--cb-block-size', '10', '--cb-shuffle-off', '--seed', '3')

    positions = output_positions(input_file, output_file)
    assert positions == sorted(positions)
    assert len(positions) < 100


@pytest.mark.parametrize('extra_arguments', [
    ['-d', 'residue', '-m', 'classification'],
    ['-d', 'sequence', '-m', 'classify'],
    ['-d', 'sequence', '-m', 'regression', '--recode-classes'],
    ['-d', 'sequence', '-m', 'regression', '--class-balance'],
    ['-d', 'sequence', '-m', 'classification', '--remove-line-duplicates', '--remove-sequence-duplicates'],
])
def test_preprocess_invalid_arguments(run_cli, tmp_path, extra_arguments):
    # A datatype or mode that is not one of argparse's choices stops with a usage message (SystemExit)
    with pytest.raises((ValueError, SystemExit)):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', data_path('seq_class_dataset.tsv'),
                                         tmp_path / 'out.tsv', *extra_arguments])


@pytest.mark.parametrize('batching', ['length', 'random'])
def test_train_batching_is_reproducible(run_cli, tmp_path, batching):
    filename, datatype, num_classes = EXAMPLE_TASKS['res_regress']
    predictions = []
    for run in range(2):
        run_train(run_cli, data_path(filename), tmp_path / f'net{run}.pt', datatype, num_classes,
                  '--batching', batching, '--seed', '5', '--silent')
        predictions.append((tmp_path / f'net{run}_predictions.tsv').read_text())
    assert predictions[0] == predictions[1]


def test_train_rejects_unknown_batching(run_cli, tmp_path):
    filename, datatype, num_classes = EXAMPLE_TASKS['seq_regress']
    with pytest.raises(SystemExit):
        run_train(run_cli, data_path(filename), tmp_path / 'net.pt', datatype, num_classes, '--batching', 'sorted',
                  '--silent')


@pytest.mark.parametrize('task_name', ['seq_regress', 'res_class'])
def test_train_with_standardized_encoding_then_predict(run_cli, tmp_path, task_name):
    filename, datatype, num_classes = EXAMPLE_TASKS[task_name]
    network_file = tmp_path / 'net.pt'
    run_train(run_cli, data_path(filename), network_file, datatype, num_classes,
              '--encode', 'biophysics', '--standardize-encoding', '--silent')

    network, encoder = load_network(str(network_file))
    assert encoder.to_config()['name'] == 'standardized'
    assert encoder.to_config()['base'] == {'name': 'biophysics'}

    # parrot-predict applies the saved standardization: its predictions for the test
    # sequences match the test-set predictions written during training
    test_lines = read_lines(tmp_path / 'net_predictions.tsv')
    sequences = [line.split()[1] for line in test_lines[1::2]]
    expected = [np.array(line.split()[2:], dtype=float) for line in test_lines[1::2]]
    seq_file = tmp_path / 'test_sequences.txt'
    seq_file.write_text(''.join(f'seq{i} {sequence}\n' for i, sequence in enumerate(sequences)))

    run_predict(run_cli, str(seq_file), network_file, tmp_path / 'out.txt')
    predicted = [np.array(line.split()[2:], dtype=float) for line in read_lines(tmp_path / 'out.txt')]
    for found, wanted in zip(predicted, expected):
        np.testing.assert_allclose(found, wanted, atol=1e-4)

    # Naming the base encoding is accepted; naming a different one is not
    run_predict(run_cli, str(seq_file), network_file, tmp_path / 'out2.txt', '--encode', 'biophysics')
    with pytest.raises(ValueError, match='does not match'):
        run_predict(run_cli, str(seq_file), network_file, tmp_path / 'out3.txt', '--encode', 'onehot')
