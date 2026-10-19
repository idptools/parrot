"""
Tests for training on regression values with uncertainties (--uncertainties): reading 'value,SD' tokens, the
Gaussian loss and the model error, the simple models' weighted fits, the metrics, and the commands.
"""

import math

import numpy as np
import pytest
import torch

from parrot import process_input_data as pid
from parrot import train_network
from parrot.architectures import build_network
from parrot.checkpoint import load_network, save_network
from parrot.encode_sequence import get_encoder
from parrot.metrics import uncertainty_metrics
from parrot.parrot_exceptions import IOExceptionParrot
from parrot.scripts import parrot_optimize, parrot_preprocess, parrot_train
from parrot.tasks import Task
from parrot.uncertainties import combine_measurements, fit_model_error, gaussian_nll, model_error_of, set_model_error

from tests.helpers import data_path, training_command

SEQUENCE_TASK = Task('sequence', 1, uncertainties=True)
RESIDUE_TASK = Task('residues', 1, uncertainties=True)

# Simulated residuals for checking the model error: how many, and the error added beyond each value's SD
SIMULATED_VALUES = 20_000
TRUE_MODEL_ERROR = 0.3

# The model error recovered from SIMULATED_VALUES residuals should be this close to the true one (its standard
# error is about 1%)
MODEL_ERROR_TOLERANCE = 0.05


def example_sd(index: int) -> float:
    """A different SD for each line of an example dataset (0.05 to 0.45)."""
    return 0.05 + 0.1 * (index % 5)


@pytest.fixture
def uncertain_dataset(write_lines):
    """Write an example regression dataset ('seq_regress_dataset.tsv' or 'res_regress_dataset.tsv') with an SD
    after every value, and return its path."""
    def _write(filename: str) -> str:
        lines = []
        with open(data_path(filename)) as handle:
            for index, line in enumerate(handle):
                fields = line.split()
                tokens = [value if value.lower() == 'nan' else f'{value},{example_sd(index):g}' for value in fields[2:]]
                lines.append(' '.join(fields[:2] + tokens))
        return write_lines('uncertain_' + filename, lines)
    return _write


def simulated_residuals(seed: int, extra_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Residuals whose SDs are log-uniform from 0.05 to 1, plus Gaussian error of SD extra_sd."""
    rng = np.random.default_rng(seed)
    sds = np.exp(rng.uniform(np.log(0.05), np.log(1.0), SIMULATED_VALUES))
    residuals = rng.normal(0, np.sqrt(sds ** 2 + extra_sd ** 2))
    return residuals, sds


# ..........................................................................................
# The task

@pytest.mark.parametrize('datatype, num_classes', [('sequence', 3), ('residues', 2), ('matrix', 1)])
def test_uncertainties_are_only_for_sequence_and_residue_regression(datatype, num_classes):
    with pytest.raises(ValueError, match='sequence and residue regression only'):
        Task(datatype, num_classes, uncertainties=True)


def test_task_with_uncertainties_is_saved_and_rebuilt():
    assert Task.from_config(SEQUENCE_TASK.to_config()) == SEQUENCE_TASK
    # Networks trained without uncertainties are saved exactly as before
    assert 'uncertainties' not in Task('sequence', 1).to_config()
    assert not Task.from_config({'datatype': 'sequence', 'num_classes': 1}).uncertainties


# ..........................................................................................
# Reading values with uncertainties

def test_values_are_read_with_their_sds():
    sequence_sample = pid.parse_lines([['a', 'ACD', '1.5,0.2']], 'sequence', uncertainties=True)[0]
    residue_sample = pid.parse_lines([['b', 'AC', '1,0', 'nan']], 'residues', uncertainties=True)[0]
    assert sequence_sample == ['a', 'ACD', 1.5, 0.2]
    np.testing.assert_array_equal(residue_sample[2], [1.0, np.nan])
    np.testing.assert_array_equal(residue_sample[3], [0.0, np.nan])


@pytest.mark.parametrize('token, message', [
    ('1.5', "every value is written as 'value,SD'"),
    ('1.5,0.1,2', "every value is written as 'value,SD'"),
    ('1.5,-0.1', 'finite number of at least 0'),
    ('1.5,inf', 'finite number of at least 0'),
    ('1.5,nan', 'finite number of at least 0'),
])
def test_bad_uncertainties_are_refused_with_the_line(token, message):
    with pytest.raises(IOExceptionParrot, match=f'Data line 1 .*{message}'):
        pid.parse_lines([['a', 'ACD', token]], 'sequence', uncertainties=True)


def test_values_with_sds_but_no_flag_suggest_the_flag():
    with pytest.raises(IOExceptionParrot, match="'1.5,0.2' are not numbers; .* give --uncertainties"):
        pid.parse_lines([['a', 'ACD', '1.5,0.2']], 'sequence')


@pytest.mark.parametrize('task, data, expected_shape', [
    (SEQUENCE_TASK, [['a', 'ACD', 1.5, 0.2], ['b', 'AC', -1.0, 0.0]], (2, 2)),
    (RESIDUE_TASK, [['a', 'ACD', np.array([1.0, np.nan, 2.0]), np.array([0.1, np.nan, 0.2])],
                    ['b', 'AC', np.array([3.0, 4.0]), np.array([0.3, 0.4])]], (2, 3, 2)),
])
def test_batches_pair_each_value_with_its_sd(task, data, expected_shape):
    dataset = pid.SequenceDataset(data)
    batch = next(iter(pid.make_dataloader(dataset, task, batch_size=2, batching='random')))
    order = [batch.names.index(name) for name in ('a', 'b')]
    targets = batch.targets[order]
    assert targets.shape[0] == expected_shape[0] and targets.shape[-1] == 2
    if task.per_residue:
        torch.testing.assert_close(targets[0, :3], torch.tensor([[1.0, 0.1], [math.nan, math.nan], [2.0, 0.2]]),
                                   equal_nan=True)
        # Padding holds zeros, and is masked out of the loss by length
        assert torch.all(targets[1, 2:] == 0)
    else:
        torch.testing.assert_close(targets, torch.tensor([[1.5, 0.2], [-1.0, 0.0]]))
    assert dataset.uncertainties is not None and pid.SequenceDataset(data[:0]).uncertainties is None


# ..........................................................................................
# The loss and the model error

def test_gaussian_nll_is_the_full_gaussian_negative_log_likelihood():
    predicted, values, sds = torch.tensor([0.0, 1.0, 2.5]), torch.tensor([0.5, 1.0, 1.0]), torch.tensor([0.1, 0.0, 2.0])
    model_error = torch.tensor(0.3)
    expected = torch.nn.GaussianNLLLoss(full=True, reduction='sum')(predicted, values, sds ** 2 + model_error ** 2)
    torch.testing.assert_close(gaussian_nll(predicted, values, sds, model_error), expected)


def test_model_error_recovers_the_error_beyond_the_sds():
    residuals, sds = simulated_residuals(seed=1, extra_sd=TRUE_MODEL_ERROR)
    assert fit_model_error(residuals, sds) == pytest.approx(TRUE_MODEL_ERROR, rel=MODEL_ERROR_TOLERANCE)


def test_model_error_maximises_the_likelihood():
    residuals, sds = simulated_residuals(seed=2, extra_sd=TRUE_MODEL_ERROR)
    fitted = fit_model_error(residuals, sds)

    def nll(model_error: float) -> float:
        return float(gaussian_nll(torch.zeros(len(residuals), dtype=torch.float64), torch.from_numpy(residuals),
                                  torch.from_numpy(sds), torch.tensor(model_error, dtype=torch.float64)))
    assert nll(fitted) < nll(fitted * 0.99) and nll(fitted) < nll(fitted * 1.01)


def test_model_error_is_zero_when_the_sds_explain_the_residuals():
    residuals, sds = simulated_residuals(seed=3, extra_sd=0.0)
    # SDs twice as large as the errors really were
    assert fit_model_error(residuals, 2 * sds) == 0.0


def test_model_error_of_exact_values_is_their_root_mean_squared_residual():
    residuals = np.array([0.1, -0.3, 0.2, 0.4])
    assert fit_model_error(residuals, np.zeros(4)) == pytest.approx(np.sqrt(np.mean(residuals ** 2)), rel=1e-9)


def test_model_error_stays_above_zero_for_exact_values_predicted_exactly():
    # One exact value is predicted exactly, so the likelihood would grow without limit as s fell to 0
    model_error = fit_model_error(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert 0 < model_error < 1e-5
    assert fit_model_error(np.zeros(3), np.zeros(3)) == 0.0


@pytest.mark.parametrize('residuals, sds, message', [
    ([], [], 'at least one residual'),
    ([0.1, 0.2], [0.1], '2 residuals but 1 SDs'),
    ([0.1], [-0.1], 'not negative'),
    ([np.nan], [0.1], 'must be finite'),
])
def test_model_error_refuses_bad_input(residuals, sds, message):
    with pytest.raises(ValueError, match=message):
        fit_model_error(np.array(residuals), np.array(sds))


def test_measurements_are_combined_by_inverse_variance_weighting():
    value, sd = combine_measurements(np.array([1.0, 3.0, 2.0]), np.array([1.0, 1.0, 0.5]))
    # Weights 1, 1 and 4
    assert value == pytest.approx((1 + 3 + 4 * 2) / 6) and sd == pytest.approx(1 / math.sqrt(6))
    assert combine_measurements(np.array([1.0, 5.0]), np.array([0.0, 1.0])) == (1.0, 0.0)


# ..........................................................................................
# Networks and training

def test_only_networks_with_uncertainties_have_a_model_error_and_weights_are_unchanged():
    torch.manual_seed(0)
    with_uncertainties = build_network('cnn', 20, SEQUENCE_TASK)
    torch.manual_seed(0)
    without = build_network('cnn', 20, Task('sequence', 1))

    assert math.isnan(float(model_error_of(with_uncertainties)))
    with pytest.raises(ValueError, match='no model error'):
        model_error_of(without)
    # Creating the model error draws no random numbers, so a seed gives the same weights either way
    weights = {name: value for name, value in with_uncertainties.state_dict().items() if name != 'model_error'}
    for name, value in without.state_dict().items():
        torch.testing.assert_close(weights[name], value)


def test_knn_cannot_weight_values():
    with pytest.raises(ValueError, match="'knn' architecture cannot be trained on values with uncertainties"):
        build_network('knn', 20, SEQUENCE_TASK)


def loaders_with_uncertainties(task: Task, filename: str) -> tuple[list, object, object]:
    """Training and validation loaders of an example dataset, with SDs added."""
    data = pid.parse_file(data_path(filename), Task(task.datatype, 1), ignore_warnings=True)
    for index, sample in enumerate(data):
        sample.append(np.where(np.isnan(sample[2]), np.nan, example_sd(index)) if task.per_residue
                      else example_sd(index))
    split = int(0.75 * len(data))
    train_loader = pid.make_dataloader(pid.SequenceDataset(data, np.arange(split)), task, 16, shuffle=True)
    val_loader = pid.make_dataloader(pid.SequenceDataset(data, np.arange(split, len(data))), task, 16)
    return data, train_loader, val_loader


@pytest.mark.parametrize('task, filename', [(SEQUENCE_TASK, 'seq_regress_dataset.tsv'),
                                            (RESIDUE_TASK, 'res_regress_dataset.tsv')])
def test_training_fits_the_model_error_and_saving_keeps_it(task, filename, tmp_path):
    _, train_loader, val_loader = loaders_with_uncertainties(task, filename)
    network = build_network('brnn', 20, task, {'hidden_size': 8, 'num_layers': 1})
    train_losses, val_losses = train_network.train(network, train_loader, val_loader, task, 'cpu', 0.01, 3,
                                                   silent=True)
    model_error = float(model_error_of(network))
    assert model_error > 0 and all(math.isfinite(loss) for loss in train_losses + val_losses)

    # The model error kept is the one fitted to the kept network on the validation set
    kept = float(model_error_of(network))
    train_network.refit_model_error(network, val_loader, task, 'cpu')
    assert float(model_error_of(network)) == pytest.approx(kept, rel=1e-5)

    save_network(str(tmp_path / 'net.pt'), network, get_encoder('onehot'))
    loaded, _ = load_network(str(tmp_path / 'net.pt'))
    assert loaded.task == task and float(model_error_of(loaded)) == model_error


def test_stop_auto_counts_a_fall_in_the_loss_rather_than_a_fraction():
    # A negative log-likelihood can be negative, where a relative fall would point the wrong way
    assert train_network.meaningfully_better(-1.006, -1.0, minimum_fall=0.005)
    assert not train_network.meaningfully_better(-1.004, -1.0, minimum_fall=0.005)
    assert train_network.meaningfully_better(-1.0, math.inf, minimum_fall=0.005)


def test_simple_models_are_weighted_by_their_sds_and_refitted():
    data, train_loader, val_loader = loaders_with_uncertainties(SEQUENCE_TASK, 'seq_regress_dataset.tsv')
    network = build_network('linear', 20, SEQUENCE_TASK)

    # Before the model error is known, every sequence counts equally
    assert network.sample_weights(np.array([0.1, 0.2])) is None
    set_model_error(network, 0.1)
    np.testing.assert_allclose(network.sample_weights(np.array([0.1, 0.0])), np.array([1 / 0.02, 1 / 0.01]) / 75)

    train_network.fit_in_one_step(network, train_loader, val_loader, SEQUENCE_TASK, 'cpu', silent=True)
    weighted = network.estimator.predict(np.eye(21))
    # The linear model fits this example almost exactly, so the SDs account for all of its error
    assert float(model_error_of(network)) == 0.0

    unweighted = build_network('linear', 20, Task('sequence', 1))
    plain_data = [sample[:3] for sample in data]
    unweighted.fit(pid.make_dataloader(pid.SequenceDataset(plain_data, np.arange(int(0.75 * len(data)))),
                                       Task('sequence', 1), 16))
    assert not np.allclose(weighted, unweighted.estimator.predict(np.eye(21)))


# ..........................................................................................
# Metrics

def test_uncertainty_metrics():
    metrics = uncertainty_metrics([np.zeros(4)], [np.ones(4)], [np.array([0.5, 1.5, -2.5, 0.0])], 0.0)
    # z-scores -0.5, -1.5, 2.5 and 0
    assert metrics['Reduced chi-squared'] == pytest.approx((0.25 + 2.25 + 6.25) / 4, abs=1e-3)
    assert (metrics['Within 1 SD'], metrics['Within 2 SD']) == (0.5, 0.75)
    # Values that are all the same leave nothing for a model to explain
    assert math.isnan(metrics['Noise ceiling R'])

    # Residue values: the missing one is left out, and the variance of 1, 2, 3, 4 is 1.25
    metrics = uncertainty_metrics([np.array([1.0, 2.0]), np.array([3.0, np.nan, 4.0])],
                                  [np.array([0.5, 0.5]), np.array([0.5, np.nan, 0.5])],
                                  [np.array([1.0, 2.0]), np.array([3.0, 0.0, 4.0])], 0.0)
    assert metrics['Noise ceiling R'] == pytest.approx(math.sqrt(1 - 0.25 / 1.25), abs=1e-3)
    assert metrics['Reduced chi-squared'] == 0.0


def test_uncertainty_metrics_of_non_finite_predictions_are_nan():
    metrics = uncertainty_metrics([np.zeros(2)], [np.ones(2)], [np.array([np.nan, 0.0])], 0.1)
    assert all(math.isnan(metrics[name]) for name in ('Reduced chi-squared', 'Within 1 SD', 'Within 2 SD'))


# ..........................................................................................
# The commands

@pytest.mark.parametrize('architecture, filename, datatype', [
    ('brnn', 'seq_regress_dataset.tsv', 'sequence'),
    ('cnn', 'res_regress_dataset.tsv', 'residues'),
    ('linear', 'seq_regress_dataset.tsv', None),
])
def test_train_with_uncertainties(run_cli, tmp_path, uncertain_dataset, architecture, filename, datatype, capsys):
    network_file = tmp_path / 'net.pt'
    epochs = ['-e', '2'] if architecture != 'linear' else []
    run_cli(parrot_train.main, training_command('parrot-train', architecture, uncertain_dataset(filename),
                                                network_file, datatype, 1, '--uncertainties', *epochs))

    assert 'Model error s = ' in capsys.readouterr().out
    stats = (tmp_path / 'net_performance_stats.txt').read_text()
    for name in ('Reduced chi-squared', 'Within 1 SD', 'Within 2 SD', 'Noise ceiling R', 'Model error s'):
        assert f'{name} : ' in stats
    info = (tmp_path / 'net_info.txt').read_text()
    assert 'regression with uncertainties' in info and 'model error' in info and 'SD of each' in info
    network, _ = load_network(str(network_file))
    # (The linear model fits the sequence example almost exactly, so its model error can be 0)
    assert network.task.uncertainties and math.isfinite(float(model_error_of(network)))


@pytest.mark.parametrize('architecture, flags', [('brnn', ['-e', '3', '-d', 'sequence']), ('svm', [])])
def test_optimize_with_uncertainties(run_cli, tmp_path, uncertain_dataset, architecture, flags):
    run_cli(parrot_optimize.main, training_command('parrot-optimize', architecture,
                                                   uncertain_dataset('seq_regress_dataset.tsv'), tmp_path / 'opt',
                                                   None, 1, '--uncertainties', '--trials', '2', '--silent', *flags))
    network, _ = load_network(str(tmp_path / 'opt.pt'))
    assert network.task.uncertainties and math.isfinite(float(model_error_of(network)))


def test_uncertainties_need_regression(run_cli, tmp_path, uncertain_dataset):
    with pytest.raises(ValueError, match='--uncertainties applies to regression only'):
        run_cli(parrot_train.main, training_command('parrot-train', 'brnn',
                                                    uncertain_dataset('seq_regress_dataset.tsv'), tmp_path / 'net',
                                                    'sequence', 3, '--uncertainties'))


def test_knn_does_not_offer_uncertainties(run_cli, tmp_path, uncertain_dataset, capsys):
    with pytest.raises(SystemExit):
        run_cli(parrot_train.main, training_command('parrot-train', 'knn', uncertain_dataset('seq_regress_dataset.tsv'),
                                                    tmp_path / 'net', None, 1, '--uncertainties'))
    assert 'unrecognized arguments: --uncertainties' in capsys.readouterr().err


def test_preprocess_merges_copies_by_their_sds(run_cli, tmp_path, write_lines):
    input_file = write_lines('data.tsv', ['a ACDE 1.0,1.0', 'b ACDE 3.0,0.5', 'c GGG 1.5,0.2'])
    output_file = tmp_path / 'out.tsv'
    run_cli(parrot_preprocess.main, ['parrot-preprocess', input_file, output_file, '-d', 'sequence', '-m',
                                     'regression', '--uncertainties', '--deduplicate', 'merge'])
    merged, kept = [line.split() for line in output_file.read_text().splitlines()]
    # Weights 1 and 4
    value, sd = (float(number) for number in merged[2].split(','))
    assert merged[:2] == ['a', 'ACDE'] and value == pytest.approx((1 + 4 * 3) / 5) and sd == pytest.approx(1 / 5 ** 0.5)
    assert kept == ['c', 'GGG', '1.5,0.2']
    assert 'regression data with uncertainties' in (tmp_path / 'out_preprocess_report.txt').read_text()


@pytest.mark.parametrize('datatype, lines', [
    ('sequence', ['a ACDE 1.0,1.0', 'b ACDE 3.0,-0.5']),
    ('sequence', ['a ACDE 1.0,1.0', 'b ACDE 3.0,nan']),
    ('residues', ['a AC 1.0,1.0 nan', 'b AC 3.0,0.5 2.0,-0.1']),
])
def test_preprocess_refuses_to_merge_copies_with_invalid_sds(run_cli, tmp_path, write_lines, datatype, lines):
    # A negative SD used to be merged as if it were positive, and a missing one made the merged value missing
    with pytest.raises(ValueError, match='Cannot merge the copies of a: the SD of every value must be a finite'):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', write_lines('data.tsv', lines), tmp_path / 'out.tsv',
                                         '-d', datatype, '-m', 'regression', '--uncertainties', '--deduplicate',
                                         'merge'])


def test_preprocess_merges_residue_values_by_their_sds(run_cli, tmp_path, write_lines):
    input_file = write_lines('residues.tsv', ['a AC 1.0,1.0 nan', 'b AC 3.0,0.5 2.0,0.1', 'c AC 5.0,0 nan'])
    output_file = tmp_path / 'out.tsv'
    run_cli(parrot_preprocess.main, ['parrot-preprocess', input_file, output_file, '-d', 'residues', '-m',
                                     'regression', '--uncertainties', '--deduplicate', 'merge'])
    fields = output_file.read_text().split()
    # The first residue has an exact copy (SD 0); the second has one labelled copy
    assert fields == ['a', 'AC', '5,0', '2,0.1']


@pytest.mark.parametrize('flags, message', [
    (['-m', 'classification'], 'sequence and residue regression only'),
    (['-m', 'regression', '--delimiter', ','], 'cannot also be separated'),
])
def test_preprocess_checks_uncertainty_options(run_cli, tmp_path, write_lines, flags, message):
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', write_lines('data.tsv', ['a ACDE 1.0,1.0']),
                                         tmp_path / 'out.tsv', '-d', 'sequence', '--uncertainties', *flags])


def test_preprocess_suggests_the_flag_for_values_with_sds(run_cli, tmp_path, write_lines):
    with pytest.raises(ValueError, match='give --uncertainties'):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', write_lines('data.tsv', ['a ACDE 1.0,1.0', 'b ACDE 2,1']),
                                         tmp_path / 'out.tsv', '-d', 'sequence', '-m', 'regression', '--deduplicate',
                                         'merge'])
