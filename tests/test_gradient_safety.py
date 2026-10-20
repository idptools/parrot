"""Opt-in clipping and finite checks must protect updates without changing ordinary training."""

import copy
import math

import numpy as np
import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.scripts import parrot_optimize, parrot_train, training_parser
from parrot.tasks import Task
from tests.helpers import data_path, training_command

TASK = Task('residues', 1)


class TinyRegressor(torch.nn.Module):
    """A linear residue regressor with a known, nonzero gradient on one-hot alanines."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 1, bias=False)
        torch.nn.init.zeros_(self.fc.weight)

    def forward(self, vectors, lengths):
        return self.fc(vectors)


def make_loader(value=1.0, length=4):
    dataset = pid.SequenceDataset([['protein_1', 'A' * length, np.full(length, value)]])
    return pid.make_dataloader(dataset, TASK, 1)


@pytest.mark.parametrize('options', [{'check_finite_gradients': True}, {'gradient_clip_norm': 1.0}])
@pytest.mark.parametrize('failure', ['nan_loss', 'inf_loss', 'nan_gradient', 'inf_gradient'])
def test_nonfinite_batch_never_reaches_optimizer(options, failure):
    network = TinyRegressor()
    if failure.endswith('loss'):
        bad = math.nan if failure.startswith('nan') else math.inf
        network.fc.register_forward_hook(lambda module, inputs, output: output + bad)
    if failure.endswith('gradient'):
        bad = math.nan if failure.startswith('nan') else math.inf
        network.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, bad))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    before = copy.deepcopy(network.state_dict())

    with pytest.raises(tn.NonFiniteTrainingError, match='gradient norm' if failure.endswith('gradient') else 'loss') \
            as raised:
        tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu', **options)

    assert 'batch 1' in str(raised.value) and 'protein_1' in str(raised.value)
    assert 'lengths 4-4' in str(raised.value)
    assert not optimizer.state
    assert all(torch.equal(before[name], value) for name, value in network.state_dict().items())


def test_finite_forward_can_have_nonfinite_recurrent_gradients():
    """An actual BRNN can overflow in backward while inputs, weights and the scalar loss remain finite."""
    network = build_network('brnn', 20, TASK, {'hidden_size': 1, 'num_layers': 1})
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        for suffix in ('', '_reverse'):
            getattr(network.lstm, 'weight_hh_l0' + suffix)[2, 0] = 2.0
            getattr(network.lstm, 'bias_ih_l0' + suffix)[[0, 1, 3]] = 5.0
        network.fc.weight.fill_(1.0)
    loader = make_loader(0.5, length=256)
    loss, outputs = tn.compute_batch_loss(network, next(iter(loader)), TASK, tn.build_loss_criterion(TASK), 'cpu')
    assert loss.item() == 64.0 and torch.isfinite(outputs).all()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01427)

    with pytest.raises(tn.NonFiniteTrainingError, match='gradient norm'):
        tn.train_epoch(network, loader, TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu',
                       check_finite_gradients=True)
    assert all(torch.isfinite(parameter).all() for parameter in network.parameters())
    assert not optimizer.state


def test_clipping_bounds_global_norm_and_parameter_update():
    network = TinyRegressor()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.5)
    statistics = tn.GradientStatistics()
    tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu',
                   gradient_clip_norm=0.1, gradient_statistics=statistics)

    assert statistics.norms == pytest.approx([8.0])
    assert network.fc.weight.grad.norm().item() == pytest.approx(0.1)
    assert network.fc.weight.norm().item() == pytest.approx(0.05)
    assert 'clipped 1/1 batches' in statistics.describe(0.1)


def test_checks_alone_do_not_change_finite_updates():
    results = []
    for options in ({}, {'check_finite_gradients': True}):
        network = TinyRegressor()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        loss, _ = tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu',
                                 **options)
        results.append((loss, network.fc.weight.detach().clone(), copy.deepcopy(optimizer.state_dict())))
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])
    for key, value in results[0][2]['state'][0].items():
        assert torch.equal(value, results[1][2]['state'][0][key])


def test_default_path_does_not_compute_gradient_norms(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Inactive gradient checks must not inspect or clip gradients')
    monkeypatch.setattr(torch.nn.utils, 'get_total_norm', unexpected)
    monkeypatch.setattr(torch.nn.utils, 'clip_grads_with_norm_', unexpected)
    network = TinyRegressor()
    tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK),
                   torch.optim.Adam(network.parameters()), 'cpu')


@pytest.mark.parametrize('threshold', [0.0, -1.0, math.nan, math.inf, -math.inf, True])
def test_invalid_clip_norm_is_rejected_before_updates(threshold):
    network = TinyRegressor()
    optimizer = torch.optim.Adam(network.parameters())
    with pytest.raises(ValueError, match='finite number greater than 0'):
        tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu',
                       gradient_clip_norm=threshold)
    assert not optimizer.state


def test_training_stops_at_first_bad_epoch_and_preserves_prior_save():
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    backwards = 0

    def fail_second_epoch(gradient):
        nonlocal backwards
        backwards += 1
        return gradient if backwards == 1 else torch.full_like(gradient, math.nan)

    network.fc.weight.register_hook(fail_second_epoch)
    saved = []
    with pytest.raises(tn.NonFiniteTrainingError, match='Epoch 2:.*protein_1'):
        tn.train(network, make_loader(), make_loader(), TASK, 'cpu', 0.01, 10, silent=True,
                 check_finite_gradients=True, on_improvement=lambda net: saved.append(copy.deepcopy(net.state_dict())))
    assert backwards == 2 and len(saved) == 1
    assert all(torch.isfinite(value).all() for value in saved[0].values())
    assert all(torch.equal(saved[0][name], value) for name, value in network.state_dict().items())


def test_opted_in_checks_reject_nonfinite_validation(monkeypatch):
    monkeypatch.setattr(tn, 'validation_loss', lambda *args: math.nan)
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    with pytest.raises(tn.NonFiniteTrainingError, match='Epoch 1: nonfinite validation loss'):
        tn.train(network, make_loader(), make_loader(), TASK, 'cpu', 0.01, 10, silent=True,
                 check_finite_gradients=True)


def test_opted_in_training_reports_norms_and_respects_silent(capsys):
    for silent in (False, True):
        network = build_network('brnn', 20, TASK, {'hidden_size': 2})
        tn.train(network, make_loader(), make_loader(), TASK, 'cpu', 0.01, 1, silent=silent,
                 gradient_clip_norm=0.1)
        output = capsys.readouterr().out
        if silent:
            assert output == ''
        else:
            assert 'Gradient norms before clipping: median' in output and 'clipped 1/1 batches' in output


@pytest.mark.parametrize('command', [parrot_train, parrot_optimize])
def test_gradient_options_are_opt_in_and_neural_only(command):
    base = ['--input-file', 'data.tsv', '--output-network', 'net', '--split', 'random', '--regression']
    parser = command.build_parser()
    args = parser.parse_args(['brnn', *base, '-d', 'residues'])
    assert training_parser.gradient_options_from_args(args) == {}
    args = parser.parse_args(['brnn', *base, '-d', 'residues', '--gradient-clip-norm', '5',
                              '--check-finite-gradients'])
    assert training_parser.gradient_options_from_args(args) == {'gradient_clip_norm': 5.0,
                                                               'check_finite_gradients': True}
    for option in (['--gradient-clip-norm', '5'], ['--check-finite-gradients']):
        with pytest.raises(SystemExit):
            parser.parse_args(['knn', *base, *option])


@pytest.mark.parametrize('command', [parrot_train, parrot_optimize])
def test_cli_rejects_invalid_clip_before_loading_data(run_cli, command):
    with pytest.raises(ValueError, match='Gradient clip norm'):
        run_cli(command.main, training_command(command.PROGRAM_NAME, 'brnn', 'missing.tsv', 'net', 'residues', 1,
                                               '--gradient-clip-norm', 'nan'))


@pytest.mark.parametrize('command, extra', [(parrot_train, []), (parrot_optimize, ['--trials', '1']),
                                          (parrot_optimize, ['--trials', '1', '--cv-folds', '2'])])
def test_cli_applies_clipping_and_records_active_settings(run_cli, tmp_path, monkeypatch, command, extra):
    original = torch.nn.utils.clip_grads_with_norm_
    observed = []

    def record(parameters, max_norm, total_norm, **kwargs):
        parameters = list(parameters)
        original(parameters, max_norm, total_norm, **kwargs)
        observed.append(torch.nn.utils.get_total_norm([p.grad for p in parameters]).item())

    monkeypatch.setattr(torch.nn.utils, 'clip_grads_with_norm_', record)
    output = tmp_path / 'guarded.pt'
    run_cli(command.main, training_command(command.PROGRAM_NAME, 'brnn', data_path('res_regress_dataset.tsv'),
                                           output, 'residues', 1, '-e', '1', '-hs', '2', '-nl', '1', '--silent',
                                           '--gradient-clip-norm', '0.1', *extra))
    assert observed and max(observed) <= 0.10001
    info = (tmp_path / 'guarded_info.txt').read_text()
    assert 'global L2 norm at most 0.1' in info and 'finite checks' in info
    assert output.is_file()


def test_search_marks_nonfinite_trial_failed_and_continues(monkeypatch):
    original = tn.train_epoch
    failures = 0

    def fail_first_trial(*args, **kwargs):
        nonlocal failures
        assert kwargs['check_finite_gradients'] is True
        if failures == 0:
            failures += 1
            raise tn.NonFiniteTrainingError('Nonfinite gradient norm in training batch 1')
        return original(*args, **kwargs)

    monkeypatch.setattr(tn, 'train_epoch', fail_first_trial)
    space = hs.SearchSpace('brnn', fixed={'hidden_size': 2, 'num_layers': 1})
    trainer = hs.TrialTrainer(space, [(make_loader(), make_loader())], 20, TASK, 'cpu',
                              check_finite_gradients=True)
    search = hs.HyperparameterSearch(space, trainer, 2, 1, rng=np.random.default_rng(0), silent=True)
    best = search.run()
    assert failures == 1 and best.status == hs.COMPLETED
    failed = [trial for trial in search.trials if trial.status == hs.FAILED]
    assert len(failed) == 1 and 'Nonfinite gradient norm' in failed[0].failure_reason


def test_unprotected_cli_does_not_record_gradient_settings(run_cli, tmp_path):
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path('res_regress_dataset.tsv'),
                                               tmp_path / 'ordinary.pt', 'residues', 1, '-e', '1', '-hs', '2', '--silent'))
    info = (tmp_path / 'ordinary_info.txt').read_text()
    assert 'gradient clipping' not in info and 'finite checks' not in info


@pytest.mark.real_devices
@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_clipping_and_nonfinite_checks_on_accelerators(device):
    available = torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f'{device} unavailable')
    network = TinyRegressor().to(device)
    optimizer = torch.optim.Adam(network.parameters())
    tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, device,
                   gradient_clip_norm=0.1)
    assert network.fc.weight.grad.norm().item() <= 0.10001
    before = copy.deepcopy(network.state_dict())
    state_before = copy.deepcopy(optimizer.state_dict())
    network.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
    with pytest.raises(tn.NonFiniteTrainingError, match='gradient norm'):
        tn.train_epoch(network, make_loader(), TASK, tn.build_loss_criterion(TASK), optimizer, device,
                       check_finite_gradients=True)
    assert all(torch.equal(before[name], value) for name, value in network.state_dict().items())
    for key, value in state_before['state'][0].items():
        assert torch.equal(value, optimizer.state_dict()['state'][0][key])
