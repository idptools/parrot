"""Scientific loss weighting, LR schedules and bounded recovery through real training paths."""

import copy
import csv
import json
import math
import random

import numpy as np
import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import prediction
from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.checkpoint import load_network
from parrot.encode_sequence import ONE_HOT
from parrot.scripts import parrot_optimize, parrot_train, training_parser
from parrot.tasks import Task
from parrot.training_control import EpochSnapshot, LearningRateController, TrainingControlState, TrainingOptions
from parrot.uncertainties import model_error_of
from tests.helpers import data_path, training_command

TASK = Task('residues', 1)


class ResidueLinear(torch.nn.Module):
    """Zero-initialized linear outputs allow exact loss and gradient calculations."""

    def __init__(self, classes=1, width=None):
        super().__init__()
        self.fc = torch.nn.Linear(20, classes, bias=False)
        torch.nn.init.zeros_(self.fc.weight)
        self.width = width

    def forward(self, vectors, lengths):
        outputs = self.fc(vectors)
        return outputs if self.width is None else outputs.unsqueeze(2).expand(-1, -1, self.width, -1)


def dataset():
    return pid.SequenceDataset([['short', 'A', np.array([1.0])], ['long', 'CCCC', np.array([3.0] * 4)]])


def loaders(batch=2, shuffle=False):
    data = dataset()
    return (pid.make_dataloader(data, TASK, batch, shuffle=shuffle), pid.make_dataloader(data, TASK, batch))


def test_equal_weights_are_mean_errors_not_equal_gradient_norms():
    network = ResidueLinear()
    batch = next(iter(loaders()[0]))
    criterion = tn.build_loss_criterion(TASK)
    ordinary, _ = tn.compute_batch_loss(network, batch, TASK, criterion, 'cpu')
    weighted, _ = tn.compute_batch_loss(network, batch, TASK, criterion, 'cpu', equal_protein_weighting=True)
    assert ordinary.item() == 37.0
    assert weighted.item() == 10.0  # 1 + 9, before averaging the two proteins
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    reported, _ = tn.train_epoch(network, [batch], TASK, criterion, optimizer, 'cpu', equal_protein_weighting=True)
    assert reported == 5.0
    assert network.fc.weight.grad[0, ONE_HOT['A']].item() == -1.0
    assert network.fc.weight.grad[0, ONE_HOT['C']].item() == -3.0  # Different errors still give different gradients


def test_equal_weight_loss_excludes_missing_labels_and_padding():
    samples = pid.SequenceDataset([['short', 'A', [1.0]], ['long', 'CCCC', [3.0, math.nan, 3.0, math.nan]]])
    batch = next(iter(pid.make_dataloader(samples, TASK, 2)))
    batch.targets[0, 1:] = 1e10
    network = ResidueLinear()
    loss, _ = tn.compute_batch_loss(network, batch, TASK, tn.build_loss_criterion(TASK), 'cpu',
                                    equal_protein_weighting=True)
    assert loss.item() == 10.0
    loss.backward()
    assert torch.isfinite(network.fc.weight.grad).all()


@pytest.mark.parametrize('batch_size', [1, 2])
def test_weighting_agrees_in_validation_and_test_loss(batch_size):
    network = ResidueLinear()
    loader = pid.make_dataloader(dataset(), TASK, batch_size)
    assert tn.evaluate_loss(network, loader, TASK, 'cpu', equal_protein_weighting=True) == 5.0
    result = prediction.predict_labeled(network, dataset(), TASK, 'cpu', batch_size,
                                        equal_protein_weighting=True)
    assert result.loss == 5.0


def test_equal_weighting_supports_classification():
    task = Task('residues', 3)
    data = pid.SequenceDataset([['short', 'A', [0]], ['long', 'CCCC', [1, 2, -100, 1]]])
    batch = next(iter(pid.make_dataloader(data, task, 2)))
    network = ResidueLinear(classes=3)
    loss, _ = tn.compute_batch_loss(network, batch, task, tn.build_loss_criterion(task), 'cpu',
                                    equal_protein_weighting=True)
    assert loss.item() == pytest.approx(2 * math.log(3))
    loss.backward()
    assert torch.isfinite(network.fc.weight.grad).all()


def test_equal_weighting_supports_matrix_cells():
    task = Task('matrix', 1, width=2)
    data = pid.SequenceDataset([['short', 'A', np.ones((1, 2))],
                                ['long', 'CCCC', np.array([[3.0, math.nan]] * 4)]])
    batch = next(iter(pid.make_dataloader(data, task, 2)))
    network = ResidueLinear(width=2)
    loss, _ = tn.compute_batch_loss(network, batch, task, tn.build_loss_criterion(task), 'cpu',
                                    equal_protein_weighting=True)
    assert loss.item() == 10.0


def test_uncertainty_model_error_uses_the_same_protein_weights():
    task = Task('residues', 1, uncertainties=True)
    data = pid.SequenceDataset([['short', 'A', np.array([0.2]), np.array([0.0])],
                                ['long', 'CCCC', np.array([2.0] * 4), np.array([0.0] * 4)]])
    loader = pid.make_dataloader(data, task, 2)
    network = build_network('brnn', 20, task, {'hidden_size': 1})
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    loss, effective_values = tn.refit_model_error(network, loader, task, 'cpu', equal_protein_weighting=True)
    assert float(model_error_of(network)) == pytest.approx(math.sqrt((0.2 ** 2 + 2.0 ** 2) / 2))
    assert effective_values == 1.0
    assert loss == pytest.approx(tn.evaluate_loss(network, loader, task, 'cpu', equal_protein_weighting=True))


def test_cosine_matches_pytorch_and_stays_at_floor():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    native_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    native = torch.optim.lr_scheduler.CosineAnnealingLR(native_optimizer, T_max=4, eta_min=0.001)
    controller = LearningRateController(optimizer, TrainingOptions(lr_schedule='cosine', lr_anneal_epochs=4,
                                                                  lr_min=0.001))
    for epoch in range(8):
        controller.prepare_epoch(epoch)
        assert controller.lrs[0] == pytest.approx(native_optimizer.param_groups[0]['lr'] if epoch <= 4 else 0.001)
        native_optimizer.step()
        native.step()


def test_exponential_decay_and_warmup():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    controller = LearningRateController(optimizer, TrainingOptions(lr_schedule='exponential', lr_gamma=0.5,
                                                                  lr_min=0.002, lr_warmup_epochs=2))
    rates = []
    for epoch in range(7):
        controller.prepare_epoch(epoch)
        rates.append(controller.lrs[0])
    assert rates == pytest.approx([0.005, 0.01, 0.01, 0.005, 0.0025, 0.002, 0.002])


def test_plateau_adapts_to_negative_validation_losses_with_cooldown():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    controller = LearningRateController(optimizer, TrainingOptions(lr_schedule='plateau', lr_factor=0.5,
                                                                  lr_patience=0, lr_cooldown=1, lr_threshold=0.01))
    rates = []
    for epoch, loss in enumerate([-1.0, -1.0, -1.0, -1.1, -1.1, -1.1]):
        controller.prepare_epoch(epoch)
        rates.append(controller.lrs[0])
        controller.finish_epoch(epoch, loss)
    assert rates == pytest.approx([0.01, 0.01, 0.005, 0.005, 0.005, 0.0025])


def test_snapshot_restores_random_generators_and_does_not_alias_adam_buffers():
    network = ResidueLinear()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    train_loader, val_loader = loaders(batch=1, shuffle=True)
    generator = torch.Generator().manual_seed(7)
    train_loader.generator = generator
    tn.train_epoch(network, train_loader, TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu')
    controller = LearningRateController(optimizer, TrainingOptions(recover_nonfinite=2))
    snapshot = EpochSnapshot.capture(network, optimizer, controller, (train_loader, val_loader), 'cpu')
    expected = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
    before = copy.deepcopy(optimizer.state_dict())
    for _ in range(2):
        tn.train_epoch(network, train_loader, TASK, tn.build_loss_criterion(TASK), optimizer, 'cpu')
        snapshot.restore(network, optimizer, controller)
        actual = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
        assert expected[:2] == actual[:2]
        assert torch.equal(expected[2], actual[2]) and torch.equal(expected[3], actual[3])
        for key, value in before['state'][0].items():
            assert torch.equal(value, optimizer.state_dict()['state'][0][key])


def test_replay_matches_clean_lower_lr_training_with_dropout_and_adam():
    results = []
    for inject_failure, initial_lr in [(True, 0.01), (False, 0.005)]:
        torch.manual_seed(31)
        network = build_network('brnn', 20, TASK, {'hidden_size': 2, 'num_layers': 2, 'dropout': 0.2})
        if inject_failure:
            backwards = 0

            def fail_second_batch(gradient):
                nonlocal backwards
                backwards += 1
                return torch.full_like(gradient, math.nan) if backwards == 2 else gradient

            network.fc.weight.register_hook(fail_second_batch)
        state = TrainingControlState()
        train_loader, val_loader = loaders(batch=1, shuffle=True)
        losses = tn.train(network, train_loader, val_loader, TASK, 'cpu', initial_lr, 3, silent=True,
                          training_options=TrainingOptions(recover_nonfinite=2, lr_schedule='exponential', lr_gamma=0.8),
                          control_state=state)
        results.append((copy.deepcopy(network.state_dict()), losses, state))
    assert results[0][1] == results[1][1]
    assert all(torch.equal(value, results[1][0][name]) for name, value in results[0][0].items())
    assert results[0][2].recoveries_used == 1
    assert [row['learning_rate'] for row in results[0][2].epoch_history] == pytest.approx([0.005, 0.004, 0.0032])


def test_recovery_budget_is_global_and_reductions_are_cumulative():
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    backwards = 0

    def fail_odd_attempts(gradient):
        nonlocal backwards
        backwards += 1
        return torch.full_like(gradient, math.nan) if backwards % 2 else gradient

    network.fc.weight.register_hook(fail_odd_attempts)
    state = TrainingControlState()
    train_loader, val_loader = loaders()
    saved = []
    with pytest.raises(tn.NonFiniteTrainingError, match='Recovery budget exhausted'):
        tn.train(network, train_loader, val_loader, TASK, 'cpu', 0.01, 10, silent=True,
                  training_options=TrainingOptions(recover_nonfinite=2), control_state=state,
                  on_improvement=lambda model: saved.append(copy.deepcopy(model.state_dict())))
    assert backwards == 5 and state.recoveries_used == 2
    assert len(state.epoch_history) == 2
    assert [event['epoch'] for event in state.recovery_events] == [1, 2]
    assert [event['new_lr'] for event in state.recovery_events] == pytest.approx([0.005, 0.0025])
    assert saved and all(torch.isfinite(value).all() for value in saved[0].values())


def test_repeated_failure_of_same_epoch_stops_after_two_replays():
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    network.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
    state = TrainingControlState()
    with pytest.raises(tn.NonFiniteTrainingError, match='Recovery budget exhausted'):
        tn.train(network, *loaders(), TASK, 'cpu', 0.01, 3, silent=True,
                  training_options=TrainingOptions(recover_nonfinite=2), control_state=state)
    assert not state.epoch_history
    assert [event['new_lr'] for event in state.recovery_events] == pytest.approx([0.005, 0.0025])
    assert all(parameter.grad is None for parameter in network.parameters())


def test_bad_input_does_not_consume_retries():
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    bad = pid.SequenceDataset([['bad_target', 'A', [math.inf]]])
    loader = pid.make_dataloader(bad, TASK, 1)
    state = TrainingControlState()
    with pytest.raises(tn.NonFiniteTrainingError, match='input or target'):
        tn.train(network, loader, loader, TASK, 'cpu', 0.01, 3, silent=True,
                  training_options=TrainingOptions(recover_nonfinite=2), control_state=state)
    assert state.recoveries_used == 0 and not state.recovery_events


def test_corrupt_adam_moment_after_step_is_rolled_back(monkeypatch):
    original = torch.optim.Adam.step
    injected = False

    def corrupt_once(optimizer, *args, **kwargs):
        nonlocal injected
        result = original(optimizer, *args, **kwargs)
        if not injected:
            next(iter(optimizer.state.values()))['exp_avg_sq'].fill_(math.inf)
            injected = True
        return result

    monkeypatch.setattr(torch.optim.Adam, 'step', corrupt_once)
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    state = TrainingControlState()
    tn.train(network, *loaders(), TASK, 'cpu', 0.01, 2, silent=True,
              training_options=TrainingOptions(recover_nonfinite=2), control_state=state)
    assert injected and state.recoveries_used == 1
    assert all(torch.isfinite(parameter).all() for parameter in network.parameters())


@pytest.mark.parametrize('schedule', ['cosine', 'exponential', 'plateau'])
@pytest.mark.parametrize('warmup', [0, 3])
def test_lr_state_and_weights_resume_identically_across_search_stages(schedule, warmup):
    results = []
    options = TrainingOptions(lr_schedule=schedule, lr_anneal_epochs=5, lr_patience=0,
                              lr_warmup_epochs=warmup, recover_nonfinite=2)
    for stops in ([2, 5], [5]):
        torch.manual_seed(13)
        space = hs.SearchSpace('brnn', fixed={'hidden_size': 2, 'num_layers': 1})
        trainer = hs.TrialTrainer(space, [loaders(shuffle=True)], 20, TASK, 'cpu', training_options=options)
        trial = hs.Trial(1, space.default_values(), 'default')
        for epochs in stops:
            trainer.train_to(trial, epochs)
        network, training, validation = trainer.trained_network(trial)
        results.append((copy.deepcopy(network.state_dict()), training, validation, trial.folds[0].training_control))
    assert results[0][1:3] == results[1][1:3]
    assert all(torch.equal(value, results[1][0][name]) for name, value in results[0][0].items())
    assert results[0][3] == results[1][3]


def test_recovery_budget_is_preserved_across_search_promotions(monkeypatch):
    original = tn.train_epoch
    calls = 0

    def fail_odd_attempts(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2:
            raise tn.NonFiniteTrainingError('Injected numerical failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(tn, 'train_epoch', fail_odd_attempts)
    space = hs.SearchSpace('brnn', fixed={'hidden_size': 2, 'num_layers': 1})
    trainer = hs.TrialTrainer(space, [loaders()], 20, TASK, 'cpu',
                              training_options=TrainingOptions(recover_nonfinite=2))
    trial = hs.Trial(1, space.default_values(), 'default')
    trainer.train_to(trial, 1)
    trainer.train_to(trial, 2)
    with pytest.raises(tn.NonFiniteTrainingError, match='Recovery budget exhausted'):
        trainer.train_to(trial, 3)
    assert calls == 5 and trial.epochs == 2
    assert trial.folds[0].training_control.recoveries_used == 2


def test_nonfinite_validation_replays_the_epoch(monkeypatch):
    original = tn.validation_loss
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return math.nan if calls == 1 else original(*args, **kwargs)

    monkeypatch.setattr(tn, 'validation_loss', fail_once)
    state = TrainingControlState()
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    tn.train(network, *loaders(), TASK, 'cpu', 0.01, 2, silent=True,
              training_options=TrainingOptions(recover_nonfinite=2), control_state=state)
    assert calls == 3 and state.recoveries_used == 1 and len(state.epoch_history) == 2
    assert state.epoch_history[0]['learning_rate'] == 0.005


def test_recovery_stops_when_lr_cannot_be_lowered():
    network = build_network('brnn', 20, TASK, {'hidden_size': 2})
    network.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
    state = TrainingControlState()
    with pytest.raises(tn.NonFiniteTrainingError, match='cannot be reduced further'):
        tn.train(network, *loaders(), TASK, 'cpu', 0.01, 2, silent=True,
                  training_options=TrainingOptions(recover_nonfinite=2, lr_min=0.01), control_state=state)
    assert state.recoveries_used == 0
    assert all(parameter.grad is None for parameter in network.parameters())


@pytest.mark.parametrize('command, extra', [(parrot_train, []), (parrot_optimize, ['--trials', '1']),
                                          (parrot_optimize, ['--trials', '1', '--cv-folds', '2'])])
def test_cli_weighting_schedules_and_recovery_metadata(run_cli, tmp_path, command, extra):
    run_cli(command.main, training_command(command.PROGRAM_NAME, 'brnn', data_path('res_regress_dataset.tsv'),
                                           tmp_path / 'controlled.pt', 'residues', 1, '-e', '2', '-hs', '2', '-nl', '1',
                                           '--equal-protein-weighting', '--recover-nonfinite', '--lr-schedule',
                                           'cosine', '--lr-warmup-epochs', '1', '--silent', *extra))
    info = (tmp_path / 'controlled_info.txt').read_text()
    assert 'equal-protein weighting' in info and 'initial learning rate' in info
    assert 'numerical recovery' in info and 'recoveries used' in info
    assert '(constant)' not in info
    with (tmp_path / 'controlled_training_history.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert len(rows) == 2 and all(float(row['learning_rate']) > 0 for row in rows)
    assert not (tmp_path / 'controlled_recovery_events.json').exists()


def test_successful_cli_recovery_writes_its_audit_trail(run_cli, tmp_path, monkeypatch):
    original = torch.optim.Adam.step
    injected = False

    def corrupt_once(optimizer, *args, **kwargs):
        nonlocal injected
        result = original(optimizer, *args, **kwargs)
        if not injected:
            next(iter(optimizer.state.values()))['exp_avg_sq'].fill_(math.inf)
            injected = True
        return result

    monkeypatch.setattr(torch.optim.Adam, 'step', corrupt_once)
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path('res_regress_dataset.tsv'),
                                               tmp_path / 'recovered.pt', 'residues', 1, '-e', '1', '-hs', '2',
                                               '--recover-nonfinite', '--silent'))
    events = json.loads((tmp_path / 'recovered_recovery_events.json').read_text())
    assert len(events) == 1 and events[0]['old_lr'] == 0.001 and events[0]['new_lr'] == 0.0005
    assert events[0]['epoch'] == 1 and 'after the optimizer update' in events[0]['reason']


def test_cli_uncertainties_use_equal_protein_test_loss(run_cli, tmp_path):
    lines = [f"protein_{index} {'A' * (index + 1)} " + ' '.join([f'{index / 10},0.1'] * (index + 1))
             for index in range(15)]
    data_file = tmp_path / 'uncertain.tsv'
    data_file.write_text('\n'.join(lines) + '\n')
    output = tmp_path / 'uncertain.pt'
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_file, output, 'residues', 1,
                                               '-e', '2', '-hs', '2', '--uncertainties', '--equal-protein-weighting',
                                               '--lr-schedule', 'plateau', '--lr-patience', '0',
                                               '--recover-nonfinite', '--save-splits', '--silent'))
    network, encoder = load_network(str(output))
    _, _, test_indices = pid.read_split_file(str(tmp_path / 'uncertain_split_file.txt'), len(lines))
    task = network.task
    data = pid.SequenceDataset(pid.parse_file(str(data_file), task, ignore_warnings=True), test_indices, encoder)
    expected = prediction.predict_labeled(network, data, task, 'cpu', equal_protein_weighting=True).loss
    info = (tmp_path / 'uncertain_info.txt').read_text()
    assert math.isfinite(expected) and f'{expected:.4f}' in info and 'equal-protein weighting' in info


@pytest.mark.parametrize('command', [parrot_train, parrot_optimize])
def test_new_controls_are_neural_only(command):
    parser = command.build_parser()
    for flags in (['--equal-protein-weighting'], ['--recover-nonfinite'], ['--lr-schedule', 'plateau']):
        with pytest.raises(SystemExit):
            parser.parse_args(training_command(command.PROGRAM_NAME, 'knn', 'data.tsv', 'out', 'sequence', 1,
                                                *flags)[1:])


@pytest.mark.real_devices
@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_snapshot_restores_accelerator_rng(device):
    available = torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f'{device} unavailable')
    network = ResidueLinear().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    controller = LearningRateController(optimizer, TrainingOptions(recover_nonfinite=2))
    snapshot = EpochSnapshot.capture(network, optimizer, controller, loaders(), device)
    expected = torch.rand(8, device=device)
    torch.rand(20, device=device)
    snapshot.restore(network, optimizer, controller)
    assert torch.equal(torch.rand(8, device=device), expected)


@pytest.mark.parametrize('flags', [['--lr-gamma', '0.9'], ['--recovery-lr-factor', '0.5'],
                                  ['--lr-factor', '0.5'], ['--lr-min', '0.001'],
                                  ['--lr-schedule', 'cosine', '--stop', 'auto']])
def test_inapplicable_or_ambiguous_controls_are_rejected(flags):
    args = parrot_train.build_parser().parse_args(training_command('parrot-train', 'brnn', 'missing.tsv', 'out',
                                                                   'residues', 1, *flags)[1:])
    with pytest.raises(ValueError):
        training_parser.training_options_from_args(args)


@pytest.mark.parametrize('kwargs', [{'recover_nonfinite': -1}, {'lr_min': math.nan}, {'lr_gamma': 1.0},
                                   {'lr_patience': -1}, {'lr_warmup_epochs': -1}, {'recovery_lr_factor': 0.0},
                                   {'lr_schedule': 'unknown'}, {'lr_anneal_epochs': 0}])
def test_training_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingOptions(**kwargs)


def test_default_options_do_not_add_control_kwargs_or_artifacts(run_cli, tmp_path):
    args = parrot_train.build_parser().parse_args(training_command('parrot-train', 'brnn', 'data.tsv', 'out',
                                                                   'residues', 1)[1:])
    assert training_parser.training_options_from_args(args) == {}
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path('res_regress_dataset.tsv'),
                                               tmp_path / 'ordinary.pt', 'residues', 1, '-e', '1', '--silent'))
    assert not (tmp_path / 'ordinary_training_history.tsv').exists()
    assert not (tmp_path / 'ordinary_recovery_events.json').exists()
