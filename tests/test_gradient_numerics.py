"""Regression coverage for overflowing reductions, genuine failures and replay."""

import copy
import json
import math

import numpy as np
import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.gradient_numerics import clip_with_norm, stable_total_norm
from parrot.scripts import diagnose_brnn, parrot_optimize, parrot_train, training_parser
from parrot.tasks import Task
from parrot.training_precision import TrainingPrecision
from tests.test_gradient_safety import TinyRegressor, make_loader


def unstable_brnn(device='cpu'):
    task = Task('residues', 1)
    network = build_network('brnn', 20, task, {'hidden_size': 1, 'num_layers': 1}).to(device)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        for suffix in ('', '_reverse'):
            getattr(network.lstm, 'weight_hh_l0' + suffix)[2, 0] = 2.
            getattr(network.lstm, 'bias_ih_l0' + suffix)[[0, 1, 3]] = 5.
        network.fc.weight.fill_(1.)
    return network, task


@pytest.mark.parametrize('amp_boundary', [False, True])
def test_actual_brnn_overflowed_norm_can_be_clipped(amp_boundary):
    network, task = unstable_brnn()
    precision = None
    if amp_boundary:
        # Exercise AMP's real guard boundary without claiming CUDA validation.
        precision = TrainingPrecision('off', 'cpu')
        precision.dtype = torch.bfloat16
        precision.autocast = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
    optimizer = torch.optim.Adam(network.parameters())
    statistics = tn.GradientStatistics()
    tn.train_epoch(network, make_loader(0.5, 48), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                   gradient_clip_norm=1., gradient_statistics=statistics, precision=precision)
    assert 1e21 < statistics.norms[0] < 1e24
    assert statistics.raw_norms == [math.inf]
    assert 'stable norm fallback on 1 batch' in statistics.describe(1.)
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in network.parameters())
    assert float(stable_total_norm([p.grad for p in network.parameters()])) == pytest.approx(1., rel=1e-5)
    assert tn.finite_tensors(network.state_dict()) and tn.finite_tensors(optimizer.state_dict())


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
def test_clipping_norm_larger_than_float32_range(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    parameter = torch.nn.Parameter(torch.zeros(100, device=device))
    parameter.grad = torch.full_like(parameter, 1e38)
    norm = stable_total_norm([parameter.grad])
    assert float(norm) == pytest.approx(1e39, rel=1e-5)
    clip_with_norm([parameter], 1., norm)
    assert torch.allclose(parameter.grad, torch.full_like(parameter, 0.1), atol=1e-6)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('amp_boundary', [False, True])
def test_actual_nonfinite_entries_are_still_rejected_and_described(bad, amp_boundary):
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, bad))
    task = Task('residues', 1)
    optimizer = torch.optim.Adam(network.parameters())
    precision = None
    if amp_boundary:
        precision = TrainingPrecision('off', 'cpu')
        precision.dtype = torch.bfloat16
        precision.autocast = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
    with pytest.raises(tn.NonFiniteTrainingError) as raised:
        tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       gradient_clip_norm=1., precision=precision)
    details = raised.value.diagnostics
    assert details['nan' if math.isnan(bad) else 'inf'] == 20
    assert details['affected_parameters'][0]['parameter'] == 'fc.weight'
    assert ('precision=torch.bfloat16' if amp_boundary else 'precision=float32') in str(raised.value)
    assert 'lengths 4-4' in str(raised.value)
    assert not optimizer.state


def test_checks_alone_allow_finite_entries_with_overflowed_reduction():
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: gradient * 1e20)
    task = Task('residues', 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-22)
    tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                   check_finite_gradients=True)
    assert float(network.fc.weight[0, 0].detach()) == pytest.approx(0.08)


def test_norm_fallback_does_not_hide_overflowing_adam_moments():
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: gradient * 1e22)
    task = Task('residues', 1)
    optimizer = torch.optim.Adam(network.parameters())
    with pytest.raises(tn.NonFiniteTrainingError, match='model or optimizer state') as raised:
        tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       check_finite_gradients=True)
    assert 'after the optimizer update' in str(raised.value)
    assert raised.value.diagnostics['nan'] == raised.value.diagnostics['inf'] == 0
    assert not tn.finite_tensors(optimizer.state_dict())


def test_search_retains_full_failure_and_structured_diagnostics(tmp_path):
    space = hs.SearchSpace('brnn', fixed={'hidden_size': 2, 'num_layers': 1})
    trainer = hs.TrialTrainer(space, [], 20, Task('residues', 1), 'cpu')
    search = hs.HyperparameterSearch(space, trainer, 1, 1, silent=True)
    trial = hs.Trial(1, {'learning_rate': .001}, 'default')
    reason = 'Nonfinite gradient norm: ' + 'long_identifier ' * 50 + ' lengths 100-2049'
    details = {'nan': 12, 'affected_parameters': [{'parameter': 'lstm.weight_hh_l0'}]}
    search._record_failure(trial, tn.NonFiniteTrainingError(reason, diagnostics=details))
    search.trials = [trial]
    assert trial.failure_reason == reason and trial.failure_details == details
    state = search.state_dict()
    search.load_state_dict(state)
    assert search.trials[0].failure_details == details
    search.write_trials(str(tmp_path / 'trials.tsv'))
    assert reason in (tmp_path / 'trials.tsv').read_text()


@pytest.mark.parametrize('command', [parrot_train, parrot_optimize])
def test_capture_flag_neural_only_and_opt_in(command):
    base = ['--input-file', 'data.tsv', '--output-network', 'net', '--split', 'random', '--regression']
    parser = command.build_parser()
    args = parser.parse_args(['brnn', *base, '-d', 'residues'])
    assert training_parser.gradient_options_from_args(args) == {}
    args = parser.parse_args(['brnn', *base, '-d', 'residues', '--capture-nonfinite', 'failure.pt'])
    assert training_parser.gradient_options_from_args(args) == {'capture_nonfinite': 'failure.pt'}
    with pytest.raises(SystemExit):
        parser.parse_args(['knn', *base, '--capture-nonfinite', 'failure.pt'])


def test_failure_capture_is_before_batch_and_replayable(tmp_path):
    network, task = unstable_brnn()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    weights = copy.deepcopy(network.state_dict())
    path = tmp_path / 'failure.pt'
    with pytest.raises(tn.NonFiniteTrainingError, match='Failure capture:'):
        tn.train_epoch(network, make_loader(0.5, 96), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       capture_nonfinite=str(path))
    saved = torch.load(path, weights_only=True)
    assert saved['diagnostics']['nan'] + saved['diagnostics']['inf'] > 0
    assert all(torch.equal(saved['model_state'][name], value) for name, value in weights.items())
    assert not saved['optimizer_state']['state']
    assert saved['batch']['lengths'].tolist() == [96]
    original = path.read_bytes()
    # Recovery/search must not overwrite the first piece of evidence.
    with pytest.raises(tn.NonFiniteTrainingError):
        tn.train_epoch(network, make_loader(0.5, 256), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       capture_nonfinite=str(path))
    assert path.read_bytes() == original
    output = tmp_path / 'replay'
    diagnose_brnn.main(['--replay', str(path), '--device', 'cpu', '--modes', 'off', '--output-dir', str(output)])
    result = json.loads((output / 'summary.json').read_text())
    assert [row['implementation'] for row in result['replay']] == ['unpacked', 'packed']
    assert all(row['status'] == 'failed' and row['diagnostics']['nan'] + row['diagnostics']['inf'] > 0
               for row in result['replay'])


def test_capture_success_writes_nothing(tmp_path):
    network, task = unstable_brnn()
    path = tmp_path / 'failure.pt'
    tn.train_epoch(network, make_loader(0.5, 8), task, tn.build_loss_criterion(task),
                   torch.optim.Adam(network.parameters()), 'cpu', capture_nonfinite=str(path))
    assert not list(tmp_path.iterdir())


def test_atomic_capture_never_clobbers_existing_file(tmp_path):
    from parrot.training_persistence import atomic_save
    path = tmp_path / 'capture.pt'
    atomic_save({'first': 1}, str(path), overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_save({'second': 2}, str(path), overwrite=False)
    assert torch.load(path, weights_only=True) == {'first': 1}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('command,extra', [(parrot_train, []), (parrot_optimize, ['--trials', '1'])])
def test_actual_cli_routes_failure_capture(tmp_path, run_cli, monkeypatch, command, extra):
    from tests.helpers import data_path, training_command
    original = tn.compute_batch_loss

    def fail_backward(*args, **kwargs):
        loss, outputs = original(*args, **kwargs)
        if loss.requires_grad:
            loss.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
        return loss, outputs

    monkeypatch.setattr(tn, 'compute_batch_loss', fail_backward)
    capture = tmp_path / 'failure.pt'
    with pytest.raises(RuntimeError):
        run_cli(command.main, training_command(command.PROGRAM_NAME, 'brnn', data_path('res_regress_dataset.tsv'),
                                               tmp_path / 'network.pt', 'residues', 1, '-e', '1', '-hs', '2',
                                               '-nl', '1', '--silent', '--capture-nonfinite', capture, *extra))
    saved = torch.load(capture, weights_only=True)
    assert saved['network']['architecture'] == 'brnn' and saved['diagnostics']['nan'] > 0


@pytest.mark.parametrize('module', [parrot_train, parrot_optimize])
def test_capture_can_be_enabled_when_resuming_existing_checkpoint(tmp_path, run_cli, monkeypatch, module):
    from parrot.scripts import search_artifacts
    from tests.test_training_resume_cli import command, data_file
    program = module.PROGRAM_NAME
    data = data_file(tmp_path)
    output = tmp_path / 'network.pt'
    extra = ['--trials', '2'] if module is parrot_optimize else []
    saving = search_artifacts if module is parrot_optimize else tn
    original = saving.atomic_save

    def interrupt(payload, path):
        original(payload, path)
        if payload['records']:
            raise RuntimeError('interrupted')

    monkeypatch.setattr(saving, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        run_cli(module.main, command(program, data, output, *extra))
    monkeypatch.setattr(saving, 'atomic_save', original)
    capture = tmp_path / 'capture.pt'
    run_cli(module.main, [program, '--resume', tmp_path / 'network_training_checkpoint.pt',
                          '--capture-nonfinite', capture])
    assert output.is_file() and not capture.exists()


def test_diagnostic_precision_comparison_uses_same_weights_and_batches(tmp_path, monkeypatch):
    def cpu_bf16(mode, device):
        precision = TrainingPrecision('off', 'cpu')
        precision.dtype = torch.bfloat16
        precision.autocast = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
        return precision

    monkeypatch.setattr(diagnose_brnn, 'TrainingPrecision', cpu_bf16)
    output = tmp_path / 'comparison'
    diagnose_brnn.main(['--device', 'cpu', '--output-dir', str(output), '--epochs', '2',
                       '--hidden-size', '2', '--num-layers', '2', '--batch', '2',
                       '--synthetic-lengths', '8', '17'])
    report = json.loads((output / 'summary.json').read_text())
    assert all(row['status'] == 'completed' for row in report['pilots'])
    assert report['pilots'][0]['initial_weights_sha256'] == report['pilots'][1]['initial_weights_sha256']
    logs = [[json.loads(line) for line in (output / name).read_text().splitlines()]
            for name in ('fp32_batches.jsonl', 'bfloat16_batches.jsonl')]
    assert [row['indices_sha256'] for row in logs[0]] == [row['indices_sha256'] for row in logs[1]]


@pytest.mark.real_devices
def test_native_cuda_long_brnn_precision_pilot(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip('Native BF16 unavailable')
    output = tmp_path / 'cuda_comparison'
    diagnose_brnn.main(['--output-dir', str(output), '--epochs', '2', '--hidden-size', '10', '--num-layers', '2',
                       '--batch', '2', '--synthetic-lengths', '129', '513', '--gradient-clip-norm', '1'])
    report = json.loads((output / 'summary.json').read_text())
    assert all(row['status'] == 'completed' for row in report['pilots'])


def test_cpu_diagnostic_pilot_and_order_match(tmp_path):
    output = tmp_path / 'pilot'
    diagnose_brnn.main(['--device', 'cpu', '--modes', 'off', '--output-dir', str(output),
                       '--epochs', '2', '--hidden-size', '2', '--num-layers', '1', '--batch', '2',
                       '--synthetic-lengths', '8', '17', '--gradient-clip-norm', '1'])
    result = json.loads((output / 'summary.json').read_text())
    assert result['pilots'][0]['status'] == 'completed'
    rows = [json.loads(line) for line in (output / 'fp32_batches.jsonl').read_text().splitlines()]
    assert len(rows) == 4 and all(row['post_clip_norm'] <= 1.00001 for row in rows)
    assert not list(output.glob('*failure.pt'))
    args = diagnose_brnn.build_parser().parse_args(['--output-dir', 'unused', '--batch', '2'])
    dataset = pid.SequenceDataset([[str(i), 'A' * length, np.zeros(length)]
                                   for i, length in enumerate([8, 9, 13, 17, 20])])
    expected = diagnose_brnn.epoch_batches(dataset, Task('residues', 1), args, 0)
    torch.randn(1000)
    assert expected == diagnose_brnn.epoch_batches(dataset, Task('residues', 1), args, 0)


@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_float64_labels_rejected_with_original_line(datatype):
    from parrot.parrot_exceptions import IOExceptionParrot
    with pytest.raises(IOExceptionParrot, match='Data line 1.*finite float32') as raised:
        pid.parse_lines([['original_protein', 'A', '1e40']], datatype)
    assert 'original_protein' in str(raised.value)


def test_collation_rejects_large_custom_encodings_and_programmatic_targets():
    with pytest.raises(ValueError, match="Encoded sequence 'original_protein'.*finite float32"):
        pid.collate_batch([('original_protein', torch.tensor([[1e40]], dtype=torch.float64), np.array([0.]))],
                           Task('residues', 1))
    with pytest.raises(ValueError, match="Targets for sequence 'original_protein'.*finite float32"):
        pid.collate_batch([('original_protein', torch.zeros(1, 20), np.array([1e40]))], Task('residues', 1))
