"""End-to-end loss, checkpoint/resume, evaluation and guarded AMP behavior."""

import copy
import csv
import math
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

from parrot import metrics, reporting
from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.checkpoint import load_network
from parrot.reproducibility import set_random_seed
from parrot.tasks import Task
from parrot.training_control import TrainingControlState, TrainingOptions
from parrot.training_persistence import (
    atomic_save,
    capture_random_state,
    load_training_checkpoint,
    restore_random_state,
)
from parrot.training_precision import TrainingPrecision, resolve_amp


def samples():
    return [['a', 'ACD', np.array([0.2, math.nan, 1.2])],
            ['b', 'CDEFGA', np.array([0.1, 0.3, 0.7, 0.8, 0.9, 0.2])],
            ['c', 'FFFF', np.array([0.5, 0.1, 0.3, 0.8])], ['d', 'ACK', np.array([0.1, 0.8, 0.4])]]


def read_tsv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def training_parts(task=None, architecture='brnn'):
    task = task or Task('residues', 1)
    values = {'hidden_size': 4, 'num_layers': 2, 'dropout': 0.3} if architecture == 'brnn' else \
        {'channels': 4, 'num_blocks': 1, 'dropout': 0.3}
    network = build_network(architecture, 20, task, values)
    dataset = pid.SequenceDataset(samples())
    return (network, pid.make_dataloader(dataset, task, 2, shuffle=True),
            pid.make_dataloader(dataset, task, 2), task)


@pytest.mark.parametrize('loss,expected', [('mae', 4.0), ('mse', 10.0), ('huber', 3.0)])
def test_selected_loss_values_and_checkpoint_roundtrip(loss, expected, tmp_path):
    task = Task('sequence', 1, regression_loss=loss)
    criterion = tn.build_loss_criterion(task)
    actual = criterion(torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [3.0]]))
    assert actual.item() == expected
    assert Task.from_config(task.to_config()) == task
    network = build_network('brnn', 20, task)
    from parrot.checkpoint import save_network
    from parrot.encode_sequence import get_encoder
    path = str(tmp_path / 'loss.pt')
    save_network(path, network, get_encoder('onehot'))
    restored, _ = load_network(path)
    assert restored.task == task


def test_huber_equal_weights_exclude_missing_and_padding():
    task = Task('residues', 1, regression_loss='huber', huber_delta=0.5)
    network, train, _, _ = training_parts(task)
    for parameter in network.parameters():
        parameter.data.zero_()
    batch = next(iter(train))
    computed, _ = tn.compute_batch_loss(network, batch, task, tn.build_loss_criterion(task), 'cpu',
                                        equal_protein_weighting=True)
    labelled = [values[:length][torch.isfinite(values[:length])]
                for values, length in zip(batch.targets[..., 0], batch.lengths)]
    expected = sum(torch.nn.functional.huber_loss(torch.zeros_like(row), row, delta=0.5, reduction='mean')
                   for row in labelled)
    assert computed.item() == pytest.approx(expected.item())
    computed.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


@pytest.mark.parametrize('task', [lambda: Task('residues', 2, regression_loss='mse'),
                                  lambda: Task('residues', 1, uncertainties=True, regression_loss='huber'),
                                  lambda: Task('sequence', 1, regression_loss='huber', huber_delta=math.nan)])
def test_incompatible_loss_settings_fail_early(task):
    with pytest.raises(ValueError):
        task()


@pytest.mark.parametrize('schedule', ['fixed', 'cosine', 'plateau'])
def test_resume_matches_uninterrupted_dropout_adamw_lr_and_randomness(tmp_path, monkeypatch, schedule):
    options = TrainingOptions(optimizer='adamw', weight_decay=0.01, lr_schedule=schedule,
                              lr_anneal_epochs=4 if schedule == 'cosine' else None, lr_warmup_epochs=1,
                              lr_patience=0, equal_protein_weighting=True)
    set_random_seed(83)
    network, train, val, task = training_parts()
    initial = copy.deepcopy(network.state_dict())
    before = torch.get_rng_state().clone()
    expected = tn.train(network, train, val, task, 'cpu', 0.003, 5, silent=True, training_options=options)
    expected_weights = copy.deepcopy(network.state_dict())
    expected_random = (torch.rand(4), np.random.rand(4), random.random())

    set_random_seed(83)
    interrupted, train, val, task = training_parts()
    interrupted.load_state_dict(initial)
    torch.set_rng_state(before)
    path, log = str(tmp_path / 'latest.pt'), str(tmp_path / 'epochs.tsv')
    real_save = tn.atomic_save

    def crash_after_second_epoch(payload, target):
        real_save(payload, target)
        if payload['epoch'] == 2:
            raise RuntimeError('simulated job termination')

    monkeypatch.setattr(tn, 'atomic_save', crash_after_second_epoch)
    with pytest.raises(RuntimeError, match='simulated'):
        tn.train(interrupted, train, val, task, 'cpu', 0.003, 5, silent=True, training_options=options,
                 checkpoint_path=path, training_log=log)
    saved = load_training_checkpoint(path, 'train')
    assert saved['epoch'] == 2
    assert len(read_tsv(log)) == 2
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    set_random_seed(917)
    resumed, train, val, task = training_parts()
    result = tn.train(resumed, train, val, task, 'cpu', 0.003, 5, silent=True, training_options=options,
                      checkpoint_path=path, resume_from=path, training_log=log)
    assert result == expected
    for key, value in resumed.state_dict().items():
        assert torch.equal(value, expected_weights[key]), key
    assert torch.equal(torch.rand(4), expected_random[0])
    assert np.array_equal(np.random.rand(4), expected_random[1])
    assert random.random() == expected_random[2]
    assert [int(row['epoch']) for row in read_tsv(log)] == [1, 2, 3, 4, 5]


def test_resume_rejects_changed_data_without_changing_checkpoint_or_log(tmp_path):
    network, train, val, task = training_parts()
    path, log = str(tmp_path / 'latest.pt'), str(tmp_path / 'epochs.tsv')
    tn.train(network, train, val, task, 'cpu', 0.001, 1, silent=True, checkpoint_path=path, training_log=log)
    original = (Path(path).read_bytes(), Path(log).read_bytes())
    train.dataset.data[0][2][0] = 19.0
    with pytest.raises(ValueError, match='loaders'):
        tn.train(network, train, val, task, 'cpu', 0.001, 2, silent=True, checkpoint_path=path,
                 resume_from=path, training_log=log)
    assert (Path(path).read_bytes(), Path(log).read_bytes()) == original


def sampled_loader(dataset, task, kind, changed=False):
    if kind.startswith('random'):
        source = range(3) if changed and kind == 'random-source' else dataset
        sampler = RandomSampler(source, replacement=changed if kind == 'random-replacement' else False,
                                num_samples=8 if changed and kind == 'random-count' else 4)
    elif kind.startswith('weighted'):
        weights = [10.0, 1.0, 1.0, 1.0] if changed and kind == 'weighted-values' else [1.0] * 4
        sampler = WeightedRandomSampler(weights, 8 if changed and kind == 'weighted-count' else 4,
                                        replacement=not changed if kind == 'weighted-replacement' else True)
    elif kind == 'subset':
        sampler = SubsetRandomSampler([3, 2, 1, 0] if changed else range(4))
    elif kind == 'sequential-source':
        sampler = SequentialSampler(range(3) if changed else dataset)
    elif kind.startswith('length'):
        lengths = [len(sequence) for sequence in dataset.sequences]
        if changed and kind == 'length-values':
            lengths.reverse()
        sampler = pid.LengthBatchSampler(lengths, 2, shuffle=not changed if kind == 'length-shuffle' else True)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(pid.collate_batch, task=task))
    else:
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, 2, drop_last=changed)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=partial(pid.collate_batch, task=task))
    return DataLoader(dataset, sampler=sampler, batch_size=2, collate_fn=partial(pid.collate_batch, task=task))


@pytest.mark.parametrize('kind', ['random-replacement', 'random-count', 'weighted-replacement', 'weighted-count',
                                  'weighted-values', 'subset', 'length-values', 'length-shuffle', 'batch-drop-last',
                                  'random-source', 'sequential-source'])
def test_resume_rejects_changed_sampling_before_touching_network_or_artifacts(tmp_path, kind):
    network, train, val, task = training_parts()
    original = sampled_loader(train.dataset, task, kind)
    changed = sampled_loader(train.dataset, task, kind, changed=True)
    path, log = str(tmp_path / 'latest.pt'), str(tmp_path / 'epochs.tsv')
    tn.train(network, original, val, task, 'cpu', 0.001, 1, silent=True, checkpoint_path=path, training_log=log)
    saved_bytes = Path(path).read_bytes(), Path(log).read_bytes()
    network, _, _, _ = training_parts(task)
    before = copy.deepcopy(network.state_dict())
    with pytest.raises(ValueError, match='loaders'):
        tn.train(network, changed, val, task, 'cpu', 0.001, 2, silent=True, checkpoint_path=path,
                 resume_from=path, training_log=log)
    assert (Path(path).read_bytes(), Path(log).read_bytes()) == saved_bytes
    assert all(torch.equal(value, before[name]) for name, value in network.state_dict().items())


@pytest.mark.parametrize('custom_batch', [False, True])
def test_checkpoint_rejects_untracked_custom_sampler_before_writing(tmp_path, custom_batch):
    class CustomSampler(SequentialSampler):
        pass

    class CustomBatchSampler(BatchSampler):
        pass

    network, train, val, task = training_parts()
    dataset = train.dataset
    sampler = SequentialSampler(dataset) if custom_batch else CustomSampler(dataset)
    batches = CustomBatchSampler(sampler, 2, False) if custom_batch else BatchSampler(sampler, 2, False)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=partial(pid.collate_batch, task=task))
    path = str(tmp_path / 'checkpoint.pt')
    with pytest.raises(ValueError, match='sampler'):
        tn.train(network, loader, val, task, 'cpu', 0.001, 1, silent=True, checkpoint_path=path)
    assert not Path(path).exists()


@pytest.mark.parametrize('kind', ['random-count', 'weighted-values', 'subset'])
def test_supported_sampling_policies_resume_with_identical_weights_and_rng(tmp_path, monkeypatch, kind):
    def parts():
        network, train, val, task = training_parts()
        return network, sampled_loader(train.dataset, task, kind), val, task

    set_random_seed(97)
    expected, train, val, task = parts()
    losses = tn.train(expected, train, val, task, 'cpu', 0.001, 3, silent=True)
    expected_rng = torch.get_rng_state().clone()
    set_random_seed(97)
    network, train, val, task = parts()
    path = str(tmp_path / 'latest.pt')
    real_save = tn.atomic_save

    def interrupt(payload, target):
        real_save(payload, target)
        if payload['epoch'] == 1:
            raise RuntimeError('job interrupted')

    monkeypatch.setattr(tn, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='job interrupted'):
        tn.train(network, train, val, task, 'cpu', 0.001, 3, silent=True, checkpoint_path=path)
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    resumed, train, val, task = parts()
    assert tn.train(resumed, train, val, task, 'cpu', 0.001, 3, silent=True, resume_from=path) == losses
    assert all(torch.equal(value, expected.state_dict()[name]) for name, value in resumed.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_resume_reconciles_log_rows_after_last_checkpoint(tmp_path, monkeypatch):
    network, train, val, task = training_parts()
    path, log = str(tmp_path / 'latest.pt'), str(tmp_path / 'epochs.tsv')
    real_save = tn.atomic_save
    def fail_third_checkpoint(payload, target):
        if payload['epoch'] == 3:
            raise OSError('disk full')
        real_save(payload, target)
    monkeypatch.setattr(tn, 'atomic_save', fail_third_checkpoint)
    with pytest.raises(OSError, match='disk full'):
        tn.train(network, train, val, task, 'cpu', 0.001, 4, silent=True, checkpoint_path=path, training_log=log)
    assert load_training_checkpoint(path)['epoch'] == 2
    assert len(read_tsv(log)) == 3
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    tn.train(network, train, val, task, 'cpu', 0.001, 4, silent=True, checkpoint_path=path, resume_from=path,
             training_log=log)
    assert [int(row['epoch']) for row in read_tsv(log)] == [1, 2, 3, 4]


def test_auto_stop_patience_continues_after_resume(tmp_path, monkeypatch):
    network, train, val, task = training_parts()
    monkeypatch.setattr(tn, 'validation_loss', lambda *args, **kwargs: 1.0)
    path = str(tmp_path / 'latest.pt')
    real_save = tn.atomic_save
    def interrupt(payload, target):
        real_save(payload, target)
        if payload['epoch'] == 2:
            raise RuntimeError('interrupted')
    monkeypatch.setattr(tn, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        tn.train(network, train, val, task, 'cpu', 0.001, 2, stop_condition='auto', silent=True, checkpoint_path=path)
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    result = tn.train(network, train, val, task, 'cpu', 0.001, 2, stop_condition='auto', silent=True,
                      checkpoint_path=path, resume_from=path)
    assert len(result[1]) == 3


def test_recovery_budget_and_reduced_rate_survive_interruption_before_replay(tmp_path, monkeypatch):
    network, train, val, task = training_parts()
    path = str(tmp_path / 'latest.pt')
    real_train_epoch, real_save = tn.train_epoch, tn.atomic_save
    def fail_batch(*args, **kwargs):
        raise tn.NonFiniteTrainingError('simulated gradient overflow')
    def interrupt_on_recovery(payload, target):
        real_save(payload, target)
        if payload['control_state']['recoveries_used']:
            raise RuntimeError('interrupted before replay')
    monkeypatch.setattr(tn, 'train_epoch', fail_batch)
    monkeypatch.setattr(tn, 'atomic_save', interrupt_on_recovery)
    options = TrainingOptions(recover_nonfinite=1)
    with pytest.raises(RuntimeError, match='before replay'):
        tn.train(network, train, val, task, 'cpu', 0.001, 2, silent=True, checkpoint_path=path, training_options=options)
    saved = load_training_checkpoint(path)
    assert saved['epoch'] == 0
    assert saved['control_state']['recoveries_used'] == 1
    assert saved['optimizer_state']['param_groups'][0]['lr'] == 0.0005
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    with pytest.raises(tn.NonFiniteTrainingError, match='budget exhausted'):
        tn.train(network, train, val, task, 'cpu', 0.001, 2, silent=True, checkpoint_path=path,
                 resume_from=path, training_options=options)
    monkeypatch.setattr(tn, 'train_epoch', real_train_epoch)


def test_atomic_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = str(tmp_path / 'latest.pt')
    atomic_save({'old': torch.tensor(1)}, path)
    def broken_save(payload, handle):
        handle.write(b'partial bytes')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(torch, 'save', broken_save)
    with pytest.raises(OSError, match='disk failure'):
        atomic_save({'new': torch.tensor(2)}, path)
    assert torch.load(path, weights_only=True)['old'].item() == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_restores_explicit_generators_and_rejects_changed_sharing():
    _, train, val, _ = training_parts()
    first, second = torch.Generator().manual_seed(71), torch.Generator().manual_seed(91)
    train.generator, train.sampler.generator, val.generator = first, second, first
    state = capture_random_state((train, val), 'cpu')
    expected = (torch.rand(4, generator=first), torch.rand(4, generator=second))
    restore_random_state(state, (train, val), 'cpu')
    assert torch.equal(torch.rand(4, generator=first), expected[0])
    assert torch.equal(torch.rand(4, generator=second), expected[1])
    val.generator = second
    with pytest.raises(ValueError, match='generator structure'):
        restore_random_state(state, (train, val), 'cpu')


def test_regression_metrics_detect_bias_and_distinguish_protein_weights():
    actual = [np.array([0.0]), np.array([1.0] * 4)]
    predicted = [np.array([10.0]), np.array([2.0] * 4)]
    result = metrics.performance_metrics(actual, predicted, Task('residues', 1))
    assert result['MAE'] == 2.8
    assert result['RMSE'] == pytest.approx(math.sqrt(104 / 5), abs=0.001)
    assert result['Mean error'] == 2.8
    assert result['Mean protein MAE'] == 5.5
    assert result['R2'] < 0


def test_metric_masks_and_length_tables_match_pooled_values(tmp_path):
    task = Task('matrix', 1, symmetric=True)
    truth = np.array([[1.0, math.nan], [math.nan, 3.0]])
    predicted = np.array([[2.0, 999.0], [999.0, 5.0]])
    from parrot.prediction import LabeledPredictions
    labeled = LabeledPredictions(['protein'], ['AC'], [truth], [predicted[..., None]], 0.0)
    result, files = reporting.report_test_results(labeled, task, str(tmp_path / 'test'))
    assert result['MAE'] == 1.5
    table = next(path for path in files if path.endswith('_per_protein_metrics.tsv'))
    row = read_tsv(table)[0]
    assert row['labelled_values'] == '2'
    assert float(row['MAE']) == 1.5


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_amp_never_silently_falls_back_to_unsupported_device(device):
    with pytest.raises(ValueError, match='CUDA'):
        resolve_amp('auto', device)
    assert resolve_amp('off', device) is None


def test_amp_capability_checks_use_selected_gpu(monkeypatch):
    import parrot.training_precision as precision
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    selected = []
    monkeypatch.setattr(torch.cuda, 'device', lambda device: selected.append(str(device)) or nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation: True)
    assert precision.resolve_amp('auto', 'cuda:2') == torch.bfloat16
    assert selected == ['cuda:2']
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation: False)
    assert precision.resolve_amp('auto', 'cuda:2') == torch.float16
    with pytest.raises(ValueError, match='native bfloat16'):
        precision.resolve_amp('bfloat16', 'cuda:2')


@pytest.mark.parametrize('always_overflow', [False, True])
def test_real_scaler_unscales_before_clipping_and_replays_instead_of_skipping(monkeypatch, always_overflow):
    """Exercise the actual GradScaler on CPU; this does not substitute for CUDA integration tests."""
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state, device: None)
    policy = TrainingPrecision('off', 'cpu')
    policy.scaler = torch.amp.GradScaler('cpu', init_scale=128.0)
    network = torch.nn.Linear(1, 1, bias=False)
    network.weight.data.fill_(1)
    network.weight.register_hook(lambda grad: grad * math.inf if always_overflow or grad.abs().max() > 128 else grad)
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network
            self.register_buffer('forwards', torch.tensor(0))
        def forward(self, vectors, lengths):
            self.forwards += 1
            return self.network(vectors)
    wrapped = Wrapped()
    batch = pid.Batch(names=['x'], vectors=torch.tensor([[[1.0]]]), lengths=torch.tensor([1]),
                      targets=torch.tensor([[[0.0]]]))
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    if always_overflow:
        with pytest.raises(tn.NonFiniteTrainingError, match='retry budget exhausted'):
            tn.train_amp_batch(wrapped, batch, Task('residues', 1), torch.nn.MSELoss(reduction='sum'),
                                optimizer, 'cpu', policy, 1, 1.0, None, False)
        assert network.weight.item() == 1.0
        assert policy.scale_retries == 8
        return
    result = tn.train_amp_batch(wrapped, batch, Task('residues', 1), torch.nn.MSELoss(reduction='sum'),
                                 optimizer, 'cpu', policy, 1, 1.0, None, False)
    assert result.item() == 1.0
    assert policy.scale_retries == 1
    assert wrapped.forwards.item() == 1  # failed forward's buffer update was restored
    assert network.weight.item() == pytest.approx(0.9)


@pytest.mark.parametrize('uncertainties', [False, True])
def test_autocast_forward_keeps_loss_and_parameter_gradients_in_fp32(uncertainties):
    """Exercise BF16 autocast/loss boundaries on CPU, independent of CUDA availability."""
    task = Task('residues', 1, uncertainties=uncertainties,
                regression_loss='auto' if uncertainties else 'huber')
    data = samples()
    if uncertainties:
        data = [row + [np.full(len(row[1]), 0.1)] for row in data]
    loader = pid.make_dataloader(pid.SequenceDataset(data), task, 2)
    network = build_network('composition', 20, task)
    if uncertainties:
        tn.refit_model_error(network, loader, task, 'cpu', equal_protein_weighting=True)
    policy = TrainingPrecision('off', 'cpu')
    policy.autocast = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
    loss, outputs = tn.compute_batch_loss(network, next(iter(loader)), task, tn.build_loss_criterion(task), 'cpu',
                                          equal_protein_weighting=True, precision=policy)
    assert outputs.dtype == loss.dtype == torch.float32
    loss.backward()
    assert all(parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters() if parameter.grad is not None)


@pytest.mark.real_devices
@pytest.mark.parametrize('amp', ['float16', 'bfloat16'])
@pytest.mark.parametrize('architecture', ['cnn', 'transformer', 'unet', 'composition', 'unet2d'])
def test_cuda_amp_training_keeps_fp32_state_and_finite_updates(architecture, amp):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 0):
        pytest.skip('A CUDA GPU supporting PARROT AMP is required')
    if amp == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip('Native BF16 support is required')
    task = Task('matrix', 1, width=2) if architecture == 'unet2d' else Task('residues', 1)
    data = pid.SequenceDataset([['x', 'ACDEFG', np.zeros((6, 2) if task.is_matrix else 6)]])
    network = build_network(architecture, 20, task).cuda()
    loader = pid.make_dataloader(data, task, 1)
    state = TrainingControlState()
    tn.train(network, loader, loader, task, 'cuda', 0.001, 2, silent=True, gradient_clip_norm=1.0,
             training_options=TrainingOptions(amp=amp, recover_nonfinite=1), control_state=state)
    assert all(parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
               for parameter in network.parameters())
    assert state.precision_state is not None


@pytest.mark.real_devices
@pytest.mark.parametrize('amp', ['float16', 'bfloat16'])
def test_cuda_amp_checkpoint_resume_restores_scaler_dropout_and_adamw(tmp_path, monkeypatch, amp):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 0):
        pytest.skip('CUDA AMP hardware required')
    if amp == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip('Native BF16 support required')
    options = TrainingOptions(amp=amp, optimizer='adamw', weight_decay=0.01, lr_schedule='exponential')
    set_random_seed(419)
    network, train, val, task = training_parts(architecture='cnn')
    network.cuda()
    expected = tn.train(network, train, val, task, 'cuda', 0.001, 3, silent=True, training_options=options)
    expected_weights = copy.deepcopy(network.state_dict())
    set_random_seed(419)
    interrupted, train, val, task = training_parts(architecture='cnn')
    interrupted.cuda()
    path = str(tmp_path / 'amp.pt')
    real_save = tn.atomic_save
    def interrupt(payload, target):
        real_save(payload, target)
        if payload['epoch'] == 1:
            raise RuntimeError('job killed')
    monkeypatch.setattr(tn, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='job killed'):
        tn.train(interrupted, train, val, task, 'cuda', 0.001, 3, silent=True, training_options=options,
                 checkpoint_path=path)
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    result = tn.train(interrupted, train, val, task, 'cuda', 0.001, 3, silent=True, training_options=options,
                      checkpoint_path=path, resume_from=path)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    for name, value in interrupted.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=1e-5, atol=1e-6)
