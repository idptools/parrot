"""Production regressions for stable norms, clipping and float32 input validation."""

import math

import numpy as np
import pytest
import torch

from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.gradient_numerics import clip_with_norm, stable_total_norm
from parrot.parrot_exceptions import IOExceptionParrot
from parrot.tasks import Task
from parrot.training_precision import TrainingPrecision
from tests.test_gradient_safety import TinyRegressor, make_loader


def test_finite_brnn_gradients_with_overflowed_norm_can_be_clipped():
    task = Task('residues', 1)
    network = build_network('brnn', 20, task, {'hidden_size': 1, 'num_layers': 1})
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        for suffix in ('', '_reverse'):
            getattr(network.lstm, 'weight_hh_l0' + suffix)[2, 0] = 2.
            getattr(network.lstm, 'bias_ih_l0' + suffix)[[0, 1, 3]] = 5.
        network.fc.weight.fill_(1.)
    optimizer = torch.optim.Adam(network.parameters())
    statistics = tn.GradientStatistics()
    tn.train_epoch(network, make_loader(0.5, 48), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                   gradient_clip_norm=1., gradient_statistics=statistics)
    assert 1e21 < statistics.norms[0] < 1e24
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in network.parameters())
    assert float(stable_total_norm([p.grad for p in network.parameters()])) == pytest.approx(1., rel=1e-5)
    assert tn.finite_tensors(network.state_dict()) and tn.finite_tensors(optimizer.state_dict())


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
@pytest.mark.real_devices
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


def test_checks_alone_allow_finite_gradients_with_overflowed_norm():
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: gradient * 1e20)
    task = Task('residues', 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-22)
    tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                   check_finite_gradients=True)
    assert float(network.fc.weight[0, 0].detach()) == pytest.approx(0.08)


def test_finite_norm_fallback_does_not_hide_overflowing_adam_moments():
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: gradient * 1e22)
    task = Task('residues', 1)
    optimizer = torch.optim.Adam(network.parameters())
    with pytest.raises(tn.NonFiniteTrainingError, match='model or optimizer state') as raised:
        tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       check_finite_gradients=True)
    assert 'after the optimizer update' in str(raised.value)
    assert not tn.finite_tensors(optimizer.state_dict())


@pytest.mark.parametrize('bad', [None, math.nan, math.inf])
def test_amp_gradient_guard_clips_finite_overflow_and_rejects_nonfinite_entries(bad):
    # CPU autocast exercises the common AMP gradient guard, not CUDA backend support.
    network = TinyRegressor()
    network.fc.weight.register_hook(lambda gradient: gradient * 1e20 if bad is None else
                                    torch.full_like(gradient, bad))
    task = Task('residues', 1)
    optimizer = torch.optim.Adam(network.parameters())
    precision = TrainingPrecision('off', 'cpu')
    precision.dtype = torch.bfloat16
    precision.autocast = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
    if bad is not None:
        with pytest.raises(tn.NonFiniteTrainingError, match='AMP gradient norm'):
            tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                           gradient_clip_norm=1., precision=precision)
        assert not optimizer.state
    else:
        tn.train_epoch(network, make_loader(), task, tn.build_loss_criterion(task), optimizer, 'cpu',
                       gradient_clip_norm=1., precision=precision)
        assert float(stable_total_norm([p.grad for p in network.parameters()])) == pytest.approx(1., rel=1e-5)
        assert tn.finite_tensors(optimizer.state_dict())


@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_float64_labels_rejected_with_original_line(datatype):
    with pytest.raises(IOExceptionParrot, match='Data line 1.*finite float32') as raised:
        pid.parse_lines([['original_protein', 'A', '1e40']], datatype)
    assert 'original_protein' in str(raised.value)


def test_collation_rejects_large_custom_encodings_and_programmatic_targets():
    with pytest.raises(ValueError, match="Encoded sequence 'original_protein'.*finite float32"):
        pid.collate_batch([('original_protein', torch.tensor([[1e40]], dtype=torch.float64), np.array([0.]))],
                           Task('residues', 1))
    with pytest.raises(ValueError, match="Targets for sequence 'original_protein'.*finite float32"):
        pid.collate_batch([('original_protein', torch.zeros(1, 20), np.array([1e40]))], Task('residues', 1))
