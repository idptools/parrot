"""
Tests for running PARROT on different devices.

- Device choice (parrot/devices.py and the --device/--gpu-id flags),
  with CUDA and MPS availability faked so these run on any machine.
- The BRNN's LSTM, which runs each direction on unpacked tensors and must give
  the same results as running the LSTM on a packed batch.
- Checkpoints, which must hold CPU tensors whatever device trained them.
- Real runs on NVIDIA (CUDA) and Apple-silicon (MPS) GPUs, each skipped on
  machines without one.
"""

import argparse
import copy
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from parrot import devices
from parrot import process_input_data as pid
from parrot.architectures.brnn import reverse_within_lengths, run_lstm_unpacked
from parrot.checkpoint import load_network, save_network
from parrot.encode_sequence import get_encoder
from parrot.py_predictor import Predictor
from parrot.scripts import parrot_optimize, parrot_predict, parrot_train
from parrot.scripts.common import select_device
from parrot.tasks import Task
from parrot.prediction import predict_labeled, predict_sequences
from parrot.train_network import evaluate_loss, train

from tests.helpers import (BUILT_IN_ARCHITECTURES, EXAMPLE_TASKS, FITTING_SEQUENCES, FULL_PRECISION, NEW_PRECISION_API,
                           TF32_EVERYWHERE, architecture_class, data_path, make_network, ready_network, supports,
                           tf32_settings, training_command)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no NVIDIA GPU (CUDA) on this machine')
requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(),
                                  reason='no Apple-silicon GPU (MPS) on this machine')

# The architectures trained by gradient descent (the neural networks)
NEURAL_ARCHITECTURES = [name for name in BUILT_IN_ARCHITECTURES if architecture_class(name).trained_by_gradient_descent]

# The GPUs PARROT can run on; tests parametrized over these run on whichever this machine has
GPUS = [pytest.param('cuda', marks=requires_cuda), pytest.param('mps', marks=requires_mps)]

# float32 results on different devices, or from differently ordered computations, agree to about this
FLOAT32_TOLERANCE = 1e-5


@pytest.fixture
def full_float32_precision() -> Iterator[None]:
    """Run the test in full float32 precision, as PARROT runs its networks (see devices.full_float32_precision)

    Only needed by tests that call a network or an LSTM directly rather than
    through PARROT's training and prediction functions.
    """
    with devices.full_float32_precision():
        yield


def fake_availability(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, cuda_devices: int = 1) -> None:
    """Pretend this machine has (or lacks) CUDA and MPS devices."""
    monkeypatch.setattr(devices, 'cuda_available', lambda: cuda)
    monkeypatch.setattr(devices, 'mps_available', lambda: mps)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: cuda_devices if cuda else 0)


# ..........................................................................................
# Choosing a device

@pytest.mark.parametrize('cuda, mps, expected', [
    (True, True, 'cuda'),
    (True, False, 'cuda'),
    (False, True, 'mps'),
    (False, False, 'cpu'),
])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    fake_availability(monkeypatch, cuda, mps)
    assert devices.resolve_device('auto') == torch.device(expected)


def test_cpu_is_always_available(monkeypatch):
    fake_availability(monkeypatch, cuda=True, mps=True)
    assert devices.resolve_device('cpu') == torch.device('cpu')


@pytest.mark.parametrize('requested', ['cuda', 'mps'])
def test_unavailable_device_fails(monkeypatch, requested):
    fake_availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match='not available'):
        devices.resolve_device(requested)


def test_gpu_id_selects_cuda_device(monkeypatch):
    fake_availability(monkeypatch, cuda=True, mps=False, cuda_devices=2)
    assert devices.resolve_device('auto', gpu_id=1) == torch.device('cuda:1')
    assert devices.resolve_device('cuda', gpu_id=0) == torch.device('cuda:0')


def test_gpu_id_out_of_range_fails(monkeypatch):
    fake_availability(monkeypatch, cuda=True, mps=False, cuda_devices=2)
    with pytest.raises(ValueError, match='does not exist'):
        devices.resolve_device('cuda', gpu_id=2)


def test_gpu_id_without_cuda_fails(monkeypatch):
    fake_availability(monkeypatch, cuda=False, mps=True)
    with pytest.raises(RuntimeError, match='CUDA'):
        devices.resolve_device('auto', gpu_id=0)


@pytest.mark.parametrize('requested', ['cpu', 'mps'])
def test_gpu_id_with_other_device_fails(monkeypatch, requested):
    fake_availability(monkeypatch, cuda=True, mps=True)
    with pytest.raises(ValueError, match='cannot be combined'):
        devices.resolve_device(requested, gpu_id=0)


def test_unknown_device_fails():
    with pytest.raises(ValueError, match='Unknown device'):
        devices.resolve_device('tpu')


def device_arguments(device: str = 'auto', gpu_id: int | None = None) -> argparse.Namespace:
    """Parsed arguments as add_device_arguments() would produce them."""
    return argparse.Namespace(device=device, gpu_id=gpu_id, silent=True)


def test_cpu_can_be_chosen_when_gpus_are_available(monkeypatch):
    fake_availability(monkeypatch, cuda=True, mps=True)
    assert select_device(device_arguments(device='cpu')) == torch.device('cpu')


@pytest.mark.parametrize('main_function, arguments', [
    (parrot_train.main, ['parrot-train', 'brnn', '--force-cpu']),
    (parrot_predict.main, ['parrot-predict', 'sequences.txt', 'net.pt', 'out.txt', '--force-cpu']),
], ids=['train', 'predict'])
def test_force_cpu_is_retired(run_cli, capsys, main_function, arguments):
    with pytest.raises(SystemExit):
        run_cli(main_function, arguments)
    assert '--force-cpu is no longer an option: use --device cpu' in capsys.readouterr().err


def test_train_on_unavailable_device_fails(run_cli, tmp_path):
    # GPUs are hidden from tests not marked real_devices (see conftest.py)
    filename, datatype, num_classes = EXAMPLE_TASKS['seq_regress']
    with pytest.raises(RuntimeError, match='MPS'):
        run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(filename), tmp_path / 'net.pt',
                                                    datatype, num_classes, '-e', '1', '--device', 'mps', '--silent'))


# ..........................................................................................
# Full float32 precision (no TF32)

def test_full_float32_precision_turns_tf32_off_and_restores_it(tf32_allowed):
    with devices.full_float32_precision():
        assert tf32_settings() == FULL_PRECISION
    assert tf32_settings() == TF32_EVERYWHERE

    with pytest.raises(RuntimeError), devices.full_float32_precision():
        raise RuntimeError('failed inside')
    assert tf32_settings() == TF32_EVERYWHERE


@pytest.mark.skipif(not NEW_PRECISION_API, reason='PyTorch before 2.9 has only the allow_tf32 flags')
def test_full_float32_precision_keeps_a_lowered_matmul_precision():
    # Reading or writing the allow_tf32 flags after this used to make PyTorch refuse to report the precision
    torch.set_float32_matmul_precision('medium')
    try:
        with devices.full_float32_precision():
            pass
        assert torch.get_float32_matmul_precision() == 'medium'
    finally:
        torch.set_float32_matmul_precision('highest')


def test_parrot_runs_networks_in_full_precision(tf32_allowed):
    task = Task('residues', 1)
    network = ready_network('cnn', task)
    seen = []
    network.register_forward_pre_hook(lambda *args: seen.append(tf32_settings()))
    data = pid.parse_file(data_path('res_regress_dataset.tsv'), task, ignore_warnings=True)[:20]
    dataset = pid.SequenceDataset(data)
    loader = pid.make_dataloader(dataset, task, batch_size=8)

    train(network, loader, loader, task, 'cpu', learning_rate=0.001, n_epochs=1, silent=True)
    evaluate_loss(network, loader, task, 'cpu')
    predict_labeled(network, dataset, task, 'cpu')
    predict_sequences(network, dataset.sequences, get_encoder('onehot'), 'cpu')

    assert seen and all(settings == FULL_PRECISION for settings in seen)
    assert tf32_settings() == TF32_EVERYWHERE


# ..........................................................................................
# The BRNN's LSTM

def run_lstm_packed(lstm: torch.nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference that run_lstm_unpacked is checked against: the LSTM run on a packed batch."""
    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, (h_n, _) = lstm(packed)
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
    return out, h_n


def padded_batch(lengths: list[int], input_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """A random zero-padded batch with the given sequence lengths."""
    x = torch.zeros(len(lengths), max(lengths), input_size)
    for i, length in enumerate(lengths):
        x[i, :length] = torch.randn(length, input_size)
    return x, torch.tensor(lengths)


def outputs_and_gradients(implementation, lstm: torch.nn.LSTM, x: torch.Tensor,
                          lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one LSTM implementation and return its outputs, final states and parameter gradients."""
    lstm.zero_grad()
    out, h_n = implementation(lstm, x, lengths)
    (out.sum() + h_n.sum()).backward()
    gradients = torch.cat([parameter.grad.flatten() for parameter in lstm.parameters()])
    return out.detach(), h_n.detach(), gradients


@pytest.mark.parametrize('num_layers', [1, 2, 3])
def test_unpacked_lstm_matches_packed(num_layers):
    lstm = torch.nn.LSTM(5, 4, num_layers, batch_first=True, bidirectional=True)
    x, lengths = padded_batch([7, 1, 3, 7, 5])

    packed = outputs_and_gradients(run_lstm_packed, lstm, x, lengths)
    unpacked = outputs_and_gradients(run_lstm_unpacked, lstm, x, lengths)

    for expected, found in zip(packed, unpacked):
        assert torch.allclose(expected, found, atol=FLOAT32_TOLERANCE)


def test_unpacked_lstm_zeroes_padding():
    lstm = torch.nn.LSTM(5, 4, 2, batch_first=True, bidirectional=True)
    x, lengths = padded_batch([6, 2])

    out, _ = run_lstm_unpacked(lstm, x, lengths)

    assert torch.all(out[1, 2:] == 0)


def test_unpacked_lstm_with_dropout_matches_packed_in_evaluation_mode():
    lstm = torch.nn.LSTM(5, 4, 2, batch_first=True, bidirectional=True, dropout=0.5).eval()
    x, lengths = padded_batch([6, 2, 4])

    out_packed, h_packed = run_lstm_packed(lstm, x, lengths)
    out_unpacked, h_unpacked = run_lstm_unpacked(lstm, x, lengths)

    assert torch.allclose(out_packed, out_unpacked, atol=FLOAT32_TOLERANCE)
    assert torch.allclose(h_packed, h_unpacked, atol=FLOAT32_TOLERANCE)


def test_unpacked_lstm_draws_no_random_numbers():
    lstm = torch.nn.LSTM(5, 4, 2, batch_first=True, bidirectional=True)
    x, lengths = padded_batch([6, 2])

    torch.manual_seed(0)
    run_lstm_unpacked(lstm, x, lengths)
    after_lstm = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(after_lstm, torch.rand(3))


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
def test_unpacked_lstm_matches_packed_on_gpu(device, full_float32_precision):
    lstm = torch.nn.LSTM(5, 4, 2, batch_first=True, bidirectional=True)
    x, lengths = padded_batch([7, 1, 3, 7, 5])

    on_cpu = outputs_and_gradients(run_lstm_packed, lstm, x, lengths)
    lstm.to(device)
    packed = outputs_and_gradients(run_lstm_packed, lstm, x.to(device), lengths)
    unpacked = outputs_and_gradients(run_lstm_unpacked, lstm, x.to(device), lengths)

    # cuDNN's LSTM kernels for unpacked tensors are accurate to a few parts per million rather than to
    # float32 precision (with cuDNN disabled, the unpacked LSTM on CUDA matches the CPU to ~1e-7), so
    # gradients summed over the batch can differ from the CPU's by a few times FLOAT32_TOLERANCE
    tolerance = 10 * FLOAT32_TOLERANCE if device == 'cuda' else FLOAT32_TOLERANCE
    for expected, from_packed, from_unpacked in zip(on_cpu, packed, unpacked):
        assert torch.allclose(expected, from_packed.cpu(), atol=tolerance)
        assert torch.allclose(expected, from_unpacked.cpu(), atol=tolerance)


def test_reverse_within_lengths():
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    lengths = torch.tensor([4, 2])

    reversed_x = reverse_within_lengths(x, lengths)

    assert reversed_x[..., 0].tolist() == [[3, 2, 1, 0], [5, 4, 6, 7]]
    assert torch.equal(reverse_within_lengths(reversed_x, lengths), x)


# ..........................................................................................
# Real runs on a GPU (CUDA or MPS)

def encoded_batch(encoder_name: str = 'onehot') -> tuple[torch.Tensor, torch.Tensor]:
    """The test fitting sequences, encoded and padded into one batch."""
    encoder = get_encoder(encoder_name)
    return pid.pad_sequence_vectors([encoder.encode(sequence) for sequence in FITTING_SEQUENCES])


@pytest.mark.real_devices
@pytest.mark.parametrize('device, description', [pytest.param('cuda', 'CUDA GPU', marks=requires_cuda),
                                                 pytest.param('mps', 'Apple-silicon GPU (MPS)', marks=requires_mps)])
def test_auto_trains_on_the_gpu(run_cli, tmp_path, capsys, device, description):
    filename, datatype, num_classes = EXAMPLE_TASKS['seq_regress']
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(filename), tmp_path / 'net.pt',
                                                datatype, num_classes, '-e', '1', '--device', 'auto'))

    assert f'Running on: {description}' in capsys.readouterr().out


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
@pytest.mark.parametrize('architecture', BUILT_IN_ARCHITECTURES)
@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_every_architecture_gives_the_same_output_on_gpu(architecture, datatype, device, full_float32_precision):
    if not supports(architecture, datatype):
        pytest.skip(f'{architecture} does not support {datatype} data')
    network = ready_network(architecture, Task(datatype, 3)).eval()
    x, lengths = encoded_batch()

    with torch.no_grad():
        on_cpu = network(x, lengths)
        on_gpu = network.to(device)(x.to(device), lengths).cpu()

    assert torch.allclose(on_cpu, on_gpu, atol=FLOAT32_TOLERANCE)


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
@pytest.mark.parametrize('architecture', NEURAL_ARCHITECTURES)
@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_gpu_predictions_match_the_cpu_with_pytorch_defaults(architecture, datatype, device, monkeypatch):
    # TF32 left as PyTorch sets it by default: PARROT's own prediction code must still compute in full precision
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)
    network = ready_network(architecture, Task(datatype, 3))
    encoder = get_encoder('onehot')

    on_cpu = predict_sequences(network, FITTING_SEQUENCES, encoder, 'cpu')
    on_gpu = predict_sequences(copy.deepcopy(network).to(device), FITTING_SEQUENCES, encoder, device)

    for sequence in FITTING_SEQUENCES:
        assert np.allclose(on_cpu[sequence], on_gpu[sequence], atol=FLOAT32_TOLERANCE)


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
def test_checkpoint_from_gpu_holds_cpu_tensors(tmp_path, device):
    network = make_network('residues', 1).to(device)
    path = tmp_path / 'net.pt'

    save_network(str(path), network, get_encoder('onehot'))

    saved = torch.load(path, weights_only=False)
    assert all(tensor.device.type == 'cpu' for tensor in saved['state_dict'].values())


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
@pytest.mark.parametrize('architecture', ['brnn', 'cnn'])
def test_train_on_gpu_then_predict_anywhere(run_cli, tmp_path, architecture, device):
    filename, datatype, num_classes = EXAMPLE_TASKS['res_regress']
    network_file = tmp_path / 'net.pt'
    run_cli(parrot_train.main, training_command('parrot-train', architecture, data_path(filename), network_file,
                                                datatype, num_classes, '-e', '2', '--device', device, '--seed', '1',
                                                '--silent'))

    # The saved network gives the same predictions on the CPU and on the GPU, from the CLI and from Predictor
    predictions = {}
    for prediction_device in ('cpu', device):
        output_file = tmp_path / f'predictions_{prediction_device}.txt'
        run_cli(parrot_predict.main, ['parrot-predict', data_path('seqfile.txt'), network_file, output_file,
                                      '--device', prediction_device, '--silent'])
        predictions[prediction_device] = [np.array(line.split()[2:], dtype=float)
                                          for line in output_file.read_text().splitlines()]

    for on_cpu, on_gpu in zip(predictions['cpu'], predictions[device]):
        assert np.allclose(on_cpu, on_gpu, atol=FLOAT32_TOLERANCE)

    sequence = FITTING_SEQUENCES[0]
    from_cpu = Predictor(network_file).predict(sequence)
    from_gpu = Predictor(network_file, device=device).predict(sequence)
    assert np.allclose(from_cpu, from_gpu, atol=FLOAT32_TOLERANCE)


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
@pytest.mark.parametrize('architecture', NEURAL_ARCHITECTURES)
def test_seeded_training_on_gpu_is_reproducible(run_cli, tmp_path, architecture, device):
    filename, datatype, num_classes = EXAMPLE_TASKS['res_class']
    outputs = []
    for run in range(2):
        network_file = tmp_path / f'net{run}.pt'
        run_cli(parrot_train.main, training_command('parrot-train', architecture, data_path(filename), network_file,
                                                    datatype, num_classes, '-e', '2', '--device', device, '--seed',
                                                    '7', '--silent'))
        outputs.append((tmp_path / f'net{run}_predictions.tsv').read_text())

    assert outputs[0] == outputs[1]


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
def test_optimize_on_gpu(run_cli, tmp_path, write_lines, device):
    lines = [' '.join(line) for line in pid.read_tsv_raw(data_path('res_regress_dataset.tsv'))[:60]]
    data_file = write_lines('subset.tsv', lines)

    run_cli(parrot_optimize.main, training_command('parrot-optimize', 'brnn', data_file, tmp_path / 'net.pt',
                                                   'residues', 1, '-e', '3', '--trials', '4', '--device', device,
                                                   '--seed', '1', '--silent'))

    assert Path(tmp_path / 'net_predictions.tsv').is_file()
    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert next(network.parameters()).device.type == 'cpu'
