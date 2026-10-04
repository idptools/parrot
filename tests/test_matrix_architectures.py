"""
Tests that every matrix architecture (parrot.matrices) must pass, plus tests of
the 2D U-Net and the shared 2D building blocks.

The architectures are read from the registry, so a matrix architecture added
later is covered automatically: output shapes for square and fixed-width maps,
for regression and classification; the same output for a sequence however much
padding it is given; symmetric maps for symmetric tasks; gradients through
padded batches; saving and loading; training; and (on machines with a GPU)
the same output on the GPU as on the CPU.
"""

import numpy as np
import pytest
import torch

from parrot import prediction, train_network
from parrot import process_input_data as pid
from parrot.architectures import build_network, get_architecture
from parrot.checkpoint import load_network, save_network
from parrot.devices import full_float32_precision
from parrot.encode_sequence import get_encoder
from parrot.matrices.layers import ChannelLayerNorm2d, cell_mask, separation_feature, symmetrise
from parrot.tasks import Task

from tests.helpers import MATRIX_ARCHITECTURES, encode_batch

# Sequences of very different lengths, including ones shorter than the U-Net's downsampling factor
SEQUENCES = ['ACDEFGHIK', 'LMN', 'PQRSTVWYACDEFGHIKLMNPQ', 'KK', 'W', 'ACDEFGHIKLMNPQRSTVWYAC']

MATRIX_TASKS = [Task('matrix', 1), Task('matrix', 3), Task('matrix', 1, width=5), Task('matrix', 3, width=5),
                Task('matrix', 1, symmetric=True)]
TASK_IDS = ['square-regression', 'square-classes', 'width5-regression', 'width5-classes', 'symmetric']

# Every architecture with its defaults, plus settings that exercise other code paths (dropout stays 0, so
# outputs are deterministic)
CONFIGURATIONS = [(architecture, {}) for architecture in MATRIX_ARCHITECTURES] + [
    ('unet2d', {'pair_features': 'outer', 'pair_channels': 4}),
    ('unet2d', {'upsampling': 'nearest', 'depth': 1}),
]
CONFIGURATION_IDS = [f"{architecture}-{'-'.join(f'{k}={v}' for k, v in values.items()) or 'defaults'}"
                     for architecture, values in CONFIGURATIONS]

# float32 results from differently padded or ordered computations agree to about this
FLOAT32_TOLERANCE = 1e-5

GPUS = [pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA GPU')),
        pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='no MPS GPU'))]


def matrix_width(task: Task, length: int) -> int:
    """The number of columns of a sequence's matrix."""
    return length if task.width is None else task.width


def build(architecture: str, task: Task, values: dict | None = None) -> torch.nn.Module:
    """A matrix network for one-hot encoded sequences, in evaluation mode."""
    return build_network(architecture, 20, task, values).eval()


# ..........................................................................................
# Behaviour every matrix architecture must have

def test_matrix_architectures_are_registered():
    assert 'unet2d' in MATRIX_ARCHITECTURES
    assert all(get_architecture(name).supported_datatypes == ('matrix',) for name in MATRIX_ARCHITECTURES)


@pytest.mark.parametrize('architecture, values', CONFIGURATIONS, ids=CONFIGURATION_IDS)
@pytest.mark.parametrize('task', MATRIX_TASKS, ids=TASK_IDS)
def test_output_shapes(architecture, values, task):
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad():
        output = build(architecture, task, values)(batch, lengths)
    padded_length = batch.shape[1]
    assert output.shape == (len(SEQUENCES), padded_length, matrix_width(task, padded_length), task.num_classes)


@pytest.mark.parametrize('architecture, values', CONFIGURATIONS, ids=CONFIGURATION_IDS)
@pytest.mark.parametrize('task', MATRIX_TASKS, ids=TASK_IDS)
def test_padding_invariance(architecture, values, task):
    """A sequence's matrix must be the same on its own as in a batch padded to a longer sequence"""
    network = build(architecture, task, values)
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad():
        batched = network(batch, lengths)
        for index, sequence in enumerate(SEQUENCES):
            alone, alone_lengths = encode_batch([sequence])
            alone_output = network(alone, alone_lengths)
            length, width = len(sequence), matrix_width(task, len(sequence))
            torch.testing.assert_close(batched[index, :length, :width], alone_output[0, :length, :width],
                                       atol=FLOAT32_TOLERANCE, rtol=FLOAT32_TOLERANCE)


@pytest.mark.parametrize('architecture, values', CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_symmetric_tasks_give_symmetric_maps(architecture, values):
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad():
        output = build(architecture, Task('matrix', 1, symmetric=True), values)(batch, lengths)
    torch.testing.assert_close(output, output.transpose(1, 2))


@pytest.mark.parametrize('architecture, values', CONFIGURATIONS, ids=CONFIGURATION_IDS)
@pytest.mark.parametrize('task', MATRIX_TASKS, ids=TASK_IDS)
def test_gradients_flow_through_padded_batches(architecture, values, task):
    network = build_network(architecture, 20, task, values).train()
    batch, lengths = encode_batch(SEQUENCES)
    output = network(batch, lengths)

    # A loss over the real cells only, as in training
    real = cell_mask(lengths, batch.shape[1], lengths if task.width is None else torch.full_like(lengths, task.width),
                     output.shape[2])
    output[real].sum().backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize('architecture', MATRIX_ARCHITECTURES)
@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_matrix_architectures_reject_other_data(architecture, datatype):
    with pytest.raises(ValueError, match='only supports matrix-mapped data'):
        build_network(architecture, 20, Task(datatype, 1))


@pytest.mark.parametrize('architecture', MATRIX_ARCHITECTURES)
@pytest.mark.parametrize('task', MATRIX_TASKS, ids=TASK_IDS)
def test_save_and_load(tmp_path, architecture, task):
    """A saved matrix network (with its width and symmetry) loads and predicts exactly as before"""
    network = build(architecture, task)
    path = str(tmp_path / 'net.pt')
    save_network(path, network, get_encoder('onehot'))
    loaded, encoder = load_network(path)

    assert loaded.task == task and loaded.name == architecture
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad():
        torch.testing.assert_close(loaded.eval()(batch, lengths), network(batch, lengths))


@pytest.mark.parametrize('architecture', MATRIX_ARCHITECTURES)
def test_predictions_are_one_matrix_per_sequence(architecture):
    task = Task('matrix', 3, width=5)
    outputs = prediction.predict_sequences(build(architecture, task), SEQUENCES, get_encoder('onehot'), 'cpu',
                                           batch_size=2)
    assert [outputs[sequence].shape for sequence in SEQUENCES] == [(len(sequence), 5, 3) for sequence in SEQUENCES]


def toy_dataset(task: Task, num_sequences: int = 40, seed: int = 0) -> pid.SequenceDataset:
    """Random sequences whose map says which pairs of residues are within 2 of each other (|i - j| <= 2)."""
    rng = np.random.default_rng(seed)
    data = []
    for index in range(num_sequences):
        length = int(rng.integers(4, 13))
        sequence = ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=length))
        positions = np.arange(length)
        near = (np.abs(positions[:, None] - positions[None, :]) <= 2).astype(float)
        data.append([f's{index}', sequence, near.astype(np.int64) if task.is_classification else near])
    return pid.SequenceDataset(data)


@pytest.mark.parametrize('architecture', MATRIX_ARCHITECTURES)
@pytest.mark.parametrize('task', [Task('matrix', 1, symmetric=True), Task('matrix', 2)], ids=['regression', 'classes'])
def test_training_learns_a_simple_map(architecture, task):
    """The map depends only on how far apart two residues are, which the input grid gives the network"""
    torch.manual_seed(0)
    network = build_network(architecture, 20, task, {'base_channels': 4, 'depth': 1, 'pair_channels': 4})
    loader = pid.make_dataloader(toy_dataset(task), task, batch_size=8, shuffle=True)
    train_losses, _ = train_network.train(network, loader, loader, task, 'cpu', learning_rate=0.01, n_epochs=15,
                                          silent=True)
    assert train_losses[-1] < 0.5 * train_losses[0]


@pytest.mark.real_devices
@pytest.mark.parametrize('device', GPUS)
@pytest.mark.parametrize('architecture', MATRIX_ARCHITECTURES)
@pytest.mark.parametrize('task', MATRIX_TASKS, ids=TASK_IDS)
def test_same_output_on_gpu(architecture, task, device):
    network = build(architecture, task)
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad(), full_float32_precision():
        on_cpu = network(batch, lengths)
        on_gpu = network.to(device)(batch.to(device), lengths).cpu()
    torch.testing.assert_close(on_cpu, on_gpu, atol=FLOAT32_TOLERANCE, rtol=FLOAT32_TOLERANCE)


# ..........................................................................................
# Shared 2D building blocks

def test_cell_mask_marks_cells_in_real_rows_and_columns():
    mask = cell_mask(torch.tensor([2, 3]), 3, torch.tensor([3, 1]), 4)
    assert mask.shape == (2, 3, 4)
    assert mask[0].tolist() == [[True, True, True, False], [True, True, True, False], [False] * 4]
    assert mask[1].sum() == 3 and mask[1, :, 0].all()


def test_separation_feature():
    separation = separation_feature(4, torch.device('cpu'), torch.float32)
    torch.testing.assert_close(separation, torch.log1p(torch.tensor([[0., 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1],
                                                                     [3, 2, 1, 0]])))


def test_symmetrise():
    outputs = torch.rand(2, 5, 5, 3)
    symmetric = symmetrise(outputs)
    torch.testing.assert_close(symmetric, symmetric.transpose(1, 2))
    torch.testing.assert_close(symmetrise(symmetric), symmetric)


def test_channel_layer_norm_2d_normalises_each_cell():
    norm = ChannelLayerNorm2d(4)
    x = torch.rand(2, 4, 3, 5) * 10
    normalised = norm(x)
    torch.testing.assert_close(normalised.mean(dim=1), torch.zeros(2, 3, 5), atol=1e-5, rtol=0)
    # A cell's result does not depend on any other cell
    changed = x.clone()
    changed[:, :, 0, 0] += 100
    torch.testing.assert_close(norm(changed)[:, :, 1:], normalised[:, :, 1:])
