"""
Tests for matrix data (parrot.matrices): the 'matrix' task, reading and checking
matrix files, padding matrices into batches, the masked loss, and unpadding
network outputs.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from parrot import prediction, train_network
from parrot import process_input_data as pid
from parrot.matrices import data as matrix_data
from parrot.parrot_exceptions import IOExceptionParrot
from parrot.tasks import IGNORED_CLASS_LABEL, Task
from parrot.tools import dataset_warnings

SQUARE = Task('matrix', 1)
SQUARE_CLASSES = Task('matrix', 3)
BINNED = Task('matrix', 1, width=5)

SEQUENCES = {'a': 'ACDE', 'b': 'FGH', 'c': 'KLMNPQ', 'd': 'RS'}


def write_matrix_data(directory: Path, matrices: dict[str, np.ndarray], sequences: dict[str, str] | None = None,
                      ) -> tuple[str, str]:
    """Write a '<ID> <sequence>' datafile and the matching .npz matrix file, returning their paths."""
    sequences = sequences if sequences is not None else {name: SEQUENCES[name] for name in matrices}
    datafile = directory / 'data.tsv'
    datafile.write_text(''.join(f'{name} {sequence}\n' for name, sequence in sequences.items()))
    matrix_file = directory / 'matrices.npz'
    np.savez_compressed(matrix_file, **matrices)
    return str(datafile), str(matrix_file)


def square_matrices(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A random square map for every sequence of SEQUENCES."""
    return {name: rng.random((len(sequence), len(sequence))) for name, sequence in SEQUENCES.items()}


def class_matrices(rng: np.random.Generator, width: int | None = None) -> dict[str, np.ndarray]:
    """Random class labels (0-2) for every sequence: square maps, or `width` columns."""
    return {name: rng.integers(0, 3, (len(sequence), width or len(sequence))).astype(float)
            for name, sequence in SEQUENCES.items()}


# ..........................................................................................
# The task

def test_matrix_task_width():
    assert SQUARE.is_matrix and SQUARE.is_square and SQUARE.width is None
    assert BINNED.is_matrix and not BINNED.is_square and BINNED.width == 5
    assert not Task('residues', 1).is_matrix and not Task('residues', 1).is_square


@pytest.mark.parametrize('arguments', [dict(datatype='residues', num_classes=1, width=4),
                                       dict(datatype='matrix', num_classes=1, width=0),
                                       dict(datatype='matrix', num_classes=1, width=2.5),
                                       dict(datatype='matrix', num_classes=1, width=True)])
def test_matrix_task_rejects_invalid_widths(arguments):
    with pytest.raises(ValueError):
        Task(**arguments)


@pytest.mark.parametrize('task', [SQUARE, SQUARE_CLASSES, BINNED, Task('sequence', 2), Task('residues', 1)])
def test_task_config_round_trip(task):
    assert Task.from_config(task.to_config()) == task


def test_sequence_and_residue_configs_are_unchanged():
    """Networks saved before matrix data existed have no 'width' in their task"""
    assert Task('residues', 3).to_config() == {'datatype': 'residues', 'num_classes': 3}


# ..........................................................................................
# Reading and checking

def test_parse_square_regression(tmp_path):
    matrices = square_matrices(np.random.default_rng(0))
    matrices['a'][0, 2] = np.nan
    datafile, matrix_file = write_matrix_data(tmp_path, matrices)

    data = pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=matrix_file)

    assert [sample[:2] for sample in data] == [[name, SEQUENCES[name]] for name in matrices]
    for (name, _, values) in data:
        np.testing.assert_array_equal(values, matrices[name])
        assert values.dtype == np.float64


def test_parse_matrices_with_a_fixed_width(tmp_path):
    rng = np.random.default_rng(1)
    matrices = {name: rng.random((len(sequence), 5)) for name, sequence in SEQUENCES.items()}
    datafile, matrix_file = write_matrix_data(tmp_path, matrices)

    data = pid.parse_file(datafile, BINNED, ignore_warnings=True, matrix_file=matrix_file)
    assert [sample[2].shape for sample in data] == [(len(sequence), 5) for sequence in SEQUENCES.values()]


def test_parse_classification_marks_missing_labels(tmp_path):
    matrices = class_matrices(np.random.default_rng(2))
    matrices['b'][1, 1] = np.nan
    datafile, matrix_file = write_matrix_data(tmp_path, matrices)

    data = pid.parse_file(datafile, SQUARE_CLASSES, ignore_warnings=True, matrix_file=matrix_file)
    labels = dict((name, values) for name, _, values in data)

    assert labels['b'].dtype == np.int64 and labels['b'][1, 1] == IGNORED_CLASS_LABEL
    np.testing.assert_array_equal(labels['a'], matrices['a'].astype(np.int64))


@pytest.mark.parametrize('change, message', [
    (lambda m: m.update(a=np.zeros((3, 4))), r'shape \(3, 4\), but should be \(4, 4\)'),
    (lambda m: m.update(a=np.zeros((4, 4, 1))), 'shape'),
    (lambda m: m.update(a=np.full((4, 4), np.inf)), 'infinite'),
    (lambda m: m.update(a=np.full((4, 4), np.nan)), 'Every value'),
    (lambda m: m.pop('b'), 'no matrix in the matrix file: b'),
])
def test_matrices_that_do_not_fit_their_sequences_are_rejected(tmp_path, change, message):
    matrices = square_matrices(np.random.default_rng(3))
    sequences = {name: SEQUENCES[name] for name in matrices}
    change(matrices)
    datafile, matrix_file = write_matrix_data(tmp_path, matrices, sequences)

    with pytest.raises(IOExceptionParrot, match=message):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=matrix_file)


def test_matrices_must_have_the_task_width(tmp_path):
    datafile, matrix_file = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(4)))
    with pytest.raises(IOExceptionParrot, match='5 columns'):
        pid.parse_file(datafile, BINNED, ignore_warnings=True, matrix_file=matrix_file)


@pytest.mark.parametrize('bad_label, message', [(1.5, 'not whole numbers'), (3.0, 'outside 0 to 2'),
                                                (-1.0, 'outside 0 to 2')])
def test_invalid_class_labels_are_rejected(tmp_path, bad_label, message):
    matrices = class_matrices(np.random.default_rng(5))
    matrices['c'][2, 3] = bad_label
    datafile, matrix_file = write_matrix_data(tmp_path, matrices)

    with pytest.raises(IOExceptionParrot, match=message):
        pid.parse_file(datafile, SQUARE_CLASSES, ignore_warnings=True, matrix_file=matrix_file)


def test_datafile_lines_must_be_an_id_and_a_sequence(tmp_path):
    datafile, matrix_file = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(6)))
    Path(datafile).write_text('a ACDE 1 2 3 4\n')
    with pytest.raises(IOExceptionParrot, match=r"should be '<ID> <sequence>'"):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=matrix_file)


def test_sequence_ids_must_be_unique(tmp_path):
    datafile, matrix_file = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(7)))
    Path(datafile).write_text('a ACDE\na ACDE\n')
    with pytest.raises(IOExceptionParrot, match='more than once: a'):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=matrix_file)


def test_matrix_file_is_needed_for_matrix_data_only(tmp_path):
    datafile, matrix_file = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(8)))
    with pytest.raises(ValueError, match='matrix file'):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True)
    with pytest.raises(ValueError, match='matrix file'):
        pid.parse_file(datafile, Task('sequence', 1), ignore_warnings=True, matrix_file=matrix_file)


def test_matrix_data_needs_sequence_ids(tmp_path):
    datafile, matrix_file = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(9)))
    with pytest.raises(ValueError, match='needs sequence IDs'):
        pid.parse_file(datafile, SQUARE, exclude_seq_id=True, ignore_warnings=True, matrix_file=matrix_file)


def test_unreadable_matrix_file(tmp_path):
    datafile, _ = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(10)))
    not_npz = tmp_path / 'matrices.txt'
    not_npz.write_text('not an archive')
    with pytest.raises(IOExceptionParrot, match='NumPy .npz'):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=str(not_npz))


def test_split_data_reads_matrix_data(tmp_path):
    rng = np.random.default_rng(11)
    sequences = {f's{i}': 'ACDEFGHIK'[:3 + i % 6] for i in range(40)}
    matrices = {name: rng.random((len(sequence), len(sequence))) for name, sequence in sequences.items()}
    datafile, matrix_file = write_matrix_data(tmp_path, matrices, sequences)

    train, val, test = pid.split_data(datafile, SQUARE, ignore_warnings=True, matrix_file=matrix_file)
    assert len(train) + len(val) + len(test) == 40
    assert all(sample[2].shape == (len(sample[1]), len(sample[1])) for sample in train.data)


# ..........................................................................................
# Batches, loss and outputs

def make_batch(task: Task, matrices: dict[str, np.ndarray]) -> pid.Batch:
    """Collate every sequence of SEQUENCES (with the given matrices) into one batch."""
    items = [(name, torch.ones(len(SEQUENCES[name]), 20), matrices[name]) for name in SEQUENCES]
    return pid.collate_batch(items, task)


def test_square_maps_are_padded_in_both_dimensions():
    matrices = square_matrices(np.random.default_rng(12))
    batch = make_batch(SQUARE, matrices)
    padded_length = batch.vectors.shape[1]

    assert batch.targets.shape == (4, padded_length, padded_length, 1)
    np.testing.assert_allclose(batch.targets[2, :6, :6, 0].numpy(), matrices['c'], rtol=1e-6)
    # Everything outside a sequence's own map is padding, marked as missing
    assert torch.isnan(batch.targets[0, 4:]).all() and torch.isnan(batch.targets[0, :, 4:]).all()


def test_fixed_width_matrices_are_padded_in_rows_only():
    rng = np.random.default_rng(13)
    matrices = {name: rng.integers(0, 3, (len(sequence), 5)).astype(np.int64) for name, sequence in SEQUENCES.items()}
    task = Task('matrix', 3, width=5)
    batch = make_batch(task, matrices)

    assert batch.targets.shape == (4, batch.vectors.shape[1], 5) and batch.targets.dtype == torch.int64
    assert (batch.targets[3, 2:] == IGNORED_CLASS_LABEL).all()
    np.testing.assert_array_equal(batch.targets[3, :2].numpy(), matrices['d'])


class ConstantOutput(torch.nn.Module):
    """A stand-in network whose output is zero everywhere, with the shape a matrix network gives."""

    def __init__(self, task: Task) -> None:
        super().__init__()
        self.task = task

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        width = x.shape[1] if self.task.width is None else self.task.width
        return torch.zeros(x.shape[0], x.shape[1], width, self.task.num_classes)


def test_regression_loss_counts_only_real_labelled_cells():
    matrices = square_matrices(np.random.default_rng(14))
    matrices['a'][1, 2] = np.nan
    batch = make_batch(SQUARE, matrices)

    loss, _ = train_network.compute_batch_loss(ConstantOutput(SQUARE), batch, SQUARE,
                                               train_network.build_loss_criterion(SQUARE), 'cpu')

    # Predicting 0 everywhere, the summed squared error is the sum of the squares of the real, labelled values
    expected = sum(np.nansum(matrix ** 2) for matrix in matrices.values())
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_classification_loss_counts_only_real_labelled_cells():
    matrices = class_matrices(np.random.default_rng(15))
    matrices['c'][0, 0] = np.nan
    labels = {name: matrix_data.matrix_class_labels(name, matrix, SQUARE_CLASSES) for name, matrix in matrices.items()}
    batch = make_batch(SQUARE_CLASSES, labels)

    loss, _ = train_network.compute_batch_loss(ConstantOutput(SQUARE_CLASSES), batch, SQUARE_CLASSES,
                                               train_network.build_loss_criterion(SQUARE_CLASSES), 'cpu')

    # Equal logits for every class cost log(3) per labelled cell
    labelled_cells = sum(np.sum(~np.isnan(matrix)) for matrix in matrices.values())
    assert loss.item() == pytest.approx(labelled_cells * math.log(3), rel=1e-5)


@pytest.mark.parametrize('task, width', [(SQUARE_CLASSES, None), (Task('matrix', 3, width=5), 5)])
def test_outputs_are_unpadded_to_each_sequence(task, width):
    lengths = torch.tensor([4, 3, 6])
    outputs = torch.zeros(3, 8, width or 8, 3)
    unpadded = prediction.unpad_outputs(outputs, lengths, task)
    assert [array.shape for array in unpadded] == [(n, width or n, 3) for n in (4, 3, 6)]


# ..........................................................................................
# Dataset warnings

def test_warnings_report_missing_matrix_values(capsys):
    matrices = square_matrices(np.random.default_rng(16))
    matrices['d'][0, 1] = np.nan
    data = [[name, SEQUENCES[name], matrix] for name, matrix in matrices.items()]

    dataset_warnings.warn_about_dataset(data, SQUARE)
    output = capsys.readouterr().out
    assert 'Some matrix values are missing' in output and '1 of 65 matrix values' in output


def test_class_imbalance_is_checked_over_every_matrix_value(capsys):
    data = [[name, sequence, np.zeros((len(sequence), len(sequence)))] for name, sequence in SEQUENCES.items()]
    data[0][2][0, 0] = 1.0
    dataset_warnings.warn_about_dataset(data, SQUARE_CLASSES)
    assert 'Severe class imbalance' in capsys.readouterr().out
