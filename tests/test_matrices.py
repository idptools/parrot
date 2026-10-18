"""
Tests for matrix data (parrot.matrices): the 'matrix' task, reading and checking
matrix files, padding matrices into batches, the masked loss, unpadding network
outputs, metrics, and every command with matrix data.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from parrot import figures, metrics, prediction, train_network
from parrot import process_input_data as pid
from parrot.checkpoint import load_network
from parrot.matrices import data as matrix_data
from parrot.matrices.output import name_matrices, write_matrix_predictions
from parrot.parrot_exceptions import IOExceptionParrot
from parrot.py_predictor import Predictor
from parrot.scripts import parrot_optimize, parrot_predict, parrot_preprocess, parrot_train
from parrot.tasks import IGNORED_CLASS_LABEL, Task
from parrot.tools import dataset_warnings

from tests.helpers import training_command

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


def test_a_matrix_file_of_object_arrays_is_refused_with_a_file_error(tmp_path):
    # Arrays of Python objects need unpickling, which PARROT never does; numpy's own ValueError named no file
    datafile, _ = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(10)))
    object_file = tmp_path / 'objects.npz'
    np.savez(object_file, a=np.array([np.zeros(2), np.zeros(3)], dtype=object))
    with pytest.raises(IOExceptionParrot, match='Cannot read the matrices in the matrix file .*objects.npz'):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=str(object_file))


def test_a_single_array_is_not_a_matrix_file(tmp_path):
    datafile, _ = write_matrix_data(tmp_path, square_matrices(np.random.default_rng(10)))
    npy_file = tmp_path / 'matrices.npy'
    np.save(npy_file, np.zeros((3, 3)))
    with pytest.raises(IOExceptionParrot, match='holds a single array, not a .npz archive'):
        pid.parse_file(datafile, SQUARE, ignore_warnings=True, matrix_file=str(npy_file))


def test_complex_matrices_are_refused():
    # Their imaginary parts would otherwise be silently dropped
    with pytest.raises(IOExceptionParrot, match='does not hold real numbers'):
        matrix_data.check_matrix('a', 'AC', np.array([[1 + 2j, 2], [3, 4j]]), SQUARE)


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


# ..........................................................................................
# Metrics

def test_metrics_count_each_pair_of_a_symmetric_map_once():
    """Only the upper triangle (and diagonal) of a symmetric map is scored, as in training"""
    true = [np.array([[1.0, 2.0], [2.0, 3.0]])]
    predicted = [np.array([[1.0, 2.0], [-50.0, 3.0]])]   # the lower triangle is never looked at
    task = Task('matrix', 1, symmetric=True)
    true_flat, predicted_flat = metrics.labelled_values(true, predicted, task)
    np.testing.assert_array_equal(true_flat, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(predicted_flat, [1.0, 2.0, 3.0])


def test_metrics_pool_matrices_of_different_sizes():
    true = [np.eye(3), np.eye(5)]
    scores = metrics.performance_metrics(true, [2 * matrix + 1 for matrix in true], Task('matrix', 1))
    assert scores['Pearson R'] == pytest.approx(1.0)


# ..........................................................................................
# Commands

def write_command_data(directory: Path, kind: str = 'square', num_sequences: int = 40) -> tuple[str, str]:
    """A dataset for the commands: symmetric square maps ('square'), class maps ('classes') or 5 columns ('width5')."""
    rng = np.random.default_rng(20)
    sequences, matrices = {}, {}
    for index in range(num_sequences):
        length = int(rng.integers(4, 16))
        sequence = ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=length))
        separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
        if kind == 'square':
            matrix = np.exp(-separation / 3.0)
        elif kind == 'classes':
            matrix = (separation <= 2).astype(float)
        else:
            matrix = np.tile(np.linspace(0, 1, 5), (length, 1))
        sequences[f'id{index}'], matrices[f'id{index}'] = sequence, matrix
    return write_matrix_data(directory, matrices, sequences)


def train_matrix(run_cli, datafile, matrix_file, network, num_classes, *arguments):
    """Run parrot-train unet2d on matrix data for 2 epochs on the CPU."""
    run_cli(parrot_train.main, training_command('parrot-train', 'unet2d', datafile, network, None, num_classes,
                                                '--matrix-file', matrix_file, '-e', 2, '--device', 'cpu', '--silent',
                                                *arguments))


def test_train_square_maps_writes_every_output(run_cli, tmp_path):
    datafile, matrix_file = write_command_data(tmp_path)
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'net.pt', 1, '--symmetric', '--include-figs')

    network, _ = load_network(str(tmp_path / 'net.pt'))
    assert network.name == 'unet2d' and network.task == Task('matrix', 1, symmetric=True)
    for suffix in ('_matrix_scatterplot.png', '_matrix_examples.png', '_train_val_loss.png',
                   '_performance_stats.txt'):
        assert (tmp_path / f'net{suffix}').is_file(), suffix

    predictions = np.load(tmp_path / 'net_predictions.npz')
    sequences = dict(line.split() for line in Path(datafile).read_text().splitlines())
    assert predictions.files and all(predictions[name].shape == (len(sequences[name]),) * 2
                                     for name in predictions.files)


def test_train_classes_and_fixed_width(run_cli, tmp_path):
    datafile, matrix_file = write_command_data(tmp_path, 'classes')
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'classes.pt', 2, '--include-figs')
    assert (tmp_path / 'classes_matrix_CM.png').is_file()

    datafile, matrix_file = write_command_data(tmp_path, 'width5')
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'width5.pt', 1, '--matrix-width', 5)
    network, _ = load_network(str(tmp_path / 'width5.pt'))
    assert network.task == Task('matrix', 1, width=5)


@pytest.mark.parametrize('arguments, message', [
    ([], 'needs --matrix-file'),
    (['--matrix-file', 'MATRICES', '--symmetric', '--matrix-width', 5], 'Only square matrix tasks'),
    (['--matrix-file', 'MATRICES', '--max-matrix-length', 10], 'longer than --max-matrix-length'),
    (['--matrix-file', 'MATRICES', '--exclude-seq-id'], 'needs sequence IDs'),
])
def test_train_rejects_invalid_matrix_options(run_cli, tmp_path, arguments, message):
    datafile, matrix_file = write_command_data(tmp_path)
    options = [matrix_file if argument == 'MATRICES' else argument for argument in arguments]
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_train.main, training_command('parrot-train', 'unet2d', datafile, tmp_path / 'net.pt', None, 1,
                                                    '-e', 1, '--device', 'cpu', '--silent', *options))


@pytest.mark.parametrize('architecture, datatype, options', [
    ('brnn', 'sequence', ['--matrix-file', 'MATRICES']),   # only matrix architectures read matrices
    ('brnn', 'matrix', []),                                # 1D networks cannot learn matrices
    ('unet2d', 'sequence', []),                            # unet2d only learns matrices
    ('unet2d', None, ['--probabilistic-classification']),  # class probabilities are for sequences
])
def test_train_offers_matrix_options_only_to_matrix_architectures(run_cli, tmp_path, architecture, datatype, options):
    datafile, matrix_file = write_command_data(tmp_path)
    options = [matrix_file if option == 'MATRICES' else option for option in options]
    with pytest.raises(SystemExit):
        run_cli(parrot_train.main, training_command('parrot-train', architecture, datafile, tmp_path / 'net.pt',
                                                    datatype, 1, *options))


def test_optimize_matrix_data(run_cli, tmp_path):
    datafile, matrix_file = write_command_data(tmp_path)
    run_cli(parrot_optimize.main, training_command('parrot-optimize', 'unet2d', datafile, tmp_path / 'opt.pt', None, 1,
                                                   '--matrix-file', matrix_file, '-e', 2, '--trials', 2, '--device',
                                                   'cpu', '--silent'))
    assert 'architecture: unet2d' in (tmp_path / 'opt_optimal_hyperparams.txt').read_text()
    assert (tmp_path / 'opt_predictions.npz').is_file()


@pytest.mark.parametrize('exclude_ids', [False, True])
def test_predict_writes_one_matrix_per_sequence(run_cli, tmp_path, exclude_ids):
    datafile, matrix_file = write_command_data(tmp_path)
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'net.pt', 1)
    lines = Path(datafile).read_text().splitlines()[:3]
    sequence_file = tmp_path / 'new.txt'
    sequence_file.write_text('\n'.join(line.split()[1] if exclude_ids else line for line in lines) + '\n')

    run_cli(parrot_predict.main, ['parrot-predict', sequence_file, tmp_path / 'net.pt', tmp_path / 'out',
                                  '--device', 'cpu', '--silent', *(['--exclude-seq-id'] if exclude_ids else [])])

    predictions = np.load(tmp_path / 'out.npz')   # '.npz' is added to the output name
    names = ['1', '2', '3'] if exclude_ids else [line.split()[0] for line in lines]
    assert predictions.files == names
    assert [predictions[name].shape for name in names] == [(len(line.split()[1]),) * 2 for line in lines]


def test_predict_refuses_repeated_matrix_ids_before_predicting(run_cli, tmp_path, monkeypatch):
    datafile, matrix_file = write_command_data(tmp_path)
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'net.pt', 1)
    sequence_file = tmp_path / 'new.txt'
    sequence_file.write_text('a ACDEFG\nb KLMNPQ\na RSTVWY\n')
    monkeypatch.setattr(prediction, 'predict_sequences', lambda *args, **kwargs: pytest.fail('predicted anyway'))

    with pytest.raises(ValueError, match='repeated: a'):
        run_cli(parrot_predict.main, ['parrot-predict', sequence_file, tmp_path / 'net.pt', tmp_path / 'out',
                                      '--device', 'cpu', '--silent'])


def test_predictor_returns_a_matrix(run_cli, tmp_path):
    datafile, matrix_file = write_command_data(tmp_path)
    train_matrix(run_cli, datafile, matrix_file, tmp_path / 'net.pt', 1, '--symmetric')
    predictor = Predictor(tmp_path / 'net.pt')
    predicted = predictor.predict('ACDEFG')
    assert predicted.shape == (6, 6) and np.allclose(predicted, predicted.T)

    # predict_batch() gives each sequence's matrix (up to float32 rounding) when batched with other lengths
    batch = predictor.predict_batch(['ACDEFG', 'KLMNPQRSTV', 'WY'], batch_size=2)
    assert [matrix.shape for matrix in batch] == [(6, 6), (10, 10), (2, 2)]
    np.testing.assert_allclose(batch[0], predicted, rtol=1e-5, atol=1e-6)


def test_preprocess_splits_matrix_data(run_cli, tmp_path):
    datafile, matrix_file = write_command_data(tmp_path)
    output = tmp_path / 'processed.tsv'
    run_cli(parrot_preprocess.main, ['parrot-preprocess', datafile, output, '-d', 'matrix', '-m', 'regression',
                                     '--split', 'random', '--max-length', 12, '--matrix-file', matrix_file,
                                     '--seed', 1])

    report = (tmp_path / 'processed_preprocess_report.txt').read_text()
    assert 'Checked: every sequence has a matrix' in report
    assert all(len(line.split()) == 2 and len(line.split()[1]) <= 12 for line in output.read_text().splitlines())
    # The original matrix file works with the processed datafile and its split-file
    train_matrix(run_cli, str(output), matrix_file, tmp_path / 'net.pt', 1, '--split',
                 tmp_path / 'processed_split_file.txt')


@pytest.mark.parametrize('arguments, message', [
    (['-d', 'matrix', '-m', 'classification', '--class-balance'], 'does not apply to matrix data'),
    (['-d', 'matrix', '-m', 'classification', '--recode-classes'], 'does not apply to matrix data'),
    (['-d', 'matrix', '-m', 'regression', '--deduplicate', 'merge'], 'does not apply to matrix data'),
    (['-d', 'sequence', '-m', 'regression', '--matrix-width', 4], 'only applies to matrix data'),
])
def test_preprocess_rejects_options_that_do_not_suit_matrix_data(run_cli, tmp_path, arguments, message):
    datafile, _ = write_command_data(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', datafile, tmp_path / 'out.tsv', *arguments])


def test_preprocess_matrix_lines_must_be_an_id_and_a_sequence(run_cli, tmp_path):
    datafile = tmp_path / 'data.tsv'
    datafile.write_text('a ACD 1 2 3\n')
    with pytest.raises(ValueError, match="'<ID> <sequence>'"):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', datafile, tmp_path / 'out.tsv', '-d', 'matrix',
                                         '-m', 'regression'])


def test_preprocess_checks_matrix_classes_against_the_matrices(run_cli, tmp_path):
    """Three-class maps pass the output check (the class count comes from the matrices)."""
    datafile, matrix_file = write_command_data(tmp_path, 'classes')
    matrices = {name: 2 * matrix for name, matrix in np.load(matrix_file).items()}   # classes 0 and 2
    np.savez(matrix_file, **matrices)
    run_cli(parrot_preprocess.main, ['parrot-preprocess', datafile, tmp_path / 'out.tsv', '-d', 'matrix',
                                     '-m', 'classification', '--matrix-file', matrix_file])
    assert 'Checked: every sequence has a matrix' in (tmp_path / 'out_preprocess_report.txt').read_text()


def test_predicted_matrices_keep_any_sequence_id(tmp_path):
    """IDs that are also numpy.savez's argument names, or hold '|' and '/', are written and read back unchanged."""
    matrices = {'file': np.eye(2), 'allow_pickle': np.ones((3, 3), np.float32), 'sp|P1|X/1-5': np.zeros((2, 4, 3))}
    path = write_matrix_predictions(str(tmp_path / 'predictions'), matrices)

    assert path.endswith('predictions.npz')
    loaded = np.load(path)
    assert loaded.files == list(matrices)
    for name, matrix in matrices.items():
        np.testing.assert_array_equal(loaded[name], matrix)
        assert loaded[name].dtype == matrix.dtype


def test_predicted_matrices_need_unique_ids():
    with pytest.raises(ValueError, match='repeated: a'):
        name_matrices(['a', 'b', 'a'], [np.eye(2)] * 3)
    with pytest.raises(ValueError, match='not empty'):
        name_matrices(['', 'b'], [np.eye(2)] * 2)


def test_large_matrix_scatterplots_show_a_warned_sample(monkeypatch):
    monkeypatch.setattr(figures, 'MAX_SCATTER_POINTS', 10)
    true_values, predicted_values = np.arange(25.0), np.arange(25.0) + 1
    with pytest.warns(UserWarning, match='shows 10 of the 25 test values'):
        true_shown, predicted_shown = figures.scatter_sample(true_values, predicted_values)
    assert len(true_shown) == 10
    np.testing.assert_array_equal(predicted_shown, true_shown + 1)   # pairs stay together
