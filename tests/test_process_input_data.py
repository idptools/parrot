"""
Tests for parrot.process_input_data: reading datafiles, validating values, building
datasets, collating batches and splitting data into train/validation/test sets.
"""

from typing import Any

import numpy as np
import pytest
import torch

from parrot import encode_sequence
from parrot import process_input_data as pid
from parrot.parrot_exceptions import IOExceptionParrot
from parrot.tasks import Task

from tests.helpers import data_path

# Module-level double-underscore functions are not name-mangled, but fetching it
# explicitly makes it obvious this is a private helper being tested
parse_lines = getattr(pid, '__parse_lines')


# ..........................................................................................
# Reading and parsing lines

def test_read_tsv_raw_skips_blank_and_comment_lines(write_lines):
    path = write_lines('data.tsv', ['# a comment', 'id1 ACD 1.0', '', '   ', '  # indented comment', 'id2 EFG 2.0'])
    assert pid.read_tsv_raw(path) == [['id1', 'ACD', '1.0'], ['id2', 'EFG', '2.0']]


def test_read_tsv_raw_custom_delimiter(write_lines):
    path = write_lines('data.csv', ['id1,ACD,1.0', 'id2,EFG,2.0'])
    assert pid.read_tsv_raw(path, delimiter=',') == [['id1', 'ACD', '1.0'], ['id2', 'EFG', '2.0']]


def test_parse_lines_sequence():
    data = parse_lines([['id1', 'ACD', '1.5'], ['id2', 'EF', '-2']], datatype='sequence')
    assert data == [['id1', 'ACD', 1.5], ['id2', 'EF', -2.0]]


def test_parse_lines_residues():
    data = parse_lines([['id1', 'ACD', '1', '2', '3.5']], datatype='residues')
    assert data[0][:2] == ['id1', 'ACD']
    np.testing.assert_array_equal(data[0][2], [1.0, 2.0, 3.5])


def test_parse_lines_invalid_datatype():
    with pytest.raises(ValueError):
        parse_lines([['id1', 'ACD', '1']], datatype='sequences')


def test_parse_lines_residue_count_mismatch():
    with pytest.raises(IOExceptionParrot, match='Inconsistent number'):
        parse_lines([['id1', 'ACD', '1', '2']], datatype='residues')


def test_parse_lines_residue_count_mismatch_not_validated():
    data = parse_lines([['id1', 'ACD', '1', '2']], datatype='residues', validate=False)
    assert len(data[0][2]) == 2


def test_parse_lines_sequence_rejects_extra_columns():
    """Residue-formatted data passed as sequence data must not be silently truncated"""
    with pytest.raises(IOExceptionParrot):
        parse_lines([['id1', 'ACD', '1', '2', '3']], datatype='sequence')


@pytest.mark.parametrize('bad_value', ['nan', 'inf', 'abc'])
def test_parse_lines_sequence_rejects_non_finite(bad_value):
    with pytest.raises(IOExceptionParrot):
        parse_lines([['id1', 'ACD', bad_value]], datatype='sequence')


@pytest.mark.parametrize('bad_value', ['inf', '-inf', 'abc'])
def test_parse_lines_residues_rejects_invalid_values(bad_value):
    with pytest.raises(IOExceptionParrot):
        parse_lines([['id1', 'ACD', '1', bad_value, '3']], datatype='residues')


@pytest.mark.parametrize('missing', ['nan', 'NaN', 'NAN'])
def test_parse_lines_residues_accepts_missing_labels(missing):
    data = parse_lines([['id1', 'ACD', '1', missing, '3']], datatype='residues')
    assert data[0][2][0] == 1.0 and np.isnan(data[0][2][1]) and data[0][2][2] == 3.0


def test_parse_lines_residues_rejects_all_missing():
    with pytest.raises(IOExceptionParrot):
        parse_lines([['id1', 'ACD', 'nan', 'nan', 'nan']], datatype='residues')


def test_missing_label_text_is_parsed_as_nan():
    assert np.isnan(float(pid.MISSING_LABEL_TEXT))


# ..........................................................................................
# parse_file: full datafile parsing and class-label validation

def test_parse_file_sequence_classification():
    data = pid.parse_file(data_path('seq_class_dataset.tsv'), Task('sequence', 3))

    assert len(data) == 300
    assert all(isinstance(sample[2], int) for sample in data)
    assert {sample[2] for sample in data} == {0, 1, 2}


def test_parse_file_residue_classification():
    data = pid.parse_file(data_path('res_class_dataset.tsv'), Task('residues', 3))

    assert len(data) == 300
    for sample in data:
        assert isinstance(sample[2], list)
        assert len(sample[2]) == len(sample[1])
        assert set(sample[2]) <= {0, 1, 2}


def test_parse_file_regression():
    data = pid.parse_file(data_path('seq_regress_dataset.tsv'), Task('sequence', 1))
    assert all(isinstance(sample[2], float) for sample in data)

    data = pid.parse_file(data_path('res_regress_dataset.tsv'), Task('residues', 1))
    assert all(len(sample[2]) == len(sample[1]) for sample in data)


def test_parse_file_exclude_seq_id(write_lines):
    path = write_lines('noids.tsv', ['ACD 1', 'EFG 0'])
    data = pid.parse_file(path, Task('sequence', 2), exclude_seq_id=True)
    assert data == [['', 'ACD', 1], ['', 'EFG', 0]]


@pytest.mark.parametrize('label', ['3', '-1'])
def test_parse_file_sequence_class_out_of_range(write_lines, label):
    path = write_lines('data.tsv', ['id1 ACD 0', f'id2 EFG {label}'])
    with pytest.raises(ValueError, match='between 0 and 2'):
        pid.parse_file(path, Task('sequence', 3))


def test_parse_file_sequence_class_must_be_integer(write_lines):
    """A label of 1.5 must not be silently truncated to class 1"""
    path = write_lines('data.tsv', ['id1 ACD 0', 'id2 EFG 1.5'])
    with pytest.raises(ValueError, match='must be integers'):
        pid.parse_file(path, Task('sequence', 3))


def test_parse_file_sequence_class_accepts_integer_valued_floats(write_lines):
    path = write_lines('data.tsv', ['id1 ACD 0.0', 'id2 EFG 2.0'])
    data = pid.parse_file(path, Task('sequence', 3))
    assert [sample[2] for sample in data] == [0, 2]


def test_parse_file_residue_class_out_of_range(write_lines):
    path = write_lines('data.tsv', ['id1 ACD 0 1 3'])
    with pytest.raises(ValueError, match='between 0 and 2'):
        pid.parse_file(path, Task('residues', 3))


def test_parse_file_residue_class_must_be_integer(write_lines):
    path = write_lines('data.tsv', ['id1 ACD 0 1.5 2'])
    with pytest.raises(ValueError, match='must be integers'):
        pid.parse_file(path, Task('residues', 3))


def test_parse_file_residue_classification_with_missing_labels(write_lines):
    path = write_lines('data.tsv', ['id1 ACDE 0 nan 2 1', 'id2 FG nan 1'])
    data = pid.parse_file(path, Task('residues', 3), ignore_warnings=True)

    ignored = pid.IGNORED_CLASS_LABEL
    assert data[0][2] == [0, ignored, 2, 1]
    assert data[1][2] == [ignored, 1]


def test_parse_file_residue_regression_with_missing_labels(write_lines):
    path = write_lines('data.tsv', ['id1 ACD 0.5 nan 2.5'])
    data = pid.parse_file(path, Task('residues', 1), ignore_warnings=True)
    assert data[0][2][0] == 0.5 and np.isnan(data[0][2][1]) and data[0][2][2] == 2.5


def test_parse_file_missing_labels_still_validated(write_lines):
    path = write_lines('data.tsv', ['id1 ACD 0 nan 5'])
    with pytest.raises(ValueError, match='between 0 and 2'):
        pid.parse_file(path, Task('residues', 3))


def test_parse_file_reports_missing_labels(write_lines, capsys):
    path = write_lines('data.tsv', ['id1 ACDE 0 nan 1 1', 'id2 FGHK 1 0 nan nan', 'id3 LMNP 0 1 0 1'])

    pid.parse_file(path, Task('residues', 2))
    output = capsys.readouterr().out
    assert '3 of 12 residue labels (25.0%) are missing, in 2 of 3 sequences' in output
    assert 'class imbalance' not in output  # missing labels are not counted as a class

    pid.parse_file(path, Task('residues', 2), ignore_warnings=True)
    assert capsys.readouterr().out == ''


def test_missing_label_mask():
    classification = Task('residues', 3)
    assert pid.missing_label_mask([0, pid.IGNORED_CLASS_LABEL, 2], classification).tolist() == [False, True, False]

    regression = Task('residues', 1)
    assert pid.missing_label_mask(np.array([0.5, np.nan]), regression).tolist() == [False, True]
    assert not pid.missing_label_mask(1.5, Task('sequence', 1))


def test_parse_file_warns_about_duplicates(write_lines, capsys):
    path = write_lines('data.tsv', ['id1 ACD 0', 'id2 ACD 1', 'id3 EFG 1', 'id4 KLM 0'])

    pid.parse_file(path, Task('sequence', 2))
    assert 'Duplicate sequences detected' in capsys.readouterr().out

    pid.parse_file(path, Task('sequence', 2), ignore_warnings=True)
    assert capsys.readouterr().out == ''


# ..........................................................................................
# SequenceDataset

def test_sequence_dataset_subset_and_items():
    data = [['a', 'ACD', 0], ['b', 'EF', 1], ['c', 'GHIK', 2]]
    dataset = pid.SequenceDataset(data, subset=np.array([2, 0]))

    assert len(dataset) == 2
    name, vector, value = dataset[0]
    assert name == 'c' and value == 2
    torch.testing.assert_close(vector, encode_sequence.one_hot('GHIK'))


def test_sequence_dataset_without_subset_uses_all_data():
    data = [['a', 'ACD', 0], ['b', 'EF', 1]]
    assert len(pid.SequenceDataset(data)) == 2


def test_sequence_dataset_empty_subset_is_empty():
    data = [['a', 'ACD', 0], ['b', 'EF', 1]]
    assert len(pid.SequenceDataset(data, subset=np.array([], dtype=int))) == 0


@pytest.mark.parametrize('encoding, expected_width', [('onehot', 20), ('biophysics', 9),
                                                     (data_path('encoding_example.txt'), 6)])
def test_sequence_dataset_encodings(encoding, expected_width):
    dataset = pid.SequenceDataset([['a', 'ACDEF', 1.0]], encoder=encode_sequence.get_encoder(encoding))
    assert dataset[0][1].shape == (5, expected_width)


def test_sequence_dataset_names_and_sequences():
    dataset = pid.SequenceDataset([['a', 'ACD', 0], ['b', 'EF', 1]], subset=np.array([1, 0]))
    assert dataset.names == ['b', 'a']
    assert dataset.sequences == ['EF', 'ACD']


# ..........................................................................................
# Collate functions

def make_batch(values: list[Any]) -> list[tuple[str, torch.Tensor, Any]]:
    """Build a batch of (name, one-hot vector, value) tuples with sequences of length 2, 4 and 3."""
    sequences = ['AC', 'DEFG', 'HIK']
    return [(f'seq{i}', encode_sequence.one_hot(seq), values[i]) for i, seq in enumerate(sequences)]


def test_pad_sequence_vectors():
    vectors = [encode_sequence.one_hot('AC'), encode_sequence.one_hot('DEFG')]
    padded, lengths = pid.pad_sequence_vectors(vectors)

    # Padded to padded_batch_length(4) = 8
    assert padded.shape == (2, 8, 20) and padded.dtype == torch.float32
    assert lengths.tolist() == [2, 4]
    torch.testing.assert_close(padded[0, :2], vectors[0].float())
    assert torch.all(padded[0, 2:] == 0) and torch.all(padded[1, 4:] == 0)


@pytest.mark.parametrize('longest, expected', [(1, 8), (4, 8), (30, 32), (40, 40), (64, 64), (65, 72), (86, 88),
                                               (300, 320), (749, 768), (1000, 1024), (5654, 6144)])
def test_padded_batch_length(longest, expected):
    assert pid.padded_batch_length(longest) == expected


def test_padded_batch_length_adds_little_padding_and_few_shapes():
    padded = np.array([pid.padded_batch_length(length) for length in range(1, 5001)])
    lengths = np.arange(1, 5001)

    assert np.all(padded >= lengths)
    assert np.all(padded[63:] - lengths[63:] < lengths[63:] / 8)  # under 1/8 extra from 64 residues up
    assert len(np.unique(padded)) < 100


def test_padded_batch_length_rejects_empty_sequences():
    with pytest.raises(ValueError):
        pid.padded_batch_length(0)


def test_collate_sequence_classification():
    names, vectors, targets, lengths = pid.collate_batch(make_batch([0, 2, 1]), Task('sequence', 3))

    assert names == ['seq0', 'seq1', 'seq2']
    assert vectors.shape == (3, 8, 20)
    assert targets.dtype == torch.int64 and targets.tolist() == [0, 2, 1]
    assert lengths.tolist() == [2, 4, 3]


def test_collate_sequence_regression():
    names, vectors, targets, lengths = pid.collate_batch(make_batch([0.5, -1.0, 2.0]), Task('sequence', 1))

    assert targets.shape == (3, 1) and targets.dtype == torch.float32
    assert targets.flatten().tolist() == [0.5, -1.0, 2.0]
    assert lengths.tolist() == [2, 4, 3]


def test_collate_residue_classification_marks_padding():
    batch = make_batch([[0, 1], [2, 2, 1, 0], [1, 1, 1]])
    names, vectors, targets, lengths = pid.collate_batch(batch, Task('residues', 3))

    padding = pid.IGNORED_CLASS_LABEL
    assert targets.shape == (3, 8) and targets.dtype == torch.int64
    assert targets.tolist() == [[0, 1] + [padding] * 6, [2, 2, 1, 0] + [padding] * 4, [1, 1, 1] + [padding] * 5]
    assert lengths.tolist() == [2, 4, 3]


def test_collate_keeps_missing_labels():
    batch = make_batch([[0, pid.IGNORED_CLASS_LABEL], [2, 2, 1, 0], [1, 1, 1]])
    _, _, targets, _ = pid.collate_batch(batch, Task('residues', 3))
    assert targets[0].tolist() == [0] + [pid.IGNORED_CLASS_LABEL] * 7

    batch = make_batch([[0.5, np.nan], [1, 2, 3, 4], [9, 8, 7]])
    _, _, targets, _ = pid.collate_batch(batch, Task('residues', 1))
    assert targets[0, 0, 0] == 0.5 and torch.isnan(targets[0, 1, 0]) and targets[0, 2, 0] == 0.0


def test_collate_residue_regression():
    batch = make_batch([[0.5, 1.0], [1, 2, 3, 4], [9, 8, 7]])
    names, vectors, targets, lengths = pid.collate_batch(batch, Task('residues', 1))

    assert targets.shape == (3, 8, 1) and targets.dtype == torch.float32
    assert targets[0, :, 0].tolist() == [0.5, 1.0] + [0.0] * 6
    assert lengths.tolist() == [2, 4, 3]


def test_make_dataloader_random_batching_keeps_order_without_shuffling():
    task = Task('sequence', 3)
    data = pid.parse_file(data_path('seq_class_dataset.tsv'), task, ignore_warnings=True)
    loader = pid.make_dataloader(pid.SequenceDataset(data), task, batch_size=32, batching='random')

    names, vectors, targets, lengths = next(iter(loader))
    assert len(names) == 32 and vectors.shape[0] == 32 and targets.shape == (32,)
    assert names == [sample[0] for sample in data[:32]]  # not shuffled
    assert sum(len(batch[0]) for batch in loader) == 300


def test_make_dataloader_length_batching_sorts_without_shuffling():
    task = Task('sequence', 3)
    data = pid.parse_file(data_path('seq_class_dataset.tsv'), task, ignore_warnings=True)
    loader = pid.make_dataloader(pid.SequenceDataset(data), task, batch_size=32)

    lengths = torch.cat([batch[3] for batch in loader])
    assert len(lengths) == 300
    assert torch.all(lengths[1:] >= lengths[:-1])


def test_make_dataloader_rejects_unknown_batching():
    task = Task('sequence', 3)
    with pytest.raises(ValueError, match='Unknown batching'):
        pid.make_dataloader(pid.SequenceDataset([['a', 'ACD', 0]]), task, batch_size=2, batching='sorted')


def sampler_epoch(lengths: list[int], batch_size: int, shuffle: bool) -> list[list[int]]:
    """One epoch of batches from a LengthBatchSampler."""
    return list(pid.LengthBatchSampler(lengths, batch_size, shuffle))


@pytest.mark.parametrize('shuffle', [False, True])
def test_length_batches_cover_every_sequence_once(shuffle):
    lengths = list(np.random.randint(1, 500, size=1003))
    batches = sampler_epoch(lengths, 16, shuffle)

    assert sorted(index for batch in batches for index in batch) == list(range(1003))
    assert len(batches) == len(pid.LengthBatchSampler(lengths, 16, shuffle)) == 63
    assert sum(len(batch) < 16 for batch in batches) == 1


def test_length_batches_waste_less_padding_than_random_batches():
    lengths = np.random.randint(1, 500, size=2000)

    def padded_positions(batches) -> int:
        return sum(len(batch) * lengths[batch].max() for batch in batches)

    length_batches = sampler_epoch(list(lengths), 32, shuffle=True)
    random_batches = np.array_split(np.random.permutation(2000), len(length_batches))

    # Uniform lengths: random batches are padded to ~the maximum, length batches to ~the mean
    assert padded_positions(length_batches) < 0.6 * padded_positions(random_batches)


def test_length_batches_change_between_epochs_but_follow_the_seed():
    lengths = list(np.random.randint(1, 500, size=500))
    sampler = pid.LengthBatchSampler(lengths, 8, shuffle=True)

    torch.manual_seed(3)
    first, second = list(sampler), list(sampler)
    torch.manual_seed(3)
    repeat = list(sampler)

    assert first != second
    assert first == repeat


def test_longest_batch_comes_first():
    lengths = list(np.random.randint(1, 100, size=300))
    lengths[123] = 1000
    assert 123 in sampler_epoch(lengths, 10, shuffle=True)[0]


def test_padding_class_label_is_ignored_by_cross_entropy():
    assert pid.IGNORED_CLASS_LABEL == torch.nn.CrossEntropyLoss().ignore_index


# ..........................................................................................
# Splitting data

def test_vector_split_sizes_and_disjointness():
    all_samples = np.arange(101)
    first, second = pid.vector_split(all_samples, 0.7)

    assert len(first) == 71  # ceil(0.7 * 101)
    assert len(second) == 30
    assert np.all(np.diff(first) > 0)  # sorted
    assert len(np.intersect1d(first, second)) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([first, second])), all_samples)


def test_vector_split_is_reproducible_with_seed():
    np.random.seed(5)
    first_a, _ = pid.vector_split(np.arange(50), 0.5)
    np.random.seed(5)
    first_b, _ = pid.vector_split(np.arange(50), 0.5)
    np.testing.assert_array_equal(first_a, first_b)


def test_random_split_indices():
    train, val, test = pid.random_split_indices(200, percent_val=0.1, percent_test=0.2)

    assert (len(train), len(val), len(test)) == (140, 20, 40)
    np.testing.assert_array_equal(np.sort(np.concatenate([train, val, test])), np.arange(200))


def test_write_and_read_split_file(tmp_path):
    path = str(tmp_path / 'split.txt')
    pid.write_split_file(path, np.array([3, 0, 1]), np.array([4]), np.array([2, 5]))

    train, val, test = pid.read_split_file(path)
    assert train.tolist() == [0, 1, 3] and val.tolist() == [4] and test.tolist() == [2, 5]


def test_cross_validation_folds():
    folds = pid.cross_validation_folds(np.arange(10, 20), 3)

    assert len(folds) == 3
    held_out = np.concatenate([fold_held_out for _, fold_held_out in folds])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(10, 20))
    for training, fold_held_out in folds:
        assert len(np.intersect1d(training, fold_held_out)) == 0
        np.testing.assert_array_equal(np.sort(np.concatenate([training, fold_held_out])), np.arange(10, 20))


def test_read_split_file_example():
    train, val, test = pid.read_split_file(data_path('splitfile.txt'), num_samples=300)

    np.testing.assert_array_equal(train, np.arange(0, 100))
    np.testing.assert_array_equal(val, np.arange(100, 200))
    np.testing.assert_array_equal(test, np.arange(200, 300))


def test_read_split_file_rejects_overlap(write_lines):
    path = write_lines('split.txt', ['0 1 2', '3 4', '4 5'])
    with pytest.raises(ValueError, match='more than one set'):
        pid.read_split_file(path)


def test_read_split_file_rejects_out_of_range(write_lines):
    path = write_lines('split.txt', ['0 1 2', '3 4', '5 6'])
    with pytest.raises(ValueError, match='outside the range'):
        pid.read_split_file(path, num_samples=6)


def test_read_split_file_requires_three_lines(write_lines):
    path = write_lines('split.txt', ['0 1 2', '3 4'])
    with pytest.raises(ValueError, match='three lines'):
        pid.read_split_file(path)


def dataset_names(dataset: pid.SequenceDataset) -> set[str]:
    """Return the set of sequence IDs in a SequenceDataset."""
    return {sample[0] for sample in dataset.data}


def test_split_data_partitions_all_samples():
    train, val, test = pid.split_data(data_path('seq_regress_dataset.tsv'), Task('sequence', 1))

    assert (len(train), len(val), len(test)) == (210, 45, 45)
    train_names, val_names, test_names = dataset_names(train), dataset_names(val), dataset_names(test)
    assert not (train_names & val_names) and not (train_names & test_names) and not (val_names & test_names)
    assert len(train_names | val_names | test_names) == 300


def test_split_data_custom_fractions():
    train, val, test = pid.split_data(data_path('seq_regress_dataset.tsv'), Task('sequence', 1),
                                      percent_val=0.1, percent_test=0.3)
    assert (len(train), len(val), len(test)) == (180, 30, 90)


def test_split_data_uses_split_file():
    train, val, test = pid.split_data(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                                      split_file=data_path('splitfile.txt'))

    all_data = pid.parse_file(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                              ignore_warnings=True)
    assert train.data == all_data[0:100]
    assert val.data == all_data[100:200]
    assert test.data == all_data[200:300]


@pytest.mark.parametrize('lines, empty_set', [(['', '0 1 2', '3 4 5'], 'training'),
                                               (['0 1 2', '', '3 4 5'], 'validation'),
                                               (['0 1 2', '3 4 5', ''], 'test')])
def test_split_data_rejects_a_split_file_with_an_empty_set(write_lines, lines, empty_set):
    data_file = write_lines('data.tsv', [f's{i} ACDEF {i}.0' for i in range(6)])
    split_file = write_lines('split.txt', lines)
    with pytest.raises(ValueError, match=f'The {empty_set} set is empty'):
        pid.split_data(data_file, Task('sequence', 1), split_file=split_file, ignore_warnings=True)


def test_split_data_rejects_a_random_split_with_an_empty_set(write_lines):
    # 6 sequences with 15% each for validation and test: rounding leaves the test set empty
    data_file = write_lines('data.tsv', [f's{i} ACDEF {i}.0' for i in range(6)])
    with pytest.raises(ValueError, match='The test set is empty'):
        pid.split_data(data_file, Task('sequence', 1), ignore_warnings=True)


def test_split_data_cv_rejects_more_folds_than_sequences(write_lines):
    # 10 sequences: 7 training + 2 validation sequences cannot make 10 folds
    data_file = write_lines('data.tsv', [f's{i} ACDEF {i}.0' for i in range(10)])
    with pytest.raises(ValueError, match='cross-validation folds'):
        pid.split_data_cv(data_file, Task('sequence', 1), n_folds=10, ignore_warnings=True)


def test_split_data_save_splits_round_trip(tmp_path):
    split_output = str(tmp_path / 'saved_split.txt')
    train, val, test = pid.split_data(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                                      save_splits_output=split_output)

    # Re-splitting with the saved file reproduces exactly the same sets
    train2, val2, test2 = pid.split_data(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                                         split_file=split_output)
    assert dataset_names(train) == dataset_names(train2)
    assert dataset_names(val) == dataset_names(val2)
    assert dataset_names(test) == dataset_names(test2)


def test_split_data_is_reproducible_with_seed():
    np.random.seed(11)
    train_a, _, _ = pid.split_data(data_path('seq_class_dataset.tsv'), Task('sequence', 3))
    np.random.seed(11)
    train_b, _, _ = pid.split_data(data_path('seq_class_dataset.tsv'), Task('sequence', 3))
    assert dataset_names(train_a) == dataset_names(train_b)


def test_split_data_cv_folds():
    n_folds = 4
    cv_sets, train, val, test = pid.split_data_cv(data_path('res_class_dataset.tsv'), Task('residues', 3),
                                                  n_folds=n_folds)

    assert len(cv_sets) == n_folds
    train_val_names = dataset_names(train) | dataset_names(val)
    test_names = dataset_names(test)

    fold_test_names = []
    for fold_train, fold_test in cv_sets:
        fold_train_names, fold_val_names = dataset_names(fold_train), dataset_names(fold_test)

        # Within a fold, the CV train and CV test sets are disjoint and together
        # cover exactly the original training + validation data
        assert not (fold_train_names & fold_val_names)
        assert fold_train_names | fold_val_names == train_val_names

        # The held-out test set is never used in cross-validation
        assert not (fold_train_names & test_names) and not (fold_val_names & test_names)
        fold_test_names.append(fold_val_names)

    # Each sample is in exactly one CV test fold
    assert sum(len(names) for names in fold_test_names) == len(train_val_names)
    assert set().union(*fold_test_names) == train_val_names


def test_split_data_cv_with_split_file():
    cv_sets, train, val, test = pid.split_data_cv(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                                                  split_file=data_path('splitfile.txt'))

    all_data = pid.parse_file(data_path('seq_class_dataset.tsv'), Task('sequence', 3),
                              ignore_warnings=True)
    assert test.data == all_data[200:300]
    assert len(cv_sets) == 5
    assert sum(len(fold_test) for _, fold_test in cv_sets) == 200
