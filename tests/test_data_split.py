"""
Tests for how parrot-train and parrot-optimize divide the data into sets
(--split, parrot/scripts/data_split.py): a file named by --split is read first,
the methods aim for --set-fractions and keep identical or similar sequences
together, and every option that does not fit the split is rejected.
"""

import argparse
import shutil

import numpy as np
import pytest

from parrot import process_input_data as pid
from parrot.scripts import data_split, parrot_train
from parrot.tasks import Task

from tests.helpers import data_path, training_command

# The sequence regression example: 300 distinct, unrelated sequences
EXAMPLE_FILE = 'seq_regress_dataset.tsv'
EXAMPLE_TASK = Task('sequence', 1)

# Families of near-identical sequences for the similarity splits
NUM_FAMILIES = 12
FAMILY_SIZE = 5
FAMILY_SEQUENCE_LENGTH = 60
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

needs_mmseqs = pytest.mark.skipif(shutil.which('mmseqs') is None, reason='MMseqs2 (mmseqs) is not installed')


def split_arguments(split: str, set_fractions: list[float] | None = None,
                    save_splits: bool = False) -> argparse.Namespace:
    """The arguments data_split_from_args() reads, as the parser would give them."""
    return argparse.Namespace(split=split, set_fractions=set_fractions, save_splits=save_splits, silent=True)


def example_data() -> list[pid.Sample]:
    """The parsed sequence regression example."""
    return pid.parse_file(data_path(EXAMPLE_FILE), EXAMPLE_TASK, ignore_warnings=True)


def family_data(seed: int = 0) -> tuple[list[pid.Sample], list[int]]:
    """Families of sequences that differ from each other by one residue, and unrelated between families.

    Returns
    -------
    list
        Parsed samples, the families interleaved
    list of int
        The family of each sample
    """
    rng = np.random.default_rng(seed)
    data, families = [], []
    ancestors = [''.join(rng.choice(list(AMINO_ACIDS), FAMILY_SEQUENCE_LENGTH)) for _ in range(NUM_FAMILIES)]
    for member in range(FAMILY_SIZE):
        for family, ancestor in enumerate(ancestors):
            position = int(rng.integers(FAMILY_SEQUENCE_LENGTH))
            sequence = ancestor[:position] + AMINO_ACIDS[member] + ancestor[position + 1:]
            data.append([f'family{family}_{member}', sequence, 0.0])
            families.append(family)
    return data, families


def set_of_each_sample(sets: tuple[np.ndarray, np.ndarray, np.ndarray], num_samples: int) -> list[str | None]:
    """The set ('train', 'val' or 'test') of each sample, or None if it is in no set."""
    names: list[str | None] = [None] * num_samples
    for name, indices in zip(('train', 'val', 'test'), sets):
        for index in indices:
            names[index] = name
    return names


def assert_families_kept_together(sets: tuple[np.ndarray, np.ndarray, np.ndarray], families: list[int]) -> None:
    """Every family's assigned members are in one set."""
    names = set_of_each_sample(sets, len(families))
    for family in set(families):
        family_sets = {names[index] for index, member_family in enumerate(families)
                       if member_family == family and names[index] is not None}
        assert len(family_sets) <= 1, (family, family_sets)


# ..........................................................................................
# What --split names

def test_a_file_named_by_split_is_read_first_even_if_it_is_a_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'random').write_text('0 1 2 3 4 5\n6 7\n8 9\n')
    split = data_split.data_split_from_args(split_arguments('random'))
    assert split.split_file == str(tmp_path / 'random') and split.method is None

    training, validation, test = split(example_data()[:10])
    assert (list(training), list(validation), list(test)) == ([0, 1, 2, 3, 4, 5], [6, 7], [8, 9])


@pytest.mark.parametrize('method', data_split.SPLIT_METHODS)
def test_a_method_is_used_when_no_file_has_its_name(tmp_path, monkeypatch, method):
    monkeypatch.chdir(tmp_path)
    if method == 'mmseqs' and shutil.which('mmseqs') is None:
        pytest.skip('MMseqs2 (mmseqs) is not installed')
    split = data_split.data_split_from_args(split_arguments(method))
    assert split.method == method and split.split_file is None
    assert split.fractions == pid.DEFAULT_SET_FRACTIONS


@pytest.mark.parametrize('arguments, message', [
    (split_arguments('randm'), 'there is no file of that name, and it is not one of mmseqs, kmer, random'),
    (split_arguments('random', set_fractions=[0.5, 0.2, 0.2]), 'must sum to 1'),
])
def test_unusable_splits_are_rejected(tmp_path, monkeypatch, arguments, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=message):
        data_split.data_split_from_args(arguments)


@pytest.mark.parametrize('option, message', [
    (dict(set_fractions=[0.6, 0.2, 0.2]), '--set-fractions applies to --split mmseqs, kmer, random'),
    (dict(save_splits=True), '--save-splits saves a split made by --split mmseqs, kmer, random'),
])
def test_options_for_methods_are_rejected_with_a_split_file(tmp_path, option, message):
    split_file = tmp_path / 'split.txt'
    split_file.write_text('0\n1\n2\n')
    with pytest.raises(ValueError, match=message):
        data_split.data_split_from_args(split_arguments(str(split_file), **option))


# ..........................................................................................
# The methods

@pytest.mark.parametrize('fractions, sizes', [((0.7, 0.15, 0.15), (210, 45, 45)), ((0.6, 0.2, 0.2), (180, 60, 60))])
@pytest.mark.parametrize('method', ['random', 'kmer'])
def test_methods_aim_for_the_set_fractions(method, fractions, sizes):
    split = data_split.DataSplit(split_file=None, method=method, fractions=fractions, silent=True)
    sets = split(example_data())
    assert tuple(len(indices) for indices in sets) == sizes == split.set_sizes
    assert sorted(np.concatenate(sets)) == list(range(300))


def test_random_split_keeps_identical_sequences_together():
    data = [[f'seq{index}', 'ACDEFGHIKL' + 'W' * (index % 10), 0.0] for index in range(100)]
    split = data_split.DataSplit(split_file=None, method='random', fractions=(0.6, 0.2, 0.2), silent=True)
    assert_families_kept_together(split(data), [index % 10 for index in range(100)])


def test_kmer_split_keeps_similar_sequences_together():
    data, families = family_data()
    split = data_split.DataSplit(split_file=None, method='kmer', fractions=(0.6, 0.2, 0.2), silent=True)
    assert_families_kept_together(split(data), families)
    assert 'k-mer' in split.description() and 'similarity criterion' in split.description()


@needs_mmseqs
def test_mmseqs_split_keeps_homologous_sequences_together():
    data, families = family_data()
    split = data_split.DataSplit(split_file=None, method='mmseqs', fractions=(0.6, 0.2, 0.2), silent=True)
    assert_families_kept_together(split(data), families)
    assert 'MMseqs2' in split.description()


def test_the_same_seed_gives_the_same_split():
    splits = []
    for _ in range(2):
        np.random.seed(7)
        split = data_split.DataSplit(split_file=None, method='random', fractions=(0.7, 0.15, 0.15), silent=True)
        splits.append([list(indices) for indices in split(example_data())])
    assert splits[0] == splits[1]


def test_too_little_data_to_fill_every_set_is_an_error():
    data = [['a', 'ACDEF', 0.0], ['b', 'ACDEF', 1.0], ['c', 'ACDEF', 2.0]]
    split = data_split.DataSplit(split_file=None, method='random', fractions=(0.7, 0.15, 0.15), silent=True)
    with pytest.raises(ValueError, match='Use more data, larger --set-fractions or a split-file'):
        split(data)


# ..........................................................................................
# Describing the split

def test_lines_a_split_file_leaves_out_are_counted(tmp_path, capsys):
    split_file = tmp_path / 'split.txt'
    split_file.write_text('0 1 2 3 4\n5 6\n7\n')
    split = data_split.DataSplit(split_file=str(split_file), method=None, fractions=None, silent=False)
    split(example_data()[:10])

    assert 'Note: 2 of 10 sequences are in no set (the split-file does not list them)' in capsys.readouterr().out
    assert split.description() == (f'from the split-file {split_file}; 2 of 10 sequences are in no set '
                                    '(the split-file does not list them)')


def test_method_description_gives_the_target_fractions():
    split = data_split.DataSplit(split_file=None, method='random', fractions=(0.8, 0.1, 0.1), silent=True)
    split(example_data())
    assert split.description() == ('at random, keeping identical sequences in the same set (as parrot-preprocess '
                                   '--split random), aiming for 0.8 of the sequences for training, 0.1 for '
                                   'validation and 0.1 for testing')


# ..........................................................................................
# Through parrot-train

def test_train_saves_a_methods_split_with_the_fractions_asked_for(run_cli, tmp_path):
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(EXAMPLE_FILE), tmp_path / 'net',
                                                'sequence', 1, '--split', 'kmer', '--set-fractions', '0.6', '0.2',
                                                '0.2', '--save-splits', '-e', '1', '--silent'))

    sets = pid.read_split_file(str(tmp_path / 'net_split_file.txt'), num_samples=300)
    assert tuple(len(indices) for indices in sets) == (180, 60, 60)
    info = (tmp_path / 'net_info.txt').read_text()
    assert 'as parrot-preprocess --split kmer' in info and 'aiming for 0.6 of the sequences for training' in info


def test_train_reads_a_split_file_called_random(run_cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'random').write_text(' '.join(str(index) for index in range(200)) + '\n'
                                     + ' '.join(str(index) for index in range(200, 250)) + '\n'
                                     + ' '.join(str(index) for index in range(250, 300)) + '\n')
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(EXAMPLE_FILE), tmp_path / 'net',
                                                'sequence', 1, '--split', 'random', '-e', '1', '--silent'))

    assert f'from the split-file {tmp_path / "random"}' in (tmp_path / 'net_info.txt').read_text()
    assert len((tmp_path / 'net_predictions.tsv').read_text().splitlines()) == 2 * 50
