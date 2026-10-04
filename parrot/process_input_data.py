"""
Module with functions for processing an input datafile into a PyTorch-compatible
format: reading and validating datafiles, building datasets, collating batches
and splitting data into training, validation and test sets.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import functools
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from parrot import parrot_exceptions
from parrot.encode_sequence import OneHotEncoder, SequenceEncoder
from parrot.tasks import Task
from parrot.tools import dataset_warnings

# Text used in residue-mapped datafiles for a residue with no label (e.g. a residue
# not observed in a structure). Any capitalisation is accepted.
MISSING_LABEL_TEXT = 'nan'

# Class label given to padded positions and to residues with missing labels in
# residue-classification data. This matches the default ignore_index of
# torch.nn.CrossEntropyLoss, so these positions never contribute to the loss.
IGNORED_CLASS_LABEL = -100

# Tolerance (in number of samples) used when converting a set fraction into a set
# size, to absorb floating-point error in the fractions themselves
SPLIT_SIZE_TOLERANCE = 1e-9

# How sequences are grouped into training batches. 'length' groups sequences of
# similar length (see LengthBatchSampler), so little of each batch is padding;
# 'random' is PARROT's original behaviour of shuffling the training data freely.
BATCHING_CHOICES = ('length', 'random')
DEFAULT_BATCHING = 'length'

# Length batching sorts pools of this many batches at a time. Bigger pools give
# batches of more similar length but less random batch composition; 50 is the
# value used by fastai's SortishSampler (Howard & Gugger 2020, Information 11:108).
LENGTH_POOL_BATCHES = 50

# Batches are padded to a "round" length rather than exactly to their longest
# sequence (see padded_batch_length). On GPUs, and Apple-silicon (MPS) GPUs in
# particular, every new tensor shape costs a compilation, so length-batched data
# padded to exact lengths spends more time compiling than computing (measured on
# MPS: epochs 2-5x slower than with rounded lengths). Rounding up in steps of at
# least PAD_MIN_STEP residues, and about 1/PAD_STEPS_PER_DOUBLING of the length,
# keeps the number of distinct shapes small while adding less than 1/8 extra padding.
PAD_MIN_STEP = 8
PAD_STEPS_PER_DOUBLING = 8

# A parsed datafile line: [sequence ID, sequence, value(s)]
Sample = list[Any]


# .................................................................
# Reading datafiles

def read_tsv_raw(tsvfile: str, delimiter: str | None = None) -> list[list[str]]:
    """Split a whitespace-separated datafile into lines of fields

    Empty lines and comment lines (lines that start with a # symbol) are
    skipped. No other checking is done.

    Parameters
    ----------
    tsvfile : str
        Path to a whitespace-separated datafile
    delimiter : str or None
        String used to split columns. Default (None) splits on any whitespace.

    Returns
    -------
    list of list of str
        One list of fields for each non-empty, non-comment line
    """

    with open(tsvfile) as fh:
        content = fh.readlines()

    lines = []
    for line in content:

        # skip empty lines
        if len(line.strip()) == 0:
            continue

        # skip comment lines
        if line.strip()[0] == '#':
            continue

        lines.append(line.strip().split(delimiter))

    return lines


def __parse_lines(lines: list[list[str]], datatype: str, validate: bool = True) -> list[Sample]:
    """Convert the fields of each datafile line into [id, sequence, value(s)]

    For residue-mapped data, individual residue values may be 'nan' to mark a
    residue with no label; these are kept as NaN.

    Parameters
    ----------
    lines : list of list of str
        Output from read_tsv_raw()
    datatype : str
        'residues' (a value per residue) or 'sequence' (a single value)
    validate : bool
        If True, ensure the number of residue values equals the number of residues

    Returns
    -------
    list
        One [id, sequence, value(s)] list per line, where value(s) is a float
        (datatype='sequence') or a numpy array of floats (datatype='residues')

    Raises
    ------
    IOExceptionParrot
        If a line cannot be parsed. The line number and offending line are reported.
    """

    if datatype not in ['residues', 'sequence']:
        raise ValueError('Invalid datatype. Must be "residues" or "sequence".')

    data: list[Sample] = []
    line_number = 0
    line: list[str] = []
    try:
        for line_number, line in enumerate(lines, start=1):

            if datatype == 'residues':
                # Missing labels ('nan') are allowed for individual residues, but every
                # sequence needs at least one label, and infinite values are never valid
                residue_data = np.array(line[2:], dtype=float)
                if np.any(np.isinf(residue_data)):
                    raise ValueError('Residue values must be finite numbers or nan (missing), not inf.')
                if len(residue_data) > 0 and np.all(np.isnan(residue_data)):
                    raise ValueError('Every residue label of this sequence is missing (nan).')

                data.append([line[0], line[1], residue_data])

            else:
                # Exactly one value is expected; extra columns usually mean a residues-
                # formatted file was passed with --datatype sequence.
                if len(line) != 3:
                    raise ValueError(f'Expected 3 columns (id, sequence, value) but found {len(line)}.')

                sequence_value = float(line[2])
                if not np.isfinite(sequence_value):
                    raise ValueError('Sequence values must be finite numbers (no NaN or inf).')

                data.append([line[0], line[1], sequence_value])

    except Exception as e:
        print('Excecption raised on parsing input file...')
        print(e)
        print('')
        raise parrot_exceptions.IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nMake sure your datafile does not have empty lines at the end of the file.\nError on line {line_number}:\n{line}")

    if validate and datatype == 'residues':
        for line_number, sample in enumerate(data, start=1):
            if len(sample[1]) != len(sample[2]):
                raise parrot_exceptions.IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nInconsistent number of residue values and residues. Error on line {line_number}:\n{sample}")

    return data


def parse_file(tsvfile: str, task: Task, exclude_seq_id: bool = False, ignore_warnings: bool = False) -> list[Sample]:
    """Parse a datafile containing sequences and values

    Each line of the file should be "<seqID> <sequence> <value(s)>", or
    "<sequence> <value(s)>" if `exclude_seq_id` is True. `value(s)` is a single
    number for sequence-mapped data or one number per residue for residue-mapped
    data. For classification, values must be integer class labels in the range
    [0, num_classes - 1]. For residue-mapped data, a residue with no label can be
    given the value 'nan'; it is ignored when training and evaluating.

    Parameters
    ----------
    tsvfile : str
        Path to a whitespace-separated datafile
    task : Task
        The task the data is for
    exclude_seq_id : bool, optional
        True if the lines do not begin with a sequence ID (default False). An
        empty ID is used for every sequence.
    ignore_warnings : bool, optional
        If False, check the dataset against basic heuristics and print warnings
        for common issues (duplicates, class imbalance, skewed values).

    Returns
    -------
    list
        One [seqID, sequence, value(s)] list per data line. Classification labels
        are ints (or lists of ints for residue-mapped data, with missing labels set
        to IGNORED_CLASS_LABEL). Missing residue regression values are NaN.

    Raises
    ------
    ValueError
        If any class label is not an integer in the range [0, num_classes - 1]
    """

    lines = read_tsv_raw(tsvfile)

    # Add a dummy seqID if none are provided
    if exclude_seq_id:
        for line in lines:
            line.insert(0, '')

    data = __parse_lines(lines, task.datatype)

    if not ignore_warnings:
        dataset_warnings.check_duplicate_sequences(data)
        if task.per_residue:
            dataset_warnings.report_missing_residue_labels(data)
        if task.is_classification:
            dataset_warnings.check_class_imbalance(data)
        else:
            dataset_warnings.check_regression_imbalance(data)

    if task.is_classification:
        for sample in data:
            sample[2] = _validate_class_labels(sample, task)

    return data


def _validate_class_labels(sample: Sample, task: Task) -> int | list[int]:
    """Convert a sample's class label(s) to int(s), checking they are valid

    Parameters
    ----------
    sample : list
        A parsed [seqID, sequence, value(s)] sample
    task : Task
        A classification task

    Returns
    -------
    int or list of int
        The class label (sequence-mapped) or labels (residue-mapped, with missing
        labels set to IGNORED_CLASS_LABEL)

    Raises
    ------
    ValueError
        If any label is not a whole number in the range [0, num_classes - 1].
        Labels are never silently truncated (e.g. 1.5 is not turned into 1).
    """

    labels = np.atleast_1d(np.asarray(sample[2], dtype=float))
    missing = np.isnan(labels)
    present = labels[~missing]

    if np.any(present != np.round(present)):
        raise ValueError(f"Invalid class label on entry {sample[0]}.\nClass labels must be integers, got {sample[2]}")

    if np.any(present < 0) or np.any(present >= task.num_classes):
        raise ValueError(f"Invalid class label on entry {sample[0]}.\nClass labels must be between 0 and {task.num_classes - 1}, got {sample[2]}")

    integer_labels = [IGNORED_CLASS_LABEL if is_missing else int(label) for label, is_missing in zip(labels, missing)]
    if task.per_residue:
        return integer_labels
    return integer_labels[0]


def missing_label_mask(values: Any, task: Task) -> np.ndarray:
    """Mark which of a sequence's (or residue's) labels are missing

    Parameters
    ----------
    values : number or array
        True value(s) as stored by parse_file()
    task : Task
        The task the values are for

    Returns
    -------
    numpy bool array
        True where the label is missing (NaN for regression, IGNORED_CLASS_LABEL
        for classification), with the same shape as `values`
    """

    values = np.asarray(values)
    if task.is_classification:
        return values == IGNORED_CLASS_LABEL
    return np.isnan(values)


# .................................................................
# Datasets and batches

class SequenceDataset(Dataset):
    """A PyTorch-compatible dataset of sequences and their values

    Sequences are stored as strings and encoded on the fly when an item is
    requested.

    Attributes
    ----------
    data : list
        Each item is a [seqID, sequence, value(s)] list
    encoder : SequenceEncoder
        Converts sequences to numeric vectors
    """

    def __init__(self, data: list[Sample], subset: np.ndarray | None = None,
                 encoder: SequenceEncoder | None = None) -> None:
        """
        Parameters
        ----------
        data : list
            Each item is a [seqID, sequence, value(s)] list
        subset : numpy int array, optional
            Indices of `data` to include. None (default) includes all of
            `data`; an empty array gives an empty dataset.
        encoder : SequenceEncoder, optional
            How to encode sequences (default one-hot)
        """

        self.data = data if subset is None else [data[i] for i in subset]

        self.encoder = encoder if encoder is not None else OneHotEncoder()

    def __len__(self) -> int:
        """Get the number of sequences in this dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, Any]:
        """Return (name, encoded sequence, value(s)) for one sequence."""
        name, sequence, value = self.data[idx]
        return name, self.encoder.encode(sequence), value

    @property
    def names(self) -> list[str]:
        """The sequence ID of every sequence, in dataset order."""
        return [sample[0] for sample in self.data]

    @property
    def sequences(self) -> list[str]:
        """Every sequence, in dataset order."""
        return [sample[1] for sample in self.data]


def padded_batch_length(longest: int) -> int:
    """The length a batch is padded to, given its longest sequence

    The longest length is rounded up to a multiple of a step that grows with
    length: PAD_MIN_STEP residues for short sequences, and otherwise the largest
    power of two that is at most 1/PAD_STEPS_PER_DOUBLING of the length (e.g.
    40 -> 40, 86 -> 88, 300 -> 320, 749 -> 768). See PAD_MIN_STEP for why.

    Parameters
    ----------
    longest : int
        Length of the longest sequence in the batch (at least 1)

    Returns
    -------
    int
        The padded length, >= `longest`
    """
    if longest < 1:
        raise ValueError(f'Sequences must have at least one residue, got a longest length of {longest}')
    step = max(PAD_MIN_STEP, 2 ** math.floor(math.log2(longest / PAD_STEPS_PER_DOUBLING)))
    return math.ceil(longest / step) * step


def pad_sequence_vectors(seq_vectors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a list of encoded sequences to a common length

    Every PARROT architecture masks padding, so the amount of padding does not
    change any result; batches are padded slightly beyond their longest
    sequence (see padded_batch_length) because that is faster on GPUs.

    Parameters
    ----------
    seq_vectors : list of torch.Tensor
        Encoded sequences, each of shape [sequence_length X input_size]

    Returns
    -------
    torch.Tensor
        Zero-padded float32 batch of shape [batch_dim X padded_length X input_size],
        where padded_length = padded_batch_length(longest sequence length)
    torch.Tensor
        The true (unpadded) length of each sequence in the batch
    """

    lengths = [len(vector) for vector in seq_vectors]
    padded_length = padded_batch_length(max(lengths))

    # Filled directly as float32 (each value is rounded to float32 exactly as a float64 batch converted
    # afterwards would be, but without building and converting a float64 copy)
    padded_seq_vectors = torch.zeros(len(seq_vectors), padded_length, seq_vectors[0].shape[-1])
    for i, vector in enumerate(seq_vectors):
        padded_seq_vectors[i, :len(vector)] = torch.as_tensor(vector)

    return padded_seq_vectors, torch.tensor(lengths, dtype=torch.long)


def pad_targets(targets: list[Any], task: Task, padded_length: int) -> torch.Tensor:
    """Stack the target values of a batch into a tensor

    Parameters
    ----------
    targets : list
        One target per sequence: a class label or value (sequence-mapped) or a
        list of labels or values (residue-mapped)
    task : Task
        The task the targets are for
    padded_length : int
        The length the batch's sequences were padded to

    Returns
    -------
    torch.Tensor
        Sequence classification: int64 [batch_dim].
        Sequence regression: float32 [batch_dim X 1].
        Residue classification: int64 [batch_dim X padded_length], with padded
        positions (and missing labels) set to IGNORED_CLASS_LABEL so the loss
        ignores them.
        Residue regression: float32 [batch_dim X padded_length X 1], with padded
        positions set to 0 and missing labels left as NaN (both must be masked
        out of the loss).
    """

    if not task.per_residue:
        if task.is_classification:
            return torch.LongTensor(targets)
        return torch.FloatTensor([[target] for target in targets])

    padding_value = IGNORED_CLASS_LABEL if task.is_classification else 0
    padded = np.full([len(targets), padded_length], padding_value, dtype=float)
    for i, target in enumerate(targets):
        padded[i][0:len(target)] = target

    if task.is_classification:
        return torch.LongTensor(padded.astype(np.int64))
    return torch.FloatTensor(padded).unsqueeze(-1)


def collate_batch(batch: list[tuple[str, torch.Tensor, Any]],
                  task: Task) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (name, encoded sequence, value) items into a padded batch

    Parameters
    ----------
    batch : list
        Items from a SequenceDataset
    task : Task
        The task the data is for (decides how targets are formatted)

    Returns
    -------
    tuple
        (names, zero-padded sequence vectors, targets, true sequence lengths).
        See pad_sequence_vectors() and pad_targets() for the shapes.
    """

    names = [item[0] for item in batch]
    padded_seq_vectors, lengths = pad_sequence_vectors([item[1] for item in batch])
    targets = pad_targets([item[2] for item in batch], task, padded_seq_vectors.shape[1])
    return names, padded_seq_vectors, targets, lengths


class LengthBatchSampler(Sampler[list[int]]):
    """Group a dataset into batches of sequences with similar lengths

    Every batch is padded to (just beyond) the length of its longest sequence,
    so batching sequences of similar length avoids computing on padding: for
    data whose lengths vary widely, most of a randomly composed batch can be
    padding.

    For training (shuffle=True) the batches are "sortish" (as in fastai's
    SortishSampler): each epoch the dataset is shuffled and cut into pools of
    LENGTH_POOL_BATCHES batches, each pool is sorted by length and cut into
    batches, and the order of the batches is shuffled. Batch composition
    therefore still changes every epoch. The batch holding the longest
    sequences comes first, so a batch too large for memory fails immediately.

    Without shuffling (for evaluation) the whole dataset is sorted by length.
    Because every PARROT architecture's output for a sequence does not depend
    on what it is batched with, this changes how fast evaluation runs but not
    its results.

    Attributes
    ----------
    lengths : numpy int array
        Length of each sequence in the dataset
    batch_size : int
        Maximum number of sequences per batch
    shuffle : bool
        Whether batches are randomised each epoch
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, shuffle: bool) -> None:
        """
        Parameters
        ----------
        lengths : sequence of int
            Length of each sequence in the dataset, in dataset order
        batch_size : int
            Maximum number of sequences per batch
        shuffle : bool
            True for training (sortish, re-randomised every epoch); False for a
            fixed order sorted by length
        """
        if batch_size < 1:
            raise ValueError(f'Batch size must be positive, got {batch_size}')
        self.lengths = np.asarray(lengths, dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the batches of one epoch, each a list of dataset indices."""
        if not self.shuffle:
            by_length = np.argsort(self.lengths, kind='stable')
            yield from self.cut_into_batches(by_length)
            return

        # Shuffle with torch's global generator, so --seed makes this reproducible
        shuffled = torch.randperm(len(self.lengths)).numpy()
        pool_size = self.batch_size * LENGTH_POOL_BATCHES
        batches = []
        for start in range(0, len(shuffled), pool_size):
            pool = shuffled[start:start + pool_size]
            batches += self.cut_into_batches(pool[np.argsort(self.lengths[pool], kind='stable')])

        batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        longest = max(range(len(batches)), key=lambda i: self.lengths[batches[i]].max())
        batches.insert(0, batches.pop(longest))
        yield from batches

    def cut_into_batches(self, indices: np.ndarray) -> list[list[int]]:
        """Cut an ordered array of dataset indices into consecutive batches."""
        return [indices[start:start + self.batch_size].tolist() for start in range(0, len(indices), self.batch_size)]


def make_dataloader(dataset: SequenceDataset, task: Task, batch_size: int, shuffle: bool = False,
                    batching: str = DEFAULT_BATCHING) -> DataLoader:
    """Wrap a dataset in a DataLoader that produces padded batches

    Parameters
    ----------
    dataset : SequenceDataset
        The dataset to load from
    task : Task
        The task the data is for
    batch_size : int
        Number of sequences per batch
    shuffle : bool, optional
        Whether to reshuffle the batches every epoch (use for training data only)
    batching : str, optional
        'length' (default) to batch sequences of similar length together (see
        LengthBatchSampler), or 'random' for freely shuffled batches (or, without
        shuffling, batches in dataset order)

    Returns
    -------
    torch.utils.data.DataLoader
        Batches in the order given by the loader's batch_sampler

    Raises
    ------
    ValueError
        If `batching` is not one of BATCHING_CHOICES
    """

    # functools.partial (rather than a closure) keeps the collate function picklable,
    # which DataLoader needs if it is ever run with multiple worker processes
    collate = functools.partial(collate_batch, task=task)

    if batching == 'random':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    if batching == 'length':
        sampler = LengthBatchSampler([len(sequence) for sequence in dataset.sequences], batch_size, shuffle)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    raise ValueError(f"Unknown batching '{batching}': choose from {', '.join(BATCHING_CHOICES)}")


# .................................................................
# Splitting data

def vector_split(v: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a vector randomly by a specified proportion

    Randomly divide the values of a vector into two, non-overlapping smaller
    vectors. The proportions of the two vectors will be `fraction` and
    (1 - `fraction`).

    Parameters
    ----------
    v : numpy array
        The vector to divide
    fraction : float
        Size proportion for the returned vectors. Should be in the range [0-1].

    Returns
    -------
    numpy array
        a sorted subset of `v` of length `fraction` * len(v) (rounding up)
    numpy array
        the remaining (1-`fraction`) * len(v) values of `v`
    """

    # Subtract a tiny tolerance before rounding up so that floating-point noise
    # (e.g. 1 - 0.1 - 0.3 = 0.6000000000000001) cannot push an exact size up by one
    segment1_size = math.ceil(fraction*len(v) - SPLIT_SIZE_TOLERANCE)
    segment1 = np.random.choice(v, size=segment1_size, replace=False)
    segment1.sort()
    segment2 = np.setdiff1d(v, segment1, assume_unique=True)
    return segment1, segment2


def random_split_indices(num_samples: int, percent_val: float,
                         percent_test: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly assign sample indices to training, validation and test sets

    Parameters
    ----------
    num_samples : int
        Total number of samples
    percent_val : float
        Fraction of samples for the validation set
    percent_test : float
        Fraction of samples for the test set. The training set gets the rest.

    Returns
    -------
    tuple of numpy int arrays
        Sorted (training, validation, test) indices
    """

    percent_train = 1 - percent_val - percent_test
    training_samples, val_test_samples = vector_split(np.arange(num_samples), percent_train)

    # Repeat procedure to split val and test sets
    val_test_fraction = percent_val / (percent_val + percent_test)
    val_samples, test_samples = vector_split(val_test_samples, val_test_fraction)

    return training_samples, val_samples, test_samples


def read_split_file(split_file: str, num_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in a split_file

    Parameters
    ----------
    split_file : str
        Path to a split-file: three lines of whitespace-separated indices for
        the training, validation and test sets
    num_samples : int, optional
        Number of sequences in the datafile the split-file refers to. If
        provided, every index must lie in the range [0, num_samples).

    Returns
    -------
    tuple of numpy int arrays
        (training, validation, test) indices

    Raises
    ------
    ValueError
        If the file does not contain three lines, if any index appears in
        more than one set (data leakage), or if any index is out of range
    """

    with open(split_file) as f:
        lines = [line.rstrip().split() for line in f]

    if len(lines) < 3:
        raise ValueError(f'Split-file {split_file} must contain three lines (train, validation, test).')

    training_samples = np.array([int(i) for i in lines[0]], dtype=int)
    val_samples = np.array([int(i) for i in lines[1]], dtype=int)
    test_samples = np.array([int(i) for i in lines[2]], dtype=int)

    # The same sample must never appear in more than one set
    all_samples = np.concatenate([training_samples, val_samples, test_samples])
    if len(np.unique(all_samples)) != len(all_samples):
        raise ValueError(f'Split-file {split_file} assigns at least one sample index to more than one set (or repeats an index).')

    if num_samples is not None:
        if np.any(all_samples < 0) or np.any(all_samples >= num_samples):
            raise ValueError(f'Split-file {split_file} contains indices outside the range 0 to {num_samples - 1}.')

    return training_samples, val_samples, test_samples


def write_split_file(split_file: str, training_samples: np.ndarray, val_samples: np.ndarray,
                     test_samples: np.ndarray) -> None:
    """Write training, validation and test indices to a split-file

    Parameters
    ----------
    split_file : str
        Path to write to
    training_samples, val_samples, test_samples : numpy int arrays
        Indices of each set (written in sorted order)
    """

    with open(split_file, 'w') as out:
        for indices in (training_samples, val_samples, test_samples):
            out.write(" ".join(np.sort(indices).astype('str')) + "\n")


def check_sets_not_empty(training_samples: np.ndarray, val_samples: np.ndarray, test_samples: np.ndarray,
                         source: str) -> None:
    """Raise if the training, validation or test set has no sequences

    Training needs all three: the training set to learn from, the validation set
    to choose the best epoch, and the test set to measure performance.

    Parameters
    ----------
    training_samples, val_samples, test_samples : numpy int arrays
        Indices of each set
    source : str
        Where the split came from, ending the error message (e.g. 'in split-file x.txt')

    Raises
    ------
    ValueError
        If any set is empty
    """
    empty = [name for name, samples in (('training', training_samples), ('validation', val_samples),
                                        ('test', test_samples)) if len(samples) == 0]
    if empty:
        sets = f'The {empty[0]} set is' if len(empty) == 1 else f"The {' and '.join(empty)} sets are"
        raise ValueError(f'{sets} empty {source}')


def load_and_split(data_file: str, task: Task, exclude_seq_id: bool = False, split_file: str | None = None,
                   percent_val: float = 0.15, percent_test: float = 0.15, ignore_warnings: bool = False,
                   save_splits_output: str | None = None) -> tuple[list[Sample], np.ndarray, np.ndarray, np.ndarray]:
    """Parse a datafile and decide which samples go in each set

    Parameters
    ----------
    data_file : str
        Path to the datafile
    task : Task
        The task the data is for
    exclude_seq_id : bool, optional
        True if the datafile has no sequence ID column
    split_file : str, optional
        Split-file assigning samples to sets. If None (default) samples are
        assigned randomly using `percent_val` and `percent_test`.
    percent_val, percent_test : float, optional
        Fractions of the data assigned to the validation and test sets when no
        split-file is given (default 0.15 each)
    ignore_warnings : bool, optional
        If True, do not print dataset warnings
    save_splits_output : str, optional
        If given (and no split-file is used), write the random split to this path

    Returns
    -------
    tuple
        (parsed data, training indices, validation indices, test indices)
    """

    data = parse_file(data_file, task, exclude_seq_id=exclude_seq_id, ignore_warnings=ignore_warnings)

    if split_file is not None:
        training_samples, val_samples, test_samples = read_split_file(split_file, len(data))
        check_sets_not_empty(training_samples, val_samples, test_samples, f'in split-file {split_file}')
        return data, training_samples, val_samples, test_samples

    training_samples, val_samples, test_samples = random_split_indices(len(data), percent_val, percent_test)
    check_sets_not_empty(training_samples, val_samples, test_samples,
                         f'when {len(data)} sequences are split with validation and test fractions of '
                         f'{percent_val} and {percent_test}. Use more data, larger fractions or a split-file.')
    if save_splits_output is not None:
        write_split_file(save_splits_output, training_samples, val_samples, test_samples)

    return data, training_samples, val_samples, test_samples


def split_data(data_file: str, task: Task, encoder: SequenceEncoder | None = None, exclude_seq_id: bool = False,
               split_file: str | None = None, percent_val: float = 0.15, percent_test: float = 0.15,
               ignore_warnings: bool = False,
               save_splits_output: str | None = None) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Divide a datafile into training, validation, and test datasets

    Parameters
    ----------
    data_file : str
        Path to the datafile
    task : Task
        The task the data is for
    encoder : SequenceEncoder, optional
        How to encode sequences (default one-hot)
    exclude_seq_id, split_file, percent_val, percent_test, ignore_warnings, save_splits_output
        See load_and_split()

    Returns
    -------
    tuple of SequenceDataset
        (training set, validation set, test set)
    """

    data, training_samples, val_samples, test_samples = load_and_split(
        data_file, task, exclude_seq_id=exclude_seq_id, split_file=split_file, percent_val=percent_val,
        percent_test=percent_test, ignore_warnings=ignore_warnings, save_splits_output=save_splits_output)

    return (SequenceDataset(data, training_samples, encoder),
            SequenceDataset(data, val_samples, encoder),
            SequenceDataset(data, test_samples, encoder))


def cross_validation_folds(samples: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Randomly divide sample indices into cross-validation folds

    Parameters
    ----------
    samples : numpy int array
        The sample indices to divide
    n_folds : int
        Number of folds

    Returns
    -------
    list of tuples of numpy int arrays
        For each fold, (training indices, held-out indices). Every sample is
        held out in exactly one fold.
    """

    shuffled = np.array(samples, copy=True)
    np.random.shuffle(shuffled)
    held_out_sets = np.array_split(shuffled, n_folds)

    folds = []
    for i, held_out in enumerate(held_out_sets):
        training = np.concatenate([held_out_sets[j] for j in range(n_folds) if j != i])
        folds.append((np.sort(training), np.sort(held_out)))
    return folds


def split_data_cv(data_file: str, task: Task, encoder: SequenceEncoder | None = None, exclude_seq_id: bool = False,
                  split_file: str | None = None, percent_val: float = 0.15, percent_test: float = 0.15,
                  n_folds: int = 5, ignore_warnings: bool = False, save_splits_output: str | None = None
                  ) -> tuple[list[tuple[SequenceDataset, SequenceDataset]], SequenceDataset, SequenceDataset, SequenceDataset]:
    """Divide a datafile into training, validation and test sets plus cross-validation folds

    The cross-validation folds are made from the combined training and
    validation data; the test set is never part of any fold.

    Parameters
    ----------
    data_file, task, encoder, exclude_seq_id, split_file, percent_val, percent_test, ignore_warnings, save_splits_output
        See split_data()
    n_folds : int, optional
        Number of cross-validation folds (default 5)

    Returns
    -------
    tuple
        (list of (fold training set, fold held-out set) pairs, training set,
        validation set, test set)
    """

    data, training_samples, val_samples, test_samples = load_and_split(
        data_file, task, exclude_seq_id=exclude_seq_id, split_file=split_file, percent_val=percent_val,
        percent_test=percent_test, ignore_warnings=ignore_warnings, save_splits_output=save_splits_output)

    cv_samples = np.append(training_samples, val_samples)
    if n_folds > len(cv_samples):
        raise ValueError(f'Cannot make {n_folds} cross-validation folds from {len(cv_samples)} training and '
                         'validation sequences')

    cv_sets = []
    for fold_training, fold_held_out in cross_validation_folds(cv_samples, n_folds):
        cv_sets.append((SequenceDataset(data, fold_training, encoder), SequenceDataset(data, fold_held_out, encoder)))

    return (cv_sets, SequenceDataset(data, training_samples, encoder), SequenceDataset(data, val_samples, encoder),
            SequenceDataset(data, test_samples, encoder))
