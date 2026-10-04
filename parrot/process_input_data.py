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
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from parrot import parrot_exceptions
from parrot.encode_sequence import OneHotEncoder, SequenceEncoder
from parrot.tasks import VALID_DATATYPES, Task
from parrot.tools import dataset_warnings

# Text used in residue-mapped datafiles for a residue with no label (e.g. a residue
# not observed in a structure). Any capitalisation is accepted.
MISSING_LABEL_TEXT = 'nan'

# Class label given to padded positions and to residues with missing labels in
# residue-classification data. This matches the default ignore_index of
# torch.nn.CrossEntropyLoss, so these positions never contribute to the loss.
IGNORED_CLASS_LABEL = -100

# Default fractions of the data in the training, validation and test sets
DEFAULT_SET_FRACTIONS = (0.7, 0.15, 0.15)

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


def parse_lines(lines: list[list[str]], datatype: str) -> list[Sample]:
    """Convert the fields of each datafile line into [id, sequence, value(s)]

    For residue-mapped data, individual residue values may be 'nan' to mark a
    residue with no label; these are kept as NaN.

    Parameters
    ----------
    lines : list of list of str
        Output from read_tsv_raw()
    datatype : str
        'residues' (a value per residue) or 'sequence' (a single value)

    Returns
    -------
    list
        One [id, sequence, value(s)] list per line, where value(s) is a float
        (datatype='sequence') or a numpy array of floats (datatype='residues')

    Raises
    ------
    IOExceptionParrot
        For the first line that cannot be parsed, saying which data line it is
        (counting only non-empty, non-comment lines), what is wrong with it, and
        what it contains
    """

    if datatype not in VALID_DATATYPES:
        raise ValueError(f"Invalid datatype '{datatype}': must be one of {', '.join(VALID_DATATYPES)}")

    data: list[Sample] = []
    for line_number, fields in enumerate(lines, start=1):
        try:
            data.append(parse_line(fields, datatype))
        except ValueError as error:
            message = f"Data line {line_number} is not valid {datatype}-mapped data: {error}\n  {' '.join(fields)}"
            raise parrot_exceptions.IOExceptionParrot(message) from error
    return data


def parse_line(fields: list[str], datatype: str) -> Sample:
    """Convert the fields of one datafile line into [id, sequence, value(s)]

    Parameters
    ----------
    fields : list of str
        The line split into fields: ID, sequence, then the value(s)
    datatype : str
        'residues' (a value per residue) or 'sequence' (a single value)

    Returns
    -------
    list
        [id, sequence, value(s)]: a float for sequence-mapped data, or a numpy
        array of floats (NaN for a missing label) for residue-mapped data

    Raises
    ------
    ValueError
        Saying what is wrong with the line
    """
    if len(fields) < 3:
        raise ValueError(f'expected an ID, a sequence and its value(s), but found {len(fields)} column(s)')

    sequence_id, sequence, values = fields[0], fields[1], fields[2:]
    if datatype == 'residues':
        return [sequence_id, sequence, parse_residue_values(values, sequence)]
    return [sequence_id, sequence, parse_sequence_value(values)]


def parse_residue_values(values: list[str], sequence: str) -> np.ndarray:
    """Convert a residue-mapped line's values to numbers: one per residue, 'nan' for a residue with no label

    Parameters
    ----------
    values : list of str
        The line's values
    sequence : str
        The line's sequence

    Returns
    -------
    numpy array of float

    Raises
    ------
    ValueError
        If a value is not a number, the number of values differs from the
        number of residues, a value is infinite, or every value is missing
    """
    numbers = np.array(values, dtype=float)
    if len(numbers) != len(sequence):
        raise ValueError(f'inconsistent number of residue values and residues ({len(numbers)} values for a '
                         f'sequence of {len(sequence)} residues)')
    if np.any(np.isinf(numbers)):
        raise ValueError('residue values must be finite numbers or nan (missing), not inf')
    if np.all(np.isnan(numbers)):
        raise ValueError('every residue label of this sequence is missing (nan)')
    return numbers


def parse_sequence_value(values: list[str]) -> float:
    """Convert a sequence-mapped line's value to a number

    Parameters
    ----------
    values : list of str
        The line's values (exactly one is expected)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If there is not exactly one value, or it is not a finite number
    """
    # Extra columns usually mean a residue-mapped file was given with --datatype sequence
    if len(values) != 1:
        raise ValueError(f'expected one value per sequence but found {len(values)} (is this residue-mapped data?)')

    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError('sequence values must be finite numbers (no nan or inf)')
    return value


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
    IOExceptionParrot
        If the file has no data lines, a line cannot be parsed, or a class label
        is not an integer in the range [0, num_classes - 1]
    """

    lines = read_tsv_raw(tsvfile)
    if not lines:
        raise parrot_exceptions.IOExceptionParrot(f'Datafile {tsvfile} contains no data lines')

    # Add a dummy seqID if none are provided
    if exclude_seq_id:
        for line in lines:
            line.insert(0, '')

    data = parse_lines(lines, task.datatype)

    # The warnings run before class labels are converted, because they read missing labels as NaN
    if not ignore_warnings:
        dataset_warnings.warn_about_dataset(data, task)

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
    IOExceptionParrot
        If any label is not a whole number in the range [0, num_classes - 1].
        Labels are never silently truncated (e.g. 1.5 is not turned into 1).
    """

    labels = np.atleast_1d(np.asarray(sample[2], dtype=float))
    missing = np.isnan(labels)
    present = labels[~missing]

    if np.any(present != np.round(present)):
        raise parrot_exceptions.IOExceptionParrot(
            f'Invalid class label on entry {sample[0]}.\nClass labels must be integers, got {sample[2]}')

    if np.any(present < 0) or np.any(present >= task.num_classes):
        raise parrot_exceptions.IOExceptionParrot(
            f'Invalid class label on entry {sample[0]}.\nClass labels must be between 0 and {task.num_classes - 1}, '
            f'got {sample[2]}')

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

    @property
    def values(self) -> list[Any]:
        """The value(s) of every sequence (a number, or one per residue), in dataset order."""
        return [sample[2] for sample in self.data]


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


class Batch(NamedTuple):
    """A padded batch of sequences, as made by collate_batch()

    It unpacks like a tuple: names, vectors, targets, lengths = batch.

    Attributes
    ----------
    names : list of str
        The name of each sequence
    vectors : torch.Tensor
        Zero-padded encoded sequences (see pad_sequence_vectors())
    targets : torch.Tensor
        Target values or class labels (see pad_targets())
    lengths : torch.Tensor
        True (unpadded) length of each sequence
    """

    names: list[str]
    vectors: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor


def collate_batch(batch: list[tuple[str, torch.Tensor, Any]], task: Task) -> Batch:
    """Collate (name, encoded sequence, value) items into a padded batch

    Parameters
    ----------
    batch : list
        Items from a SequenceDataset
    task : Task
        The task the data is for (decides how targets are formatted)

    Returns
    -------
    Batch
        The names, zero-padded sequence vectors, targets and true lengths
    """

    names = [item[0] for item in batch]
    padded_seq_vectors, lengths = pad_sequence_vectors([item[1] for item in batch])
    targets = pad_targets([item[2] for item in batch], task, padded_seq_vectors.shape[1])
    return Batch(names, padded_seq_vectors, targets, lengths)


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

def vector_split(values: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the values of a vector at random into two non-overlapping parts

    Parameters
    ----------
    values : numpy array
        The (unique) values to divide, e.g. sample indices
    fraction : float
        Fraction of the values in the first part, between 0 and 1

    Returns
    -------
    numpy array
        A sorted, random `fraction` of the values (rounding the number up)
    numpy array
        The rest of the values, sorted
    """

    # Subtract a tiny tolerance before rounding up so that floating-point noise
    # (e.g. 1 - 0.1 - 0.3 = 0.6000000000000001) cannot push an exact size up by one
    first_size = math.ceil(fraction * len(values) - SPLIT_SIZE_TOLERANCE)
    first = np.random.choice(values, size=first_size, replace=False)
    first.sort()
    rest = np.setdiff1d(values, first, assume_unique=True)
    return first, rest


def random_split_indices(num_samples: int, val_fraction: float,
                         test_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly assign sample indices to training, validation and test sets

    Parameters
    ----------
    num_samples : int
        Total number of samples
    val_fraction : float
        Fraction of samples for the validation set
    test_fraction : float
        Fraction of samples for the test set. The training set gets the rest.

    Returns
    -------
    tuple of numpy int arrays
        Sorted (training, validation, test) indices
    """

    train_fraction = 1 - val_fraction - test_fraction
    training_samples, val_test_samples = vector_split(np.arange(num_samples), train_fraction)

    # Then divide what is left between the validation and test sets
    val_samples, test_samples = vector_split(val_test_samples, val_fraction / (val_fraction + test_fraction))

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
                   percent_val: float = DEFAULT_SET_FRACTIONS[1], percent_test: float = DEFAULT_SET_FRACTIONS[2],
                   ignore_warnings: bool = False,
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
        Fractions (not percentages; the names are kept from earlier versions of
        PARROT) of the data assigned to the validation and test sets when no
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
               split_file: str | None = None, percent_val: float = DEFAULT_SET_FRACTIONS[1],
               percent_test: float = DEFAULT_SET_FRACTIONS[2], ignore_warnings: bool = False,
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
                  split_file: str | None = None, percent_val: float = DEFAULT_SET_FRACTIONS[1],
                  percent_test: float = DEFAULT_SET_FRACTIONS[2], n_folds: int = 5, ignore_warnings: bool = False,
                  save_splits_output: str | None = None
                  ) -> tuple[list[tuple[SequenceDataset, SequenceDataset]], SequenceDataset, SequenceDataset,
                             SequenceDataset]:
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
