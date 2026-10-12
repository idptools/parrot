"""
The steps of parrot-preprocess that clean, deduplicate and recode a datafile,
and write the processed datafile.

Every function here works on parsed datafile lines ([ID, sequence, value(s)],
all as text, as read by process_input_data.read_tsv_raw()) and records what it
did in a ProcessingReport rather than printing it. Splitting is in splitting.py
and class balancing in class_balancing.py.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from parrot import process_input_data as pid
from parrot.sequence_encoder import STANDARD_AMINO_ACIDS
from parrot.tools.splitting import SET_NAMES

# A datafile line, split into fields: [ID, sequence, value, value, ...]
Line = list[str]

# How identical sequences are deduplicated (see deduplicate())
DEDUPLICATION_MODES = ('merge', 'first', 'drop')

# Whole numbers smaller than this are written without a decimal point (larger ones lose precision as integers)
LARGEST_EXACT_WHOLE_NUMBER = 1e15


class ProcessingReport:
    """Records what parrot-preprocess did, for printing and for the report file

    Attributes
    ----------
    lines : list of str
        Report lines, in order
    verbose : bool
        Print each line as it is added
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Parameters
        ----------
        verbose : bool, optional
            Print each line as it is added
        """
        self.lines: list[str] = []
        self.verbose = verbose

    def add(self, text: str = '') -> None:
        """Add a line to the report."""
        self.lines.append(text)
        if self.verbose:
            print(text)

    def section(self, title: str) -> None:
        """Start a new titled section."""
        self.add()
        self.add(title)
        self.add('-' * len(title))

    def write(self, path: str) -> None:
        """Write the report to a file."""
        with open(path, 'w') as fh:
            fh.write('\n'.join(self.lines) + '\n')


# .................................................................
# Cleaning sequences

def clean_sequences(lines: list[Line], report: ProcessingReport, uppercase: bool = False,
                    remove_nonstandard: bool = False, min_length: int | None = None,
                    max_length: int | None = None) -> list[Line]:
    """Optionally upper-case sequences, and remove sequences with non-standard residues or outside a length range

    Parameters
    ----------
    lines : list
        Parsed datafile lines ([id, sequence, value(s)...])
    report : ProcessingReport
        Where to record what was changed
    uppercase : bool
        Convert every sequence to upper case
    remove_nonstandard : bool
        Remove sequences containing anything other than the 20 standard amino acids
    min_length, max_length : int, optional
        Remove sequences shorter or longer than these

    Returns
    -------
    list
        The kept lines
    """
    if uppercase:
        changed = sum(1 for line in lines if line[1] != line[1].upper())
        lines = [[line[0], line[1].upper()] + line[2:] for line in lines]
        report.add(f'Converted {changed:,} sequences to upper case')

    if remove_nonstandard:
        lines = remove_nonstandard_sequences(lines, report)

    if min_length is not None:
        kept = [line for line in lines if len(line[1]) >= min_length]
        report.add(f'Removed {len(lines) - len(kept):,} sequences shorter than {min_length} residues')
        lines = kept

    if max_length is not None:
        kept = [line for line in lines if len(line[1]) <= max_length]
        report.add(f'Removed {len(lines) - len(kept):,} sequences longer than {max_length} residues')
        lines = kept
    return lines


def remove_nonstandard_sequences(lines: list[Line], report: ProcessingReport) -> list[Line]:
    """Remove sequences containing anything other than the 20 standard amino acids, reporting which letters

    Parameters
    ----------
    lines : list
        Parsed datafile lines
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines
    """
    standard = set(STANDARD_AMINO_ACIDS)
    letters: Counter[str] = Counter()
    kept = []
    for line in lines:
        unknown = set(line[1]) - standard
        if unknown:
            letters.update(unknown)
        else:
            kept.append(line)
    report.add(f'Removed {len(lines) - len(kept):,} sequences containing residues other than the 20 standard '
               f'amino acids' + (f' ({", ".join(sorted(letters))})' if letters else ''))
    return kept


# .................................................................
# Duplicates

def remove_line_duplicates(lines: list[Line]) -> list[Line]:
    """Remove lines that are exact copies of an earlier line (same ID, sequence and values)

    The first copy of each line is kept, and the lines stay in their input order.

    Parameters
    ----------
    lines : list
        Parsed datafile lines

    Returns
    -------
    list
        The kept lines
    """
    seen: set[tuple[str, ...]] = set()
    kept = []
    for line in lines:
        fields = tuple(line)
        if fields not in seen:
            seen.add(fields)
            kept.append(line)
    return kept


def count_repeated_sequences(lines: list[Line]) -> int:
    """How many sequences appear on more than one line

    Parameters
    ----------
    lines : list
        Parsed datafile lines

    Returns
    -------
    int
    """
    return sum(count > 1 for count in Counter(line[1] for line in lines).values())


@dataclass
class MergeSummary:
    """What merging the copies of duplicated sequences did, for the report (see deduplicate())

    Attributes
    ----------
    conflicting_sequences : int
        Sequence classification: sequences whose copies had different labels (removed)
    disagreeing_residues : int
        Residue classification: residues whose labelled copies disagreed (made missing)
    unlabelled_sequences : int
        Residue classification: sequences left with no labelled residue (removed)
    spreads : list of float
        Sequence regression: the range of each sequence's copies' values
    """

    conflicting_sequences: int = 0
    disagreeing_residues: int = 0
    unlabelled_sequences: int = 0
    spreads: list[float] = field(default_factory=list)


def deduplicate(lines: list[Line], mode: str, datatype: str, classification: bool,
                report: ProcessingReport) -> list[Line]:
    """Deal with sequences that appear more than once, keeping one line per unique sequence (or none)

    Parameters
    ----------
    lines : list
        Parsed datafile lines ([id, sequence, value(s)...])
    mode : str
        'merge': combine the copies' values. Regression values are averaged
        (residue by residue for residue data, ignoring missing labels).
        Classification copies are kept (once) only if their labels agree
        (labels that are the same number, such as '1' and '1.0', agree); for
        residue data each residue keeps its label if every labelled copy
        agrees, and otherwise becomes missing, and a sequence left with no
        labelled residue is removed.
        'first': keep the first copy as it is.
        'drop': remove every copy (we cannot know which is right).
    datatype : str
        'sequence' or 'residues'
    classification : bool
        True if the values are class labels
    report : ProcessingReport
        Where to record what was done

    Returns
    -------
    list
        One line per kept sequence, in order of first appearance, with the first copy's ID
    """
    if mode not in DEDUPLICATION_MODES:
        raise ValueError(f"Unknown deduplication mode '{mode}': choose from {', '.join(DEDUPLICATION_MODES)}")

    copies_of: dict[str, list[Line]] = defaultdict(list)
    for line in lines:
        copies_of[line[1]].append(line)
    duplicated = {sequence: copies for sequence, copies in copies_of.items() if len(copies) > 1}
    report.add(f'{len(duplicated):,} sequences appear more than once '
               f'({sum(len(copies) for copies in duplicated.values()) - len(duplicated):,} extra copies)')

    summary = MergeSummary()
    merged = {sequence: merge_copies(copies, mode, datatype, classification, summary)
              for sequence, copies in duplicated.items()}

    # Each duplicated sequence's merged line (if any) takes the place of its first copy
    kept = []
    for line in lines:
        sequence = line[1]
        if sequence not in duplicated:
            kept.append(line)
        elif sequence in merged:
            merged_line = merged.pop(sequence)
            if merged_line is not None:
                kept.append(merged_line)

    if duplicated:
        report.add(describe_merge(mode, datatype, classification, summary))
    report.add(f'{len(kept):,} lines remain')
    return kept


def merge_copies(copies: list[Line], mode: str, datatype: str, classification: bool,
                 summary: MergeSummary) -> Line | None:
    """The one line that replaces the copies of a duplicated sequence, or None to remove them all

    Parameters
    ----------
    copies : list
        Every line with this sequence, in order
    mode, datatype, classification
        See deduplicate()
    summary : MergeSummary
        Updated with what was done

    Returns
    -------
    list or None
    """
    if mode == 'first':
        return copies[0]
    if mode == 'drop':
        return None

    if datatype == 'sequence':
        return merge_sequence_labels(copies, summary) if classification else merge_sequence_values(copies, summary)

    if any(len(copy) != len(copies[0]) for copy in copies):
        raise ValueError(f'Copies of {copies[0][0]} have different numbers of values')
    return merge_residue_labels(copies, summary) if classification else merge_residue_values(copies)


def merge_sequence_labels(copies: list[Line], summary: MergeSummary) -> Line | None:
    """Sequence classification: keep the first labelled copy if every labelled copy has the same label, otherwise none

    Missing labels ('nan') are ignored, as they are when residue labels and regression values are merged; if no
    copy has a label, the first copy is kept.

    Parameters
    ----------
    copies : list
        Every line with this sequence
    summary : MergeSummary
        Counts the sequences whose copies disagree

    Returns
    -------
    list or None
    """
    labelled = [copy for copy in copies if not is_missing_label(copy[2])]
    if len({label_value(copy[2]) for copy in labelled}) > 1:
        summary.conflicting_sequences += 1
        return None
    return labelled[0] if labelled else copies[0]


def merge_sequence_values(copies: list[Line], summary: MergeSummary) -> Line:
    """Sequence regression: one line with the average of the copies' values (missing values ignored)

    Parameters
    ----------
    copies : list
        Every line with this sequence
    summary : MergeSummary
        Records the range of the copies' values

    Returns
    -------
    list
    """
    numbers = np.array([copy[2] for copy in copies], dtype=float)
    labelled = ~np.isnan(numbers)
    summary.spreads.append(float(np.nanmax(numbers) - np.nanmin(numbers)) if labelled.any() else 0.0)
    mean = np.nanmean(numbers) if labelled.any() else np.nan
    return [copies[0][0], copies[0][1], format_number(mean)]


def merge_residue_labels(copies: list[Line], summary: MergeSummary) -> Line | None:
    """Residue classification: each residue keeps its label if every labelled copy agrees, otherwise it becomes missing

    Parameters
    ----------
    copies : list
        Every line with this sequence (all with the same number of values)
    summary : MergeSummary
        Counts the residues whose labelled copies disagree, and the sequences left with no label

    Returns
    -------
    list or None
        None if no residue keeps a label (a line with no labels cannot be trained on), unless no copy had any
    """
    first = copies[0]
    labels = []
    for position in range(2, len(first)):
        # The first copy's text of each distinct label (labels that are the same number agree)
        present: dict[float | str, str] = {}
        for copy in copies:
            if not is_missing_label(copy[position]):
                present.setdefault(label_value(copy[position]), copy[position])
        if len(present) > 1:
            summary.disagreeing_residues += 1
            labels.append(pid.MISSING_LABEL_TEXT)
        else:
            labels.append(next(iter(present.values())) if present else pid.MISSING_LABEL_TEXT)

    labelled_before = any(not is_missing_label(label) for copy in copies for label in copy[2:])
    if labelled_before and all(is_missing_label(label) for label in labels):
        summary.unlabelled_sequences += 1
        return None
    return [first[0], first[1]] + labels


def merge_residue_values(copies: list[Line]) -> Line:
    """Residue regression: each residue gets the average of its labelled copies (missing if none is labelled)

    Parameters
    ----------
    copies : list
        Every line with this sequence (all with the same number of values)

    Returns
    -------
    list
    """
    numbers = np.array([copy[2:] for copy in copies], dtype=float)
    labelled = ~np.isnan(numbers)
    with np.errstate(invalid='ignore'):
        means = np.where(labelled.any(axis=0), np.nansum(numbers, axis=0) / labelled.sum(axis=0), np.nan)
    return [copies[0][0], copies[0][1]] + [format_number(value) for value in means]


def describe_merge(mode: str, datatype: str, classification: bool, summary: MergeSummary) -> str:
    """The report line describing how the copies of duplicated sequences were dealt with

    Parameters
    ----------
    mode, datatype, classification
        See deduplicate()
    summary : MergeSummary

    Returns
    -------
    str
    """
    if mode == 'first':
        return 'Kept the first copy of each'
    if mode == 'drop':
        return 'Removed every copy of each (we cannot know which is right)'
    if datatype == 'sequence' and classification:
        return (f'Removed {summary.conflicting_sequences:,} of them whose copies had different labels; kept one '
                'copy of the rest')
    if datatype == 'sequence':
        return (f'Merged the copies of each by averaging their values (median range between copies '
                f'{np.median(summary.spreads):.3g}, largest {np.max(summary.spreads):.3g})')
    if classification:
        return (f'Merged the copies of each residue by residue; {summary.disagreeing_residues:,} residues whose '
                'labelled copies disagreed became missing (nan), and '
                f'{summary.unlabelled_sequences:,} sequences left with no labelled residue were removed')
    return 'Merged the copies of each residue by residue, averaging their labelled values'


# .................................................................
# Labels

def is_missing_label(label: str) -> bool:
    """True if a label marks a missing value: 'nan' in any capitalisation, exactly as PARROT reads datafiles

    Residues with a missing label are ignored in training.

    Parameters
    ----------
    label : str
        A class label or value as written in a datafile

    Returns
    -------
    bool
    """
    try:
        return math.isnan(float(label))
    except ValueError:
        return False


def label_value(label: str) -> float | str:
    """What a class label stands for, for comparing labels: its number if it is one ('1' and '1.0' are the same
    class), otherwise its text

    Parameters
    ----------
    label : str

    Returns
    -------
    float or str
    """
    try:
        return float(label)
    except ValueError:
        return label


def labels_need_recoding(lines: list[Line]) -> bool:
    """True if any class label is not a whole number from 0 (as PARROT needs; see recode_classes())

    Parameters
    ----------
    lines : list
        Parsed datafile lines ([id, sequence, label(s)...])

    Returns
    -------
    bool
    """
    for line in lines:
        for label in line[2:]:
            value = label_value(label)
            if isinstance(value, str) or (not math.isnan(value) and not (value >= 0 and value.is_integer())):
                return True
    return False


def format_number(value: float) -> str:
    """Format a value for a datafile: whole numbers without a decimal point, missing values as 'nan'.

    Parameters
    ----------
    value : float

    Returns
    -------
    str
    """
    if np.isnan(value):
        return pid.MISSING_LABEL_TEXT
    if float(value).is_integer() and abs(value) < LARGEST_EXACT_WHOLE_NUMBER:
        return str(int(value))
    return repr(float(value))


def recode_classes(lines: list[Line]) -> tuple[list[Line], dict[str, int]]:
    """Renumber class labels as consecutive whole numbers from 0, as PARROT needs (residue or sequence data)

    If every label is a number, classes are ordered by value and labels with the same value
    (e.g. '1' and '1.0') are the same class; otherwise labels are ordered alphabetically.
    Missing labels ('nan') are not a class and are written out unchanged.

    Parameters
    ----------
    lines : list
        Parsed datafile lines ([id, sequence, label(s)...])

    Returns
    -------
    list
        The lines with each label replaced by its class number
    dict
        Maps each label found in the input to its class number
    """

    labels = {label for line in lines for label in line[2:] if not is_missing_label(label)}

    try:
        values = {label: float(label) for label in labels}
    except ValueError:
        values = None

    if values is not None:
        # numeric labels: one class per distinct value, in increasing order
        class_numbers = {value: number for number, value in enumerate(sorted(set(values.values())))}
        class_mapping = {label: class_numbers[values[label]]
                         for label in sorted(labels, key=lambda label: (values[label], label))}
    else:
        class_mapping = {label: number for number, label in enumerate(sorted(labels))}

    new_lines = []
    for line in lines:
        new_labels = [label if is_missing_label(label) else str(class_mapping[label]) for label in line[2:]]
        new_lines.append([line[0], line[1]] + new_labels)

    return new_lines, class_mapping


# .................................................................
# Writing

def write_class_mapping(path: str, mapping: dict[str, int]) -> None:
    """Write which class number each original label was given (see recode_classes())

    Parameters
    ----------
    path : str
        File to write
    mapping : dict
        Original label -> class number
    """
    with open(path, 'w') as fh:
        fh.write('File generated by parrot-preprocess\n\n')
        fh.write(f'File generated at {datetime.now()}\n')
        fh.write(f'Number of classes: {len(set(mapping.values()))}\n\n')
        for label, number in mapping.items():
            fh.write(f'{label} = {number}\n')


def write_dataset(lines: list[Line], splits: list[str] | None, datafile: str, split_file: str,
                  report: ProcessingReport) -> None:
    """Write a PARROT datafile and, if sets were assigned, the matching split-file

    With sets, lines are written in the order training, validation, test, and the
    split-file lists the line numbers of each set, so the data can be trained on
    with parrot-train --split.

    Parameters
    ----------
    lines : list
        Lines to write
    splits : list of str or None
        The set ('train', 'val' or 'test') of each line, or None to write the
        lines in order without a split-file
    datafile, split_file : str
        Where to write them
    report : ProcessingReport
        Where to record what was written
    """
    if splits is not None:
        order = [i for name in SET_NAMES for i in range(len(lines)) if splits[i] == name]
        if len(order) != len(lines):
            raise ValueError('Every line must be assigned to train, val or test')
        lines = [lines[i] for i in order]
        splits = [splits[i] for i in order]

    with open(datafile, 'w') as fh:
        for line in lines:
            fh.write(' '.join(line) + '\n')

    if splits is None:
        report.add(f'Wrote {len(lines):,} lines to {datafile}')
        return

    indices = {name: np.array([i for i, split in enumerate(splits) if split == name], dtype=int) for name in SET_NAMES}
    pid.write_split_file(split_file, indices['train'], indices['val'], indices['test'])
    report.add(f'Wrote {len(lines):,} lines to {datafile} (training {len(indices["train"]):,}, validation '
               f'{len(indices["val"]):,}, test {len(indices["test"]):,}) and the matching split-file {split_file}')
