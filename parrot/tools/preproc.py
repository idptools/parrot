"""
Module with functions used in parrot-preprocess

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import math
from collections import Counter, defaultdict

import numpy as np

from parrot import process_input_data as pid

# The 20 standard amino acids (the residues every PARROT encoding understands)
STANDARD_AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')

# How identical sequences are deduplicated (see deduplicate())
DEDUPLICATION_MODES = ('merge', 'first', 'drop')


# .................................................................
#
#
class ProcessingReport:
    """Records what parrot-preprocess did, for printing and for the report file

    Attributes
    ----------
    lines : list of str
        Report lines, in order
    verbose : bool
        Print each line as it is added
    """

    def __init__(self, verbose=False):
        self.lines = []
        self.verbose = verbose

    def add(self, text=''):
        """Add a line to the report."""
        self.lines.append(text)
        if self.verbose:
            print(text)

    def section(self, title):
        """Start a new titled section."""
        self.add()
        self.add(title)
        self.add('-' * len(title))

    def write(self, path):
        """Write the report to a file."""
        with open(path, 'w') as fh:
            fh.write('\n'.join(self.lines) + '\n')


# .................................................................
#
#
def clean_sequences(lines, report, uppercase=False, remove_nonstandard=False, min_length=None, max_length=None):
    """
    Optionally upper-case sequences, and remove sequences with non-standard residues or outside a length range.

    Parameters
    -----------
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
    ----------
    list
        The kept lines
    """
    if uppercase:
        changed = sum(1 for line in lines if line[1] != line[1].upper())
        lines = [[line[0], line[1].upper()] + line[2:] for line in lines]
        report.add(f'Converted {changed:,} sequences to upper case')

    if remove_nonstandard:
        letters = Counter()
        kept = []
        for line in lines:
            unknown = set(line[1]) - STANDARD_AMINO_ACIDS
            if unknown:
                letters.update(unknown)
            else:
                kept.append(line)
        report.add(f'Removed {len(lines) - len(kept):,} sequences containing residues other than the 20 standard '
                   f'amino acids' + (f' ({", ".join(sorted(letters))})' if letters else ''))
        lines = kept

    for limit, too_far, words in ((min_length, lambda n: n < min_length, 'shorter than'),
                                  (max_length, lambda n: n > max_length, 'longer than')):
        if limit is not None:
            kept = [line for line in lines if not too_far(len(line[1]))]
            report.add(f'Removed {len(lines) - len(kept):,} sequences {words} {limit} residues')
            lines = kept
    return lines


def format_number(value):
    """Format a value for a datafile: whole numbers without a decimal point, missing values as 'nan'."""
    if np.isnan(value):
        return pid.MISSING_LABEL_TEXT
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def deduplicate(lines, mode, datatype, classification, report):
    """
    Deal with sequences that appear more than once, keeping one line per unique sequence (or none).

    Parameters
    -----------
    lines : list
        Parsed datafile lines ([id, sequence, value(s)...])
    mode : str
        'merge': combine the copies' values. Regression values are averaged
        (residue by residue for residue data, ignoring missing labels).
        Classification copies are kept (once) only if their labels agree; for
        residue data each residue keeps its label if every labelled copy
        agrees, and otherwise becomes missing.
        'first': keep the first copy as it is.
        'drop': remove every copy (we cannot know which is right).
    datatype : str
        'sequence' or 'residues'
    classification : bool
        True if the values are class labels
    report : ProcessingReport
        Where to record what was done

    Returns
    ----------
    list
        One line per kept sequence, in order of first appearance, with the first copy's ID
    """
    if mode not in DEDUPLICATION_MODES:
        raise ValueError(f"Unknown deduplication mode '{mode}': choose from {', '.join(DEDUPLICATION_MODES)}")

    copies_of = defaultdict(list)
    for line in lines:
        copies_of[line[1]].append(line)
    duplicated = {sequence: copies for sequence, copies in copies_of.items() if len(copies) > 1}
    report.add(f'{len(duplicated):,} sequences appear more than once '
               f'({sum(len(copies) for copies in duplicated.values()) - len(duplicated):,} extra copies)')

    merged = {}
    conflicting = disagreeing_residues = 0
    spreads = []
    for sequence, copies in duplicated.items():
        if mode == 'first':
            merged[sequence] = copies[0]
            continue
        if mode == 'drop':
            merged[sequence] = None
            continue

        first = copies[0]
        if datatype == 'sequence':
            values = [copy[2] for copy in copies]
            if classification:
                if len(set(values)) > 1:
                    conflicting += 1
                    merged[sequence] = None
                else:
                    merged[sequence] = first
            else:
                numbers = np.array(values, dtype=float)
                spreads.append(float(np.nanmax(numbers) - np.nanmin(numbers)) if np.any(~np.isnan(numbers)) else 0.0)
                mean = np.nanmean(numbers) if np.any(~np.isnan(numbers)) else np.nan
                merged[sequence] = [first[0], sequence, format_number(mean)]
        else:
            if any(len(copy) != len(first) for copy in copies):
                raise ValueError(f'Copies of {first[0]} have different numbers of values')
            if classification:
                labels = []
                for position in range(2, len(first)):
                    present = {copy[position] for copy in copies if not is_missing_label(copy[position])}
                    if len(present) > 1:
                        disagreeing_residues += 1
                        labels.append(pid.MISSING_LABEL_TEXT)
                    else:
                        labels.append(present.pop() if present else pid.MISSING_LABEL_TEXT)
            else:
                numbers = np.array([copy[2:] for copy in copies], dtype=float)
                labelled = ~np.isnan(numbers)
                with np.errstate(invalid='ignore'):
                    means = np.where(labelled.any(axis=0), np.nansum(numbers, axis=0) / labelled.sum(axis=0), np.nan)
                labels = [format_number(value) for value in means]
            merged[sequence] = [first[0], sequence] + labels

    kept = []
    for line in lines:
        sequence = line[1]
        if sequence not in duplicated:
            kept.append(line)
        elif sequence in merged:
            if merged[sequence] is not None:
                kept.append(merged[sequence])
            del merged[sequence]

    if duplicated:
        if mode == 'first':
            report.add('Kept the first copy of each')
        elif mode == 'drop':
            report.add('Removed every copy of each (we cannot know which is right)')
        elif datatype == 'sequence' and classification:
            report.add(f'Removed {conflicting:,} of them whose copies had different labels; kept one copy of the rest')
        elif datatype == 'sequence':
            report.add(f'Merged the copies of each by averaging their values (median range between copies '
                       f'{np.median(spreads):.3g}, largest {np.max(spreads):.3g})')
        elif classification:
            report.add(f'Merged the copies of each residue by residue; {disagreeing_residues:,} residues whose '
                       'labelled copies disagreed became missing (nan)')
        else:
            report.add('Merged the copies of each residue by residue, averaging their labelled values')
    report.add(f'{len(kept):,} lines remain')
    return kept


def write_dataset(lines, splits, datafile, split_file, report):
    """
    Write a PARROT datafile and, if sets were assigned, the matching split-file.

    With sets, lines are written in the order training, validation, test, and the
    split-file lists the line numbers of each set, so the data can be trained on
    with parrot-train --split.

    Parameters
    -----------
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
        order = [i for name in ('train', 'val', 'test') for i in range(len(lines)) if splits[i] == name]
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

    indices = {name: np.array([i for i, split in enumerate(splits) if split == name], dtype=int)
               for name in ('train', 'val', 'test')}
    pid.write_split_file(split_file, indices['train'], indices['val'], indices['test'])
    report.add(f'Wrote {len(lines):,} lines to {datafile} (training {len(indices["train"]):,}, validation '
               f'{len(indices["val"]):,}, test {len(indices["test"]):,}) and the matching split-file {split_file}')


# .................................................................
#
#
def is_missing_label(label):
    """
    True if a label marks a missing value: 'nan' in any capitalisation, exactly as
    PARROT reads datafiles (residues with a missing label are ignored in training).

    Parameters
    -----------
    label : str
        A class label or value as written in a datafile

    Returns
    ----------
    bool
    """
    try:
        return math.isnan(float(label))
    except ValueError:
        return False


# .................................................................
#
#
def recode_classes(lines):
    """
    Function which takes a set of data read in (lines) and converts it into a PARROT-compliant set of 
    lines files where classes start at 0. This works for both residue and sequence datatypes.

    If every label is a number, classes are ordered by value and labels with the same value
    (e.g. '1' and '1.0') are the same class; otherwise labels are ordered alphabetically.
    Missing labels ('nan') are not a class and are written out unchanged.

    Parameters
    -----------
    lines : list
        The lines list is a list of lists, where each sublist corresponds to a line in a parsed
        datafile. If the dataype is sequence then each sublist has three elements (id, sequence, class/score),
        while if the datatype is residues then each sublists has n elements (id, sequence, per-residue scores or 
        classes for each residue).


    Returns
    -------------
    list
        Returns a revised lines list, where sequence or residue classes are numbered consecutively
        from 0.

    dict
        Maps each label found in the input to its new class number
    
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

    n_classes = len(set(class_mapping.values()))
    if n_classes > 100:
        print(f'WARNING: {n_classes} individual classes found. This is probably not a good classification problem...')

    # build a new lines list, where the sequence or residue classes are mapped via class_mapping
    new_lines = []
    for line in lines:
        new_labels = [label if is_missing_label(label) else str(class_mapping[label]) for label in line[2:]]
        new_lines.append([line[0], line[1]] + new_labels)

    return (new_lines, class_mapping)



# .................................................................
#
#
def remove_line_duplicates(lines, verbose=False):
    """
    Function that removes true duplicates and warns about potential psuedoduplicates if
    verbose is set to true.

    Parameters
    -------------
    lines : list
        Lines list file generated by parrot.process_input_data.__read_tsv(<filename>).

    verbose : bool
        Flag which tells the function how loud to be. Default = False.

    Returns
    ------------
    list
        Returns an updated lines file that has had all the true duplicate lines removed (i.e.
        lines where the id, sequence, and data were all identical. 
    

    """

    if verbose:
        print('---------------------------------------------------------------')
        print('Performing duplicate checking/removal --remove-line-duplicates ')
        print('---------------------------------------------------------------')

    found = {}
    all_sequences = set([])
    duplicate_count = 0
    seq_duplicate_count = 0
    seq_duplicate_count_full = 0

    # for each line
    for i in lines:

        # this part ONLY keeps track of sequences. Note these are not removed
        # because which one should you remove!
        if i[1] in all_sequences:
            seq_duplicate_count_full = seq_duplicate_count_full + 1
        else:
            all_sequences.add(i[1])

        # if the ID code is has not yet been seen
        if i[0] not in found:
            found[i[0]] = [i]

        # else if the ID code (first column in datafile) HAS been seen
        else:
            
            # set the unique flag to true
            this_is_a_duplicate = False

            # for EVERY time it was seen (hits is a previous line)
            for hits in found[i[0]]:

                # if the sequence ALSO MATCHED
                if hits[1] == i[1]:
                    seq_duplicate_count = seq_duplicate_count + 1
                    
                    if hits[2:] == i[2:]:
                        duplicate_count = duplicate_count + 1
                        #
                        this_is_a_duplicate = True

            # if we get here then this was 
            if this_is_a_duplicate is False:
                found[i[0]].append(i)

    if verbose:
        print(f'   Found {duplicate_count} full duplicate lines in this file. These have been removed.\n')
        print(f'   Also found {seq_duplicate_count} ID-sequence duplicates in this file (same ID, same sequence, different data.\n   If any of these were full duplicates (likely) they will have been removed.\n')
        print(f'   Found {seq_duplicate_count_full} sequence duplicates in this file (same sequence with different ID and different data).\n   If any of these were full duplicates (likely) they will have been removed.\n')

    new_lines = []
    for id_key in found:
        for x in found[id_key]:
            new_lines.append(x)

    return new_lines
        
                




