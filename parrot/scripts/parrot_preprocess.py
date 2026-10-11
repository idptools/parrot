#!/usr/bin/env python
"""
Usage: $ parrot-preprocess raw_data_file processed_data_file <flags>

parrot-preprocess turns a PARROT datafile into one ready for training: it can
clean sequences, recode class labels, deal with duplicate sequences, split the
data into training, validation and test sets without leakage between them
(keeping homologous sequences in the same set, with MMseqs2), and balance the
classes of the training set. It writes the processed datafile, a matching
split-file (if the data were split) and a report of everything it did.

Each step is one function below, run in this order by main(): read, clean,
recode, deduplicate, split, balance, write.

For more information on usage, use the '-h' flag.

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from parrot import process_input_data
from parrot.matrices.data import read_matrix_file
from parrot.package_version import PARROT_VERSION
from parrot.reproducibility import random_generator, set_random_seed
from parrot.scripts import arguments
from parrot.scripts.examples import add_examples_argument
from parrot.scripts.common import COMMAND_LINE_DATATYPES, LEGACY_NOTE
from parrot.tasks import Task
from parrot.tools import class_balancing, kmer_similarity, mmseqs_search, preproc, splitting
from parrot.tools.preproc import Line, ProcessingReport

PROGRAM_NAME = 'parrot-preprocess'

# The two kinds of data, as given with --mode
MODES = ('classification', 'regression')

# How the data can be split into sets (see --split)
SPLIT_METHODS = ('homology', 'kmer', 'random')

# Allowed range of MMseqs2's search sensitivity (-s)
MIN_MMSEQS_SENSITIVITY = 1.0
MAX_MMSEQS_SENSITIVITY = 7.5

# Recoding to more classes than this probably means the data are not really classes
MANY_CLASSES = 100


# .................................................................
# Arguments

def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-preprocess

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME, description='parrot-preprocess cleans, deduplicates, splits (without leakage between sets) and class-balances '
                    'a PARROT datafile, writing a processed datafile, a matching split-file and a report.')
    add_examples_argument(parser, PROGRAM_NAME)
    add_input_arguments(parser)
    add_cleaning_arguments(parser)
    add_duplicate_arguments(parser)
    add_splitting_arguments(parser)
    add_balancing_arguments(parser)
    add_matrix_arguments(parser)
    return parser


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """The files, the kind of data, and general options."""
    parser.add_argument('input_data_file', help="the datafile to process: '<ID> <sequence> <value(s)>' on each line "
                                                "(for matrix data, '<ID> <sequence>')")
    parser.add_argument('output_data_file', help='where to write the processed datafile (an existing file is '
                                                 'overwritten)')
    parser.add_argument('-d', '--datatype', required=True, choices=COMMAND_LINE_DATATYPES,
                        help="REQUIRED. Format of the datafile: 'sequence' (one value per sequence), 'residues' (one "
                             "value per residue) or 'matrix' (a matrix per sequence, in a separate .npz file)")
    parser.add_argument('-m', '--mode', required=True, choices=MODES,
                        help="REQUIRED. 'classification' (the values are class labels) or 'regression'")
    parser.add_argument('--delimiter', type=str, metavar='character',
                        help='the character separating the columns of the datafile (default: any whitespace)')
    parser.add_argument('-v', '--verbose', action='store_true', help='print the report of what was done as it happens')
    parser.add_argument('--seed', type=int, default=None, metavar='seed',
                        help='random seed, for a reproducible split and class balancing (default: not seeded)')


def add_cleaning_arguments(parser: argparse.ArgumentParser) -> None:
    """Cleaning sequences, and recoding class labels."""
    cleaning = parser.add_argument_group('cleaning sequences')
    cleaning.add_argument('--uppercase', action='store_true', help='convert every sequence to upper case')
    cleaning.add_argument('--remove-nonstandard', dest='remove_nonstandard', action='store_true',
                          help='remove sequences containing anything other than the 20 standard amino acids')
    cleaning.add_argument('--min-length', dest='min_length', type=int, metavar='residues',
                          help='remove sequences shorter than this')
    cleaning.add_argument('--max-length', dest='max_length', type=int, metavar='residues',
                          help='remove sequences longer than this')

    parser.add_argument('--recode-classes', action='store_true',
                        help='classification: recode the class labels, whatever they are (e.g. text), as 0, 1, ..., '
                             'n-1 for n classes, keeping missing labels (nan); the mapping is written to '
                             '<output>_class_mapping.txt')


def add_duplicate_arguments(parser: argparse.ArgumentParser) -> None:
    """Dealing with duplicate lines and sequences."""
    duplicates = parser.add_argument_group('duplicate sequences')
    duplicates.add_argument('--deduplicate', choices=preproc.DEDUPLICATION_MODES,
                            help="keep one line per unique sequence: 'merge' averages the copies' values (regression; "
                                 "residue by residue for residue data) or keeps one copy if their labels agree "
                                 "(classification; residues whose copies disagree become nan); 'first' keeps the "
                                 "first copy; 'drop' removes every copy")
    duplicates.add_argument('--remove-line-duplicates', action='store_true',
                            help='remove lines that are exact copies of an earlier line (same ID, sequence and values)')
    duplicates.add_argument('--remove-sequence-duplicates', action='store_true',
                            help='remove every copy of any sequence that appears more than once (same as '
                                 '--deduplicate drop)')


def add_splitting_arguments(parser: argparse.ArgumentParser) -> None:
    """Splitting into training, validation and test sets."""
    defaults = mmseqs_search.HomologyCriteria()
    default_fractions = ' '.join(str(fraction) for fraction in process_input_data.DEFAULT_SET_FRACTIONS)
    group = parser.add_argument_group('splitting into training, validation and test sets')
    group.add_argument('--split', dest='split_method', choices=SPLIT_METHODS,
                       help="write a split-file for the output, keeping similar sequences in the same set so no set "
                            "contains a sequence similar to one in another (records similar to a higher-priority set "
                            "are removed, test > validation > training). 'homology' finds homologues with MMseqs2 "
                            "(most sensitive; MMseqs2 must be installed); 'kmer' uses PARROT's own k-mer search, which "
                            "needs nothing extra but only finds closer relatives (most at >= 40-50%% identity); "
                            "'random' splits at random, keeping only identical sequences together")
    group.add_argument('--set-fractions', dest='set_fractions', nargs=3, type=float,
                       default=process_input_data.DEFAULT_SET_FRACTIONS, metavar=('train', 'val', 'test'),
                       help='target proportions of the training, validation and test sets '
                            f'(default {default_fractions})')
    group.add_argument('--groups', metavar='file',
                       help='file of "<sequence ID> <group>" lines: records in the same group (e.g. overlapping tiles '
                            'of one protein) are always put in the same set')
    group.add_argument('--homology-evalue', dest='homology_evalue', type=float, metavar='evalue',
                       help='homology and kmer: a similarity with at most this E-value links two sequences '
                            f'(default {defaults.max_evalue:g})')
    group.add_argument('--homology-coverage', dest='homology_coverage', type=float, metavar='fraction',
                       help='homology and kmer: ...if it covers at least this fraction of the shorter sequence '
                            f'(default {defaults.min_coverage:g})')
    group.add_argument('--no-masking', dest='no_masking', action='store_true',
                       help='homology and kmer: do not mask low-complexity regions, so similarity in them also links '
                            'sequences (even more stringent)')
    group.add_argument('--mmseqs', metavar='path', help='homology: MMseqs2 executable (default: mmseqs on the PATH)')
    group.add_argument('--threads', type=int, metavar='threads', help='homology: threads for MMseqs2 (default: every CPU)')
    group.add_argument('--mmseqs-sensitivity', dest='mmseqs_sensitivity', type=float, metavar='sensitivity',
                       help=f'homology: MMseqs2 search sensitivity, {MIN_MMSEQS_SENSITIVITY:g} to '
                            f'{MAX_MMSEQS_SENSITIVITY:g} (default {defaults.sensitivity:g}, the most sensitive)')
    group.add_argument('--kmer-size', dest='kmer_size', type=int, metavar='residues',
                       help=f'kmer: length of the shared words, {kmer_similarity.MIN_KMER_SIZE} to '
                            f'{kmer_similarity.MAX_KMER_SIZE} (default {kmer_similarity.KmerCriteria.kmer_size}; '
                            'smaller finds more distant relatives but is slower and more easily misled by chance '
                            'matches)')


def add_balancing_arguments(parser: argparse.ArgumentParser) -> None:
    """Class balancing (classification only)."""
    group = parser.add_argument_group('class balancing (classification only)')
    group.add_argument('--class-balance', action='store_true',
                       help='down-sample the more common classes so classes are balanced (only the training set, if '
                            'the data are split); for residue data, the balance after each removal is written to '
                            '<output>_class_balance_trace.csv')
    group.add_argument('--cb-block-size', type=int, default=None, metavar='sequences',
                       help='residue data: the number of randomly drawn sequences considered for each removal '
                            f'(default {class_balancing.DEFAULT_BLOCK_SIZE}; larger makes better choices, more slowly)')
    group.add_argument('--cb-max-fraction-removed', type=float, default=class_balancing.DEFAULT_MAX_FRACTION_REMOVED,
                       metavar='fraction',
                       help='the largest fraction of the sequences balancing may remove '
                            f'(default {class_balancing.DEFAULT_MAX_FRACTION_REMOVED})')
    group.add_argument('--cb-balance-threshold', type=float, default=class_balancing.DEFAULT_BALANCE_THRESHOLD,
                       metavar='ratio',
                       help='the data count as balanced once the most common class has at most this many times as '
                            f'many members as the least common (default {class_balancing.DEFAULT_BALANCE_THRESHOLD})')
    group.add_argument('--cb-shuffle-off', action='store_true',
                       help=f'no longer has any effect: balancing always keeps the input order {LEGACY_NOTE}')


def add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    """Matrix data (-d matrix): the datafile holds '<ID> <sequence>' lines, and the values are in a matrix file."""
    group = parser.add_argument_group('matrix data (-d matrix)')
    group.add_argument('--matrix-file', dest='matrix_file', metavar='file.npz',
                       help="the NumPy .npz file of each sequence's matrix, named by its sequence ID: if given, the "
                            'output is checked against it (the file itself is not changed, and works with the output '
                            'as it is, since every sequence keeps its ID)')
    group.add_argument('--matrix-width', dest='matrix_width', type=int, metavar='N',
                       help='every matrix has N columns (default: one per residue); used when checking the output')


def check_arguments(args: argparse.Namespace) -> None:
    """Check the options before reading any data

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments

    Raises
    ------
    ValueError
        If the options are invalid or inconsistent
    RuntimeError
        If --split homology was asked for but MMseqs2 cannot be found
    FileNotFoundError
        If the input datafile does not exist
    """
    arguments.check_file_exists(args.input_data_file, 'Datafile')
    if args.recode_classes and args.mode != 'classification':
        raise ValueError('--recode-classes only applies to classification data (-m classification)')
    if args.class_balance and args.mode != 'classification':
        raise ValueError('Class balancing (--class-balance) only applies to classification data '
                         '(--mode classification)')
    if args.remove_line_duplicates and args.remove_sequence_duplicates:
        raise ValueError('Give either --remove-line-duplicates or --remove-sequence-duplicates, not both')
    if args.remove_sequence_duplicates and args.deduplicate is not None:
        raise ValueError('--remove-sequence-duplicates is the same as --deduplicate drop; give one or the other')

    for name, value in (('--min-length', args.min_length), ('--max-length', args.max_length)):
        if value is not None and value < 1:
            raise ValueError(f'{name} must be at least 1, got {value}')
    if args.min_length is not None and args.max_length is not None and args.min_length > args.max_length:
        raise ValueError('--min-length cannot be larger than --max-length')

    check_options_match_split_method(args)
    if args.split_method is not None:
        check_split_settings(args)
    check_matrix_options(args)


def check_matrix_options(args: argparse.Namespace) -> None:
    """Check the options make sense for the datatype: matrix options only for matrix data, and no option that
    changes values for it (its values are in the matrix file, which preprocessing does not change)

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments

    Raises
    ------
    ValueError
        For the first option that does not
    FileNotFoundError
        If the matrix file does not exist
    """
    if args.datatype != 'matrix':
        for flag, given in (('--matrix-file', args.matrix_file is not None),
                            ('--matrix-width', args.matrix_width is not None)):
            if given:
                raise ValueError(f'{flag} only applies to matrix data (-d matrix)')
        return

    for flag, given in (('--recode-classes', args.recode_classes), ('--class-balance', args.class_balance),
                        ('--deduplicate merge', args.deduplicate == 'merge')):
        if given:
            raise ValueError(f'{flag} changes values, so it does not apply to matrix data (whose values are in the '
                             'matrix file); cleaning, removing duplicates and splitting do')
    if args.matrix_file is not None:
        arguments.check_file_exists(args.matrix_file, 'Matrix file')


def check_options_match_split_method(args: argparse.Namespace) -> None:
    """Check that options which only some splitting methods use are only given with those methods

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments

    Raises
    ------
    ValueError
        For the first option given without a method that uses it
    """
    similarity_methods = ('homology', 'kmer')
    fractions_given = tuple(args.set_fractions) != process_input_data.DEFAULT_SET_FRACTIONS
    for flag, given, methods in (('--groups', args.groups is not None, SPLIT_METHODS),
                                 ('--set-fractions', fractions_given, SPLIT_METHODS),
                                 ('--homology-evalue', args.homology_evalue is not None, similarity_methods),
                                 ('--homology-coverage', args.homology_coverage is not None, similarity_methods),
                                 ('--no-masking', args.no_masking, similarity_methods),
                                 ('--mmseqs', args.mmseqs is not None, ('homology',)),
                                 ('--threads', args.threads is not None, ('homology',)),
                                 ('--mmseqs-sensitivity', args.mmseqs_sensitivity is not None, ('homology',)),
                                 ('--kmer-size', args.kmer_size is not None, ('kmer',))):
        if given and args.split_method not in methods:
            raise ValueError(f'{flag} only applies to --split ' + ' or '.join(methods))


def check_split_settings(args: argparse.Namespace) -> None:
    """Check the values of the splitting options, and that MMseqs2 can be found if it is needed

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments

    Raises
    ------
    ValueError
        If a value is out of range
    RuntimeError
        If --split homology was asked for but MMseqs2 cannot be found
    """
    arguments.check_set_fractions(args.set_fractions)
    if args.groups is not None:
        arguments.check_file_exists(args.groups, 'Groups file')
    if args.split_method == 'homology':
        mmseqs_search.find_mmseqs(args.mmseqs)  # fail before doing any work if MMseqs2 is missing
    if args.homology_coverage is not None and not 0 < args.homology_coverage <= 1:
        raise ValueError(f'--homology-coverage must be between 0 and 1, got {args.homology_coverage}')
    if args.homology_evalue is not None and args.homology_evalue <= 0:
        raise ValueError(f'--homology-evalue must be positive, got {args.homology_evalue}')
    sensitivity = args.mmseqs_sensitivity
    if sensitivity is not None and not MIN_MMSEQS_SENSITIVITY <= sensitivity <= MAX_MMSEQS_SENSITIVITY:
        raise ValueError(f'--mmseqs-sensitivity must be between {MIN_MMSEQS_SENSITIVITY:g} and '
                         f'{MAX_MMSEQS_SENSITIVITY:g}, got {sensitivity}')
    kmer_size = args.kmer_size
    if kmer_size is not None and not kmer_similarity.MIN_KMER_SIZE <= kmer_size <= kmer_similarity.MAX_KMER_SIZE:
        raise ValueError(f'--kmer-size must be between {kmer_similarity.MIN_KMER_SIZE} and '
                         f'{kmer_similarity.MAX_KMER_SIZE}, got {kmer_size}')


# .................................................................
# The steps

def main() -> None:
    """parrot-preprocess's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Process a datafile, step by step, and write the result, a split-file (if split) and a report."""
    args = build_parser().parse_args()
    arguments.print_header(PROGRAM_NAME, silent=False)
    check_arguments(args)
    set_random_seed(args.seed)
    rng = random_generator()

    output = os.path.abspath(args.output_data_file)
    output_prefix, _ = arguments.output_prefix_and_directory(output)
    classification = args.mode == 'classification'
    report = ProcessingReport(verbose=args.verbose)
    report.add(f'{PROGRAM_NAME} (PARROT {PARROT_VERSION}), {datetime.now().isoformat(timespec="seconds")}')
    report.add(f'Input: {os.path.abspath(args.input_data_file)} ({args.datatype}-mapped {args.mode})')

    lines = read_lines(args, report)
    lines = clean(lines, args, report)
    lines = recode(lines, args, output_prefix, report)
    before_deduplication = lines
    lines = remove_duplicates(lines, args, classification, report)
    lines, splits = split(lines, before_deduplication, args, rng, report)
    lines, splits = balance_classes(lines, splits, args, output_prefix, report)
    write_outputs(lines, splits, args, output, output_prefix, classification, report)


def read_lines(args: argparse.Namespace, report: ProcessingReport) -> list[Line]:
    """Read the input datafile, checking every line has an ID, a sequence and the right number of values.

    Parameters
    ----------
    args : argparse.Namespace
    report : ProcessingReport

    Returns
    -------
    list
        The lines, split into fields (on --delimiter if given, otherwise on any whitespace)

    Raises
    ------
    ValueError
        If the datafile has no data lines, or a line is not usable (see check_lines)
    """
    lines = process_input_data.read_tsv_raw(args.input_data_file, delimiter=args.delimiter)
    if not lines:
        raise ValueError(f'Datafile {args.input_data_file} has no data lines')
    check_lines(lines, args.datatype)
    report.add(f'Read {len(lines):,} lines')
    return lines


def check_lines(lines: list[Line], datatype: str) -> None:
    """Check every line has an ID, a sequence and the right number of values (none, for matrix data)

    Parameters
    ----------
    lines : list
        Lines split into fields
    datatype : str
        'sequence', 'residues' or 'matrix'

    Raises
    ------
    ValueError
        For the first line that does not, or that has an empty field or a field containing whitespace (possible
        with --delimiter; PARROT datafiles separate fields by whitespace, so the output could not be read)
    """
    for number, line in enumerate(lines, start=1):
        if any(field == '' for field in line):
            raise ValueError(f'Data line {number} has an empty column (a delimiter at the start or end of the line, '
                             f'or two in a row): {line}')
        spaced = [field for field in line if len(field.split()) > 1]
        if spaced:
            raise ValueError(f"Data line {number} has a field containing whitespace ('{spaced[0]}'), which a PARROT "
                             'datafile cannot hold (its columns are separated by whitespace); change it, e.g. '
                             'replace the spaces in IDs with underscores')
        if datatype == 'matrix':
            if len(line) != 2:
                raise ValueError(f"Data line {number} has {len(line)} column(s); matrix data has '<ID> <sequence>' "
                                 f'on each line (the values are in the matrix file): {" ".join(line)}')
            continue
        if len(line) < 3:
            raise ValueError(f'Data line {number} has {len(line)} column(s); expected <ID> <sequence> <value(s)>: '
                             f'{" ".join(line)}')
        if datatype == 'sequence' and len(line) != 3:
            raise ValueError(f'Data line {number} ({line[0]}) has {len(line) - 2} values; sequence-mapped data has '
                             'one value per sequence')
        if datatype == 'residues' and len(line) - 2 != len(line[1]):
            raise ValueError(f'Data line {number} ({line[0]}) has {len(line) - 2} values for a sequence of '
                             f'{len(line[1])} residues')


def clean(lines: list[Line], args: argparse.Namespace, report: ProcessingReport) -> list[Line]:
    """Clean sequences (--uppercase, --remove-nonstandard, --min-length, --max-length), if asked.

    Parameters
    ----------
    lines : list
    args : argparse.Namespace
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines

    Raises
    ------
    ValueError
        If cleaning removes every line
    """
    if not (args.uppercase or args.remove_nonstandard or args.min_length is not None or args.max_length is not None):
        return lines

    report.section('Cleaning sequences')
    first_report_line = len(report.lines)
    kept = preproc.clean_sequences(lines, report, uppercase=args.uppercase, remove_nonstandard=args.remove_nonstandard,
                                   min_length=args.min_length, max_length=args.max_length)
    check_lines_remain(kept, 'Cleaning', report.lines[first_report_line:])
    return kept


def check_lines_remain(lines: list[Line], step: str, what_was_done: list[str]) -> None:
    """Stop if a step removed every line, saying what it did (there would be nothing to split or write)

    Parameters
    ----------
    lines : list
        The lines kept by the step
    step : str
        The step, to start the message (e.g. 'Cleaning')
    what_was_done : list of str
        The step's report lines

    Raises
    ------
    ValueError
        If no lines remain
    """
    if not lines:
        raise ValueError(f'{step} removed every line, so nothing was written: ' + '; '.join(what_was_done))


def recode(lines: list[Line], args: argparse.Namespace, output_prefix: str, report: ProcessingReport) -> list[Line]:
    """Renumber class labels as 0, 1, 2, ... (--recode-classes), writing the mapping to a file.

    Parameters
    ----------
    lines : list
    args : argparse.Namespace
    output_prefix : str
        Path and filename prefix of the output files
    report : ProcessingReport

    Returns
    -------
    list
        The lines, recoded if asked
    """
    if not args.recode_classes:
        return lines

    report.section('Recoding classes')
    lines, mapping = preproc.recode_classes(lines)
    mapping_file = output_prefix + '_class_mapping.txt'
    preproc.write_class_mapping(mapping_file, mapping)

    num_classes = len(set(mapping.values()))
    report.add(f'{len(mapping):,} labels recoded as {num_classes:,} classes (0, 1, 2, ...); mapping written to '
               f'{mapping_file}')
    if num_classes > MANY_CLASSES:
        report.add(f'WARNING: {num_classes} individual classes found. This is probably not a good classification '
                   'problem...')
    return lines


def remove_duplicates(lines: list[Line], args: argparse.Namespace, classification: bool,
                      report: ProcessingReport) -> list[Line]:
    """Remove exact duplicate lines (--remove-line-duplicates), and deal with duplicated sequences (--deduplicate).

    Parameters
    ----------
    lines : list
    args : argparse.Namespace
    classification : bool
        True if the values are class labels
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines
    """
    if args.remove_line_duplicates:
        report.section('Duplicate lines')
        before = len(lines)
        lines = preproc.remove_line_duplicates(lines)
        report.add(f'Removed {before - len(lines):,} lines that were exact copies of an earlier line')
        repeated = preproc.count_repeated_sequences(lines)
        if repeated:
            report.add(f'{repeated:,} sequences still appear more than once, with different IDs or values '
                       '(--deduplicate decides what to do with them)')

    deduplication = 'drop' if args.remove_sequence_duplicates else args.deduplicate
    if deduplication is not None:
        report.section(f'Duplicate sequences ({deduplication})')
        first_report_line = len(report.lines)
        lines = preproc.deduplicate(lines, deduplication, args.datatype, classification, report)
        check_lines_remain(lines, 'Removing duplicates', report.lines[first_report_line:])
    return lines


def split(lines: list[Line], before_deduplication: list[Line], args: argparse.Namespace, rng: np.random.Generator,
          report: ProcessingReport) -> tuple[list[Line], list[str] | None]:
    """Assign every line to the training, validation or test set (--split), keeping similar sequences together

    Parameters
    ----------
    lines : list
    before_deduplication : list
        The lines before duplicated sequences were removed: a removed copy's group (--groups) still holds the
        line that replaced it
    args : argparse.Namespace
    rng : numpy.random.Generator
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines (records similar to a higher-priority set are removed)
    list of str or None
        The set of each kept line, or None if the data are not split
    """
    if args.split_method is None:
        return lines, None

    report.section(f'Splitting into training, validation and test sets ({args.split_method})')
    groups = read_groups(args.groups) if args.groups is not None else {}
    items = [splitting.SplitItem(index, line[1], groups.get(line[0], f'line {index}'))
             for index, line in enumerate(lines)]
    if groups:
        report.add(f'{sum(line[0] in groups for line in lines):,} of {len(lines):,} lines belong to a group from '
                   f'{args.groups}; records in a group are kept together')
        links = removed_copy_links(items, before_deduplication, groups)
        if links:
            splitting.merge_groups(items, links)
            report.add(f'{len(links):,} lines also belong to the group of a copy of their sequence that '
                       'deduplication removed')
    fractions = dict(zip(splitting.SET_NAMES, args.set_fractions))
    report.add('Target fractions: ' + ', '.join(f'{splitting.SET_LABELS[name]} {fraction:g}'
                                                for name, fraction in fractions.items()))

    if args.split_method == 'random':
        kept = splitting.random_split(items, fractions, rng, report.add)
    else:
        kept = splitting.similarity_split(items, fractions, rng, report.add, similarity_search(args))

    kept.sort(key=lambda item: item.index)
    report_final_sets(kept, len(lines), report)
    return [lines[item.index] for item in kept], [str(item.split) for item in kept]


def removed_copy_links(items: list[splitting.SplitItem], before_deduplication: list[Line],
                       groups: dict[str, str]) -> list[tuple[str, str]]:
    """Links between each record's group and the groups of the copies of its sequence that deduplication removed

    Deduplication keeps one line per sequence; the copies it removed may have
    belonged to other groups, whose other records must still stay in the same
    set as the line that replaced them.

    Parameters
    ----------
    items : list of SplitItem
        The records to split
    before_deduplication : list
        Every line before deduplication
    groups : dict
        Sequence ID -> group (see read_groups)

    Returns
    -------
    list of (str, str)
        Pairs of groups to join (only groups some record still has)
    """
    groups_of_sequence: dict[str, set[str]] = defaultdict(set)
    for line in before_deduplication:
        if line[0] in groups:
            groups_of_sequence[line[1]].add(groups[line[0]])

    present = {item.group for item in items}
    return [(item.group, group) for item in items for group in sorted(groups_of_sequence.get(item.sequence, ()))
            if group != item.group and group in present]


def similarity_search(args: argparse.Namespace) -> splitting.SimilaritySearch:
    """The search engine for --split homology (MMseqs2) or kmer, with the options given (defaults otherwise)

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    SimilaritySearch
    """
    given: dict[str, Any] = {name: value for name, value in (('max_evalue', args.homology_evalue),
                                                             ('min_coverage', args.homology_coverage))
                             if value is not None}
    given['mask_low_complexity'] = not args.no_masking

    if args.split_method == 'homology':
        if args.mmseqs_sensitivity is not None:
            given['sensitivity'] = args.mmseqs_sensitivity
        return mmseqs_search.MMseqsSearch(mmseqs_search.HomologyCriteria(**given), args.mmseqs, args.threads)

    if args.kmer_size is not None:
        given['kmer_size'] = args.kmer_size
    return kmer_similarity.KmerSearch(kmer_similarity.KmerCriteria(**given))


def report_final_sets(kept: list[splitting.SplitItem], num_lines: int, report: ProcessingReport) -> None:
    """Report how many lines were removed to separate the sets, and how big each set is

    Parameters
    ----------
    kept : list of SplitItem
        The kept records, with their sets
    num_lines : int
        Number of lines before splitting
    report : ProcessingReport

    Raises
    ------
    ValueError
        If any set is empty
    """
    counts = {name: sum(item.split == name for item in kept) for name in splitting.SET_NAMES}
    removed = num_lines - len(kept)
    if removed:
        report.add(f'Removed {removed:,} of {num_lines:,} lines ({100 * removed / num_lines:.1f}%) to separate '
                   'the sets')
    report.add('Final sets: ' + ', '.join(f'{splitting.SET_LABELS[name]} {counts[name]:,} '
                                          f'({100 * counts[name] / len(kept):.1f}%)' for name in counts))
    training, validation, test = (np.array([item.index for item in kept if item.split == name])
                                  for name in splitting.SET_NAMES)
    process_input_data.check_sets_not_empty(training, validation, test,
                                            'after splitting. Use more data or larger fractions.')


def read_groups(path: str) -> dict[str, str]:
    """Read a groups file: '<sequence ID> <group>' on each line (blank and '#' lines are ignored)

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
        Sequence ID -> group name

    Raises
    ------
    ValueError
        If a line does not have exactly two fields
    """
    groups = {}
    for line in process_input_data.read_tsv_raw(path):
        if len(line) != 2:
            raise ValueError(f"Lines of the groups file {path} should be '<sequence ID> <group>', found: "
                             f"{' '.join(line)}")
        groups[line[0]] = f'group {line[1]}'
    return groups


def balance_classes(lines: list[Line], splits: list[str] | None, args: argparse.Namespace, output_prefix: str,
                    report: ProcessingReport) -> tuple[list[Line], list[str] | None]:
    """Down-sample the more common classes (--class-balance), of the training set only if the data are split

    Balancing draws its own random numbers (see class_balancing), seeded by --seed.

    Parameters
    ----------
    lines : list
    splits : list of str or None
        The set of each line, if the data were split
    args : argparse.Namespace
    output_prefix : str
        Path and filename prefix of the output files
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines (the training set first, if split)
    list of str or None
        The set of each kept line
    """
    if not args.class_balance:
        return lines, splits

    report.section('Class balancing' + (' (training set only)' if splits is not None else ''))
    if args.cb_shuffle_off:
        print('Note: --cb-shuffle-off no longer has any effect (balancing always keeps the input order)')

    if splits is None:
        to_balance, others, other_splits = lines, [], []
    else:
        to_balance = [line for line, split in zip(lines, splits) if split == 'train']
        others = [line for line, split in zip(lines, splits) if split != 'train']
        other_splits = [split for split in splits if split != 'train']

    classes, before = class_balancing.class_count_matrix(to_balance)
    report.add(f'Before: {class_balancing.format_class_counts(classes, before.sum(axis=0))} '
               f'(imbalance {imbalance_text(before.sum(axis=0))}; target <= {args.cb_balance_threshold:g}, removing '
               f'at most {100 * args.cb_max_fraction_removed:g}% of {len(to_balance):,} sequences)')

    if args.datatype == 'residues':
        balanced = balance_residue_classes(to_balance, args, output_prefix, report)
    else:
        if args.cb_block_size is not None:
            print('Warning: --cb-block-size does nothing for sequence class balancing')
        balanced = class_balancing.run_class_balance_sequences(to_balance, balance_threshold=args.cb_balance_threshold,
                                                               max_fraction_removed=args.cb_max_fraction_removed)

    classes, after = class_balancing.class_count_matrix(balanced)
    report.add(f'After:  {class_balancing.format_class_counts(classes, after.sum(axis=0))} '
               f'(imbalance {imbalance_text(after.sum(axis=0))})')
    report.add(f'Kept {len(balanced):,} of {len(to_balance):,} sequences'
               + (' of the training set; the validation and test sets keep their natural class mix'
                  if splits is not None else ''))

    if splits is None:
        return balanced, None
    return balanced + others, ['train'] * len(balanced) + other_splits


def balance_residue_classes(lines: list[Line], args: argparse.Namespace, output_prefix: str,
                            report: ProcessingReport) -> list[Line]:
    """Balance residue classes, writing the imbalance after each removal to <output>_class_balance_trace.csv

    Parameters
    ----------
    lines : list
        The lines to balance
    args : argparse.Namespace
    output_prefix : str
    report : ProcessingReport

    Returns
    -------
    list
        The kept lines, in input order
    """
    block_size = args.cb_block_size if args.cb_block_size is not None else class_balancing.DEFAULT_BLOCK_SIZE
    balanced, tracker = class_balancing.run_class_balance_residues(
        lines, balance_threshold=args.cb_balance_threshold, block_size=block_size,
        max_fraction_removed=args.cb_max_fraction_removed)

    budget = class_balancing.removal_budget(len(lines), args.cb_max_fraction_removed)
    report.add('Stopped because ' + class_balancing.residue_balance_stop_reason(tracker, args.cb_balance_threshold,
                                                                                 budget))
    trace_file = output_prefix + '_class_balance_trace.csv'
    with open(trace_file, 'w') as fh:
        for step, imbalance in enumerate(tracker, start=1):
            fh.write(f'{step}, {imbalance:.3f}\n')
    report.add(f'Imbalance after each removal written to {trace_file}')
    return balanced


def imbalance_text(totals: np.ndarray) -> str:
    """The imbalance (most / least common class) of class totals, as text."""
    return f'{totals.max() / totals.min():.3g}' if totals.min() > 0 else 'inf'


def write_outputs(lines: list[Line], splits: list[str] | None, args: argparse.Namespace, output: str,
                  output_prefix: str, classification: bool, report: ProcessingReport) -> None:
    """Write the processed datafile, the split-file (if split) and the report, and check PARROT can read them

    Parameters
    ----------
    lines : list
    splits : list of str or None
    args : argparse.Namespace
    output : str
        The processed datafile
    output_prefix : str
    classification : bool
    report : ProcessingReport
    """
    report.section('Output')
    split_file = output_prefix + '_split_file.txt'
    preproc.write_dataset(lines, splits, output, split_file, report)
    if args.datatype == 'matrix':
        check_matrix_output(output, split_file if splits is not None else None, args, report)
    else:
        check_output(output, split_file if splits is not None else None, args.datatype, classification, report)

    report_file = output_prefix + '_preprocess_report.txt'
    report.write(report_file)
    if not args.verbose:
        print(f'Wrote {output}' + (f', {split_file}' if splits is not None else '') + f' and {report_file}')


def check_output(datafile: str, split_file: str | None, datatype: str, classification: bool,
                 report: ProcessingReport) -> None:
    """Check PARROT can read what was written, recording the result (a warning, not an error, if it cannot)

    Parameters
    ----------
    datafile : str
    split_file : str or None
    datatype : str
    classification : bool
    report : ProcessingReport
    """
    lines = process_input_data.read_tsv_raw(datafile)
    try:
        if classification:
            labels = [float(value) for line in lines for value in line[2:] if not preproc.is_missing_label(value)]
            num_classes = int(max(labels)) + 1 if labels else 1
        else:
            num_classes = 1
        data = process_input_data.parse_file(datafile, Task(datatype, max(num_classes, 2) if classification else 1),
                                             ignore_warnings=True)
        if split_file is not None:
            process_input_data.read_split_file(split_file, num_samples=len(data))
    except ValueError as error:
        hint = classification and preproc.labels_need_recoding(lines)
        report.add(f'WARNING: PARROT cannot train on the output as it is: {error}' +
                   (' (use --recode-classes to turn class labels into 0, 1, 2, ...)' if hint else ''))
        print(report.lines[-1])
        return
    report.add(f'Checked: PARROT reads the output as {datatype}-mapped '
               + (f'classification data with {num_classes} classes' if classification else 'regression data')
               + (', and the split-file matches it' if split_file is not None else ''))



def matrix_class_count(matrix_file: str) -> int:
    """How many classes the matrices hold: one more than the largest label (and at least 2)

    Labels that are not whole numbers are left for parse_file() to report.

    Parameters
    ----------
    matrix_file : str
        A .npz matrix file

    Returns
    -------
    int
    """
    largest_labels = [np.nanmax(matrix) for matrix in read_matrix_file(matrix_file).values()
                      if not np.all(np.isnan(matrix))]
    return max(int(max(largest_labels, default=0)) + 1, 2)


def check_matrix_output(datafile: str, split_file: str | None, args: argparse.Namespace,
                        report: ProcessingReport) -> None:
    """Check PARROT can read the output as matrix data (with the matrix file, if one was given)

    Parameters
    ----------
    datafile : str
    split_file : str or None
    args : argparse.Namespace
    report : ProcessingReport
    """
    if args.matrix_file is None:
        report.add('Not checked against the matrices: give --matrix-file to check every sequence has a matrix that '
                   'fits it')
        return

    try:
        num_classes = 1 if args.mode == 'regression' else matrix_class_count(args.matrix_file)
        task = Task('matrix', num_classes, width=args.matrix_width)
        data = process_input_data.parse_file(datafile, task, ignore_warnings=True, matrix_file=args.matrix_file)
        if split_file is not None:
            process_input_data.read_split_file(split_file, num_samples=len(data))
    except ValueError as error:
        report.add(f'WARNING: PARROT cannot train on the output with {args.matrix_file} as it is: {error}')
        print(report.lines[-1])
        return
    report.add(f'Checked: every sequence has a matrix in {args.matrix_file} that fits it'
               + (', and the split-file matches the output' if split_file is not None else ''))


if __name__ == '__main__':
    main()
