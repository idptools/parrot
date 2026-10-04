#!/usr/bin/env python
"""
Usage: $ parrot-preprocess raw_data_file processed_data_file <flags>

parrot-preprocess turns a PARROT datafile into one ready for training: it can
clean sequences, recode class labels, deal with duplicate sequences, split the
data into training, validation and test sets without leakage between them
(keeping homologous sequences in the same set, with MMseqs2), and balance the
classes of the training set. It writes the processed datafile, a matching
split-file (if the data were split) and a report of everything it did.

For more information on usage, use the '-h' flag.

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""
import argparse
import math
import os
from datetime import datetime

import numpy as np

from parrot.package_version import PARROT_VERSION
from parrot import process_input_data
from parrot.tasks import Task
from parrot.tools import class_balancing, cli, homology, kmer_similarity, preproc, validate_args

# Default proportions of the data in the training, validation and test sets
DEFAULT_SET_FRACTIONS = (0.7, 0.15, 0.15)


def main():

    parser = argparse.ArgumentParser(
        description='parrot-preprocess cleans, deduplicates, splits (without leakage between sets) and class-balances '
                    'a PARROT datafile, writing a processed datafile, a matching split-file and a report.')

    parser.add_argument('input_data_file', help='path to tsv file with format: <idx> <sequence> <data>. This is the file that will be manipulated in some way')
    parser.add_argument('output_data_file', help='path to be written to for an output data file. Will overwrite an existing file')
    parser.add_argument('-d', '--datatype', metavar='dtype', type=str, required=True, help="REQUIRED. Format of the input data file, must be 'sequence' or 'residues'")
    parser.add_argument("-m", "--mode", help="Must be either 'classification' or 'regression'", type=str, required=True, metavar='mode')
    parser.add_argument("--delimiter", help="If a non-whitespace delimiter was used in an input file, this allows the user to define an alternative character.", type=str)
    parser.add_argument("-v", "--verbose", help="Print the report of what was done as it happens", action='store_true')
    parser.add_argument("--seed", type=int, default=None, help="Random seed for a reproducible split and class balancing (default: not seeded)")

    cleaning = parser.add_argument_group('cleaning sequences')
    cleaning.add_argument('--uppercase', action='store_true', help='convert every sequence to upper case')
    cleaning.add_argument('--remove-nonstandard', dest='remove_nonstandard', action='store_true',
                          help='remove sequences containing anything other than the 20 standard amino acids')
    cleaning.add_argument('--min-length', dest='min_length', type=int, help='remove sequences shorter than this')
    cleaning.add_argument('--max-length', dest='max_length', type=int, help='remove sequences longer than this')

    # recode flag
    parser.add_argument("--recode-classes", help="Flag which, if set to true, will read in class data regardless of if it would have been valid in PARROT and recode it in a parrot-compliant format (0,1,2,...[n]) where there are n-1 classes. Missing labels (nan) are kept as they are. The mapping is written to <output>_class_mapping.txt next to the output file.", action='store_true')

    duplicates = parser.add_argument_group('duplicate sequences')
    duplicates.add_argument('--deduplicate', choices=preproc.DEDUPLICATION_MODES,
                            help="keep one line per unique sequence: 'merge' averages the copies' values (regression; "
                                 "residue by residue for residue data) or keeps one copy if their labels agree "
                                 "(classification; residues whose copies disagree become nan); 'first' keeps the "
                                 "first copy; 'drop' removes every copy")
    duplicates.add_argument("--remove-line-duplicates", help="Remove lines that are exact copies of an earlier line (same ID, sequence and values)", action='store_true')
    duplicates.add_argument("--remove-sequence-duplicates", help="Remove every copy of any sequence that appears more than once (same as --deduplicate drop)", action='store_true')

    splitting = parser.add_argument_group('splitting into training, validation and test sets')
    splitting.add_argument('--split', choices=('homology', 'kmer', 'random'),
                           help="write a split-file for the output, keeping similar sequences in the same set so no "
                                "set contains a sequence similar to one in another (records similar to a "
                                "higher-priority set are removed, test > validation > training). 'homology' finds "
                                "homologues with MMseqs2 (most sensitive; MMseqs2 must be installed); 'kmer' uses "
                                "PARROT's own k-mer search, which needs nothing extra but only finds closer relatives "
                                "(most at >= 40-50%% identity); 'random' splits at random, keeping only identical "
                                "sequences together")
    splitting.add_argument('--set-fractions', dest='set_fractions', nargs=3, type=float, default=DEFAULT_SET_FRACTIONS,
                           metavar=('train', 'val', 'test'),
                           help='target proportions of the training, validation and test sets (default 0.7 0.15 0.15)')
    splitting.add_argument('--groups', metavar='file',
                           help='file of "<sequence ID> <group>" lines: records in the same group (e.g. overlapping '
                                'tiles of one protein) are always put in the same set')
    splitting.add_argument('--homology-evalue', dest='homology_evalue', type=float,
                           help='homology and kmer: a similarity with at most this E-value links two sequences '
                                '(default 1e-3)')
    splitting.add_argument('--homology-coverage', dest='homology_coverage', type=float,
                           help='homology and kmer: ...if it covers at least this fraction of the shorter sequence '
                                '(default 0.3)')
    splitting.add_argument('--no-masking', dest='no_masking', action='store_true',
                           help='homology and kmer: do not mask low-complexity regions, so similarity in them also '
                                'links sequences (even more stringent)')
    splitting.add_argument('--mmseqs', metavar='path', help='homology: MMseqs2 executable (default: mmseqs on the PATH)')
    splitting.add_argument('--threads', type=int, help='homology: threads for MMseqs2 (default: every CPU)')
    splitting.add_argument('--mmseqs-sensitivity', dest='mmseqs_sensitivity', type=float,
                           help='homology: MMseqs2 search sensitivity, 1 to 7.5 (default 7.5, the most sensitive)')
    splitting.add_argument('--kmer-size', dest='kmer_size', type=int,
                           help=f'kmer: length of the shared words, 3 to 8 (default {kmer_similarity.KmerCriteria.kmer_size}; '
                                'smaller finds more distant relatives but is slower and more easily misled by chance '
                                'matches)')

    # class balance
    balancing = parser.add_argument_group('class balancing (classification only)')
    balancing.add_argument("--class-balance",   help="Down-sample the more common classes so classes are balanced (only the training set, if the data are split). For residues, the balance score after each removal is written to <output>_class_balance_trace.csv next to the output file", action='store_true')
    balancing.add_argument("--cb-block-size", help="If --class-balance used with residues, the number of randomly drawn sequences considered for each removal (default 100; larger makes better choices, more slowly)", type=int, default=100)
    balancing.add_argument("--cb-max-fraction-removed", help="If --class-balance used, defines what fraction of the data could be removed in pursuit of a balanced dataset. Default = 0.25 ", type=float, default=0.25)
    balancing.add_argument("--cb-balance-threshold", help="If --class-balance used, defines the balance ratio (most numerous class / least numerous class) threshold used to say if a dataset is 'balanced'. Default = 1.4", type=float, default=1.4)
    balancing.add_argument("--cb-shuffle-off", help=argparse.SUPPRESS, action='store_true')  # no longer has any effect

    args = parser.parse_args()
    check_arguments(args)
    cli.set_random_seed(args.seed)
    rng = np.random.default_rng(np.random.randint(2 ** 31))

    output = os.path.abspath(args.output_data_file)
    output_prefix, _ = validate_args.split_file_and_directory(output)
    report = preproc.ProcessingReport(verbose=args.verbose)
    report.add(f'parrot-preprocess (PARROT {PARROT_VERSION}), {datetime.now().isoformat(timespec="seconds")}')
    report.add(f'Input: {os.path.abspath(args.input_data_file)} ({args.datatype}-mapped {args.mode})')
    classification = args.mode == 'classification'

    # read in and parse the TSV file. Note that if no args.delimiter is passed, this passes 'None' which
    # is the default value and splits on any whitespace.
    lines = process_input_data.read_tsv_raw(args.input_data_file, delimiter=args.delimiter)
    check_lines(lines, args.datatype)
    report.add(f'Read {len(lines):,} lines')

    if args.uppercase or args.remove_nonstandard or args.min_length is not None or args.max_length is not None:
        report.section('Cleaning sequences')
        lines = preproc.clean_sequences(lines, report, uppercase=args.uppercase,
                                        remove_nonstandard=args.remove_nonstandard, min_length=args.min_length,
                                        max_length=args.max_length)

    if args.recode_classes:
        report.section('Recoding classes')
        (lines, mapping) = preproc.recode_classes(lines)
        mapping_file = output_prefix + '_class_mapping.txt'
        with open(mapping_file, 'w') as fh:
            fh.write('File generated by parrot-preprocess\n\n')
            fh.write(f'File generated at {datetime.now()}\n')
            fh.write(f'Number of classes: {len(set(mapping.values()))}\n\n')
            for i in mapping:
                fh.write(f'{i} = {mapping[i]}\n')
        report.add(f'{len(mapping):,} labels recoded as {len(set(mapping.values())):,} classes (0, 1, 2, ...); '
                   f'mapping written to {mapping_file}')

    if args.remove_line_duplicates:
        report.section('Duplicate lines')
        before = len(lines)
        lines = preproc.remove_line_duplicates(lines, args.verbose)
        report.add(f'Removed {before - len(lines):,} lines that were exact copies of an earlier line')

    deduplication = 'drop' if args.remove_sequence_duplicates else args.deduplicate
    if deduplication is not None:
        report.section(f'Duplicate sequences ({deduplication})')
        lines = preproc.deduplicate(lines, deduplication, args.datatype, classification, report)

    splits = None
    if args.split is not None:
        report.section(f'Splitting into training, validation and test sets ({args.split})')
        lines, splits = split_lines(lines, args, rng, report)

    if args.class_balance:
        report.section('Class balancing' + (' (training set only)' if splits is not None else ''))
        lines, splits = balance_classes(lines, splits, args, output_prefix, report)

    report.section('Output')
    split_file = output_prefix + '_split_file.txt'
    preproc.write_dataset(lines, splits, output, split_file, report)
    check_output(output, split_file if splits is not None else None, args.datatype, classification, report)

    report_file = output_prefix + '_preprocess_report.txt'
    report.write(report_file)
    if not args.verbose:
        print(f'Wrote {output}' + (f', {split_file}' if splits is not None else '') + f' and {report_file}')


def check_arguments(args):
    """Check the options before reading any data

    Raises
    ------
    ValueError
        If the options are invalid or inconsistent
    """
    cli.validate_args(args.datatype, ['residues', 'sequence'], "Please pass either 'residues' or 'sequence' with the --datatype flag")
    cli.validate_args(args.mode, ['classification', 'regression'], "Please pass either 'classification' or 'regression' with the --mode flag")

    if args.recode_classes and args.mode != 'classification':
        raise ValueError('If we are trying to recode classes the data mode must be classification')
    if args.class_balance and args.mode != 'classification':
        raise ValueError('Class balancing (--class-balance) only applies to classification data (--mode classification)')
    if args.remove_line_duplicates and args.remove_sequence_duplicates:
        raise ValueError('Cannot request to remove both line duplicates (--remove-line-duplicates) and sequence duplicates (--remove-sequence-duplicates). Must be one or the other')
    if args.remove_sequence_duplicates and args.deduplicate is not None:
        raise ValueError('--remove-sequence-duplicates is the same as --deduplicate drop; give one or the other')

    for name, value in (('--min-length', args.min_length), ('--max-length', args.max_length)):
        if value is not None and value < 1:
            raise ValueError(f'{name} must be at least 1, got {value}')
    if args.min_length is not None and args.max_length is not None and args.min_length > args.max_length:
        raise ValueError('--min-length cannot be larger than --max-length')

    # Options that only some splitting methods use
    for flag, given, methods in (('--groups', args.groups is not None, ('homology', 'kmer', 'random')),
                                 ('--set-fractions', tuple(args.set_fractions) != DEFAULT_SET_FRACTIONS,
                                  ('homology', 'kmer', 'random')),
                                 ('--homology-evalue', args.homology_evalue is not None, ('homology', 'kmer')),
                                 ('--homology-coverage', args.homology_coverage is not None, ('homology', 'kmer')),
                                 ('--no-masking', args.no_masking, ('homology', 'kmer')),
                                 ('--mmseqs', args.mmseqs is not None, ('homology',)),
                                 ('--threads', args.threads is not None, ('homology',)),
                                 ('--mmseqs-sensitivity', args.mmseqs_sensitivity is not None, ('homology',)),
                                 ('--kmer-size', args.kmer_size is not None, ('kmer',))):
        if given and args.split not in methods:
            raise ValueError(f'{flag} only applies to --split ' + ' or '.join(methods))

    if args.split is not None:
        for fraction in args.set_fractions:
            if not 0 < fraction < 1:
                raise ValueError(f'Set fractions must be between 0 and 1, got {fraction}')
        if not math.isclose(sum(args.set_fractions), 1.0):
            raise ValueError('Set fractions must sum to 1')
        if args.groups is not None:
            validate_args.check_file_exists(args.groups, 'Groups file')
        if args.split == 'homology':
            homology.find_mmseqs(args.mmseqs)  # fail before doing any work if MMseqs2 is missing
        if args.homology_coverage is not None and not 0 < args.homology_coverage <= 1:
            raise ValueError(f'--homology-coverage must be between 0 and 1, got {args.homology_coverage}')
        if args.homology_evalue is not None and args.homology_evalue <= 0:
            raise ValueError(f'--homology-evalue must be positive, got {args.homology_evalue}')
        if args.mmseqs_sensitivity is not None and not 1 <= args.mmseqs_sensitivity <= 7.5:
            raise ValueError(f'--mmseqs-sensitivity must be between 1 and 7.5, got {args.mmseqs_sensitivity}')
        if args.kmer_size is not None and not 3 <= args.kmer_size <= 8:
            raise ValueError(f'--kmer-size must be between 3 and 8, got {args.kmer_size}')


def check_lines(lines, datatype):
    """Check every line has an ID, a sequence and the right number of values

    Raises
    ------
    ValueError
        For the first line that does not
    """
    for number, line in enumerate(lines, start=1):
        if len(line) < 3:
            raise ValueError(f'Data line {number} has {len(line)} column(s); expected <ID> <sequence> <value(s)>: '
                             f'{" ".join(line)}')
        if datatype == 'sequence' and len(line) != 3:
            raise ValueError(f'Data line {number} ({line[0]}) has {len(line) - 2} values; sequence-mapped data has '
                             'one value per sequence')
        if datatype == 'residues' and len(line) - 2 != len(line[1]):
            raise ValueError(f'Data line {number} ({line[0]}) has {len(line) - 2} values for a sequence of '
                             f'{len(line[1])} residues')


def split_lines(lines, args, rng, report):
    """Assign every line to a set, returning the kept lines and their sets."""
    groups = read_groups(args.groups) if args.groups is not None else {}
    items = [homology.SplitItem(index, line[1], groups.get(line[0], f'line {index}')) for index, line in enumerate(lines)]
    if groups:
        report.add(f'{sum(line[0] in groups for line in lines):,} of {len(lines):,} lines belong to a group from '
                   f'{args.groups}; records in a group are kept together')
    fractions = dict(zip(('train', 'val', 'test'), args.set_fractions))
    report.add('Target fractions: ' + ', '.join(f'{homology.SET_LABELS[name]} {fraction:g}'
                                                for name, fraction in fractions.items()))

    if args.split == 'random':
        kept = homology.random_split(items, fractions, rng, report.add)
    else:
        # Options left out keep each method's defaults
        given = {name: value for name, value in (('max_evalue', args.homology_evalue),
                                                 ('min_coverage', args.homology_coverage)) if value is not None}
        given['mask_low_complexity'] = not args.no_masking
        if args.split == 'homology':
            if args.mmseqs_sensitivity is not None:
                given['sensitivity'] = args.mmseqs_sensitivity
            search = homology.MMseqsSearch(homology.HomologyCriteria(**given), args.mmseqs, args.threads)
        else:
            if args.kmer_size is not None:
                given['kmer_size'] = args.kmer_size
            search = kmer_similarity.KmerSearch(kmer_similarity.KmerCriteria(**given))
        kept = homology.similarity_split(items, fractions, rng, report.add, search)

    kept.sort(key=lambda item: item.index)
    counts = {name: sum(item.split == name for item in kept) for name in ('train', 'val', 'test')}
    if len(kept) < len(lines):
        report.add(f'Removed {len(lines) - len(kept):,} of {len(lines):,} lines ({100 * (len(lines) - len(kept)) / len(lines):.1f}%) '
                   'to separate the sets')
    report.add('Final sets: ' + ', '.join(f'{homology.SET_LABELS[name]} {counts[name]:,} '
                                          f'({100 * counts[name] / len(kept):.1f}%)' for name in counts))
    process_input_data.check_sets_not_empty(*(np.array([item.index for item in kept if item.split == name])
                                              for name in ('train', 'val', 'test')),
                                            'after splitting. Use more data or larger fractions.')
    return [lines[item.index] for item in kept], [item.split for item in kept]


def read_groups(path):
    """Read a groups file: '<sequence ID> <group>' on each line (blank and '#' lines are ignored)."""
    groups = {}
    for line in process_input_data.read_tsv_raw(path):
        if len(line) != 2:
            raise ValueError(f"Lines of the groups file {path} should be '<sequence ID> <group>', found: {' '.join(line)}")
        groups[line[0]] = f'group {line[1]}'
    return groups


def balance_classes(lines, splits, args, output_prefix, report):
    """Down-sample the more common classes (of the training set only, if the data are split)."""
    if splits is None:
        to_balance, others, other_splits = lines, [], []
    else:
        to_balance = [line for line, split in zip(lines, splits) if split == 'train']
        others = [line for line, split in zip(lines, splits) if split != 'train']
        other_splits = [split for split in splits if split != 'train']

    if args.cb_shuffle_off:
        print('Note: --cb-shuffle-off no longer has any effect (balancing always keeps the input order)')
    classes, before = class_balancing.class_count_matrix(to_balance)
    report.add(f'Before: {class_balancing.format_class_counts(classes, before.sum(axis=0))} '
               f'(imbalance {imbalance_text(before.sum(axis=0))}; target <= {args.cb_balance_threshold:g}, removing '
               f'at most {100 * args.cb_max_fraction_removed:g}% of {len(to_balance):,} sequences)')

    if args.datatype == 'residues':
        balanced, tracker = class_balancing.run_class_balance_residues(
            to_balance, balance_threshold=args.cb_balance_threshold, block_size=args.cb_block_size,
            max_fraction_removed=args.cb_max_fraction_removed, verbose=args.verbose)
        trace_file = output_prefix + '_class_balance_trace.csv'
        with open(trace_file, 'w') as fh:
            for x in range(len(tracker)):
                fh.write(f'{x+1}, {tracker[x]:.3f}\n')
        report.add(f'Imbalance after each removal written to {trace_file}')
    else:
        if args.cb_block_size != 100:
            print('Warning: --cb-block-size does nothing for sequence class balancing')
        balanced = class_balancing.run_class_balance_sequences(
            to_balance, balance_threshold=args.cb_balance_threshold,
            max_fraction_removed=args.cb_max_fraction_removed, verbose=args.verbose)

    classes, after = class_balancing.class_count_matrix(balanced)
    report.add(f'After:  {class_balancing.format_class_counts(classes, after.sum(axis=0))} '
               f'(imbalance {imbalance_text(after.sum(axis=0))})')
    report.add(f'Kept {len(balanced):,} of {len(to_balance):,} sequences'
               + (' of the training set; the validation and test sets keep their natural class mix'
                  if splits is not None else ''))

    if splits is None:
        return balanced, None
    return balanced + others, ['train'] * len(balanced) + other_splits


def imbalance_text(totals):
    """The imbalance (most / least common class) of class totals, as text."""
    return f'{totals.max() / totals.min():.3g}' if totals.min() > 0 else 'inf'


def check_output(datafile, split_file, datatype, classification, report):
    """Check PARROT can read what was written, recording the result (a warning, not an error, if it cannot)."""
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
        report.add(f'WARNING: PARROT cannot train on the output as it is: {error}' +
                   (' (use --recode-classes to turn class labels into 0, 1, 2, ...)' if classification else ''))
        print(report.lines[-1])
        return
    report.add(f'Checked: PARROT reads the output as {datatype}-mapped '
               + (f'classification data with {num_classes} classes' if classification else 'regression data')
               + (', and the split-file matches it' if split_file is not None else ''))
