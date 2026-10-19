#!/usr/bin/env python
"""
Usage: $ parrot-predict seq_file saved_network output_file <flags>

Driver script for generating predictions on a list of sequences using a trained
network. For more information on usage, use the '-h' flag.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import os
from typing import Any

from parrot import nardini, plm, prediction, reporting
from parrot import process_input_data as pid
from parrot.checkpoint import load_network
from parrot.encode_sequence import StandardizedEncoder, get_encoder
from parrot.sequence_encoder import SequenceEncoder
from parrot.architectures import ParrotArchitecture
from parrot.matrices.output import check_matrix_names
from parrot.scripts import arguments
from parrot.scripts.examples import add_examples_argument
from parrot.scripts.common import (COMMAND_LINE_DATATYPES, LEGACY_NOTE, add_device_arguments,
                                   add_embedding_cache_argument, add_embedding_context_arguments,
                                   check_embedding_context_arguments, check_probabilistic_classification,
                                   place_in_context, prepare_encoding, select_device, write_context_table)

PROGRAM_NAME = 'parrot-predict'


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-predict

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description='Make predictions with a trained PARROT network.')
    add_examples_argument(parser, PROGRAM_NAME)

    parser.add_argument('seq_file', help="the sequences to predict: '<ID> <sequence>' on each line (or just "
                                         "'<sequence>', with --exclude-seq-id)")
    parser.add_argument('saved_network', help='the trained network (from parrot-train or parrot-optimize)')
    parser.add_argument('output_file', help='where to write the predictions (for matrix networks, a NumPy .npz '
                                            'archive of one matrix per sequence, named by sequence ID)')

    parser.add_argument('--exclude-seq-id', dest='exclude_seq_id', action='store_true',
                        help='the sequence file has no sequence IDs, just one sequence per line')
    parser.add_argument('--probabilistic-classification', dest='probabilistic_classification',
                        action='store_true',
                        help='classification: report the probability of each class, not just the most likely '
                             'class (for residue networks, each residue\'s probabilities are one comma-separated '
                             'token)')
    add_embedding_cache_argument(parser, plm.PREDICT_CACHE_DIRECTORY)
    add_embedding_context_arguments(parser)
    parser.add_argument('--embed-in-isolation', dest='embed_in_isolation', action='store_true',
                        help='networks trained with --embedding-context only: embed each sequence on its own instead '
                             '(e.g. sequences with no full-length protein). The embeddings then differ from those the '
                             'network was trained on, so predictions may be less accurate')
    add_device_arguments(parser)
    parser.add_argument('--verbose', '-v', action='store_true', help='print every setting before predicting')
    parser.add_argument('--silent', action='store_true', help='print nothing')
    parser.add_argument('--print-frequency', default=prediction.DEFAULT_PRINT_FREQUENCY, type=int,
                        metavar='num_sequences',
                        help=f'print progress every this many sequences (default {prediction.DEFAULT_PRINT_FREQUENCY})')
    parser.add_argument('-b', '--batch', default=prediction.DEFAULT_PREDICTION_BATCH, type=int, metavar='batch_size',
                        help=f'sequences predicted at once (default {prediction.DEFAULT_PREDICTION_BATCH}); a smaller '
                             'batch uses less memory, which large matrix networks may need, and changes the '
                             'predictions only by float32 rounding')

    # Everything below is read from networks saved by this version of PARROT
    saved = parser.add_argument_group(
        'recorded in the saved network',
        'Networks saved by this version of PARROT record these, so -d, -c and --encode, if given, are only checked '
        'against the network. Networks saved by earlier versions recorded only their weights: they need -d (and '
        "--encode, if it was not 'onehot'), and their number of classes and size are read from the weights.")
    saved.add_argument('-d', '--datatype', choices=COMMAND_LINE_DATATYPES,
                       help="'sequence' or 'residues' (networks saved by earlier versions are never 'matrix' networks)")
    saved.add_argument('-c', '--classes', type=int, metavar='num_classes',
                       help='number of output classes (1 for regression)')
    saved.add_argument('--encode', default=None, type=str, metavar='encoding_scheme',
                       help="the encoding the network was trained with (default for earlier networks: 'onehot')")
    saved.add_argument('-hs', '--hidden-size', type=int, metavar='hidden_size',
                       help=f'ignored: the network size is read from the saved network {LEGACY_NOTE}')
    saved.add_argument('-nl', '--num-layers', type=int, metavar='num_layers',
                       help=f'ignored: the network size is read from the saved network {LEGACY_NOTE}')
    return parser


def main() -> None:
    """parrot-predict's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Predict values for every sequence in a sequence file with a saved network, and write them to a file."""
    args = build_parser().parse_args()
    arguments.print_header(PROGRAM_NAME, args.silent)
    if (args.hidden_size is not None or args.num_layers is not None) and not args.silent:
        print('Note: --hidden-size and --num-layers are ignored; the network size is read from the saved network')

    seq_file = arguments.check_file_exists(args.seq_file, 'Sequence file')
    saved_network = arguments.check_file_exists(args.saved_network, 'Saved network')
    output_file = os.path.abspath(args.output_file)
    output_prefix, _ = arguments.output_prefix_and_directory(output_file)
    arguments.check_positive(args.print_frequency, '--print-frequency')
    arguments.check_positive(args.batch, 'Batch size')

    device = select_device(args)
    network, encoder = load_network(saved_network, device, legacy_datatype=args.datatype,
                                    legacy_encoding=args.encode or 'onehot')
    check_arguments_match_network(args, network, encoder)
    check_probabilistic_classification(args.probabilistic_classification, network.task)
    check_embedding_context(args, encoder)

    seq_ids, sequences = read_sequence_file(seq_file, args.exclude_seq_id)
    encoder.check_sequences(sequences)
    if network.task.is_matrix:
        check_matrix_names(reporting.matrix_prediction_names(seq_ids, len(sequences)))
    if args.verbose and not args.silent:
        print_prediction_settings(args, seq_file, saved_network, output_file, network, encoder)

    # A network trained on sequences embedded within their full-length proteins embeds these in the same way, unless
    # told to embed each on its own
    if seq_ids is not None:
        context = place_in_context(args, encoder, seq_ids, sequences)
        if context is not None:
            write_context_table(context, output_prefix, args.silent)
    if args.embed_in_isolation:
        assert isinstance(encoder, plm.PLMEncoder)
        encoder.add_contexts({sequence: None for sequence in sequences})
        if not args.silent:
            print('\nNote: this network was trained on sequences embedded within their full-length proteins, but '
                  '--embed-in-isolation embeds each of these on its own')

    # Protein language model and NARDINI encodings: encode every sequence once, into a cache next to the output (or
    # in --embedding-cache)
    prepare_encoding(encoder, sequences, args.embedding_cache,
                     os.path.join(os.path.dirname(output_file), plm.PREDICT_CACHE_DIRECTORY), device,
                     PROGRAM_NAME, args.silent)

    if not args.silent:
        print(f'Predicting {len(sequences):,} sequences with a {network.name} network '
              f'({network.task.datatype}-mapped {network.task.problem_type}; {encoder.name} encoding)')
    print_frequency = None if args.silent else args.print_frequency
    outputs = prediction.predict_sequences(network, sequences, encoder, device, batch_size=args.batch,
                                           print_frequency=print_frequency)

    written = reporting.write_unlabeled_predictions(seq_ids, sequences, outputs, network.task, output_file,
                                                    args.probabilistic_classification)
    if not args.silent:
        print(f'Predictions written to {written}')


def print_prediction_settings(args: argparse.Namespace, seq_file: str, saved_network: str, output_file: str,
                              network: ParrotArchitecture, encoder: SequenceEncoder) -> None:
    """Print the files, network and options a prediction run uses (with the same names as parrot-train's settings)

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    seq_file, saved_network, output_file : str
        Absolute paths of the sequence file, the network and the output file
    network : ParrotArchitecture
        The loaded network
    encoder : SequenceEncoder
        The encoding saved with it
    """
    task = network.task
    arguments.print_settings(False, sequence_file=seq_file, network_file=saved_network, output_file=output_file,
                             datatype=task.datatype, task=task.problem_type, classes=task.num_classes,
                             encoding=encoder.name, architecture=network.name, **network.hyperparameter_values,
                             exclude_seq_id=args.exclude_seq_id, batch_size=args.batch,
                             probabilistic_classification=args.probabilistic_classification)


def check_arguments_match_network(args: argparse.Namespace, network: ParrotArchitecture,
                                  encoder: SequenceEncoder) -> None:
    """Check that any datatype, class count or encoding given on the command line matches the network

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    network : ParrotArchitecture
        The loaded network
    encoder : SequenceEncoder
        The encoder loaded with the network

    Raises
    ------
    ValueError
        If any of them disagree with the saved network
    """

    task = network.task
    if args.datatype is not None and args.datatype != task.datatype:
        raise ValueError(f"--datatype {args.datatype} does not match the saved network, which is a "
                         f"'{task.datatype}' network")

    if args.classes is not None and args.classes != task.num_classes:
        raise ValueError(f'--classes {args.classes} does not match the saved network, which has '
                         f'{task.num_classes} output(s)')

    if args.encode is not None and comparable_config(get_encoder(args.encode)) != comparable_config(encoder):
        raise ValueError(f"--encode {args.encode} does not match the encoding stored with the saved network "
                         f"('{encoder.name}'). The --encode flag can be left out.")


def check_embedding_context(args: argparse.Namespace, encoder: SequenceEncoder) -> None:
    """Check the sequences will be embedded as the network's training sequences were: in context or on their own

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    encoder : SequenceEncoder
        The encoder loaded with the network

    Raises
    ------
    ValueError
        If the network was trained with --embedding-context and neither --embedding-context nor
        --embed-in-isolation was given; if either was given for a network trained without it; if both were
        given; or if --embedding-context is incomplete or used with --exclude-seq-id
    FileNotFoundError
        If a file given with --embedding-context does not exist
    """
    check_embedding_context_arguments(args, encoder)
    trained_in_context = isinstance(encoder, plm.PLMEncoder) and encoder.in_context
    context_given = args.embedding_context is not None

    if context_given and args.embed_in_isolation:
        raise ValueError('Give --embedding-context or --embed-in-isolation, not both')
    if (context_given or args.embed_in_isolation) and not trained_in_context:
        flag = '--embedding-context' if context_given else '--embed-in-isolation'
        raise ValueError(f'{flag} only applies to networks trained with --embedding-context; this network was trained '
                         'on sequences each embedded on its own, so these are embedded on their own too')
    if trained_in_context and not context_given and not args.embed_in_isolation:
        raise ValueError('This network was trained on sequences embedded within their full-length proteins '
                         '(--embedding-context). Give --embedding-context and --embedding-context-map for the '
                         'sequences to predict, or --embed-in-isolation to embed each on its own (embeddings unlike '
                         'those the network was trained on, so predictions may be less accurate)')


def comparable_config(encoder: SequenceEncoder) -> dict[str, Any]:
    """What an encoding given with --encode must match: the saved encoding, without the settings --encode cannot give

    A standardized encoding is compared by the encoding it standardizes, a
    NARDINI encoding by its per-residue encoding alone (its number of scrambles,
    seed and SPARROW version are settings of the saved network), and a protein
    language model without whether it embeds in context (which
    --embedding-context, not --encode, says), subtracts each embedding's mean
    or uses an earlier layer (settings of the saved network, as is the
    description of the layer's output).

    Parameters
    ----------
    encoder : SequenceEncoder

    Returns
    -------
    dict
    """
    if isinstance(encoder, StandardizedEncoder):
        return comparable_config(encoder.base)
    if isinstance(encoder, nardini.NardiniEncoder):
        residue = encoder.residue_encoder
        return {'name': nardini.NARDINI_ENCODING,
                'residue_encoding': comparable_config(residue) if residue is not None else None}
    config = encoder.to_config()
    for setting in (plm.IN_CONTEXT_CONFIG_KEY, plm.MEAN_SUBTRACTED_CONFIG_KEY, plm.LAYER_CONFIG_KEY, 'representation'):
        config.pop(setting, None)
    return config


def read_sequence_file(seq_file: str, exclude_seq_id: bool) -> tuple[list[str] | None, list[str]]:
    """Read the sequences (and optional IDs) to predict on

    Blank lines and comment lines (starting with '#') are skipped, exactly as
    they are for training datafiles.

    Parameters
    ----------
    seq_file : str
            Path to a file with one sequence per line, optionally preceded by a
            whitespace-separated sequence ID
    exclude_seq_id : bool
            If True, each line contains only a sequence. If False, each line
            contains a sequence ID followed by a sequence.

    Returns
    -------
    list of str or None
            The sequence ID on each line, or None if `exclude_seq_id` is True
    list of str
            The sequence on each line, in file order (duplicates are kept)

    Raises
    ------
    ValueError
            If any line has the wrong number of columns
    """

    lines = pid.read_tsv_raw(seq_file)

    if exclude_seq_id:
        expected_columns = 1
        expected_format = '<sequence>'
    else:
        expected_columns = 2
        expected_format = '<seqID> <sequence>'

    for line in lines:
        if len(line) == expected_columns:
            continue
        if len(line) > expected_columns:
            hint = ('it looks like a datafile, with values to predict; the sequence file has no values' if
                    len(line) > 2 else 'use --exclude-seq-id only if the file has no sequence IDs')
        else:
            hint = 'use --exclude-seq-id if the file has no sequence IDs'
        raise ValueError(f"Lines in the sequence file should have the format {expected_format}, but found: "
                         f"{' '.join(line)} ({hint})")

    if exclude_seq_id:
        return None, [line[0] for line in lines]

    return [line[0] for line in lines], [line[1] for line in lines]


if __name__ == '__main__':
    main()
