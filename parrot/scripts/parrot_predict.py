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

from parrot import process_input_data as pid
from parrot import prediction, reporting
from parrot.checkpoint import load_network
from parrot.encode_sequence import StandardizedEncoder, get_encoder
from parrot import plm
from parrot.scripts.common import (add_device_arguments, add_embedding_cache_argument, prepare_plm_embeddings,
                                  select_device)
from parrot.tools import cli, validate_args



def main():
    parser = argparse.ArgumentParser(description='Make predictions with a trained PARROT network.')

    parser.add_argument('seq_file', help='path to tsv file with format: <idx> <sequence>')
    parser.add_argument('saved_network', help='path to the trained network file')
    parser.add_argument('output_file', help='file with sequences and their predicted values')

    parser.add_argument('-d', '--datatype', metavar='dtype', type=str,
                        help="'sequence' or 'residues'. Only needed for networks saved by PARROT versions "
                             "that did not record it; otherwise checked against the saved network.")
    parser.add_argument('-c', '--classes', type=int, metavar='num_classes',
                        help='Number of output classes (1 for regression). Optional; checked against the saved network.')
    parser.add_argument('--encode', default=None, type=str, metavar='encoding_scheme',
                        help="'onehot', 'biophysics', or a path to a user-created scheme. Only needed for networks "
                             "saved by older PARROT versions that were not trained with one-hot encoding.")
    parser.add_argument('-hs', '--hidden-size', type=int, metavar='hidden_size',
                        help='no longer needed: the network size is read from the saved network')
    parser.add_argument('-nl', '--num-layers', type=int, metavar='num_layers',
                        help='no longer needed: the network size is read from the saved network')

    parser.add_argument('--exclude-seq-id', dest='excludeSeqID', action='store_true',
                        help='use if data_file lacks sequence IDs in the first column of each line')
    parser.add_argument('--probabilistic-classification', dest='probabilistic_classification',
                        action='store_true', help='Optional implementation for sequence classification')
    add_device_arguments(parser)
    add_embedding_cache_argument(parser, plm.PREDICT_CACHE_DIRECTORY)
    parser.add_argument('--silent', action='store_true',
                        help="Flag which, if provided, ensures no output is generated to the terminal")
    parser.add_argument('--print-frequency', default=1000, type=int,
                        help="Value that defines how often status updates should be printed (in number of sequences predicted. Default=1000")

    args = parser.parse_args()

    cli.print_startup('parrot-predict', args.silent)

    seq_file = validate_args.check_file_exists(args.seq_file, 'Sequence file')
    saved_network = validate_args.check_file_exists(args.saved_network, 'Saved network')
    output_file = os.path.abspath(args.output_file)
    validate_args.split_file_and_directory(output_file)

    device = select_device(args)
    network, encoder = load_network(saved_network, device, legacy_datatype=args.datatype,
                                    legacy_encoding=args.encode or 'onehot')
    check_arguments_match_network(args, network, encoder)
    task = network.task

    if args.probabilistic_classification and (task.per_residue or not task.is_classification):
        raise ValueError('Probabilistic classification only implemented for sequence classification')

    cli.print_settings(args.silent, sequence_file=seq_file, network_file=saved_network, output_file=output_file,
                       architecture=network.name, **network.hyperparameter_values, dtype=task.datatype,
                       num_classes=task.num_classes, problem_type=task.problem_type, encode=encoder.name,
                       excludeSeqID=args.excludeSeqID, probabilistic_classification=args.probabilistic_classification)

    # Convert sequence file to a list of (seqID, sequence) entries, one per line
    seq_ids, sequences = read_sequence_file(seq_file, args.excludeSeqID)
    encoder.check_sequences(sequences)

    # Protein language model encodings: embed every sequence once, into a cache next to the output
    prepare_plm_embeddings(encoder, sequences, args.embedding_cache,
                           os.path.join(os.path.dirname(output_file), plm.PREDICT_CACHE_DIRECTORY), device,
                           'parrot-predict', args.silent)

    if not args.silent:
        print('\n---> Starting predictions...\n')

    print_frequency = None if args.silent else args.print_frequency
    outputs = prediction.predict_sequences(network, sequences, encoder, device, print_frequency=print_frequency)

    if not args.silent:
        print('\n---> Prediction complete!')

    reporting.write_unlabeled_predictions(None if args.excludeSeqID else seq_ids, sequences, outputs, task,
                                          output_file, args.probabilistic_classification)


def check_arguments_match_network(args, network, encoder):
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

    # A standardized network's encoding is checked against the encoding it standardizes
    saved_base = encoder.base if isinstance(encoder, StandardizedEncoder) else encoder
    if args.encode is not None and get_encoder(args.encode).to_config() != saved_base.to_config():
        raise ValueError(f"--encode {args.encode} does not match the encoding stored with the saved network "
                         f"('{encoder.name}'). The --encode flag can be left out.")


def read_sequence_file(seq_file, excludeSeqID):
    """Read the sequences (and optional IDs) to predict on

    Blank lines and comment lines (starting with '#') are skipped, exactly as
    they are for training datafiles.

    Parameters
    ----------
    seq_file : str
            Path to a file with one sequence per line, optionally preceded by a
            whitespace-separated sequence ID
    excludeSeqID : bool
            If True, each line contains only a sequence. If False, each line
            contains a sequence ID followed by a sequence.

    Returns
    -------
    list of str or None
            The sequence ID on each line, or None if `excludeSeqID` is True
    list of str
            The sequence on each line, in file order (duplicates are kept)

    Raises
    ------
    ValueError
            If any line has the wrong number of columns
    """

    lines = pid.read_tsv_raw(seq_file)

    if excludeSeqID:
        expected_columns = 1
        expected_format = '<sequence>'
    else:
        expected_columns = 2
        expected_format = '<seqID> <sequence>'

    for line in lines:
        if len(line) != expected_columns:
            raise ValueError(f"Lines in the sequence file should have the format {expected_format}, but found: {' '.join(line)}\n"
                             "(use --exclude-seq-id if the file has no sequence IDs)")

    if excludeSeqID:
        return None, [line[0] for line in lines]

    return [line[0] for line in lines], [line[1] for line in lines]
