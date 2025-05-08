#!/usr/bin/env python
"""
Usage: $ parrot-predict data_file saved_network output_file <flags>
  
Driver script for generating predictions on a list of sequences using a trained
bidirectional recurrent neural network. For more information on usage, use the '-h'
flag.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import os
import sys

import numpy as np
import torch
import argparse

from parrot import brnn_architecture
from parrot import process_input_data as pid
from parrot import train_network
from parrot.tools import cli, validate_args


# Parse the command line arguments
def main():

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Make predictions with a bi-directional RNN.')

    parser.add_argument('seq_file', help='path to tsv file with format: <idx> <sequence>')

    parser.add_argument('saved_network', help='path to the trained network weights file')

    parser.add_argument('output_file', help='file with sequences and their predicted values')

    parser.add_argument('-d', '--datatype', metavar='dtype', type=str, required=True,
                        help="REQUIRED. Format of the input data file, must be 'sequence' or 'residues'")

    parser.add_argument('-c', '--classes', type=int, metavar='num_classes', required=True,
                        help='REQUIRED. Number of output classes, for regression put 1')

    parser.add_argument('-hs', '--hidden-size', default=10, type=int, metavar='hidden_size',
                        help='hidden vector size (def=10)')

    parser.add_argument('-nl', '--num-layers', default=1, type=int, metavar='num_layers',
                        help='number of layers per direction (def=1)')

    parser.add_argument('--encode', default='onehot', type=str, metavar='encoding_scheme',
                                            help="'onehot' (default), 'biophysics', or specify a path to a user-created scheme")

    parser.add_argument('--exclude-seq-id', dest='excludeSeqID', action='store_true',
                        help='use if data_file lacks sequence IDs in the first column of each line')

    parser.add_argument('--probabilistic-classification', dest='probabilistic_classification',
                        action='store_true', help='Optional implementation for sequence classificaion')

    parser.add_argument('--silent', action='store_true',
                        help="Flag which, if provided, ensures no output is generated to the terminal")

    parser.add_argument('--print-frequency', default=1000, type=int, 
                        help="Value that defines how often status updates should be printed (in number of sequences predicted. Default=1000")

    args = parser.parse_args()

    # print startup 
    cli.print_startup('parrot-predict', args.silent)
    device = 'cpu'

    # Hyper-parameters
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    # Data format
    dtype = args.datatype
    num_classes = args.classes

    # Other flags
    encode = args.encode
    excludeSeqID = args.excludeSeqID
    probabilistic_classification = args.probabilistic_classification


    if args.silent:
        print_frequency = None
    else:
        print_frequency = args.print_frequency


    ###############################################################################
    ################    Validate arguments and initialize:      ###################

    # Ensure that provided sequence file exists
    seq_file = validate_args.check_file_exists(args.seq_file, 'Sequence file')

    # Ensure that the saved weights file exists
    saved_weights = validate_args.check_file_exists(args.saved_network, 'Saved network')

    # Ensure that output file location is valid
    output_file = os.path.abspath(args.output_file)
    output_filename, output_dir = validate_args.split_file_and_directory(output_file)
    validate_args.check_directory(output_dir, "Output directory")

    # Set encoding scheme and/or validate user scheme
    encoding_scheme, encoder, input_size = validate_args.set_encoding_scheme(encode)

    # Initialize network as classifier or regressor
    if num_classes > 1:
        problem_type = 'classification'
    elif num_classes == 1:
        problem_type = 'regression'
    else:
        raise ValueError('Number of classes must be a positive integer.')

    # Ensure that hidden size and num layers are both positive ints
    validate_args.check_positive(hidden_size, 'Hidden vector size')
    validate_args.check_positive(num_layers, 'Number of layers')

    # Ensure that task is binary sequence classification if
    # probabilistic_classfication is set
    if probabilistic_classification:
        if dtype != 'sequence':
            raise ValueError('Proportional classification only implemented for sequence classification')

    # Initialize network architecture depending on data format
    if dtype == 'sequence':
        # Use a many-to-one architecture
        brnn_network = brnn_architecture.BRNN_MtO(input_size, hidden_size,
                                                num_layers, num_classes, device).to(device)
    elif dtype == 'residues':
        # Use a many-to-many architecture
        brnn_network = brnn_architecture.BRNN_MtM(input_size, hidden_size,
                                                num_layers, num_classes, device).to(device)
    else:
        raise ValueError('Invalid argument `--datatype`: must be "residues" or "sequence"')


    # print info on the setup
    cli.print_settings(args.silent, 
                    sequence_file=seq_file,
                    weights_file=saved_weights,
                    output_file=output_file,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    num_classes=num_classes,
                    problem_type=problem_type,
                    encode=encode,
                    excludeSeqID=excludeSeqID,
                    probabilistic_classification=probabilistic_classification)

    ###############################################################################

    brnn_network.load_state_dict(torch.load(saved_weights,
                                            map_location=torch.device(device)))

    # Convert sequence file to list of sequences:
    if excludeSeqID:
        with open(seq_file) as f:
            sequences = [line.strip() for line in f]
    else:
        with open(seq_file) as f:
            lines = [line.strip().split() for line in f]

        sequences = []
        seq_id_dict = {}
        for line in lines:
            sequences.append(line[1])
            seq_id_dict[line[1]] = line[0]

    if args.silent is False:
        print('\n---> Starting predictions...\n')

    pred_dict = train_network.test_unlabeled_data(brnn_network, sequences, device,
                                                encoding_scheme=encoding_scheme, encoder=encoder, 
                                                print_frequency=print_frequency)

    if args.silent is False:
        print('\n---> Prediction complete!')

    if problem_type == 'classification':
        if dtype == 'sequence':
            # probabilistic_classification
            if probabilistic_classification:
                for seq, values in pred_dict.items():
                    softmax = np.exp(values[0])
                    pred_dict[seq] = list(np.around(softmax / np.sum(softmax), decimals=4))
            else:
                for seq, values in pred_dict.items():
                    pred_dict[seq] = [np.argmax(values[0])]
        else:
            for seq, values in pred_dict.items():
                int_vals = []
                for row in values[0]:
                    int_vals.append(np.argmax(row))
                pred_dict[seq] = int_vals
    else:
        if dtype == 'sequence':
            for seq, values in pred_dict.items():
                pred_dict[seq] = list(values[0])
        else:
            for seq, values in pred_dict.items():
                pred_dict[seq] = list(values[0].flatten())


    with open(output_file, 'w') as f:
        for seq, values in pred_dict.items():
            if excludeSeqID:
                out_str = seq + ' ' + ' '.join(map(str, values)) + '\n'
            else:
                out_str = seq_id_dict[seq] + ' ' + seq + ' ' + ' '.join(map(str, values)) + '\n'
            f.write(out_str)
