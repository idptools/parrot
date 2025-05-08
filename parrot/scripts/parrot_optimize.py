#!/usr/bin/env python
"""
Usage: $ parrot-optimize data_file output_network <flags>
  
Driver script for finding optimal hyperparameters for a bidirectional recurrent 
neural network on a given dataset, then training a network with those parameters
For more information on usage, use the '-h' flag.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from parrot import process_input_data as pid
from parrot import brnn_architecture
from parrot import train_network
from parrot import brnn_plot
from parrot import bayesian_optimization
from parrot.tools import validate_args
from parrot.tools import dataset_warnings

# Parse the command line arguments
def main():

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Train and test a bi-directional RNN using entire sequence.')

    parser.add_argument('data_file', help='path to tsv file with format: <idx> <sequence> <data>')

    parser.add_argument('output_network', help='location to save the trained network')

    parser.add_argument('-d', '--datatype', metavar='dtype', type=str, required=True,
                        help="REQUIRED. Format of the input data file, must be 'sequence' or 'residues'")

    parser.add_argument('-c', '--classes', type=int, metavar='num_classes', required=True,
                        help='REQUIRED. Number of output classes, for regression put 1')

    parser.add_argument('-b', '--batch', default=32, type=int, metavar='batch_size',
                        help='size of training batch (def=32)')

    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='num_epochs',
                        help='number of training epochs (def=100)')

    parser.add_argument('--max-iter', default=50, type=int, metavar='max_iter',
                        help='Maximum number of iterations for the optimization procedure (def=50)')

    parser.add_argument('--split', default='', metavar='split_file', type=str,
                        help="file indicating how to split datafile into training, validation, and testing sets")

    parser.add_argument('--set-fractions', nargs=3, default=[0.7, 0.15, 0.15], type=float,
                        dest='setFractions', metavar=('train', 'val', 'test'),
                        help='Proportion of dataset that should be divided into training, validation, and test sets')

    parser.add_argument('--encode', default='onehot', type=str, metavar='encoding_scheme',
                        help="'onehot' (default), 'biophysics', or specify a path to a user-created scheme")

    parser.add_argument('--exclude-seq-id', dest='excludeSeqID', action='store_true',
                        help='use if data_file lacks sequence IDs in the first column of each line')

    parser.add_argument('--probabilistic-classification', dest='probabilistic_classification',
                        action='store_true', help='Optional implementation for sequence classificaion')

    parser.add_argument('--include-figs', dest='include_figs', action='store_true',
                        help='Generate figures from training results and save to same location as network')

    parser.add_argument('--no-stats', dest='ignore_metrics', action='store_true',
                        help='If passed, do not output a perfomance stats file.')

    parser.add_argument('--force-cpu', dest='forceCPU', action='store_true',
                        help='force network to train on CPU, even if GPU is available')

    parser.add_argument('--ignore-warnings', '-w', dest='ignore_warnings', action='store_true',
                        help='Do not display warnings for dataset structure')

    parser.add_argument('--save-splits', dest='save_splits', action='store_true',
                        help='Save a split-file using the random splits from this run')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Flag which, if provided, causes output to terminal to be more descriptive')

    parser.add_argument('--silent', action='store_true',
                        help="Flag which, if provided, ensures no output is generated to the terminal")

    args = parser.parse_args()

    # Hyper-parameters
    batch_size = args.batch
    num_epochs = args.epochs
    max_iterations = args.max_iter

    # Data format
    dtype = args.datatype
    num_classes = args.classes

    # Other flags
    split_file = args.split
    encode = args.encode
    verbose = args.verbose
    silent = args.silent
    forceCPU = args.forceCPU
    setFractions = args.setFractions
    excludeSeqID = args.excludeSeqID
    probabilistic_classification = args.probabilistic_classification
    include_figs = args.include_figs
    ignore_warnings = args.ignore_warnings
    ignore_metrics = args.ignore_metrics
    save_splits = args.save_splits

    # Device configuration
    if forceCPU:
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###############################################################################
    ################    Validate arguments and initialize:      ###################

    # Ensure that provided data file exists
    data_file = validate_args.check_file_exists(args.data_file, 'Datafile')

    # Extract output directory and output prediction file name
    network_file = os.path.abspath(args.output_network)
    filename_prefix, output_dir = validate_args.split_file_and_directory(network_file)

    # If provided, check that split_file exists
    if split_file != '':
        split_file = validate_args.check_file_exists(split_file, 'Split-file')
    else:
        split_file = None

    # If specified, get location where randomly generated train/val/test splits will be saved
    if save_splits:
        save_splits_output = filename_prefix + '_split_file.txt'
    else:
        save_splits_output = None

    # Set encoding scheme and/or validate user scheme
    encoding_scheme, encoder, input_size = validate_args.set_encoding_scheme(encode)

    # Initialize network as classifier or regressor
    problem_type, collate_function = validate_args.set_ml_task(num_classes, dtype)

    # Ensure that network hyperparams (not being optimized) are valid
    validate_args.check_positive(num_epochs, 'Number of epochs')
    validate_args.check_positive(batch_size, 'Batch size')

    # Ensure that the sum of setFractions adds up to 1
    for frac in setFractions:
        validate_args.check_between_zero_and_one(frac, 'Set fractions')
    if sum(setFractions) != 1.0:
        raise ValueError('Set fractions must sum to 1.')

    # Ensure that task is binary sequence classification if
    # probabilistic_classfication is set
    if probabilistic_classification:
        if dtype != 'sequence' or num_classes < 2:
            raise ValueError('Probabilistic classification only implemented for sequence classification')

    # Set ignore_warnings to True if --silent is provided
    if silent:
        ignore_warnings = True

    ###############################################################################
    ################################  Main code  ##################################

    # Split data
    cvs, train, val, test = pid.split_data_cv(data_file, datatype=dtype, problem_type=problem_type,
                                            num_classes=num_classes, excludeSeqID=excludeSeqID,
                                            split_file=split_file, encoding_scheme=encoding_scheme,
                                            encoder=encoder, ignoreWarnings=ignore_warnings,
                                            percent_val=setFractions[1], percent_test=setFractions[2],
                                            save_splits_output=save_splits_output)

    # Assess batch size compared to training set size
    if not ignore_warnings:
        dataset_warnings.eval_batch_size(batch_size, len(cvs[0][0]))

    # Convert CV datasets to dataloaders
    cv_loaders = []
    for cv_train, cv_val in cvs:
        cv_train_loader = torch.utils.data.DataLoader(dataset=cv_train, batch_size=batch_size,
                                                    collate_fn=collate_function, shuffle=True)
        cv_val_loader = torch.utils.data.DataLoader(dataset=cv_val, batch_size=batch_size,
                                                    collate_fn=collate_function, shuffle=False)
        cv_loaders.append((cv_train_loader, cv_val_loader))

    # Output to std out
    # TODO move to tools.cli
    if silent is False:
        print()
        print("PARROT with hyperparameter optimization")
        print("---------------------------------------")
        if verbose:
            print('Train on:\t%s' % device)
            print("Datatype:\t%s" % dtype)
            print("ML Task:\t%s" % problem_type)
            print("Batch size:\t%d" % batch_size)
            print("Number of epochs:\t%d" % num_epochs)
            print("Number of optimization iterations:\t%d\n" % max_iterations)

    # Optimization procedure
    optimizer = bayesian_optimization.BayesianOptimizer(cv_loaders, input_size, num_epochs,
                                                        num_classes, dtype, network_file,
                                                        max_iterations, device, silent)

    best_hyperparams = optimizer.optimize()
    lr = 10**best_hyperparams[0]
    nl = int(best_hyperparams[1])
    hs = int(best_hyperparams[2])

    # Save these hyperparamters to a file so that the user has a record
    # TODO: move to helper function
    params_file = filename_prefix + '_optimal_hyperparams.txt'
    with open(params_file, 'w') as f:
        f.write('Learning rate: %.5f\n' % lr)
        f.write('Num Layers: %d\n' % nl)
        f.write('Hidden vector size: %d\n' % hs)

    # Use these best hyperparams to train the network from scratch using the entire train/val sets
    # Add data to dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                            batch_size=batch_size,
                                            collate_fn=collate_function,
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val,
                                            batch_size=batch_size,
                                            collate_fn=collate_function,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                            batch_size=1,		# Set test batch size to 1
                                            collate_fn=collate_function,
                                            shuffle=False)

    # Initialize network:
    if dtype == 'sequence':
        brnn_network = brnn_architecture.BRNN_MtO(input_size, hs, nl, num_classes, device).to(device)
    else:  # dtype == 'residues'
        brnn_network = brnn_architecture.BRNN_MtM(input_size, hs, nl, num_classes, device).to(device)

    # Train network
    if silent is False:
        print('Training with optimal hyperparams:')
    train_loss, val_loss = train_network.train(brnn_network, train_loader, val_loader, datatype=dtype,
                                            problem_type=problem_type, weights_file=network_file, 
                                            stop_condition='iter', device=device, learn_rate=lr, 
                                            n_epochs=num_epochs*2, verbose=verbose, silent=silent)

    if include_figs:  # Plot training & validation loss per epoch
        brnn_plot.training_loss(train_loss, val_loss, output_file_prefix=filename_prefix)

    # Test network
    test_loss, test_set_predictions = train_network.test_labeled_data(brnn_network, test_loader,
                                                    datatype=dtype, problem_type=problem_type,
                                                    weights_file=network_file, num_classes=num_classes,
                                                    probabilistic_classification=probabilistic_classification,
                                                    include_figs=include_figs, device=device,
                                                    output_file_prefix=filename_prefix)
    if silent is False:
        print('\nTest Loss: %.4f' % test_loss)

    # Output performance metrics
    if not ignore_metrics:
        brnn_plot.write_performance_metrics(test_set_predictions, dtype, problem_type,
                                        probabilistic_classification, filename_prefix)
        
    # Output the test set predictions to a text file
    brnn_plot.output_predictions_to_file(test_set_predictions, excludeSeqID, encoding_scheme,
                                        probabilistic_classification, encoder, output_file_prefix=filename_prefix)
