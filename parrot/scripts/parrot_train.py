#!/usr/bin/env python
"""
Usage: $ parrot-train data_file output_network <flags>

Driver script for training a network with user-specified hyperparameters. For
more information on usage, use the '-h' flag.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse

from parrot import train_network
from parrot.architectures import build_network
from parrot.scripts import common
from parrot.tools import validate_args


def main():
    parser = argparse.ArgumentParser(description='Train and test a PARROT network with user-specified hyperparameters.')
    common.add_training_arguments(parser)
    common.add_hyperparameter_arguments(parser)

    parser.add_argument('-lr', '--learning-rate', default=train_network.LEARNING_RATE.default, type=float,
                        metavar='learning_rate', help=f'(def={train_network.LEARNING_RATE.default})')
    parser.add_argument('--stop', default='iter', metavar='stop_condition', type=str,
                        help="training stop condition: either 'auto' or 'iter' (default 'iter')")

    args = parser.parse_args()

    # Check the training-specific arguments before any data is loaded
    validate_args.check_between_zero_and_one(args.learning_rate, 'Learning rate')
    validate_args.check_stop_condition(args.stop, args.epochs)
    hyperparameters = common.hyperparameters_from_args(args)

    run = common.prepare_training_run(args)
    network = build_network(args.architecture, len(run.encoder), run.task, hyperparameters).to(run.device)

    common.print_run_header('PARROT with user-specified parameters', run,
                            dict(architecture=args.architecture, **hyperparameters,
                                 learning_rate=args.learning_rate, batch_size=args.batch, batching=args.batching))
    if not run.silent:
        if network.trained_by_gradient_descent:
            print("Validation set loss per epoch:")
        else:
            print(f"The '{args.architecture}' model is fitted in one step, so the learning rate, number of "
                  "epochs and stop condition are not used.")

    train_losses, val_losses = train_network.train(network, run.loader(run.train_set, shuffle=True),
                                                   run.loader(run.val_set), run.task, run.device,
                                                   learning_rate=args.learning_rate, n_epochs=args.epochs,
                                                   stop_condition=args.stop, on_improvement=run.save,
                                                   verbose=args.verbose, silent=args.silent)

    common.finish_training_run(run, network, train_losses, val_losses)
