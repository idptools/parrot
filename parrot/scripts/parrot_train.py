#!/usr/bin/env python
"""
Usage: $ parrot-train <architecture> --input-file <datafile> --output-network <name> --split <method or split-file>
       (--regression | --classification --num-classes N) <flags>

Driver script for training a network with user-specified hyperparameters. Each
architecture is a subcommand with its own options (see training_parser.py);
use 'parrot-train -h' for the architectures, and 'parrot-train <architecture> -h'
for an architecture's options.

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

from parrot import train_network
from parrot.architectures import build_network
from parrot.scripts import arguments, common, training_artifacts, training_parser
from parrot.training_control import TrainingControlState

PROGRAM_NAME = 'parrot-train'

# With --stop auto, --epochs is how many epochs performance must stagnate for, so beyond this many epochs
# training can take a long time (a rule of thumb, warned about rather than refused)
LONG_AUTO_STOP_EPOCHS = 10


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-train: one subcommand per architecture

    Returns
    -------
    argparse.ArgumentParser
    """
    return training_parser.build_parser(PROGRAM_NAME, 'Train and test a PARROT network with user-specified '
                                        'hyperparameters.', training_loop_options=[learning_options()])


def learning_options() -> argparse.ArgumentParser:
    """parrot-train's own options for networks trained by gradient descent: the learning rate and when to stop

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('training')
    group.add_argument('-lr', '--learning-rate', default=train_network.LEARNING_RATE.default, type=float,
                       metavar='learning_rate',
                       help=f'initial learning rate of Adam, fixed unless LR control is enabled '
                            f'(default {train_network.LEARNING_RATE.default})')
    group.add_argument('--stop', default='iter', choices=train_network.STOP_CONDITIONS,
                       help="training stop condition: 'iter' trains for --epochs epochs (default); 'auto' trains "
                            "until --epochs epochs in a row have not lowered the validation loss by at least "
                            f"{100 * train_network.AUTO_STOP_RELATIVE_IMPROVEMENT:g}%% (with --uncertainties, whose "
                            'loss can be negative, by at least '
                            f'{train_network.AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT:.2g} per labelled value, a '
                            f'{100 * train_network.AUTO_STOP_RELATIVE_IMPROVEMENT:g}%% gain in likelihood)')
    return options


def main() -> None:
    """parrot-train's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Train a network with the given hyperparameters, save it, and report its performance on the test set."""
    args = training_parser.parse_arguments(build_parser())
    arguments.print_header(PROGRAM_NAME, args.silent)

    # Check the training-specific arguments before any data is loaded
    arguments.check_fraction(args.learning_rate, 'Learning rate')
    training_options = training_parser.training_options_from_args(args)
    control_state = TrainingControlState() if 'training_options' in training_options else None
    hyperparameters = training_parser.hyperparameters_from_args(args)
    output_prefix = os.path.splitext(os.path.abspath(common.network_path(args.output_network)))[0]
    checkpoint_path, log_path = training_artifacts.artifact_paths(args, output_prefix)
    if args.stop == 'auto' and args.epochs > LONG_AUTO_STOP_EPOCHS and not args.silent:
        print(f"Warning: Stop condition is set to 'auto' and num_epochs > {LONG_AUTO_STOP_EPOCHS}. "
              'Network training may take a long time.\n')

    run = common.prepare_training_run(args, PROGRAM_NAME)
    network = build_network(args.architecture, len(run.encoder), run.task, hyperparameters).to(run.device)
    if not run.silent:
        flags = {hyperparameter.name: hyperparameter.cli_flag for hyperparameter in network.hyperparameters}
        for name, reason in network.settings_without_effect().items():
            print(f'Note: {flags[name]} has no effect here ({reason}), so it is recorded but not used')

    settings: dict[str, Any] = dict(architecture=args.architecture, **hyperparameters)
    if network.trained_by_gradient_descent:
        settings.update(learning_rate=args.learning_rate, epochs=args.epochs, stop=args.stop, batch_size=args.batch,
                        batching=args.batching)
        settings.update(training_parser.active_training_settings(training_options))
    if run.task.regression_loss != 'auto':
        settings['regression_loss'] = run.task.regression_loss
        if run.task.regression_loss == 'huber':
            settings['huber_delta'] = run.task.huber_delta
    common.print_run_header(f'Training {args.architecture}: {network.description}', run, settings)
    if not run.silent and network.trained_by_gradient_descent:
        print('Validation set loss per epoch:')

    timing = train_network.TrainingTiming()
    call_options = dict(training_options)
    if control_state is not None:
        call_options['control_state'] = control_state
    if checkpoint_path is not None or args.resume is not None or log_path is not None:
        call_options.update(checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every or 1,
                            resume_from=args.resume, training_log=log_path,
                            checkpoint_metadata=training_artifacts.run_metadata(args, run))
    train_losses, val_losses = train_network.train(network, run.loader(run.train_set, shuffle=True),
                                                   run.loader(run.val_set), run.task, run.device,
                                                   learning_rate=args.learning_rate, n_epochs=args.epochs,
                                                   stop_condition=args.stop, on_improvement=run.save,
                                                   verbose=args.verbose, silent=args.silent, timing=timing,
                                                   **call_options)

    training = dict(learning_rate=args.learning_rate, epochs=args.epochs, stop=args.stop)
    training.update(training_parser.active_training_settings(training_options))
    if control_state is not None:
        training.update(control_state.report())
    if checkpoint_path is not None:
        training['training_checkpoint'] = checkpoint_path
    if log_path is not None:
        training['continuous_training_log'] = log_path
    if args.resume is not None:
        training['resumed_from'] = os.path.abspath(args.resume)
    files = [path for path in (checkpoint_path, log_path) if path is not None]
    common.finish_training_run(run, network, train_losses, val_losses, training, timing=timing, other_files=files)


if __name__ == '__main__':
    main()
