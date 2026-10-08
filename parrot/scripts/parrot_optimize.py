#!/usr/bin/env python
"""
Usage: $ parrot-optimize <architecture> data_file output_network <flags>

Search for good hyperparameters for a network architecture on a dataset, then
train and test a network with the best ones. See parrot/hyperparameter_search.py
for how the search works. Each architecture is a subcommand with its own options
(see training_parser.py); use 'parrot-optimize -h' for the architectures, and
'parrot-optimize <architecture> -h' for an architecture's options.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
from collections import Counter
from typing import Any

from torch.utils.data import DataLoader

from parrot import hyperparameter_search, train_network
from parrot.architectures import HyperParameterValue, ParrotArchitecture, build_network, get_architecture
from parrot.reproducibility import random_generator
from parrot.scripts import arguments, common, training_parser

PROGRAM_NAME = 'parrot-optimize'

# Number of configurations tried by default
DEFAULT_TRIALS = 50


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments of parrot-optimize: one subcommand per architecture

    Returns
    -------
    argparse.ArgumentParser
    """
    return training_parser.build_parser(PROGRAM_NAME, 'Search for good hyperparameters, then train and test a PARROT '
                                        'network with them.', training_loop_options=[successive_halving_options()],
                                        command_options=[search_options()], describe_search=True)


def search_options() -> argparse.ArgumentParser:
    """The options of the search itself (every architecture)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('hyperparameter search')
    group.add_argument('--trials', type=int, default=None, metavar='trials',
                       help=f'number of hyperparameter configurations to try (default {DEFAULT_TRIALS})')
    group.add_argument('--max-iter', dest='max_iter', type=int, default=None, metavar='trials',
                       help=f'same as --trials {common.LEGACY_NOTE}')
    group.add_argument('--cv-folds', dest='cv_folds', type=int, default=None, metavar='folds',
                       help='score each configuration by k-fold cross-validation over the training and validation '
                            'data, instead of on the validation set (about k times slower; for small datasets)')
    return options


def successive_halving_options() -> argparse.ArgumentParser:
    """How long configurations are trained for at each stage of the search (networks trained by gradient descent)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('hyperparameter search')
    group.add_argument('--min-epochs', dest='min_epochs', type=int, default=None, metavar='epochs',
                       help='epochs every configuration is first trained for; the best ones continue for longer, up '
                            'to --epochs (default: --epochs / reduction-factor^2, at least 1)')
    group.add_argument('--reduction-factor', dest='reduction_factor', type=int, metavar='factor',
                       default=hyperparameter_search.REDUCTION_FACTOR,
                       help='at each stage the best 1/reduction-factor of the configurations continue, for '
                            f'reduction-factor times as many epochs (default {hyperparameter_search.REDUCTION_FACTOR})')
    return options


def main() -> None:
    """parrot-optimize's entry point: run it, reporting mistakes in its input as a message (see arguments.run_command)."""
    arguments.run_command(PROGRAM_NAME, run)


def run() -> None:
    """Search for good hyperparameters, then save and report on a network trained with the best ones."""
    args = training_parser.parse_arguments(build_parser())
    arguments.print_header(PROGRAM_NAME, args.silent)
    n_trials = check_search_arguments(args)

    # Any architecture hyperparameter given on the command line is fixed rather than searched
    fixed_hyperparameters = training_parser.given_hyperparameters(args)

    run = common.prepare_training_run(args, PROGRAM_NAME, n_folds=args.cv_folds)
    settings: dict[str, Any] = dict(architecture=args.architecture, **fixed_hyperparameters)
    if get_architecture(args.architecture).trained_by_gradient_descent:
        settings.update(batch_size=args.batch, batching=args.batching, epochs=args.epochs)
    settings.update(trials=n_trials, cv_folds=args.cv_folds or 'none (validation set)')
    description = get_architecture(args.architecture).description
    common.print_run_header(f'Searching hyperparameters for {args.architecture}: {description}', run, settings)

    space = hyperparameter_search.SearchSpace(args.architecture, fixed_hyperparameters)
    trainer = hyperparameter_search.TrialTrainer(space, search_loaders(run), len(run.encoder), run.task, run.device)
    searcher, best_trial = run_search(args, run, space, trainer, n_trials)
    best = searcher.best_values(best_trial)
    hyperparameters_file = run.output_prefix + '_optimal_hyperparams.txt'
    write_hyperparameters(hyperparameters_file, args.architecture, best)

    if run.cv_sets is None:
        # The best trial already is a full training run on the training set, keeping its best epoch
        network, train_losses, val_losses = trainer.trained_network(best_trial)
    else:
        network, train_losses, val_losses = train_with_best_values(args, run, best)

    training = dict(learning_rate=best.get(train_network.LEARNING_RATE.name, train_network.LEARNING_RATE.default),
                    epochs=args.epochs, stop='iter')
    common.finish_training_run(run, network, train_losses, val_losses, training,
                               search=describe_search(args, searcher, best_trial),
                               other_files=[hyperparameters_file, trials_file(run)])


def search_loaders(run: common.TrainingRun) -> list[tuple[DataLoader, DataLoader]]:
    """The (training, validation) loaders configurations are scored on: the validation split, or each fold

    Parameters
    ----------
    run : TrainingRun

    Returns
    -------
    list of (DataLoader, DataLoader)
    """
    if run.cv_sets is None:
        return [(run.loader(run.train_set, shuffle=True), run.loader(run.val_set))]
    return [(run.loader(fold_train, shuffle=True), run.loader(fold_held_out))
            for fold_train, fold_held_out in run.cv_sets]


def run_search(args: argparse.Namespace, run: common.TrainingRun, space: hyperparameter_search.SearchSpace,
               trainer: hyperparameter_search.TrialTrainer,
               n_trials: int) -> tuple[hyperparameter_search.HyperparameterSearch, hyperparameter_search.Trial]:
    """Run the search, writing the trials table after every step so far

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    run : TrainingRun
    space : SearchSpace
    trainer : TrialTrainer
    n_trials : int
        Number of configurations to try

    Returns
    -------
    HyperparameterSearch
        The finished search
    Trial
        Its best trial
    """
    searcher = hyperparameter_search.HyperparameterSearch(
        space, trainer, n_trials, args.epochs, min_epochs=args.min_epochs, reduction_factor=args.reduction_factor,
        rng=random_generator(), silent=run.silent, on_progress=lambda progress: progress.write_trials(trials_file(run)))
    return searcher, searcher.run()


def trials_file(run: common.TrainingRun) -> str:
    """The file every trial is recorded in, as the search goes."""
    return run.output_prefix + '_hyperparameter_trials.tsv'


def describe_search(args: argparse.Namespace, searcher: hyperparameter_search.HyperparameterSearch,
                    best_trial: hyperparameter_search.Trial) -> list[tuple[str, str]]:
    """The hyperparameter search, for the network's info file: how it worked, what it searched, and what it chose

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    searcher : HyperparameterSearch
        The finished search
    best_trial : Trial
        The configuration chosen

    Returns
    -------
    list of (str, str)
        'name : value' entries
    """
    outcomes = Counter(trial.status for trial in searcher.trials)
    entries = [('method', 'successive halving, with new configurations proposed by a tree-structured Parzen '
                          'estimator (TPE) and a third drawn at random'),
               ('configurations', f'{len(searcher.trials)} ({outcomes[hyperparameter_search.COMPLETED]} trained for '
                                  f'every epoch, {outcomes[hyperparameter_search.STOPPED]} stopped early, '
                                  f'{outcomes[hyperparameter_search.FAILED]} failed)')]
    if len(searcher.stages) > 1:
        entries.append(('stages', f'{" -> ".join(str(epochs) for epochs in searcher.stages)} epochs; the best '
                                  f'1/{searcher.reduction_factor} of each stage continued to the next'))
    entries.append(('scored by', f'average lowest validation loss over {args.cv_folds}-fold cross-validation'
                                 if args.cv_folds else 'lowest validation loss'))
    for hyperparameter in searcher.space.dimensions:
        entries.append((f'searched {hyperparameter.name}', training_parser.search_description(hyperparameter)
                        .removeprefix('searched over ')))
    for name, value in searcher.space.fixed.items():
        entries.append((f'fixed {name}', hyperparameter_search.format_value(value)))
    entries.append(('chosen', f'trial {best_trial.number}: {hyperparameter_search.format_hyperparameters(best_trial.values)}'))
    return entries


def train_with_best_values(args: argparse.Namespace, run: common.TrainingRun,
                           best: dict[str, HyperParameterValue]) -> tuple[ParrotArchitecture, list[float], list[float]]:
    """Train the final network on the training set with the best hyperparameters (after a cross-validated search)

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    run : TrainingRun
    best : dict
        The best configuration (including the learning rate)

    Returns
    -------
    ParrotArchitecture
        The trained network
    list of float
        Training loss per epoch
    list of float
        Validation loss per epoch
    """
    if not run.silent:
        print('Training with the best hyperparameters:')
    architecture_values = dict(best)
    learning_rate = float(architecture_values.pop(train_network.LEARNING_RATE.name,
                                                  train_network.LEARNING_RATE.default))
    network = build_network(args.architecture, len(run.encoder), run.task, architecture_values).to(run.device)
    train_losses, val_losses = train_network.train(network, run.loader(run.train_set, shuffle=True),
                                                   run.loader(run.val_set), run.task, run.device,
                                                   learning_rate=learning_rate, n_epochs=args.epochs,
                                                   stop_condition='iter', on_improvement=run.save,
                                                   verbose=args.verbose, silent=args.silent)
    return network, train_losses, val_losses


def check_search_arguments(args: argparse.Namespace) -> int:
    """Validate the search options, returning the number of trials

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments

    Returns
    -------
    int
        The number of configurations to try

    Raises
    ------
    ValueError
        If any search option is invalid
    """
    if args.trials is not None and args.max_iter is not None:
        raise ValueError('Give either --trials or --max-iter (an older name for it), not both')
    n_trials = next((value for value in (args.trials, args.max_iter) if value is not None), DEFAULT_TRIALS)
    if n_trials < 1:
        raise ValueError(f'--trials must be at least 1, got {n_trials}')
    if args.min_epochs is not None and not 1 <= args.min_epochs <= args.epochs:
        raise ValueError(f'--min-epochs must be between 1 and --epochs ({args.epochs}), got {args.min_epochs}')
    if args.reduction_factor < 2:
        raise ValueError(f'--reduction-factor must be at least 2, got {args.reduction_factor}')
    if args.cv_folds is not None and args.cv_folds < 2:
        raise ValueError(f'--cv-folds must be at least 2, got {args.cv_folds}')
    return n_trials


def write_hyperparameters(path: str, architecture: str, hyperparameters: dict[str, Any]) -> None:
    """Record the chosen hyperparameters as "<name>: <value>" lines

    Parameters
    ----------
    path : str
        File to write
    architecture : str
        Name of the architecture
    hyperparameters : dict
        Hyperparameter names and values (including the learning rate)
    """

    with open(path, 'w') as f:
        f.write(f'architecture: {architecture}\n')
        for name, value in hyperparameters.items():
            f.write(f'{name}: {hyperparameter_search.format_value(value)}\n')


if __name__ == '__main__':
    main()
