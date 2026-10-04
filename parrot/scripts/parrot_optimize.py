#!/usr/bin/env python
"""
Usage: $ parrot-optimize data_file output_network <flags>

Search for good hyperparameters for a network architecture on a dataset, then
train and test a network with the best ones. See parrot/hyperparameter_search.py
for how the search works, and use the '-h' flag for the options.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse

import numpy as np

from parrot import hyperparameter_search, train_network
from parrot.architectures import build_network
from parrot.scripts import common

# Number of configurations tried by default
DEFAULT_TRIALS = 50


def main():
    parser = argparse.ArgumentParser(description='Search for good hyperparameters, then train and test a PARROT network '
                                                 'with them.')
    common.add_training_arguments(parser)
    common.add_hyperparameter_arguments(parser)

    search = parser.add_argument_group('hyperparameter search')
    search.add_argument('--trials', type=int, default=None,
                        help=f'number of hyperparameter configurations to try (default {DEFAULT_TRIALS})')
    search.add_argument('--max-iter', dest='max_iter', type=int, default=None,
                        help='same as --trials (kept so that older scripts keep working)')
    search.add_argument('--min-epochs', dest='min_epochs', type=int, default=None,
                        help='epochs every configuration is first trained for; the best ones continue for longer, up '
                             'to --epochs (default: --epochs / reduction-factor^2, at least 1)')
    search.add_argument('--reduction-factor', dest='reduction_factor', type=int,
                        default=hyperparameter_search.REDUCTION_FACTOR,
                        help='at each stage the best 1/reduction-factor of the configurations continue, for '
                             f'reduction-factor times as many epochs (default {hyperparameter_search.REDUCTION_FACTOR})')
    search.add_argument('--cv-folds', dest='cv_folds', type=int, default=None,
                        help='score each configuration by k-fold cross-validation over the training and validation '
                             'data, instead of on the validation set (about k times slower; for small datasets)')

    args = parser.parse_args()
    n_trials = check_search_arguments(args)

    # Any architecture hyperparameter given on the command line is fixed rather than searched
    fixed_hyperparameters = common.given_hyperparameters(args)

    run = common.prepare_training_run(args, n_folds=args.cv_folds)
    common.print_run_header('PARROT with hyperparameter search', run,
                            dict(architecture=args.architecture, **fixed_hyperparameters, batch_size=args.batch,
                                 batching=args.batching, epochs=args.epochs, trials=n_trials,
                                 cv_folds=args.cv_folds or 'none (validation set)'))

    if run.cv_sets is None:
        loaders = [(run.loader(run.train_set, shuffle=True), run.loader(run.val_set))]
    else:
        loaders = [(run.loader(fold_train, shuffle=True), run.loader(fold_held_out))
                   for fold_train, fold_held_out in run.cv_sets]

    space = hyperparameter_search.SearchSpace(args.architecture, fixed_hyperparameters)
    trainer = hyperparameter_search.TrialTrainer(space, loaders, len(run.encoder), run.task, run.device)
    trials_file = run.output_prefix + '_hyperparameter_trials.tsv'
    # Seeded by --seed (through NumPy's global generator), so the search is reproducible
    searcher = hyperparameter_search.HyperparameterSearch(
        space, trainer, n_trials, args.epochs, min_epochs=args.min_epochs, reduction_factor=args.reduction_factor,
        rng=np.random.default_rng(np.random.randint(2 ** 31)), silent=run.silent,
        on_progress=lambda progress: progress.write_trials(trials_file))
    best_trial = searcher.run()
    best = searcher.best_values(best_trial)
    write_hyperparameters(run.output_prefix + '_optimal_hyperparams.txt', args.architecture, best)

    if run.cv_sets is None:
        # The best trial already is a full training run on the training set, keeping its best epoch
        network, train_losses, val_losses = trainer.trained_network(best_trial)
    else:
        # Cross-validation trained on folds, so train the final network on the training set
        if not run.silent:
            print('Training with the best hyperparameters:')
        learning_rate = best.pop(train_network.LEARNING_RATE.name, train_network.LEARNING_RATE.default)
        network = build_network(args.architecture, len(run.encoder), run.task, best).to(run.device)
        train_losses, val_losses = train_network.train(network, run.loader(run.train_set, shuffle=True),
                                                       run.loader(run.val_set), run.task, run.device,
                                                       learning_rate=learning_rate, n_epochs=args.epochs,
                                                       stop_condition='iter', on_improvement=run.save,
                                                       verbose=args.verbose, silent=args.silent)

    common.finish_training_run(run, network, train_losses, val_losses)


def check_search_arguments(args):
    """Validate the search options, returning the number of trials

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


def write_hyperparameters(path, architecture, hyperparameters):
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
