"""
Hyperparameter search for any PARROT architecture (used by parrot-optimize).

The search space is the learning rate (for networks trained by gradient descent)
plus every hyperparameter the architecture marks as searchable: numbers over a
`search_range` (optionally on a log scale) and settings with a fixed set of
values over their `search_choices`. Hyperparameters fixed by the user are not
searched, and the rest keep their defaults. Each configuration is scored by its
best validation loss (or, with cross-validation, the average over folds of each
fold's best validation loss), exactly as parrot-train chooses its best epoch.

Two ideas make the search efficient, and need nothing beyond NumPy:

- Successive halving (Jamieson & Talwalkar 2016, AISTATS; in its asynchronous
  form ASHA, Li et al. 2020, MLSys). Every configuration is first trained for a
  few epochs. Once enough configurations have reached a stage, the best
  1/REDUCTION_FACTOR of them continue for REDUCTION_FACTOR times as many
  epochs, carrying on from where they stopped, up to the full number of epochs.
  Poor configurations are therefore abandoned early, and most of the training
  goes to the promising ones.
- Model-based sampling with a tree-structured Parzen estimator (TPE; Bergstra
  et al. 2011, NeurIPS), in the multivariate form used by BOHB (Falkner et al.
  2018, ICML). After PARROT's default hyperparameters and a space-filling initial
  design, the configurations scored at the furthest stage with enough results
  are split into the best GOOD_FRACTION and the rest, a kernel density estimate
  is fitted to each, candidates are drawn from the density of good
  configurations, and the candidate with the highest ratio of good to bad
  density is tried next. A fraction of new configurations is always drawn at
  random, so the whole space keeps being explored.

The search space is defined in search_space.py and the TPE model in
tpe_sampler.py. This module trains configurations stage by stage
(TrialTrainer) and runs the search that ties everything together
(HyperparameterSearch).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from parrot import train_network
from parrot.architectures import HyperParameterValue, ParrotArchitecture, build_network
from parrot.devices import cpu_copy
from parrot.search_space import SearchSpace
from parrot.tasks import Task
from parrot.tpe_sampler import TPESampler

# Successive halving: each stage keeps the best 1/REDUCTION_FACTOR of its configurations and trains them
# REDUCTION_FACTOR times as long
REDUCTION_FACTOR = 3

# Fraction of new configurations drawn at random rather than from the TPE model, so the whole space keeps
# being explored (BOHB's default random_fraction; Falkner et al. 2018)
RANDOM_FRACTION = 1 / 3

# How many times a new configuration is drawn before looking for one that has not been tried
MAX_DRAWS_PER_TRIAL = 100

# A search space with at most this many configurations (every hyperparameter a whole number or a choice)
# is listed in full to find the ones not yet tried
MAX_LISTED_CONFIGURATIONS = 100_000

# The space-filling initial design tries this many configurations, or two per searched hyperparameter if
# that is more (PARROT's choice: enough to cover the space coarsely before the TPE model takes over)
MIN_INITIAL_DESIGN = 8

# How much of an error message is kept as a failed trial's reason (so the trials table stays readable)
MAX_FAILURE_MESSAGE_LENGTH = 200

# The status of a trial
WAITING = 'waiting'        # trained to the end of a stage, waiting to be promoted to the next one
STOPPED = 'stopped'        # not promoted (the search is over, or it can never be in the best 1/REDUCTION_FACTOR)
COMPLETED = 'completed'    # trained for the full number of epochs
FAILED = 'failed'          # training raised an error (see Trial.failure_reason)


def format_value(value: HyperParameterValue) -> str:
    """Format a hyperparameter value for printing (floats to 6 significant figures).

    Parameters
    ----------
    value : int, float, str or bool

    Returns
    -------
    str
    """
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def format_hyperparameters(values: Mapping[str, HyperParameterValue]) -> str:
    """Format hyperparameter values as 'name=value' pairs for printing.

    Parameters
    ----------
    values : dict
        Hyperparameter values by name

    Returns
    -------
    str
    """
    return '  '.join(f'{name}={format_value(value)}' for name, value in values.items())


def stage_epochs(max_epochs: int, min_epochs: int, reduction_factor: int = REDUCTION_FACTOR) -> list[int]:
    """The number of epochs a configuration has been trained for at the end of each stage of successive halving

    Parameters
    ----------
    max_epochs : int
        Epochs at the final stage (a full training run)
    min_epochs : int
        Smallest number of epochs a configuration is trained for
    reduction_factor : int, optional
        Each stage trains reduction_factor times as long as the one before

    Returns
    -------
    list of int
        Increasing epoch counts ending with max_epochs, e.g. [3, 10, 30] for 30 and 3
    """
    if not 1 <= min_epochs <= max_epochs:
        raise ValueError(f'The minimum number of epochs must be between 1 and {max_epochs}, got {min_epochs}')
    if reduction_factor < 2:
        raise ValueError(f'The reduction factor must be at least 2, got {reduction_factor}')

    epochs = [float(max_epochs)]
    while epochs[0] / reduction_factor >= min_epochs:
        epochs.insert(0, epochs[0] / reduction_factor)
    return sorted({max(1, round(value)) for value in epochs})


# .................................................................
# Training a configuration

@dataclass
class FoldProgress(train_network.TrainingHistory):
    """A configuration's training on one fold, kept on the CPU while it waits for the next stage

    As well as the losses and best weights (see TrainingHistory), it keeps
    where training stopped, so training can continue exactly from there.

    Attributes
    ----------
    network_state : dict or None
        The network's weights where training stopped
    optimizer_state : dict or None
        The optimizer's state where training stopped
    """

    network_state: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None


@dataclass
class Trial:
    """One configuration tried by the search

    Attributes
    ----------
    number : int
        Trials are numbered from 1 in the order they were started
    values : dict
        The searched hyperparameter values (including the learning rate)
    origin : str
        'default' (PARROT's defaults), 'initial' (space-filling design),
        'model' (TPE) or 'random'
    stage_losses : list of float
        The score at the end of each stage reached (inf if training failed or diverged)
    epochs : int
        Epochs trained so far
    status : str
        WAITING, STOPPED, COMPLETED or FAILED (see the constants at the top of this module)
    failure_reason : str
        Why training failed (empty unless the status is FAILED)
    folds : list of FoldProgress
        Training progress on each fold (dropped once the trial cannot continue)
    """

    number: int
    values: dict[str, HyperParameterValue]
    origin: str
    stage_losses: list[float] = field(default_factory=list)
    epochs: int = 0
    status: str = WAITING
    failure_reason: str = ''
    folds: list[FoldProgress] = field(default_factory=list)

    @property
    def status_text(self) -> str:
        """The status as written in the trials table and printed, e.g. 'completed' or 'failed: <reason>'."""
        if self.status == FAILED:
            return f'{FAILED}: {self.failure_reason}'
        return self.status


class TrialTrainer:
    """Trains configurations stage by stage, on the validation split or on cross-validation folds

    Between stages each configuration's weights, optimizer state and best
    weights are stored on the CPU, so configurations waiting for the next stage
    take no GPU memory, and training resumes exactly where it stopped.

    Attributes
    ----------
    space : SearchSpace
    loaders : list of (DataLoader, DataLoader)
        (training, validation) loaders: one pair for the validation split, or one per fold
    input_size : int
        Length of the per-residue encoding vectors
    task : Task
    device : torch.device
    """

    def __init__(self, space: SearchSpace, loaders: list[tuple[DataLoader, DataLoader]], input_size: int, task: Task,
                 device: str | torch.device) -> None:
        """
        Parameters
        ----------
        space : SearchSpace
            The space being searched
        loaders : list of (DataLoader, DataLoader)
            (training, validation) loaders, one pair per fold
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task being trained
        device : str or torch.device
            Device to train on
        """
        self.space = space
        self.loaders = loaders
        self.input_size = input_size
        self.task = task
        self.device = device

    def _build(self, trial: Trial) -> ParrotArchitecture:
        """A freshly initialised network for the trial's configuration, on the device

        Initialising the weights draws random numbers. The global random state
        is restored afterwards (fork_rng), so a configuration's starting weights
        do not depend on how many networks were built before it, and rebuilding
        a network to resume its training does not change the random numbers
        training then uses (e.g. to shuffle batches).

        Parameters
        ----------
        trial : Trial

        Returns
        -------
        ParrotArchitecture
        """
        with torch.random.fork_rng(devices=[]):
            network = build_network(self.space.architecture, self.input_size, self.task,
                                    self.space.architecture_values(trial.values))
        return network.to(self.device)

    def train_to(self, trial: Trial, epochs: int) -> float:
        """Continue training a trial until it has been trained for `epochs` epochs

        Parameters
        ----------
        trial : Trial
        epochs : int
            Epochs the trial should have been trained for (ignored for models
            fitted in one step)

        Returns
        -------
        float
            The trial's score: its best validation loss so far, averaged over folds
            (inf if any fold failed to improve on infinity, e.g. after diverging)
        """
        if not trial.folds:
            trial.folds = [FoldProgress() for _ in self.loaders]

        for progress, (train_loader, val_loader) in zip(trial.folds, self.loaders):
            if self.space.trained_by_gradient_descent:
                self._train_fold(trial, progress, train_loader, val_loader, epochs)
            elif progress.network_state is None:
                self._fit_fold(trial, progress, train_loader, val_loader)

        if self.space.trained_by_gradient_descent:
            trial.epochs = max(trial.epochs, epochs)
        losses = [progress.best_loss for progress in trial.folds]
        return float(np.mean(losses)) if all(math.isfinite(loss) for loss in losses) else math.inf

    def _train_fold(self, trial: Trial, progress: FoldProgress, train_loader: DataLoader, val_loader: DataLoader,
                    epochs: int) -> None:
        """Continue training a trial on one fold from where it stopped, up to `epochs` epochs

        Parameters
        ----------
        trial : Trial
        progress : FoldProgress
            The trial's progress on this fold (updated)
        train_loader, val_loader : DataLoader
            The fold's training and validation data
        epochs : int
            Epochs to have been trained for at the end
        """
        network = self._build(trial)
        if progress.network_state is not None:
            network.load_state_dict(progress.network_state)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.space.learning_rate(trial.values))
        if progress.optimizer_state is not None:
            optimizer.load_state_dict(progress.optimizer_state)
        criterion = train_network.build_loss_criterion(self.task)

        for _ in range(trial.epochs, epochs):
            train_loss = train_network.train_epoch(network, train_loader, self.task, criterion, optimizer, self.device)
            val_loss = train_network.evaluate_loss(network, val_loader, self.task, self.device)
            progress.record_epoch(train_loss, val_loss, network)

        progress.network_state = cpu_copy(network.state_dict())
        progress.optimizer_state = cpu_copy(optimizer.state_dict())

    def _fit_fold(self, trial: Trial, progress: FoldProgress, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Fit a model that is fitted in one step to one fold

        Parameters
        ----------
        trial : Trial
        progress : FoldProgress
            The trial's progress on this fold (updated)
        train_loader, val_loader : DataLoader
            The fold's training and validation data
        """
        network = self._build(trial)
        _, val_losses = train_network.fit_in_one_step(network, train_loader, val_loader, self.task, self.device,
                                                      silent=True)
        progress.best_loss = val_losses[0]
        progress.network_state = progress.best_weights = cpu_copy(network.state_dict())

    def trained_network(self, trial: Trial) -> tuple[ParrotArchitecture, list[float], list[float]]:
        """The trial's network with the weights of its best epoch, and its loss curves (validation split only).

        Parameters
        ----------
        trial : Trial
            A trial scored on the validation split

        Returns
        -------
        ParrotArchitecture
            The network, on the device
        list of float
            Training loss per epoch
        list of float
            Validation loss per epoch
        """
        if len(self.loaders) != 1:
            raise ValueError('Only a trial scored on the validation split has a single trained network')
        progress = trial.folds[0]
        network = self._build(trial)
        state = progress.best_weights if progress.best_weights is not None else progress.network_state
        assert state is not None
        network.load_state_dict(state)
        return network, list(progress.train_losses), list(progress.val_losses)


# .................................................................
# The search

class HyperparameterSearch:
    """Successive halving over configurations suggested by TPE

    Attributes
    ----------
    space : SearchSpace
    trainer : TrialTrainer
    n_trials : int
        Number of configurations to try
    stages : list of int
        Epochs trained by the end of each stage ([1] for models fitted in one step)
    reduction_factor : int
    trials : list of Trial
        Every trial so far, in the order started
    """

    def __init__(self, space: SearchSpace, trainer: TrialTrainer, n_trials: int, max_epochs: int,
                 min_epochs: int | None = None, reduction_factor: int = REDUCTION_FACTOR,
                 rng: np.random.Generator | None = None, silent: bool = False,
                 on_progress: Callable[['HyperparameterSearch'], None] | None = None) -> None:
        """
        Parameters
        ----------
        space : SearchSpace
        trainer : TrialTrainer
        n_trials : int
            Number of configurations to try (1 if nothing is searched)
        max_epochs : int
            Epochs of a full training run (the final stage)
        min_epochs : int, optional
            Epochs of the first stage (default max_epochs / reduction_factor**2, so
            there are three stages, but at least 1)
        reduction_factor : int, optional
            Each stage keeps the best 1/reduction_factor of its configurations and
            trains them reduction_factor times as long
        rng : numpy.random.Generator, optional
            Source of randomness for choosing configurations
        silent : bool, optional
            If True, print nothing
        on_progress : callable, optional
            Called with the search after every step (e.g. to save the trials so far)
        """
        if n_trials < 1:
            raise ValueError(f'The number of trials must be at least 1, got {n_trials}')
        self.space = space
        self.trainer = trainer
        self.n_trials = n_trials if len(space) else 1
        self.reduction_factor = reduction_factor
        if space.trained_by_gradient_descent:
            if min_epochs is None:
                min_epochs = max(1, round(max_epochs / reduction_factor ** 2))
            self.stages = stage_epochs(max_epochs, min_epochs, reduction_factor)
        else:
            self.stages = [1]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.silent = silent
        self.on_progress = on_progress
        self.sampler = TPESampler(space, self.rng)
        self.trials: list[Trial] = []

        # Points of the space-filling initial design not yet used (None until it is drawn)
        self._initial: list[np.ndarray] | None = None
        # Every configuration tried so far, as a tuple of values (see _key())
        self._tried: set[tuple[Any, ...]] = set()
        # The first error that made a configuration fail, reported if every configuration fails
        self._first_failure: BaseException | None = None

    def run(self) -> Trial:
        """Run the search

        Returns
        -------
        Trial
            The best trial: the lowest score among those trained for the full
            number of epochs
        """
        if not self.silent:
            self.print_plan()

        self._run_successive_halving()
        self._train_best_to_completion()

        for trial in self.trials:
            if trial.status == WAITING:
                trial.status = STOPPED
        if self.on_progress is not None:
            self.on_progress(self)

        best_trial = min((trial for trial in self.trials if trial.status == COMPLETED),
                         key=lambda trial: trial.stage_losses[-1])
        if not self.silent:
            print(f'\nBest configuration (trial {best_trial.number}, validation loss '
                  f'{best_trial.stage_losses[-1]:.4g}):\n  {format_hyperparameters(self.best_values(best_trial))}\n')
        return best_trial

    def best_values(self, trial: Trial) -> dict[str, HyperParameterValue]:
        """The learning rate (if trained by gradient descent) and every architecture hyperparameter of a trial.

        Parameters
        ----------
        trial : Trial

        Returns
        -------
        dict
        """
        values: dict[str, HyperParameterValue] = {}
        if self.space.trained_by_gradient_descent:
            values[train_network.LEARNING_RATE.name] = self.space.learning_rate(trial.values)
        values.update(self.space.architecture_values(trial.values))
        return values

    # ..... the main loop

    def _run_successive_halving(self) -> None:
        """Promote trials to their next stage and start new ones, until neither is possible

        Promotion comes first, so the most promising configurations are trained
        further as soon as they have earned it.
        """
        while True:
            trial = self._next_promotion()
            if trial is None:
                trial = self._start_trial()
            if trial is None:
                return
            self._train_stage(trial)

    def _start_trial(self) -> Trial | None:
        """Start a new trial, unless n_trials have been started or every configuration has been tried.

        Returns
        -------
        Trial or None
        """
        if len(self.trials) >= self.n_trials:
            return None

        suggestion = self._next_configuration()
        if suggestion is None:
            # Every possible configuration has been tried: from now on, search as if only these had been asked for
            self.n_trials = len(self.trials)
            return None

        values, origin = suggestion
        self._tried.add(self._key(values))
        trial = Trial(len(self.trials) + 1, values, origin)
        self.trials.append(trial)
        return trial

    def _train_best_to_completion(self) -> None:
        """Make sure at least one configuration has been trained for the full number of epochs

        Normally the best configurations reach the final stage during the
        search, but with few trials none may have. If so, the trial furthest
        along (the best, among equals) is trained on, one stage at a time.

        Raises
        ------
        RuntimeError
            If no configuration can be trained further (each failed or diverged)
        """
        while not any(trial.status == COMPLETED for trial in self.trials):
            continuing = [trial for trial in self.trials
                          if trial.status == WAITING and math.isfinite(trial.stage_losses[-1])]
            if not continuing:
                raise RuntimeError('Every configuration failed (or diverged), so no network could be trained') \
                    from self._first_failure

            furthest = max(len(trial.stage_losses) for trial in continuing)
            best = min((trial for trial in continuing if len(trial.stage_losses) == furthest),
                       key=lambda trial: trial.stage_losses[-1])
            self._train_stage(best)

    def _train_stage(self, trial: Trial) -> None:
        """Train a trial to the end of its next stage and record its score.

        A configuration that cannot be trained (e.g. one that does not suit this
        dataset, or one too large for the GPU) fails, and the search goes on.

        Parameters
        ----------
        trial : Trial
        """
        stage = len(trial.stage_losses)
        try:
            loss = self.trainer.train_to(trial, self.stages[stage])
        except (ValueError, torch.OutOfMemoryError) as error:
            loss = math.inf
            self._record_failure(trial, error)

        trial.stage_losses.append(loss)
        if trial.status != FAILED:
            trial.status = COMPLETED if stage == len(self.stages) - 1 else WAITING

        if not self.silent:
            self._print_stage(trial, stage, loss)
        if self.on_progress is not None:
            self.on_progress(self)

    def _record_failure(self, trial: Trial, error: BaseException) -> None:
        """Mark a trial as failed, and free what it was using.

        Parameters
        ----------
        trial : Trial
        error : Exception
            Why training failed
        """
        trial.status = FAILED
        trial.failure_reason = str(error).splitlines()[0][:MAX_FAILURE_MESSAGE_LENGTH] if str(error) else repr(error)
        trial.folds = []
        if self._first_failure is None:
            self._first_failure = error
        if isinstance(error, torch.OutOfMemoryError) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ..... successive halving

    def _ranked(self, stage: int) -> list[Trial]:
        """Trials that finished `stage`, best first.

        Parameters
        ----------
        stage : int

        Returns
        -------
        list of Trial
        """
        finished = [trial for trial in self.trials if len(trial.stage_losses) > stage]
        return sorted(finished, key=lambda trial: trial.stage_losses[stage])

    @staticmethod
    def _waiting_after(trial: Trial, stage: int) -> bool:
        """True if a trial has finished `stage` and is waiting to start the next one."""
        return trial.status == WAITING and len(trial.stage_losses) == stage + 1

    def _next_promotion(self) -> Trial | None:
        """The trial that should continue to its next stage now, if any

        A trial continues once it is in the best 1/reduction_factor of all
        trials that have finished its stage (furthest stages first). Trials that
        can never get there are stopped first (see _stop_hopeless_trials()).

        Returns
        -------
        Trial or None
        """
        self._stop_hopeless_trials()
        for stage in reversed(range(len(self.stages) - 1)):
            ranked = self._ranked(stage)
            for trial in ranked[:len(ranked) // self.reduction_factor]:
                if self._waiting_after(trial, stage) and math.isfinite(trial.stage_losses[stage]):
                    return trial
        return None

    def _stop_hopeless_trials(self) -> None:
        """Stop the waiting trials that can never be promoted, and drop their training state

        At most n_trials / reduction_factor trials can ever be promoted from a
        stage, so a trial ranked below that many others can never be (its rank
        can only fall as more trials finish the stage).
        """
        most_promoted = max(1, self.n_trials // self.reduction_factor)
        for stage in range(len(self.stages) - 1):
            for rank, trial in enumerate(self._ranked(stage)):
                if rank >= most_promoted and self._waiting_after(trial, stage):
                    trial.status = STOPPED
                    trial.folds = []

    # ..... choosing new configurations

    def _key(self, values: Mapping[str, HyperParameterValue]) -> tuple[Any, ...]:
        """A configuration as a tuple of values, to remember which configurations have been tried."""
        return tuple(values[hyperparameter.name] for hyperparameter in self.space.dimensions)

    def _is_new(self, values: Mapping[str, HyperParameterValue]) -> bool:
        """True if a configuration has not been tried yet."""
        return self._key(values) not in self._tried

    def _next_configuration(self) -> tuple[dict[str, HyperParameterValue], str] | None:
        """The next configuration to try and how it was chosen

        PARROT's defaults come first, then the space-filling initial design, then
        configurations drawn from the TPE model (or at random). If drawing keeps
        finding configurations already tried, a small space is listed in full to
        find one that has not been.

        Returns
        -------
        (dict, str) or None
            The configuration and its origin ('default', 'initial', 'model' or
            'random'), or None if every configuration has been tried
        """
        if not self.trials:
            return self.space.default_values(), 'default'

        initial = self._next_initial_configuration()
        if initial is not None:
            return initial, 'initial'

        drawn = self._draw_configuration()
        if drawn is not None:
            return drawn

        untried = self._random_untried_configuration()
        if untried is not None:
            return untried, 'random'
        return None

    def _next_initial_configuration(self) -> dict[str, HyperParameterValue] | None:
        """The next configuration of the space-filling initial design that has not been tried, if any

        The design is drawn the first time this is called (for the second
        configuration, after the defaults).

        Returns
        -------
        dict or None
        """
        if self._initial is None:
            n_initial = min(self.n_trials - 1, max(MIN_INITIAL_DESIGN, 2 * len(self.space)))
            self._initial = self.space.initial_design(n_initial, self.rng)

        while self._initial:
            values = self.space.from_point(self._initial.pop(0))
            if self._is_new(values):
                return values
        return None

    def _draw_configuration(self) -> tuple[dict[str, HyperParameterValue], str] | None:
        """A new configuration from the TPE model, or drawn at random, if one is found within MAX_DRAWS_PER_TRIAL draws

        Once enough configurations have been scored to fit the model, it is used
        for all but RANDOM_FRACTION of the draws.

        Returns
        -------
        (dict, str) or None
            The configuration and its origin ('model' or 'random')
        """
        model_data = self._model_data()
        for _ in range(MAX_DRAWS_PER_TRIAL):
            if model_data is not None and self.rng.random() >= RANDOM_FRACTION:
                for candidate in self.sampler.suggest(*model_data):
                    values = self.space.from_point(candidate)
                    if self._is_new(values):
                        return values, 'model'

            values = self.space.from_point(self.space.random_point(self.rng))
            if self._is_new(values):
                return values, 'random'
        return None

    def _random_untried_configuration(self) -> dict[str, HyperParameterValue] | None:
        """A configuration that has not been tried, chosen at random by listing every one (small spaces only)

        Returns
        -------
        dict or None
            None if the space has continuous numbers, is too large to list, or
            has no configurations left
        """
        options = self.space.discrete_values()
        if options is None or math.prod(len(values) for values in options) > MAX_LISTED_CONFIGURATIONS:
            return None

        untried = [combination for combination in itertools.product(*options) if combination not in self._tried]
        if not untried:
            return None
        names = [hyperparameter.name for hyperparameter in self.space.dimensions]
        return dict(zip(names, untried[self.rng.integers(len(untried))]))

    def _model_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Points and losses at the furthest stage with enough scored configurations to fit TPE, if any.

        Returns
        -------
        (numpy array, numpy array) or None
            [n X dimensions] points and their n losses
        """
        for stage in reversed(range(len(self.stages))):
            scored = [trial for trial in self.trials if len(trial.stage_losses) > stage]
            if self.sampler.model_can_be_fitted(len(scored)):
                return (np.array([self.space.to_point(trial.values) for trial in scored]),
                        np.array([trial.stage_losses[stage] for trial in scored]))
        return None

    # ..... reporting

    def print_plan(self) -> None:
        """Describe the search before it starts."""
        print(f'\nSearching {self.n_trials} configuration(s) of the {self.space.architecture} architecture')
        if self.space.dimensions:
            for hyperparameter in self.space.dimensions:
                if self.space.is_numeric(hyperparameter):
                    assert hyperparameter.search_range is not None
                    low, high = hyperparameter.search_range
                    scale = ', log scale' if hyperparameter.log_scale else ''
                    print(f'  {hyperparameter.name}: {format_value(low)} to {format_value(high)}{scale}')
                else:
                    choices = ', '.join(format_value(choice) for choice in hyperparameter.search_choices or ())
                    print(f'  {hyperparameter.name}: {choices}')
        else:
            print('  Every hyperparameter is fixed or not searchable, so a single network is trained.')
        if self.space.fixed:
            print(f'Fixed: {format_hyperparameters(self.space.fixed)}')
        if len(self.stages) > 1:
            print(f'Epochs per stage: {" -> ".join(str(epochs) for epochs in self.stages)} (the best '
                  f'1/{self.reduction_factor} of each stage continue to the next)')
        print()

    def _print_stage(self, trial: Trial, stage: int, loss: float) -> None:
        """Print one line about a trial that has just finished a stage.

        Parameters
        ----------
        trial : Trial
        stage : int
            The stage it finished
        loss : float
            Its score at that stage
        """
        stage_text = f'epochs {self.stages[stage]:4d}' if self.space.trained_by_gradient_descent else 'fitted'
        outcome = trial.status_text if trial.status == FAILED else f'val loss {loss:.4g}'
        print(f'  trial {trial.number:3d} ({trial.origin:7s})  {stage_text}  {outcome:16s}  '
              f'{format_hyperparameters(trial.values)}', flush=True)

    def write_trials(self, path: str) -> None:
        """Write every trial so far as a tab-separated table (one row per trial).

        Parameters
        ----------
        path : str
            File to write
        """
        names = [hyperparameter.name for hyperparameter in self.space.dimensions]
        if self.space.trained_by_gradient_descent:
            score_columns = [f'val_loss_{epochs}_epochs' for epochs in self.stages]
        else:
            score_columns = ['val_loss']
        with open(path, 'w') as handle:
            handle.write(f'# parrot-optimize trials for the {self.space.architecture} architecture\n')
            if self.space.fixed:
                handle.write(f'# fixed: {format_hyperparameters(self.space.fixed)}\n')
            handle.write('\t'.join(['trial', 'origin', 'status', 'epochs', *score_columns, *names]) + '\n')
            for trial in self.trials:
                scores = [format_value(trial.stage_losses[stage]) if stage < len(trial.stage_losses) else ''
                          for stage in range(len(score_columns))]
                handle.write('\t'.join([str(trial.number), trial.origin, trial.status_text, str(trial.epochs),
                                        *scores, *(format_value(trial.values[name]) for name in names)]) + '\n')
