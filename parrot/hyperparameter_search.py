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

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp, ndtr
from torch.utils.data import DataLoader

from parrot import train_network
from parrot.architectures import (HyperParameter, HyperParameterValue, ParrotArchitecture, build_network,
                                  get_architecture, resolve_hyperparameters)
from parrot.architectures.classical import ClassicalModel
from parrot.tasks import Task

# Successive halving: each stage keeps the best 1/REDUCTION_FACTOR of its configurations and trains them
# REDUCTION_FACTOR times as long
REDUCTION_FACTOR = 3

# TPE settings, as in BOHB (Falkner et al. 2018)
GOOD_FRACTION = 0.15            # fraction of scored configurations whose density counts as "good" (at least
                                # one more than the number of hyperparameters)
RANDOM_FRACTION = 1 / 3         # fraction of new configurations drawn at random rather than from the model
CANDIDATES_PER_SUGGESTION = 64  # candidates drawn from the good density for each new configuration
BANDWIDTH_FACTOR = 3.0          # candidates are drawn from the good density with its kernels widened this much
MIN_BANDWIDTH = 0.05            # narrowest kernel for a number, as a fraction of its search range
CATEGORY_SMOOTHING = 0.2        # probability a choice's kernel spreads over the other choices

# How many times a new configuration is drawn before looking for one that has not been tried
MAX_DRAWS_PER_TRIAL = 100

# A search space with at most this many configurations (every hyperparameter a whole number or a choice)
# is listed in full to find the ones not yet tried
MAX_LISTED_CONFIGURATIONS = 100_000


def format_value(value: HyperParameterValue) -> str:
    """Format a hyperparameter value for printing (floats to 6 significant figures)."""
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def format_hyperparameters(values: Mapping[str, HyperParameterValue]) -> str:
    """Format hyperparameter values as 'name=value' pairs for printing."""
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
# The search space

class SearchSpace:
    """The hyperparameters being searched, and conversions between their values and points

    A point has one coordinate per searched hyperparameter, in the order of
    `dimensions`: for a number, its position in the search range scaled to
    [0, 1] (on a log scale if declared; whole numbers each get an equal share of
    the range); for a setting with search choices, the index of the choice.

    Attributes
    ----------
    architecture : str
        Name of the architecture
    fixed : dict
        Architecture hyperparameter values fixed by the user (not searched)
    dimensions : list of HyperParameter
        The searched hyperparameters, learning rate first (for networks trained
        by gradient descent)
    trained_by_gradient_descent : bool
        Whether the architecture is trained epoch by epoch
    """

    def __init__(self, architecture: str, fixed: Mapping[str, HyperParameterValue] | None = None) -> None:
        """
        Parameters
        ----------
        architecture : str
            Name of a registered architecture
        fixed : dict, optional
            Architecture hyperparameters fixed at these values instead of searched
        """
        architecture_class = get_architecture(architecture)
        self.architecture = architecture
        self.fixed = dict(fixed or {})
        self.trained_by_gradient_descent = architecture_class.trained_by_gradient_descent
        searched = [hyperparameter for hyperparameter in architecture_class.hyperparameters
                    if hyperparameter.searchable and hyperparameter.name not in self.fixed]
        learning_rate = [train_network.LEARNING_RATE] if self.trained_by_gradient_descent else []
        self.dimensions: list[HyperParameter] = learning_rate + searched

    def __len__(self) -> int:
        return len(self.dimensions)

    @staticmethod
    def is_numeric(hyperparameter: HyperParameter) -> bool:
        """True for a hyperparameter searched over a range of numbers (rather than a set of choices)."""
        return hyperparameter.search_range is not None

    # ..... single coordinates

    @staticmethod
    def _to_unit(hyperparameter: HyperParameter, value: float) -> float:
        """Position of a value in its search range, scaled to [0, 1]."""
        assert hyperparameter.search_range is not None
        low, high = hyperparameter.search_range
        whole = hyperparameter.value_type is int
        if hyperparameter.log_scale:
            top = high + 1 if whole else high
            position = math.log(value + 0.5 if whole else value)
            return float(np.clip((position - math.log(low)) / (math.log(top) - math.log(low)), 0, 1))
        if whole:
            return (value - low + 0.5) / (high - low + 1)
        return (value - low) / (high - low)

    @staticmethod
    def _from_unit(hyperparameter: HyperParameter, unit: float) -> int | float:
        """The value at a position in [0, 1] of the search range."""
        assert hyperparameter.search_range is not None
        low, high = hyperparameter.search_range
        unit = float(np.clip(unit, 0.0, 1.0))
        if hyperparameter.value_type is int:
            if hyperparameter.log_scale:
                value = math.floor(math.exp(math.log(low) + unit * (math.log(high + 1) - math.log(low))))
            else:
                value = round(low - 0.5 + unit * (high - low + 1))
            return int(np.clip(value, math.ceil(low), math.floor(high)))
        if hyperparameter.log_scale:
            return float(math.exp(math.log(low) + unit * (math.log(high) - math.log(low))))
        return float(low + unit * (high - low))

    # ..... whole configurations

    def to_point(self, values: Mapping[str, HyperParameterValue]) -> np.ndarray:
        """The point of a configuration (a {name: value} dictionary of the searched hyperparameters)."""
        point = []
        for hyperparameter in self.dimensions:
            value = values[hyperparameter.name]
            if self.is_numeric(hyperparameter):
                point.append(self._to_unit(hyperparameter, float(value)))
            else:
                assert hyperparameter.search_choices is not None
                point.append(float(hyperparameter.search_choices.index(value)))
        return np.array(point, dtype=float)

    def from_point(self, point: Sequence[float] | np.ndarray) -> dict[str, HyperParameterValue]:
        """The configuration at a point."""
        values: dict[str, HyperParameterValue] = {}
        for hyperparameter, coordinate in zip(self.dimensions, point, strict=True):
            if self.is_numeric(hyperparameter):
                values[hyperparameter.name] = self._from_unit(hyperparameter, coordinate)
            else:
                assert hyperparameter.search_choices is not None
                values[hyperparameter.name] = hyperparameter.search_choices[int(coordinate)]
        return values

    def default_values(self) -> dict[str, HyperParameterValue]:
        """PARROT's default value of every searched hyperparameter (moved into its search range if outside it)."""
        point = []
        for hyperparameter in self.dimensions:
            if self.is_numeric(hyperparameter):
                assert hyperparameter.search_range is not None
                low, high = hyperparameter.search_range
                default = float(np.clip(float(hyperparameter.default), low, high))
                point.append(self._to_unit(hyperparameter, default))
            else:
                assert hyperparameter.search_choices is not None
                choices = hyperparameter.search_choices
                point.append(float(choices.index(hyperparameter.default) if hyperparameter.default in choices else 0))
        return self.from_point(point)

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        """A point drawn uniformly at random."""
        return np.array([rng.random() if self.is_numeric(hyperparameter)
                         else float(rng.integers(len(hyperparameter.search_choices or ())))
                         for hyperparameter in self.dimensions], dtype=float)

    def initial_design(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """n space-filling points: a Latin hypercube for numbers, and every choice used equally often

        In a Latin hypercube each number's range is cut into n equal slices and
        each slice is used by exactly one point, so the points spread over every
        range (unlike a one-at-a-time sweep or plain random points).
        """
        if n <= 0:
            return []
        columns = []
        for hyperparameter in self.dimensions:
            if self.is_numeric(hyperparameter):
                columns.append((rng.permutation(n) + rng.random(n)) / n)
            else:
                count = len(hyperparameter.search_choices or ())
                columns.append(rng.permutation(np.arange(n) % count).astype(float))
        return [np.array(row, dtype=float) for row in zip(*columns)] if columns else [np.zeros(0) for _ in range(n)]

    def discrete_values(self) -> list[list[HyperParameterValue]] | None:
        """Every value of each searched hyperparameter, if each takes a finite set of values (None otherwise)."""
        options: list[list[HyperParameterValue]] = []
        for hyperparameter in self.dimensions:
            if not self.is_numeric(hyperparameter):
                options.append(list(hyperparameter.search_choices or ()))
            elif hyperparameter.value_type is int:
                assert hyperparameter.search_range is not None
                low, high = hyperparameter.search_range
                options.append(list(range(math.ceil(low), math.floor(high) + 1)))
            else:
                return None
        return options

    def architecture_values(self, values: Mapping[str, HyperParameterValue]) -> dict[str, HyperParameterValue]:
        """Every architecture hyperparameter: searched values, then fixed values, then defaults (no learning rate)."""
        searched = {name: value for name, value in values.items() if name != train_network.LEARNING_RATE.name}
        return resolve_hyperparameters(get_architecture(self.architecture), {**self.fixed, **searched})

    @staticmethod
    def learning_rate(values: Mapping[str, HyperParameterValue]) -> float:
        """The learning rate of a configuration (PARROT's default if it is not searched)."""
        return float(values.get(train_network.LEARNING_RATE.name, train_network.LEARNING_RATE.default))


# .................................................................
# Model-based sampling (TPE)

class TPESampler:
    """Suggests configurations where good results have clustered (tree-structured Parzen estimator)

    Attributes
    ----------
    space : SearchSpace
    rng : numpy.random.Generator
    """

    def __init__(self, space: SearchSpace, rng: np.random.Generator) -> None:
        self.space = space
        self.rng = rng
        self._numeric = np.array([space.is_numeric(hyperparameter) for hyperparameter in space.dimensions])
        self._choice_counts = np.array([len(hyperparameter.search_choices or ()) for hyperparameter in space.dimensions])

    def n_good(self, n_observations: int) -> int:
        """How many of the best configurations make up the good density."""
        return max(len(self.space) + 1, math.ceil(GOOD_FRACTION * n_observations))

    def model_can_be_fitted(self, n_observations: int) -> bool:
        """Whether there are enough scored configurations at one stage to fit the densities

        As in BOHB, both the good and the bad density need at least one more
        configuration than there are hyperparameters, so neither collapses onto a
        few points.
        """
        return n_observations - self.n_good(n_observations) >= len(self.space) + 1

    def suggest(self, points: np.ndarray, losses: np.ndarray) -> np.ndarray:
        """The candidate point with the highest ratio of good to bad density

        Parameters
        ----------
        points : numpy array
            [n X dimensions] points scored at one stage
        losses : numpy array
            Their losses (inf for failed configurations)

        Returns
        -------
        numpy array
            [CANDIDATES_PER_SUGGESTION X dimensions] candidates, best first
        """
        n_good = self.n_good(len(points))
        order = np.argsort(losses, kind='stable')
        good, bad = points[order[:n_good]], points[order[n_good:]]

        good_bandwidth, bad_bandwidth = self._bandwidths(good), self._bandwidths(bad)
        candidates = self._sample(good, good_bandwidth * BANDWIDTH_FACTOR, CANDIDATES_PER_SUGGESTION)
        score = self._log_density(candidates, good, good_bandwidth) - self._log_density(candidates, bad, bad_bandwidth)
        return candidates[np.argsort(-score, kind='stable')]

    def _bandwidths(self, points: np.ndarray) -> np.ndarray:
        """Kernel width for each numeric coordinate (Scott's rule), clipped to [MIN_BANDWIDTH, 1]."""
        scott = points.std(axis=0) * len(points) ** (-1.0 / (len(self.space) + 4))
        return np.clip(scott, MIN_BANDWIDTH, 1.0)

    def _sample(self, centres: np.ndarray, bandwidth: np.ndarray, n: int) -> np.ndarray:
        """Draw n points from the kernel density centred on `centres`."""
        samples = centres[self.rng.integers(len(centres), size=n)].copy()
        for dimension in range(len(self.space)):
            if self._numeric[dimension]:
                samples[:, dimension] = self._truncated_normal(samples[:, dimension], bandwidth[dimension])
            else:
                count = self._choice_counts[dimension]
                switch = self.rng.random(n) < min(1 - 1 / count, CATEGORY_SMOOTHING * BANDWIDTH_FACTOR)
                offsets = self.rng.integers(1, count, size=n)
                samples[switch, dimension] = (samples[switch, dimension] + offsets[switch]) % count
        return samples

    def _truncated_normal(self, means: np.ndarray, width: float) -> np.ndarray:
        """Normal samples around `means`, redrawn until they fall in [0, 1] (clipped as a last resort)."""
        values = self.rng.normal(means, width)
        for _ in range(20):
            outside = (values < 0) | (values > 1)
            if not outside.any():
                break
            values[outside] = self.rng.normal(means[outside], width)
        return np.clip(values, 0, 1)

    def _log_density(self, x: np.ndarray, centres: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """Log of the kernel density estimate (a product of one kernel per coordinate) at each row of x."""
        log_kernels = np.zeros((len(x), len(centres)))
        for dimension in range(len(self.space)):
            xs, cs = x[:, dimension][:, None], centres[:, dimension][None, :]
            if self._numeric[dimension]:
                width = bandwidth[dimension]
                inside_mass = ndtr((1 - cs) / width) - ndtr(-cs / width)
                log_kernels += -0.5 * ((xs - cs) / width) ** 2 - math.log(width * math.sqrt(2 * math.pi)) \
                    - np.log(inside_mass)
            else:
                count = self._choice_counts[dimension]
                same = np.isclose(xs, cs)
                log_kernels += np.where(same, math.log(1 - CATEGORY_SMOOTHING), math.log(CATEGORY_SMOOTHING / (count - 1)))
        return logsumexp(log_kernels, axis=1) - math.log(len(centres))


# .................................................................
# Training a configuration

@dataclass
class FoldProgress:
    """A configuration's training on one fold, kept on the CPU while it waits for the next stage

    Attributes
    ----------
    network_state, optimizer_state : dict or None
        The network's weights and the optimizer's state where training stopped
    best_state : dict or None
        The weights from the epoch with the lowest validation loss
    best_loss : float
        That validation loss
    train_losses, val_losses : list of float
        Loss per epoch so far
    """

    network_state: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_loss: float = math.inf
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


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
        'waiting' (for promotion to the next stage), 'stopped' (not promoted),
        'completed' (trained for the full number of epochs) or 'failed: <reason>'
    folds : list of FoldProgress
        Training progress on each fold (dropped once the trial cannot continue)
    """

    number: int
    values: dict[str, HyperParameterValue]
    origin: str
    stage_losses: list[float] = field(default_factory=list)
    epochs: int = 0
    status: str = 'waiting'
    folds: list[FoldProgress] = field(default_factory=list)


def _to_cpu(value: Any) -> Any:
    """A copy of a (nested) state dictionary with every tensor on the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


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
        self.space = space
        self.loaders = loaders
        self.input_size = input_size
        self.task = task
        self.device = device

    def _build(self, trial: Trial) -> ParrotArchitecture:
        """A network for the trial's configuration, on the device (building it draws no random numbers)."""
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
            network = self._build(trial)
            if not self.space.trained_by_gradient_descent:
                if progress.network_state is None:
                    # Only scikit-learn models are fitted in one step
                    assert isinstance(network, ClassicalModel)
                    train_network.fit_in_one_step(network, train_loader, val_loader, self.task, self.device,
                                                  silent=True)
                    progress.best_loss = train_network.evaluate_loss(network, val_loader, self.task, self.device)
                    progress.network_state = progress.best_state = _to_cpu(network.state_dict())
                continue

            if progress.network_state is not None:
                network.load_state_dict(progress.network_state)
            optimizer = torch.optim.Adam(network.parameters(), lr=self.space.learning_rate(trial.values))
            if progress.optimizer_state is not None:
                optimizer.load_state_dict(progress.optimizer_state)
            criterion = train_network.build_loss_criterion(self.task)

            for _ in range(trial.epochs, epochs):
                progress.train_losses.append(train_network.train_epoch(network, train_loader, self.task, criterion,
                                                                       optimizer, self.device))
                val_loss = train_network.evaluate_loss(network, val_loader, self.task, self.device)
                progress.val_losses.append(val_loss)
                if val_loss < progress.best_loss:
                    progress.best_loss = val_loss
                    progress.best_state = _to_cpu(network.state_dict())

            progress.network_state = _to_cpu(network.state_dict())
            progress.optimizer_state = _to_cpu(optimizer.state_dict())

        trial.epochs = max(trial.epochs, epochs) if self.space.trained_by_gradient_descent else trial.epochs
        losses = [progress.best_loss for progress in trial.folds]
        return float(np.mean(losses)) if all(math.isfinite(loss) for loss in losses) else math.inf

    def trained_network(self, trial: Trial) -> tuple[ParrotArchitecture, list[float], list[float]]:
        """The trial's network with the weights of its best epoch (validation split only), and its loss curves."""
        if len(self.loaders) != 1:
            raise ValueError('Only a trial scored on the validation split has a single trained network')
        progress = trial.folds[0]
        network = self._build(trial)
        state = progress.best_state if progress.best_state is not None else progress.network_state
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
        self._initial: list[np.ndarray] = []
        self._tried: set[tuple[Any, ...]] = set()

    # ..... choosing new configurations

    def _key(self, values: Mapping[str, HyperParameterValue]) -> tuple[Any, ...]:
        return tuple(values[hyperparameter.name] for hyperparameter in self.space.dimensions)

    def _next_configuration(self) -> tuple[dict[str, HyperParameterValue], str] | None:
        """The next configuration to try and how it was chosen, or None if every configuration has been tried."""
        if not self.trials:
            return self.space.default_values(), 'default'

        if not self._initial and len(self.trials) == 1:
            n_initial = min(self.n_trials - 1, max(8, 2 * len(self.space)))
            self._initial = self.space.initial_design(n_initial, self.rng)
        while self._initial:
            values = self.space.from_point(self._initial.pop(0))
            if self._key(values) not in self._tried:
                return values, 'initial'

        for _ in range(MAX_DRAWS_PER_TRIAL):
            stage_points = self._model_data()
            if stage_points is not None and self.rng.random() >= RANDOM_FRACTION:
                for candidate in self.sampler.suggest(*stage_points):
                    values = self.space.from_point(candidate)
                    if self._key(values) not in self._tried:
                        return values, 'model'
            values = self.space.from_point(self.space.random_point(self.rng))
            if self._key(values) not in self._tried:
                return values, 'random'

        # Drawing keeps finding tried configurations: in a small space, list the ones left
        options = self.space.discrete_values()
        if options is not None and math.prod(len(values) for values in options) <= MAX_LISTED_CONFIGURATIONS:
            untried = [combination for combination in itertools.product(*options) if combination not in self._tried]
            if untried:
                names = [hyperparameter.name for hyperparameter in self.space.dimensions]
                return dict(zip(names, untried[self.rng.integers(len(untried))])), 'random'
        return None

    def _model_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Points and losses at the furthest stage with enough scored configurations to fit TPE, if any."""
        for stage in reversed(range(len(self.stages))):
            scored = [trial for trial in self.trials if len(trial.stage_losses) > stage]
            if self.sampler.model_can_be_fitted(len(scored)):
                return (np.array([self.space.to_point(trial.values) for trial in scored]),
                        np.array([trial.stage_losses[stage] for trial in scored]))
        return None

    # ..... successive halving

    def _ranked(self, stage: int) -> list[Trial]:
        """Trials that finished `stage`, best first."""
        finished = [trial for trial in self.trials if len(trial.stage_losses) > stage]
        return sorted(finished, key=lambda trial: trial.stage_losses[stage])

    def _promotable(self) -> Trial | None:
        """The best trial due to continue to the next stage (checking the furthest stages first), if any

        A trial continues once it is in the best 1/reduction_factor of all trials
        that have finished its stage. Trials that could never get there (ranked
        below the best n_trials/reduction_factor, the most that can ever be
        promoted) are stopped and their training state is dropped.
        """
        most_promoted = max(1, self.n_trials // self.reduction_factor)
        for stage in reversed(range(len(self.stages) - 1)):
            ranked = self._ranked(stage)
            for rank, trial in enumerate(ranked):
                if trial.status == 'waiting' and len(trial.stage_losses) == stage + 1 and rank >= most_promoted:
                    trial.status = 'stopped'
                    trial.folds = []
            for trial in ranked[:len(ranked) // self.reduction_factor]:
                if trial.status == 'waiting' and len(trial.stage_losses) == stage + 1 \
                        and math.isfinite(trial.stage_losses[stage]):
                    return trial
        return None

    def _train_stage(self, trial: Trial) -> None:
        """Train a trial to the end of its next stage and record its score."""
        stage = len(trial.stage_losses)
        try:
            loss = self.trainer.train_to(trial, self.stages[stage])
        except (ValueError, torch.OutOfMemoryError) as error:
            # e.g. a configuration that does not suit this dataset, or one too large for the GPU
            loss = math.inf
            trial.status = f'failed: {str(error).splitlines()[0][:200]}'
            trial.folds = []
            if isinstance(error, torch.OutOfMemoryError) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        trial.stage_losses.append(loss)
        if not trial.status.startswith('failed'):
            trial.status = 'completed' if stage == len(self.stages) - 1 else 'waiting'

        if not self.silent:
            if self.space.trained_by_gradient_descent:
                stage_text = f'epochs {self.stages[stage]:4d}'
            else:
                stage_text = 'fitted'
            outcome = trial.status if trial.status.startswith('failed') else f'val loss {loss:.4g}'
            print(f'  trial {trial.number:3d} ({trial.origin:7s})  {stage_text}  {outcome:16s}  '
                  f'{format_hyperparameters(trial.values)}', flush=True)
        if self.on_progress is not None:
            self.on_progress(self)

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

        while True:
            trial = self._promotable()
            if trial is None and len(self.trials) < self.n_trials:
                suggestion = self._next_configuration()
                if suggestion is not None:
                    values, origin = suggestion
                    self._tried.add(self._key(values))
                    trial = Trial(len(self.trials) + 1, values, origin)
                    self.trials.append(trial)
                else:
                    self.n_trials = len(self.trials)  # every possible configuration has been tried
            if trial is None:
                break
            self._train_stage(trial)

        # Make sure the best configuration has been trained for the full number of epochs
        while not any(trial.status == 'completed' for trial in self.trials):
            continuing = [trial for trial in self.trials if trial.status in ('waiting', 'stopped') and trial.folds
                          and math.isfinite(trial.stage_losses[-1])]
            if not continuing:
                raise RuntimeError('Every configuration failed, so no network could be trained')
            best = max(len(trial.stage_losses) for trial in continuing)
            self._train_stage(min((trial for trial in continuing if len(trial.stage_losses) == best),
                                  key=lambda trial: trial.stage_losses[-1]))

        for trial in self.trials:
            if trial.status == 'waiting':
                trial.status = 'stopped'
        if self.on_progress is not None:
            self.on_progress(self)
        best_trial = min((trial for trial in self.trials if trial.status == 'completed'),
                         key=lambda trial: trial.stage_losses[-1])
        if not self.silent:
            print(f'\nBest configuration (trial {best_trial.number}, validation loss '
                  f'{best_trial.stage_losses[-1]:.4g}):\n  {format_hyperparameters(self.best_values(best_trial))}\n')
        return best_trial

    def best_values(self, trial: Trial) -> dict[str, HyperParameterValue]:
        """The learning rate (for networks trained by gradient descent) and every architecture hyperparameter of a trial."""
        values: dict[str, HyperParameterValue] = {}
        if self.space.trained_by_gradient_descent:
            values[train_network.LEARNING_RATE.name] = self.space.learning_rate(trial.values)
        values.update(self.space.architecture_values(trial.values))
        return values

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

    def write_trials(self, path: str) -> None:
        """Write every trial so far as a tab-separated table (one row per trial)."""
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
                handle.write('\t'.join([str(trial.number), trial.origin, trial.status, str(trial.epochs), *scores,
                                        *(format_value(trial.values[name]) for name in names)]) + '\n')
