"""
Core training module of PARROT: the loss functions, and training any registered
architecture (or fitting it, for models fitted in one step).

Nothing in this module depends on which architecture is being used; networks
only need to follow the ParrotArchitecture interface (see
parrot/architectures/base.py). Running a trained network is in prediction.py.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import itertools
import math
import time
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parrot import process_input_data as pid
from parrot.architectures import HyperParameter, ParrotArchitecture
from parrot.architectures.layers import padding_mask
from parrot.devices import cpu_copy, full_float32_precision
from parrot.tasks import Task
from parrot.uncertainties import fit_model_error, gaussian_nll, model_error_of, set_model_error

# The learning rate is the one hyperparameter shared by every architecture
LEARNING_RATE = HyperParameter('learning_rate', float, default=0.001,
                               help="the Adam optimizer's learning rate (constant throughout training)",
                               search_range=(1e-5, 1e-2), log_scale=True)

# When training stops: after a fixed number of epochs ('iter'), or once the validation loss stops improving ('auto')
STOP_CONDITIONS = ('iter', 'auto')

# With --stop auto, the validation loss must fall by at least this fraction below the loss of the last epoch that
# improved this much, for training to count as still improving (so a tiny new best does not keep training going)
AUTO_STOP_RELATIVE_IMPROVEMENT = 0.005

# With uncertainties the loss is a negative log-likelihood, which can be negative, so a relative fall means nothing.
# Instead it must fall by this much per labelled value: the same 0.5% improvement, measured in likelihood
# (a likelihood 1/(1 - 0.005) times as large lowers its negative log by log(1/(1 - 0.005)))
AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT = -math.log(1 - AUTO_STOP_RELATIVE_IMPROVEMENT)

# Models fitted in one step with uncertainties are refitted as their model error is re-estimated, until it changes
# by less than this fraction (PARROT's choice: well below the model error's own uncertainty)...
MODEL_ERROR_TOLERANCE = 0.01

# ...or they have been refitted this many times (PARROT's choice: it usually settles within two or three)
MAX_MODEL_ERROR_REFITS = 5

# How often (in epochs) the validation loss is printed when not in verbose mode
PRINT_EVERY_N_EPOCHS = 5


# .................................................................
# Losses

def build_loss_criterion(task: Task) -> nn.Module:
    """Build the loss function used for training, validation and testing

    Losses are summed over every sequence (and, for residue-mapped and matrix
    data, every labelled residue or matrix cell) in a batch. Callers divide the
    accumulated total by the number of sequences, so all reported losses are
    average loss per sequence.

    Parameters
    ----------
    task : Task
        The task being trained

    Returns
    -------
    torch.nn.Module
        MSE for residue and matrix regression, L1 for sequence regression, and
        cross-entropy for classification. The cross-entropy loss ignores padded
        positions and missing labels, which carry the label IGNORED_CLASS_LABEL.
        For regression with uncertainties, the Gaussian negative log-likelihood
        (see UncertaintyLoss).
    """

    if task.uncertainties:
        return UncertaintyLoss()
    if task.is_classification:
        return nn.CrossEntropyLoss(reduction='sum', ignore_index=pid.IGNORED_CLASS_LABEL)
    if task.per_residue or task.is_matrix:
        return nn.MSELoss(reduction='sum')
    return nn.L1Loss(reduction='sum')


class UncertaintyLoss(nn.Module):
    """The loss for values with uncertainties: the Gaussian negative log-likelihood (see parrot/uncertainties.py)

    Unlike the other losses it needs the network's model error, so
    compute_batch_loss() calls it with (predicted, values, SDs, model error).
    """

    def forward(self, predicted: torch.Tensor, values: torch.Tensor, sds: torch.Tensor,
                model_error: torch.Tensor) -> torch.Tensor:
        """The summed negative log-likelihood (see uncertainties.gaussian_nll)

        Parameters
        ----------
        predicted, values, sds : torch.Tensor
            The predictions, values and SDs of every labelled value (1D)
        model_error : torch.Tensor
            The network's model error s

        Returns
        -------
        torch.Tensor
            A scalar
        """
        return gaussian_nll(predicted, values, sds, model_error)


def labelled_with_uncertainties(outputs: torch.Tensor, targets: torch.Tensor,
                                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The prediction, value and SD of every labelled value in a batch with uncertainties (leaving out padding)

    Parameters
    ----------
    outputs : torch.Tensor
        The network's outputs: [batch_dim X 1] or [batch_dim X padded_length X 1]
    targets : torch.Tensor
        The batch's targets, on the same device: the same shape but with a last
        dimension of 2, holding each value and its SD (see pid.pad_targets)
    lengths : torch.Tensor
        The true length of each sequence

    Returns
    -------
    tuple of torch.Tensor
        (predictions, values, SDs), each 1D
    """
    if outputs.dim() == 3:
        mask = padding_mask(lengths.to(outputs.device), outputs.shape[1]) & ~torch.isnan(targets[..., 0])
        outputs, targets = outputs[mask], targets[mask]
    return outputs[..., 0], targets[..., 0], targets[..., 1]


def compute_batch_loss(network: nn.Module, batch: pid.Batch, task: Task, criterion: nn.Module,
                       device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a batch through the network and compute its summed loss

    Padded positions and residues (or matrix cells) with missing labels never
    contribute to the loss: for regression they are masked out here, and for
    classification they carry the label IGNORED_CLASS_LABEL, which `criterion`
    ignores.

    Parameters
    ----------
    network : torch.nn.Module
        A network following the ParrotArchitecture interface
    batch : Batch
        A padded batch, as produced by collate_batch()
    task : Task
        The task being trained
    criterion : torch.nn.Module
        The loss function returned by build_loss_criterion()
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    torch.Tensor
        Scalar loss summed over the batch
    torch.Tensor
        The network outputs: [batch_dim X num_classes],
        [batch_dim X padded_length X num_classes] or (matrix tasks)
        [batch_dim X padded_length X width X num_classes]
    """

    vectors = batch.vectors.to(device)
    targets = batch.targets.to(device)

    outputs = network(vectors, batch.lengths)

    if task.uncertainties:
        loss = criterion(*labelled_with_uncertainties(outputs, targets, batch.lengths), model_error_of(network))
    elif task.is_classification and (task.per_residue or task.is_matrix):
        # CrossEntropyLoss expects the class dimension second ([batch, classes, length(, width)])
        loss = criterion(outputs.movedim(-1, 1), targets)
    elif task.per_residue:
        # Only real residues with a label (missing labels are NaN)
        mask = padding_mask(batch.lengths.to(outputs.device), outputs.shape[1]) & ~torch.isnan(targets[..., 0])
        loss = criterion(outputs[mask], targets[mask])
    elif task.is_matrix:
        # Padded cells are NaN, like missing ones (see matrices.data.pad_matrix_targets)
        mask = ~torch.isnan(targets[..., 0])
        loss = criterion(outputs[mask], targets[mask])
    else:
        loss = criterion(outputs, targets)

    return loss, outputs


def sum_batch_losses(batch_losses: list[torch.Tensor]) -> float:
    """Add up per-batch losses that were kept on the device

    Calling .item() on every batch's loss would make the CPU wait for the GPU
    after every batch. Keeping the losses on the device and adding them up once
    gives the same total (each loss is the same float32 value .item() would
    return, summed in float64 on the CPU) without those waits.

    Parameters
    ----------
    batch_losses : list of torch.Tensor
        Scalar, detached loss of each batch

    Returns
    -------
    float
        The total loss (0.0 for an empty list)
    """
    if not batch_losses:
        return 0.0
    # Move to the CPU before converting: MPS tensors cannot be converted to float64
    return float(torch.stack(batch_losses).cpu().double().sum())


@full_float32_precision()
def evaluate_loss(network: nn.Module, loader: DataLoader, task: Task, device: str | torch.device) -> float:
    """Average loss per sequence over a dataset, without updating the network

    The network is left in evaluation mode. With uncertainties, the loss uses
    the network's model error as it is (see refit_model_error() to fit it).

    Parameters
    ----------
    network : torch.nn.Module
        The network to evaluate
    loader : DataLoader
        Batches of the dataset to evaluate on
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        Total loss divided by the number of sequences
    """

    criterion = build_loss_criterion(task)
    batch_losses = []
    n_sequences = 0

    network.eval()
    with torch.no_grad():
        for batch in loader:
            loss, _ = compute_batch_loss(network, batch, task, criterion, device)
            batch_losses.append(loss.detach())
            n_sequences += len(batch.names)
    total_loss = sum_batch_losses(batch_losses)

    return total_loss / n_sequences


@full_float32_precision()
def refit_model_error(network: ParrotArchitecture, loader: DataLoader, task: Task,
                      device: str | torch.device) -> tuple[float, float]:
    """Fit a network's model error to its residuals on a dataset (with uncertainties), and return the loss there

    The model error is set to its maximum-likelihood value given the
    network's predictions of the dataset (see uncertainties.fit_model_error),
    and the loss is then computed with it. Training does this on the
    validation set after every epoch.

    Parameters
    ----------
    network : ParrotArchitecture
        A network built for a task with uncertainties (left in evaluation mode)
    loader : DataLoader
        Batches of the dataset
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        The loss with the new model error: total divided by the number of sequences
    float
        The average number of labelled values per sequence

    Raises
    ------
    ValueError
        If the task has no uncertainties
    """
    if not task.uncertainties:
        raise ValueError('Only a network trained on values with uncertainties has a model error to fit')
    predicted: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    sds: list[torch.Tensor] = []
    n_sequences = 0

    network.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = network(batch.vectors.to(device), batch.lengths)
            for collected, part in zip((predicted, values, sds),
                                       labelled_with_uncertainties(outputs, batch.targets.to(device), batch.lengths)):
                collected.append(part)
            n_sequences += len(batch.names)

    all_predicted, all_values, all_sds = (torch.cat(parts) for parts in (predicted, values, sds))
    residuals = (all_values - all_predicted).cpu().double().numpy()
    # A network that has diverged (non-finite predictions) keeps its model error; its loss is then not finite
    if np.all(np.isfinite(residuals)):
        set_model_error(network, fit_model_error(residuals, all_sds.cpu().double().numpy()))
    total = float(gaussian_nll(all_predicted, all_values, all_sds, model_error_of(network)))
    return total / n_sequences, len(residuals) / n_sequences


def validation_loss(network: ParrotArchitecture, loader: DataLoader, task: Task, device: str | torch.device) -> float:
    """The validation loss after an epoch; with uncertainties, the model error is first refitted to the validation set

    Parameters
    ----------
    network : ParrotArchitecture
        The network being trained
    loader : DataLoader
        Batches of the validation set
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        Average loss per sequence
    """
    if task.uncertainties:
        return refit_model_error(network, loader, task, device)[0]
    return evaluate_loss(network, loader, task, device)


# .................................................................
# Training

@dataclass
class TrainingHistory:
    """A network's losses after each epoch of training, and the weights from its best epoch

    Attributes
    ----------
    train_losses : list of float
        Average training loss per sequence, one per epoch
    val_losses : list of float
        Average validation loss per sequence, one per epoch
    best_loss : float
        The lowest validation loss so far (inf before the first epoch)
    best_epoch : int
        The epoch (counting from 0) with that loss (0 until an epoch's loss is finite)
    best_weights : dict or None
        A CPU copy of the network's weights at that epoch (None until then)
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_loss: float = math.inf
    best_epoch: int = 0
    best_weights: dict[str, Any] | None = None

    def record_epoch(self, train_loss: float, val_loss: float, network: nn.Module) -> bool:
        """Add an epoch's losses, keeping a copy of the network's weights if it is the best epoch so far

        Parameters
        ----------
        train_loss : float
            The epoch's average training loss per sequence
        val_loss : float
            The epoch's average validation loss per sequence
        network : torch.nn.Module
            The network, as it is at the end of the epoch

        Returns
        -------
        bool
            True if this epoch's validation loss is the lowest so far
        """
        epoch = len(self.val_losses)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Written this way round so that a NaN loss never counts as an improvement
        if not val_loss < self.best_loss:
            return False

        self.best_loss = val_loss
        self.best_epoch = epoch
        self.best_weights = cpu_copy(network.state_dict())
        return True

    def restore_best_weights(self, network: nn.Module) -> None:
        """Load the weights of the best epoch into the network (if any epoch had a finite loss)

        Parameters
        ----------
        network : torch.nn.Module
            The network that was trained
        """
        if self.best_weights is not None:
            network.load_state_dict(self.best_weights)


@dataclass
class TrainingTiming:
    """How long training took, epoch by epoch, recorded by train() when it is given one

    Attributes
    ----------
    train_seconds : list of float
        Seconds spent training in each epoch (for a model fitted in one step,
        the time to fit it)
    train_sequences : list of int
        Sequences trained on in each epoch
    val_seconds : list of float
        Seconds spent measuring the validation loss after each epoch
    """

    train_seconds: list[float] = field(default_factory=list)
    train_sequences: list[int] = field(default_factory=list)
    val_seconds: list[float] = field(default_factory=list)

    def record_epoch(self, train_seconds: float, train_sequences: int, val_seconds: float) -> None:
        """Add one epoch's timings

        Parameters
        ----------
        train_seconds : float
            Seconds spent training in the epoch
        train_sequences : int
            Sequences trained on in the epoch
        val_seconds : float
            Seconds spent measuring the validation loss after it
        """
        self.train_seconds.append(train_seconds)
        self.train_sequences.append(train_sequences)
        self.val_seconds.append(val_seconds)

    @property
    def total_seconds(self) -> float:
        """Seconds spent training and validating, over every epoch."""
        return sum(self.train_seconds) + sum(self.val_seconds)

    @property
    def training_speed(self) -> float:
        """Sequences trained on per second, while training (not validating); NaN if no training was timed."""
        seconds = sum(self.train_seconds)
        return sum(self.train_sequences) / seconds if seconds > 0 else math.nan


def meaningfully_better(val_loss: float, reference_loss: float, minimum_fall: float | None = None) -> bool:
    """True if a validation loss improves on a reference by at least AUTO_STOP_RELATIVE_IMPROVEMENT (--stop auto)

    Parameters
    ----------
    val_loss : float
        Validation loss of the current epoch
    reference_loss : float
        Validation loss of the last epoch that improved meaningfully (inf before the first)
    minimum_fall : float, optional
        If given, the loss must instead fall by at least this much (for a loss
        that can be negative; see AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT)

    Returns
    -------
    bool
        False for a NaN loss
    """
    if minimum_fall is not None:
        return val_loss < reference_loss - minimum_fall
    return val_loss < reference_loss * (1 - AUTO_STOP_RELATIVE_IMPROVEMENT)


@full_float32_precision()
def train(network: ParrotArchitecture, train_loader: DataLoader, val_loader: DataLoader, task: Task,
          device: str | torch.device, learning_rate: float, n_epochs: int, stop_condition: str = 'iter',
          on_improvement: Callable[[ParrotArchitecture], None] | None = None, verbose: bool = False,
          silent: bool = False, timing: TrainingTiming | None = None) -> tuple[list[float], list[float]]:
    """Train a network, keeping the weights that perform best on the validation set

    Every epoch the network is trained on the training set and then evaluated
    on the validation set. When training finishes, the network holds the
    weights from the epoch with the lowest validation loss. With
    uncertainties, the model error is fitted to the validation set before
    training and after every epoch (see refit_model_error()), and the network
    keeps the model error of the epoch it keeps.

    Models that are not trained by gradient descent (see architectures/classical.py)
    are instead fitted in one step; the learning rate, number of epochs and stop
    condition are not used, and the returned lists each hold a single loss.

    Parameters
    ----------
    network : ParrotArchitecture
        The network to train (already on `device`)
    train_loader : DataLoader
        Batches of the training set (should shuffle)
    val_loader : DataLoader
        Batches of the validation set
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on
    learning_rate : float
        Learning rate of the Adam optimiser (constant throughout training)
    n_epochs : int
        With stop_condition='iter', the number of epochs to train for. With
        stop_condition='auto', stop once this many epochs in a row have not
        lowered the validation loss by at least AUTO_STOP_RELATIVE_IMPROVEMENT
        (relative to the last epoch that did; with uncertainties, by
        AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT per labelled value), so training
        lasts at least n_epochs + 1 epochs.
    stop_condition : str, optional
        'iter' (default) or 'auto'
    on_improvement : callable, optional
        Called with the network every time the validation loss reaches a new
        minimum, e.g. to save it to disk so an interrupted run is not lost.
    verbose : bool, optional
        If True, print the validation loss every epoch rather than every 5 epochs
    silent : bool, optional
        If True, print nothing
    timing : TrainingTiming, optional
        If given, how long each epoch's training and validation took is
        recorded in it

    Returns
    -------
    list of float
        Average training loss per sequence, for each epoch
    list of float
        Average validation loss per sequence, for each epoch

    Raises
    ------
    ValueError
        If `stop_condition` is not one of STOP_CONDITIONS
    """

    if stop_condition not in STOP_CONDITIONS:
        raise ValueError(f"Invalid stop condition '{stop_condition}': must be 'auto' or 'iter'.")

    if not network.trained_by_gradient_descent:
        return fit_in_one_step(network, train_loader, val_loader, task, device, on_improvement, silent, timing)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = build_loss_criterion(task)
    history = TrainingHistory()

    # With uncertainties, the first epoch is weighted by a model error fitted to the untrained network's residuals,
    # and --stop auto counts the improvement per labelled value (see AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT)
    minimum_fall = None
    if task.uncertainties:
        _, values_per_sequence = refit_model_error(network, val_loader, task, device)
        minimum_fall = AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT * values_per_sequence

    # With 'auto', training ends once n_epochs epochs in a row have not improved the validation loss meaningfully
    # (see meaningfully_better), so there is no fixed number of epochs
    patience_epochs = n_epochs
    reference_loss, last_meaningful_epoch = math.inf, 0
    epochs = range(n_epochs) if stop_condition == 'iter' else itertools.count()

    for epoch in epochs:
        start = time.perf_counter()
        train_loss, train_sequences = train_epoch(network, train_loader, task, criterion, optimizer, device)
        trained = time.perf_counter()
        val_loss = validation_loss(network, val_loader, task, device)
        if timing is not None:
            timing.record_epoch(trained - start, train_sequences, time.perf_counter() - trained)

        if meaningfully_better(val_loss, reference_loss, minimum_fall):
            reference_loss, last_meaningful_epoch = val_loss, epoch
        stop_now = stop_condition == 'auto' and epoch - last_meaningful_epoch >= patience_epochs

        improved = history.record_epoch(train_loss, val_loss, network)
        if improved and on_improvement is not None:
            on_improvement(network)

        # Epochs are shown counting from 1: the first, every PRINT_EVERY_N_EPOCHS-th and the last
        number = epoch + 1
        last = stop_now or (stop_condition == 'iter' and number == n_epochs)
        if not silent and (verbose or number == 1 or number % PRINT_EVERY_N_EPOCHS == 0 or last):
            print(f'Epoch {number}\tLoss {val_loss:.4f}')

        if stop_now:
            break

    history.restore_best_weights(network)
    if not silent:
        print_training_summary(history, stopped_early=stop_condition == 'auto')
        if task.uncertainties:
            print_model_error(network)
    return history.train_losses, history.val_losses


def print_training_summary(history: TrainingHistory, stopped_early: bool) -> None:
    """Say how training ended, and which epoch's network is kept (the one with the lowest validation loss)

    Parameters
    ----------
    history : TrainingHistory
        The finished training
    stopped_early : bool
        True if training stopped once the validation loss plateaued (--stop auto)
    """
    epochs = len(history.val_losses)
    if stopped_early:
        print(f'Stopped after {epochs} epochs, as the validation loss had stopped improving')
    if history.best_weights is None:
        print('No epoch had a finite validation loss, so the network after the last epoch is kept')
        return
    print(f'Lowest validation loss {history.best_loss:.4f}, at epoch {history.best_epoch + 1} of {epochs}: the '
          'network from that epoch is kept')


def print_model_error(network: ParrotArchitecture) -> None:
    """Report a network's model error (with uncertainties), in the units of the values

    Parameters
    ----------
    network : ParrotArchitecture
        A network built for a task with uncertainties
    """
    print(f'Model error s = {float(model_error_of(network)):.4g}: the SD of the error in its predictions, beyond each '
          "value's own SD (fitted on the validation set)")


@full_float32_precision()
def train_epoch(network: nn.Module, loader: DataLoader, task: Task, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str | torch.device) -> tuple[float, int]:
    """Train a network for one epoch (one pass over the training data)

    Parameters
    ----------
    network : torch.nn.Module
        The network to train (already on `device`)
    loader : DataLoader
        Batches of the training set
    task : Task
        The task being trained
    criterion : torch.nn.Module
        The loss function returned by build_loss_criterion()
    optimizer : torch.optim.Optimizer
        Optimizer of the network's parameters
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        Average training loss per sequence
    int
        Number of sequences trained on
    """

    network.train()
    batch_losses = []
    n_train_sequences = 0
    for batch in loader:
        loss, _ = compute_batch_loss(network, batch, task, criterion, device)
        batch_losses.append(loss.detach())
        n_train_sequences += len(batch.names)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_batch_losses(batch_losses) / n_train_sequences, n_train_sequences


def fit_in_one_step(network: ParrotArchitecture, train_loader: DataLoader, val_loader: DataLoader, task: Task,
                    device: str | torch.device,
                    on_improvement: Callable[[ParrotArchitecture], None] | None = None,
                    silent: bool = False, timing: TrainingTiming | None = None) -> tuple[list[float], list[float]]:
    """Fit a model that is not trained by gradient descent, and report its losses

    Parameters
    ----------
    network : ParrotArchitecture
        The model to fit (one with trained_by_gradient_descent = False)
    train_loader : DataLoader
        Batches of the training set
    val_loader : DataLoader
        Batches of the validation set
    task : Task
        The task being trained
    device : str or torch.device
        Device the model's outputs are placed on
    on_improvement : callable, optional
        Called once with the fitted model (e.g. to save it)
    silent : bool, optional
        If True, print nothing
    timing : TrainingTiming, optional
        If given, how long fitting and validation took is recorded in it (as one epoch)

    Returns
    -------
    list of float
        The average training loss per sequence (a single value)
    list of float
        The average validation loss per sequence (a single value)
    """

    start = time.perf_counter()
    if task.uncertainties:
        fit_with_model_error(network, train_loader, val_loader, task, device)
    else:
        network.fit(train_loader)
    fitted = time.perf_counter()
    train_loss = evaluate_loss(network, train_loader, task, device)
    validating = time.perf_counter()
    val_loss = evaluate_loss(network, val_loader, task, device)
    if timing is not None:
        timing.record_epoch(fitted - start, len(cast(Sized, train_loader.dataset)), time.perf_counter() - validating)

    if on_improvement is not None:
        on_improvement(network)

    if not silent:
        print(f'Fitted in one step: validation loss {val_loss:.4f}')
        if task.uncertainties:
            print_model_error(network)

    return [train_loss], [val_loss]


def fit_with_model_error(network: ParrotArchitecture, train_loader: DataLoader, val_loader: DataLoader, task: Task,
                         device: str | torch.device) -> None:
    """Fit a model fitted in one step to values with uncertainties, re-estimating its model error as it is refitted

    The model is first fitted with every sequence weighted equally. Then, in
    turn, the model error is fitted to the validation set (see
    refit_model_error()) and the model refitted with weights
    1 / (SD^2 + s^2), until the model error changes by less than
    MODEL_ERROR_TOLERANCE (or after MAX_MODEL_ERROR_REFITS refits). The
    model error is left fitted to the final model.

    Parameters
    ----------
    network : ParrotArchitecture
        A model fitted in one step, built for a task with uncertainties
    train_loader, val_loader : DataLoader
        Batches of the training and validation sets
    task : Task
        The task being trained
    device : str or torch.device
        Device the model's outputs are placed on
    """
    set_model_error(network, math.nan)
    network.fit(train_loader)
    previous = math.nan
    for _ in range(MAX_MODEL_ERROR_REFITS):
        refit_model_error(network, val_loader, task, device)
        model_error = float(model_error_of(network))
        if abs(model_error - previous) <= MODEL_ERROR_TOLERANCE * previous:
            return
        previous = model_error
        network.fit(train_loader)
    refit_model_error(network, val_loader, task, device)
