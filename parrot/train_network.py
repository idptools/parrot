"""
Core training module of PARROT: the loss functions, and training any registered
architecture (or fitting it, for models fitted in one step), including fitting
the model error of a network trained on values with uncertainties (see
uncertainties.py).

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parrot import process_input_data as pid
from parrot.architectures import HyperParameter, ParrotArchitecture
from parrot.architectures.layers import padding_mask
from parrot.devices import cpu_copy, full_float32_precision
from parrot.gradient_numerics import clip_with_norm, stable_total_norm
from parrot.tasks import Task
from parrot.training_control import (EpochSnapshot, LearningRateController, TrainingControlState, TrainingOptions,
                                     finite_tensors)
from parrot.training_precision import BatchReplay, MAX_SCALE_RETRIES, TrainingPrecision, validate_amp_architecture
from parrot.training_persistence import (FORMAT_KEY, FORMAT_VERSION, EpochLog, atomic_save, capture_random_state,
                                         load_training_checkpoint, loader_signature, restore_random_state,
                                         validate_signature)
from parrot.uncertainties import (MODEL_ERROR_SIGNIFICANT_FIGURES, fit_model_error, gaussian_nll, model_error_of,
                                  set_model_error)

# The learning rate is the one hyperparameter shared by every architecture
LEARNING_RATE = HyperParameter('learning_rate', float, default=0.001,
                               help="the Adam optimizer's initial learning rate (fixed unless LR control is enabled)",
                               search_range=(1e-5, 1e-2), log_scale=True)

# When training stops: after a fixed number of epochs ('iter'), or once the validation loss stops improving ('auto')
STOP_CONDITIONS = ('iter', 'auto')

# With --stop auto, the validation loss must fall by at least this fraction below the loss of the last epoch that
# improved this much, for training to count as still improving (so a tiny new best does not keep training going)
AUTO_STOP_RELATIVE_IMPROVEMENT = 0.005

# With uncertainties the loss is a negative log-likelihood, which can be negative, so a relative fall means nothing.
# Instead it must fall by this much (in nats) per labelled value: the same 0.5% improvement, measured in likelihood
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
    if task.loss_name == 'huber':
        return nn.HuberLoss(reduction='sum', delta=task.huber_delta)
    if task.loss_name == 'mse':
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
                       device: str | torch.device, equal_protein_weighting: bool = False,
                       precision: TrainingPrecision | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
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
    equal_protein_weighting : bool, optional
        Return the sum of per-protein mean labelled losses instead of the
        sum over all labels. Callers average across proteins.

    Returns
    -------
    torch.Tensor
        Scalar loss summed over the batch
    torch.Tensor
        The network outputs: [batch_dim X num_classes],
        [batch_dim X padded_length X num_classes] or (matrix tasks)
        [batch_dim X padded_length X width X num_classes]
    """

    if precision is not None:
        precision.validate_architecture(getattr(network, 'name', ''))
    vectors = batch.vectors.to(device)
    targets = batch.targets.to(device)

    if precision is None:
        outputs = network(vectors, batch.lengths)
    else:
        with precision.autocast():
            outputs = network(vectors, batch.lengths)
        # Loss reductions, squared errors and Gaussian NLL always use FP32.
        outputs = outputs.float()

    if equal_protein_weighting:
        loss = equal_protein_loss(network, outputs, targets, batch.lengths, task, criterion)
    elif task.uncertainties:
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


def equal_protein_loss(network: nn.Module, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor,
                       task: Task, criterion: nn.Module) -> torch.Tensor:
    """Sum per-protein mean losses, excluding missing labels and padding before loss arithmetic.

    Evaluation divides this sum by the dataset's number of proteins; training
    also divides by the batch size before backward. Equal objective weights do
    not imply equal gradient vectors or gradient norms.
    """
    if not (task.per_residue or task.is_matrix):
        raise ValueError('Equal-protein weighting applies to residue or matrix tasks')
    real = padding_mask(lengths.to(outputs.device), outputs.shape[1]) if task.per_residue else None
    labelled_mask = targets != pid.IGNORED_CLASS_LABEL if task.is_classification else ~torch.isnan(targets[..., 0])
    if real is not None:
        labelled_mask = labelled_mask & real
    counts = labelled_mask.flatten(start_dim=1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('Equal-protein weighting needs at least one labelled value per protein')
    shape = [outputs.shape[0]] + [1] * (labelled_mask.dim() - 1)
    weights = counts.to(outputs.dtype).reciprocal().reshape(shape).expand_as(labelled_mask)[labelled_mask]
    selected, labelled = outputs[labelled_mask], targets[labelled_mask]
    if task.uncertainties and isinstance(criterion, UncertaintyLoss):
        return gaussian_nll(selected[..., 0], labelled[..., 0], labelled[..., 1], model_error_of(network),
                            weights=weights)
    if isinstance(criterion, nn.CrossEntropyLoss):
        per_value_losses = nn.functional.cross_entropy(selected, labelled, weight=criterion.weight, reduction='none',
                                                        ignore_index=criterion.ignore_index,
                                                        label_smoothing=criterion.label_smoothing)
        return (per_value_losses * weights).sum()
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
        loss_function = nn.functional.mse_loss if isinstance(criterion, nn.MSELoss) else nn.functional.l1_loss
        per_value_losses = loss_function(selected, labelled, reduction='none').sum(dim=-1)
    elif isinstance(criterion, nn.HuberLoss):
        per_value_losses = nn.functional.huber_loss(selected, labelled, reduction='none',
                                                    delta=criterion.delta).sum(dim=-1)
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.HuberLoss)):
        return (per_value_losses * weights).sum()

    # Preserve support for a caller's custom summed criterion.
    losses = []
    for index in range(outputs.shape[0]):
        target = targets[index]
        mask = labelled_mask[index]
        selected, labelled = outputs[index][mask], target[mask]
        count = labelled.shape[0]
        if task.uncertainties:
            summed = criterion(selected[..., 0], labelled[..., 0], labelled[..., 1], model_error_of(network))
        else:
            summed = criterion(selected, labelled)
        losses.append(summed / count)
    return torch.stack(losses).sum()


def sum_batch_losses(batch_losses: list[torch.Tensor]) -> float:
    """Add up per-batch losses that were kept on the device

    Calling .item() on every batch's loss would make the CPU wait for the GPU
    after every batch. Keeping the losses on the device and adding them up once
    gives the same total up to float64 rounding (each loss is the same float32
    value .item() would return, summed in float64 on the CPU) without those
    waits.

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
def evaluate_loss(network: nn.Module, loader: DataLoader, task: Task, device: str | torch.device,
                  equal_protein_weighting: bool = False) -> float:
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
    equal_protein_weighting : bool, optional
        Average each protein's labelled losses before averaging proteins.

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
            options: dict[str, Any] = {'equal_protein_weighting': True} if equal_protein_weighting else {}
            loss, _ = compute_batch_loss(network, batch, task, criterion, device, **options)
            batch_losses.append(loss.detach())
            n_sequences += len(batch.names)
    total_loss = sum_batch_losses(batch_losses)

    return total_loss / n_sequences


@full_float32_precision()
def refit_model_error(network: ParrotArchitecture, loader: DataLoader, task: Task,
                      device: str | torch.device, equal_protein_weighting: bool = False) -> tuple[float, float]:
    """Fit a network's model error to its residuals on a dataset (with uncertainties), and return the loss there

    The model error is set to its maximum-likelihood value given the
    network's predictions of the dataset (see uncertainties.fit_model_error),
    and the loss is then computed with it. Training does this on the
    validation set before the first epoch and after every epoch. If any
    prediction is not finite (the network has diverged), the model error is
    left as it is, and the loss is not finite.

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
    equal_protein_weighting : bool, optional
        Weight each labelled value by the reciprocal of its protein's labelled
        count in both model-error fitting and the returned loss.

    Returns
    -------
    float
        The loss with the new model error: total divided by the number of sequences
    float
        The average number of labelled values per sequence, or 1 with
        equal-protein weighting (the total effective weight per sequence)

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
    weights: list[torch.Tensor] = []
    n_sequences = 0

    network.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = network(batch.vectors.to(device), batch.lengths)
            if equal_protein_weighting:
                real = padding_mask(batch.lengths.to(device), outputs.shape[1])
                labelled = real & ~torch.isnan(batch.targets.to(device)[..., 0])
                counts = labelled.sum(dim=1)
                if bool((counts == 0).any()):
                    raise ValueError('Equal-protein weighting needs labelled values in every protein')
                per_value = counts.to(outputs.dtype).reciprocal().unsqueeze(1).expand_as(labelled)
                weights.append(per_value[labelled])
            for collected, part in zip((predicted, values, sds),
                                       labelled_with_uncertainties(outputs, batch.targets.to(device), batch.lengths)):
                collected.append(part)
            n_sequences += len(batch.names)

    all_predicted, all_values, all_sds = (torch.cat(parts) for parts in (predicted, values, sds))
    residuals = (all_values - all_predicted).cpu().double().numpy()
    all_weights = torch.cat(weights) if equal_protein_weighting else None
    # A network that has diverged (non-finite predictions) keeps its model error; its loss is then not finite
    if np.all(np.isfinite(residuals)):
        fit_options = {} if all_weights is None else {'weights': all_weights.cpu().double().numpy()}
        set_model_error(network, fit_model_error(residuals, all_sds.cpu().double().numpy(), **fit_options))
    loss_options = {} if all_weights is None else {'weights': all_weights}
    total = float(gaussian_nll(all_predicted, all_values, all_sds, model_error_of(network), **loss_options))
    return total / n_sequences, 1.0 if equal_protein_weighting else len(residuals) / n_sequences


def validation_loss(network: ParrotArchitecture, loader: DataLoader, task: Task, device: str | torch.device,
                    equal_protein_weighting: bool = False) -> float:
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
    options = {'equal_protein_weighting': True} if equal_protein_weighting else {}
    if task.uncertainties:
        return refit_model_error(network, loader, task, device, **options)[0]
    return evaluate_loss(network, loader, task, device, **options)


# .................................................................
# Training

class NonFiniteTrainingError(RuntimeError):
    """Opt-in checks found a nonfinite loss or gradient; training must stop."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def validate_gradient_clip_norm(gradient_clip_norm: float | None) -> None:
    """A clipping threshold is a finite, strictly positive global L2 norm, or None (off)."""
    if gradient_clip_norm is not None and (isinstance(gradient_clip_norm, bool)
                                          or not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0):
        raise ValueError('Gradient clip norm must be a finite number greater than 0')


@dataclass
class GradientStatistics:
    """Pre-clipping global L2 norms for one epoch, collected only with opt-in gradient checks or clipping."""

    norms: list[float] = field(default_factory=list)

    def describe(self, clip_norm: float | None) -> str:
        """A calibration summary in the units of the current, summed training loss."""
        if not self.norms:
            return 'Gradient norms: no batches'
        summary = (f'Gradient norms before clipping: median {np.median(self.norms):.4g}, '
                   f'p95 {np.percentile(self.norms, 95):.4g}, max {max(self.norms):.4g}')
        if clip_norm is not None:
            clipped = sum(norm > clip_norm for norm in self.norms)
            summary += f'; clipped {clipped}/{len(self.norms)} batches at {clip_norm:g}'
        return summary


def nonfinite_batch_error(what: str, batch: pid.Batch, batch_number: int, retryable: bool = True,
                          after_update: bool = False) -> NonFiniteTrainingError:
    """Identify a failing batch without printing an unbounded list of sequence IDs."""
    names = ', '.join(repr(name[:80]) for name in batch.names[:5])
    if len(batch.names) > 5:
        names += f', ... ({len(batch.names)} sequences)'
    lengths = batch.lengths
    action = ('The failure occurred after the optimizer update.' if after_update else
              'The optimizer was not updated for this batch.')
    return NonFiniteTrainingError(
        f'Nonfinite {what} in training batch {batch_number} (sequence IDs: {names}; '
        f'lengths {int(lengths.min())}-{int(lengths.max())}). '
        f'{action} The attempt stopped without skipping batches.', retryable=retryable)


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
        The lowest validation loss so far (inf until an epoch's loss is finite)
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
          silent: bool = False, timing: TrainingTiming | None = None, gradient_clip_norm: float | None = None,
          check_finite_gradients: bool = False, training_options: TrainingOptions | None = None,
          control_state: TrainingControlState | None = None, checkpoint_path: str | None = None,
          checkpoint_every: int = 1, resume_from: str | None = None, training_log: str | None = None,
          checkpoint_metadata: dict[str, Any] | None = None) -> tuple[list[float], list[float]]:
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
        Initial learning rate of Adam (fixed unless LR control is enabled)
    n_epochs : int
        With stop_condition='iter', the number of epochs to train for. With
        stop_condition='auto', stop once this many epochs in a row have not
        lowered the validation loss by at least AUTO_STOP_RELATIVE_IMPROVEMENT
        (relative to the last epoch that did; with uncertainties, by
        AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT per labelled value), so training
        lasts at least n_epochs + 1 epochs. With uncertainties and equal-protein
        weighting, the absolute improvement is in mean per-protein NLL instead.
        Stopping patience is independent of LR control and does not reset on
        LR changes.
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
    gradient_clip_norm : float, optional
        Clip the global L2 gradient norm to this positive, finite value before
        every update. Off by default. Implies the finite checks below.
    check_finite_gradients : bool, optional
        Stop on a nonfinite training loss, gradient norm or validation loss.
        Off by default. Either option also reports gradient norm summaries.
    training_options : TrainingOptions, optional
        Opt-in equal-protein weighting, LR schedules and bounded in-process
        recovery. Fixed LR and the original summed loss remain the defaults.
        Cosine duration is resolved against n_epochs minus warmup; with
        stop_condition='auto', specify lr_anneal_epochs explicitly. Recovery
        requires loaders with num_workers=0 and cannot resume a crashed process.
    control_state : TrainingControlState, optional
        Collect successful-epoch LR history and recovery diagnostics.
    checkpoint_path : str, optional
        Opt in to atomic resumable disk checkpoints, separate from the best
        inference network. Includes model, optimizer, best weights, stopping,
        scheduler/scaler, recovery and random states. Requires PARROT
        SequenceDataset loaders with num_workers=0.
    checkpoint_every : int, optional
        Positive interval in successful epochs (default 1). Also save before
        training, at the final epoch and after each recovery rollback.
    resume_from : str, optional
        Resume a training checkpoint after validating task, data, encoding,
        batching, policies, device type and PyTorch version. n_epochs remains
        the total duration, not the additional epochs to run.
    training_log : str, optional
        Opt in to a TSV record flushed after every successful epoch. Resume
        reconciles rows with the checkpoint before continuing.
    checkpoint_metadata : dict, optional
        Caller provenance and identity to bind into checkpoint validation.

    Returns
    -------
    list of float
        Average training loss per sequence, for each epoch
    list of float
        Average validation loss per sequence, for each epoch

    Raises
    ------
    ValueError
        If `stop_condition` is not one of STOP_CONDITIONS, or a clipping
        threshold is not finite and strictly positive
    NonFiniteTrainingError
        If opt-in checks find a nonfinite loss or gradient norm
    """

    if stop_condition not in STOP_CONDITIONS:
        raise ValueError(f"Invalid stop condition '{stop_condition}': must be 'auto' or 'iter'.")
    validate_gradient_clip_norm(gradient_clip_norm)
    options = (training_options or TrainingOptions()).resolved(n_epochs, stop_condition)
    validate_amp_architecture(network.name, options.amp)
    options.validate_task(task)
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError('Checkpoint interval must be a positive integer')
    persistence = checkpoint_path is not None or resume_from is not None or training_log is not None
    if persistence and not network.trained_by_gradient_descent:
        raise ValueError('Epoch checkpoints, resume and continuous logs apply to neural networks only')
    saved = load_training_checkpoint(resume_from, 'train') if resume_from is not None else None
    if saved is not None:
        saved_signature = saved['signature']
        validate_amp_architecture(saved_signature.get('architecture', ''),
                                  saved_signature.get('training_options', {}).get('amp', 'off'))
    if checkpoint_path is not None:
        if training_log is not None and Path(checkpoint_path).absolute() == Path(training_log).absolute():
            raise ValueError('Training checkpoint and epoch log must have different paths')
        if Path(checkpoint_path).exists() and (resume_from is None
                                               or Path(checkpoint_path).absolute() != Path(resume_from).absolute()):
            prior = load_training_checkpoint(checkpoint_path)
            final_search = False
            if checkpoint_metadata is not None and 'final_search' in checkpoint_metadata:
                final_search = prior['kind'] == 'search' and prior['metadata']['identity'] == checkpoint_metadata['identity']
            if not final_search:
                raise ValueError('Training checkpoint already exists; use resume_from or choose a new path')
    signature = None
    if checkpoint_path is not None or saved is not None:
        signature = {'architecture': network.name, 'hyperparameters': network.hyperparameter_values,
                     'task': task.to_config(), 'input_size': network.input_size,
                     'training_options': asdict(options), 'learning_rate': learning_rate,
                     'stop_condition': stop_condition, 'patience': n_epochs if stop_condition == 'auto' else None,
                     'gradient_clip_norm': gradient_clip_norm, 'check_finite_gradients': check_finite_gradients,
                     'device_type': torch.device(device).type, 'torch_version': str(torch.__version__),
                     'loaders': [loader_signature(loader) for loader in (train_loader, val_loader)],
                     'metadata': checkpoint_metadata.get('identity', checkpoint_metadata)
                     if checkpoint_metadata is not None else None}
        # Validate worker/generator structure before any network state or output is changed.
        capture_random_state((train_loader, val_loader), device)
        if saved is not None:
            validate_signature(saved['signature'], signature)
            if stop_condition == 'iter' and n_epochs < saved['epoch']:
                raise ValueError('Requested total epochs precede the resumed checkpoint epoch')

    # Omit inactive options entirely, including for callers that wrap train_epoch.
    gradient_options: dict[str, Any] = {}
    if gradient_clip_norm is not None:
        gradient_options['gradient_clip_norm'] = gradient_clip_norm
    if check_finite_gradients:
        gradient_options['check_finite_gradients'] = True

    if not network.trained_by_gradient_descent:
        if options.active:
            raise ValueError('Training controls apply to neural networks trained by gradient descent')
        return fit_in_one_step(network, train_loader, val_loader, task, device, on_improvement, silent, timing)

    optimizer = options.build_optimizer(network, learning_rate)
    criterion = build_loss_criterion(task)
    history = TrainingHistory()
    state = control_state if control_state is not None else TrainingControlState()
    controller = LearningRateController(optimizer, options) if options.controls_lr else None
    precision = TrainingPrecision(options.amp, device) if options.amp != 'off' else None
    if precision is not None and state.precision_state is not None:
        precision.load_state_dict(state.precision_state)

    start_epoch = 0
    reference_loss, last_meaningful_epoch = math.inf, 0
    minimum_fall = None
    if saved is not None:
        network.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        history = TrainingHistory(**saved['history'])
        restored = TrainingControlState(**saved['control_state'])
        state.__dict__.update(restored.__dict__)
        if controller is not None:
            if state.scheduler_state is None:
                raise ValueError('Training checkpoint is missing LR-controller state')
            controller.load_state_dict(state.scheduler_state)
        if precision is not None and state.precision_state is not None:
            precision.load_state_dict(state.precision_state)
        start_epoch = saved['epoch']
        reference_loss, last_meaningful_epoch = saved['reference_loss'], saved['last_meaningful_epoch']
        minimum_fall = saved['minimum_fall']
        if timing is not None and saved['timing'] is not None:
            timing.__dict__.update(saved['timing'])
        restore_random_state(saved['random_state'], (train_loader, val_loader), device)
    log = EpochLog(training_log, saved.get('records', []) if saved else None, resume=saved is not None)

    def save_progress(epoch: int, stopped: bool = False) -> None:
        if checkpoint_path is None:
            return
        if controller is not None:
            state.scheduler_state = controller.state_dict()
        if precision is not None:
            state.precision_state = precision.state_dict()
        if not finite_tensors(network.state_dict()) or not finite_tensors(optimizer.state_dict()):
            raise NonFiniteTrainingError('Refusing to overwrite a clean checkpoint with nonfinite model/optimizer state')
        payload = {FORMAT_KEY: FORMAT_VERSION, 'kind': 'train', 'signature': signature,
                   'metadata': checkpoint_metadata, 'epoch': epoch, 'stopped': stopped,
                   'model_state': cpu_copy(network.state_dict()), 'optimizer_state': cpu_copy(optimizer.state_dict()),
                   'history': asdict(history), 'control_state': asdict(state), 'reference_loss': reference_loss,
                   'last_meaningful_epoch': last_meaningful_epoch, 'minimum_fall': minimum_fall,
                   'timing': asdict(timing) if timing is not None else None, 'records': log.rows,
                   'random_state': capture_random_state((train_loader, val_loader), device)}
        atomic_save(payload, checkpoint_path)

    # With uncertainties, the first epoch is weighted by a model error fitted to the untrained network's residuals,
    # and --stop auto counts the improvement per labelled value (see AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT)
    if task.uncertainties and saved is None:
        loss_options = {'equal_protein_weighting': True} if options.equal_protein_weighting else {}
        _, values_per_sequence = refit_model_error(network, val_loader, task, device, **loss_options)
        minimum_fall = AUTO_STOP_LOG_LIKELIHOOD_IMPROVEMENT * values_per_sequence

    # With 'auto', training ends once n_epochs epochs in a row have not improved the validation loss meaningfully
    # (see meaningfully_better), so there is no fixed number of epochs
    patience_epochs = n_epochs
    epochs = range(start_epoch, n_epochs) if stop_condition == 'iter' else itertools.count(start_epoch)
    if saved is not None and saved['stopped'] and stop_condition == 'auto':
        epochs = range(0)
    if checkpoint_path is not None and saved is None:
        save_progress(0)

    for epoch in epochs:
        start = time.perf_counter()
        used_lr = optimizer.param_groups[0]['lr']
        gradient_statistics = GradientStatistics() if gradient_options or options.recover_nonfinite or precision else None
        epoch_options = dict(gradient_options)
        if gradient_statistics is not None:
            epoch_options['gradient_statistics'] = gradient_statistics
        control_options: dict[str, Any] = {}
        if precision is not None:
            control_options['precision'] = precision
        if checkpoint_path is not None:
            control_options['on_recovery'] = lambda: save_progress(epoch)
        try:
            if options.active:
                train_loss, train_sequences, val_loss, trained = controlled_epoch(
                    network, train_loader, val_loader, task, criterion, optimizer, device, options, state,
                    controller, epoch, epoch_options, silent=silent, **control_options)
            else:
                train_loss, train_sequences = train_epoch(network, train_loader, task, criterion, optimizer, device,
                                                           **epoch_options)
                trained = time.perf_counter()
                val_loss = validation_loss(network, val_loader, task, device)
        except NonFiniteTrainingError as error:
            raise NonFiniteTrainingError(f'Epoch {epoch + 1}: {error}', retryable=error.retryable) from error
        if gradient_options and not math.isfinite(val_loss):
            raise NonFiniteTrainingError(f'Epoch {epoch + 1}: nonfinite validation loss; training stopped. '
                                         'Any previously saved best network is preserved.')
        if persistence and (not math.isfinite(train_loss) or not math.isfinite(val_loss)):
            raise NonFiniteTrainingError(f'Epoch {epoch + 1}: nonfinite loss; the last clean checkpoint is preserved')
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
        if persistence:
            if controller is not None:
                used_lr = state.epoch_history[-1]['learning_rate']
            row = {'epoch': number, 'training_loss': train_loss, 'validation_loss': val_loss,
                   'learning_rate': used_lr, 'recoveries': state.recoveries_used,
                   'training_seconds': trained - start, 'validation_seconds': time.perf_counter() - trained,
                   'amp_scale_retries': precision.scale_retries if precision is not None else 0,
                   'amp_dtype': str(precision.dtype) if precision is not None else 'off',
                   'loss_scale': precision.scaler.get_scale() if precision is not None and precision.scaler else 1.0}
            if gradient_statistics is not None and gradient_statistics.norms:
                norms = np.asarray(gradient_statistics.norms)
                row.update(gradient_norm_median=float(np.median(norms)), gradient_norm_p95=float(np.percentile(norms, 95)),
                           gradient_norm_max=float(np.max(norms)),
                           clipped_fraction=float(np.mean(norms > gradient_clip_norm))
                           if gradient_clip_norm is not None else 0.0)
            log.append(row)
            if number % checkpoint_every == 0 or last:
                save_progress(number, stopped=stop_now)
        if not silent and (verbose or number == 1 or number % PRINT_EVERY_N_EPOCHS == 0 or last):
            rate = f"\tLR {state.epoch_history[-1]['learning_rate']:.6g}" if controller is not None else ''
            print(f'Epoch {number}\tLoss {val_loss:.4f}{rate}')
            if gradient_statistics is not None:
                print(gradient_statistics.describe(gradient_clip_norm))

        if stop_now:
            break

    history.restore_best_weights(network)
    if not silent:
        print_training_summary(history, stopped_early=stop_condition == 'auto')
        if task.uncertainties:
            print_model_error(network)
    return history.train_losses, history.val_losses


def controlled_epoch(network: ParrotArchitecture, train_loader: DataLoader, val_loader: DataLoader, task: Task,
                      criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str | torch.device,
                      options: TrainingOptions, state: TrainingControlState,
                      controller: LearningRateController | None, epoch: int, gradient_options: dict[str, Any],
                      silent: bool = True, precision: TrainingPrecision | None = None,
                      on_recovery: Callable[[], None] | None = None) -> tuple[float, int, float, float]:
    """One successful epoch, or bounded replay from its clean boundary; shared with search training."""
    loss_options = {'equal_protein_weighting': True} if options.equal_protein_weighting else {}
    epoch_options = {**gradient_options, **loss_options}
    if precision is not None:
        epoch_options['precision'] = precision
    if controller is not None:
        controller.prepare_epoch(epoch)
    if options.recover_nonfinite:
        epoch_options.update(check_finite_gradients=True, check_optimizer_state=True)
        if not finite_tensors(network.state_dict()) or not finite_tensors(optimizer.state_dict()):
            raise NonFiniteTrainingError('Cannot recover from an already nonfinite epoch boundary', retryable=False)
        assert controller is not None
        snapshot = EpochSnapshot.capture(network, optimizer, controller, (train_loader, val_loader), device,
                                          **({'precision': precision} if precision is not None else {}))
    else:
        snapshot = None

    while True:
        used_lrs = controller.lrs if controller is not None else []
        statistics = epoch_options.get('gradient_statistics')
        if statistics is not None:
            statistics.norms.clear()
        try:
            train_loss, count = train_epoch(network, train_loader, task, criterion, optimizer, device, **epoch_options)
            trained = time.perf_counter()
            val_loss = validation_loss(network, val_loader, task, device, **loss_options)
            if (gradient_options or options.recover_nonfinite or options.lr_schedule == 'plateau' or precision) \
                    and not math.isfinite(val_loss):
                raise NonFiniteTrainingError('Nonfinite validation loss')
            if options.recover_nonfinite and (not finite_tensors(network.state_dict())
                                               or not finite_tensors(optimizer.state_dict())):
                raise NonFiniteTrainingError('Nonfinite model or optimizer state after validation')
        except NonFiniteTrainingError as error:
            if snapshot is None:
                raise
            assert controller is not None
            if not error.retryable:
                snapshot.restore(network, optimizer, controller, precision=precision)
                raise
            if state.recoveries_used >= options.recover_nonfinite:
                snapshot.restore(network, optimizer, controller, precision=precision)
                raise NonFiniteTrainingError(
                    f'Recovery budget exhausted ({state.recoveries_used}/{options.recover_nonfinite}); {error} '
                    'Any previously saved best network is preserved.') from error
            failed_lrs = controller.lrs
            snapshot.restore(network, optimizer, controller, precision=precision)
            try:
                reduced = controller.reduce_after_rollback(failed_lrs)
            except ValueError as reduction_error:
                raise NonFiniteTrainingError(f'{reduction_error}. Original failure: {error}') from reduction_error
            state.recoveries_used += 1
            state.recovery_events.append({'epoch': epoch + 1, 'attempt': state.recoveries_used,
                                          'old_lr': failed_lrs[0], 'new_lr': reduced[0], 'reason': str(error)})
            if not silent:
                print(f'Epoch {epoch + 1}: recovery {state.recoveries_used}/{options.recover_nonfinite}, '
                      f'LR {failed_lrs[0]:.6g} -> {reduced[0]:.6g}; replaying the epoch ({error})')
            if on_recovery is not None:
                on_recovery()
            continue
        if controller is not None:
            controller.finish_epoch(epoch, val_loss)
            state.scheduler_state = controller.state_dict()
            state.epoch_history.append({'epoch': epoch + 1, 'training_loss': train_loss,
                                         'validation_loss': val_loss, 'learning_rate': used_lrs[0],
                                         'recoveries': state.recoveries_used})
        if precision is not None:
            state.precision_state = precision.state_dict()
        return train_loss, count, val_loss, trained


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
    model_error = float(model_error_of(network))
    print(f'Model error s = {model_error:.{MODEL_ERROR_SIGNIFICANT_FIGURES}g}: the SD of the error in its predictions, '
          "beyond each value's own SD (fitted on the validation set)")


@full_float32_precision()
def train_epoch(network: nn.Module, loader: DataLoader, task: Task, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str | torch.device, gradient_clip_norm: float | None = None,
                check_finite_gradients: bool = False,
                gradient_statistics: GradientStatistics | None = None, equal_protein_weighting: bool = False,
                check_optimizer_state: bool = False, precision: TrainingPrecision | None = None) -> tuple[float, int]:
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
    gradient_clip_norm : float, optional
        Maximum global L2 gradient norm. Off by default; implies finite checks.
    check_finite_gradients : bool, optional
        Stop before an update with a nonfinite loss or gradient norm.
    gradient_statistics : GradientStatistics, optional
        Collect pre-clipping norms when either option above is active.
    equal_protein_weighting : bool, optional
        Average labelled losses within each protein and then across the batch.
    check_optimizer_state : bool, optional
        Recovery's additional checks of input values and model/optimizer
        tensors, including Adam moments after every update.
    precision : TrainingPrecision, optional
        CUDA AMP policy. Enables input, loss, unscaled-gradient and
        post-update checks even without the separate gradient-check flag.

    Returns
    -------
    float
        Average training loss per sequence
    int
        Number of sequences trained on
    """

    validate_gradient_clip_norm(gradient_clip_norm)
    if precision is not None:
        precision.validate_architecture(getattr(network, 'name', ''))
    check_optimizer_state = check_optimizer_state or precision is not None
    check_gradients = check_finite_gradients or gradient_clip_norm is not None or check_optimizer_state
    network.train()
    batch_losses = []
    n_train_sequences = 0
    for batch_number, batch in enumerate(loader, start=1):
        if check_optimizer_state:
            # Invalid data cannot be repaired by lowering the learning rate. Missing labels remain allowed.
            invalid_targets = torch.isinf(batch.targets).any()
            if task.uncertainties:
                values, sds = batch.targets[..., 0], batch.targets[..., 1]
                invalid_targets = invalid_targets | ((~torch.isnan(values)) &
                                                       (~torch.isfinite(sds) | (sds < 0))).any()
            if not task.per_residue and not task.is_matrix:
                invalid_targets = invalid_targets | ~torch.isfinite(batch.targets).all()
            if not torch.isfinite(batch.vectors).all() or bool(invalid_targets):
                raise nonfinite_batch_error('input or target', batch, batch_number, retryable=False)
        loss_options: dict[str, Any] = {'equal_protein_weighting': True} if equal_protein_weighting else {}
        if precision is not None:
            loss = train_amp_batch(network, batch, task, criterion, optimizer, device, precision, batch_number,
                                    gradient_clip_norm, gradient_statistics, equal_protein_weighting)
            batch_losses.append(loss.detach())
            n_train_sequences += len(batch.names)
            if check_optimizer_state and (not finite_tensors(network.state_dict())
                                           or not finite_tensors(optimizer.state_dict())):
                raise nonfinite_batch_error('model or optimizer state', batch, batch_number, after_update=True)
            continue
        loss, _ = compute_batch_loss(network, batch, task, criterion, device, **loss_options)
        if check_gradients and not torch.isfinite(loss):
            raise nonfinite_batch_error('loss', batch, batch_number)
        batch_losses.append(loss.detach())
        n_train_sequences += len(batch.names)

        optimizer.zero_grad()
        backward_loss = loss / len(batch.names) if equal_protein_weighting else loss
        backward_loss.backward()
        norm_fallback = False
        if check_gradients:
            parameters = [parameter for parameter in network.parameters() if parameter.grad is not None]
            gradients = [cast(torch.Tensor, parameter.grad) for parameter in parameters]
            norm = nn.utils.get_total_norm(gradients)
            norm_fallback = not bool(torch.isfinite(norm))
            if norm_fallback:
                norm = stable_total_norm(gradients, norm)
            if not torch.isfinite(norm):
                raise nonfinite_batch_error('gradient norm', batch, batch_number)
            if gradient_statistics is not None:
                gradient_statistics.norms.append(float(norm.detach()))
            if gradient_clip_norm is not None:
                clip_with_norm(parameters, gradient_clip_norm, norm)
        optimizer.step()
        # A finite fallback norm does not prevent enormous unclipped gradients
        # from overflowing Adam's squared-gradient moments.
        if (check_optimizer_state or norm_fallback) and (not finite_tensors(network.state_dict())
                                                         or not finite_tensors(optimizer.state_dict())):
            raise nonfinite_batch_error('model or optimizer state', batch, batch_number, after_update=True)

    return sum_batch_losses(batch_losses) / n_train_sequences, n_train_sequences


def train_amp_batch(network: nn.Module, batch: pid.Batch, task: Task, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str | torch.device, precision: TrainingPrecision,
                    batch_number: int, clip_norm: float | None, statistics: GradientStatistics | None,
                    equal_protein_weighting: bool) -> torch.Tensor:
    """One committed AMP update, with bounded replay for FP16 loss-scaling overflow.

    Gradients are unscaled before checking or clipping. FP16 scaled-gradient
    overflow lowers the scale and replays the retained batch with the same
    randomness and buffers. Eight retries are allowed; no batch is skipped.
    Nonfinite forward losses and BF16 gradients use normal failure/recovery.
    """
    precision.validate_architecture(getattr(network, 'name', ''))
    scaler = precision.scaler
    replay = BatchReplay(network, precision.device) if scaler is not None else None
    for attempt in range(MAX_SCALE_RETRIES + 1):
        optimizer.zero_grad()
        loss, _ = compute_batch_loss(network, batch, task, criterion, device,
                                     equal_protein_weighting=equal_protein_weighting, precision=precision)
        if not torch.isfinite(loss):
            raise nonfinite_batch_error('AMP forward loss', batch, batch_number)
        backward_loss = loss / len(batch.names) if equal_protein_weighting else loss
        if scaler is None:
            backward_loss.backward()
        else:
            scaler.scale(backward_loss).backward()
            scaler.unscale_(optimizer)
        parameters = [parameter for parameter in network.parameters() if parameter.grad is not None]
        gradients = [cast(torch.Tensor, parameter.grad) for parameter in parameters]
        norm = stable_total_norm(gradients)
        if not torch.isfinite(norm):
            # A finite norm already establishes finite gradients. Only inspect
            # entries on failure, distinguishing scaling overflow from overflow
            # of the norm of otherwise finite (but enormous) gradients.
            if scaler is not None and not finite_tensors(gradients):
                if attempt < MAX_SCALE_RETRIES and precision.lower_scale():
                    assert replay is not None
                    replay.restore(network)
                    continue
                raise nonfinite_batch_error('AMP gradient (loss-scaling retry budget exhausted)', batch, batch_number)
            raise nonfinite_batch_error('AMP gradient norm', batch, batch_number)
        if statistics is not None:
            statistics.norms.append(float(norm.detach()))
        if clip_norm is not None:
            clip_with_norm(parameters, clip_norm, norm)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        return loss
    raise AssertionError('AMP retry loop must return or raise')


def fit_in_one_step(network: ParrotArchitecture, train_loader: DataLoader, val_loader: DataLoader, task: Task,
                    device: str | torch.device,
                    on_improvement: Callable[[ParrotArchitecture], None] | None = None,
                    silent: bool = False, timing: TrainingTiming | None = None) -> tuple[list[float], list[float]]:
    """Fit a model that is not trained by gradient descent, and report its losses

    With uncertainties, the model is refitted as its model error is
    re-estimated (see fit_with_model_error()).

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
    1 / (SD^2 + s^2), until the model error changes by less than the fraction
    MODEL_ERROR_TOLERANCE of its value (or after MAX_MODEL_ERROR_REFITS
    refits). The model error is left fitted to the final model.

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
