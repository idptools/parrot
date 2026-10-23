"""Opt-in loss weighting, learning-rate control and in-process training rollback.

Ordinary training does not construct these controllers or take snapshots.
Rollback snapshots include the model, optimizer, LR controller and random states;
they are held in memory and are separate from inference checkpoints.
"""

import copy
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from parrot.devices import cpu_copy
from parrot.tasks import Task
from parrot.training_precision import AMP_MODES, TrainingPrecision

LR_SCHEDULES = ('fixed', 'cosine', 'exponential', 'plateau')


@dataclass(frozen=True)
class TrainingOptions:
    """Optional neural-network training policies; defaults preserve the original behavior.

    Pass to ``train(..., training_options=...)`` or ``TrialTrainer``. Gradient
    clipping and finite checks are separate arguments to those functions.
    See the Training controls guide for loss units, timing and CLI equivalents.

    Parameters
    ----------
    equal_protein_weighting : bool, optional
        For residue and matrix tasks, average labelled losses within each
        protein, then across proteins. Default False: sum labelled losses.
        Validation and test loss must use the same weighting explicitly when
        evaluated outside training. Standard metrics still pool labelled values.
    recover_nonfinite : int, optional
        Maximum rollbacks across the run, including promoted search stages.
        Default 0 (off). Restore a clean epoch boundary and replay at a lower
        LR; invalid inputs and arbitrary runtime errors are not retried.
        Recovery requires loaders with num_workers=0 and keeps CPU snapshots
        in memory; it does not support restarting a crashed process.
    recovery_lr_factor : float, optional
        LR multiplier per rollback, strictly between 0 and 1 (default 0.5).
    lr_schedule : str, optional
        'fixed' (default), 'cosine', 'exponential' or validation-driven 'plateau'.
    lr_anneal_epochs : int or None, optional
        Positive cosine decay duration after warmup. ``resolved()`` fills an
        omitted duration from the full run minus warmup, at least 1. An
        explicit duration is required with automatic stopping. Cosine holds
        at the floor after its duration rather than starting another cycle.
    lr_min : float, optional
        Finite, nonnegative LR floor for warmup, schedules and recovery
        (default 0); cannot exceed the initial LR.
    lr_gamma : float, optional
        Exponential multiplier per completed decay epoch (default 0.95).
    lr_factor : float, optional
        Plateau reduction factor (default 0.5). Like lr_gamma, strictly
        between 0 and 1.
    lr_patience : int, optional
        Tolerated epochs without sufficient improvement before plateau
        reduces LR (default 5; reduction after the sixth).
    lr_threshold : float, optional
        Minimum absolute validation-loss improvement for plateau (default
        0.0001); supports negative Gaussian losses.
    lr_cooldown : int, optional
        Plateau cooldown epochs after a reduction (default 0).
    lr_warmup_epochs : int, optional
        Linear warmup from initial-LR/N to initial LR over N epochs, before
        any schedule (default 0). The floor also applies during warmup.
    optimizer : str, optional
        'adam' (default) or 'adamw'. The same optimizer is used for every search
        fold and final retraining.
    weight_decay : float, optional
        Finite nonnegative coefficient, default 0 for both optimizers. AdamW
        applies decoupled decay; Adam uses an L2 penalty in its adaptive update.
    amp : str, optional
        'off' (default), 'auto', 'float16' or 'bfloat16'. CUDA only, with hardware
        checks. Auto prefers native BF16. FP16 uses bounded same-batch scaling
        retries; model parameters and loss arithmetic remain FP32. BRNN
        training supports 'off' only, including checkpoint resume.
    """

    equal_protein_weighting: bool = False
    recover_nonfinite: int = 0
    recovery_lr_factor: float = 0.5
    lr_schedule: str = 'fixed'
    lr_anneal_epochs: int | None = None
    lr_min: float = 0.0
    lr_gamma: float = 0.95
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_threshold: float = 1e-4
    lr_cooldown: int = 0
    lr_warmup_epochs: int = 0
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    amp: str = 'off'

    def __post_init__(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f'LR schedule must be one of {LR_SCHEDULES}')
        if self.optimizer not in ('adam', 'adamw'):
            raise ValueError('Optimizer must be adam or adamw')
        if isinstance(self.weight_decay, bool) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Weight decay must be finite and nonnegative')
        if self.amp not in AMP_MODES:
            raise ValueError(f'AMP mode must be one of {AMP_MODES}')
        for name in ('recover_nonfinite', 'lr_patience', 'lr_cooldown', 'lr_warmup_epochs'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.lr_anneal_epochs is not None and (isinstance(self.lr_anneal_epochs, bool)
                                                  or not isinstance(self.lr_anneal_epochs, int)
                                                  or self.lr_anneal_epochs < 1):
            raise ValueError('LR anneal epochs must be a positive integer')
        for name in ('recovery_lr_factor', 'lr_gamma', 'lr_factor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be finite and strictly between 0 and 1')
        for name in ('lr_min', 'lr_threshold'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    @property
    def controls_lr(self) -> bool:
        return self.lr_schedule != 'fixed' or self.lr_warmup_epochs > 0 or self.recover_nonfinite > 0

    @property
    def active(self) -> bool:
        return (self.equal_protein_weighting or self.controls_lr or self.optimizer != 'adam'
                or self.weight_decay != 0 or self.amp != 'off')

    def build_optimizer(self, network: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        """Construct the same configured optimizer for direct training and every search fold."""
        optimizer = torch.optim.AdamW if self.optimizer == 'adamw' else torch.optim.Adam
        if self.weight_decay or self.optimizer == 'adamw':
            return optimizer(network.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        return optimizer(network.parameters(), lr=learning_rate)

    def resolved(self, n_epochs: int, stop_condition: str = 'iter') -> 'TrainingOptions':
        """Resolve a cosine horizon once, against the full run rather than a search stage."""
        if self.lr_schedule == 'cosine' and self.lr_anneal_epochs is None:
            if stop_condition == 'auto':
                raise ValueError('Cosine annealing with --stop auto needs --lr-anneal-epochs')
            return replace(self, lr_anneal_epochs=max(1, n_epochs - self.lr_warmup_epochs))
        return self

    def validate_task(self, task: Task) -> None:
        if self.equal_protein_weighting and not (task.per_residue or task.is_matrix):
            raise ValueError('Equal-protein weighting applies to residue or matrix tasks, not sequence tasks')

    def active_settings(self) -> dict[str, Any]:
        """Only effective settings, suitable for headers and the network info file."""
        settings: dict[str, Any] = {}
        if self.optimizer != 'adam' or self.weight_decay:
            settings.update(optimizer=self.optimizer, weight_decay=self.weight_decay)
        if self.amp != 'off':
            settings['amp'] = self.amp
        if self.equal_protein_weighting:
            settings['equal_protein_weighting'] = True
        if self.controls_lr:
            settings.update(lr_schedule=self.lr_schedule, lr_min=self.lr_min)
        if self.lr_warmup_epochs:
            settings['lr_warmup_epochs'] = self.lr_warmup_epochs
        if self.lr_schedule == 'cosine':
            settings['lr_anneal_epochs'] = self.lr_anneal_epochs
        if self.lr_schedule == 'exponential':
            settings['lr_gamma'] = self.lr_gamma
        if self.lr_schedule == 'plateau':
            settings.update(lr_factor=self.lr_factor, lr_patience=self.lr_patience,
                            lr_threshold=self.lr_threshold, lr_cooldown=self.lr_cooldown)
        if self.recover_nonfinite:
            settings.update(recover_nonfinite=self.recover_nonfinite, recovery_lr_factor=self.recovery_lr_factor)
        return settings


@dataclass
class TrainingControlState:
    """LR state, recovery budget and diagnostics, also carried across search stages.

    Pass a fresh instance to ``train(..., control_state=...)`` to inspect its
    diagnostics. History is populated when LR control or recovery is enabled;
    clipping, finite checks or equal-protein weighting alone do not populate it.

    Attributes
    ----------
    scheduler_state : dict or None
        State retained between promoted search stages, including the permanent
        recovery multiplier. Not a complete checkpoint for resuming training.
    recoveries_used : int
        Cumulative rollbacks used, without resetting after successful epochs.
    epoch_history : list of dict
        One row per successful epoch: epoch (1-based), training_loss,
        validation_loss, learning_rate used, and cumulative recoveries.
    recovery_events : list of dict
        Each rollback's epoch, attempt, old_lr, new_lr and failure reason.
    """

    scheduler_state: dict[str, Any] | None = None
    recoveries_used: int = 0
    epoch_history: list[dict[str, Any]] = field(default_factory=list)
    recovery_events: list[dict[str, Any]] = field(default_factory=list)
    precision_state: dict[str, Any] | None = None

    def report(self) -> dict[str, Any]:
        """Return nonempty history and recovery records for reporting, without writing files."""
        result: dict[str, Any] = {}
        if self.epoch_history:
            result['learning_rate_history'] = self.epoch_history
        if self.recovery_events:
            result['recovery_events'] = self.recovery_events
        if self.precision_state is not None:
            result.update(amp_dtype=self.precision_state['dtype'],
                          amp_scale_retries=self.precision_state['scale_retries'])
        return result


class LearningRateController:
    """Epoch-based schedules with a permanent recovery multiplier and a lower LR bound.

    Cosine decay is clamped at its horizon, so it never starts another cycle.
    Plateau reductions use PyTorch's scheduler with an absolute loss threshold,
    which also works with negative Gaussian log-likelihood losses.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, options: TrainingOptions) -> None:
        self.optimizer = optimizer
        self.options = options
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        if any(not math.isfinite(lr) or lr <= 0 or options.lr_min > lr for lr in self.base_lrs):
            raise ValueError('LR control needs a positive finite initial LR and --lr-min no greater than it')
        if options.lr_schedule == 'cosine' and options.lr_anneal_epochs is None:
            raise ValueError('Cosine annealing needs a full-run annealing horizon')
        self.recovery_scale = 1.0
        self.plateau = None
        if options.lr_schedule == 'plateau':
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=options.lr_factor, patience=options.lr_patience,
                threshold=options.lr_threshold, threshold_mode='abs', cooldown=options.lr_cooldown,
                min_lr=options.lr_min, eps=0.0)

    @property
    def lrs(self) -> list[float]:
        return [float(group['lr']) for group in self.optimizer.param_groups]

    def prepare_epoch(self, epoch: int) -> None:
        """Set the rate for a zero-based epoch, before its first optimizer update."""
        options = self.options
        warmup = options.lr_warmup_epochs
        if self.plateau is not None and epoch >= warmup and not (warmup and epoch == warmup):
            return
        elapsed = max(0, epoch - warmup)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < warmup:
                nominal = base * (epoch + 1) / warmup
            elif options.lr_schedule == 'cosine':
                assert options.lr_anneal_epochs is not None
                fraction = min(elapsed / options.lr_anneal_epochs, 1.0)
                nominal = options.lr_min + (base - options.lr_min) * (1 + math.cos(math.pi * fraction)) / 2
            elif options.lr_schedule == 'exponential':
                nominal = base * options.lr_gamma ** elapsed
            else:
                nominal = base
            group['lr'] = max(options.lr_min, nominal * self.recovery_scale)

    def finish_epoch(self, epoch: int, val_loss: float) -> None:
        """Observe validation only after successful training and validation, never after a failed attempt."""
        if self.plateau is not None and epoch >= self.options.lr_warmup_epochs:
            self.plateau.step(val_loss)

    def reduce_after_rollback(self, failed_lrs: list[float]) -> list[float]:
        """Reduce cumulatively even when several retries restore the same epoch snapshot."""
        targets = [max(self.options.lr_min, lr * self.options.recovery_lr_factor) for lr in failed_lrs]
        if any(target <= 0 or target >= failed for target, failed in zip(targets, failed_lrs)):
            raise ValueError('Cannot recover: the learning rate cannot be reduced further above its lower bound')
        restored = self.lrs
        self.recovery_scale *= min(target / old for target, old in zip(targets, restored))
        for group, target in zip(self.optimizer.param_groups, targets):
            group['lr'] = target
        return targets

    def state_dict(self) -> dict[str, Any]:
        return {'base_lrs': list(self.base_lrs), 'recovery_scale': self.recovery_scale,
                'plateau': None if self.plateau is None else copy.deepcopy(self.plateau.state_dict())}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = list(state['base_lrs'])
        self.recovery_scale = state['recovery_scale']
        if self.plateau is not None and state['plateau'] is not None:
            self.plateau.load_state_dict(state['plateau'])


@dataclass
class EpochSnapshot:
    """A clean epoch boundary for replay, including explicit DataLoader generators when present."""

    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    python_rng: Any
    numpy_rng: Any
    torch_rng: torch.Tensor
    accelerator_rng: torch.Tensor | None
    generators: list[tuple[torch.Generator, torch.Tensor]]
    device: torch.device
    precision: dict[str, Any] | None = None

    @classmethod
    def capture(cls, network: torch.nn.Module, optimizer: torch.optim.Optimizer,
                controller: LearningRateController, loaders: tuple[DataLoader, ...],
                device: str | torch.device, precision: TrainingPrecision | None = None) -> 'EpochSnapshot':
        selected = torch.device(device)
        generators = []
        seen = set()
        for loader in loaders:
            if loader.num_workers:
                raise ValueError('Automatic recovery requires DataLoaders with num_workers=0 for reproducible replay')
            for owner in (loader, loader.sampler, loader.batch_sampler):
                generator = getattr(owner, 'generator', None)
                if isinstance(generator, torch.Generator) and id(generator) not in seen:
                    seen.add(id(generator))
                    generators.append((generator, generator.get_state().clone()))
        accelerator = None
        if selected.type == 'cuda':
            accelerator = torch.cuda.get_rng_state(selected)
        elif selected.type == 'mps':
            accelerator = torch.mps.get_rng_state()
        return cls(cpu_copy(network.state_dict()), cpu_copy(optimizer.state_dict()), controller.state_dict(),
                   random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
                   accelerator, generators, selected, precision.state_dict() if precision is not None else None)

    def restore(self, network: torch.nn.Module, optimizer: torch.optim.Optimizer,
                controller: LearningRateController, precision: TrainingPrecision | None = None) -> None:
        network.load_state_dict(self.model)
        # load_state_dict can retain CPU tensor references, so each replay gets fresh optimizer buffers.
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        optimizer.zero_grad(set_to_none=True)
        controller.load_state_dict(self.scheduler)
        if precision is not None and self.precision is not None:
            precision.load_state_dict(self.precision)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        if self.accelerator_rng is not None:
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state(self.accelerator_rng, self.device)
            elif self.device.type == 'mps':
                torch.mps.set_rng_state(self.accelerator_rng)
        for generator, state in self.generators:
            generator.set_state(state)


def finite_tensors(value: Any) -> bool:
    """Check nested model/optimizer tensors with one host synchronization per device."""
    checks: dict[torch.device, list[torch.Tensor]] = {}

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            checks.setdefault(item.device, []).append(torch.isfinite(item).all())
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    return all(bool(torch.stack(group).all()) for group in checks.values())
