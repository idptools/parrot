"""Explicit CUDA AMP policy, with bounded loss-scaling retries and FP32 losses.

Parameters and optimizer moments stay float32. Autocast wraps only the network
forward pass; evaluation and model-error fitting retain PARROT's FP32 behavior.
"""

import copy
import math
import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from parrot.devices import cpu_copy

AMP_MODES = ('off', 'auto', 'float16', 'bfloat16')
MAX_SCALE_RETRIES = 8


def validate_amp_architecture(architecture: str, mode: str) -> None:
    """Reject unsupported BRNN AMP before data, model state or CUDA are touched."""
    if architecture == 'brnn' and mode != 'off':
        raise ValueError('AMP is currently unsupported for BRNN training because of numerical failures in the '
                         'CUDA LSTM autocast pathway. Use --amp off (TrainingOptions(amp="off") in Python). '
                         'Start a new FP32 run; BRNN AMP checkpoints cannot be resumed.')


def resolve_amp(mode: str, device: str | torch.device) -> torch.dtype | None:
    """Validate an explicitly requested CUDA mode on the selected GPU.

    This implementation requires CUDA compute capability 7.0 or newer. BF16
    requires native hardware support; auto chooses native BF16, else FP16.
    CPU and MPS AMP are intentionally unsupported and never silently substituted.
    """
    if mode not in AMP_MODES:
        raise ValueError(f'AMP mode must be one of {AMP_MODES}')
    if mode == 'off':
        return None
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('AMP requires an available CUDA GPU; CPU and MPS AMP are not supported by PARROT')
    with torch.cuda.device(selected):
        if torch.cuda.get_device_capability(selected) < (7, 0):
            raise ValueError('PARROT AMP requires CUDA compute capability 7.0 or newer')
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if mode == 'bfloat16' and not bf16:
        raise ValueError('The selected CUDA GPU does not support native bfloat16 AMP; use float16 or disable AMP')
    return torch.bfloat16 if mode == 'bfloat16' or (mode == 'auto' and bf16) else torch.float16


class TrainingPrecision:
    """AMP runtime and serializable scaler state for one training run or search fold."""

    def __init__(self, mode: str, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.dtype = resolve_amp(mode, device)
        self.scaler = torch.amp.GradScaler('cuda') if self.dtype == torch.float16 else None
        self.scale_retries = 0

    def autocast(self) -> Any:
        return nullcontext() if self.dtype is None else torch.autocast('cuda', dtype=self.dtype)

    def validate_architecture(self, architecture: str) -> None:
        """Apply the same restriction to callers using the low-level runtime API."""
        validate_amp_architecture(architecture, 'off' if self.dtype is None else 'auto')

    def state_dict(self) -> dict[str, Any]:
        return {'dtype': str(self.dtype), 'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                'scale_retries': self.scale_retries}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state['dtype'] != str(self.dtype):
            raise ValueError('AMP dtype changed since the checkpoint; resume on a compatible GPU with the same dtype')
        if self.scaler is not None:
            # A failed backward may leave the old scaler in its UNSCALED stage.
            # load_state_dict alone does not clear that per-optimizer bookkeeping.
            self.scaler = torch.amp.GradScaler('cuda')
            self.scaler.load_state_dict(copy.deepcopy(state['scaler']))
        self.scale_retries = state['scale_retries']

    def lower_scale(self) -> bool:
        """Consume GradScaler's found-inf result, without taking an optimizer step."""
        assert self.scaler is not None
        previous = self.scaler.get_scale()
        self.scaler.update()
        current = self.scaler.get_scale()
        self.scale_retries += 1
        return math.isfinite(current) and 0 < current < previous


class BatchReplay:
    """Restore dropout randomness and forward-updated buffers for an FP16 scaling retry.

    The data batch is retained, and parameters have not had an optimizer update.
    No batch is discarded. Only buffers are copied, avoiding full parameter
    snapshots on every batch.
    """

    def __init__(self, network: torch.nn.Module, device: torch.device) -> None:
        self.buffers = cpu_copy(dict(network.named_buffers()))
        self.python = random.getstate()
        self.numpy = copy.deepcopy(np.random.get_state())
        self.cpu = torch.get_rng_state().clone()
        self.cuda = torch.cuda.get_rng_state(device)
        self.device = device

    def restore(self, network: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, buffer in network.named_buffers():
                buffer.copy_(self.buffers[name])
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state(self.cuda, self.device)
