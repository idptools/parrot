"""Explicit, bounded capture of one numerical failure for exact batch replay."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from parrot.devices import cpu_copy
from parrot.training_persistence import atomic_save, capture_random_state


def environment(device: str | torch.device) -> dict[str, Any]:
    """Record the actual runtime rather than a separately installed CUDA compiler."""
    selected = torch.device(device)
    result = {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'device': str(selected),
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        cudnn: Any = torch.backends.cudnn
        result.update(matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                      cudnn_rnn_precision=cudnn.rnn.fp32_precision,
                      cudnn_conv_precision=cudnn.conv.fp32_precision)
    else:
        result.update(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_tf32=torch.backends.cudnn.allow_tf32)
    if selected.type == 'cuda':
        with torch.cuda.device(selected):
            result.update(gpu=torch.cuda.get_device_name(selected),
                          capability=list(torch.cuda.get_device_capability(selected)),
                          native_bfloat16=torch.cuda.is_bf16_supported(including_emulation=False))
    return result


class FailureCapture:
    """Snapshot before a batch only when requested and no first failure exists.

    Capturing copies model and optimizer state to CPU every batch until a
    failure, so it is deliberately a debugging option. No files are written on
    success. Existing captures are never replaced, including during recovery.
    """

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path is not None else None

    @contextmanager
    def watch(self, network, optimizer, batch, task, device, batch_number, clip_norm,
              equal_protein_weighting, precision, error_type):
        if self.path is None or self.path.exists():
            yield
            return
        snapshot = {
            'parrot_failure_capture': 1,
            'network': {'architecture': getattr(network, 'name', None),
                        'input_size': getattr(network, 'input_size', None),
                        'hyperparameters': getattr(network, 'hyperparameter_values', None)},
            'task': task.to_config(), 'model_state': cpu_copy(network.state_dict()),
            'optimizer': type(optimizer).__name__, 'optimizer_state': cpu_copy(optimizer.state_dict()),
            'batch': {name: cpu_copy(value) for name, value in batch._asdict().items()},
            'batch_number': batch_number, 'random_state': capture_random_state((), device),
            'gradient_clip_norm': clip_norm, 'equal_protein_weighting': equal_protein_weighting,
            'precision_state': precision.state_dict() if precision is not None else None,
            'environment': environment(device),
        }
        try:
            yield
        except error_type as error:
            snapshot['error'] = str(error)
            snapshot['diagnostics'] = getattr(error, 'diagnostics', {})
            try:
                atomic_save(snapshot, str(self.path), overwrite=False)
            except FileExistsError:
                error.add_note(f'Existing failure capture retained: {self.path}')
            except OSError as capture_error:
                error.add_note(f'Could not save failure capture: {capture_error}')
            else:
                # Include the location in ordinary CLI errors too (not only a traceback's notes).
                error.args = (f'{error} Failure capture: {self.path}',)
            raise
