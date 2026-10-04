"""
Choosing the device PARROT runs on: the CPU, an NVIDIA GPU (CUDA) or an
Apple-silicon GPU (MPS).

Every command that trains or predicts takes the same choice ('auto', 'cpu',
'cuda' or 'mps'). 'auto' uses the fastest device available, in the order CUDA,
then MPS, then CPU. Asking for a device that is not available is an error rather
than a silent fallback, because training on the CPU instead of a GPU can be an
order of magnitude slower.

PARROT always computes in full float32 precision, so a network gives the same
results (to float32 precision) on every device; see full_float32_precision().

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

# Values accepted wherever a device is chosen (the --device flag and Predictor)
DEVICE_CHOICES = ('auto', 'cpu', 'cuda', 'mps')


def mps_available() -> bool:
    """True if PyTorch can use an Apple-silicon GPU (MPS) on this machine."""
    return torch.backends.mps.is_available()


def cuda_available() -> bool:
    """True if PyTorch can use an NVIDIA GPU (CUDA) on this machine."""
    return torch.cuda.is_available()


def resolve_device(requested: str = 'auto', gpu_id: int | None = None) -> torch.device:
    """Turn a device choice into the torch.device to run on

    Parameters
    ----------
    requested : str, optional
        'auto' (default: CUDA if available, else MPS if available, else the
        CPU), 'cpu', 'cuda' or 'mps'
    gpu_id : int, optional
        Index of the CUDA device to use. Only valid when running on CUDA.

    Returns
    -------
    torch.device

    Raises
    ------
    ValueError
        If `requested` is not a valid choice, or `gpu_id` is given for a
        device other than CUDA or is not a valid CUDA device index
    RuntimeError
        If the requested device (or the CUDA device implied by `gpu_id`) is not
        available on this machine
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{requested}': choose from {', '.join(DEVICE_CHOICES)}")

    # A GPU index is an explicit request for CUDA
    if gpu_id is not None:
        if requested in ('cpu', 'mps'):
            raise ValueError(f"A CUDA device index (--gpu-id {gpu_id}) cannot be combined with device '{requested}'")
        requested = 'cuda'

    if requested == 'cpu':
        return torch.device('cpu')

    if requested == 'cuda' or (requested == 'auto' and cuda_available()):
        return _cuda_device(gpu_id)

    if requested == 'mps' or (requested == 'auto' and mps_available()):
        if not mps_available():
            reason = ('this PyTorch build does not include MPS support' if not torch.backends.mps.is_built()
                      else 'this machine has no Apple-silicon GPU, or macOS is too old')
            raise RuntimeError(f'The MPS (Apple-silicon GPU) device was requested but is not available: {reason}')
        return torch.device('mps')

    return torch.device('cpu')


def _cuda_device(gpu_id: int | None) -> torch.device:
    """The default CUDA device, or the one with index `gpu_id`, checking it exists."""
    if not cuda_available():
        raise RuntimeError('A CUDA (NVIDIA GPU) device was requested but CUDA is not available on this machine')
    if gpu_id is None:
        return torch.device('cuda')

    num_devices = torch.cuda.device_count()
    if not 0 <= gpu_id < num_devices:
        raise ValueError(f'CUDA device {gpu_id} does not exist: this machine has {num_devices} CUDA device(s), '
                         f'numbered from 0')
    return torch.device(f'cuda:{gpu_id}')


@contextlib.contextmanager
def full_float32_precision() -> Iterator[None]:
    """Compute float32 convolutions, LSTMs and matrix products in full precision on NVIDIA GPUs

    On Ampere and newer NVIDIA GPUs, PyTorch lets cuDNN compute float32
    convolutions and LSTMs in TF32 by default (and matrix products too, if
    torch.set_float32_matmul_precision has been lowered). TF32 keeps only 10
    bits of the mantissa: BRNN, CNN and U-Net predictions on CUDA differed from
    the same network's predictions on the CPU or an Apple-silicon GPU by up to
    ~1e-2 (in the units of the predicted values), against ~1e-5 between the CPU
    and MPS. In full precision CUDA agrees with them to ~1e-4. This costs
    nothing for the BRNN; CNNs and U-Nets on protein language model embeddings
    train up to ~1.5x slower (measured on an RTX A4000).

    PARROT runs every training, evaluation, prediction and embedding step inside
    this context. The previous settings are restored afterwards, so other code
    running in the same process is unaffected. Has no effect on other devices.

    Can also be used as a decorator.
    """
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def describe_device(device: torch.device) -> str:
    """A short human-readable name for a device, for printing."""
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f'CUDA GPU {index} ({torch.cuda.get_device_name(index)})'
    if device.type == 'mps':
        return 'Apple-silicon GPU (MPS)'
    return 'CPU'


def cpu_copy(value: Any) -> Any:
    """Copy a tensor, or a (nested) dictionary or list of tensors such as a state dict, to the CPU

    Tensors are always copied, even if they are already on the CPU, so the copy
    stays as it is while training goes on changing the original. Anything that
    is not a tensor (e.g. a fitted scikit-learn model in a classical network's
    state dict) is kept as it is.

    Parameters
    ----------
    value : torch.Tensor, dict, list or anything else
        What to copy, e.g. network.state_dict() or optimizer.state_dict()

    Returns
    -------
    The same structure, with every tensor replaced by a detached copy on the CPU
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return value
