"""Stable opt-in gradient checks without changing parameter or gradient dtypes."""

from collections.abc import Iterable

import torch
from torch import nn


@torch.no_grad()
def stable_total_norm(gradients: list[torch.Tensor], initial_norm: torch.Tensor | None = None
                      ) -> torch.Tensor:
    """Keep the ordinary fast reduction, falling back only when it overflows.

    Actual NaN/Inf entries remain errors. CPU/CUDA accumulate the fallback in
    float64. MPS cannot use float64 tensors, so rescale on device and combine
    the two scalar factors in Python; return that scalar as a CPU double.
    """
    norm = nn.utils.get_total_norm(gradients) if initial_norm is None else initial_norm
    if bool(torch.isfinite(norm)) or any(not bool(torch.isfinite(g).all()) for g in gradients):
        return norm
    if not gradients:
        return norm
    if gradients[0].device.type != 'mps':
        norms = [torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients]
        return torch.linalg.vector_norm(torch.stack(norms))
    maximum = max(float(g.abs().max()) for g in gradients if g.numel())
    if maximum == 0:
        return torch.tensor(0., dtype=torch.float64)
    scaled = [torch.linalg.vector_norm(g / maximum) for g in gradients]
    value = maximum * float(torch.linalg.vector_norm(torch.stack(scaled)))
    return torch.tensor(value, dtype=torch.float64)


@torch.no_grad()
def clip_with_norm(parameters: Iterable[nn.Parameter], max_norm: float, norm: torch.Tensor) -> None:
    """Clip original gradients, including the CPU-double norm used for MPS."""
    parameters = list(parameters)
    coefficient = min(1., max_norm / (float(norm) + 1e-6)) if norm.dtype == torch.float64 else None
    if parameters and coefficient is not None and (parameters[0].device.type == 'mps'
                                                   or coefficient < torch.finfo(parameters[0].dtype).tiny):
        # Two factors avoid underflow of a tiny multiplier on accelerators that
        # flush subnormal float32 values to zero. Intermediate values stay <= 1.
        gradients = [p.grad for p in parameters if p.grad is not None]
        maximum = max(float(g.abs().max()) for g in gradients if g.numel())
        if coefficient == 1. or maximum == 0:
            return
        target_scaled_norm = max_norm / (float(norm) / maximum)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(maximum).mul_(target_scaled_norm)
    else:
        nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
