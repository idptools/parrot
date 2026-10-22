"""Stable opt-in gradient checks without changing parameter or gradient dtypes."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


@torch.no_grad()
def stable_total_norm(gradients: list[torch.Tensor], raw_norms: list[float] | None = None) -> torch.Tensor:
    """Keep the ordinary fast reduction, falling back only when it overflows.

    Actual NaN/Inf entries remain errors. CPU/CUDA accumulate the fallback in
    float64. MPS cannot use float64 tensors, so rescale on device and combine
    the two scalar factors in Python; return that scalar as a CPU double.
    """
    norm = nn.utils.get_total_norm(gradients)
    if raw_norms is not None:
        raw_norms.append(float(norm))
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


@torch.no_grad()
def gradient_diagnostics(network: nn.Module, norm: torch.Tensor | None = None) -> dict[str, Any]:
    """Detailed failure-only statistics; never replace NaN/Inf entries in training."""
    nan_count = inf_count = 0
    maximum = 0.
    affected = []
    dtypes = set()
    for name, parameter in network.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        dtypes.add(str(gradient.dtype))
        nans, infs = int(torch.isnan(gradient).sum()), int(torch.isinf(gradient).sum())
        nan_count += nans
        inf_count += infs
        if nans or infs:
            affected.append({'parameter': name, 'nan': nans, 'inf': infs})
        finite = gradient[torch.isfinite(gradient)]
        if finite.numel():
            maximum = max(maximum, float(finite.abs().max()))
    return {'nan': nan_count, 'inf': inf_count, 'max_finite_abs': maximum,
            'norm': float(norm) if norm is not None else None,
            'gradient_dtypes': sorted(dtypes), 'affected_parameters': affected}


def diagnostic_text(details: dict[str, Any]) -> str:
    """Put numeric evidence ahead of potentially long sequence identifiers."""
    norm = details['norm']
    norm_text = f'{norm:.6g}' if norm is not None and math.isfinite(norm) else str(norm)
    names = ','.join(item['parameter'] for item in details['affected_parameters'][:3]) or 'none'
    return (f"nan={details['nan']} inf={details['inf']} max_finite_abs={details['max_finite_abs']:.6g} "
            f"norm={norm_text} affected={names}")
