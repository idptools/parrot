"""
Training on regression values that come with uncertainties (--uncertainties).

Each value y_i is a measurement of a true value, with a measurement error whose
standard deviation (SD) sigma_i is given in the datafile. A network cannot
predict the true values exactly either, so we take the difference between a
value and its prediction mu_i to be the sum of two independent Gaussian errors:
the measurement error (SD sigma_i) and the network's own error (SD s, the
"model error", the same for every value). Then

    y_i - mu_i ~ Normal(0, sigma_i^2 + s^2)

and training minimises the negative log-likelihood of the values,

    NLL = sum over i of 1/2 [log(2 pi (sigma_i^2 + s^2)) + (y_i - mu_i)^2 / (sigma_i^2 + s^2)]

so each value is weighted by 1 / (sigma_i^2 + s^2): precise values pull the
network harder than noisy ones, but no value counts for more than 1 / s^2,
however small its SD (and an SD of 0 is allowed).

s is not a network weight. Before training and after every epoch it is set to
its maximum-likelihood value given the network's residuals on the validation
set (see fit_model_error() and train_network.refit_model_error()), so it matches
the network at the end of each epoch (and the network kept from training keeps
the s of its epoch), and describes how far the network's predictions are from
the true values on data it was not trained on. Models fitted in one step
instead alternate between fitting their weights and s (see
train_network.fit_with_model_error()). This is the likelihood estimate of the
between-study variance in a random-effects meta-analysis with known
within-study variances (Hardy & Thompson 1996, Statistics in Medicine
15:619-629), with the predictions in place of the pooled mean.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math

import numpy as np
import torch
import torch.nn as nn

# Name of the buffer holding a network's model error s (only networks trained with uncertainties have one)
MODEL_ERROR_BUFFER = 'model_error'

# log(2 pi), the constant of the Gaussian log-likelihood
LOG_2PI = math.log(2 * math.pi)

# The model error is found by bisection, halving the interval it lies in this many times (enough to locate it to
# about 1e-10 of the largest residual)
MODEL_ERROR_BISECTION_STEPS = 64

# When some values have an SD of 0, the model error must stay above 0 (or those values would have no variance at
# all): it is not taken below this fraction of the largest residual (PARROT's choice: far below any error that
# matters, but enough to keep the loss finite)
MIN_MODEL_ERROR_FRACTION = 1e-6

# Smallest total variance used in the loss, so a value with an SD of 0 predicted exactly (a residual and a model
# error of exactly 0) gives a finite loss rather than log(0)
MIN_VARIANCE = float(torch.finfo(torch.float32).tiny)

# Significant figures the model error s is reported to (it is in the units of the values, so decimal places would
# suit some datasets and not others)
MODEL_ERROR_SIGNIFICANT_FIGURES = 4


def gaussian_nll(predicted: torch.Tensor, values: torch.Tensor, sds: torch.Tensor,
                 model_error: torch.Tensor) -> torch.Tensor:
    """The negative log-likelihood of values given their predictions, summed (see the module docstring)

    Parameters
    ----------
    predicted : torch.Tensor
        The predictions mu_i (any shape)
    values : torch.Tensor
        The measured values y_i (same shape)
    sds : torch.Tensor
        The SD sigma_i of each value's measurement error (same shape)
    model_error : torch.Tensor
        The model error s (a scalar)

    Returns
    -------
    torch.Tensor
        A scalar: the sum over every value of
        1/2 [log(2 pi (sigma^2 + s^2)) + (y - mu)^2 / (sigma^2 + s^2)],
        with sigma^2 + s^2 taken to be at least MIN_VARIANCE
    """
    variance = torch.clamp(sds ** 2 + model_error ** 2, min=MIN_VARIANCE)
    return 0.5 * torch.sum(LOG_2PI + torch.log(variance) + (values - predicted) ** 2 / variance)


def fit_model_error(residuals: np.ndarray, sds: np.ndarray) -> float:
    """The maximum-likelihood model error s, given residuals and the SDs of their values' measurement errors

    The log-likelihood of s (see the module docstring) is greatest where its
    derivative with respect to t = s^2,

        sum over i of (r_i^2 - sigma_i^2 - t) / (sigma_i^2 + t)^2,

    is zero, so s is found by bisecting on the sign of that sum. If the
    measurement errors alone explain the residuals (the sum is not positive
    even at t = 0), s is 0. If some SDs are 0, s is not taken below
    MIN_MODEL_ERROR_FRACTION of the largest residual.

    Parameters
    ----------
    residuals : numpy array
        Value minus prediction, for every labelled value (any shape)
    sds : numpy array
        The SD of each value's measurement error (same shape)

    Returns
    -------
    float
        s, in the units of the values (0 if every residual is 0)

    Raises
    ------
    ValueError
        If there are no residuals, the shapes differ, or a residual or SD is
        not finite or an SD is negative
    """
    squared_residuals = np.ravel(np.asarray(residuals, dtype=np.float64)) ** 2
    variances = np.ravel(np.asarray(sds, dtype=np.float64)) ** 2
    if squared_residuals.shape != variances.shape:
        raise ValueError(f'Got {squared_residuals.size} residuals but {variances.size} SDs')
    if squared_residuals.size == 0:
        raise ValueError('The model error needs at least one residual')
    if not (np.all(np.isfinite(squared_residuals)) and np.all(np.isfinite(variances)) and np.all(np.ravel(sds) >= 0)):
        raise ValueError('Residuals and SDs must be finite, and SDs not negative')

    def slope(model_variance: float) -> float:
        """The derivative of the log-likelihood with respect to s^2 (up to a factor of 1/2)."""
        total = variances + model_variance
        return float(np.sum((squared_residuals - total) / total ** 2))

    highest = float(squared_residuals.max())
    if highest == 0:
        return 0.0
    # The slope is not positive at the largest squared residual (no term is), so the maximum lies at or below it
    lowest = 0.0 if np.all(variances > 0) else (MIN_MODEL_ERROR_FRACTION ** 2) * highest
    if slope(lowest) <= 0:
        return math.sqrt(lowest)

    for _ in range(MODEL_ERROR_BISECTION_STEPS):
        middle = (lowest + highest) / 2
        if slope(middle) > 0:
            lowest = middle
        else:
            highest = middle
    return math.sqrt((lowest + highest) / 2)


def model_error_of(network: nn.Module) -> torch.Tensor:
    """The model error buffer of a network trained with uncertainties (a scalar tensor on the network's device)

    Parameters
    ----------
    network : torch.nn.Module
        A network built for a task with uncertainties

    Returns
    -------
    torch.Tensor
        The buffer itself (so filling it changes the network's model error),
        NaN until the model error has first been fitted

    Raises
    ------
    ValueError
        If the network has no model error (it was not built for a task with uncertainties)
    """
    buffers = dict(network.named_buffers(recurse=False))
    if MODEL_ERROR_BUFFER not in buffers:
        raise ValueError('This network was not built for values with uncertainties, so it has no model error')
    return buffers[MODEL_ERROR_BUFFER]


def set_model_error(network: nn.Module, model_error: float) -> None:
    """Store a network's model error (see fit_model_error())

    Parameters
    ----------
    network : torch.nn.Module
        A network built for a task with uncertainties
    model_error : float
        The model error s, or NaN for one not yet fitted

    Raises
    ------
    ValueError
        If the network has no model error, or `model_error` is negative
    """
    if model_error < 0:
        raise ValueError(f'The model error cannot be negative, got {model_error}')
    with torch.no_grad():
        model_error_of(network).fill_(model_error)


def combine_measurements(values: np.ndarray, sds: np.ndarray) -> tuple[float, float]:
    """Combine independent measurements of one quantity into one value and SD (inverse-variance weighting)

    The combined value is the mean of the values weighted by 1 / SD^2, and its
    SD is 1 / sqrt(sum of 1 / SD^2), the standard combination of independent
    estimates with Gaussian errors. Measurements with an SD of 0 are exact: if
    there are any, the value is their mean and the SD is 0.

    Parameters
    ----------
    values : numpy array
        The measurements (at least one)
    sds : numpy array
        The SD of each measurement's error

    Returns
    -------
    float
        The combined value
    float
        Its SD

    Raises
    ------
    ValueError
        If there are no measurements, or the shapes differ
    """
    values = np.asarray(values, dtype=np.float64)
    sds = np.asarray(sds, dtype=np.float64)
    if values.shape != sds.shape or values.size == 0:
        raise ValueError(f'Need at least one measurement and one SD for each, got {values.size} and {sds.size}')

    exact = sds == 0
    if exact.any():
        return float(values[exact].mean()), 0.0
    weights = 1 / sds ** 2
    return float(np.sum(weights * values) / np.sum(weights)), float(1 / math.sqrt(np.sum(weights)))
