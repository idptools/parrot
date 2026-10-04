"""
Making PARROT's random choices reproducible.

PARROT draws random numbers from NumPy (data splits, cross-validation folds,
class balancing, the hyperparameter search) and PyTorch (network initialisation
and batch shuffling). set_random_seed() seeds both, so a command run with
--seed gives the same result every time on the same machine.

Code that needs its own random number generator gets one from
random_generator(), which is seeded from NumPy's global generator, so it too
follows --seed.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import random

import numpy as np
import torch

# Exclusive upper bound of the seeds drawn for new random number generators
MAX_GENERATOR_SEED = 2 ** 31


def set_random_seed(seed: int | None) -> None:
    """Seed every random number generator PARROT draws from

    Seeds NumPy, PyTorch (which also seeds every CUDA device and the MPS
    device) and Python's own random module (which PARROT does not use, but
    other code run alongside it may). On NVIDIA GPUs, cuDNN is also told to use
    only deterministic algorithms: some of its default convolution algorithms
    are not, which made seeded U-Net training differ between runs (at no
    measurable cost in speed on an RTX A4000).

    Parameters
    ----------
    seed : int or None
        The seed to use. If None, nothing is seeded and runs are not reproducible.
    """

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_generator() -> np.random.Generator:
    """A new NumPy random number generator, seeded from NumPy's global generator so that --seed controls it

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(np.random.randint(MAX_GENERATOR_SEED))
