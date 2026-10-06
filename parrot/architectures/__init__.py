"""
Network architectures available to PARROT.

Each architecture lives in its own module in this package and registers itself
with `register_architecture`. Importing this package imports every built-in
architecture, so they are all available by name to the command-line tools.

To add a new architecture, see docs/adding_architectures.rst.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from parrot.architectures.base import ParrotArchitecture
from parrot.architectures.hyperparameters import HyperParameter, HyperParameterValue
from parrot.architectures.registry import (available_architectures, build_network, get_architecture,
                                           register_architecture, resolve_hyperparameters)

# Built-in architectures. Importing each module registers it.
from parrot.architectures import brnn, classical, cnn, composition, transformer, unet  # noqa: E402,F401

# Architectures for matrix data live with the rest of the matrix code
import parrot.matrices.architectures  # noqa: E402,F401

# The architecture of commands that do not choose one (written before parrot-train had a subcommand per architecture)
DEFAULT_ARCHITECTURE = 'brnn'

__all__ = ['HyperParameter', 'HyperParameterValue', 'ParrotArchitecture', 'available_architectures', 'build_network',
           'get_architecture', 'register_architecture', 'resolve_hyperparameters', 'DEFAULT_ARCHITECTURE']
