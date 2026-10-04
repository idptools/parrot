"""
Matrix architectures: networks that predict a matrix per sequence (see matrices/base.py).

Each architecture lives in its own module and registers itself with the same
registry as every other PARROT architecture, declaring 'matrix' as the only
datatype it supports. parrot.architectures imports this package, so they are
available by name everywhere.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

# Importing each module registers it
from parrot.matrices.architectures import unet2d  # noqa: F401
