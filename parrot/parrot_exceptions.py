"""
The errors PARROT raises for input it cannot use.

ParrotException is the parent of PARROT's own errors. It is a ValueError, so
code that catches ValueError (the usual error for a bad value) also catches
PARROT's errors. IOExceptionParrot is raised for a datafile (or other input
file) that PARROT cannot read as the data it was told to expect.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""


class ParrotException(ValueError):
    """Input PARROT cannot use, such as invalid settings or data."""


class IOExceptionParrot(ParrotException):
    """A datafile (or other input file) that is not in the format PARROT expects."""
