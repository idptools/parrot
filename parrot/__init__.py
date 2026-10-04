"""
PARROT
A framework for training neural networks that map protein sequences to values,
either one value per sequence or one value per residue.
"""

import os

# Modules whose contents have always been available directly from `parrot`
from parrot.encode_sequence import *
from parrot.process_input_data import *
from parrot.brnn_architecture import *
from parrot.train_network import *

# The main building blocks for working with any architecture (re-exported for convenience)
from parrot.architectures import (HyperParameter, ParrotArchitecture, available_architectures,  # noqa: F401
                                  build_network, get_architecture, register_architecture)
from parrot.checkpoint import load_network, save_network  # noqa: F401
from parrot.parrot_exceptions import IOExceptionParrot, ParrotException  # noqa: F401
from parrot.tasks import Task  # noqa: F401


def _version_from_source_checkout():
    """Ask versioningit for the version when running from a raw source checkout.

    ``parrot/_version.py`` is written by versioningit when the package is built or
    installed, so it does not exist when PARROT is imported straight from the
    repository (e.g. when running the tests or building the docs). In that case we
    ask versioningit directly, pointing it at the repository root rather than at the
    current working directory.

    Returns
    -------
    str
        The version string, or '0+unknown' if versioningit is not installed.
    """
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        import versioningit
    except ImportError:
        return '0+unknown'

    return versioningit.get_version(repository_root)


try:
    from parrot._version import __version__ as __version__
except ImportError:
    __version__ = _version_from_source_checkout()


# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_directory(path='../data'):
    return os.path.join(_ROOT, path)
