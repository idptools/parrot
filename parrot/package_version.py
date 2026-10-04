"""
The version of PARROT that is running.

The version lives in its own small module, with no other PARROT imports, so any
module can import it (e.g. to record it in a saved network) without importing
the whole package.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import os


def version_from_source_checkout() -> str:
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

    version: str = versioningit.get_version(repository_root)
    return version


try:
    from parrot._version import __version__ as PARROT_VERSION
except ImportError:
    PARROT_VERSION = version_from_source_checkout()
