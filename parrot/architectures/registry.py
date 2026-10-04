"""
Registry that maps architecture names to architecture classes.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Mapping
from typing import Any

from parrot.architectures.base import HyperParameterValue, ParrotArchitecture
from parrot.tasks import Task

# Maps each architecture's name to its class. Populated by register_architecture().
_ARCHITECTURES: dict[str, type[ParrotArchitecture]] = {}


def register_architecture(architecture_class: type[ParrotArchitecture]) -> type[ParrotArchitecture]:
    """Class decorator that makes an architecture available to PARROT.

    Parameters
    ----------
    architecture_class : subclass of ParrotArchitecture
        The architecture to register. Its `name` must be unique.

    Returns
    -------
    subclass of ParrotArchitecture
        The same class, unchanged, so this can be used as a decorator.

    Raises
    ------
    ValueError
        If the name is already taken or two hyperparameters share a name.
    """
    name = architecture_class.name
    if name in _ARCHITECTURES:
        raise ValueError(f"An architecture called '{name}' is already registered")

    hyperparameter_names = [hyperparameter.name for hyperparameter in architecture_class.hyperparameters]
    if len(set(hyperparameter_names)) != len(hyperparameter_names):
        raise ValueError(f"Architecture '{name}' declares the same hyperparameter more than once")

    _ARCHITECTURES[name] = architecture_class
    return architecture_class


def available_architectures() -> list[str]:
    """Return the names of every registered architecture, sorted alphabetically."""
    return sorted(_ARCHITECTURES)


def get_architecture(name: str) -> type[ParrotArchitecture]:
    """Look up a registered architecture class by name.

    Parameters
    ----------
    name : str
        The architecture's registry name

    Returns
    -------
    subclass of ParrotArchitecture

    Raises
    ------
    ValueError
        If no architecture with that name is registered
    """
    if name not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Available architectures: {', '.join(available_architectures())}")
    return _ARCHITECTURES[name]


def fill_legacy_values(architecture_class: type[ParrotArchitecture],
                       values: Mapping[str, Any]) -> dict[str, Any]:
    """Complete the saved hyperparameter values of a network with any hyperparameters added since it was saved

    A hyperparameter added to PARROT after a network was saved is missing from
    the values saved with it. If the hyperparameter declares a legacy_value (the
    behaviour networks had before it existed), that value is used, so the network
    is rebuilt exactly as it was trained; other missing hyperparameters are left
    to take their defaults.

    Parameters
    ----------
    architecture_class : subclass of ParrotArchitecture
        The saved network's architecture
    values : dict
        The hyperparameter values saved with the network

    Returns
    -------
    dict
        `values`, plus the legacy value of every missing hyperparameter that has one
    """
    filled = dict(values)
    for hyperparameter in architecture_class.hyperparameters:
        if hyperparameter.name not in filled and hyperparameter.legacy_value is not None:
            filled[hyperparameter.name] = hyperparameter.legacy_value
    return filled


def resolve_hyperparameters(architecture_class: type[ParrotArchitecture],
                            values: Mapping[str, Any] | None = None) -> dict[str, HyperParameterValue]:
    """Fill in defaults for, and validate, an architecture's hyperparameter values.

    Parameters
    ----------
    architecture_class : subclass of ParrotArchitecture
        The architecture whose hyperparameters are being set
    values : dict, optional
        Hyperparameter values to use. Any hyperparameter not given here takes
        its declared default.

    Returns
    -------
    dict
        A value for every declared hyperparameter, converted to its declared type

    Raises
    ------
    ValueError
        If a value is given for a hyperparameter the architecture does not have,
        or a value is invalid.
    """
    values = dict(values or {})
    declared = {hyperparameter.name: hyperparameter for hyperparameter in architecture_class.hyperparameters}

    unknown = sorted(set(values) - set(declared))
    if unknown:
        raise ValueError(f"Architecture '{architecture_class.name}' has no hyperparameter(s) {', '.join(unknown)}. "
                         f"Its hyperparameters are: {', '.join(declared) or 'none'}")

    resolved = {}
    for name, hyperparameter in declared.items():
        resolved[name] = hyperparameter.validate(values.get(name, hyperparameter.default))
    return resolved


def build_network(architecture: str, input_size: int, task: Task,
                  hyperparameter_values: Mapping[str, Any] | None = None) -> ParrotArchitecture:
    """Build a network of a registered architecture.

    Parameters
    ----------
    architecture : str
        Registry name of the architecture (e.g. 'brnn')
    input_size : int
        Length of the per-residue encoding vectors
    task : Task
        The task the network is for
    hyperparameter_values : dict, optional
        Hyperparameter values; anything not given takes its default

    Returns
    -------
    ParrotArchitecture
        A freshly initialised network on the CPU
    """
    architecture_class = get_architecture(architecture)
    values = resolve_hyperparameters(architecture_class, hyperparameter_values)

    # Subclasses take their hyperparameters as keyword arguments (see ParrotArchitecture),
    # which the base-class constructor signature cannot express
    return architecture_class(input_size, task, **values)  # type: ignore[arg-type]
