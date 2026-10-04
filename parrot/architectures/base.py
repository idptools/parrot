"""
The interface every PARROT network architecture implements.

An architecture is a PyTorch module that takes a zero-padded batch of encoded
sequences (plus the true length of each sequence) and returns either one output
vector per sequence or one output vector per residue, depending on the task.
Each architecture also declares its hyperparameters, which is what lets the
command-line tools build flags for them and the optimiser search over them
without knowing anything else about the architecture.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from parrot.tasks import Task


# The value of a hyperparameter: a number, an on/off switch, or one of a fixed set of named options
HyperParameterValue = bool | int | float | str


@dataclass(frozen=True)
class HyperParameter:
    """Declaration of a single hyperparameter.

    Hyperparameters are numbers (int or float), on/off switches (bool), or text
    with a fixed set of allowed `choices` (e.g. a pooling method).

    A bool hyperparameter's command-line flag switches it away from its default:
    '--<name>-off' for one that is on by default (e.g. '--attention-pooling-off'),
    '--<name>' for one that is off by default.

    Parameters
    ----------
    name : str
        Python identifier for the hyperparameter (e.g. 'hidden_size'). The
        command-line flag is derived from it (e.g. '--hidden-size').
    value_type : type
        int, float, bool or str. A str hyperparameter must declare `choices`.
    default : int, float or str
        Value used when none is provided. Must itself be a valid value.
    help : str
        One-line description shown in command-line help.
    cli_aliases : tuple of str, optional
        Additional command-line flags (e.g. ('-hs',)).
    minimum : int or float, optional
        Smallest allowed value (inclusive). None means no lower bound.
    maximum : int or float, optional
        Largest allowed value (inclusive). None means no upper bound.
    choices : tuple, optional
        The only values allowed. Required for str hyperparameters.
    search_range : tuple of (int or float), optional
        Inclusive (low, high) range searched by hyperparameter optimisation
        (parrot-optimize), for numeric hyperparameters. None means the range is
        not searched.
    log_scale : bool, optional
        If True the optimiser searches this hyperparameter on a log10 scale.
    search_choices : tuple, optional
        The values hyperparameter optimisation tries, for hyperparameters with a
        fixed set of values (`choices`, or True/False). None means they are not
        searched. A hyperparameter has either a search_range or search_choices.
    legacy_value : int, float, str or bool, optional
        The value used for networks saved before this hyperparameter existed, if
        that differs from `default` (e.g. a feature that is now on by default
        but that older networks did not have), so they are rebuilt exactly as
        they were trained. None means such networks use the default.
    """

    name: str
    value_type: type[int] | type[float] | type[str]
    default: HyperParameterValue
    help: str
    cli_aliases: tuple[str, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[HyperParameterValue, ...] | None = None
    search_range: tuple[int | float, int | float] | None = None
    log_scale: bool = False
    legacy_value: HyperParameterValue | None = None
    search_choices: tuple[HyperParameterValue, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Hyperparameter name '{self.name}' must be a valid Python identifier")

        if self.value_type not in (int, float, str, bool):
            raise ValueError(f"Hyperparameter '{self.name}' must have value_type int, float, str or bool")

        if self.value_type is bool and (self.choices is not None or self.minimum is not None
                                        or self.maximum is not None or self.search_range is not None
                                        or self.log_scale):
            raise ValueError(f"On/off hyperparameter '{self.name}' cannot have choices, bounds or a search range")

        if self.value_type is str and self.choices is None:
            raise ValueError(f"Text hyperparameter '{self.name}' must declare its choices")

        if self.search_range is not None:
            if self.value_type is str:
                raise ValueError(f"Hyperparameter '{self.name}' is not numeric, so it cannot have a search range")
            if self.search_range[0] >= self.search_range[1]:
                raise ValueError(f"Hyperparameter '{self.name}' has an empty search range {self.search_range}")

        if self.search_choices is not None:
            if self.search_range is not None:
                raise ValueError(f"Hyperparameter '{self.name}' cannot have both a search range and search choices")
            if len(set(self.search_choices)) < 2:
                raise ValueError(f"Hyperparameter '{self.name}' needs at least two search choices")
            for choice in self.search_choices:
                self.validate(choice)

        # Catch mistakes in the declaration itself as early as possible
        self.validate(self.default)
        if self.legacy_value is not None:
            self.validate(self.legacy_value)

    @property
    def cli_flag(self) -> str:
        """The long command-line flag for this hyperparameter, e.g. '--hidden-size' or '--attention-pooling-off'."""
        flag = '--' + self.name.replace('_', '-')
        if self.value_type is bool and self.default:
            return flag + '-off'
        return flag

    @property
    def searchable(self) -> bool:
        """True if hyperparameter optimisation should search over this hyperparameter."""
        return self.search_range is not None or self.search_choices is not None

    def validate(self, value: Any) -> HyperParameterValue:
        """Convert a value to this hyperparameter's type and check it is allowed.

        Parameters
        ----------
        value : int, float, str or bool
            The value to check

        Returns
        -------
        int, float, str or bool
            The value converted to `value_type`

        Raises
        ------
        ValueError
            If the value is the wrong type, cannot be represented exactly as
            `value_type` (e.g. 2.5 for an integer hyperparameter), is outside
            [minimum, maximum], or is not one of `choices`.
        """
        if self.value_type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Hyperparameter '{self.name}' must be True or False, got {value!r}")
            return value

        if self.value_type is str:
            if not isinstance(value, str):
                raise ValueError(f"Hyperparameter '{self.name}' must be text, got {value!r}")
            converted: HyperParameterValue = value
        else:
            number: int | float = int(value) if self.value_type is int else float(value)
            if number != value:
                raise ValueError(f"Hyperparameter '{self.name}' must be of type {self.value_type.__name__}, got {value}")

            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"Hyperparameter '{self.name}' must be at least {self.minimum}, got {value}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"Hyperparameter '{self.name}' must be at most {self.maximum}, got {value}")
            converted = number

        if self.choices is not None and converted not in self.choices:
            raise ValueError(f"Hyperparameter '{self.name}' must be one of {self.choices}, got {value!r}")

        return converted


class ParrotArchitecture(nn.Module):
    """Base class for every PARROT network architecture.

    To add a new architecture, subclass this class, set the three class
    attributes below, implement __init__ and forward, and decorate the class
    with `parrot.architectures.register_architecture`.

    __init__ must accept `input_size` and `task` followed by one keyword argument
    per declared hyperparameter, and must pass the hyperparameter values up to
    this class so they are saved alongside the network weights.

    forward(x, lengths) receives a zero-padded batch `x` of shape
    [batch_dim X padded_length X input_size] and the true length of each
    sequence (or None if the batch is unpadded). It must return a tensor of
    shape [batch_dim X num_classes] for sequence-mapped tasks or
    [batch_dim X padded_length X num_classes] for residue-mapped tasks, and the
    output for a sequence must not depend on how much padding it was given.
    For classification the outputs are unnormalised logits.

    Attributes
    ----------
    name : str
        Registry key used to select the architecture (e.g. 'brnn').
    description : str
        One-line description shown in command-line help.
    hyperparameters : tuple of HyperParameter
        The architecture's hyperparameters.
    supported_datatypes : tuple of str
        The datatypes ('sequence', 'residues') the architecture can be used for.
        Defaults to both.
    trained_by_gradient_descent : bool
        True (default) for networks trained epoch by epoch with a learning rate.
        False for models fitted in a single step (see architectures/classical.py).
    input_size : int
        Length of the per-residue encoding vectors.
    task : Task
        The task this network was built for.
    hyperparameter_values : dict
        The hyperparameter values this network was built with.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    hyperparameters: ClassVar[tuple[HyperParameter, ...]]
    supported_datatypes: ClassVar[tuple[str, ...]] = ('sequence', 'residues')
    trained_by_gradient_descent: ClassVar[bool] = True

    def __init__(self, input_size: int, task: Task, hyperparameter_values: Mapping[str, HyperParameterValue]) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        hyperparameter_values : dict
            The value of every declared hyperparameter

        Raises
        ------
        ValueError
            If the architecture does not support the task's datatype
        """
        if task.datatype not in self.supported_datatypes:
            raise ValueError(f"The '{self.name}' architecture only supports "
                             f"{' and '.join(self.supported_datatypes)}-mapped data, not '{task.datatype}' data")

        super().__init__()
        self.input_size = input_size
        self.task = task
        self.hyperparameter_values = dict(hyperparameter_values)

    @property
    def num_classes(self) -> int:
        """Number of output values per sequence (or per residue)."""
        return self.task.num_classes

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Map a batch of encoded sequences to outputs (see the class docstring)."""
        raise NotImplementedError
