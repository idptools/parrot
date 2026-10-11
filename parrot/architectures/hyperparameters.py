"""
Declaring the hyperparameters of a network architecture.

Each architecture lists its hyperparameters as HyperParameter declarations.
Everything else about a hyperparameter follows from its declaration: its
command-line flag and help (parrot-train), whether and how parrot-optimize
searches it, the checks applied to its value, and the value used for networks
saved before it existed.

Hyperparameters that several architectures have in common are declared once,
at the end of this module, so the same flag means the same thing for each of
them.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any

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
    default : int, float, bool or str
        Value used when none is provided. Must itself be a valid value.
    help : str
        One-line description shown in command-line help.
    cli_aliases : tuple of str, optional
        Additional command-line flags (e.g. ('-hs',)).
    minimum : int or float, optional
        Smallest allowed value (inclusive, unless `exclusive_minimum`). None
        means no lower bound.
    maximum : int or float, optional
        Largest allowed value (inclusive, unless `exclusive_maximum`). None
        means no upper bound.
    exclusive_minimum : bool, optional
        If True the value must be strictly greater than `minimum` (e.g. a
        penalty strength that must be positive: minimum=0, exclusive_minimum=True).
    exclusive_maximum : bool, optional
        If True the value must be strictly less than `maximum` (e.g. a dropout
        probability, which at 1 would drop everything).
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
    value_type: type[bool] | type[int] | type[float] | type[str]
    default: HyperParameterValue
    help: str
    cli_aliases: tuple[str, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: tuple[HyperParameterValue, ...] | None = None
    search_range: tuple[int | float, int | float] | None = None
    log_scale: bool = False
    legacy_value: HyperParameterValue | None = None
    search_choices: tuple[HyperParameterValue, ...] | None = None

    def __post_init__(self) -> None:
        """Check the declaration itself is consistent, so a mistake in it fails as soon as it is imported.

        Raises
        ------
        ValueError
            If the name is not a Python identifier, the type is not supported,
            options are given that do not apply to the type, the search range is
            empty, or the default, legacy value or a search choice is invalid.
        """
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

        if self.exclusive_minimum and self.minimum is None:
            raise ValueError(f"Hyperparameter '{self.name}' has an exclusive minimum but no minimum")
        if self.exclusive_maximum and self.maximum is None:
            raise ValueError(f"Hyperparameter '{self.name}' has an exclusive maximum but no maximum")

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
            the allowed range, or is not one of `choices`.
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
            converted = self.validate_number(value)

        if self.choices is not None and converted not in self.choices:
            raise ValueError(f"Hyperparameter '{self.name}' must be one of {self.choices}, got {value!r}")

        return converted

    def validate_number(self, value: Any) -> int | float:
        """Convert a value to this (numeric) hyperparameter's type and check it is in range.

        Parameters
        ----------
        value : int or float
            The value to check

        Returns
        -------
        int or float
            The value as `value_type`

        Raises
        ------
        ValueError
            If the value is not a finite number (True and False are not numbers
            here), cannot be represented exactly as `value_type`, or is outside
            the allowed range
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"Hyperparameter '{self.name}' must be a finite number, got {value!r}")
        # NumPy's numbers are converted too; a float that is a whole number is accepted for an int hyperparameter
        number: int | float
        if self.value_type is int:
            number = int(value) if isinstance(value, numbers.Integral) else int(float(value))
        else:
            number = float(value)
        if number != value:
            raise ValueError(f"Hyperparameter '{self.name}' must be of type {self.value_type.__name__}, got {value}")

        if self.minimum is not None:
            if self.exclusive_minimum and number <= self.minimum:
                raise ValueError(f"Hyperparameter '{self.name}' must be greater than {self.minimum}, got {value}")
            if number < self.minimum:
                raise ValueError(f"Hyperparameter '{self.name}' must be at least {self.minimum}, got {value}")

        if self.maximum is not None:
            if self.exclusive_maximum and number >= self.maximum:
                raise ValueError(f"Hyperparameter '{self.name}' must be less than {self.maximum}, got {value}")
            if number > self.maximum:
                raise ValueError(f"Hyperparameter '{self.name}' must be at most {self.maximum}, got {value}")

        return number


# ..........................................................................................
# Hyperparameters several architectures have in common, declared once so their flags, defaults and help agree.
# (Each architecture has its own parrot-train subcommand, so an architecture that needs a different default can
# declare its own.)

DROPOUT = HyperParameter('dropout', float, default=0.0, help='dropout probability (at least 0, below 1)', minimum=0.0,
                         maximum=1.0, exclusive_maximum=True, search_range=(0.0, 0.5))

# Allowed convolution widths. Widths must be odd so convolutions are centred on each residue.
KERNEL_SIZES = (3, 5, 7, 9, 11)

# Ways of turning per-residue features into one vector per sequence when attention pooling is off
POOLING_METHODS = ('mean', 'max')

POOLING = HyperParameter(
    'pooling', str, default='mean', choices=POOLING_METHODS,
    help="sequence tasks with attention pooling off: how residue features are pooled, 'mean' or 'max'")

# On/off options that are on by default. Networks saved before each existed did not have it, so they are
# rebuilt without it (legacy_value=False).
GLOBAL_CONTEXT = HyperParameter(
    'global_context', bool, default=True, legacy_value=False,
    help='add a summary of the whole sequence to every position, so every residue sees the whole sequence')

ATTENTION_POOLING = HyperParameter(
    'attention_pooling', bool, default=True, legacy_value=False,
    help='sequence tasks only: pool residue features with learned attention weights (when off, --pooling is used)')

LOGLENGTH_FEEDFORWARD = HyperParameter(
    'loglength_feedforward', bool, default=True, legacy_value=False,
    help='sequence tasks only: also give the output layer log(sequence length)')
