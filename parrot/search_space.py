"""
The space of hyperparameter configurations searched by parrot-optimize.

A configuration is a value for every searched hyperparameter: the learning rate
(for networks trained by gradient descent) and every hyperparameter the
architecture marks as searchable that the user has not fixed and that has an
effect for the task and encoding. To search the
space, each configuration is also represented as a point: one coordinate in
[0, 1] per number and one choice index per setting with a fixed set of values
(see SearchSpace).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from parrot import train_network
from parrot.architectures import HyperParameter, HyperParameterValue, get_architecture, resolve_hyperparameters
from parrot.tasks import Task


class SearchSpace:
    """The hyperparameters being searched, and conversions between their values and points

    A point has one coordinate per searched hyperparameter, in the order of
    `dimensions`: for a number, its position in the search range scaled to
    [0, 1] (on a log scale if declared; whole numbers each get an equal share of
    the range); for a setting with search choices, the index of the choice.

    Attributes
    ----------
    architecture : str
        Name of the architecture
    fixed : dict
        Architecture hyperparameter values fixed by the user (not searched)
    unused : dict
        Searchable hyperparameters that have no effect for the task and
        encoding (not searched; they keep their defaults), with the reason
    dimensions : list of HyperParameter
        The searched hyperparameters, learning rate first (for networks trained
        by gradient descent)
    trained_by_gradient_descent : bool
        Whether the architecture is trained epoch by epoch
    """

    def __init__(self, architecture: str, fixed: Mapping[str, HyperParameterValue] | None = None,
                 task: Task | None = None, input_size: int | None = None) -> None:
        """
        Parameters
        ----------
        architecture : str
            Name of a registered architecture
        fixed : dict, optional
            Architecture hyperparameters fixed at these values instead of searched
        task : Task, optional
        input_size : int, optional
            The task and the length of the per-residue encoding vectors: if both
            are given, hyperparameters that have no effect for them are not
            searched (see ParrotArchitecture.unused_hyperparameters)
        """
        architecture_class = get_architecture(architecture)
        self.architecture = architecture
        self.fixed = dict(fixed or {})
        self.trained_by_gradient_descent = architecture_class.trained_by_gradient_descent
        self.unused: dict[str, str] = {}
        if task is not None and input_size is not None:
            self.unused = {name: reason for name, reason in
                           architecture_class.unused_hyperparameters(task, input_size).items()
                           if name not in self.fixed}
        searched = [hyperparameter for hyperparameter in architecture_class.hyperparameters
                    if hyperparameter.searchable and hyperparameter.name not in self.fixed
                    and hyperparameter.name not in self.unused]
        learning_rate = [train_network.LEARNING_RATE] if self.trained_by_gradient_descent else []
        self.dimensions: list[HyperParameter] = learning_rate + searched

    def __len__(self) -> int:
        """The number of searched hyperparameters."""
        return len(self.dimensions)

    @staticmethod
    def is_numeric(hyperparameter: HyperParameter) -> bool:
        """True for a hyperparameter searched over a range of numbers (rather than a set of choices)."""
        return hyperparameter.search_range is not None

    # ..... single coordinates
    #
    # Whole numbers each get an equal share of [0, 1]. On a linear scale, the value v of a range low..high
    # covers [v - 0.5, v + 0.5], so the whole range is stretched to [low - 0.5, high + 0.5] (high - low + 1
    # units); on a log scale v covers [v, v + 1), so the log axis runs from low to high + 1. A value's
    # coordinate is the middle of its share (v on a linear scale, v + 0.5 on a log scale), and any
    # coordinate in a value's share converts back to that value.

    @staticmethod
    def _to_unit(hyperparameter: HyperParameter, value: float) -> float:
        """Position of a value in its search range, scaled to [0, 1] (see the comment above).

        Parameters
        ----------
        hyperparameter : HyperParameter
            A numeric hyperparameter with a search range
        value : float

        Returns
        -------
        float
            Between 0 and 1
        """
        assert hyperparameter.search_range is not None
        low, high = hyperparameter.search_range
        whole = hyperparameter.value_type is int
        if hyperparameter.log_scale:
            top = high + 1 if whole else high
            position = math.log(value + 0.5 if whole else value)
            return float(np.clip((position - math.log(low)) / (math.log(top) - math.log(low)), 0, 1))
        if whole:
            return (value - low + 0.5) / (high - low + 1)
        return (value - low) / (high - low)

    @staticmethod
    def _from_unit(hyperparameter: HyperParameter, unit: float) -> int | float:
        """The value at a position in [0, 1] of the search range (the inverse of _to_unit()).

        Parameters
        ----------
        hyperparameter : HyperParameter
            A numeric hyperparameter with a search range
        unit : float
            Position in the range (clipped to [0, 1])

        Returns
        -------
        int or float
            The value, of the hyperparameter's type
        """
        assert hyperparameter.search_range is not None
        low, high = hyperparameter.search_range
        unit = float(np.clip(unit, 0.0, 1.0))
        if hyperparameter.value_type is int:
            if hyperparameter.log_scale:
                value = math.floor(math.exp(math.log(low) + unit * (math.log(high + 1) - math.log(low))))
            else:
                value = round(low - 0.5 + unit * (high - low + 1))
            return int(np.clip(value, math.ceil(low), math.floor(high)))
        if hyperparameter.log_scale:
            return float(math.exp(math.log(low) + unit * (math.log(high) - math.log(low))))
        return float(low + unit * (high - low))

    # ..... whole configurations

    def to_point(self, values: Mapping[str, HyperParameterValue]) -> np.ndarray:
        """The point of a configuration.

        Parameters
        ----------
        values : dict
            Value of every searched hyperparameter, by name

        Returns
        -------
        numpy array
            One coordinate per searched hyperparameter (see the class docstring)
        """
        point = []
        for hyperparameter in self.dimensions:
            value = values[hyperparameter.name]
            if self.is_numeric(hyperparameter):
                point.append(self._to_unit(hyperparameter, float(value)))
            else:
                assert hyperparameter.search_choices is not None
                point.append(float(hyperparameter.search_choices.index(value)))
        return np.array(point, dtype=float)

    def from_point(self, point: Sequence[float] | np.ndarray) -> dict[str, HyperParameterValue]:
        """The configuration at a point (the inverse of to_point()).

        Hyperparameters that have no effect, given the others, are set to their
        defaults (see normalized()).

        Parameters
        ----------
        point : numpy array or sequence of float
            One coordinate per searched hyperparameter

        Returns
        -------
        dict
            Value of every searched hyperparameter, by name
        """
        values: dict[str, HyperParameterValue] = {}
        for hyperparameter, coordinate in zip(self.dimensions, point, strict=True):
            if self.is_numeric(hyperparameter):
                values[hyperparameter.name] = self._from_unit(hyperparameter, coordinate)
            else:
                assert hyperparameter.search_choices is not None
                values[hyperparameter.name] = hyperparameter.search_choices[int(coordinate)]
        return self.normalized(values)

    def normalized(self, values: Mapping[str, HyperParameterValue]) -> dict[str, HyperParameterValue]:
        """A configuration with every hyperparameter that has no effect, given the others, at its default value

        The default is as default_point_values() gives it. Configurations that
        differ only in such hyperparameters are the same network, so they are
        tried only once (see ParrotArchitecture.inactive_hyperparameters).

        Parameters
        ----------
        values : dict
            Value of every searched hyperparameter, by name

        Returns
        -------
        dict
        """
        defaults = self.default_point_values()
        inactive = get_architecture(self.architecture).inactive_hyperparameters({**self.fixed, **values})
        return {name: defaults[name] if name in inactive else value for name, value in values.items()}

    def default_values(self) -> dict[str, HyperParameterValue]:
        """PARROT's default value of every searched hyperparameter (moved into its search range if outside it).

        Returns
        -------
        dict
            Value of every searched hyperparameter, by name
        """
        return self.normalized(self.default_point_values())

    def default_point_values(self) -> dict[str, HyperParameterValue]:
        """PARROT's default value of every searched hyperparameter, moved into its search range if outside it, as it
        would be at a point (see default_values)

        A setting whose default is not one of its search choices takes its first
        search choice instead.

        Returns
        -------
        dict
        """
        values: dict[str, HyperParameterValue] = {}
        for hyperparameter in self.dimensions:
            if self.is_numeric(hyperparameter):
                assert hyperparameter.search_range is not None
                low, high = hyperparameter.search_range
                default = float(np.clip(float(hyperparameter.default), low, high))
                values[hyperparameter.name] = self._from_unit(hyperparameter, self._to_unit(hyperparameter, default))
            else:
                assert hyperparameter.search_choices is not None
                choices = hyperparameter.search_choices
                values[hyperparameter.name] = (hyperparameter.default if hyperparameter.default in choices
                                               else choices[0])
        return values

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        """A point drawn uniformly at random.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        numpy array
        """
        return np.array([rng.random() if self.is_numeric(hyperparameter)
                         else float(rng.integers(len(hyperparameter.search_choices or ())))
                         for hyperparameter in self.dimensions], dtype=float)

    def initial_design(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """n space-filling points: a Latin hypercube for numbers, and every choice used equally often

        In a Latin hypercube (McKay, Beckman & Conover 1979, Technometrics
        21:239) each number's range is cut into n equal slices and each slice
        is used by exactly one point, so the points spread over every range
        (unlike a one-at-a-time sweep or plain random points). Each choice is
        used equally often, as nearly as n allows.

        Parameters
        ----------
        n : int
            Number of points
        rng : numpy.random.Generator

        Returns
        -------
        list of numpy arrays
        """
        if n <= 0:
            return []
        columns = []
        for hyperparameter in self.dimensions:
            if self.is_numeric(hyperparameter):
                # Slice i of n is [i / n, (i + 1) / n): each point gets a different slice, at a random place in it
                columns.append((rng.permutation(n) + rng.random(n)) / n)
            else:
                count = len(hyperparameter.search_choices or ())
                columns.append(rng.permutation(np.arange(n) % count).astype(float))
        return [np.array(row, dtype=float) for row in zip(*columns)] if columns else [np.zeros(0) for _ in range(n)]

    def discrete_values(self) -> list[list[HyperParameterValue]] | None:
        """Every value of each searched hyperparameter, if each takes a finite set of values.

        Returns
        -------
        list of lists, or None
            The possible values of each searched hyperparameter, or None if any
            of them is a continuous number
        """
        options: list[list[HyperParameterValue]] = []
        for hyperparameter in self.dimensions:
            if not self.is_numeric(hyperparameter):
                options.append(list(hyperparameter.search_choices or ()))
            elif hyperparameter.value_type is int:
                assert hyperparameter.search_range is not None
                low, high = hyperparameter.search_range
                options.append(list(range(math.ceil(low), math.floor(high) + 1)))
            else:
                return None
        return options

    def architecture_values(self, values: Mapping[str, HyperParameterValue]) -> dict[str, HyperParameterValue]:
        """Every architecture hyperparameter: searched values, then fixed values, then defaults (no learning rate).

        Parameters
        ----------
        values : dict
            A configuration (the searched hyperparameters)

        Returns
        -------
        dict
            A value for every hyperparameter the architecture declares
        """
        searched = {name: value for name, value in values.items() if name != train_network.LEARNING_RATE.name}
        return resolve_hyperparameters(get_architecture(self.architecture), {**self.fixed, **searched})

    @staticmethod
    def learning_rate(values: Mapping[str, HyperParameterValue]) -> float:
        """The learning rate of a configuration (PARROT's default if it is not searched).

        Parameters
        ----------
        values : dict
            A configuration

        Returns
        -------
        float
        """
        return float(values.get(train_network.LEARNING_RATE.name, train_network.LEARNING_RATE.default))
