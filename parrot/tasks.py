"""
Description of the machine-learning task a PARROT network is trained for.

Every PARROT network maps a protein sequence to a single value
(sequence-mapped data), to one value per residue (residue-mapped data), or to a
matrix of values with one row per residue (matrix data; see parrot.matrices),
and each value is either a class label (classification) or a real number
(regression). A Task bundles these choices so that the rest of the code can
pass around one object instead of separate datatype / problem-type /
number-of-classes arguments.

This module also holds the conventions for missing labels, which every part of
PARROT that reads or scores data shares.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import numbers
from dataclasses import dataclass
from typing import Any

# The supported data formats: one value per sequence, one value per residue, or a matrix of values per
# sequence with one row per residue
VALID_DATATYPES = ('sequence', 'residues', 'matrix')

# Text used in datafiles for a value that is missing (e.g. a residue not observed in a structure). Any
# capitalisation is accepted.
MISSING_LABEL_TEXT = 'nan'

# Class label given to padded positions and to missing labels in residue and matrix classification data. This
# matches the default ignore_index of torch.nn.CrossEntropyLoss, so these positions never contribute to the loss.
IGNORED_CLASS_LABEL = -100


@dataclass(frozen=True)
class Task:
    """The machine-learning task a network is trained for.

    Parameters
    ----------
    datatype : str
        'sequence' if there is one value per sequence, 'residues' if there is
        one value per residue, or 'matrix' if there is a matrix of values per
        sequence, with one row per residue.
    num_classes : int
        Number of classes for a classification task, or 1 for a regression task.
    width : int or None, optional
        'matrix' tasks only: the number of columns of each sequence's matrix.
        None (the default) means one column per residue, i.e. a square map of
        every pair of residues (such as a contact map); an integer N means N
        columns for every sequence (such as N ordered bins or time points).

    Raises
    ------
    ValueError
        If `datatype` is not one of VALID_DATATYPES, `num_classes` is not a
        positive integer, or `width` is given for a task that is not 'matrix'
        or is not a positive integer.
    """

    datatype: str
    num_classes: int
    width: int | None = None

    def __post_init__(self) -> None:
        if self.datatype not in VALID_DATATYPES:
            raise ValueError(f"Invalid datatype '{self.datatype}': must be one of {', '.join(VALID_DATATYPES)}")

        if (isinstance(self.num_classes, bool) or not isinstance(self.num_classes, numbers.Integral)
                or self.num_classes < 1):
            raise ValueError('Number of classes must be a positive integer.')
        # Store a plain int even if given a NumPy integer (e.g. read from an array's shape)
        object.__setattr__(self, 'num_classes', int(self.num_classes))

        if self.width is not None:
            if self.datatype != 'matrix':
                raise ValueError(f"Only 'matrix' tasks have a width, not '{self.datatype}' tasks")
            if isinstance(self.width, bool) or not isinstance(self.width, numbers.Integral) or self.width < 1:
                raise ValueError(f'The width of a matrix task must be a positive integer, got {self.width!r}')
            object.__setattr__(self, 'width', int(self.width))

    @property
    def is_classification(self) -> bool:
        """True for classification tasks (more than one class)."""
        return self.num_classes > 1

    @property
    def problem_type(self) -> str:
        """'classification' or 'regression'."""
        if self.is_classification:
            return 'classification'
        return 'regression'

    @property
    def per_residue(self) -> bool:
        """True if the network predicts a value for every residue."""
        return self.datatype == 'residues'

    @property
    def is_matrix(self) -> bool:
        """True if the network predicts a matrix of values for every sequence (see parrot.matrices)."""
        return self.datatype == 'matrix'

    @property
    def is_square(self) -> bool:
        """True for a matrix task with one column per residue (a map of every pair of residues)."""
        return self.is_matrix and self.width is None

    def to_config(self) -> dict[str, Any]:
        """Return a plain dictionary describing this task (used when saving networks)."""
        config: dict[str, Any] = {'datatype': self.datatype, 'num_classes': self.num_classes}
        if self.is_matrix:
            config['width'] = self.width
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Task':
        """Rebuild a Task from the dictionary written by to_config()."""
        return cls(datatype=config['datatype'], num_classes=int(config['num_classes']), width=config.get('width'))
