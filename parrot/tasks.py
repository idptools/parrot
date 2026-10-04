"""
Description of the machine-learning task a PARROT network is trained for.

Every PARROT network maps a protein sequence either to a single value
(sequence-mapped data) or to one value per residue (residue-mapped data), and
that value is either a class label (classification) or a real number
(regression). A Task bundles these two choices so that the rest of the code
can pass around one object instead of separate datatype / problem-type /
number-of-classes arguments.

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

# The two supported data formats: one value per sequence, or one value per residue
VALID_DATATYPES = ('sequence', 'residues')


@dataclass(frozen=True)
class Task:
    """The machine-learning task a network is trained for.

    Parameters
    ----------
    datatype : str
        'sequence' if there is one value per sequence, or 'residues' if there is
        one value per residue.
    num_classes : int
        Number of classes for a classification task, or 1 for a regression task.

    Raises
    ------
    ValueError
        If `datatype` is not 'sequence' or 'residues', or `num_classes` is not a
        positive integer.
    """

    datatype: str
    num_classes: int

    def __post_init__(self) -> None:
        if self.datatype not in VALID_DATATYPES:
            raise ValueError(f"Invalid datatype '{self.datatype}': must be 'residues' or 'sequence'")

        if (isinstance(self.num_classes, bool) or not isinstance(self.num_classes, numbers.Integral)
                or self.num_classes < 1):
            raise ValueError('Number of classes must be a positive integer.')
        # Store a plain int even if given a NumPy integer (e.g. read from an array's shape)
        object.__setattr__(self, 'num_classes', int(self.num_classes))

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

    def to_config(self) -> dict[str, Any]:
        """Return a plain dictionary describing this task (used when saving networks)."""
        return {'datatype': self.datatype, 'num_classes': self.num_classes}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Task':
        """Rebuild a Task from the dictionary written by to_config()."""
        return cls(datatype=config['datatype'], num_classes=int(config['num_classes']))
