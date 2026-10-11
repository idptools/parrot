"""
The interface every PARROT network architecture implements.

An architecture is a PyTorch module that takes a zero-padded batch of encoded
sequences (plus the true length of each sequence) and returns either one output
vector per sequence or one output vector per residue, depending on the task.
Each architecture also declares its hyperparameters (see hyperparameters.py),
which is what lets the command-line tools build flags for them and the
optimiser search over them without knowing anything else about the architecture.

Most architectures end the same way: per-residue features are mapped to the
output by a linear layer, after (for sequence tasks) pooling them into one
vector per sequence. ParrotArchitecture provides that "readout" once, in
build_readout() and readout().

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parrot.architectures.hyperparameters import HyperParameter, HyperParameterValue
from parrot.architectures.layers import AttentionPool, append_log_length, masked_pool
from parrot.tasks import Task


class ParrotArchitecture(nn.Module):
    """Base class for every PARROT network architecture.

    To add a new architecture, subclass this class, set the class attributes
    below, implement __init__ and forward, and decorate the class with
    `parrot.architectures.register_architecture` (see docs/adding_architectures.rst).

    __init__ must accept `input_size` and `task` followed by one keyword argument
    per declared hyperparameter, and must pass every hyperparameter value up to
    this class, which checks them and saves them alongside the network weights.

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
        One-line description, shown in the list of architectures in
        command-line help. `name` is the architecture's subcommand of
        parrot-train and parrot-optimize.
    hyperparameters : tuple of HyperParameter
        The architecture's hyperparameters.
    supported_datatypes : tuple of str
        The datatypes ('sequence', 'residues', 'matrix') the architecture can be
        used for. Defaults to ('sequence', 'residues').
    trained_by_gradient_descent : bool
        True (default) for networks trained epoch by epoch with a learning rate.
        False for models fitted in a single step by fit() (see classical.py).
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
            If the architecture does not support the task's datatype, a declared
            hyperparameter is missing from `hyperparameter_values` (or an
            undeclared one is present), or a value is not allowed
        """
        if task.datatype not in self.supported_datatypes:
            raise ValueError(f"The '{self.name}' architecture only supports "
                             f"{' and '.join(self.supported_datatypes)}-mapped data, not '{task.datatype}' data")

        super().__init__()
        self.input_size = input_size
        self.task = task
        self.hyperparameter_values = self.checked_hyperparameter_values(hyperparameter_values)

    @classmethod
    def checked_hyperparameter_values(cls, values: Mapping[str, HyperParameterValue]) -> dict[str, HyperParameterValue]:
        """Check that `values` gives exactly the declared hyperparameters, each with an allowed value.

        The values are saved with the network and used to rebuild it, so a
        hyperparameter left out here (or misspelt) would silently be rebuilt with
        its default when the network is loaded.

        Parameters
        ----------
        values : dict
            Hyperparameter values, by name

        Returns
        -------
        dict
            The values, each converted to its declared type, in declaration order

        Raises
        ------
        ValueError
            If the names differ from the declared ones or a value is not allowed
        """
        declared = {hyperparameter.name: hyperparameter for hyperparameter in cls.hyperparameters}
        missing = sorted(set(declared) - set(values))
        undeclared = sorted(set(values) - set(declared))
        if missing or undeclared:
            raise ValueError(f"The '{cls.name}' architecture was given the wrong hyperparameters: "
                             f"missing {missing or 'none'}, not declared {undeclared or 'none'}")

        return {name: hyperparameter.validate(values[name]) for name, hyperparameter in declared.items()}

    @classmethod
    def unused_hyperparameters(cls, task: Task, input_size: int) -> dict[str, str]:
        """Hyperparameters that have no effect, or cannot be used, for a task and encoding (parrot-optimize does not
        search them). An architecture overrides this if it has any; by default there are none.

        Parameters
        ----------
        task : Task
        input_size : int
            Length of the per-residue encoding vectors

        Returns
        -------
        dict
            Why each one is not used, by name
        """
        return {}

    @classmethod
    def inactive_hyperparameters(cls, values: Mapping[str, HyperParameterValue]) -> set[str]:
        """Hyperparameters that have no effect given the values of the others (e.g. a setting of one kernel when
        another kernel is chosen). parrot-optimize gives them their default value, so configurations that differ
        only in them count as the same. By default there are none.

        Parameters
        ----------
        values : dict
            Hyperparameter values, by name (not necessarily all of them)

        Returns
        -------
        set of str
        """
        return set()

    @property
    def num_classes(self) -> int:
        """Number of output values per sequence (or per residue)."""
        return self.task.num_classes

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Map a batch of encoded sequences to outputs (see the class docstring)."""
        raise NotImplementedError

    def fit(self, loader: DataLoader[Any]) -> None:
        """Fit a model that is not trained by gradient descent (trained_by_gradient_descent = False) in one step.

        Parameters
        ----------
        loader : DataLoader
            Batches of the training data
        """
        raise NotImplementedError(f"The '{self.name}' architecture is trained by gradient descent, not fitted")

    # ..........................................................................................
    # The readout: from per-residue features to the network's output

    def build_readout(self, num_features: int, attention_pooling: bool = False, pooling: str = 'mean',
                      log_length: bool = False) -> None:
        """Create the layers that map per-residue features to the network's output.

        For residue-mapped tasks, a linear layer `fc` maps each residue's
        features to its outputs. For sequence-mapped tasks, the residues are
        first pooled into one vector per sequence, by learned attention weights
        (`attention_pool`) if `attention_pooling` is on and otherwise by
        `pooling`, and with `log_length` the log of the sequence length is
        appended to that vector before `fc`. Pooling and log(length) only apply
        to sequence tasks, so they are ignored for residue tasks.

        Saved networks store the weights under the names `attention_pool` and
        `fc`, so these names must not change. The attention pool is created
        before `fc` so that a network built from a given random seed always
        starts from the same weights.

        Parameters
        ----------
        num_features : int
            Number of features per residue
        attention_pooling : bool, optional
            Sequence tasks: pool residues with learned attention weights
        pooling : str, optional
            Sequence tasks without attention pooling: 'mean' or 'max'
        log_length : bool, optional
            Sequence tasks: also give `fc` log(sequence length)
        """
        is_sequence_task = not self.task.per_residue
        self.pooling = pooling
        self.uses_log_length = log_length and is_sequence_task
        self.attention_pool = AttentionPool(num_features) if attention_pooling and is_sequence_task else None
        self.fc = nn.Linear(num_features + int(self.uses_log_length), self.task.num_classes)

    def readout(self, features: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Map per-residue features to the network's output, with the layers made by build_readout().

        Parameters
        ----------
        features : torch.Tensor
            Per-residue features [batch_dim X padded_length X num_features]
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues
        lengths : torch.Tensor
            True length of each sequence, on the same device as `features`

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X num_classes] for residue tasks, or
            [batch_dim X num_classes] for sequence tasks
        """
        if self.task.per_residue:
            per_residue_output: torch.Tensor = self.fc(features)
            return per_residue_output

        return self.sequence_output(self.pool_residues(features, mask), lengths)

    def pool_residues(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool per-residue features into one vector per sequence, by attention or by `self.pooling`.

        Parameters
        ----------
        features : torch.Tensor
            Per-residue features [batch_dim X padded_length X num_features]
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            [batch_dim X num_features]
        """
        if self.attention_pool is not None:
            pooled: torch.Tensor = self.attention_pool(features, mask)
            return pooled
        return masked_pool(features, mask, self.pooling)

    def sequence_output(self, summary: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """The output for each sequence: `fc` applied to its summary vector (plus log(length), if used).

        Parameters
        ----------
        summary : torch.Tensor
            One vector per sequence [batch_dim X num_features]
        lengths : torch.Tensor
            True length of each sequence, on the same device as `summary`

        Returns
        -------
        torch.Tensor
            [batch_dim X num_classes]
        """
        if self.uses_log_length:
            summary = append_log_length(summary, lengths)
        output: torch.Tensor = self.fc(summary)
        return output
