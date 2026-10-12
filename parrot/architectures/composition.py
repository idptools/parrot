"""
An order-independent baseline: predictions from residue composition alone.

For sequence-mapped tasks, the encoded residues are averaged over the whole
sequence (for one-hot encoding this is exactly the amino-acid composition) and,
together with the log of the sequence length, passed through a small
fully connected network. For residue-mapped tasks, each residue is described by
the composition of a window centred on it.

This model cannot use the order of residues (within the window, for residue
tasks). Comparing other architectures against it shows how much of their
performance actually comes from sequence order rather than composition and
length.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import torch
import torch.nn.functional as F

from parrot.architectures.base import ParrotArchitecture
from parrot.architectures.hyperparameters import DROPOUT, HyperParameter
from parrot.architectures.layers import (composition_and_log_length, multilayer_perceptron, padding_mask,
                                         resolve_lengths, zero_padding)
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task


@register_architecture
class Composition(ParrotArchitecture):
    """Composition (plus log length, for sequence tasks) fed to a multilayer perceptron

    The parameter names (`mlp.N`) are stored in saved networks, so they must not change.

    Attributes
    ----------
    window_radius : int
        For residue tasks, the number of residues either side of each residue
        included in its composition window
    mlp : torch.nn.Sequential
        The fully connected network applied to the composition features
    """

    name = 'composition'
    description = 'order-independent baseline: composition (and log length) fed to a multilayer perceptron'
    hyperparameters = (
        HyperParameter('hidden_size', int, default=32, help='width of each hidden layer',
                       cli_aliases=('-hs',), minimum=1, search_range=(8, 128)),
        HyperParameter('num_hidden_layers', int, default=1, help='number of hidden layers (0 gives a linear model)',
                       minimum=0, search_range=(0, 3)),
        DROPOUT,
        HyperParameter('window_radius', int, default=7,
                       help='residue tasks only: residues either side of each residue in its composition window',
                       minimum=0, search_range=(1, 25)),
    )

    def __init__(self, input_size: int, task: Task, hidden_size: int, num_hidden_layers: int, dropout: float,
                 window_radius: int) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        hidden_size : int
            Width of each hidden layer
        num_hidden_layers : int
            Number of hidden layers (0 gives a linear model)
        dropout : float
            Dropout probability after each hidden layer
        window_radius : int
            Residue tasks only: residues either side of each residue in its window
        """
        super().__init__(input_size, task, {'hidden_size': hidden_size, 'num_hidden_layers': num_hidden_layers,
                                            'dropout': dropout, 'window_radius': window_radius})
        self.window_radius = window_radius

        # Sequence tasks get one extra input feature: the log of the sequence length
        num_features = input_size if task.per_residue else input_size + 1
        self.mlp = multilayer_perceptron(num_features, hidden_size, num_hidden_layers, task.num_classes, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Predict from composition (see the module docstring)

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch in the format [batch_dim X padded_length X input_size]
        lengths : torch.Tensor, optional
            The true length of each sequence (None if the batch is unpadded)

        Returns
        -------
        torch.Tensor
            [batch_dim X num_classes] or [batch_dim X padded_length X num_classes]
        """
        lengths = resolve_lengths(x, lengths)
        mask = padding_mask(lengths, x.shape[1])
        x = zero_padding(x, mask)

        if self.task.per_residue:
            output: torch.Tensor = self.mlp(self.window_composition(x, mask))
            return output

        output = self.mlp(composition_and_log_length(x, lengths))
        return output

    @classmethod
    def unused_hyperparameters(cls, task: Task, input_size: int) -> dict[str, str]:
        """The composition window is only used for residue tasks (see ParrotArchitecture.unused_hyperparameters)."""
        if task.per_residue:
            return {}
        return {'window_radius': 'sequence tasks use the whole sequence\'s composition, not a window'}

    def window_composition(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average encoded residues over a window centred on each residue

        Windows are truncated at the ends of each sequence, and the average is
        taken over the real residues in the window only.

        Parameters
        ----------
        x : torch.Tensor
            Masked batch [batch_dim X padded_length X input_size]
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X input_size]
        """
        window = 2 * self.window_radius + 1
        num_features = x.shape[2]

        # Sum every feature over the window with a fixed all-ones convolution
        summing_kernel = torch.ones(num_features, 1, window, dtype=x.dtype, device=x.device)
        window_sums = F.conv1d(x.transpose(1, 2), summing_kernel, padding=self.window_radius, groups=num_features)

        # Count the real residues in each window. Clamping only affects padded
        # positions far from any real residue, whose outputs are never used.
        counting_kernel = torch.ones(1, 1, window, dtype=x.dtype, device=x.device)
        counts = F.conv1d(mask.unsqueeze(1).to(x.dtype), counting_kernel, padding=self.window_radius).clamp(min=1)

        return (window_sums / counts).transpose(1, 2)
