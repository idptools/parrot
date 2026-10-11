"""
The interface every matrix architecture implements.

A matrix architecture takes the same input as every PARROT network, a
zero-padded batch of encoded sequences [batch_dim X padded_length X input_size]
and their true lengths, and returns one matrix per sequence:
[batch_dim X padded_length X width X num_classes], where width is the padded
length for square maps (Task.width None) or the task's width.

Most of that is the same for every 2D network, so MatrixArchitecture provides
it, leaving each architecture only its own 2D body:

- build_input_grid() / input_grid(): turn the per-residue encoding into a 2D
  grid of features, one cell per output value. Each residue's encoding is first
  projected to `pair_channels` features. For square maps, cell (i, j) holds the
  features of residues i and j, combined side by side ('concatenate') or as
  every product of the two ('outer'), plus their separation along the sequence.
  (Nick's original 2D U-Net used a one-hot code of each pair of amino acids,
  which is the 'outer' combination of two one-hot encodings.) For matrices with
  a fixed number of columns, cell (i, j) holds residue i's features and a
  learned embedding of column j, so the network can tell the columns apart.
- build_matrix_readout() / matrix_readout(): map each cell's features to its
  outputs, and make square maps symmetric for a symmetric task.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from typing import ClassVar

import torch
import torch.nn as nn

from parrot.architectures.base import ParrotArchitecture
from parrot.architectures.hyperparameters import HyperParameter
from parrot.matrices.layers import cell_mask, separation_feature, symmetrise
from parrot.tasks import Task

# Ways of combining the features of two residues into the features of their cell of a square map
PAIR_FEATURE_METHODS = ('concatenate', 'outer')

# Hyperparameters every matrix architecture declares, declared once so their flags and help agree
PAIR_CHANNELS = HyperParameter(
    'pair_channels', int, default=16, minimum=1, search_range=(4, 32),
    help='features each residue is projected to before they are combined into the 2D input grid')
PAIR_FEATURES = HyperParameter(
    'pair_features', str, default='concatenate', choices=PAIR_FEATURE_METHODS, search_choices=PAIR_FEATURE_METHODS,
    help="square maps only: how the features of residues i and j make up cell (i, j): 'concatenate' (side by side) "
         "or 'outer' (every product of the two; pair_channels^2 features, so keep pair_channels small)")


class MatrixArchitecture(ParrotArchitecture):
    """Base class for architectures that predict a matrix per sequence (see the module docstring)

    Subclasses call build_input_grid() and build_matrix_readout() in __init__,
    and in forward() get the input grid from input_grid() and return
    matrix_readout() of their final features.

    Attributes
    ----------
    pair_features : str
        'concatenate' or 'outer' (square maps)
    residue_projection : torch.nn.Linear
        Projects each residue's encoding to pair_channels features
    column_embedding : torch.nn.Embedding or None
        A learned vector for each column (matrices with a fixed number of columns)
    grid_channels : int
        Number of features in each cell of the input grid
    fc : torch.nn.Linear
        Maps each cell's final features to its outputs
    """

    supported_datatypes: ClassVar[tuple[str, ...]] = ('matrix',)

    @classmethod
    def unused_hyperparameters(cls, task: Task, input_size: int) -> dict[str, str]:
        """How pairs of residues are combined only matters for square maps (see
        ParrotArchitecture.unused_hyperparameters)."""
        if task.is_square:
            return {}
        return {'pair_features': 'matrices with a fixed number of columns do not combine pairs of residues'}

    # ..... the input grid

    def build_input_grid(self, pair_channels: int, pair_features: str) -> None:
        """Create the layers that turn per-residue encodings into the 2D input grid

        Parameters
        ----------
        pair_channels : int
            Features each residue is projected to
        pair_features : str
            'concatenate' or 'outer' (square maps only)
        """
        self.pair_features = pair_features
        self.residue_projection = nn.Linear(self.input_size, pair_channels)

        if self.task.is_square:
            self.column_embedding: nn.Embedding | None = None
            combined = 2 * pair_channels if pair_features == 'concatenate' else pair_channels ** 2
            # One more channel for the separation of the two residues
            self.grid_channels = combined + 1
        else:
            assert self.task.width is not None
            self.column_embedding = nn.Embedding(self.task.width, pair_channels)
            self.grid_channels = 2 * pair_channels

    def input_grid(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The 2D input grid of a batch, and the mask of its real cells

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch [batch_dim X padded_length X input_size]
        lengths : torch.Tensor
            True length of each sequence, on the same device as `x`

        Returns
        -------
        torch.Tensor
            Channels-first grid [batch_dim X grid_channels X padded_length X width],
            zero at padded cells
        torch.Tensor
            Boolean [batch_dim X padded_length X width] mask of the real cells
        """
        residues = self.residue_projection(x)
        if self.task.is_square:
            features = self.square_features(residues)
            mask = cell_mask(lengths, x.shape[1], lengths, x.shape[1])
        else:
            features = self.column_features(residues)
            assert self.task.width is not None
            all_columns = torch.full_like(lengths, self.task.width)
            mask = cell_mask(lengths, x.shape[1], all_columns, self.task.width)

        features = features * mask.unsqueeze(-1).to(features.dtype)
        return features.movedim(-1, 1), mask

    def square_features(self, residues: torch.Tensor) -> torch.Tensor:
        """Cell (i, j) of a square map: the features of residues i and j, combined, and their separation

        Parameters
        ----------
        residues : torch.Tensor
            Projected residue features [batch_dim X padded_length X pair_channels]

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X padded_length X grid_channels]
        """
        batch_size, length, channels = residues.shape
        row_residue = residues.unsqueeze(2).expand(batch_size, length, length, channels)
        column_residue = residues.unsqueeze(1).expand(batch_size, length, length, channels)
        if self.pair_features == 'concatenate':
            combined = torch.cat([row_residue, column_residue], dim=-1)
        else:
            # Every product of a row residue's feature and a column residue's feature
            combined = (row_residue.unsqueeze(-1) * column_residue.unsqueeze(-2)).flatten(start_dim=-2)

        separation = separation_feature(length, residues.device, residues.dtype)
        separation = separation.expand(batch_size, length, length).unsqueeze(-1)
        return torch.cat([combined, separation], dim=-1)

    def column_features(self, residues: torch.Tensor) -> torch.Tensor:
        """Cell (i, j) of a matrix with a fixed number of columns: residue i's features and column j's embedding

        Parameters
        ----------
        residues : torch.Tensor
            Projected residue features [batch_dim X padded_length X pair_channels]

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X width X grid_channels]
        """
        assert self.column_embedding is not None
        batch_size, length, channels = residues.shape
        width = self.column_embedding.num_embeddings
        row_residue = residues.unsqueeze(2).expand(batch_size, length, width, channels)
        columns = self.column_embedding.weight.expand(batch_size, length, width, channels)
        return torch.cat([row_residue, columns], dim=-1)

    # ..... the readout

    def build_matrix_readout(self, num_features: int) -> None:
        """Create the layer that maps each cell's final features to its outputs (`fc`)

        Parameters
        ----------
        num_features : int
            Features per cell of the network's final grid
        """
        self.fc = nn.Linear(num_features, self.task.num_classes)

    def matrix_readout(self, features: torch.Tensor) -> torch.Tensor:
        """Each cell's outputs, made symmetric for a symmetric task

        Parameters
        ----------
        features : torch.Tensor
            Channels-first final grid [batch_dim X num_features X padded_length X width]

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X width X num_classes]. Outputs at padded
            cells carry no information.
        """
        outputs: torch.Tensor = self.fc(features.movedim(1, -1))
        if self.task.symmetric:
            outputs = symmetrise(outputs)
        return outputs
