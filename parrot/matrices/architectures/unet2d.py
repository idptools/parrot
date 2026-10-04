"""
A 2D U-Net for matrix data, after the 2D U-Net written by Nick (GitHub user
razonickj) on PARROT's parrot-lightning branch (parrot/unet_architecture.py,
added in commit fedab04).

The encoder repeatedly halves the grid in both directions (max pooling) while
doubling the number of channels; the decoder doubles it back, and at each level
concatenates the encoder's features from the same resolution (skip
connections) before two more convolutions. Each output cell can therefore
combine fine local detail with context from a large part of the map.

Differences from Nick's original, made so that the network works on padded
batches of sequences of any length and gives the same output for a sequence
however much padding it was given (as the 1D U-Net in architectures/unet.py):

- the input grid is built from per-residue encodings by MatrixArchitecture
  (see matrices/base.py) rather than read from a one-hot code of residue pairs;
- layer normalisation over channels replaces batch normalisation, whose
  statistics would mix padded cells and other sequences into each cell;
- padded cells are re-zeroed after every layer;
- the number of levels is a hyperparameter (`depth`; Nick's had four), and the
  grid is padded internally to a multiple of 2**depth, so sequences of any
  length can be used;
- the decoder upsamples with a transposed convolution or nearest-neighbour
  repetition (both purely local).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from parrot.architectures.hyperparameters import DROPOUT, KERNEL_SIZES, HyperParameter
from parrot.architectures.layers import resolve_lengths
from parrot.architectures.registry import register_architecture
from parrot.architectures.unet import UPSAMPLING_METHODS, ceil_divide
from parrot.matrices.base import PAIR_CHANNELS, PAIR_FEATURES, MatrixArchitecture
from parrot.matrices.layers import ChannelLayerNorm2d, cell_mask, grid_channel_mask
from parrot.tasks import Task


class DoubleConv2d(nn.Module):
    """Two (convolution -> layer norm -> ReLU) steps followed by dropout, on a channels-first grid

    Attributes
    ----------
    first_conv, second_conv : torch.nn.Conv2d
    first_norm, second_norm : ChannelLayerNorm2d
    dropout : torch.nn.Dropout
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Width of each (square) convolution (odd)
        dropout : float
            Dropout probability applied to the output
        """
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.first_norm = ChannelLayerNorm2d(out_channels)
        self.second_conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.second_norm = ChannelLayerNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X rows X columns]
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X rows X columns] mask of real cells at this resolution

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X rows X columns], zero at padded cells
        """
        x = torch.relu(self.first_norm(self.first_conv(x * channel_mask))) * channel_mask
        x = torch.relu(self.second_norm(self.second_conv(x))) * channel_mask
        output: torch.Tensor = self.dropout(x)
        return output


class Up2d(nn.Module):
    """Double the grid in both directions, concatenate the skip connection, then DoubleConv2d

    Attributes
    ----------
    upsample : torch.nn.Module
        Doubles the grid and maps `in_channels` to `out_channels`
    conv : DoubleConv2d
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float,
                 upsampling: str) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Channels of the lower-resolution input
        out_channels : int
            Channels of the skip connection and of the output
        kernel_size : int
            Width of each convolution (odd)
        dropout : float
            Dropout probability
        upsampling : str
            'transposed' (learned transposed convolution) or 'nearest' (repeat
            each cell, then a 1x1 convolution)
        """
        super().__init__()
        if upsampling == 'transposed':
            self.upsample: nn.Module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling == 'nearest':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            raise ValueError(f"Unknown upsampling method '{upsampling}': must be one of {UPSAMPLING_METHODS}")

        self.conv = DoubleConv2d(2 * out_channels, out_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X rows / 2 X columns / 2]
        skip : torch.Tensor
            Encoder features [batch_dim X out_channels X rows X columns]
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X rows X columns] mask of real cells

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X rows X columns]
        """
        output: torch.Tensor = self.conv(torch.cat([skip, self.upsample(x)], dim=1), channel_mask)
        return output


@register_architecture
class UNet2D(MatrixArchitecture):
    """2D U-Net for matrix data (see the module docstring)

    The parameter names (`residue_projection`, `column_embedding`, `input_block`,
    `down_blocks`, `up_blocks`, `fc` and those inside them) are stored in saved
    networks, so they must not change.

    Attributes
    ----------
    depth : int
        Number of times the grid is halved
    input_block : DoubleConv2d
    down_blocks : torch.nn.ModuleList of DoubleConv2d
    up_blocks : torch.nn.ModuleList of Up2d
    residue_projection, column_embedding, fc
        The input grid and readout (see MatrixArchitecture)
    """

    name = 'unet2d'
    description = '2D U-Net for matrix data (after the 2D U-Net written by Nick, GitHub user razonickj)'
    hyperparameters = (
        PAIR_CHANNELS,
        PAIR_FEATURES,
        HyperParameter('base_channels', int, default=16, minimum=1, search_range=(8, 64),
                       help='channels at full resolution (doubled at each level down)'),
        HyperParameter('depth', int, default=3, minimum=1, search_range=(1, 5),
                       help='number of times the sequence length (or the grid) is halved'),
        HyperParameter('kernel_size', int, default=3, help='width of each convolution', choices=KERNEL_SIZES,
                       search_choices=KERNEL_SIZES),
        DROPOUT,
        HyperParameter('upsampling', str, default='transposed', choices=UPSAMPLING_METHODS,
                       search_choices=UPSAMPLING_METHODS, help='how the decoder doubles the sequence length (or grid)'),
    )

    def __init__(self, input_size: int, task: Task, pair_channels: int, pair_features: str, base_channels: int,
                 depth: int, kernel_size: int, dropout: float, upsampling: str) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            A matrix task
        pair_channels : int
            Features each residue is projected to for the input grid
        pair_features : str
            'concatenate' or 'outer' (square maps)
        base_channels : int
            Channels at full resolution; doubled at each level down
        depth : int
            Number of times the grid is halved
        kernel_size : int
            Width of each convolution (odd)
        dropout : float
            Dropout probability
        upsampling : str
            'transposed' or 'nearest'
        """
        super().__init__(input_size, task, {'pair_channels': pair_channels, 'pair_features': pair_features,
                                            'base_channels': base_channels, 'depth': depth,
                                            'kernel_size': kernel_size, 'dropout': dropout,
                                            'upsampling': upsampling})
        self.depth = depth
        self.build_input_grid(pair_channels, pair_features)

        # Channels at each level: level 0 is full resolution, level `depth` the bottleneck
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.input_block = DoubleConv2d(self.grid_channels, channels[0], kernel_size, dropout)
        self.down_blocks = nn.ModuleList([DoubleConv2d(channels[level - 1], channels[level], kernel_size, dropout)
                                          for level in range(1, depth + 1)])
        self.up_blocks = nn.ModuleList([Up2d(channels[level], channels[level - 1], kernel_size, dropout, upsampling)
                                        for level in range(depth, 0, -1)])
        self.build_matrix_readout(channels[0])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Predict a matrix for each sequence of a batch

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch in the format [batch_dim X padded_length X input_size]
        lengths : torch.Tensor, optional
            The true length of each sequence (None if the batch is unpadded)

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X width X num_classes]
        """
        lengths = resolve_lengths(x, lengths)
        grid, _ = self.input_grid(x, lengths)
        rows, columns = grid.shape[2], grid.shape[3]

        # Pad so the grid can be halved `depth` times in both directions without remainder
        multiple = 2 ** self.depth
        internal_rows = int(ceil_divide(rows, multiple)) * multiple
        internal_columns = int(ceil_divide(columns, multiple)) * multiple
        grid = F.pad(grid, (0, internal_columns - columns, 0, internal_rows - rows))
        masks = self.level_masks(lengths, internal_rows, internal_columns, grid.dtype)

        level_features = self.encode(grid, masks)
        hidden = self.decode(level_features[-1], level_features, masks)
        return self.matrix_readout(hidden[:, :, :rows, :columns])

    def level_masks(self, lengths: torch.Tensor, internal_rows: int, internal_columns: int,
                    dtype: torch.dtype) -> list[torch.Tensor]:
        """The channel mask of real cells at every level, from full resolution (level 0) to the bottleneck

        A cell at level k covers 2**k x 2**k cells at full resolution, and is
        real if any of them is.

        Parameters
        ----------
        lengths : torch.Tensor
            True length of each sequence
        internal_rows, internal_columns : int
            The padded grid size at full resolution (multiples of 2**depth)
        dtype : torch.dtype
            Floating-point type of the features

        Returns
        -------
        list of torch.Tensor
            depth + 1 float masks; the one at level k is
            [batch_dim X 1 X internal_rows / 2**k X internal_columns / 2**k]
        """
        real_columns = lengths if self.task.is_square else torch.full_like(lengths, self.task.width or 0)
        masks = []
        for level in range(self.depth + 1):
            scale = 2 ** level
            row_lengths = ceil_divide(lengths, scale)
            column_lengths = ceil_divide(real_columns, scale)
            assert isinstance(row_lengths, torch.Tensor) and isinstance(column_lengths, torch.Tensor)
            mask = cell_mask(row_lengths, internal_rows // scale, column_lengths, internal_columns // scale)
            masks.append(grid_channel_mask(mask, dtype))
        return masks

    def encode(self, grid: torch.Tensor, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """The encoder: features at full resolution, then at each level down (halving the grid each time)

        Parameters
        ----------
        grid : torch.Tensor
            Padded input grid [batch_dim X grid_channels X internal_rows X internal_columns]
        masks : list of torch.Tensor
            Channel masks at every level (see level_masks())

        Returns
        -------
        list of torch.Tensor
            The features at every level, full resolution first and the bottleneck
            last; all but the last become the decoder's skip connections
        """
        hidden = self.input_block(grid, masks[0])
        level_features = [hidden]
        for level, block in enumerate(self.down_blocks, start=1):
            hidden = block(F.max_pool2d(hidden, 2), masks[level])
            level_features.append(hidden)
        return level_features

    def decode(self, bottleneck: torch.Tensor, level_features: list[torch.Tensor],
               masks: list[torch.Tensor]) -> torch.Tensor:
        """The decoder: from the bottleneck back up to full resolution, joining the encoder's features at each level

        Parameters
        ----------
        bottleneck : torch.Tensor
            Features at the deepest level
        level_features : list of torch.Tensor
            The encoder's features at every level (see encode())
        masks : list of torch.Tensor
            Channel masks at every level

        Returns
        -------
        torch.Tensor
            Full-resolution features [batch_dim X base_channels X internal_rows X internal_columns]
        """
        hidden = bottleneck
        for level, block in zip(range(self.depth - 1, -1, -1), self.up_blocks):
            hidden = block(hidden, level_features[level], masks[level])
        return hidden
