"""
Building blocks shared by matrix architectures, mostly for handling padding in two dimensions.

Matrix networks work on channels-first grids, [batch_dim X channels X rows X
columns], with one row per residue and either one column per residue (square
maps) or a fixed number of columns. Sequences are zero-padded at the end, so a
grid has padded rows (and, for square maps, padded columns), and, as in every
PARROT network, a sequence's output must not depend on how much padding it was
given. The masks here mark the real cells; re-applying them after any layer
that could write into padding keeps the padding at zero.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import torch
import torch.nn as nn

from parrot.architectures.layers import padding_mask


# ..........................................................................................
# Masks

def cell_mask(row_lengths: torch.Tensor, padded_rows: int, column_lengths: torch.Tensor,
              padded_columns: int) -> torch.Tensor:
    """Boolean mask of the real cells of a batch of grids: those in a real row and a real column

    Parameters
    ----------
    row_lengths : torch.Tensor
        Number of real rows of each grid
    padded_rows : int
        Number of rows every grid was padded to
    column_lengths : torch.Tensor
        Number of real columns of each grid
    padded_columns : int
        Number of columns every grid was padded to

    Returns
    -------
    torch.Tensor
        [batch_dim X padded_rows X padded_columns], True at real cells
    """
    rows = padding_mask(row_lengths, padded_rows)
    columns = padding_mask(column_lengths, padded_columns)
    return rows.unsqueeze(2) & columns.unsqueeze(1)


def grid_channel_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn a boolean [batch_dim X rows X columns] cell mask into a float [batch_dim X 1 X rows X columns] one

    Multiplying a channels-first grid by the result zeroes its padded cells.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean cell mask (see cell_mask())
    dtype : torch.dtype
        Floating-point type of the grids the mask will multiply

    Returns
    -------
    torch.Tensor
    """
    return mask.unsqueeze(1).to(dtype)


# ..........................................................................................
# Features and outputs

def separation_feature(padded_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """How far apart residues i and j are along the sequence, as log(1 + abs(i - j)), for every cell of a square map

    The log keeps the feature on the same scale as the others however long the
    sequence, while still telling near neighbours apart.

    Parameters
    ----------
    padded_length : int
    device : torch.device
    dtype : torch.dtype

    Returns
    -------
    torch.Tensor
        [padded_length X padded_length]
    """
    positions = torch.arange(padded_length, device=device, dtype=dtype)
    return torch.log1p(torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0)))


def symmetrise(outputs: torch.Tensor) -> torch.Tensor:
    """Make square maps symmetric: the average of each map and its transpose

    Parameters
    ----------
    outputs : torch.Tensor
        [batch_dim X length X length X num_classes]

    Returns
    -------
    torch.Tensor
        Same shape, with output[i, j] equal to output[j, i]
    """
    return (outputs + outputs.transpose(1, 2)) / 2


class ChannelLayerNorm2d(nn.Module):
    """Layer normalisation over the channels of a channels-first [batch X channels X rows X columns] grid

    Each cell is normalised on its own, so (unlike batch normalisation, which
    Nick's original 2D U-Net used) padded cells and other sequences in the
    batch cannot affect the result.

    Attributes
    ----------
    norm : torch.nn.LayerNorm
    """

    def __init__(self, channels: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of channels
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X channels X rows X columns]

        Returns
        -------
        torch.Tensor
            Same shape, each cell normalised over its channels
        """
        # nn.LayerNorm normalises the last dimension, so move the channels there and back
        normalised: torch.Tensor = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return normalised
