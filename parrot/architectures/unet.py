"""
A 1D U-Net, adapted from Nick's 2D U-Net on the parrot-lightning branch
(parrot/unet_architecture.py).

The encoder repeatedly halves the sequence length (max pooling) while doubling
the number of channels; the decoder doubles the length back, and at each level
concatenates the encoder's features from the same resolution (skip connections)
before two more convolutions. This lets every residue combine fine local detail
with context from a long stretch of sequence. By default a summary of the whole
sequence is also added to every position at the deepest level (see
layers.GlobalContext), so every residue sees the whole sequence however long it
is (--global-context-off removes it).

Differences from the 2D original, all made so that the network fits PARROT's
sequence and residue tasks and gives the same output for a sequence however
much padding it is given:

- convolutions and pooling run along the sequence (1D) instead of over an n x n map;
- layer normalisation over channels replaces batch normalisation, whose
  statistics would mix padded positions and other sequences into each residue;
- padded positions are re-zeroed after every layer;
- the decoder upsamples with a transposed convolution or nearest-neighbour
  repetition (both purely local) rather than bilinear interpolation, which
  behaves differently at the end of a padded and an unpadded sequence;
- channels simply double at each level, controlled by `base_channels` and
  `depth`, rather than through a separate `first_down_channels` setting;
- the input is padded internally to a multiple of 2**depth, so sequences of
  any length can be used.

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

from parrot.architectures.base import HyperParameter, ParrotArchitecture
from parrot.architectures.layers import (ATTENTION_POOLING, GLOBAL_CONTEXT, LOGLENGTH_FEEDFORWARD, POOLING_HELP,
                                         POOLING_METHODS, AttentionPool, ChannelLayerNorm, GlobalContext, padding_mask,
                                         resolve_lengths, sequence_features)
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task

# Allowed convolution widths. Widths must be odd so convolutions are centred on each residue.
KERNEL_SIZES = (3, 5, 7, 9, 11)

# Ways the decoder can double the sequence length
UPSAMPLING_METHODS = ('transposed', 'nearest')


class DoubleConv(nn.Module):
    """Two (convolution -> layer norm -> ReLU) steps followed by dropout

    Attributes
    ----------
    first_conv, second_conv : torch.nn.Conv1d
    first_norm, second_norm : ChannelLayerNorm
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
            Width of each convolution (odd)
        dropout : float
            Dropout probability applied to the output
        """
        super().__init__()
        self.first_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.first_norm = ChannelLayerNorm(out_channels)
        self.second_conv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.second_norm = ChannelLayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X length]
        mask : torch.Tensor
            Float [batch_dim X 1 X length] mask of real positions at this resolution

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X length], zero at padded positions
        """
        x = torch.relu(self.first_norm(self.first_conv(x * mask))) * mask
        x = torch.relu(self.second_norm(self.second_conv(x))) * mask
        output: torch.Tensor = self.dropout(x)
        return output


class Up(nn.Module):
    """Double the sequence length, concatenate the skip connection, then DoubleConv

    Attributes
    ----------
    upsample : torch.nn.Module
        Doubles the length and maps `in_channels` to `out_channels`
    conv : DoubleConv
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
            each position, then a 1x1 convolution)
        """
        super().__init__()
        if upsampling == 'transposed':
            self.upsample: nn.Module = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling == 'nearest':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          nn.Conv1d(in_channels, out_channels, kernel_size=1))
        else:
            raise ValueError(f"Unknown upsampling method '{upsampling}': must be one of {UPSAMPLING_METHODS}")

        self.conv = DoubleConv(2 * out_channels, out_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X length / 2]
        skip : torch.Tensor
            Encoder features [batch_dim X out_channels X length]
        mask : torch.Tensor
            Float [batch_dim X 1 X length] mask of real positions

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X length]
        """
        output: torch.Tensor = self.conv(torch.cat([skip, self.upsample(x)], dim=1), mask)
        return output


@register_architecture
class UNet(ParrotArchitecture):
    """1D U-Net (see the module docstring)

    Attributes
    ----------
    depth : int
        Number of times the sequence length is halved
    pooling : str
        How residue features are pooled for sequence-mapped tasks when attention pooling is off
    loglength_feedforward : bool
        Whether the output layer also sees log(sequence length) (sequence-mapped tasks only)
    input_block : DoubleConv
    down_blocks : torch.nn.ModuleList of DoubleConv
    global_context : GlobalContext or None
        Adds a summary of the whole sequence to every position at the deepest level, if used
    up_blocks : torch.nn.ModuleList of Up
    attention_pool : AttentionPool or None
        Pools residue features for sequence-mapped tasks, if attention pooling is on
    fc : torch.nn.Linear
        Maps the full-resolution features to outputs
    """

    name = 'unet'
    description = "1D U-Net (adapted from Nick's 2D U-Net)"
    hyperparameters = (
        HyperParameter('base_channels', int, default=16, minimum=1, search_range=(8, 64),
                       help='channels at full resolution (doubled at each level down)'),
        HyperParameter('depth', int, default=3, minimum=1, search_range=(1, 5),
                       help='number of times the sequence length is halved'),
        HyperParameter('kernel_size', int, default=3, help='width of each convolution', choices=KERNEL_SIZES,
                       search_choices=KERNEL_SIZES),
        HyperParameter('dropout', float, default=0.0, help='dropout probability', minimum=0.0, maximum=1.0,
                       search_range=(0.0, 0.5)),
        HyperParameter('upsampling', str, default='transposed', choices=UPSAMPLING_METHODS,
                       search_choices=UPSAMPLING_METHODS, help='how the decoder doubles the sequence length'),
        HyperParameter('pooling', str, default='mean', choices=POOLING_METHODS, help=POOLING_HELP),
        GLOBAL_CONTEXT,
        ATTENTION_POOLING,
        LOGLENGTH_FEEDFORWARD,
    )

    def __init__(self, input_size: int, task: Task, base_channels: int, depth: int, kernel_size: int,
                 dropout: float, upsampling: str, pooling: str, global_context: bool, attention_pooling: bool,
                 loglength_feedforward: bool) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        base_channels : int
            Channels at full resolution; doubled at each level down
        depth : int
            Number of times the sequence length is halved
        kernel_size : int
            Width of each convolution (odd)
        dropout : float
            Dropout probability
        upsampling : str
            'transposed' or 'nearest'
        pooling : str
            'mean' or 'max'; how residue features are pooled for sequence tasks when attention pooling is off
        global_context : bool
            Add a summary of the whole sequence to every position at the deepest level
        attention_pooling : bool
            Sequence tasks: pool residue features with learned attention weights
        loglength_feedforward : bool
            Sequence tasks: also give the output layer log(sequence length)
        """
        super().__init__(input_size, task, {'base_channels': base_channels, 'depth': depth,
                                            'kernel_size': kernel_size, 'dropout': dropout,
                                            'upsampling': upsampling, 'pooling': pooling,
                                            'global_context': global_context, 'attention_pooling': attention_pooling,
                                            'loglength_feedforward': loglength_feedforward})
        self.depth = depth
        self.pooling = pooling
        self.loglength_feedforward = loglength_feedforward and not task.per_residue

        # Channels at each level: level 0 is full resolution, level `depth` the bottleneck
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.input_block = DoubleConv(input_size, channels[0], kernel_size, dropout)
        self.down_blocks = nn.ModuleList([DoubleConv(channels[level - 1], channels[level], kernel_size, dropout)
                                          for level in range(1, depth + 1)])
        self.global_context = GlobalContext(channels[depth]) if global_context else None
        self.up_blocks = nn.ModuleList([Up(channels[level], channels[level - 1], kernel_size, dropout, upsampling)
                                        for level in range(depth, 0, -1)])
        self.attention_pool = AttentionPool(channels[0]) if attention_pooling and not task.per_residue else None
        self.fc = nn.Linear(channels[0] + int(self.loglength_feedforward), task.num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Propagate a batch of encoded sequences through the network

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
        original_length = x.shape[1]

        # Pad so the length can be halved `depth` times without remainder
        multiple = 2 ** self.depth
        internal_length = -(-original_length // multiple) * multiple
        hidden = F.pad(x.transpose(1, 2), (0, internal_length - original_length))

        # A position at level k summarises 2**k residues, and is real if any of them is
        masks = []
        for level in range(self.depth + 1):
            level_lengths = -(-lengths // 2 ** level)
            level_mask = padding_mask(level_lengths, internal_length // 2 ** level)
            masks.append(level_mask.unsqueeze(1).to(x.dtype))

        # Encoder: keep the features at every resolution for the skip connections
        hidden = self.input_block(hidden, masks[0])
        skips = [hidden]
        for level, block in enumerate(self.down_blocks, start=1):
            hidden = block(F.max_pool1d(hidden, 2), masks[level])
            skips.append(hidden)

        # At the bottleneck, every position also gets a summary of the whole sequence
        if self.global_context is not None:
            hidden = (hidden + self.global_context(hidden, masks[self.depth])) * masks[self.depth]

        # Decoder: from the bottleneck back up to full resolution
        for level, block in zip(range(self.depth - 1, -1, -1), self.up_blocks):
            hidden = block(hidden, skips[level], masks[level])

        features = hidden[:, :, :original_length].transpose(1, 2)
        if self.task.per_residue:
            per_residue_output: torch.Tensor = self.fc(features)
            return per_residue_output

        mask = padding_mask(lengths, original_length)
        per_sequence_output: torch.Tensor = self.fc(sequence_features(features, mask, lengths, self.attention_pool,
                                                                      self.pooling, self.loglength_feedforward))
        return per_sequence_output
