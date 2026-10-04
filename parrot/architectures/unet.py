"""
A 1D U-Net, adapted from the 2D U-Net written by Nick (GitHub user razonickj)
on PARROT's parrot-lightning branch (parrot/unet_architecture.py, added in
commit fedab04).

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

from parrot.architectures.base import ParrotArchitecture
from parrot.architectures.hyperparameters import (ATTENTION_POOLING, DROPOUT, GLOBAL_CONTEXT, KERNEL_SIZES,
                                                  LOGLENGTH_FEEDFORWARD, POOLING, HyperParameter)
from parrot.architectures.layers import ChannelLayerNorm, GlobalContext, channel_mask, padding_mask, resolve_lengths
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task

# Ways the decoder can double the sequence length
UPSAMPLING_METHODS = ('transposed', 'nearest')


def ceil_divide(numerator: torch.Tensor | int, denominator: int) -> torch.Tensor | int:
    """numerator / denominator rounded up to a whole number, for ints or integer tensors

    Floor division rounds down, so dividing the negated numerator and negating
    the result rounds up: -(-7 // 2) = -(-4) = 4.

    Parameters
    ----------
    numerator : int or torch.Tensor
    denominator : int

    Returns
    -------
    int or torch.Tensor
    """
    return -(-numerator // denominator)


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

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X length]
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X length] mask of real positions at this resolution

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X length], zero at padded positions
        """
        x = torch.relu(self.first_norm(self.first_conv(x * channel_mask))) * channel_mask
        x = torch.relu(self.second_norm(self.second_conv(x))) * channel_mask
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X in_channels X length / 2]
        skip : torch.Tensor
            Encoder features [batch_dim X out_channels X length]
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X length] mask of real positions

        Returns
        -------
        torch.Tensor
            [batch_dim X out_channels X length]
        """
        output: torch.Tensor = self.conv(torch.cat([skip, self.upsample(x)], dim=1), channel_mask)
        return output


@register_architecture
class UNet(ParrotArchitecture):
    """1D U-Net (see the module docstring)

    The parameter names (`input_block`, `down_blocks`, `global_context`,
    `up_blocks`, `attention_pool`, `fc` and those inside them) are stored in
    saved networks, so they must not change.

    Attributes
    ----------
    depth : int
        Number of times the sequence length is halved
    input_block : DoubleConv
    down_blocks : torch.nn.ModuleList of DoubleConv
    global_context : GlobalContext or None
        Adds a summary of the whole sequence to every position at the deepest level, if used
    up_blocks : torch.nn.ModuleList of Up
    attention_pool, fc, pooling, uses_log_length
        The readout (see ParrotArchitecture.build_readout)
    """

    name = 'unet'
    description = '1D U-Net (encoder-decoder convolutional network with skip connections)'
    hyperparameters = (
        HyperParameter('base_channels', int, default=16, minimum=1, search_range=(8, 64),
                       help='channels at full resolution (doubled at each level down)'),
        HyperParameter('depth', int, default=3, minimum=1, search_range=(1, 5),
                       help='number of times the sequence length is halved'),
        HyperParameter('kernel_size', int, default=3, help='width of each convolution', choices=KERNEL_SIZES,
                       search_choices=KERNEL_SIZES),
        DROPOUT,
        HyperParameter('upsampling', str, default='transposed', choices=UPSAMPLING_METHODS,
                       search_choices=UPSAMPLING_METHODS, help='how the decoder doubles the sequence length'),
        POOLING,
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

        # Channels at each level: level 0 is full resolution, level `depth` the bottleneck
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.input_block = DoubleConv(input_size, channels[0], kernel_size, dropout)
        self.down_blocks = nn.ModuleList([DoubleConv(channels[level - 1], channels[level], kernel_size, dropout)
                                          for level in range(1, depth + 1)])
        self.global_context = GlobalContext(channels[depth]) if global_context else None
        self.up_blocks = nn.ModuleList([Up(channels[level], channels[level - 1], kernel_size, dropout, upsampling)
                                        for level in range(depth, 0, -1)])
        self.build_readout(channels[0], attention_pooling=attention_pooling, pooling=pooling,
                           log_length=loglength_feedforward)

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

        # Pad so the length can be halved `depth` times without remainder. Convolutions work
        # channels-first: [batch_dim X channels X internal_length]
        multiple = 2 ** self.depth
        internal_length = int(ceil_divide(original_length, multiple)) * multiple
        hidden = F.pad(x.transpose(1, 2), (0, internal_length - original_length))
        masks = self.level_masks(lengths, internal_length, x.dtype)

        level_features = self.encode(hidden, masks)
        bottleneck = self.add_global_context(level_features[-1], masks[-1])
        hidden = self.decode(bottleneck, level_features, masks)

        features = hidden[:, :, :original_length].transpose(1, 2)
        return self.readout(features, padding_mask(lengths, original_length), lengths)

    def level_masks(self, lengths: torch.Tensor, internal_length: int, dtype: torch.dtype) -> list[torch.Tensor]:
        """The channel mask of real positions at every level, from full resolution (level 0) to the bottleneck

        A position at level k summarises 2**k residues, and is real if any of them is.

        Parameters
        ----------
        lengths : torch.Tensor
            True length of each sequence
        internal_length : int
            The padded length at full resolution (a multiple of 2**depth)
        dtype : torch.dtype
            Floating-point type of the features

        Returns
        -------
        list of torch.Tensor
            depth + 1 float masks; the one at level k is [batch_dim X 1 X internal_length / 2**k]
        """
        masks = []
        for level in range(self.depth + 1):
            level_lengths = ceil_divide(lengths, 2 ** level)
            assert isinstance(level_lengths, torch.Tensor)
            masks.append(channel_mask(padding_mask(level_lengths, internal_length // 2 ** level), dtype))
        return masks

    def encode(self, hidden: torch.Tensor, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """The encoder: features at full resolution, then at each level down (halving the length each time)

        Parameters
        ----------
        hidden : torch.Tensor
            Padded input [batch_dim X input_size X internal_length]
        masks : list of torch.Tensor
            Channel masks at every level (see level_masks())

        Returns
        -------
        list of torch.Tensor
            The features at every level, full resolution first and the bottleneck
            last; all but the last become the decoder's skip connections
        """
        hidden = self.input_block(hidden, masks[0])
        level_features = [hidden]
        for level, block in enumerate(self.down_blocks, start=1):
            hidden = block(F.max_pool1d(hidden, 2), masks[level])
            level_features.append(hidden)
        return level_features

    def add_global_context(self, bottleneck: torch.Tensor, bottleneck_mask: torch.Tensor) -> torch.Tensor:
        """Add a summary of the whole sequence to every bottleneck position (if global context is on)

        Parameters
        ----------
        bottleneck : torch.Tensor
            Features at the deepest level [batch_dim X channels X length / 2**depth]
        bottleneck_mask : torch.Tensor
            Channel mask of real positions at that level

        Returns
        -------
        torch.Tensor
            Same shape as `bottleneck`, zero at padded positions
        """
        if self.global_context is None:
            return bottleneck
        context: torch.Tensor = self.global_context(bottleneck, bottleneck_mask)
        return (bottleneck + context) * bottleneck_mask

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
            Full-resolution features [batch_dim X base_channels X internal_length]
        """
        hidden = bottleneck
        for level, block in zip(range(self.depth - 1, -1, -1), self.up_blocks):
            hidden = block(hidden, level_features[level], masks[level])
        return hidden
