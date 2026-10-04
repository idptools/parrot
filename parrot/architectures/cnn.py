"""
A dilated residual 1D convolutional network.

Each residual block normalises its input, applies a 1D convolution and adds the
result back to its input. The spacing between the residues each convolution
looks at (the dilation) doubles from one block to the next (1, 2, 4, 8) and then
repeats, so a few blocks can see a long stretch of sequence while every position
is still computed in parallel. This is the ByteNet design (Kalchbrenner et al.
2016, arXiv:1610.10099), which CARP showed to be competitive with transformers
for protein sequences (Yang et al. 2022, bioRxiv doi:10.1101/2022.05.19.492714).

The convolutions only reach a fixed number of residues (61 with the defaults),
so by default each block also adds a summary of the whole sequence to every
position (see layers.GlobalContext), letting every residue see the whole
sequence however long it is (--global-context-off removes it).

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

# The dilation doubles with each block and resets after this many blocks (1, 2, 4, 8, 1, 2, ...)
DILATION_CYCLE = 4


class ResidualConvBlock(nn.Module):
    """Layer norm -> GELU -> dilated convolution (+ whole-sequence summary) -> dropout, added back to the input

    Attributes
    ----------
    norm : ChannelLayerNorm
    conv : torch.nn.Conv1d
    context : GlobalContext or None
        Adds a summary of the whole sequence to every position, if used
    dropout : torch.nn.Dropout
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, global_context: bool) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels
        kernel_size : int
            Width of the convolution (odd)
        dilation : int
            Spacing between the residues the convolution looks at
        dropout : float
            Dropout probability applied to the convolution output
        global_context : bool
            Also add a summary of the whole sequence to every position
        """
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.context = GlobalContext(channels) if global_context else None

        # This padding keeps the output the same length as the input
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                              padding=dilation * (kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X channels X padded_length], zero at padded positions
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            Same shape as `x`, zero at padded positions
        """
        # Mask before the convolution so padding never leaks into real residues
        activated = F.gelu(self.norm(x)) * channel_mask
        update = self.conv(activated)
        if self.context is not None:
            update = update + self.context(activated, channel_mask)
        output: torch.Tensor = (x + self.dropout(update)) * channel_mask
        return output


@register_architecture
class DilatedCNN(ParrotArchitecture):
    """Dilated residual 1D convolutional network

    The parameter names (`input_projection`, `blocks.N.norm`, `blocks.N.conv`,
    `blocks.N.context`, `output_norm`, `attention_pool`, `fc`) are stored in
    saved networks, so they must not change.

    Attributes
    ----------
    input_projection : torch.nn.Conv1d
        Maps each residue's encoding to `channels` features
    blocks : torch.nn.ModuleList
        The residual convolution blocks
    output_norm : torch.nn.LayerNorm
    attention_pool, fc, pooling, uses_log_length
        The readout (see ParrotArchitecture.build_readout)
    """

    name = 'cnn'
    description = 'dilated residual 1D convolutional network'
    hyperparameters = (
        HyperParameter('channels', int, default=64, help='number of convolutional channels', minimum=1,
                       search_range=(16, 128)),
        HyperParameter('num_blocks', int, default=4, help='number of residual convolution blocks', minimum=1,
                       search_range=(1, 8)),
        HyperParameter('kernel_size', int, default=5, help='width of each convolution', choices=KERNEL_SIZES,
                       search_choices=KERNEL_SIZES),
        DROPOUT,
        POOLING,
        GLOBAL_CONTEXT,
        ATTENTION_POOLING,
        LOGLENGTH_FEEDFORWARD,
    )

    def __init__(self, input_size: int, task: Task, channels: int, num_blocks: int, kernel_size: int,
                 dropout: float, pooling: str, global_context: bool, attention_pooling: bool,
                 loglength_feedforward: bool) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        channels : int
            Number of convolutional channels
        num_blocks : int
            Number of residual convolution blocks
        kernel_size : int
            Width of each convolution (odd)
        dropout : float
            Dropout probability
        pooling : str
            'mean' or 'max'; how residue features are pooled for sequence tasks when attention pooling is off
        global_context : bool
            Add a summary of the whole sequence to every position in every block
        attention_pooling : bool
            Sequence tasks: pool residue features with learned attention weights
        loglength_feedforward : bool
            Sequence tasks: also give the output layer log(sequence length)
        """
        super().__init__(input_size, task, {'channels': channels, 'num_blocks': num_blocks,
                                            'kernel_size': kernel_size, 'dropout': dropout, 'pooling': pooling,
                                            'global_context': global_context, 'attention_pooling': attention_pooling,
                                            'loglength_feedforward': loglength_feedforward})
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, kernel_size, 2 ** (i % DILATION_CYCLE), dropout,
                                                       global_context)
                                     for i in range(num_blocks)])
        self.output_norm = nn.LayerNorm(channels)
        self.build_readout(channels, attention_pooling=attention_pooling, pooling=pooling,
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
        mask = padding_mask(lengths, x.shape[1])
        real_positions = channel_mask(mask, x.dtype)

        # Convolutions work channels-first: [batch_dim X channels X padded_length]
        hidden = self.input_projection(x.transpose(1, 2)) * real_positions
        for block in self.blocks:
            hidden = block(hidden, real_positions)

        features = self.output_norm(hidden.transpose(1, 2))
        return self.readout(features, mask, lengths)
