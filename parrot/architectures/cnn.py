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

from parrot.architectures.base import HyperParameter, ParrotArchitecture
from parrot.architectures.layers import (ATTENTION_POOLING, GLOBAL_CONTEXT, LOGLENGTH_FEEDFORWARD, POOLING_HELP,
                                         POOLING_METHODS, AttentionPool, ChannelLayerNorm, GlobalContext, padding_mask,
                                         resolve_lengths, sequence_features)
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task

# The dilation doubles with each block and resets after this many blocks (1, 2, 4, 8, 1, 2, ...)
DILATION_CYCLE = 4

# Allowed convolution widths. Widths must be odd so convolutions are centred on each residue.
KERNEL_SIZES = (3, 5, 7, 9, 11)


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X channels X padded_length], zero at padded positions
        mask : torch.Tensor
            Float [batch_dim X 1 X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            Same shape as `x`, zero at padded positions
        """
        # Mask before the convolution so padding never leaks into real residues
        activated = F.gelu(self.norm(x)) * mask
        update = self.conv(activated)
        if self.context is not None:
            update = update + self.context(activated, mask)
        output: torch.Tensor = (x + self.dropout(update)) * mask
        return output


@register_architecture
class DilatedCNN(ParrotArchitecture):
    """Dilated residual 1D convolutional network

    Attributes
    ----------
    pooling : str
        How residue features are pooled for sequence-mapped tasks when attention pooling is off
    loglength_feedforward : bool
        Whether the output layer also sees log(sequence length) (sequence-mapped tasks only)
    input_projection : torch.nn.Conv1d
        Maps each residue's encoding to `channels` features
    blocks : torch.nn.ModuleList
        The residual convolution blocks
    output_norm : torch.nn.LayerNorm
    attention_pool : AttentionPool or None
        Pools residue features for sequence-mapped tasks, if attention pooling is on
    fc : torch.nn.Linear
        Maps features to outputs
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
        HyperParameter('dropout', float, default=0.0, help='dropout probability', minimum=0.0, maximum=1.0,
                       search_range=(0.0, 0.5)),
        HyperParameter('pooling', str, default='mean', choices=POOLING_METHODS, help=POOLING_HELP),
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
        self.pooling = pooling
        self.loglength_feedforward = loglength_feedforward and not task.per_residue
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, kernel_size, 2 ** (i % DILATION_CYCLE), dropout,
                                                       global_context)
                                     for i in range(num_blocks)])
        self.output_norm = nn.LayerNorm(channels)
        self.attention_pool = AttentionPool(channels) if attention_pooling and not task.per_residue else None
        self.fc = nn.Linear(channels + int(self.loglength_feedforward), task.num_classes)

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
        channel_mask = mask.unsqueeze(1).to(x.dtype)

        hidden = self.input_projection(x.transpose(1, 2)) * channel_mask
        for block in self.blocks:
            hidden = block(hidden, channel_mask)

        features = self.output_norm(hidden.transpose(1, 2))
        if self.task.per_residue:
            per_residue_output: torch.Tensor = self.fc(features)
            return per_residue_output

        per_sequence_output: torch.Tensor = self.fc(sequence_features(features, mask, lengths, self.attention_pool,
                                                                      self.pooling, self.loglength_feedforward))
        return per_sequence_output
