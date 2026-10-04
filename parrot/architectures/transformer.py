"""
A transformer encoder.

Each residue's encoding is projected to the model width (num_heads x
head_size), then passed through a stack of pre-norm transformer blocks
(Vaswani et al. 2017, NeurIPS; pre-norm as in Xiong et al. 2020, ICML): layer
norm -> multi-head self-attention -> residual, then layer norm -> feed-forward
network -> residual. Every residue can attend to every other residue of its
sequence, so each block sees the whole sequence.

Residue order enters only through the positional encoding:

- 'rotary' (default) rotates queries and keys by an angle proportional to the
  residue's position (RoFormer, Su et al. 2024, Neurocomputing 568:127063), so
  attention depends on the distance between residues. ESM-2 uses it.
- 'sinusoidal' adds the fixed sine/cosine position vectors of Vaswani et al.
  2017 to the input, giving each residue its absolute position.
- 'none' uses no positions: the network sees each sequence as a set of residues
  (useful as an order-free comparison).

Both encodings are computed on the fly for any length, so there is no maximum
sequence length and nothing position-related is stored in saved networks.

Padded positions are masked out of attention as keys, so a sequence's output
never depends on how much padding it was given.

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
from parrot.architectures.hyperparameters import (ATTENTION_POOLING, DROPOUT, LOGLENGTH_FEEDFORWARD, POOLING,
                                                  HyperParameter)
from parrot.architectures.layers import padding_mask, resolve_lengths, zero_padding
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task

# Ways of telling the network where each residue is
POSITIONAL_ENCODINGS = ('rotary', 'sinusoidal', 'none')

# Base of the geometric series of wavelengths used by both positional encodings (Vaswani et al. 2017)
POSITION_BASE = 10000.0


def position_frequencies(dimensions: int, device: torch.device) -> torch.Tensor:
    """The angular frequency of each of dimensions // 2 sine/cosine pairs, highest first."""
    return POSITION_BASE ** (-torch.arange(0, dimensions // 2, device=device, dtype=torch.float32)
                             * 2 / dimensions)


def sinusoidal_positions(length: int, dimensions: int, device: torch.device) -> torch.Tensor:
    """[length X dimensions] sine/cosine position vectors (Vaswani et al. 2017); an odd last dimension is zero."""
    angles = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1) * \
        position_frequencies(dimensions, device)
    num_pairs = angles.shape[1]

    # Pair i fills dimensions 2i (sine) and 2i + 1 (cosine)
    positions = torch.zeros(length, dimensions, device=device)
    positions[:, 0:2 * num_pairs:2] = torch.sin(angles)
    positions[:, 1:2 * num_pairs:2] = torch.cos(angles)
    return positions


def apply_rotary(x: torch.Tensor) -> torch.Tensor:
    """Rotate queries or keys by their positions (rotary position embedding)

    Each pair of features (i, i + d/2) of a vector at position p is rotated by
    the angle p x frequency_i, so the dot product of a query and a key depends
    on how far apart they are rather than where they are. With an odd number of
    features the last one is left unrotated.

    Parameters
    ----------
    x : torch.Tensor
        [batch_dim X num_heads X padded_length X head_size]

    Returns
    -------
    torch.Tensor
        Same shape as `x`
    """
    head_size = x.shape[-1]
    half = head_size // 2
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32).unsqueeze(1) * \
        position_frequencies(2 * half, x.device)
    cos, sin = torch.cos(angles).to(x.dtype), torch.sin(angles).to(x.dtype)

    first, second, rest = x[..., :half], x[..., half:2 * half], x[..., 2 * half:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin, rest], dim=-1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the real residues of each sequence

    Attributes
    ----------
    num_heads : int
    rotary : bool
        Whether queries and keys are rotated by their positions
    dropout_probability : float
        Dropout probability applied to the attention weights
    qkv : torch.nn.Linear
        Computes the queries, keys and values of every head at once
    output : torch.nn.Linear
        Combines the heads
    """

    def __init__(self, num_heads: int, head_size: int, rotary: bool, dropout: float) -> None:
        """
        Parameters
        ----------
        num_heads : int
            Number of attention heads
        head_size : int
            Width of each head's queries, keys and values
        rotary : bool
            Rotate queries and keys by their positions (rotary position embedding)
        dropout : float
            Dropout probability applied to the attention weights
        """
        super().__init__()
        self.num_heads = num_heads
        self.rotary = rotary
        self.dropout_probability = dropout
        model_size = num_heads * head_size
        self.qkv = nn.Linear(model_size, 3 * model_size)
        self.output = nn.Linear(model_size, model_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X padded_length X model_size]
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            [batch_dim X padded_length X model_size]
        """
        batch_size, length, model_size = x.shape

        # [3 X batch_dim X num_heads X padded_length X head_size]
        qkv = self.qkv(x).view(batch_size, length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        if self.rotary:
            queries, keys = apply_rotary(queries), apply_rotary(keys)

        # Every position attends only to the real residues of its own sequence
        key_mask = mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=key_mask,
                                                  dropout_p=self.dropout_probability if self.training else 0.0)
        output: torch.Tensor = self.output(attended.transpose(1, 2).reshape(batch_size, length, model_size))
        return output


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then x + feed_forward(norm(x))

    Attributes
    ----------
    attention_norm, feed_forward_norm : torch.nn.LayerNorm
    attention : SelfAttention
    feed_forward : torch.nn.Sequential
    dropout : torch.nn.Dropout
        Applied to each sublayer's output before it is added back
    """

    def __init__(self, num_heads: int, head_size: int, feedforward_multiplier: int, rotary: bool,
                 dropout: float) -> None:
        """
        Parameters
        ----------
        num_heads, head_size, rotary, dropout
            As for SelfAttention
        feedforward_multiplier : int
            Width of the feed-forward network's hidden layer, as a multiple of the model width
        """
        super().__init__()
        model_size = num_heads * head_size
        self.attention_norm = nn.LayerNorm(model_size)
        self.attention = SelfAttention(num_heads, head_size, rotary, dropout)
        self.feed_forward_norm = nn.LayerNorm(model_size)
        self.feed_forward = nn.Sequential(nn.Linear(model_size, feedforward_multiplier * model_size), nn.GELU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(feedforward_multiplier * model_size, model_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X padded_length X model_size], zero at padded positions
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            Same shape as `x`, zero at padded positions
        """
        x = x + self.dropout(self.attention(self.attention_norm(x), mask))
        x = x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))
        return zero_padding(x, mask)


@register_architecture
class Transformer(ParrotArchitecture):
    """Transformer encoder

    The parameter names (`input_projection`, `blocks.N.attention.qkv` and the
    others inside each block, `output_norm`, `attention_pool`, `fc`) are stored
    in saved networks, so they must not change.

    Attributes
    ----------
    positional_encoding : str
        'rotary', 'sinusoidal' or 'none'
    model_size : int
        Width of every residue's representation (num_heads x head_size)
    input_projection : torch.nn.Linear
        Maps each residue's encoding to `model_size` features
    blocks : torch.nn.ModuleList
        The transformer blocks
    output_norm : torch.nn.LayerNorm
    attention_pool, fc, pooling, uses_log_length
        The readout (see ParrotArchitecture.build_readout)
    """

    name = 'transformer'
    description = 'transformer encoder (multi-head self-attention)'
    hyperparameters = (
        HyperParameter('num_layers', int, default=2, help='number of transformer blocks', minimum=1,
                       cli_aliases=('-nl',), search_range=(1, 6)),
        HyperParameter('num_heads', int, default=4, help='number of attention heads', minimum=1,
                       search_range=(1, 8)),
        HyperParameter('head_size', int, default=16, minimum=1, search_range=(8, 64),
                       help='width of each attention head; the model width is num_heads x head_size'),
        HyperParameter('feedforward_multiplier', int, default=4, minimum=1, search_range=(1, 4),
                       help='width of the feed-forward layer in each block, as a multiple of the model width'),
        DROPOUT,
        HyperParameter('positional_encoding', str, default='rotary', choices=POSITIONAL_ENCODINGS,
                       search_choices=('rotary', 'sinusoidal'),
                       help="how residue positions are encoded: 'rotary' (relative), 'sinusoidal' (absolute) or "
                            "'none' (order-free)"),
        POOLING,
        ATTENTION_POOLING,
        LOGLENGTH_FEEDFORWARD,
    )

    def __init__(self, input_size: int, task: Task, num_layers: int, num_heads: int, head_size: int,
                 feedforward_multiplier: int, dropout: float, positional_encoding: str, pooling: str,
                 attention_pooling: bool, loglength_feedforward: bool) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        num_layers : int
            Number of transformer blocks
        num_heads : int
            Number of attention heads
        head_size : int
            Width of each attention head
        feedforward_multiplier : int
            Width of each feed-forward layer, as a multiple of the model width
        dropout : float
            Dropout probability
        positional_encoding : str
            'rotary', 'sinusoidal' or 'none'
        pooling : str
            'mean' or 'max'; how residue features are pooled for sequence tasks when attention pooling is off
        attention_pooling : bool
            Sequence tasks: pool residue features with learned attention weights
        loglength_feedforward : bool
            Sequence tasks: also give the output layer log(sequence length)
        """
        super().__init__(input_size, task, {'num_layers': num_layers, 'num_heads': num_heads,
                                            'head_size': head_size, 'feedforward_multiplier': feedforward_multiplier,
                                            'dropout': dropout, 'positional_encoding': positional_encoding,
                                            'pooling': pooling, 'attention_pooling': attention_pooling,
                                            'loglength_feedforward': loglength_feedforward})
        self.positional_encoding = positional_encoding
        self.model_size = num_heads * head_size
        self.input_projection = nn.Linear(input_size, self.model_size)
        self.blocks = nn.ModuleList([TransformerBlock(num_heads, head_size, feedforward_multiplier,
                                                      positional_encoding == 'rotary', dropout)
                                     for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(self.model_size)
        self.build_readout(self.model_size, attention_pooling=attention_pooling, pooling=pooling,
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

        hidden = self.input_projection(x)
        if self.positional_encoding == 'sinusoidal':
            hidden = hidden + sinusoidal_positions(x.shape[1], self.model_size, x.device).to(hidden.dtype)
        hidden = zero_padding(hidden, mask)

        for block in self.blocks:
            hidden = block(hidden, mask)

        features = self.output_norm(hidden)
        return self.readout(features, mask, lengths)
