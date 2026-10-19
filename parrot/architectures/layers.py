"""
Building blocks shared by several architectures, mostly for handling padding.

PARROT batches sequences of different lengths by zero-padding them at the end,
and every architecture must give the same output for a sequence, up to float32
rounding, however much padding it was given. The helpers here make that easy:
build a mask of the real residues, re-zero the padding after any layer that
could write into it, normalise each position independently, and pool only over
real residues.

Tensor layouts. Batches are [batch_dim X padded_length X features]
("residues-first"), as PARROT's data loaders produce them and as linear layers,
attention and LSTMs expect. Convolutions work on [batch_dim X channels X
padded_length] ("channels-first") instead. Masks follow the same split:

- a padding mask is a boolean [batch_dim X padded_length] tensor, True at real
  residues (padding_mask());
- a channel mask is the same mask as a float [batch_dim X 1 X padded_length]
  tensor (channel_mask()), which multiplies a channels-first tensor to zero its
  padding.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import torch
import torch.nn as nn

from parrot.architectures.hyperparameters import POOLING_METHODS


# ..........................................................................................
# Lengths and masks

def resolve_lengths(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Return the true length of each sequence in a batch, on the batch's device

    Parameters
    ----------
    x : torch.Tensor
        Batch of shape [batch_dim X padded_length X input_size]
    lengths : torch.Tensor or None
        True lengths, or None if every position of `x` is a real residue

    Returns
    -------
    torch.Tensor
        int64 tensor of shape [batch_dim]
    """

    if lengths is None:
        return torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
    return lengths.to(device=x.device, dtype=torch.long)


def padding_mask(lengths: torch.Tensor, padded_length: int) -> torch.Tensor:
    """Build a boolean mask that is True at real residues and False at padding

    Parameters
    ----------
    lengths : torch.Tensor
        The true length of each sequence in the batch
    padded_length : int
        The length every sequence in the batch was padded to

    Returns
    -------
    torch.Tensor
        Boolean mask of shape [batch_dim X padded_length], on the same device as `lengths`
    """

    positions = torch.arange(padded_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def channel_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn a boolean [batch_dim X padded_length] padding mask into a float [batch_dim X 1 X padded_length] one

    Multiplying a channels-first tensor by the result zeroes its padded positions.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean padding mask (see padding_mask())
    dtype : torch.dtype
        Floating-point type of the tensors the mask will multiply

    Returns
    -------
    torch.Tensor
        1.0 at real residues and 0.0 at padding, [batch_dim X 1 X padded_length]
    """
    return mask.unsqueeze(1).to(dtype)


def zero_padding(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set every padded position of a residues-first batch to zero

    Parameters
    ----------
    x : torch.Tensor
        [batch_dim X padded_length X features]
    mask : torch.Tensor
        Boolean [batch_dim X padded_length] padding mask

    Returns
    -------
    torch.Tensor
        `x` with zeros at padded positions
    """
    return x * mask.unsqueeze(-1).to(x.dtype)


# ..........................................................................................
# Summaries of a whole sequence

def mean_over_residues(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Average each sequence's encoded residues (for one-hot encoding: its composition)

    Parameters
    ----------
    x : torch.Tensor
        [batch_dim X padded_length X input_size], zero at padded positions
    lengths : torch.Tensor
        The true length of each sequence

    Returns
    -------
    torch.Tensor
        [batch_dim X input_size]
    """

    return x.sum(dim=1) / lengths.unsqueeze(-1).to(x.dtype)


def append_log_length(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Add log(sequence length) as one more feature of each sequence

    Parameters
    ----------
    features : torch.Tensor
        [batch_dim X num_features]
    lengths : torch.Tensor
        The true length of each sequence

    Returns
    -------
    torch.Tensor
        [batch_dim X num_features + 1], with log(length) last
    """
    return torch.cat([features, torch.log(lengths.to(features.dtype)).unsqueeze(-1)], dim=-1)


def composition_and_log_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Each sequence's average encoded residue (its composition, for one-hot encoding) followed by log(length)

    Parameters
    ----------
    x : torch.Tensor
        [batch_dim X padded_length X input_size], zero at padded positions
    lengths : torch.Tensor
        The true length of each sequence

    Returns
    -------
    torch.Tensor
        [batch_dim X input_size + 1]
    """
    return append_log_length(mean_over_residues(x, lengths), lengths)


def adjacent_pair_composition(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Average, over each sequence, the outer product of every residue's encoding with the next one's

    For one-hot encoding this is the dipeptide composition: the fraction of
    adjacent residue pairs that are each of the 400 possible dipeptides.

    Parameters
    ----------
    x : torch.Tensor
        [batch_dim X padded_length X input_size], zero at padded positions (so
        pairs that include padding contribute nothing)
    lengths : torch.Tensor
        The true length of each sequence

    Returns
    -------
    torch.Tensor
        [batch_dim X input_size**2]. All zeros for sequences of length 1, which
        have no adjacent pairs.
    """

    # pair_sums[b, d, e] = sum over positions l of x[b, l, d] * x[b, l + 1, e]
    pair_sums = torch.einsum('bld,ble->bde', x[:, :-1], x[:, 1:])
    num_pairs = (lengths - 1).clamp(min=1).to(x.dtype)
    return pair_sums.flatten(start_dim=1) / num_pairs.unsqueeze(-1)


# ..........................................................................................
# Pooling residue features into one vector per sequence

def masked_pool(features: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
    """Pool per-residue features into one vector per sequence, ignoring padding

    Parameters
    ----------
    features : torch.Tensor
        [batch_dim X padded_length X channels]
    mask : torch.Tensor
        Boolean [batch_dim X padded_length] mask of real residues
    method : str
        'mean' or 'max'

    Returns
    -------
    torch.Tensor
        [batch_dim X channels]

    Raises
    ------
    ValueError
        If `method` is not one of POOLING_METHODS
    """

    if method == 'mean':
        weights = mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1)

    if method == 'max':
        return features.masked_fill(~mask.unsqueeze(-1), float('-inf')).amax(dim=1)

    raise ValueError(f"Unknown pooling method '{method}': must be one of {POOLING_METHODS}")


class AttentionPool(nn.Module):
    """Pool per-residue features into one vector per sequence with learned attention weights

    Attention-based pooling (Ilse et al. 2018, ICML): each real residue gets a
    score w . tanh(V h), a softmax over the sequence's real residues turns the
    scores into weights, and the pooled vector is the weighted average of the
    residue features. Unlike the mean, the network learns which residues matter;
    unlike the maximum, every residue contributes. Padded positions get zero weight.

    Attributes
    ----------
    score : torch.nn.Sequential
        Computes each residue's score
    """

    def __init__(self, channels: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of features per residue
        """
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, channels), nn.Tanh(), nn.Linear(channels, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            [batch_dim X padded_length X channels]
        mask : torch.Tensor
            Boolean [batch_dim X padded_length] mask of real residues

        Returns
        -------
        torch.Tensor
            [batch_dim X channels]
        """
        scores = self.score(features).squeeze(-1)

        # A score of -inf gives a weight of exactly zero, so padding never contributes
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)

        # pooled[b, c] = sum over residues l of weights[b, l] * features[b, l, c]
        pooled: torch.Tensor = torch.einsum('bl,blc->bc', weights, features)
        return pooled


# ..........................................................................................
# Layers for channels-first (convolutional) networks

class GlobalContext(nn.Module):
    """A summary of the whole sequence, to be added to every position of a channels-first batch

    The summary is the average of the features over the sequence's real
    positions, passed through a linear layer (as in global-context and
    squeeze-and-excitation networks: Hu et al. 2018, CVPR; Cao et al. 2019,
    ICCV Workshops). Added to every position, it lets a convolutional network
    see the whole sequence however long it is. The linear layer starts at zero,
    without drawing any random numbers, so a new network starts out exactly as
    the same network without the summary would from the same seed (its other
    layers get the same initial weights).

    Attributes
    ----------
    linear : torch.nn.Linear
    """

    def __init__(self, channels: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of channels
        """
        super().__init__()
        # skip_init leaves the weights uninitialised rather than drawing them at random, which would shift every
        # random number later layers draw
        self.linear = nn.utils.skip_init(nn.Linear, channels, channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_dim X channels X padded_length]
        channel_mask : torch.Tensor
            Float [batch_dim X 1 X padded_length] mask of real positions

        Returns
        -------
        torch.Tensor
            [batch_dim X channels X 1], the same for every position of a sequence
        """
        summary = (x * channel_mask).sum(dim=-1) / channel_mask.sum(dim=-1)
        context: torch.Tensor = self.linear(summary).unsqueeze(-1)
        return context


class ChannelLayerNorm(nn.Module):
    """Layer normalisation over the channels of a channels-first [batch X channels X length] tensor

    Each position is normalised on its own, so (unlike batch normalisation)
    padded positions and other sequences in the batch cannot affect the result.
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
            [batch_dim X channels X length]

        Returns
        -------
        torch.Tensor
            Same shape as `x`, each position normalised over its channels
        """
        # nn.LayerNorm normalises the last dimension, so move the channels there and back
        normalised: torch.Tensor = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return normalised


# ..........................................................................................
# Fully connected networks

def multilayer_perceptron(input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int,
                          dropout: float) -> nn.Sequential:
    """Fully connected network applied to the last dimension of its input

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Width of each hidden layer
    num_hidden_layers : int
        Number of (linear -> ReLU -> dropout) hidden layers. 0 gives a single
        linear layer.
    output_size : int
        Number of outputs
    dropout : float
        Dropout probability after each hidden layer

    Returns
    -------
    torch.nn.Sequential
    """

    layers: list[nn.Module] = []
    layer_input_size = input_size
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(layer_input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layer_input_size = hidden_size
    layers.append(nn.Linear(layer_input_size, output_size))
    return nn.Sequential(*layers)
