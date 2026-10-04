"""
The bidirectional LSTM network that PARROT has always used.

For sequence-mapped tasks each sequence is summarised by attention-pooling the
LSTM output at every residue (by default), or by concatenating the final hidden
states of the forward and reverse directions (as PARROT always did, and with
--attention-pooling-off). The summary, by default with log(sequence length), is
passed through a linear layer ("many-to-one").
For residue-mapped tasks the linear layer is applied to the LSTM output at
every residue ("many-to-many").

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import torch
import torch.nn as nn
from torch.func import functional_call

from parrot.architectures.base import HyperParameter, ParrotArchitecture
from parrot.architectures.layers import (ATTENTION_POOLING, LOGLENGTH_FEEDFORWARD, AttentionPool, append_log_length,
                                        padding_mask, resolve_lengths)
from parrot.architectures.registry import register_architecture
from parrot.tasks import Task


def run_lstm(lstm: nn.LSTM, x: torch.Tensor,
             lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a batch_first bidirectional LSTM over a (possibly zero-padded) batch of sequences

    Sequences in a batch are zero-padded at the end up to the length of the
    longest sequence. If the LSTM simply ran over those padding positions, the
    reverse direction would start inside the padding, so a sequence's output
    would depend on which other sequences it was batched with. When `lengths` is
    given, each direction is run separately on unpacked tensors
    (run_lstm_unpacked) so the LSTM sees only the real residues of each
    sequence. This gives the same result, to floating-point precision, as
    packing the batch with pack_padded_sequence, but is faster on every device:
    packed sequences are ~1.7x slower on the CPU, ~16x slower on Apple-silicon
    GPUs (MPS), and up to ~7x slower on NVIDIA GPUs (CUDA), where the gap grows
    with sequence length.

    Parameters
    ----------
    lstm : torch.nn.LSTM
        A bidirectional LSTM constructed with batch_first=True
    x : torch.Tensor
        Input batch in the format [batch_dim X padded_length X input_size]
    lengths : torch.Tensor, optional
        The true (unpadded) length of each sequence in the batch. If None, every
        position of `x` is treated as a real residue (i.e. no padding).

    Returns
    -------
    torch.Tensor
        Per-position LSTM output in the format
        [batch_dim X padded_length X hidden_size*2]. Padded positions are zero.
    torch.Tensor
        Final hidden state of each layer and direction, in the format
        [num_layers*2 X batch_dim X hidden_size] (ordered as torch.nn.LSTM
        orders them), taken at each sequence's true final residue.
    """

    # Initial hidden and cell states default to zeros on the same device as x
    if lengths is None:
        out, (h_n, _) = lstm(x)
        return out, h_n

    return run_lstm_unpacked(lstm, x, lengths)


def run_lstm_unpacked(lstm: nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """run_lstm() without packing, by running each direction of each layer separately

    The forward direction can run straight over the padded batch: its output at
    a real residue only depends on the residues before it, and the padding comes
    after every real residue. For the reverse direction, each sequence is
    reversed within its own length (the padding stays at the end), run forwards,
    and reversed back. Layers are computed one at a time because each layer's
    input is both directions of the layer below. The result equals running the
    bidirectional LSTM on packed sequences.

    The weights are always the bidirectional `lstm`'s own parameters, so saved
    networks are unaffected; each direction is computed by calling a weightless
    single-layer LSTM with those parameters (torch.func.functional_call).

    Parameters
    ----------
    lstm, x, lengths
        As for run_lstm()

    Returns
    -------
    As for run_lstm()
    """
    if lstm.proj_size != 0 or not lstm.bidirectional or not lstm.batch_first:
        raise ValueError('run_lstm_unpacked expects a batch_first bidirectional LSTM without projections')

    lengths = lengths.to(device=x.device, dtype=torch.long)
    mask = padding_mask(lengths, x.shape[1]).unsqueeze(-1)
    last_residue = lengths - 1
    batch_indices = torch.arange(x.shape[0], device=x.device)
    parameter_names = ['weight_ih', 'weight_hh'] + (['bias_ih', 'bias_hh'] if lstm.bias else [])

    layer_input = x
    final_hidden_states = []
    for layer in range(lstm.num_layers):
        # A single-layer, one-direction LSTM with no weights of its own; building it on the
        # 'meta' device allocates nothing and draws no random numbers
        direction_lstm = nn.LSTM(layer_input.shape[-1], lstm.hidden_size, batch_first=True, bias=lstm.bias,
                                 device='meta')

        direction_outputs = []
        for suffix, reverse in (('', False), ('_reverse', True)):
            weights = {f'{name}_l0': getattr(lstm, f'{name}_l{layer}{suffix}') for name in parameter_names}
            direction_input = reverse_within_lengths(layer_input, lengths) if reverse else layer_input
            direction_output, _ = functional_call(direction_lstm, weights, (direction_input,))
            if reverse:
                direction_output = reverse_within_lengths(direction_output, lengths)
            direction_outputs.append(direction_output)

        forward_output, reverse_output = direction_outputs
        # Final states: forward after the last real residue, reverse after the first residue
        final_hidden_states += [forward_output[batch_indices, last_residue], reverse_output[:, 0]]

        layer_input = torch.cat(direction_outputs, dim=-1)
        if layer < lstm.num_layers - 1 and lstm.dropout > 0:
            layer_input = nn.functional.dropout(layer_input, p=lstm.dropout, training=lstm.training)

    # Zero the padded positions, as unpacking a packed sequence does
    return layer_input * mask, torch.stack(final_hidden_states)


def reverse_within_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence of a padded batch within its own length

    Position t < length becomes position length-1-t; padding positions stay
    where they are. Applying this twice returns the original batch.

    Parameters
    ----------
    x : torch.Tensor
        Batch in the format [batch_dim X padded_length X features]
    lengths : torch.Tensor
        True length of each sequence, on the same device as `x`

    Returns
    -------
    torch.Tensor
        Same shape as `x`
    """
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    true_lengths = lengths.unsqueeze(1)
    source = torch.where(positions < true_lengths, true_lengths - 1 - positions, positions)
    return x.gather(1, source.unsqueeze(-1).expand_as(x))


@register_architecture
class BRNN(ParrotArchitecture):
    """Bidirectional LSTM followed by a single fully connected layer.

    The submodules are called `lstm` and `fc`; networks saved by every earlier
    version of PARROT use exactly these parameter names, so they must not change.

    Attributes
    ----------
    hidden_size : int
        Size of the LSTM hidden vectors
    num_layers : int
        Number of stacked LSTM layers in each direction
    loglength_feedforward : bool
        Whether the output layer also sees log(sequence length) (sequence-mapped tasks only)
    lstm : torch.nn.LSTM
        The bidirectional LSTM layer(s)
    attention_pool : AttentionPool or None
        Sequence-mapped tasks with attention pooling: pools the LSTM output at
        every residue into one vector per sequence. None means the final hidden
        states of the two directions are used instead.
    fc : torch.nn.Linear
        Fully connected layer mapping the LSTM output to the network output
    """

    name = 'brnn'
    description = 'bidirectional LSTM (the original PARROT network)'
    hyperparameters = (
        HyperParameter('hidden_size', int, default=10, help='size of the LSTM hidden vectors',
                       cli_aliases=('-hs',), minimum=1, search_range=(5, 50)),
        HyperParameter('num_layers', int, default=1, help='number of LSTM layers per direction',
                       cli_aliases=('-nl',), minimum=1, search_range=(1, 5)),
        ATTENTION_POOLING,
        LOGLENGTH_FEEDFORWARD,
    )

    def __init__(self, input_size: int, task: Task, hidden_size: int, num_layers: int, attention_pooling: bool,
                 loglength_feedforward: bool) -> None:
        """
        Parameters
        ----------
        input_size : int
            Length of the per-residue encoding vectors
        task : Task
            The task the network is being built for
        hidden_size : int
            Size of the LSTM hidden vectors
        num_layers : int
            Number of stacked LSTM layers in each direction
        attention_pooling : bool
            Sequence tasks: summarise each sequence by attention-pooling the LSTM
            output at every residue, rather than by the final hidden states
        loglength_feedforward : bool
            Sequence tasks: also give the output layer log(sequence length)
        """
        super().__init__(input_size, task, {'hidden_size': hidden_size, 'num_layers': num_layers,
                                            'attention_pooling': attention_pooling,
                                            'loglength_feedforward': loglength_feedforward})
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loglength_feedforward = loglength_feedforward and not task.per_residue
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # *2 because the forward and reverse directions are concatenated
        use_attention = attention_pooling and not task.per_residue
        self.attention_pool = AttentionPool(hidden_size * 2) if use_attention else None

        # *2 because the forward and reverse directions are concatenated (+1 for log(length))
        self.fc = nn.Linear(in_features=hidden_size * 2 + int(self.loglength_feedforward),
                            out_features=task.num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Propagate a batch of encoded sequences through the network

        Parameters
        ----------
        x : torch.Tensor
            Zero-padded batch in the format [batch_dim X padded_length X input_size]
        lengths : torch.Tensor, optional
            The true length of each sequence. If None (default) every position
            is treated as a real residue, which is correct for a single
            unpadded sequence.

        Returns
        -------
        torch.Tensor
            [batch_dim X num_classes] for sequence-mapped tasks, or
            [batch_dim X padded_length X num_classes] for residue-mapped tasks.
            Outputs at padded positions carry no information.
        """
        out, h_n = run_lstm(self.lstm, x, lengths)

        if self.task.per_residue:
            per_residue_output: torch.Tensor = self.fc(out)
            return per_residue_output

        lengths = resolve_lengths(x, lengths)
        if self.attention_pool is not None:
            # Every residue's output contributes, weighted by learned attention
            summary = self.attention_pool(out, padding_mask(lengths, out.shape[1]))
        else:
            # Final hidden state of the last layer in each direction: h_n[-2] is the
            # forward direction (after seq[n]) and h_n[-1] the reverse (after seq[0])
            summary = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        if self.loglength_feedforward:
            summary = append_log_length(summary, lengths)
        per_sequence_output: torch.Tensor = self.fc(summary)
        return per_sequence_output
