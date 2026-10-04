"""
Saving and loading trained PARROT networks.

Networks are saved as a single self-describing file containing the weights
together with everything needed to rebuild the network: the architecture name,
its hyperparameters, the task and the sequence encoding (including the full
table for user-defined encodings). Files written by earlier versions of PARROT
contain only the weights of a BRNN; these can still be loaded, but the datatype
(and the encoding, if it was not one-hot) must then be supplied by the caller.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from typing import Any

import torch

from parrot.architectures import ParrotArchitecture, build_network, get_architecture
from parrot.architectures.registry import fill_legacy_values
from parrot.encode_sequence import SequenceEncoder, encoder_from_config, get_encoder
from parrot.package_version import PARROT_VERSION
from parrot.tasks import Task

# Version of the saved-network format. Increase this if the format changes.
CHECKPOINT_FORMAT_VERSION = 1

# Key that identifies a file as a self-describing PARROT checkpoint
FORMAT_VERSION_KEY = 'parrot_checkpoint_format'


def save_network(path: str, network: ParrotArchitecture, encoder: SequenceEncoder) -> None:
    """Save a network, and everything needed to rebuild it, to a single file

    Parameters
    ----------
    path : str
        File to write (conventionally with a .pt extension)
    network : ParrotArchitecture
        The network to save
    encoder : SequenceEncoder
        The encoder the network was trained with
    """

    checkpoint = {
        FORMAT_VERSION_KEY: CHECKPOINT_FORMAT_VERSION,
        'parrot_version': PARROT_VERSION,
        'architecture': network.name,
        'hyperparameters': network.hyperparameter_values,
        'input_size': network.input_size,
        'task': network.task.to_config(),
        'encoding': encoder.to_config(),
        'state_dict': cpu_state_dict(network),
    }
    torch.save(checkpoint, path)


def cpu_state_dict(network: torch.nn.Module) -> dict[str, Any]:
    """A network's state dict with every tensor copied to the CPU

    Saving CPU tensors means a network trained on any device (CUDA or MPS) can
    be loaded on any machine, including by code that calls torch.load directly.

    Parameters
    ----------
    network : torch.nn.Module

    Returns
    -------
    dict
        Parameter name -> CPU tensor (non-tensor entries, such as the fitted
        scikit-learn model of a classical network, are kept as they are)
    """
    return {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in network.state_dict().items()}


def load_network(path: str, device: str | torch.device = 'cpu', legacy_datatype: str | None = None,
                 legacy_encoding: str = 'onehot') -> tuple[ParrotArchitecture, SequenceEncoder]:
    """Load a saved network and the encoder it was trained with

    Parameters
    ----------
    path : str
        File written by save_network(), or a weights-only file from an earlier
        version of PARROT
    device : str or torch.device, optional
        Device to load the network onto (default 'cpu')
    legacy_datatype : str, optional
        Only used for weights-only files: 'sequence' or 'residues'. Required for
        those files because they do not record it.
    legacy_encoding : str, optional
        Only used for weights-only files: the encoding the network was trained
        with ('onehot' (default), 'biophysics' or a path to an encoding file)

    Returns
    -------
    ParrotArchitecture
        The network, on `device`, in evaluation mode
    SequenceEncoder
        The encoder to use with the network

    Raises
    ------
    ValueError
        If a weights-only file is loaded without `legacy_datatype`, or its
        encoding does not match the network's input size
    """

    # Load onto the CPU first, so files saved from any device can be read, then move
    saved = torch.load(path, map_location='cpu')

    if is_checkpoint(saved):
        # Hyperparameters added since the network was saved take the value that reproduces it
        hyperparameters = fill_legacy_values(get_architecture(saved['architecture']), saved['hyperparameters'])
        network = build_network(saved['architecture'], saved['input_size'], Task.from_config(saved['task']),
                                hyperparameters)
        network.load_state_dict(saved['state_dict'])
        encoder = encoder_from_config(saved['encoding'])
    else:
        network, encoder = _network_from_legacy_weights(saved, legacy_datatype, legacy_encoding)

    network.to(device)
    network.eval()
    return network, encoder


def is_checkpoint(saved: Any) -> bool:
    """True if a loaded file is a self-describing checkpoint rather than bare weights."""
    return isinstance(saved, dict) and FORMAT_VERSION_KEY in saved


def infer_legacy_brnn_shape(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Work out the shape of a BRNN from its weights

    Parameters
    ----------
    state_dict : dict
        Weights of a BRNN saved by an earlier version of PARROT

    Returns
    -------
    dict
        'input_size', 'hidden_size', 'num_layers' and 'num_classes'
    """

    # The LSTM has weights named lstm.weight_ih_l0, lstm.weight_ih_l1, ... one per layer
    num_layers = 0
    while f'lstm.weight_ih_l{num_layers}' in state_dict:
        num_layers += 1

    # The LSTM stacks its four gates along the first dimension of weight_ih, so that
    # dimension is 4*hidden_size, and the second dimension is the input size
    first_layer_weights = state_dict['lstm.weight_ih_l0']
    return {'input_size': int(first_layer_weights.shape[1]),
            'hidden_size': int(first_layer_weights.shape[0] // 4),
            'num_layers': num_layers,
            'num_classes': int(state_dict['fc.bias'].shape[0])}


def _network_from_legacy_weights(state_dict: dict[str, torch.Tensor], datatype: str | None,
                                 encoding: str) -> tuple[ParrotArchitecture, SequenceEncoder]:
    """Rebuild a BRNN from a weights-only file written by an earlier version of PARROT."""

    if datatype is None:
        raise ValueError('This network was saved by an older version of PARROT and does not record its datatype; '
                         "please specify whether it is a 'sequence' or 'residues' network.")

    shape = infer_legacy_brnn_shape(state_dict)
    encoder = get_encoder(encoding)
    if len(encoder) != shape['input_size']:
        raise ValueError(f"The network expects input vectors of length {shape['input_size']}, but the "
                         f"'{encoding}' encoding produces vectors of length {len(encoder)}. "
                         'Please specify the encoding the network was trained with.')

    task = Task(datatype, shape['num_classes'])
    network = build_network('brnn', shape['input_size'], task,
                            fill_legacy_values(get_architecture('brnn'), {'hidden_size': shape['hidden_size'],
                                                                         'num_layers': shape['num_layers']}))
    network.load_state_dict(state_dict)
    return network, encoder
