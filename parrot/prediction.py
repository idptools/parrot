"""
Running a trained network: on a labeled dataset (e.g. the test set, to see how
well it does) or on new, unlabeled sequences, and turning its raw outputs into
the predictions reported to the user.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from parrot import process_input_data as pid
from parrot.architectures import ParrotArchitecture
from parrot.devices import full_float32_precision
from parrot.matrices.data import unpad_matrix_outputs
from parrot.sequence_encoder import SequenceEncoder
from parrot.tasks import Task
from parrot.train_network import build_loss_criterion, compute_batch_loss, sum_batch_losses


@dataclass
class LabeledPredictions:
    """Network outputs for every sequence in a labeled dataset

    Attributes
    ----------
    names : list of str
        Sequence IDs
    sequences : list of str
        The sequences
    true_values : list of numpy arrays
        The true value(s) for each sequence: a scalar for sequence-mapped data,
        or one value per residue for residue-mapped data
    outputs : list of numpy arrays
        The raw network output for each sequence: [num_classes] for
        sequence-mapped data or [sequence_length X num_classes] for residue-mapped
        data (classification outputs are logits)
    loss : float
        Average loss per sequence, computed with the training loss function
    """

    names: list[str]
    sequences: list[str]
    true_values: list[np.ndarray]
    outputs: list[np.ndarray]
    loss: float


def unpad_outputs(outputs: torch.Tensor, lengths: torch.Tensor, task: Task) -> list[np.ndarray]:
    """Split a batch of network outputs into one numpy array per sequence

    Parameters
    ----------
    outputs : torch.Tensor
        Batch output of a network
    lengths : torch.Tensor
        The true length of each sequence
    task : Task
        The task the network is for

    Returns
    -------
    list of numpy arrays
        [num_classes] per sequence, [sequence_length X num_classes] per
        sequence, or (matrix tasks) [sequence_length X width X num_classes] per
        sequence, with the padding removed
    """

    outputs_array = outputs.cpu().numpy()
    if task.is_matrix:
        return unpad_matrix_outputs(outputs_array, lengths.tolist(), task)
    if not task.per_residue:
        return list(outputs_array)
    return [outputs_array[i, :length] for i, length in enumerate(lengths.tolist())]


@full_float32_precision()
def predict_labeled(network: nn.Module, dataset: pid.SequenceDataset, task: Task, device: str | torch.device,
                    batch_size: int = 32) -> LabeledPredictions:
    """Run a trained network over a labeled dataset (e.g. the test set)

    Parameters
    ----------
    network : torch.nn.Module
        The trained network
    dataset : SequenceDataset
        The labeled sequences
    task : Task
        The task the network is for
    device : str or torch.device
        Device the network lives on
    batch_size : int, optional
        Number of sequences evaluated at once (does not affect the results)

    Returns
    -------
    LabeledPredictions
    """

    # Sequences are batched by length (which only affects speed), so the outputs come
    # out in the batch sampler's order and are put back into dataset order at the end
    loader = pid.make_dataloader(dataset, task, batch_size, shuffle=False, batching='length')
    assert loader.batch_sampler is not None
    loader_order = [index for batch_indices in loader.batch_sampler for index in batch_indices]
    criterion = build_loss_criterion(task)
    batch_losses = []
    outputs_in_loader_order = []

    network.eval()
    with torch.no_grad():
        for batch in loader:
            loss, batch_outputs = compute_batch_loss(network, batch, task, criterion, device)
            batch_losses.append(loss.detach())
            outputs_in_loader_order.extend(unpad_outputs(batch_outputs, batch.lengths, task))

    outputs: list[np.ndarray] = [np.empty(0)] * len(dataset)
    for index, output in zip(loader_order, outputs_in_loader_order, strict=True):
        outputs[index] = output

    return LabeledPredictions(names=dataset.names, sequences=dataset.sequences,
                              true_values=[np.asarray(value) for value in dataset.values],
                              outputs=outputs, loss=sum_batch_losses(batch_losses) / len(dataset))


@full_float32_precision()
def predict_sequences(network: ParrotArchitecture, sequences: list[str], encoder: SequenceEncoder,
                      device: str | torch.device, batch_size: int = 64,
                      print_frequency: int | None = None) -> dict[str, np.ndarray]:
    """Run a trained network over unlabeled sequences

    Parameters
    ----------
    network : ParrotArchitecture
        The trained network (its `task` attribute says how to read the output)
    sequences : list of str
        Sequences to predict. Each unique sequence is only run once.
    encoder : SequenceEncoder
        The encoder the network was trained with
    device : str or torch.device
        Device the network lives on
    batch_size : int, optional
        Number of sequences run through the network at once (does not affect
        the results, since padding is masked)
    print_frequency : int, optional
        If given, print a progress update every `print_frequency` sequences

    Returns
    -------
    dict
        Maps each unique sequence to its raw network output: [num_classes] for
        sequence-mapped networks, [sequence_length X num_classes] for
        residue-mapped networks
    """

    # Batch sequences of similar length together, so little computation is spent on padding
    unique_sequences = list(dict.fromkeys(sequences))
    by_length = sorted(unique_sequences, key=len)
    predictions = {}

    network.eval()
    for start in range(0, len(by_length), batch_size):
        batch_sequences = by_length[start:start + batch_size]

        vectors, lengths = pid.pad_sequence_vectors([encoder.encode(seq) for seq in batch_sequences])
        with torch.no_grad():
            outputs = network(vectors.to(device), lengths)

        for sequence, output in zip(batch_sequences, unpad_outputs(outputs, lengths, network.task)):
            predictions[sequence] = output

        # Report progress each time another print_frequency sequences have been predicted
        done = start + len(batch_sequences)
        if print_frequency is not None and done // print_frequency > start // print_frequency:
            print(f'  predicted {done:,} of {len(unique_sequences):,} sequences')

    # Return the sequences in the order they were first given
    return {sequence: predictions[sequence] for sequence in unique_sequences}


def output_to_prediction(output: np.ndarray, task: Task, probabilistic: bool = False) -> np.ndarray:
    """Convert a raw network output into the prediction reported to the user

    Parameters
    ----------
    output : numpy array
        Raw output for one sequence: [num_classes] or [sequence_length X num_classes]
    task : Task
        The task the network is for
    probabilistic : bool, optional
        For classification, return class probabilities instead of the most
        likely class

    Returns
    -------
    numpy array
        Regression: the predicted value (scalar) or one value per residue.
        Classification: the predicted class (scalar) or one class per residue,
        or, if `probabilistic`, the probability of each class.
    """

    if not task.is_classification:
        return output[..., 0]
    if probabilistic:
        return softmax(output, axis=-1)
    return np.argmax(output, axis=-1)
