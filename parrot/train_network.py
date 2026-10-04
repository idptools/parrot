"""
Core training module of PARROT: training any registered architecture, evaluating
it on labeled data and making predictions on new sequences.

Nothing in this module depends on which architecture is being used; networks
only need to follow the ParrotArchitecture interface (see
parrot/architectures/base.py).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.utils.data import DataLoader

from parrot import process_input_data as pid
from parrot.architectures import HyperParameter, ParrotArchitecture
from parrot.devices import full_float32_precision
from parrot.architectures.classical import ClassicalModel
from parrot.architectures.layers import padding_mask
from parrot.encode_sequence import SequenceEncoder
from parrot.tasks import Task

# The learning rate is the one hyperparameter shared by every architecture
LEARNING_RATE = HyperParameter('learning_rate', float, default=0.001, help='initial learning rate',
                               search_range=(1e-5, 1e-2), log_scale=True)

# With --stop auto, the validation loss must fall by at least this fraction relative
# to every recent epoch to count as still improving
AUTO_STOP_RELATIVE_IMPROVEMENT = 0.005

# How often (in epochs) the validation loss is printed when not in verbose mode
PRINT_EVERY_N_EPOCHS = 5


# .................................................................
# Losses

def build_loss_criterion(task: Task) -> nn.Module:
    """Build the loss function used for training, validation and testing

    Losses are summed over every sequence (and, for residue-mapped data, every
    real residue) in a batch. Callers divide the accumulated total by the number
    of sequences, so all reported losses are average loss per sequence.

    Parameters
    ----------
    task : Task
        The task being trained

    Returns
    -------
    torch.nn.Module
        MSE for residue regression, L1 for sequence regression, and cross-
        entropy for classification. The cross-entropy loss ignores padded
        residues and residues with missing labels, which carry the label
        IGNORED_CLASS_LABEL.
    """

    if task.is_classification:
        return nn.CrossEntropyLoss(reduction='sum', ignore_index=pid.IGNORED_CLASS_LABEL)
    if task.per_residue:
        return nn.MSELoss(reduction='sum')
    return nn.L1Loss(reduction='sum')


def compute_batch_loss(network: nn.Module, batch: tuple, task: Task, criterion: nn.Module,
                       device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a batch through the network and compute its summed loss

    Padded positions and residues with missing labels never contribute to the
    loss: for regression they are masked out here, and for classification they
    carry the label IGNORED_CLASS_LABEL, which `criterion` ignores.

    Parameters
    ----------
    network : torch.nn.Module
        A network following the ParrotArchitecture interface
    batch : tuple
        (names, padded vectors, targets, lengths) as produced by collate_batch()
    task : Task
        The task being trained
    criterion : torch.nn.Module
        The loss function returned by build_loss_criterion()
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    torch.Tensor
        Scalar loss summed over the batch
    torch.Tensor
        The network outputs: [batch_dim X num_classes] or
        [batch_dim X padded_length X num_classes]
    """

    names, vectors, targets, lengths = batch
    vectors = vectors.to(device)
    targets = targets.to(device)

    outputs = network(vectors, lengths)

    if task.is_classification and task.per_residue:
        # CrossEntropyLoss expects the class dimension second
        loss = criterion(outputs.permute(0, 2, 1), targets)
    elif task.per_residue:
        # Only real residues with a label (missing labels are NaN)
        mask = padding_mask(lengths.to(outputs.device), outputs.shape[1]) & ~torch.isnan(targets[..., 0])
        loss = criterion(outputs[mask], targets[mask])
    else:
        loss = criterion(outputs, targets)

    return loss, outputs


def sum_batch_losses(batch_losses: list[torch.Tensor]) -> float:
    """Add up per-batch losses that were kept on the device

    Calling .item() on every batch's loss would make the CPU wait for the GPU
    after every batch. Keeping the losses on the device and adding them up once
    gives the same total (each loss is the same float32 value .item() would
    return, summed in float64 on the CPU) without those waits.

    Parameters
    ----------
    batch_losses : list of torch.Tensor
        Scalar, detached loss of each batch

    Returns
    -------
    float
        The total loss (0.0 for an empty list)
    """
    if not batch_losses:
        return 0.0
    # Move to the CPU before converting: MPS tensors cannot be converted to float64
    return float(torch.stack(batch_losses).cpu().double().sum())


@full_float32_precision()
def evaluate_loss(network: nn.Module, loader: DataLoader, task: Task, device: str | torch.device) -> float:
    """Average loss per sequence over a dataset, without updating the network

    Parameters
    ----------
    network : torch.nn.Module
        The network to evaluate
    loader : DataLoader
        Batches of the dataset to evaluate on
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        Total loss divided by the number of sequences
    """

    criterion = build_loss_criterion(task)
    batch_losses = []
    n_sequences = 0

    network.eval()
    with torch.no_grad():
        for batch in loader:
            loss, _ = compute_batch_loss(network, batch, task, criterion, device)
            batch_losses.append(loss.detach())
            n_sequences += len(batch[0])
    total_loss = sum_batch_losses(batch_losses)

    return total_loss / n_sequences


# .................................................................
# Training

def plateau_reached(val_loss: float, recent_val_losses: list[float], epochs_since_best: int, patience: int) -> bool:
    """Decide whether training with --stop auto should end

    Training ends when the current validation loss is not meaningfully lower
    than every one of the last `patience` epochs, and the best validation loss
    is more than `patience` epochs old.

    Parameters
    ----------
    val_loss : float
        Validation loss of the current epoch
    recent_val_losses : list of float
        Validation losses of the previous `patience` epochs
    epochs_since_best : int
        Number of epochs since the best validation loss so far was reached
    patience : int
        The number of epochs over which performance must have stagnated

    Returns
    -------
    bool
        True if training should stop
    """

    still_improving = all(val_loss < previous * (1 - AUTO_STOP_RELATIVE_IMPROVEMENT)
                          for previous in recent_val_losses)
    return not still_improving and epochs_since_best > patience


@full_float32_precision()
def train(network: nn.Module, train_loader: DataLoader, val_loader: DataLoader, task: Task,
          device: str | torch.device, learn_rate: float, n_epochs: int, stop_condition: str = 'iter',
          on_improvement: Callable[[nn.Module], None] | None = None, verbose: bool = False,
          silent: bool = False) -> tuple[list[float], list[float]]:
    """Train a network, keeping the weights that perform best on the validation set

    Every epoch the network is trained on the training set and then evaluated
    on the validation set. When training finishes, the network holds the
    weights from the epoch with the lowest validation loss.

    Models that are not trained by gradient descent (see architectures/classical.py)
    are instead fitted in one step; the learning rate, number of epochs and stop
    condition are not used, and the returned lists each hold a single loss.

    Parameters
    ----------
    network : torch.nn.Module
        The network to train (already on `device`)
    train_loader : DataLoader
        Batches of the training set (should shuffle)
    val_loader : DataLoader
        Batches of the validation set
    task : Task
        The task being trained
    device : str or torch.device
        Device the network lives on
    learn_rate : float
        Initial learning rate for the Adam optimiser
    n_epochs : int
        With stop_condition='iter', the number of epochs to train for. With
        stop_condition='auto', train for at least this many epochs and stop once
        performance has stagnated for this many epochs.
    stop_condition : str, optional
        'iter' (default) or 'auto'
    on_improvement : callable, optional
        Called with the network every time the validation loss reaches a new
        minimum, e.g. to save it to disk so an interrupted run is not lost.
    verbose : bool, optional
        If True, print the validation loss every epoch rather than every 5 epochs
    silent : bool, optional
        If True, print nothing

    Returns
    -------
    list of float
        Average training loss per sequence, for each epoch
    list of float
        Average validation loss per sequence, for each epoch
    """

    if stop_condition not in ('iter', 'auto'):
        raise ValueError(f"Invalid stop condition '{stop_condition}': must be 'auto' or 'iter'.")

    if isinstance(network, ClassicalModel):
        return fit_in_one_step(network, train_loader, val_loader, task, device, on_improvement, silent)

    optimizer = torch.optim.Adam(network.parameters(), lr=learn_rate)
    criterion = build_loss_criterion(task)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = np.inf
    best_epoch = 0
    best_weights = None

    # With 'auto' there is no fixed number of epochs: training ends once performance plateaus
    epochs = range(n_epochs) if stop_condition == 'iter' else itertools.count()

    for epoch in epochs:
        train_loss = train_epoch(network, train_loader, task, criterion, optimizer, device)
        val_loss = evaluate_loss(network, val_loader, task, device)

        stop_now = (stop_condition == 'auto' and epoch >= n_epochs
                    and plateau_reached(val_loss, val_losses[-n_epochs:], epoch - best_epoch, n_epochs))

        # Keep a copy of the best weights seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
            if on_improvement is not None:
                on_improvement(network)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if not silent and (verbose or epoch % PRINT_EVERY_N_EPOCHS == 0):
            print('Epoch %d\tLoss %.4f' % (epoch, val_loss))

        if stop_now:
            break

    if best_weights is not None:
        network.load_state_dict(best_weights)

    return train_losses, val_losses


@full_float32_precision()
def train_epoch(network: nn.Module, loader: DataLoader, task: Task, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str | torch.device) -> float:
    """Train a network for one epoch (one pass over the training data)

    Parameters
    ----------
    network : torch.nn.Module
        The network to train (already on `device`)
    loader : DataLoader
        Batches of the training set
    task : Task
        The task being trained
    criterion : torch.nn.Module
        The loss function returned by build_loss_criterion()
    optimizer : torch.optim.Optimizer
        Optimizer of the network's parameters
    device : str or torch.device
        Device the network lives on

    Returns
    -------
    float
        Average training loss per sequence
    """

    network.train()
    batch_losses = []
    n_train_sequences = 0
    for batch in loader:
        loss, _ = compute_batch_loss(network, batch, task, criterion, device)
        batch_losses.append(loss.detach())
        n_train_sequences += len(batch[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_batch_losses(batch_losses) / n_train_sequences


def fit_in_one_step(network: ClassicalModel, train_loader: DataLoader, val_loader: DataLoader, task: Task,
                    device: str | torch.device, on_improvement: Callable[[nn.Module], None] | None = None,
                    silent: bool = False) -> tuple[list[float], list[float]]:
    """Fit a model that is not trained by gradient descent, and report its losses

    Parameters
    ----------
    network : ClassicalModel
        The model to fit
    train_loader : DataLoader
        Batches of the training set
    val_loader : DataLoader
        Batches of the validation set
    task : Task
        The task being trained
    device : str or torch.device
        Device the model's outputs are placed on
    on_improvement : callable, optional
        Called once with the fitted model (e.g. to save it)
    silent : bool, optional
        If True, print nothing

    Returns
    -------
    list of float
        The average training loss per sequence (a single value)
    list of float
        The average validation loss per sequence (a single value)
    """

    network.fit(train_loader)
    train_loss = evaluate_loss(network, train_loader, task, device)
    val_loss = evaluate_loss(network, val_loader, task, device)

    if on_improvement is not None:
        on_improvement(network)

    if not silent:
        print('Fitted in one step\tLoss %.4f' % val_loss)

    return [train_loss], [val_loss]


# .................................................................
# Evaluation and prediction

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
        [num_classes] per sequence, or [sequence_length X num_classes] per
        sequence with the padding removed
    """

    outputs_array = outputs.cpu().numpy()
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
            outputs_in_loader_order.extend(unpad_outputs(batch_outputs, batch[3], task))

    outputs: list[np.ndarray] = [np.empty(0)] * len(dataset)
    for index, output in zip(loader_order, outputs_in_loader_order, strict=True):
        outputs[index] = output

    return LabeledPredictions(names=dataset.names, sequences=dataset.sequences,
                              true_values=[np.asarray(sample[2]) for sample in dataset.data],
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

        if print_frequency is not None:
            for index in range(start, start + len(batch_sequences)):
                if index % print_frequency == 0:
                    print(f'On {index} of {len(unique_sequences)}')

        vectors, lengths = pid.pad_sequence_vectors([encoder.encode(seq) for seq in batch_sequences])
        with torch.no_grad():
            outputs = network(vectors.to(device), lengths)

        for sequence, output in zip(batch_sequences, unpad_outputs(outputs, lengths, network.task)):
            predictions[sequence] = output

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
