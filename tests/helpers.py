"""
Shared constants and helper functions for the PARROT test suite.
"""

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parrot import encode_sequence
from parrot import hyperparameter_search as hs
from parrot import process_input_data as pid
from parrot.architectures import (HyperParameter, ParrotArchitecture, available_architectures, build_network,
                                  get_architecture)
from parrot.architectures.classical import ClassicalModel
from parrot.architectures.layers import GlobalContext
from parrot.tasks import Task

# The example datasets in the repository's /data folder, and files made just for the tests
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPOSITORY_ROOT, 'data')
TEST_DATA_DIR = os.path.join(REPOSITORY_ROOT, 'tests', 'test_data')

# Seed applied before every test (see conftest.py)
TEST_SEED = 2020

# The four example datasets in /data, one per supported machine-learning task:
# (filename, datatype, number of classes)
EXAMPLE_TASKS: dict[str, tuple[str, str, int]] = {
    'seq_class': ('seq_class_dataset.tsv', 'sequence', 3),
    'seq_regress': ('seq_regress_dataset.tsv', 'sequence', 1),
    'res_class': ('res_class_dataset.tsv', 'residues', 3),
    'res_regress': ('res_regress_dataset.tsv', 'residues', 1),
}


def data_path(filename: str) -> str:
    """Return the absolute path to a file in the /data folder."""
    return os.path.join(DATA_DIR, filename)


def make_network(datatype: str, num_classes: int, input_size: int = 20, hidden_size: int = 7,
                 num_layers: int = 2, attention_pooling: bool = False,
                 loglength_feedforward: bool = False) -> ParrotArchitecture:
    """Build a small, randomly initialised BRNN for testing.

    By default it is built without attention pooling or the log(length) input,
    exactly as earlier versions of PARROT built a BRNN, so its weights can also
    stand in for a network saved by an earlier version.

    Parameters
    ----------
    datatype : str
        'sequence' or 'residues'
    num_classes : int
        Number of output classes (1 for regression)
    input_size : int
        Length of the per-residue encoding vector
    hidden_size : int
        Size of the LSTM hidden vectors
    num_layers : int
        Number of LSTM layers per direction
    attention_pooling : bool
        Sequence tasks: summarise the sequence by attention pooling instead of
        the final hidden states
    loglength_feedforward : bool
        Sequence tasks: give the output layer log(sequence length)

    Returns
    -------
    ParrotArchitecture
        The network, on the CPU
    """
    return build_network('brnn', input_size, Task(datatype, num_classes),
                         {'hidden_size': hidden_size, 'num_layers': num_layers,
                          'attention_pooling': attention_pooling, 'loglength_feedforward': loglength_feedforward})



def encode_batch(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot encode and zero-pad a list of sequences into a batch.

    Parameters
    ----------
    sequences : list of str
        The sequences, in the order they appear in the batch

    Returns
    -------
    torch.Tensor
        The padded float32 batch [batch_dim X padded_length X 20]
    torch.Tensor
        The true length of each sequence
    """
    return pid.pad_sequence_vectors([encode_sequence.one_hot(sequence) for sequence in sequences])

class LinearTestArchitecture(ParrotArchitecture):
    """A tiny architecture used only to check that new architectures plug into PARROT.

    Each residue goes through a small two-layer perceptron. For sequence-mapped
    tasks the per-residue outputs are averaged over the real (unpadded) residues.
    It is deliberately not registered at import time; the
    `test_architecture` fixture in conftest.py registers it for one test.
    """

    name = 'test_linear'
    description = 'test-only per-residue perceptron'
    hyperparameters = (
        HyperParameter('hidden_units', int, default=4, help='number of hidden units', minimum=1,
                       search_range=(2, 8)),
    )

    def __init__(self, input_size: int, task: Task, hidden_units: int) -> None:
        super().__init__(input_size, task, {'hidden_units': hidden_units})
        self.hidden = nn.Linear(input_size, hidden_units)
        self.output = nn.Linear(hidden_units, task.num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        per_residue = self.output(torch.relu(self.hidden(x)))
        if self.task.per_residue:
            return per_residue

        if lengths is None:
            return per_residue.mean(dim=1)

        lengths = lengths.to(x.device)
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        return (per_residue * mask).sum(dim=1) / lengths[:, None]


# Every built-in architecture for sequence or residue data, plus the test-only one. The built-in
# list is read from the registry, so architectures added to PARROT later are covered by the
# shared tests automatically. Tests that use these must request the `test_architecture` fixture.
# Architectures for matrix data have their own shared tests (test_matrix_architectures.py).
BUILT_IN_ARCHITECTURES = [name for name in available_architectures()
                          if set(get_architecture(name).supported_datatypes) & {'sequence', 'residues'}]
ARCHITECTURES_UNDER_TEST = BUILT_IN_ARCHITECTURES + [LinearTestArchitecture.name]

# Every built-in architecture for matrix data
MATRIX_ARCHITECTURES = [name for name in available_architectures()
                        if 'matrix' in get_architecture(name).supported_datatypes]


# Simple, varied sequences used to fit models that are fitted in one step
FITTING_SEQUENCES = ['ACDEFGHIK', 'KKKKRRRDE', 'LLLLIIIVVV', 'DEDEDEDEAA', 'GGSGGSGGS', 'PPPAPPPA', 'W', 'MNQ',
                     'YYYFFFWW', 'STSTSTSTST', 'AKAKAKAKAK', 'EEEEELLLLL']


def architecture_class(architecture: str) -> type[ParrotArchitecture]:
    """Look up an architecture class, including the test-only one (which is not registered at import time)."""
    if architecture == LinearTestArchitecture.name:
        return LinearTestArchitecture
    return get_architecture(architecture)


def supports(architecture: str, datatype: str) -> bool:
    """True if an architecture can be used for data of the given datatype."""
    return datatype in architecture_class(architecture).supported_datatypes


def randomize_global_context(network: nn.Module) -> None:
    """Give every GlobalContext layer of a network random weights (they start at zero)."""
    generator = torch.Generator().manual_seed(0)
    for module in network.modules():
        if isinstance(module, GlobalContext):
            with torch.no_grad():
                module.linear.weight.copy_(torch.randn(module.linear.weight.shape, generator=generator) * 0.5)
                module.linear.bias.copy_(torch.randn(module.linear.bias.shape, generator=generator) * 0.5)


def ready_network(architecture: str, task: Task, hyperparameter_values: dict | None = None,
                  encoding: str = 'onehot') -> ParrotArchitecture:
    """Build a network that can be run straight away.

    Neural networks can be run as soon as they are built. Models fitted in one
    step are fitted here to FITTING_SEQUENCES, with every class represented for
    classification tasks.

    Parameters
    ----------
    architecture : str
        Registry name of the architecture
    task : Task
        The task the network is for
    hyperparameter_values : dict, optional
        Hyperparameter values (defaults for any not given)
    encoding : str, optional
        Encoding the network will be used with ('onehot' or 'biophysics')

    Returns
    -------
    ParrotArchitecture
    """
    encoder = encode_sequence.get_encoder(encoding)
    network = build_network(architecture, len(encoder), task, hyperparameter_values)

    # Whole-sequence summaries start at zero (contributing nothing); give them random weights so that
    # every test exercises them
    randomize_global_context(network)

    if isinstance(network, ClassicalModel):
        targets = [i % task.num_classes if task.is_classification else float(i) for i in range(len(FITTING_SEQUENCES))]
        data = [[f'seq{i}', sequence, target] for i, (sequence, target) in enumerate(zip(FITTING_SEQUENCES, targets))]
        network.fit(pid.make_dataloader(pid.SequenceDataset(data, encoder=encoder), task, batch_size=5))

    return network


# ..........................................................................................
# Hyperparameter search

# The task of the sequence regression example, used by the search tests
SEARCH_TASK = Task('sequence', 1)

# The learning rate FakeTrainer scores configurations against
FAKE_BEST_LEARNING_RATE = 0.003


def make_loaders(n_folds: int | None = None, n_samples: int = 60) -> list[tuple[DataLoader, DataLoader]]:
    """Build (training, validation) loaders from the sequence regression example.

    Parameters
    ----------
    n_folds : int, optional
        If given, one pair of loaders per cross-validation fold; otherwise a
        single pair, with the first 75% of the sequences used for training.
    n_samples : int, optional
        How many sequences of the example to use (the first n_samples)

    Returns
    -------
    list of (DataLoader, DataLoader)
        The training and validation loaders, in batches of 16
    """
    data = pid.parse_file(data_path('seq_regress_dataset.tsv'), SEARCH_TASK, ignore_warnings=True)[:n_samples]
    split = int(0.75 * len(data))
    if n_folds is None:
        sets = [(np.arange(split), np.arange(split, len(data)))]
    else:
        sets = pid.cross_validation_folds(np.arange(len(data)), n_folds)
    return [(pid.make_dataloader(pid.SequenceDataset(data, train), SEARCH_TASK, 16, shuffle=True),
             pid.make_dataloader(pid.SequenceDataset(data, held_out), SEARCH_TASK, 16)) for train, held_out in sets]


class FakeTrainer:
    """A stand-in for hyperparameter_search.TrialTrainer that scores configurations without training.

    The loss of a configuration is how far its learning rate is from
    FAKE_BEST_LEARNING_RATE (in log10 units), plus 1/epochs so that training
    for longer always helps. This makes the search's decisions exactly
    reproducible on any machine.

    Attributes
    ----------
    space : SearchSpace
        The search space being searched
    fail_if : callable
        Called with a trial's hyperparameter values; if it returns True the
        trial fails, as a configuration that does not suit the data would
    calls : list of (int, int)
        Every (trial number, epochs) the search asked for, in order
    """

    def __init__(self, space: hs.SearchSpace, fail_if: Callable[[dict[str, Any]], bool] = lambda values: False) -> None:
        self.space = space
        self.fail_if = fail_if
        self.calls: list[tuple[int, int]] = []

    def train_to(self, trial: hs.Trial, epochs: int) -> float:
        """Pretend to train a trial to `epochs` epochs and return its loss (see the class docstring)."""
        self.calls.append((trial.number, epochs))
        if self.fail_if(trial.values):
            raise ValueError('this configuration does not suit the data')
        trial.folds = [hs.FoldProgress()]
        trial.epochs = epochs
        return abs(math.log10(trial.values['learning_rate'] / FAKE_BEST_LEARNING_RATE)) + 1 / epochs


def make_search(n_trials: int = 30, max_epochs: int = 27, seed: int = 0, **kwargs: Any) -> hs.HyperparameterSearch:
    """Build a silent search of the BRNN's hyperparameters, scored by FakeTrainer unless a `trainer` is given.

    Parameters
    ----------
    n_trials : int, optional
        Number of configurations to try
    max_epochs : int, optional
        Epochs a configuration is trained for at the final stage
    seed : int, optional
        Seed of the search's random number generator
    **kwargs
        `trainer`, plus any other keyword arguments of HyperparameterSearch

    Returns
    -------
    HyperparameterSearch
    """
    space = hs.SearchSpace('brnn')
    return hs.HyperparameterSearch(space, kwargs.pop('trainer', FakeTrainer(space)), n_trials, max_epochs,
                                   rng=np.random.default_rng(seed), silent=True, **kwargs)
