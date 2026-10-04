"""
Tests for parrot.train_network: loss construction and masking, the training loop,
evaluation on labeled data and prediction on unlabeled data.
"""

import numpy as np
import pytest
import torch

from parrot import encode_sequence
from parrot import process_input_data as pid
from parrot import train_network
from parrot.architectures import ParrotArchitecture, build_network
from parrot.tasks import Task

from tests.helpers import (ARCHITECTURES_UNDER_TEST, EXAMPLE_TASKS, architecture_class, data_path, make_network,
                           ready_network, supports)

TASK_NAMES = list(EXAMPLE_TASKS.keys())

# Every (architecture, example task) combination the architecture supports
ARCHITECTURE_TASKS = [(architecture, task_name) for architecture in ARCHITECTURES_UNDER_TEST
                      for task_name in TASK_NAMES if supports(architecture, EXAMPLE_TASKS[task_name][1])]


def load_task(task_name: str) -> tuple[pid.SequenceDataset, pid.SequenceDataset, pid.SequenceDataset, Task]:
    """Load and split one of the example datasets into (train, val, test, task)."""
    filename, datatype, num_classes = EXAMPLE_TASKS[task_name]
    task = Task(datatype, num_classes)
    train_set, val_set, test_set = pid.split_data(data_path(filename), task, ignore_warnings=True)
    return train_set, val_set, test_set, task


def loader(dataset: pid.SequenceDataset, task: Task, batch_size: int = 32,
           shuffle: bool = False) -> torch.utils.data.DataLoader:
    return pid.make_dataloader(dataset, task, batch_size, shuffle=shuffle)


def train_briefly(task_name: str, architecture: str = 'brnn',
                  n_epochs: int = 3) -> tuple[ParrotArchitecture, pid.SequenceDataset, Task]:
    """Train a small network on an example task; return (network, test set, task)."""
    train_set, val_set, test_set, task = load_task(task_name)
    network = build_network(architecture, 20, task)
    train_network.train(network, loader(train_set, task, shuffle=True), loader(val_set, task), task, 'cpu',
                        learn_rate=0.01, n_epochs=n_epochs, silent=True)
    return network, test_set, task


# ..........................................................................................
# Loss construction and masking

def test_build_loss_criterion():
    assert isinstance(train_network.build_loss_criterion(Task('residues', 1)), torch.nn.MSELoss)
    assert isinstance(train_network.build_loss_criterion(Task('sequence', 1)), torch.nn.L1Loss)

    criterion = train_network.build_loss_criterion(Task('residues', 3))
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.ignore_index == pid.IGNORED_CLASS_LABEL
    assert criterion.reduction == 'sum'


@pytest.mark.parametrize('architecture, task_name', ARCHITECTURE_TASKS)
def test_batch_loss_equals_sum_of_single_sequence_losses(test_architecture, architecture, task_name):
    """Padding must not change the loss: a padded batch gives the same total as each sequence alone"""
    train_set, _, _, task = load_task(task_name)
    network = ready_network(architecture, task)
    criterion = train_network.build_loss_criterion(task)
    samples = [train_set[i] for i in range(10)]

    with torch.no_grad():
        batch_loss, _ = train_network.compute_batch_loss(network, pid.collate_batch(samples, task), task,
                                                         criterion, 'cpu')
        single_losses = [train_network.compute_batch_loss(network, pid.collate_batch([sample], task), task,
                                                          criterion, 'cpu')[0].item()
                         for sample in samples]

    # The sequences in this batch really do have different lengths
    assert len(set(len(sample[1]) for sample in samples)) > 1
    assert batch_loss.item() == pytest.approx(sum(single_losses), rel=1e-5)


def test_residue_regression_loss_ignores_padded_targets():
    task = Task('residues', 1)
    network = make_network('residues', 1)
    criterion = train_network.build_loss_criterion(task)
    vectors, lengths = pid.pad_sequence_vectors([torch.ones(3, 20), torch.ones(5, 20)])
    targets = torch.zeros(2, vectors.shape[1], 1)

    with torch.no_grad():
        loss_a, _ = train_network.compute_batch_loss(network, ([], vectors, targets, lengths), task, criterion, 'cpu')
        targets[0, 3:] = 1000.0  # junk in the padded positions of sequence 0
        loss_b, _ = train_network.compute_batch_loss(network, ([], vectors, targets, lengths), task, criterion, 'cpu')

    assert loss_a.item() == pytest.approx(loss_b.item())


@pytest.mark.parametrize('num_classes', [1, 3])
def test_residue_loss_ignores_missing_labels(num_classes):
    """A residue with a missing label contributes nothing, whatever the network predicts there"""
    task = Task('residues', num_classes)
    network = make_network('residues', num_classes)
    criterion = train_network.build_loss_criterion(task)
    missing = np.nan if num_classes == 1 else pid.IGNORED_CLASS_LABEL
    present = 0.5 if num_classes == 1 else 1

    def loss_for(labels):
        sample = ('s', encode_sequence.one_hot('ACDE'), labels)
        with torch.no_grad():
            return train_network.compute_batch_loss(network, pid.collate_batch([sample], task), task, criterion,
                                                    'cpu')[0].item()

    with_missing = loss_for([present, missing, present, missing])
    assert np.isfinite(with_missing)

    # The loss equals the sum of the losses at the two labelled residues alone
    first_only = loss_for([present, missing, missing, missing])
    third_only = loss_for([missing, missing, present, missing])
    assert with_missing == pytest.approx(first_only + third_only, rel=1e-5)


# ..........................................................................................
# Training loop

def test_train_returns_losses_and_keeps_best_weights():
    train_set, val_set, _, task = load_task('seq_regress')
    network = make_network('sequence', 1)
    improvements = []

    train_losses, val_losses = train_network.train(network, loader(train_set, task, shuffle=True),
                                                   loader(val_set, task), task, 'cpu', learn_rate=0.01, n_epochs=6,
                                                   on_improvement=lambda net: improvements.append(1), silent=True)

    assert len(train_losses) == 6 and len(val_losses) == 6
    assert all(np.isfinite(train_losses)) and all(np.isfinite(val_losses))

    # on_improvement is called once per new best, and the network ends with the best weights
    assert len(improvements) == sum(val_losses[i] < min(val_losses[:i], default=np.inf) for i in range(6))
    assert train_network.evaluate_loss(network, loader(val_set, task), task, 'cpu') == pytest.approx(min(val_losses),
                                                                                                    rel=1e-5)


GRADIENT_ARCHITECTURE_TASKS = [(architecture, task_name) for architecture, task_name in ARCHITECTURE_TASKS
                               if architecture_class(architecture).trained_by_gradient_descent]
ONE_STEP_ARCHITECTURE_TASKS = [(architecture, task_name) for architecture, task_name in ARCHITECTURE_TASKS
                               if not architecture_class(architecture).trained_by_gradient_descent]


@pytest.mark.parametrize('architecture, task_name', GRADIENT_ARCHITECTURE_TASKS)
def test_training_reduces_training_loss(test_architecture, architecture, task_name):
    train_set, val_set, _, task = load_task(task_name)
    network = build_network(architecture, 20, task)

    train_losses, _ = train_network.train(network, loader(train_set, task, 16, shuffle=True), loader(val_set, task),
                                          task, 'cpu', learn_rate=0.01, n_epochs=8, silent=True)

    assert train_losses[-1] < train_losses[0]


@pytest.mark.parametrize('architecture, task_name', ONE_STEP_ARCHITECTURE_TASKS)
def test_one_step_models_are_fitted_by_train(architecture, task_name, capsys):
    """train() fits these models in one step, ignoring the learning rate and number of epochs"""
    train_set, val_set, _, task = load_task(task_name)
    network = build_network(architecture, 20, task)
    saved = []

    train_losses, val_losses = train_network.train(network, loader(train_set, task), loader(val_set, task), task,
                                                   'cpu', learn_rate=0.01, n_epochs=8,
                                                   on_improvement=lambda net: saved.append(net))

    assert len(train_losses) == 1 and len(val_losses) == 1
    assert np.isfinite(train_losses[0]) and np.isfinite(val_losses[0])
    assert saved == [network]
    assert 'Fitted in one step' in capsys.readouterr().out
    assert train_network.evaluate_loss(network, loader(val_set, task), task, 'cpu') == pytest.approx(val_losses[0])


def test_train_auto_stops_when_validation_loss_plateaus():
    """With a learning rate of zero the loss never changes, so 'auto' stops after min_epochs + 2 epochs"""
    train_set, val_set, _, task = load_task('seq_class')
    min_epochs = 3

    _, val_losses = train_network.train(make_network('sequence', 3), loader(train_set, task), loader(val_set, task),
                                        task, 'cpu', learn_rate=0.0, n_epochs=min_epochs, stop_condition='auto',
                                        silent=True)

    assert len(val_losses) == min_epochs + 2


def test_train_rejects_unknown_stop_condition():
    train_set, val_set, _, task = load_task('seq_class')
    with pytest.raises(ValueError, match='stop condition'):
        train_network.train(make_network('sequence', 3), loader(train_set, task), loader(val_set, task), task, 'cpu',
                            learn_rate=0.01, n_epochs=1, stop_condition='never')


def test_plateau_reached():
    assert train_network.plateau_reached(1.0, [1.0, 1.0], epochs_since_best=3, patience=2)
    assert not train_network.plateau_reached(1.0, [1.0, 1.0], epochs_since_best=2, patience=2)
    assert not train_network.plateau_reached(0.5, [1.0, 1.0], epochs_since_best=3, patience=2)


def test_train_prints_progress(capsys):
    train_set, val_set, _, task = load_task('seq_class')

    def run(**kwargs):
        train_network.train(make_network('sequence', 3), loader(train_set, task), loader(val_set, task), task, 'cpu',
                            learn_rate=0.01, n_epochs=6, **kwargs)
        return capsys.readouterr().out

    assert run(silent=True) == ''
    assert run().count('Epoch') == 2  # epochs 0 and 5
    assert run(verbose=True).count('Epoch') == 6


# ..........................................................................................
# Evaluation on labeled data

@pytest.mark.parametrize('task_name', TASK_NAMES)
def test_predict_labeled(task_name):
    network, test_set, task = train_briefly(task_name)

    labeled = train_network.predict_labeled(network, test_set, task, 'cpu')

    assert np.isfinite(labeled.loss)
    assert labeled.names == test_set.names and labeled.sequences == test_set.sequences
    for sequence, true_value, output in zip(labeled.sequences, labeled.true_values, labeled.outputs):
        if task.per_residue:
            assert true_value.shape == (len(sequence),)
            assert output.shape == (len(sequence), task.num_classes)
        else:
            assert true_value.shape == ()
            assert output.shape == (task.num_classes,)


@pytest.mark.parametrize('task_name', TASK_NAMES)
def test_predict_labeled_returns_outputs_in_dataset_order(task_name):
    # Batches are sorted by length, so outputs must be put back in dataset order
    network, test_set, task = train_briefly(task_name)
    labeled = train_network.predict_labeled(network, test_set, task, 'cpu', batch_size=4)

    for (_, sequence, _), output in zip(test_set.data, labeled.outputs):
        with torch.no_grad():
            expected = network(test_set.encoder.encode(sequence).float().unsqueeze(0))[0].numpy()
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('task_name', TASK_NAMES)
def test_predict_labeled_is_independent_of_batch_size(task_name):
    network, test_set, task = train_briefly(task_name)

    one_at_a_time = train_network.predict_labeled(network, test_set, task, 'cpu', batch_size=1)
    batched = train_network.predict_labeled(network, test_set, task, 'cpu', batch_size=16)

    assert one_at_a_time.loss == pytest.approx(batched.loss, rel=1e-5)
    for single, batch in zip(one_at_a_time.outputs, batched.outputs):
        np.testing.assert_allclose(single, batch, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('task, output, probabilistic, expected', [
    (Task('sequence', 1), np.array([0.5]), False, 0.5),
    (Task('residues', 1), np.array([[0.5], [1.5]]), False, [0.5, 1.5]),
    (Task('sequence', 3), np.array([0.1, 2.0, 0.3]), False, 1),
    (Task('residues', 2), np.array([[2.0, 0.1], [0.1, 2.0]]), False, [0, 1]),
    (Task('sequence', 2), np.array([0.0, 0.0]), True, [0.5, 0.5]),
])
def test_output_to_prediction(task, output, probabilistic, expected):
    np.testing.assert_allclose(train_network.output_to_prediction(output, task, probabilistic), expected)


# ..........................................................................................
# Predicting on unlabeled data

PREDICTION_CASES = [(architecture, task) for architecture in ARCHITECTURES_UNDER_TEST
                    for task in [Task('sequence', 1), Task('sequence', 3), Task('residues', 1), Task('residues', 2)]
                    if supports(architecture, task.datatype)]


@pytest.mark.parametrize('architecture, task', PREDICTION_CASES,
                         ids=[f'{architecture}-{task.datatype}-{task.num_classes}' for architecture, task in PREDICTION_CASES])
def test_predict_sequences(test_architecture, architecture, task):
    network = ready_network(architecture, task)
    encoder = encode_sequence.OneHotEncoder()
    sequences = ['ACDEFG', 'KLMNPQRS', 'ACDEFG', 'W']

    predictions = train_network.predict_sequences(network, sequences, encoder, 'cpu', batch_size=2)

    assert list(predictions.keys()) == ['ACDEFG', 'KLMNPQRS', 'W']
    for sequence, output in predictions.items():
        with torch.no_grad():
            expected = network(encoder.encode(sequence).float().unsqueeze(0))[0].numpy()
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_predict_sequences_with_biophysics_encoding():
    network = make_network('sequence', 1, input_size=9)
    predictions = train_network.predict_sequences(network, ['ACDEF'], encode_sequence.BiophysicsEncoder(), 'cpu')
    assert predictions['ACDEF'].shape == (1,)


def test_predict_sequences_print_frequency(capsys):
    train_network.predict_sequences(make_network('sequence', 1), ['ACD', 'EFG', 'HIK'],
                                    encode_sequence.OneHotEncoder(), 'cpu', batch_size=2, print_frequency=2)
    assert capsys.readouterr().out.splitlines() == ['On 0 of 3', 'On 2 of 3']
