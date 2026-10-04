"""
Tests for parrot.architectures.classical: the simple models fitted in one step
(linear, svm, random_forest, knn).
"""

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr

from parrot import process_input_data as pid
from parrot import train_network
from parrot.architectures import build_network
from parrot.architectures.classical import MINIMUM_PROBABILITY, ClassicalModel
from parrot.architectures.layers import adjacent_pair_composition, mean_over_residues
from parrot.tasks import Task
from parrot.tools import cli

from tests.helpers import data_path, encode_batch, ready_network

ONE_STEP_ARCHITECTURES = ['linear', 'svm', 'random_forest', 'knn']



def example_loaders(filename: str, task: Task) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Training and test loaders for an example dataset."""
    train_set, _, test_set = pid.split_data(data_path(filename), task, ignore_warnings=True)
    return pid.make_dataloader(train_set, task, 32), pid.make_dataloader(test_set, task, 32)


# ..........................................................................................
# Features

def test_mean_over_residues_is_composition():
    batch, lengths = encode_batch(['AAC', 'CW'])
    composition = mean_over_residues(batch, lengths)

    assert composition[0, 0] == pytest.approx(2 / 3) and composition[0, 1] == pytest.approx(1 / 3)
    assert composition[1, 1] == pytest.approx(0.5) and composition[1, 18] == pytest.approx(0.5)
    torch.testing.assert_close(composition.sum(dim=1), torch.ones(2))


def test_adjacent_pair_composition_is_dipeptide_composition():
    batch, lengths = encode_batch(['AAC', 'W'])
    pairs = adjacent_pair_composition(batch, lengths).view(2, 20, 20)

    # 'AAC' has two dipeptides, AA and AC; 'W' has none
    assert pairs[0, 0, 0] == pytest.approx(0.5) and pairs[0, 0, 1] == pytest.approx(0.5)
    assert pairs[0].sum() == pytest.approx(1.0)
    assert torch.all(pairs[1] == 0)


@pytest.mark.parametrize('feature_set, expected_size', [('composition', 21), ('dipeptides', 421)])
def test_sequence_features(feature_set, expected_size):
    network = build_network('linear', 20, Task('sequence', 1), {'features': feature_set})
    batch, lengths = encode_batch(['AAC', 'CWKLM'])

    features = network.sequence_features(batch, lengths)

    assert features.shape == (2, expected_size)
    torch.testing.assert_close(features[:, :20], mean_over_residues(batch, lengths))
    torch.testing.assert_close(features[:, 20], torch.log(torch.tensor([3.0, 5.0])))


# ..........................................................................................
# Fitting and predicting

def test_linear_model_recovers_composition_target():
    """The example regression target is 10 x the hydrophobic fraction, which is linear in composition"""
    task = Task('sequence', 1)
    train_loader, test_loader = example_loaders('seq_regress_dataset.tsv', task)
    network = build_network('linear', 20, task, {'alpha': 1e-3})

    network.fit(train_loader)
    labeled = train_network.predict_labeled(network, test_loader.dataset, task, 'cpu')

    predicted = np.array([output[0] for output in labeled.outputs])
    assert pearsonr(np.array(labeled.true_values), predicted)[0] > 0.99


@pytest.mark.parametrize('architecture', ONE_STEP_ARCHITECTURES)
def test_classifiers_learn_the_example_classes(architecture):
    task = Task('sequence', 3)
    train_loader, test_loader = example_loaders('seq_class_dataset.tsv', task)
    network = build_network(architecture, 20, task)

    network.fit(train_loader)
    labeled = train_network.predict_labeled(network, test_loader.dataset, task, 'cpu')

    accuracy = np.mean([np.argmax(output) == true for output, true in zip(labeled.outputs, labeled.true_values)])
    assert accuracy > 0.8


@pytest.mark.parametrize('architecture', ONE_STEP_ARCHITECTURES)
def test_classification_outputs_are_log_probabilities(architecture):
    network = ready_network(architecture, Task('sequence', 3))
    batch, lengths = encode_batch(['ACDEF', 'KKKRR', 'W'])

    with torch.no_grad():
        probabilities = torch.exp(network(batch, lengths))

    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3), rtol=1e-4, atol=1e-4)


def test_classes_missing_from_training_get_small_probability():
    task = Task('sequence', 3)
    data = [[f's{i}', sequence, i % 2] for i, sequence in enumerate(['AAAA', 'KKKK', 'AAAC', 'KKKR', 'AACA', 'KKRK'])]
    network = build_network('linear', 20, task)
    network.fit(pid.make_dataloader(pid.SequenceDataset(data), task, batch_size=4))

    with torch.no_grad():
        log_probabilities = network(*encode_batch(['AAAA']))

    assert torch.isfinite(log_probabilities).all()
    assert log_probabilities[0, 2] == pytest.approx(np.log(MINIMUM_PROBABILITY))


@pytest.mark.parametrize('architecture', ONE_STEP_ARCHITECTURES)
def test_unfitted_model_cannot_predict(architecture):
    network = build_network(architecture, 20, Task('sequence', 1))
    with pytest.raises(RuntimeError, match='has not been fitted'):
        network(*encode_batch(['ACDEF']))


@pytest.mark.parametrize('architecture', ['svm', 'random_forest'])
def test_fitting_is_reproducible_with_seed(architecture):
    task = Task('sequence', 3)
    train_loader, _ = example_loaders('seq_class_dataset.tsv', task)
    batch, lengths = encode_batch(['ACDEFGHIK', 'KKRRKDE'])
    outputs = []
    for _ in range(2):
        cli.set_random_seed(5)
        network = build_network(architecture, 20, task)
        network.fit(train_loader)
        with torch.no_grad():
            outputs.append(network(batch, lengths))

    torch.testing.assert_close(outputs[0], outputs[1])


def test_fitted_model_survives_state_dict_round_trip():
    network = ready_network('random_forest', Task('sequence', 1))
    copy = build_network('random_forest', 20, Task('sequence', 1))
    copy.load_state_dict(network.state_dict())

    batch, lengths = encode_batch(['ACDEF', 'W'])
    with torch.no_grad():
        torch.testing.assert_close(copy(batch, lengths), network(batch, lengths))


def test_one_step_models_have_no_parameters_to_train():
    network = build_network('svm', 20, Task('sequence', 1))
    assert isinstance(network, ClassicalModel)
    assert not network.trained_by_gradient_descent
    assert list(network.parameters()) == []


# ..........................................................................................
# Input checks

@pytest.mark.parametrize('architecture, values', [('linear', {'alpha': 0.0}), ('svm', {'svm_c': 0.0}),
                                                  ('svm', {'relative_gamma': -1.0})])
def test_positive_hyperparameters(architecture, values):
    with pytest.raises(ValueError, match='greater than 0'):
        build_network(architecture, 20, Task('sequence', 1), values)


def test_knn_needs_enough_training_sequences():
    task = Task('sequence', 1)
    data = [[f's{i}', 'ACDEF'[: i + 1], float(i)] for i in range(4)]
    network = build_network('knn', 20, task, {'num_neighbors': 10})

    with pytest.raises(ValueError, match='larger than the number of training sequences'):
        network.fit(pid.make_dataloader(pid.SequenceDataset(data), task, batch_size=4))


@pytest.mark.parametrize('architecture', ONE_STEP_ARCHITECTURES)
def test_residue_data_is_not_supported(architecture):
    with pytest.raises(ValueError, match='only supports sequence-mapped data'):
        build_network(architecture, 20, Task('residues', 1))
