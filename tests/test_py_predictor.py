"""
Tests for parrot.py_predictor: loading saved networks into Python workflows.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from parrot import encode_sequence, py_predictor
from parrot.architectures import build_network
from parrot.checkpoint import save_network
from parrot.encode_sequence import SequenceEncoder, get_encoder
from parrot.tasks import Task

from tests.helpers import ARCHITECTURES_UNDER_TEST, make_network, ready_network, supports


def save_checkpoint(tmp_path: Path, task: Task, encoding: str = 'onehot', architecture: str = 'brnn',
                    hyperparameters: dict | None = None) -> tuple[torch.nn.Module, SequenceEncoder, str]:
    """Save a small random network as a self-describing checkpoint; return (network, encoder, path)."""
    encoder = get_encoder(encoding)
    network = build_network(architecture, len(encoder), task, hyperparameters)
    path = str(tmp_path / 'network.pt')
    save_network(path, network, encoder)
    return network, encoder, path


def save_legacy_weights(tmp_path: Path, datatype: str, num_classes: int, input_size: int = 20, hidden_size: int = 7,
                        num_layers: int = 2) -> tuple[torch.nn.Module, str]:
    """Save only the weights of a small BRNN, as earlier versions of PARROT did; return (network, path)."""
    network = make_network(datatype, num_classes, input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers)
    path = str(tmp_path / 'legacy.pt')
    torch.save(network.state_dict(), path)
    return network, path


def raw_output(network: torch.nn.Module, encoder: SequenceEncoder, sequence: str) -> np.ndarray:
    with torch.no_grad():
        return network(encoder.encode(sequence).float().unsqueeze(0))[0].numpy()


# ..........................................................................................
# Networks saved by this version of PARROT

@pytest.mark.parametrize('task', [Task('sequence', 1), Task('sequence', 3), Task('residues', 1), Task('residues', 4)],
                         ids=lambda task: f'{task.datatype}-{task.num_classes}')
def test_predictor_attributes(tmp_path, task):
    _, _, path = save_checkpoint(tmp_path, task, hyperparameters={'hidden_size': 5, 'num_layers': 3})
    predictor = py_predictor.Predictor(path)

    assert predictor.dtype == task.datatype
    assert predictor.n_classes == task.num_classes
    assert predictor.task == task.problem_type
    assert predictor.num_layers == 3 and predictor.hidden_vector_size == 5


def test_predictor_sequence_regression(tmp_path):
    network, encoder, path = save_checkpoint(tmp_path, Task('sequence', 1))
    prediction = py_predictor.Predictor(path).predict('ACDEFGHIK')

    assert prediction.shape == (1,)
    np.testing.assert_allclose(prediction, raw_output(network, encoder, 'ACDEFGHIK'), rtol=1e-5)


def test_predictor_sequence_classification_returns_probabilities(tmp_path):
    network, encoder, path = save_checkpoint(tmp_path, Task('sequence', 3))
    prediction = py_predictor.Predictor(path).predict('ACDEFGHIK')

    assert prediction.shape == (3,)
    assert prediction.sum() == pytest.approx(1.0)
    assert np.argmax(prediction) == np.argmax(raw_output(network, encoder, 'ACDEFGHIK'))


def test_predictor_residue_regression(tmp_path):
    network, encoder, path = save_checkpoint(tmp_path, Task('residues', 1))
    prediction = py_predictor.Predictor(path).predict('ACDEFG')

    assert prediction.shape == (6,)
    np.testing.assert_allclose(prediction, raw_output(network, encoder, 'ACDEFG').flatten(), rtol=1e-5)


def test_predictor_residue_classification_returns_probabilities(tmp_path):
    network, encoder, path = save_checkpoint(tmp_path, Task('residues', 4))
    prediction = py_predictor.Predictor(path).predict('ACDEFG')

    assert prediction.shape == (6, 4)
    np.testing.assert_allclose(prediction.sum(axis=1), np.ones(6), rtol=1e-6)
    np.testing.assert_array_equal(np.argmax(prediction, axis=1),
                                  np.argmax(raw_output(network, encoder, 'ACDEFG'), axis=1))


def test_predictor_biophysics_network(tmp_path):
    network, encoder, path = save_checkpoint(tmp_path, Task('sequence', 1), encoding='biophysics')
    prediction = py_predictor.Predictor(path).predict('ACDEF')
    np.testing.assert_allclose(prediction, raw_output(network, encoder, 'ACDEF'), rtol=1e-5)


@pytest.mark.parametrize('architecture', ARCHITECTURES_UNDER_TEST)
def test_predictor_every_architecture(tmp_path, test_architecture, architecture):
    task = Task('residues', 1) if supports(architecture, 'residues') else Task('sequence', 1)
    network = ready_network(architecture, task).eval()
    encoder = get_encoder('onehot')
    path = str(tmp_path / 'network.pt')
    save_network(path, network, encoder)

    prediction = py_predictor.Predictor(path).predict('ACDEF')
    np.testing.assert_allclose(prediction, raw_output(network, encoder, 'ACDEF').flatten(), rtol=1e-5, atol=1e-6)


def test_predictor_brnn_attributes_are_none_for_other_architectures(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1), architecture='cnn')
    predictor = py_predictor.Predictor(path)
    assert predictor.num_layers is None and predictor.hidden_vector_size is None


def test_predictor_dtype_must_match(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))

    assert py_predictor.Predictor(path, dtype='sequence').dtype == 'sequence'
    with pytest.raises(ValueError, match='does not match'):
        py_predictor.Predictor(path, dtype='residues')


def test_predictor_accepts_lowercase(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    predictor = py_predictor.Predictor(path)
    np.testing.assert_allclose(predictor.predict('acdef'), predictor.predict('ACDEF'))


def test_predictor_rejects_invalid_residue(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(ValueError, match='Invalid amino acid'):
        py_predictor.Predictor(path).predict('ACDXF')


def test_predictor_accepts_the_matrix_dtype(tmp_path):
    # 'matrix' is a datatype a saved network can have, so it is checked against the network like the others
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(ValueError, match="does not match the saved network"):
        py_predictor.Predictor(path, dtype='matrix')


def test_predictor_rejects_invalid_dtype(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(ValueError, match='dtype'):
        py_predictor.Predictor(path, dtype='sequences')


# ..........................................................................................
# Weights-only networks saved by earlier versions of PARROT

@pytest.mark.parametrize('datatype, num_classes, num_layers, hidden_size', [
    ('sequence', 1, 1, 5), ('sequence', 3, 2, 7), ('residues', 1, 3, 4), ('residues', 2, 2, 9),
])
def test_predictor_legacy_network(tmp_path, datatype, num_classes, num_layers, hidden_size):
    network, path = save_legacy_weights(tmp_path, datatype, num_classes, hidden_size=hidden_size,
                                        num_layers=num_layers)
    predictor = py_predictor.Predictor(path, dtype=datatype)

    assert predictor.num_layers == num_layers
    assert predictor.hidden_vector_size == hidden_size
    assert predictor.n_classes == num_classes

    output = raw_output(network, encode_sequence.OneHotEncoder(), 'ACDEFG')
    if num_classes == 1:
        np.testing.assert_allclose(predictor.predict('ACDEFG'), output.flatten(), rtol=1e-5)
    else:
        np.testing.assert_array_equal(np.argmax(predictor.predict('ACDEFG'), axis=-1), np.argmax(output, axis=-1))


def test_predictor_legacy_network_requires_dtype(tmp_path):
    _, path = save_legacy_weights(tmp_path, 'sequence', 1)
    with pytest.raises(ValueError, match='datatype'):
        py_predictor.Predictor(path)


def test_predictor_rejects_non_one_hot_legacy_network(tmp_path):
    _, path = save_legacy_weights(tmp_path, 'sequence', 1, input_size=9)
    with pytest.raises(ValueError, match='length 9'):
        py_predictor.Predictor(path, dtype='sequence')


def test_predictor_softmax_is_numerically_stable():
    probabilities = py_predictor.softmax(np.array([1000.0, 1000.0, -1000.0]))
    np.testing.assert_allclose(probabilities, [0.5, 0.5, 0.0])
