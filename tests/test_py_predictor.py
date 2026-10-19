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
# Predicting many sequences at once

# Sequences of different lengths, one repeated and one in lowercase, so batches mix lengths and need padding
BATCH_SEQUENCES = ['ACDEFGHIK', 'mkvlaagivg', 'W', 'ACDEFGHIK', 'QRSTVWYQRSTVWYQRSTVWY', 'GSGSGS']


@pytest.mark.parametrize('task', [Task('sequence', 1), Task('sequence', 3), Task('residues', 1), Task('residues', 4)],
                         ids=lambda task: f'{task.datatype}-{task.num_classes}')
def test_predict_batch_gives_what_predict_gives_each_sequence(tmp_path, task):
    _, _, path = save_checkpoint(tmp_path, task)
    predictor = py_predictor.Predictor(path)

    batch = predictor.predict_batch(BATCH_SEQUENCES, batch_size=2)

    assert len(batch) == len(BATCH_SEQUENCES)
    for sequence, predicted in zip(BATCH_SEQUENCES, batch):
        single = predictor.predict(sequence)
        assert predicted.shape == single.shape
        np.testing.assert_allclose(predicted, single, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('architecture', ARCHITECTURES_UNDER_TEST)
def test_predict_batch_every_architecture(tmp_path, test_architecture, architecture):
    task = Task('residues', 1) if supports(architecture, 'residues') else Task('sequence', 1)
    network = ready_network(architecture, task).eval()
    path = str(tmp_path / 'network.pt')
    save_network(path, network, get_encoder('onehot'))
    predictor = py_predictor.Predictor(path)

    for sequence, predicted in zip(BATCH_SEQUENCES, predictor.predict_batch(BATCH_SEQUENCES, batch_size=4)):
        np.testing.assert_allclose(predicted, predictor.predict(sequence), rtol=1e-5, atol=1e-6)


def test_predict_batch_runs_the_network_in_batches_of_distinct_sequences(tmp_path, monkeypatch):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    predictor = py_predictor.Predictor(path)
    batch_sizes = []
    original_forward = type(predictor.network).forward

    def recording_forward(self, x, lengths):
        batch_sizes.append(x.shape[0])
        return original_forward(self, x, lengths)

    monkeypatch.setattr(type(predictor.network), 'forward', recording_forward)
    predictor.predict_batch(BATCH_SEQUENCES, batch_size=2)

    # Six sequences, of which five are distinct: batches of 2, 2 and 1
    assert batch_sizes == [2, 2, 1]


def test_predict_batch_of_no_sequences_is_empty(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    predictor = py_predictor.Predictor(path)
    assert predictor.predict_batch([]) == []
    # Its other arguments are still checked (they used to be skipped when there were no sequences)
    with pytest.raises(ValueError, match='full_lengths only applies'):
        predictor.predict_batch([], full_lengths=[])
    with pytest.raises(ValueError, match='starts only applies with full_lengths'):
        predictor.predict_batch([], starts=[])


def test_predict_batch_needs_a_list_of_sequences(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(TypeError, match='takes a list of sequences'):
        py_predictor.Predictor(path).predict_batch('ACDEF')


def test_predict_batch_checks_its_batch_size(tmp_path):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(ValueError, match='batch_size must be at least 1'):
        py_predictor.Predictor(path).predict_batch(['ACDEF'], batch_size=0)


def test_predict_batch_checks_every_sequence_before_predicting(tmp_path, monkeypatch):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    predictor = py_predictor.Predictor(path)
    monkeypatch.setattr(py_predictor.prediction, 'predict_sequences',
                        lambda *args, **kwargs: pytest.fail('predicted anyway'))

    with pytest.raises(ValueError, match='Invalid amino acid'):
        predictor.predict_batch(['ACDEF', 'ACDXF'])


@pytest.mark.parametrize('arguments, message', [
    ({'full_lengths': ['AACDEFAA']}, 'full_lengths only applies to networks trained with --embedding-context'),
    ({'starts': [3]}, 'starts only applies with full_lengths'),
])
def test_predict_batch_refuses_contexts_for_a_network_trained_without_them(tmp_path, arguments, message):
    _, _, path = save_checkpoint(tmp_path, Task('sequence', 1))
    with pytest.raises(ValueError, match=message):
        py_predictor.Predictor(path).predict_batch(['ACDEF'], **arguments)


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
