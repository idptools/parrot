"""
Tests for parrot.checkpoint: saving and loading networks, including weights-only
files written by earlier versions of PARROT.
"""

import pytest
import torch

from parrot import encode_sequence
from parrot.architectures import build_network, get_architecture, resolve_hyperparameters
from parrot.checkpoint import infer_legacy_brnn_shape, is_checkpoint, load_network, save_network
from parrot.encode_sequence import get_encoder
from parrot.tasks import Task

from tests.helpers import ARCHITECTURES_UNDER_TEST, data_path, make_network, ready_network, supports


def outputs_for(network: torch.nn.Module, encoder: encode_sequence.SequenceEncoder, sequence: str) -> torch.Tensor:
    with torch.no_grad():
        return network(encoder.encode(sequence).float().unsqueeze(0))


@pytest.mark.parametrize('encoding', ['onehot', 'biophysics', 'encoding_example.txt'])
@pytest.mark.parametrize('task', [Task('sequence', 1), Task('residues', 3)], ids=['seq-regress', 'res-class'])
def test_checkpoint_round_trip(tmp_path, encoding, task):
    encoder = get_encoder(encoding if encoding in ('onehot', 'biophysics') else data_path(encoding))
    network = build_network('brnn', len(encoder), task, {'hidden_size': 6, 'num_layers': 2})
    path = str(tmp_path / 'network.pt')

    save_network(path, network, encoder)
    loaded_network, loaded_encoder = load_network(path)

    assert loaded_network.name == 'brnn'
    assert loaded_network.task == task
    assert loaded_network.hyperparameter_values == {'hidden_size': 6, 'num_layers': 2, 'attention_pooling': True,
                                                    'loglength_feedforward': True}
    assert loaded_encoder.to_config() == encoder.to_config()
    assert not loaded_network.training
    torch.testing.assert_close(outputs_for(loaded_network, loaded_encoder, 'ACDEFGHIK'),
                               outputs_for(network, encoder, 'ACDEFGHIK'))


def test_checkpoint_stores_user_encoding_table(tmp_path):
    """A saved network does not need the original encoding file"""
    encoder = get_encoder(data_path('nucleotide_encoding.txt'))
    network = build_network('brnn', len(encoder), Task('sequence', 1))
    path = str(tmp_path / 'network.pt')

    save_network(path, network, encoder)
    _, loaded_encoder = load_network(path)

    assert loaded_encoder.encode_file is None
    assert loaded_encoder.encode_dict == encoder.encode_dict


def test_checkpoint_contents(tmp_path):
    path = str(tmp_path / 'network.pt')
    save_network(path, make_network('sequence', 2), get_encoder('onehot'))

    saved = torch.load(path)
    assert is_checkpoint(saved)
    assert saved['architecture'] == 'brnn'
    assert saved['task'] == {'datatype': 'sequence', 'num_classes': 2}
    assert saved['encoding'] == {'name': 'onehot'}
    assert isinstance(saved['parrot_version'], str)


@pytest.mark.parametrize('architecture', ARCHITECTURES_UNDER_TEST)
def test_checkpoint_round_trip_for_every_architecture(tmp_path, test_architecture, architecture):
    encoder = get_encoder('onehot')
    task = Task('residues', 3) if supports(architecture, 'residues') else Task('sequence', 3)
    network = ready_network(architecture, task)
    path = str(tmp_path / 'network.pt')

    save_network(path, network, encoder)
    loaded_network, loaded_encoder = load_network(path)

    assert type(loaded_network) is type(network)
    assert loaded_network.hyperparameter_values == network.hyperparameter_values
    torch.testing.assert_close(outputs_for(loaded_network, loaded_encoder, 'ACDEFGHIK'),
                               outputs_for(network.eval(), encoder, 'ACDEFGHIK'))


def test_checkpoint_with_text_hyperparameters(tmp_path):
    network = build_network('unet', 20, Task('sequence', 1), {'upsampling': 'nearest', 'pooling': 'max'})
    path = str(tmp_path / 'network.pt')

    save_network(path, network, get_encoder('onehot'))
    loaded_network, _ = load_network(path)

    assert loaded_network.hyperparameter_values['upsampling'] == 'nearest'
    assert loaded_network.hyperparameter_values['pooling'] == 'max'


# ..........................................................................................
# Weights-only files from earlier versions of PARROT

def save_legacy_weights(tmp_path, datatype: str, num_classes: int, input_size: int = 20) -> tuple[torch.nn.Module, str]:
    network = make_network(datatype, num_classes, input_size=input_size, hidden_size=9, num_layers=3)
    path = str(tmp_path / 'legacy.pt')
    torch.save(network.state_dict(), path)
    return network, path


def test_infer_legacy_brnn_shape(tmp_path):
    network, _ = save_legacy_weights(tmp_path, 'residues', 4)
    assert infer_legacy_brnn_shape(network.state_dict()) == {'input_size': 20, 'hidden_size': 9, 'num_layers': 3,
                                                             'num_classes': 4}


@pytest.mark.parametrize('datatype, num_classes', [('sequence', 1), ('residues', 3)])
def test_load_legacy_weights(tmp_path, datatype, num_classes):
    original, path = save_legacy_weights(tmp_path, datatype, num_classes)

    assert not is_checkpoint(torch.load(path))
    network, encoder = load_network(path, legacy_datatype=datatype)

    assert network.task == Task(datatype, num_classes)
    assert encoder.name == 'onehot'
    torch.testing.assert_close(outputs_for(network, encoder, 'ACDEFG'), outputs_for(original, encoder, 'ACDEFG'))


def test_load_legacy_weights_requires_datatype(tmp_path):
    _, path = save_legacy_weights(tmp_path, 'sequence', 1)
    with pytest.raises(ValueError, match='datatype'):
        load_network(path)


def test_load_legacy_weights_with_other_encoding(tmp_path):
    _, path = save_legacy_weights(tmp_path, 'sequence', 1, input_size=9)

    with pytest.raises(ValueError, match='length 9'):
        load_network(path, legacy_datatype='sequence')

    network, encoder = load_network(path, legacy_datatype='sequence', legacy_encoding='biophysics')
    assert encoder.name == 'biophysics'


@pytest.mark.parametrize('architecture', ['brnn', 'cnn', 'unet', 'transformer'])
@pytest.mark.parametrize('task', [Task('sequence', 2), Task('residues', 1)], ids=['sequence', 'residues'])
def test_networks_saved_before_on_off_options_load_without_them(tmp_path, architecture, task):
    """A checkpoint that records none of the newer on/off hyperparameters is rebuilt with them all off"""
    new_options = {name for name in ('global_context', 'attention_pooling', 'loglength_feedforward')
                   if name in resolve_hyperparameters(get_architecture(architecture))}
    network = build_network(architecture, 20, task, {name: False for name in new_options})
    path = str(tmp_path / 'network.pt')
    save_network(path, network, get_encoder('onehot'))

    # Remove the options from the file, as if it had been saved by an earlier version of PARROT
    saved = torch.load(path, weights_only=True)
    for name in new_options:
        del saved['hyperparameters'][name]
    torch.save(saved, path)

    loaded, encoder = load_network(path)
    assert all(loaded.hyperparameter_values[name] is False for name in new_options)
    torch.testing.assert_close(outputs_for(loaded, encoder, 'ACDEFGHIKLMN'), outputs_for(network, encoder, 'ACDEFGHIKLMN'))
