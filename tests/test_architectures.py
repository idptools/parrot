"""
Tests for parrot.architectures: hyperparameter declarations, the architecture
registry, shared layers, the behaviour every architecture must have, the
individual architectures, and the backwards-compatible BRNN_MtO / BRNN_MtM classes.
"""

import math

import numpy as np
import pytest
import torch

from parrot import brnn_architecture, encode_sequence
from parrot import process_input_data as pid
from parrot.architectures import (HyperParameter, ParrotArchitecture, available_architectures, build_network,
                                  get_architecture, register_architecture, registry, resolve_hyperparameters)
from parrot.architectures.brnn import BRNN, run_lstm
from parrot.architectures.hyperparameters import DROPOUT
from parrot.architectures.cnn import DILATION_CYCLE
from parrot.architectures.layers import (AttentionPool, ChannelLayerNorm, GlobalContext, append_log_length,
                                         masked_pool, multilayer_perceptron, padding_mask, resolve_lengths)
from parrot.architectures.registry import fill_legacy_values
from parrot.architectures.transformer import apply_rotary, sinusoidal_positions
from parrot.architectures.unet import Up
from parrot.tasks import Task

from tests.helpers import (ARCHITECTURES_UNDER_TEST, LinearTestArchitecture, architecture_class, encode_batch,
                           make_network, randomize_global_context, ready_network, supports)

# Sequences of very different lengths, including ones shorter than the U-Net's downsampling factor
SEQUENCES = ['ACDEFGHIK', 'LMN', 'PQRSTVWYACDEFGHIKLMNPQ', 'KK', 'W', 'ACDEFGHIKLMNPQRSTVWYAC']

ALL_TASKS = [Task('sequence', 1), Task('sequence', 3), Task('residues', 1), Task('residues', 3)]

# Every architecture with its defaults, plus non-default settings that exercise other code paths.
# Dropout is left at 0 so outputs are deterministic.
CONFIGURATIONS = [(architecture, {}) for architecture in ARCHITECTURES_UNDER_TEST] + [
    ('brnn', {'num_layers': 2}),
    ('cnn', {'pooling': 'max', 'kernel_size': 3, 'num_blocks': 6}),
    ('composition', {'num_hidden_layers': 0, 'window_radius': 0}),
    ('composition', {'num_hidden_layers': 2, 'window_radius': 20}),
    ('unet', {'depth': 1}),
    ('unet', {'depth': 4, 'upsampling': 'nearest', 'pooling': 'max', 'kernel_size': 5}),
    ('transformer', {'positional_encoding': 'sinusoidal', 'pooling': 'max', 'num_layers': 3}),
    ('transformer', {'positional_encoding': 'none', 'num_heads': 1, 'head_size': 8, 'feedforward_multiplier': 1}),
    ('transformer', {'num_heads': 3, 'head_size': 5}),  # odd head size: one feature is not rotated
    ('brnn', {'attention_pooling': False, 'loglength_feedforward': False}),
    ('brnn', {'attention_pooling': False}),
    ('cnn', {'global_context': False, 'attention_pooling': False, 'loglength_feedforward': False}),
    ('unet', {'global_context': False, 'attention_pooling': False, 'pooling': 'max'}),
    ('transformer', {'attention_pooling': False, 'loglength_feedforward': False}),
    ('linear', {'features': 'dipeptides', 'alpha': 0.01}),
    ('svm', {'kernel': 'linear'}),
    ('svm', {'kernel': 'poly', 'features': 'dipeptides'}),
    ('random_forest', {'num_trees': 10, 'min_samples_leaf': 3}),
    ('knn', {'num_neighbors': 3, 'neighbor_weights': 'distance'}),
]


def configuration_id(architecture: str, values: dict) -> str:
    return f"{architecture}-{'-'.join(f'{k}={v}' for k, v in values.items()) or 'defaults'}"


# Every (configuration, task) combination the architecture supports
CASES = [(architecture, values, task) for architecture, values in CONFIGURATIONS for task in ALL_TASKS
         if supports(architecture, task.datatype)]
CASE_IDS = [f'{configuration_id(architecture, values)}-{task.datatype}-{task.num_classes}'
            for architecture, values, task in CASES]

# Configurations of architectures trained by gradient descent
GRADIENT_CONFIGURATIONS = [(architecture, values) for architecture, values in CONFIGURATIONS
                           if architecture_class(architecture).trained_by_gradient_descent]


def encode_single(sequence: str) -> torch.Tensor:
    """One-hot encode a single sequence as a batch of one (no padding)."""
    return encode_sequence.one_hot(sequence).float().view(1, len(sequence), -1)


# ..........................................................................................
# HyperParameter

def test_hyperparameter_cli_flag():
    assert HyperParameter('hidden_size', int, 10, 'help').cli_flag == '--hidden-size'


def test_hyperparameter_validate():
    hyperparameter = HyperParameter('num_layers', int, 1, 'help', minimum=1)

    assert hyperparameter.validate(3) == 3
    assert hyperparameter.validate(3.0) == 3 and isinstance(hyperparameter.validate(3.0), int)
    with pytest.raises(ValueError, match='at least 1'):
        hyperparameter.validate(0)
    with pytest.raises(ValueError, match='type int'):
        hyperparameter.validate(2.5)


def test_hyperparameter_bounds_and_choices():
    bounded = HyperParameter('dropout', float, 0.0, 'help', minimum=0.0, maximum=1.0)
    assert bounded.validate(0.5) == 0.5
    with pytest.raises(ValueError, match='at most 1.0'):
        bounded.validate(1.5)

    sized = HyperParameter('kernel_size', int, 3, 'help', choices=(3, 5))
    assert sized.validate(5) == 5
    with pytest.raises(ValueError, match='one of'):
        sized.validate(4)


def test_hyperparameter_exclusive_minimum():
    penalty = HyperParameter('alpha', float, 1.0, 'help', minimum=0, exclusive_minimum=True)
    assert penalty.validate(1e-9) == 1e-9
    with pytest.raises(ValueError, match='greater than 0'):
        penalty.validate(0)


def test_hyperparameter_exclusive_maximum():
    assert DROPOUT.validate(0.99) == 0.99
    with pytest.raises(ValueError, match='less than 1.0'):
        DROPOUT.validate(1.0)


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan, True, False, '0.5', None])
def test_numeric_hyperparameters_must_be_finite_numbers(value):
    with pytest.raises(ValueError, match='must be a finite number'):
        HyperParameter('alpha', float, 1.0, 'help').validate(value)
    with pytest.raises(ValueError, match='must be a finite number'):
        HyperParameter('channels', int, 16, 'help', minimum=1).validate(value)


def test_numeric_hyperparameters_accept_numpy_numbers():
    assert HyperParameter('channels', int, 16, 'help').validate(np.int64(24)) == 24
    assert HyperParameter('alpha', float, 1.0, 'help').validate(np.float32(0.5)) == 0.5


def test_text_hyperparameter():
    pooling = HyperParameter('pooling', str, 'mean', 'help', choices=('mean', 'max'))

    assert pooling.validate('max') == 'max'
    assert not pooling.searchable
    with pytest.raises(ValueError, match='one of'):
        pooling.validate('sum')
    with pytest.raises(ValueError, match='must be text'):
        pooling.validate(1)


def test_searchable():
    assert HyperParameter('depth', int, 3, 'help', search_range=(1, 5)).searchable
    assert not HyperParameter('depth', int, 3, 'help').searchable


@pytest.mark.parametrize('kwargs', [
    {'name': 'hidden size'},
    {'value_type': list},
    {'value_type': str, 'default': 'a'},                                    # text without choices
    {'value_type': str, 'default': 'a', 'choices': ('a',), 'search_range': (0, 1)},
    {'search_range': (5, 5)},
    {'default': 0, 'minimum': 1},                                           # invalid default
    {'value_type': bool, 'default': 1},                                     # on/off must be True or False
    {'value_type': bool, 'default': True, 'choices': (True, False)},
    {'value_type': bool, 'default': True, 'search_range': (0, 1)},
    {'value_type': bool, 'default': True, 'legacy_value': 'no'},
    {'exclusive_minimum': True},                                            # exclusive minimum without a minimum
    {'exclusive_maximum': True},                                            # exclusive maximum without a maximum
])
def test_hyperparameter_rejects_invalid_declarations(kwargs):
    arguments = dict(name='hidden_size', value_type=int, default=10, help='help')
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        HyperParameter(**arguments)


# ..........................................................................................
# Registry

def test_built_in_architectures_are_registered():
    assert {'brnn', 'cnn', 'composition', 'unet'} <= set(available_architectures())
    assert get_architecture('brnn') is BRNN


def test_on_off_hyperparameter():
    on_by_default = HyperParameter('attention_pooling', bool, True, 'help', legacy_value=False)
    off_by_default = HyperParameter('extra_layer', bool, False, 'help')

    assert on_by_default.cli_flag == '--attention-pooling-off'
    assert off_by_default.cli_flag == '--extra-layer'
    assert on_by_default.validate(False) is False
    assert not on_by_default.searchable
    with pytest.raises(ValueError, match='True or False'):
        on_by_default.validate('off')


def test_fill_legacy_values():
    # A CNN saved before the on/off options existed is rebuilt with all of them off
    saved = {'channels': 8, 'num_blocks': 2, 'kernel_size': 3, 'dropout': 0.0, 'pooling': 'max'}
    assert fill_legacy_values(get_architecture('cnn'), saved) == {
        **saved, 'global_context': False, 'attention_pooling': False, 'loglength_feedforward': False}
    assert fill_legacy_values(get_architecture('cnn'), {'attention_pooling': True})['attention_pooling'] is True

    # Likewise a BRNN saved before attention pooling and log(length) existed
    assert fill_legacy_values(get_architecture('brnn'), {'hidden_size': 5, 'num_layers': 1}) == {
        'hidden_size': 5, 'num_layers': 1, 'attention_pooling': False, 'loglength_feedforward': False}


def test_get_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture 'not_an_architecture'.*brnn"):
        get_architecture('not_an_architecture')


def test_register_architecture(monkeypatch):
    monkeypatch.setattr(registry, '_ARCHITECTURES', dict(registry._ARCHITECTURES))

    assert register_architecture(LinearTestArchitecture) is LinearTestArchitecture
    assert 'test_linear' in available_architectures()

    with pytest.raises(ValueError, match='already registered'):
        register_architecture(LinearTestArchitecture)


def test_register_rejects_duplicate_hyperparameters(monkeypatch):
    monkeypatch.setattr(registry, '_ARCHITECTURES', dict(registry._ARCHITECTURES))

    class Duplicated(LinearTestArchitecture):
        name = 'duplicated'
        hyperparameters = LinearTestArchitecture.hyperparameters * 2

    with pytest.raises(ValueError, match='more than once'):
        register_architecture(Duplicated)


def test_resolve_hyperparameters():
    assert resolve_hyperparameters(BRNN) == {'hidden_size': 10, 'num_layers': 1, 'attention_pooling': True,
                                             'loglength_feedforward': True}
    assert resolve_hyperparameters(BRNN, {'num_layers': 3}) == {'hidden_size': 10, 'num_layers': 3,
                                                                'attention_pooling': True,
                                                                'loglength_feedforward': True}

    with pytest.raises(ValueError, match='no hyperparameter'):
        resolve_hyperparameters(BRNN, {'kernel_size': 3})
    with pytest.raises(ValueError, match='at least 1'):
        resolve_hyperparameters(BRNN, {'hidden_size': 0})


def test_build_network():
    network = build_network('brnn', 9, Task('residues', 2), {'hidden_size': 6})

    assert isinstance(network, BRNN) and isinstance(network, ParrotArchitecture)
    assert network.input_size == 9 and network.num_classes == 2
    assert network.hyperparameter_values == {'hidden_size': 6, 'num_layers': 1, 'attention_pooling': True,
                                             'loglength_feedforward': True}
    assert network.hidden_size == 6 and network.num_layers == 1


# ..........................................................................................
# Shared layers

def test_resolve_lengths():
    x = torch.zeros(3, 7, 20)
    assert resolve_lengths(x, None).tolist() == [7, 7, 7]
    assert resolve_lengths(x, torch.tensor([7, 2, 5])).tolist() == [7, 2, 5]


def test_padding_mask():
    mask = padding_mask(torch.tensor([2, 4, 1]), 4)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True], [True, False, False, False]]


def test_masked_pool():
    features = torch.tensor([[[1.0], [3.0], [100.0]], [[-2.0], [-4.0], [-6.0]]])
    mask = padding_mask(torch.tensor([2, 3]), 3)

    torch.testing.assert_close(masked_pool(features, mask, 'mean'), torch.tensor([[2.0], [-4.0]]))
    torch.testing.assert_close(masked_pool(features, mask, 'max'), torch.tensor([[3.0], [-2.0]]))
    with pytest.raises(ValueError, match='pooling'):
        masked_pool(features, mask, 'sum')


def test_channel_layer_norm_normalises_each_position():
    x = torch.randn(2, 6, 5) * 10 + 3
    normalised = ChannelLayerNorm(6)(x)

    torch.testing.assert_close(normalised.mean(dim=1), torch.zeros(2, 5), atol=1e-5, rtol=0)
    # Changing one position does not affect any other
    x_changed = x.clone()
    x_changed[:, :, 4] = 1000.0
    torch.testing.assert_close(ChannelLayerNorm(6)(x_changed)[:, :, :4], normalised[:, :, :4])


@pytest.mark.parametrize('num_hidden_layers, expected_linear_layers', [(0, 1), (1, 2), (3, 4)])
def test_multilayer_perceptron(num_hidden_layers, expected_linear_layers):
    mlp = multilayer_perceptron(10, 8, num_hidden_layers, 3, dropout=0.0)

    assert sum(isinstance(layer, torch.nn.Linear) for layer in mlp) == expected_linear_layers
    assert mlp(torch.randn(4, 7, 10)).shape == (4, 7, 3)


# ..........................................................................................
# Behaviour every architecture must have

@pytest.mark.parametrize('architecture, values, task', CASES, ids=CASE_IDS)
def test_output_shapes(test_architecture, architecture, values, task):
    network = ready_network(architecture, task, values)
    batch, lengths = encode_batch(SEQUENCES)
    padded_length = pid.padded_batch_length(max(len(seq) for seq in SEQUENCES))

    expected = (len(SEQUENCES), padded_length, task.num_classes) if task.per_residue else (len(SEQUENCES), task.num_classes)
    assert network(batch, lengths).shape == expected


@pytest.mark.parametrize('architecture, values, task', CASES, ids=CASE_IDS)
def test_padding_invariance(test_architecture, architecture, values, task):
    """A sequence must give the same output whether or not it is padded in a batch"""
    network = ready_network(architecture, task, values)
    batch, lengths = encode_batch(SEQUENCES)

    with torch.no_grad():
        batched = network(batch, lengths)
        for i, sequence in enumerate(SEQUENCES):
            alone = network(encode_single(sequence))[0]
            if task.per_residue:
                torch.testing.assert_close(batched[i, :len(sequence)], alone, rtol=1e-4, atol=1e-5)
            else:
                torch.testing.assert_close(batched[i], alone, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('architecture, values', GRADIENT_CONFIGURATIONS,
                         ids=[configuration_id(*configuration) for configuration in GRADIENT_CONFIGURATIONS])
def test_gradients_flow_through_padded_batches(test_architecture, architecture, values):
    network = build_network(architecture, 20, Task('residues', 2), values)
    batch, lengths = encode_batch(SEQUENCES)

    network(batch, lengths).sum().backward()

    assert all(parameter.grad is not None for parameter in network.parameters())


@pytest.mark.parametrize('architecture', ARCHITECTURES_UNDER_TEST)
def test_other_encoding_sizes(test_architecture, architecture):
    network = ready_network(architecture, Task('sequence', 2), encoding='biophysics')
    batch, lengths = pid.pad_sequence_vectors([encode_sequence.biophysics(seq) for seq in SEQUENCES])
    assert network(batch, lengths).shape == (len(SEQUENCES), 2)


@pytest.mark.parametrize('architecture', ARCHITECTURES_UNDER_TEST)
def test_architecture_declarations(test_architecture, architecture):
    """Every architecture declares its interface and records every hyperparameter it was built with"""
    declared = get_architecture(architecture)
    assert declared.name == architecture
    assert isinstance(declared.description, str) and declared.description
    assert set(declared.supported_datatypes) <= {'sequence', 'residues'} and declared.supported_datatypes

    network = build_network(architecture, 20, Task('sequence', 1))
    assert network.hyperparameter_values == resolve_hyperparameters(declared)


def test_architectures_record_exactly_their_declared_hyperparameters(test_architecture):
    """Values saved with a network rebuild it, so a missing or misspelt one must fail when the network is built"""
    class ForgetfulArchitecture(test_architecture):
        def __init__(self, input_size, task, hidden_units):
            ParrotArchitecture.__init__(self, input_size, task, {'hidden_unit': hidden_units})

    with pytest.raises(ValueError, match=r"missing \['hidden_units'\], not declared \['hidden_unit'\]"):
        ForgetfulArchitecture(20, Task('sequence', 1), hidden_units=4)


def test_architectures_check_hyperparameter_values_when_built_directly():
    """Building a network without going through build_network() still checks its hyperparameters"""
    with pytest.raises(ValueError, match="'hidden_size' must be at least 1"):
        BRNN(20, Task('sequence', 1), hidden_size=0, num_layers=1, attention_pooling=True,
             loglength_feedforward=True)


@pytest.mark.parametrize('architecture', [name for name in ARCHITECTURES_UNDER_TEST if not supports(name, 'residues')])
def test_unsupported_datatype_is_rejected(architecture):
    with pytest.raises(ValueError, match='only supports sequence-mapped data'):
        build_network(architecture, 20, Task('residues', 1))


# ..........................................................................................
# Composition baseline

def test_composition_uses_composition_and_log_length():
    """With no hidden layers the model is linear in [composition, log(length)]"""
    network = build_network('composition', 20, Task('sequence', 1), {'num_hidden_layers': 0})
    sequence = 'AACDE'
    fractions = encode_sequence.one_hot(sequence).float().mean(dim=0)
    features = torch.cat([fractions, torch.log(torch.tensor([float(len(sequence))]))])

    with torch.no_grad():
        torch.testing.assert_close(network(encode_single(sequence))[0], network.mlp(features))


def test_composition_ignores_residue_order():
    network = build_network('composition', 20, Task('sequence', 3))
    with torch.no_grad():
        torch.testing.assert_close(network(encode_single('AAKKDDEE')), network(encode_single('KDAEKADE')))


def test_composition_window_average():
    network = build_network('composition', 20, Task('residues', 1), {'window_radius': 1})
    sequence = 'ACDK'
    encoded = encode_sequence.one_hot(sequence).float()

    with torch.no_grad():
        windows = network.window_composition(encoded.unsqueeze(0), torch.ones(1, 4, dtype=torch.bool))[0]

    # Windows are truncated at the sequence ends
    expected = torch.stack([encoded[0:2].mean(0), encoded[0:3].mean(0), encoded[1:4].mean(0), encoded[2:4].mean(0)])
    torch.testing.assert_close(windows, expected)


# ..........................................................................................
# Dilated CNN

def test_cnn_dilation_schedule():
    network = build_network('cnn', 20, Task('residues', 1), {'num_blocks': 6, 'kernel_size': 3})
    dilations = [block.conv.dilation[0] for block in network.blocks]
    assert dilations == [2 ** (i % DILATION_CYCLE) for i in range(6)] == [1, 2, 4, 8, 1, 2]


def test_cnn_rejects_even_kernel_size():
    with pytest.raises(ValueError, match='one of'):
        build_network('cnn', 20, Task('residues', 1), {'kernel_size': 4})


def test_cnn_sees_distant_residues():
    """Changing a residue changes outputs well beyond a single convolution's reach"""
    network = build_network('cnn', 20, Task('residues', 1), {'num_blocks': 4, 'kernel_size': 3,
                                                               'global_context': False})
    original = 'A' * 40
    changed = 'A' * 39 + 'W'

    with torch.no_grad():
        difference = (network(encode_single(original)) - network(encode_single(changed)))[0, :, 0].abs()

    # Receptive field of 4 blocks with kernel 3 and dilations 1, 2, 4, 8 is 1 + 2 * 15 = 31 residues
    assert difference[39 - 15] > 0
    assert difference[39 - 16] == 0


@pytest.mark.parametrize('architecture, values', [('cnn', {'num_blocks': 1, 'kernel_size': 3}), ('unet', {'depth': 1})])
def test_global_context_lets_every_residue_see_the_whole_sequence(architecture, values):
    original, changed = 'A' * 300, 'A' * 299 + 'W'
    differences = {}
    for global_context in (False, True):
        network = build_network(architecture, 20, Task('residues', 1), {**values, 'global_context': global_context})
        randomize_global_context(network)
        with torch.no_grad():
            differences[global_context] = (network(encode_single(original)) - network(encode_single(changed)))[0, 0, 0]

    assert differences[False] == 0  # without it, the first residue cannot see the last
    assert differences[True] != 0


def test_global_context_starts_at_zero():
    """A new network starts out exactly as it would without the whole-sequence summary"""
    torch.manual_seed(1)
    with_context = build_network('cnn', 20, Task('residues', 1), {'global_context': True})
    torch.manual_seed(1)
    without_context = build_network('cnn', 20, Task('residues', 1), {'global_context': False})
    without_context.load_state_dict({name: value for name, value in with_context.state_dict().items()
                                     if '.context.' not in name})
    batch, lengths = encode_batch(SEQUENCES)
    with torch.no_grad():
        torch.testing.assert_close(with_context(batch, lengths), without_context(batch, lengths))


@pytest.mark.parametrize('architecture', ['cnn', 'unet'])
def test_global_context_leaves_the_other_initial_weights_as_they_would_be_without_it(architecture):
    # Its layer used to draw random numbers before being zeroed, so every later layer started differently
    outputs = []
    for global_context in (True, False):
        torch.manual_seed(1)
        network = build_network(architecture, 20, Task('residues', 1), {'global_context': global_context})
        with torch.no_grad():
            outputs.append(network(encode_single('ACDEFGHIKLMNPQRSTVWY' * 3)))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_global_context_averages_real_positions_only():
    layer = GlobalContext(2)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2))
    x = torch.tensor([[[1., 3., 100.], [2., 4., 100.]]])  # [batch X channels X length]; the last position is padding
    mask = torch.tensor([[[1., 1., 0.]]])
    torch.testing.assert_close(layer(x, mask), torch.tensor([[[2.], [3.]]]))


# ..........................................................................................
# Attention pooling and log(length)

def test_attention_pool_weights_real_residues_only():
    pool = AttentionPool(3)
    features = torch.randn(2, 5, 3)
    mask = padding_mask(torch.tensor([5, 2]), 5)

    # With every score equal, attention pooling is the mean over real residues
    with torch.no_grad():
        pool.score[-1].weight.zero_()
        pool.score[-1].bias.zero_()
        torch.testing.assert_close(pool(features, mask), masked_pool(features, mask, 'mean'))

        # Padding never receives weight, whatever it contains
        padded = features.clone()
        padded[1, 2:] = 1000.0
        pool.score[-1].weight.normal_()
        torch.testing.assert_close(pool(padded, mask)[1], pool(features, mask)[1])


def test_append_log_length():
    features = torch.zeros(2, 3)
    torch.testing.assert_close(append_log_length(features, torch.tensor([1, 10]))[:, -1],
                               torch.log(torch.tensor([1., 10.])))


def test_loglength_feedforward_makes_length_visible():
    """An order-free transformer cannot tell a sequence from the same sequence repeated, except through log(length)"""
    outputs = {}
    for loglength_feedforward in (False, True):
        torch.manual_seed(0)
        network = build_network('transformer', 20, Task('sequence', 1),
                                {'positional_encoding': 'none', 'attention_pooling': False,
                                 'loglength_feedforward': loglength_feedforward})
        with torch.no_grad():
            outputs[loglength_feedforward] = (network(encode_single('AKAKAKAKAK' * 4)),
                                              network(encode_single('AKAKAKAKAK' * 8)))

    torch.testing.assert_close(*outputs[False], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(*outputs[True])


@pytest.mark.parametrize('architecture', ['brnn', 'cnn', 'unet', 'transformer'])
def test_loglength_feedforward_widens_the_output_layer(architecture):
    widths = {value: build_network(architecture, 20, Task('sequence', 3), {'loglength_feedforward': value}).fc.in_features
              for value in (False, True)}
    assert widths[True] == widths[False] + 1


@pytest.mark.parametrize('architecture', ['brnn', 'cnn', 'unet', 'transformer'])
def test_sequence_options_add_no_parameters_for_residue_tasks(architecture):
    """Attention pooling and log(length) only apply to sequence tasks"""
    values = {'attention_pooling': False} if architecture != 'brnn' else {}
    on = build_network(architecture, 20, Task('residues', 2))
    off = build_network(architecture, 20, Task('residues', 2), {**values, 'loglength_feedforward': False})
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())


# ..........................................................................................
# U-Net

@pytest.mark.parametrize('depth, expected_channels', [(1, [16, 32]), (3, [16, 32, 64, 128])])
def test_unet_channels_double_with_depth(depth, expected_channels):
    network = build_network('unet', 20, Task('residues', 1), {'depth': depth})

    assert network.input_block.first_conv.out_channels == expected_channels[0]
    assert [block.first_conv.out_channels for block in network.down_blocks] == expected_channels[1:]
    assert len(network.up_blocks) == depth


@pytest.mark.parametrize('length', [1, 7, 8, 9, 33])
def test_unet_handles_any_length(length):
    network = build_network('unet', 20, Task('residues', 2), {'depth': 3})
    sequence = ('ACDEFGHIKLMNPQRSTVWY' * 2)[:length]
    assert network(encode_single(sequence)).shape == (1, length, 2)


def test_unet_rejects_unknown_upsampling():
    with pytest.raises(ValueError, match='upsampling'):
        Up(8, 4, kernel_size=3, dropout=0.0, upsampling='bilinear')


# ..........................................................................................
# Transformer

def test_transformer_model_width():
    network = build_network('transformer', 20, Task('residues', 1), {'num_heads': 3, 'head_size': 5})
    assert network.model_size == 15
    assert network.blocks[0].attention.qkv.weight.shape == (45, 15)


def test_rotary_attention_depends_only_on_relative_position():
    torch.manual_seed(0)
    query, key = torch.randn(6), torch.randn(6)
    # The same query and key vectors at every position of a sequence of 12 residues
    queries = apply_rotary(query.expand(1, 1, 12, 6))[0, 0]
    keys = apply_rotary(key.expand(1, 1, 12, 6))[0, 0]

    torch.testing.assert_close(queries.norm(dim=-1), query.norm().expand(12))  # rotations preserve length
    torch.testing.assert_close(queries[2] @ keys[5], queries[8] @ keys[11])  # same offset, same score
    assert not torch.isclose(queries[2] @ keys[5], queries[2] @ keys[6])
    torch.testing.assert_close(queries[0], query)  # position 0 is not rotated


def test_sinusoidal_positions():
    positions = sinusoidal_positions(5, 7, torch.device('cpu'))

    assert positions.shape == (5, 7)
    torch.testing.assert_close(positions[0], torch.tensor([0., 1., 0., 1., 0., 1., 0.]))
    assert torch.all(positions[:, 6] == 0)  # an odd last dimension is unused
    torch.testing.assert_close(positions[3, :2], torch.tensor([torch.sin(torch.tensor(3.)), torch.cos(torch.tensor(3.))]))


def test_transformer_without_positions_ignores_residue_order():
    network = build_network('transformer', 20, Task('residues', 1), {'positional_encoding': 'none'})
    order = [3, 0, 6, 1, 7, 2, 5, 4]
    sequence = 'ACDKLMWY'
    shuffled = ''.join(sequence[i] for i in order)

    with torch.no_grad():
        torch.testing.assert_close(network(encode_single(sequence))[0, order], network(encode_single(shuffled))[0])


@pytest.mark.parametrize('positional_encoding', ['rotary', 'sinusoidal'])
@pytest.mark.parametrize('sizes', [{}, {'head_size': 2, 'num_heads': 2}])
def test_transformer_positions_make_residue_order_matter(positional_encoding, sizes):
    network = build_network('transformer', 20, Task('sequence', 1),
                            {'positional_encoding': positional_encoding, **sizes})
    with torch.no_grad():
        assert not torch.allclose(network(encode_single('AAKKDDEE')), network(encode_single('KDAEKADE')))


def test_transformer_heads_must_be_wide_enough_to_encode_positions():
    # A head 1 wide has no pair of features to rotate, so rotary positions did nothing
    with pytest.raises(ValueError, match="'head_size' must be at least 2"):
        build_network('transformer', 20, Task('sequence', 1), {'head_size': 1})


def test_transformer_sees_the_whole_sequence():
    """Even one block lets the first residue's output depend on the last residue"""
    network = build_network('transformer', 20, Task('residues', 1), {'num_layers': 1})
    with torch.no_grad():
        difference = network(encode_single('A' * 300)) - network(encode_single('A' * 299 + 'W'))
    assert difference[0, 0, 0] != 0


def test_transformer_handles_long_sequences():
    network = build_network('transformer', 20, Task('sequence', 2))
    with torch.no_grad():
        assert network(encode_single(('ACDEFGHIKLMNPQRSTVWY' * 150)[:2500])).shape == (1, 2)


# ..........................................................................................
# BRNN specifics

def test_padding_without_lengths_changes_brnn_predictions():
    """Documents why lengths are needed: unmasked zero-padding alters the prediction"""
    network = make_network('sequence', 1)
    single = encode_single('EEEKKKLLLAAA')
    padded = torch.cat([single, torch.zeros(1, 20, 20)], dim=1)

    with torch.no_grad():
        assert not torch.allclose(network(single), network(padded))
        torch.testing.assert_close(network(single), network(padded, torch.tensor([12])))


@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_brnn_full_length_lengths_match_no_lengths(datatype):
    network = make_network(datatype, 3)
    single = encode_single('ACDEFGHIKLMN')

    with torch.no_grad():
        torch.testing.assert_close(network(single), network(single, torch.tensor([12])))


def test_run_lstm_zeroes_padded_positions():
    lstm = torch.nn.LSTM(20, 5, 1, batch_first=True, bidirectional=True)
    batch, lengths = encode_batch(['ACDEF', 'GH'])

    out, h_n = run_lstm(lstm, batch, lengths)

    assert out.shape == (2, 8, 10)  # padded to padded_batch_length(5) = 8
    assert h_n.shape == (2, 2, 5)
    assert torch.all(out[1, 2:] == 0)


@pytest.mark.parametrize('datatype', ['sequence', 'residues'])
def test_brnn_state_dict_keys_match_legacy_format(datatype):
    """Saved networks (e.g. those shipped with SPARROW) depend on these exact parameter names"""
    network = make_network(datatype, 3, hidden_size=7, num_layers=2)

    expected_keys = {'fc.weight', 'fc.bias'}
    for layer in range(2):
        for suffix in ['', '_reverse']:
            for kind in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                expected_keys.add(f'lstm.{kind}_l{layer}{suffix}')

    state_dict = network.state_dict()
    assert set(state_dict.keys()) == expected_keys
    assert state_dict['lstm.weight_ih_l0'].shape == (4 * 7, 20)
    assert state_dict['fc.weight'].shape == (3, 2 * 7)


@pytest.mark.parametrize('attention_pooling', [True, False])
def test_brnn_sequence_readout(attention_pooling):
    """The BRNN summarises a sequence by attention pooling or by its final hidden states"""
    network = build_network('brnn', 20, Task('sequence', 2), {'hidden_size': 6, 'attention_pooling': attention_pooling})

    assert isinstance(network.attention_pool, AttentionPool) is attention_pooling
    assert network.fc.in_features == 2 * 6 + 1  # both directions, plus log(length)


def test_brnn_attention_pooling_weights_every_residue():
    """With attention pooling, the output layer sees a weighted average of the LSTM output at every residue"""
    network = build_network('brnn', 20, Task('sequence', 1), {'hidden_size': 6, 'loglength_feedforward': False}).eval()
    x = encode_single('ACDEFGHIKLMNPQRSTVWY')

    with torch.no_grad():
        out, _ = run_lstm(network.lstm, x, torch.tensor([20]))
        weights = torch.softmax(network.attention_pool.score(out).squeeze(-1), dim=-1)
        expected = network.fc((weights.unsqueeze(-1) * out).sum(dim=1))
        torch.testing.assert_close(network(x), expected)


def test_brnn_attention_pooling_is_only_for_sequence_tasks():
    assert build_network('brnn', 20, Task('residues', 2), {'attention_pooling': True}).attention_pool is None


# ..........................................................................................
# Backwards-compatible classes

@pytest.mark.parametrize('legacy_class, datatype', [(brnn_architecture.BRNN_MtO, 'sequence'),
                                                    (brnn_architecture.BRNN_MtM, 'residues')])
def test_legacy_classes(legacy_class, datatype):
    """BRNN_MtO / BRNN_MtM keep their constructor, attributes and behaviour"""
    legacy = legacy_class(20, 11, 2, 3, 'cpu')

    assert isinstance(legacy, BRNN)
    assert legacy.task == Task(datatype, 3)
    assert legacy.hidden_size == 11 and legacy.num_layers == 2 and legacy.num_classes == 3
    assert legacy.device == 'cpu'

    # Weights load interchangeably between the legacy class and the registry-built network
    modern = make_network(datatype, 3, hidden_size=11, num_layers=2)
    legacy.load_state_dict(modern.state_dict())
    single = encode_single('ACDEFG')
    with torch.no_grad():
        torch.testing.assert_close(legacy(single), modern(single))
