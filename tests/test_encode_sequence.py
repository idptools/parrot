"""
Tests for parrot.encode_sequence: one-hot, biophysical and user-defined encodings.
"""

import numpy as np
import pytest
import torch

from parrot import encode_sequence
from parrot import process_input_data as pid

from tests.helpers import data_path

ALL_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


# ..........................................................................................
# One-hot encoding

def test_one_hot_shape_and_values():
    encoded = encode_sequence.one_hot('ACY')

    assert encoded.shape == (3, 20)
    assert torch.all(encoded.sum(dim=1) == 1)
    assert encoded[0, 0] == 1 and encoded[1, 1] == 1 and encoded[2, 19] == 1


def test_one_hot_glutamate_matches_docstring():
    expected = np.zeros(20)
    expected[3] = 1
    np.testing.assert_array_equal(encode_sequence.one_hot('E')[0].numpy(), expected)


def test_one_hot_empty_sequence():
    assert encode_sequence.one_hot('').shape == (0, 20)


@pytest.mark.parametrize('bad_sequence', ['ACXD', 'acd', 'AC D', 'B'])
def test_one_hot_rejects_invalid_residues(bad_sequence):
    with pytest.raises(ValueError, match='Invalid amino acid'):
        encode_sequence.one_hot(bad_sequence)


# ..........................................................................................
# Biophysical encoding

def test_biophysics_shape_and_values():
    encoded = encode_sequence.biophysics('EG')

    assert encoded.shape == (2, 9)
    np.testing.assert_allclose(encoded[0].numpy(), [-3.5, -1, 3.2, 147.1, 0, 1, 161.8, 68.1, -107.3])
    np.testing.assert_allclose(encoded[1].numpy(), encode_sequence.BIOPHYSICS['G'])


def test_biophysics_table_is_complete():
    assert set(encode_sequence.BIOPHYSICS.keys()) == set(ALL_AMINO_ACIDS)
    assert all(len(vector) == 9 for vector in encode_sequence.BIOPHYSICS.values())


@pytest.mark.parametrize('bad_sequence', ['ACXD', 'acd', 'B'])
def test_biophysics_rejects_invalid_residues(bad_sequence):
    with pytest.raises(ValueError, match='Invalid amino acid'):
        encode_sequence.biophysics(bad_sequence)


# ..........................................................................................
# User-defined encoding

def test_parse_encode_file_example():
    encode_dict, vector_length = encode_sequence.parse_encode_file(data_path('encoding_example.txt'))

    assert vector_length == 6
    assert set(encode_dict.keys()) == set(ALL_AMINO_ACIDS)
    assert encode_dict['A'] == [1.1, 2.0, 6.0, 7.0, 0.0, -1.0]


def test_user_encoder_from_example_file():
    encoder = encode_sequence.UserEncoder(data_path('encoding_example.txt'))
    encoded = encoder.encode('AC')

    assert len(encoder) == 6
    assert encoded.shape == (2, 6)
    np.testing.assert_allclose(encoded[0].numpy(), [1.1, 2, 6.0, 7, 0.0, -1])
    np.testing.assert_allclose(encoded[1].numpy(), [1.5, 0, 5.1, 16, 0.1, -1])


def test_user_encoder_nucleotides():
    encoder = encode_sequence.UserEncoder(data_path('nucleotide_encoding.txt'))

    assert len(encoder) == 4
    np.testing.assert_array_equal(encoder.encode('ACGT').numpy(), np.eye(4))


def test_user_encoder_rejects_unknown_character():
    encoder = encode_sequence.UserEncoder(data_path('nucleotide_encoding.txt'))
    with pytest.raises(ValueError, match='Invalid amino acid'):
        encoder.encode('ACGU')


def test_user_encoder_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        encode_sequence.UserEncoder(str(tmp_path / 'does_not_exist.txt'))


def test_parse_encode_file_ignores_blank_lines(write_lines):
    path = write_lines('encoding.txt', ['A 1 0', '', 'C 0 1', '', ''])
    encode_dict, vector_length = encode_sequence.parse_encode_file(path)

    assert vector_length == 2
    assert encode_dict == {'A': [1.0, 0.0], 'C': [0.0, 1.0]}


def test_parse_encode_file_inconsistent_lengths(write_lines):
    path = write_lines('encoding.txt', ['A 1 0', 'C 0 1 1'])
    with pytest.raises(ValueError, match='same length'):
        encode_sequence.parse_encode_file(path)


def test_parse_encode_file_non_numeric(write_lines):
    path = write_lines('encoding.txt', ['A 1 0', 'C 0 one'])
    with pytest.raises(ValueError, match='Non-numeric'):
        encode_sequence.parse_encode_file(path)


def test_parse_encode_file_ignores_a_byte_order_mark(tmp_path):
    path = tmp_path / 'encoding.txt'
    path.write_text('A 1 0\nC 0 1\nD 1 1\n', encoding='utf-8-sig')
    encode_dict, vector_length = encode_sequence.parse_encode_file(str(path))
    assert sorted(encode_dict) == ['A', 'C', 'D'] and vector_length == 2


@pytest.mark.parametrize('lines, message', [(['AB 1 0', 'C 0 1'], "single character .* 'AB'"),
                                            (['A', 'C'], "'A' has no values")])
def test_parse_encode_file_needs_one_character_and_its_values_per_line(write_lines, lines, message):
    with pytest.raises(ValueError, match=message):
        encode_sequence.parse_encode_file(write_lines('encoding.txt', lines))


def test_parse_encode_file_repeated_character(write_lines):
    path = write_lines('encoding.txt', ['A 1 0', 'C 0 1', 'A 0 0'])
    with pytest.raises(ValueError, match="'A' is listed more than once"):
        encode_sequence.parse_encode_file(path)


def test_parse_encode_file_empty(write_lines):
    path = write_lines('encoding.txt', [''])
    with pytest.raises(ValueError, match='empty'):
        encode_sequence.parse_encode_file(path)


# ..........................................................................................
# Encoder objects and factories

@pytest.mark.parametrize('encoder_class, input_size', [(encode_sequence.OneHotEncoder, 20),
                                                       (encode_sequence.BiophysicsEncoder, 9)])
def test_built_in_encoders(encoder_class, input_size):
    encoder = encoder_class()

    assert len(encoder) == input_size
    assert encoder.encode('ACD').shape == (3, input_size)
    assert encode_sequence.encoder_from_config(encoder.to_config()).name == encoder.name


def test_get_encoder():
    assert isinstance(encode_sequence.get_encoder('onehot'), encode_sequence.OneHotEncoder)
    assert isinstance(encode_sequence.get_encoder('biophysics'), encode_sequence.BiophysicsEncoder)

    user_encoder = encode_sequence.get_encoder(data_path('nucleotide_encoding.txt'))
    assert isinstance(user_encoder, encode_sequence.UserEncoder) and len(user_encoder) == 4


def test_get_encoder_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='onehot, biophysics'):
        encode_sequence.get_encoder(str(tmp_path / 'missing.txt'))


def test_user_encoder_from_table_round_trip():
    original = encode_sequence.UserEncoder(data_path('encoding_example.txt'))
    rebuilt = encode_sequence.encoder_from_config(original.to_config())

    assert rebuilt.encode_file is None
    assert len(rebuilt) == len(original)
    torch.testing.assert_close(rebuilt.encode(ALL_AMINO_ACIDS), original.encode(ALL_AMINO_ACIDS))


def test_user_encoder_requires_exactly_one_source():
    with pytest.raises(ValueError, match='exactly one'):
        encode_sequence.UserEncoder()
    with pytest.raises(ValueError, match='exactly one'):
        encode_sequence.UserEncoder(data_path('nucleotide_encoding.txt'), table={'A': [1.0]})


def test_user_encoder_table_lengths_must_match():
    with pytest.raises(ValueError, match='same length'):
        encode_sequence.UserEncoder(table={'A': [1.0, 0.0], 'C': [1.0]})


def test_encoder_from_unknown_config():
    with pytest.raises(ValueError, match='Unknown encoding'):
        encode_sequence.encoder_from_config({'name': 'esm2'})


# ..........................................................................................
# Standardized encodings

STANDARDIZABLE_ENCODINGS = ['onehot', 'biophysics', data_path('encoding_example.txt'),
                            data_path('nucleotide_encoding.txt')]


def encoding_table(encoder: encode_sequence.SequenceEncoder) -> np.ndarray:
    """The vector of every character an encoder defines, one row per character."""
    return np.stack([encoder.encode(character).numpy()[0] for character in encoder.alphabet])


@pytest.mark.parametrize('encoding', STANDARDIZABLE_ENCODINGS)
def test_standardized_features_have_mean_zero_and_unit_sd(encoding):
    table = encoding_table(encode_sequence.get_encoder(encoding, standardize=True))
    np.testing.assert_allclose(table.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(table.std(axis=0), 1, atol=1e-12)


def test_standardized_encoding_is_a_linear_transform_of_the_base():
    base = encode_sequence.BiophysicsEncoder()
    standardized = encode_sequence.StandardizedEncoder(base)
    raw_table = encoding_table(base)

    sequence = 'MKWVTFISLLFLFSSAYS'
    expected = (base.encode(sequence).numpy() - raw_table.mean(axis=0)) / raw_table.std(axis=0)
    np.testing.assert_allclose(standardized.encode(sequence).numpy(), expected)
    assert standardized.input_size == base.input_size == 9
    assert standardized.name == 'standardized biophysics'


@pytest.mark.parametrize('encoding', STANDARDIZABLE_ENCODINGS)
def test_standardized_encoding_round_trips(encoding):
    encoder = encode_sequence.get_encoder(encoding, standardize=True)
    rebuilt = encode_sequence.encoder_from_config(encoder.to_config())

    sequence = encoder.alphabet * 2
    np.testing.assert_array_equal(rebuilt.encode(sequence).numpy(), encoder.encode(sequence).numpy())


def test_saved_standardization_is_used_as_stored():
    # A saved network must keep its exact standardization even if the base table were to change
    encoder = encode_sequence.StandardizedEncoder(encode_sequence.OneHotEncoder(), mean=[0.0] * 20, scale=[2.0] * 20)
    np.testing.assert_allclose(encoder.encode('A').numpy()[0, 0], 0.5)


def test_constant_feature_is_zeroed_with_a_warning():
    base = encode_sequence.UserEncoder(table={'A': [1.0, 5.0], 'C': [2.0, 5.0], 'D': [3.0, 5.0]})
    with pytest.warns(UserWarning, match=r'feature\(s\) \[1\]'):
        standardized = encode_sequence.StandardizedEncoder(base)

    table = encoding_table(standardized)
    np.testing.assert_allclose(table[:, 1], 0)
    np.testing.assert_allclose(table[:, 0].std(), 1)


def test_standardization_parameters_must_match_the_encoding():
    with pytest.raises(ValueError, match='both or neither'):
        encode_sequence.StandardizedEncoder(encode_sequence.OneHotEncoder(), mean=[0.0] * 20)
    with pytest.raises(ValueError, match='20 values'):
        encode_sequence.StandardizedEncoder(encode_sequence.OneHotEncoder(), mean=[0.0] * 9, scale=[1.0] * 9)


# ..........................................................................................
# LookupTable (fast encoding of whole sequences)

def test_lookup_table_matches_character_by_character_encoding():
    table = encode_sequence.LookupTable(encode_sequence.BIOPHYSICS)
    sequence = 'MKWVTFISLLFLFSSAYS'
    expected = torch.tensor([encode_sequence.BIOPHYSICS[residue] for residue in sequence], dtype=torch.float64)
    assert torch.equal(table.encode(sequence), expected)


def test_lookup_table_reports_the_first_invalid_character():
    with pytest.raises(ValueError, match='Invalid amino acid detected: b$'):
        encode_sequence.LookupTable({'A': [1.0], 'C': [2.0]}).encode('ACbXc')


def test_lookup_table_handles_any_characters_and_empty_sequences():
    table = encode_sequence.LookupTable({'α': [1.0, 2.0], 'A': [3.0, 4.0]})
    assert torch.equal(table.encode('AαA'), torch.tensor([[3., 4.], [1., 2.], [3., 4.]], dtype=torch.float64))
    assert table.encode('').shape == (0, 2)
    with pytest.raises(ValueError, match='Invalid amino acid detected: β'):
        table.encode('Aβ')


def test_padded_batches_are_float32_whatever_the_encoding_dtype():
    vectors, lengths = pid.pad_sequence_vectors([encode_sequence.one_hot('ACD'),  # float64
                                                 torch.ones(2, 20, dtype=torch.float32)])
    assert vectors.dtype == torch.float32 and lengths.dtype == torch.int64
    assert torch.equal(vectors[0, :3], encode_sequence.one_hot('ACD').float())
    assert torch.all(vectors[1, 2:] == 0)
