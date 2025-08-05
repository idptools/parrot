# Import necessary libraries for testing
import pytest
from pathlib import Path
import torch
import re # Import the 're' module for regular expression operations
import numpy as np
from omegaconf import OmegaConf, DictConfig

# Import the encoder classes from the 'parrot.encode_sequence' module.
# These are the classes we are testing.
from parrot.encode_sequence import ParrotLightningEncoder
from parrot.encode_sequence import TableParrotEncoder
from parrot.encode_sequence import FunctionalParrotEncoder
from parrot.encode_sequence import MatrixParrotEncoder
from parrot.encode_sequence import BaseParrotEncoder

# --- Temporary File Content Definitions ---
# These multi-line strings (using triple quotes """...""") define the content
# that will be written to temporary files during the tests. This approach
# keeps the test data self-contained within the test file, making tests
# portable and easy to understand without external dependencies.

# Content for the temporary files
TABLE_TSV_CONTENT = """
A	1.0	0.0	0.0
C	0.0	1.0	0.0
G	0.0	0.0	1.0
"""

FUNCTION_PY_CONTENT = """
import torch
import numpy as np

# Define a simple mapping for 'A' and 'C' to 2D vectors.
_map = {'A': [1.0, 0.0], 'C': [0.0, 1.0]}
_rev_map = {tuple(v): k for k, v in _map.items()}

# Custom encoding function: takes a sequence string and returns a PyTorch tensor.
# It converts characters to uppercase and looks up their vector representation.
def custom_encode(sequence: str) -> torch.Tensor:
    vectors = [_map[c.upper()] for c in sequence]
    return torch.tensor(vectors, dtype=torch.float32)

# Custom decoding function: takes a NumPy array of vectors and returns a sequence string.
# It converts each vector row to a tuple for lookup in the reverse map.
def custom_decode(seq_vector_np: np.ndarray) -> str:
    chars = [_rev_map.get(tuple(row), '?') for row in seq_vector_np.tolist()]
    return "".join(chars)
"""

@pytest.fixture(scope="module")
# --- Pytest Fixture: encoder_data ---
# A fixture is a special function that Pytest runs before tests that depend on it.
# It's used to set up a baseline environment for tests.
# 'scope="module"' means this fixture will run only once for all tests in this module,
# which is efficient for creating temporary files.
# 'tmp_path_factory' is a built-in Pytest fixture that provides a factory for creating
# temporary directories and files, ensuring tests are isolated and clean up after themselves.
def encoder_data(tmp_path_factory):
    """Pytest fixture to create temporary data and config files for tests."""
    tmp_path = tmp_path_factory.mktemp("encoder_data")

    # Create data files
    table_file = tmp_path / "test_table.tsv"
    table_file.write_text(TABLE_TSV_CONTENT.strip())

    module_file = tmp_path / "test_functions.py"
    module_file.write_text(FUNCTION_PY_CONTENT.strip())

    # Create Hydra config dicts
    table_config = {
        "type": "table",
        "table_file_path": str(table_file),
        "alphabet": "ACG"
    }

    functional_config = {
        "type": "function",
        "module_path": str(module_file),
        "encode_function_name": "custom_encode",
        "decode_function_name": "custom_decode",
        "alphabet": "AC",
        "input_size": 2
    }

    # Return the created OmegaConf objects. These will be passed to tests that
    # request 'encoder_data' as an argument.
    return {
        "table_config": OmegaConf.create(table_config),
        "functional_config": OmegaConf.create(functional_config)
    }


# --- Test Suite for TableParrotEncoder ---
# This class groups tests specifically for the TableParrotEncoder, which uses
# a lookup table for encoding/decoding.
class TestTableEncoder:
    def test_initialization_success(self, encoder_data):
        """
        Tests if TableParrotEncoder can be initialized successfully with valid configuration.
        It checks if the created object is an instance of BaseParrotEncoder (due to inheritance),
        and if its length (input_size) and alphabet are correctly set.
        """
        encoder = TableParrotEncoder(encoder_data["table_config"])
        assert isinstance(encoder, BaseParrotEncoder)
        assert len(encoder) == 3
        assert encoder.get_alphabet() == {'A', 'C', 'G'}

    def test_encode_decode_cycle(self, encoder_data):
        """
        Tests the full encode-decode cycle for the TableParrotEncoder.
        It encodes a sequence, then attempts to decode the resulting tensor,
        verifying that the original sequence is recovered.
        """
        encoder = TableParrotEncoder(encoder_data["table_config"])
        sequence = "ACG"
        encoded = encoder.encode(sequence)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (3, 3) # (sequence_length, input_size)
        
        decoded = encoder.decode(encoded)
        assert decoded == [sequence] # decode returns a list of strings.

    def test_batch_decode(self, encoder_data):
        """
        Tests the batch decoding capability of the TableParrotEncoder.
        It encodes two sequences, converts them to NumPy arrays, and then
        decodes them as a list of arrays, expecting both original sequences back.
        """
        encoder = TableParrotEncoder(encoder_data["table_config"])
        seq1 = "ACG"
        seq2 = "GCA"
        encoded1 = encoder.encode(seq1)
        encoded2 = encoder.encode(seq2)

        # Pass a list of NumPy arrays to simulate a batch of encoded sequences.
        decoded_list = encoder.decode([encoded1.numpy(), encoded2.numpy()])
        assert decoded_list == [seq1, seq2]

    def test_encode_invalid_char(self, encoder_data):
        """
        Tests that encoding a sequence with an invalid character (not in the alphabet)
        raises a ValueError, as expected for strict alphabet validation.
        'pytest.raises' is used to assert that a specific exception is raised.
        'match' uses a regex to check if the exception message contains the expected substring.
        """
        encoder = TableParrotEncoder(encoder_data["table_config"])
        with pytest.raises(ValueError, match="Invalid character 'X'"):
            encoder.encode("ACGX")

    def test_missing_file_raises_error(self, encoder_data):
        """
        Tests that initializing the TableParrotEncoder with a non-existent table file path
        raises a FileNotFoundError.
        """
        bad_config = encoder_data["table_config"].copy()
        bad_config.table_file_path = "non_existent_file.tsv"
        with pytest.raises(FileNotFoundError):
            TableParrotEncoder(bad_config)

    def test_alphabet_mismatch_raises_error(self, encoder_data):
        """
        Tests that if the specified alphabet contains characters not present in the
        provided table file, a ValueError is raised during initialization.
        """
        bad_config = encoder_data["table_config"].copy()
        bad_config.alphabet = "ACGY" # Y is not in the table
        with pytest.raises(ValueError, match="Character 'Y' from specified alphabet not found in table file"):
            TableParrotEncoder(bad_config)


# --- Test Suite for FunctionalParrotEncoder ---
# This class groups tests specifically for the FunctionalParrotEncoder, which uses
# user-provided Python functions for encoding/decoding.
class TestFunctionalEncoder:
    def test_initialization_success(self, encoder_data):
        """
        Tests if FunctionalParrotEncoder can be initialized successfully with valid configuration.
        Checks inheritance, length (input_size), and alphabet.
        """
        encoder = FunctionalParrotEncoder(encoder_data["functional_config"])
        assert isinstance(encoder, BaseParrotEncoder)
        assert len(encoder) == 2
        assert encoder.get_alphabet() == {'A', 'C'}

    def test_encode_decode_cycle(self, encoder_data):
        """
        Tests the full encode-decode cycle for the FunctionalParrotEncoder,
        using the custom functions defined in the temporary Python module.
        """
        encoder = FunctionalParrotEncoder(encoder_data["functional_config"])
        sequence = "ACAC"
        encoded = encoder.encode(sequence)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (4, 2) # (sequence_length, input_size)

        decoded = encoder.decode(encoded)
        assert decoded == [sequence]

    def test_encode_invalid_char(self, encoder_data):
        """
        Tests that encoding a sequence with an invalid character (not in the alphabet)
        raises a ValueError for the FunctionalParrotEncoder.
        """
        encoder = FunctionalParrotEncoder(encoder_data["functional_config"])
        with pytest.raises(ValueError, match="Invalid character 'G'"):
            encoder.encode("ACAG")

    def test_missing_module_raises_error(self, encoder_data):
        """
        Tests that initializing the FunctionalParrotEncoder with a non-existent
        module file path raises a FileNotFoundError.
        """
        bad_config = encoder_data["functional_config"].copy()
        bad_config.module_path = "non_existent_module.py"
        with pytest.raises(FileNotFoundError):
            FunctionalParrotEncoder(bad_config)

    def test_missing_function_raises_error(self, encoder_data):
        """
        Tests that if the specified encode function name is not found within the
        provided module, an AttributeError is raised.
        """
        bad_config = encoder_data["functional_config"].copy()
        bad_config.encode_function_name = "non_existent_function"
        with pytest.raises(AttributeError, match="Encode function 'non_existent_function' not found"):
            FunctionalParrotEncoder(bad_config)

    def test_missing_input_size_raises_error(self, encoder_data):
        """
        Tests that if 'input_size' is missing from the configuration for a
        functional encoder, a ValueError is raised.
        're.escape()' is crucial here because the error message contains parentheses
        which are special characters in regular expressions. Escaping them ensures
        they are treated as literal characters in the match pattern.
        """
        bad_config = encoder_data["functional_config"].copy()
        del bad_config.input_size
        with pytest.raises(ValueError, match=re.escape("For 'function' encoder, 'input_size' (a positive integer) must be specified.")):
            FunctionalParrotEncoder(bad_config)


# --- Test Suite for ParrotLightningEncoderFactory ---
# This class tests the top-level ParrotLightningEncoder, which acts as a factory
# or dispatcher, creating and delegating to the appropriate concrete encoder.
class TestParrotLightningEncoderFactory:
    def test_factory_creates_table_encoder(self, encoder_data):
        """
        Tests that the ParrotLightningEncoder correctly instantiates a TableParrotEncoder
        when given a 'table' type configuration.
        """
        factory_encoder = ParrotLightningEncoder(encoder_data["table_config"])
        assert factory_encoder.encoder_type == "table"
        assert len(factory_encoder) == 3

    def test_factory_creates_functional_encoder(self, encoder_data):
        """
        Tests that the ParrotLightningEncoder correctly instantiates a FunctionalParrotEncoder
        when given a 'function' type configuration.
        """
        factory_encoder = ParrotLightningEncoder(encoder_data["functional_config"])
        assert factory_encoder.encoder_type == "function"
        assert len(factory_encoder) == 2

    def test_factory_delegates_encode(self, encoder_data):
        """
        Tests that the ParrotLightningEncoder correctly delegates the 'encode' call
        to the underlying concrete encoder (TableParrotEncoder in this case) and
        produces the expected output tensor.
        'torch.equal' is used for precise tensor comparison.
        """
        factory_encoder = ParrotLightningEncoder(encoder_data["table_config"])
        sequence = "ACG"
        encoded = factory_encoder.encode(sequence)
        
        # Define the expected output tensor based on the 'test_table.tsv' content.
        expected_tensor = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        assert torch.equal(encoded, expected_tensor)

    def test_factory_delegates_decode(self, encoder_data):
        """
        Tests that the ParrotLightningEncoder correctly delegates the 'decode' call
        to the underlying concrete encoder.
        """
        factory_encoder = ParrotLightningEncoder(encoder_data["table_config"])
        sequence = "GCA"
        encoded = factory_encoder.encode(sequence)
        decoded = factory_encoder.decode(encoded)
        assert decoded == [sequence]

    def test_factory_raises_on_unknown_type(self):
        """
        Tests that providing an unsupported 'type' in the configuration to the
        ParrotLightningEncoder raises a ValueError.
        """
        bad_config = OmegaConf.create({"type": "unknown_encoder"})
        with pytest.raises(ValueError, match="Unsupported encoder type: 'unknown_encoder'"):
            ParrotLightningEncoder(bad_config)

    def test_factory_raises_on_missing_type(self):
        """
        Tests that if the 'type' field is missing from the configuration,
        ParrotLightningEncoder raises a ValueError.
        """
        bad_config = OmegaConf.create({"alphabet": "AC"})
        with pytest.raises(ValueError, match="Encoder 'type' must be specified"):
            ParrotLightningEncoder(bad_config)

    def test_integration_with_sequencedataset(self, encoder_data):
        """
        A simple integration test to demonstrate how the `ParrotLightningEncoder`
        would be used by the `SequenceDataset` (or any other component that needs
        to encode sequences).

        This test doesn't fully mock `SequenceDataset` but simulates the key
        interaction: passing a sequence to the encoder's `encode` method.
        It verifies that the encoder can be instantiated and its `encode` method
        can be called, returning a tensor of the expected shape.
        """
        # In a real scenario, SequenceDataset would be imported and initialized
        # with this encoder object. For this test, we focus on the encoder's role.
        # Example: dataset = SequenceDataset(..., encoder=encoder, ...)

        # Create the encoder
        encoder = ParrotLightningEncoder(encoder_data["table_config"])

        # Simulate how SequenceDataset's __getitem__ or a similar function
        # would use the encoder:
        sequence = "ACG"
        # The 'if encoder:' check is a common pattern to ensure the encoder is present.
        if encoder:
            sequence_vector = encoder.encode(sequence)
        
        # Assert that the encoding was successful and the output has the correct shape.
        assert sequence_vector is not None
        assert sequence_vector.shape == (3, 3) # (sequence_length, input_size)

    def test_multi_sequence_encoding(self, encoder_data):
        """
        Test the multi-sequence encoding functionality.
        """
        encoder = ParrotLightningEncoder(encoder_data["table_config"])
        sequences = ["ACG", "GCA", "AC"]
        
        # Test encode_sequences
        encoded_list = encoder.encode_sequences(sequences)
        assert len(encoded_list) == 3
        assert encoded_list[0].shape == (3, 3)  # First sequence "ACG"
        assert encoded_list[1].shape == (3, 3)  # Second sequence "GCA"
        assert encoded_list[2].shape == (2, 3)  # Third sequence "AC"
        
        # Test encode_sequences_padded
        padded_tensor = encoder.encode_sequences_padded(sequences)
        assert padded_tensor.shape == (3, 3, 3)  # (batch_size, max_seq_len, input_size)
        
        # Verify padding worked correctly
        # The third sequence should be padded with zeros in the last position
        assert torch.all(padded_tensor[2, 2, :] == 0.0)  # Last position should be padding
        
        # Verify that non-padded parts match original encodings
        assert torch.equal(padded_tensor[0], encoded_list[0])
        assert torch.equal(padded_tensor[1], encoded_list[1])
        assert torch.equal(padded_tensor[2, :2, :], encoded_list[2])

    def test_multi_sequence_encoding_functional(self, encoder_data):
        """
        Test the multi-sequence encoding functionality with functional encoder.
        """
        encoder = ParrotLightningEncoder(encoder_data["functional_config"])
        sequences = ["AC", "CA", "A"]
        
        # Test encode_sequences
        encoded_list = encoder.encode_sequences(sequences)
        assert len(encoded_list) == 3
        assert encoded_list[0].shape == (2, 2)  # First sequence "AC"
        assert encoded_list[1].shape == (2, 2)  # Second sequence "CA"
        assert encoded_list[2].shape == (1, 2)  # Third sequence "A"
        
        # Test encode_sequences_padded
        padded_tensor = encoder.encode_sequences_padded(sequences)
        assert padded_tensor.shape == (3, 2, 2)  # (batch_size, max_seq_len, input_size)


# --- Test Suite for MatrixParrotEncoder ---
class TestMatrixEncoder:
    def test_initialization_vectorial_with_gap(self):
        """Test MatrixParrotEncoder initialization with vectorial encoding and gap character."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ACG",
            "gap_char": "*", 
            "use_gap": True,
            "encoding_type": "vectorial"
        })
        
        encoder = MatrixParrotEncoder(config)
        assert isinstance(encoder, BaseParrotEncoder)
        # Base alphabet is 3 chars, so input_size = 3*3 + 1 = 10 (+ 1 for gap dimension)
        assert len(encoder) == 10
        assert encoder.get_alphabet() == {'A', 'C', 'G', '*'}
        
        # Check that gap character is always last in mappings
        assert encoder._char_to_idx['*'] == 3  # Should be index 3 (after A=0, C=1, G=2)

    def test_initialization_vectorial_without_gap(self):
        """Test MatrixParrotEncoder initialization with vectorial encoding and no gap character."""
        config = OmegaConf.create({
            "type": "matrix", 
            "alphabet": "ACG",
            "use_gap": False,
            "encoding_type": "vectorial"
        })
        
        encoder = MatrixParrotEncoder(config)
        assert len(encoder) == 9  # 3*3 = 9 
        assert encoder.get_alphabet() == {'A', 'C', 'G'}

    def test_initialization_numerical_with_gap(self):
        """Test MatrixParrotEncoder initialization with numerical encoding and gap character.""" 
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ACG",
            "gap_char": "*",
            "use_gap": True,
            "encoding_type": "numerical"
        })
        
        encoder = MatrixParrotEncoder(config)
        assert len(encoder) == 1  # Numerical encoding always has input_size = 1
        assert encoder.get_alphabet() == {'A', 'C', 'G', '*'}

    def test_gap_character_not_in_initial_alphabet(self):
        """Test that gap character is properly handled even if included in initial alphabet."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ACG*",  # Gap char included in alphabet
            "gap_char": "*",
            "use_gap": True,
            "encoding_type": "vectorial"
        })
        
        encoder = MatrixParrotEncoder(config)
        # Should still have correct input size: 3*3 + 1 = 10
        assert len(encoder) == 10
        # Gap character should still be last
        assert encoder._char_to_idx['*'] == 3

    def test_custom_gap_character_ordering(self):
        """Test that custom gap character is always placed last regardless of alphabetical order."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ZAB",  # Z would normally be last alphabetically
            "gap_char": "A",   # But A is our gap character
            "use_gap": True,
            "encoding_type": "vectorial"
        })
        
        encoder = MatrixParrotEncoder(config)
        # Base alphabet should be {'Z', 'B'} (A removed as it's the gap char)
        # So input_size = 2*2 + 1 = 5
        assert len(encoder) == 5
        assert encoder.get_alphabet() == {'Z', 'B', 'A'}
        # Gap character (A) should be last despite alphabetical order
        assert encoder._char_to_idx['A'] == 2  # Last index

    def test_encode_decode_cycle_vectorial(self):
        """Test full encode-decode cycle with vectorial encoding."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "AC",
            "use_gap": False,
            "encoding_type": "vectorial"
        })
        
        encoder = MatrixParrotEncoder(config)
        sequence = "AC" 
        encoded = encoder.encode(sequence)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (2, 2, 4)  # (seq_len, seq_len, input_size)
        
        decoded = encoder.decode(encoded)
        assert decoded == [sequence]

    def test_encode_decode_cycle_numerical(self):
        """Test full encode-decode cycle with numerical encoding."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "AC",
            "use_gap": False,
            "encoding_type": "numerical"
        })
        
        encoder = MatrixParrotEncoder(config)
        sequence = "AC"
        encoded = encoder.encode(sequence)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (2, 2, 1)  # (seq_len, seq_len, input_size=1)
        
        decoded = encoder.decode(encoded)
        assert decoded == [sequence]

    def test_factory_creates_matrix_encoder(self):
        """Test that ParrotLightningEncoder correctly creates MatrixParrotEncoder."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ACG",
            "use_gap": True,
            "encoding_type": "vectorial"
        })
        
        factory_encoder = ParrotLightningEncoder(config)
        assert factory_encoder.encoder_type == "matrix"
        assert len(factory_encoder) == 10  # 3*3 + 1 = 10

    def test_multi_sequence_encoding_matrix(self):
        """Test multi-sequence encoding functionality with matrix encoder."""
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "AC",
            "use_gap": False,
            "encoding_type": "vectorial"
        })
        
        encoder = ParrotLightningEncoder(config)
        sequences = ["AC", "CA", "A"]
        
        # Test encode_sequences
        encoded_list = encoder.encode_sequences(sequences)
        assert len(encoded_list) == 3
        assert encoded_list[0].shape == (2, 2, 4)  # First sequence "AC"
        assert encoded_list[1].shape == (2, 2, 4)  # Second sequence "CA"
        assert encoded_list[2].shape == (1, 1, 4)  # Third sequence "A"
        
        # Test encode_sequences_padded
        padded_tensor = encoder.encode_sequences_padded(sequences)
        assert padded_tensor.shape == (3, 2, 2, 4)  # (batch_size, max_seq_len, max_seq_len, input_size)
        
        # Verify that non-padded parts match original encodings
        assert torch.equal(padded_tensor[0], encoded_list[0])
        assert torch.equal(padded_tensor[1], encoded_list[1])
        assert torch.equal(padded_tensor[2, :1, :1, :], encoded_list[2])


# --- Test Suite for Save/Load Functionality ---
class TestEncoderSaveLoad:
    """Test suite for save/load functionality across all encoder types."""
    
    def test_table_encoder_save_load_with_file(self, encoder_data, tmp_path):
        """Test save/load cycle for TableParrotEncoder created from file."""
        # Create original encoder
        original_encoder = TableParrotEncoder(encoder_data["table_config"])
        
        # Test a sequence
        test_sequence = "ACG"
        original_encoded = original_encoder.encode(test_sequence)
        original_decoded = original_encoder.decode(original_encoded)
        
        # Save the encoder
        save_path = tmp_path / "table_encoder.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Verify the loaded encoder is the correct type
        assert isinstance(loaded_encoder, TableParrotEncoder)
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        
        # Test that encoding/decoding works the same
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        assert torch.equal(original_encoded, loaded_encoded)
        assert original_decoded == loaded_decoded

    def test_table_encoder_save_load_alphabet_only(self, tmp_path):
        """Test save/load cycle for TableParrotEncoder created from alphabet only."""
        from omegaconf import OmegaConf
        
        # Create encoder with alphabet only (one-hot encoding)
        config = OmegaConf.create({
            "type": "table",
            "alphabet": "ACGT"
        })
        original_encoder = TableParrotEncoder(config)
        
        # Test a sequence
        test_sequence = "ACGT"
        original_encoded = original_encoder.encode(test_sequence)
        
        # Save the encoder
        save_path = tmp_path / "table_encoder_alphabet.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Verify the loaded encoder works the same
        assert isinstance(loaded_encoder, TableParrotEncoder)
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        
        loaded_encoded = loaded_encoder.encode(test_sequence)
        assert torch.equal(original_encoded, loaded_encoded)

    def test_functional_encoder_save_load(self, encoder_data, tmp_path):
        """Test save/load cycle for FunctionalParrotEncoder."""
        # Create original encoder
        original_encoder = FunctionalParrotEncoder(encoder_data["functional_config"])
        
        # Test a sequence
        test_sequence = "ACAC"
        original_encoded = original_encoder.encode(test_sequence)
        original_decoded = original_encoder.decode(original_encoded)
        
        # Save the encoder
        save_path = tmp_path / "functional_encoder.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Verify the loaded encoder is the correct type
        assert isinstance(loaded_encoder, FunctionalParrotEncoder)
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        assert loaded_encoder.module_path == original_encoder.module_path
        
        # Test that encoding/decoding works the same
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        assert torch.equal(original_encoded, loaded_encoded)
        assert original_decoded == loaded_decoded

    def test_matrix_encoder_save_load_vectorial(self, tmp_path):
        """Test save/load cycle for MatrixParrotEncoder with vectorial encoding."""
        from omegaconf import OmegaConf
        
        # Create original encoder
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "ACG",
            "gap_char": "*",
            "use_gap": True,
            "encoding_type": "vectorial"
        })
        original_encoder = MatrixParrotEncoder(config)
        
        # Test a sequence
        test_sequence = "AC*G"
        original_encoded = original_encoder.encode(test_sequence)
        original_decoded = original_encoder.decode(original_encoded)
        
        # Save the encoder
        save_path = tmp_path / "matrix_encoder_vectorial.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Verify the loaded encoder is the correct type
        assert isinstance(loaded_encoder, MatrixParrotEncoder)
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        assert loaded_encoder.encoding_type == original_encoder.encoding_type
        assert loaded_encoder.use_gap == original_encoder.use_gap
        assert loaded_encoder.gap_char == original_encoder.gap_char
        
        # Test that encoding/decoding works the same
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        assert torch.equal(original_encoded, loaded_encoded)
        assert original_decoded == loaded_decoded

    def test_matrix_encoder_save_load_numerical(self, tmp_path):
        """Test save/load cycle for MatrixParrotEncoder with numerical encoding."""
        from omegaconf import OmegaConf
        
        # Create original encoder
        config = OmegaConf.create({
            "type": "matrix",
            "alphabet": "AC",
            "use_gap": False,
            "encoding_type": "numerical"
        })
        original_encoder = MatrixParrotEncoder(config)
        
        # Test a sequence
        test_sequence = "ACAC"
        original_encoded = original_encoder.encode(test_sequence)
        original_decoded = original_encoder.decode(original_encoded)
        
        # Save the encoder
        save_path = tmp_path / "matrix_encoder_numerical.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Verify the loaded encoder works the same
        assert isinstance(loaded_encoder, MatrixParrotEncoder)
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        assert loaded_encoder.encoding_type == original_encoder.encoding_type
        
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        assert torch.equal(original_encoded, loaded_encoded)
        assert original_decoded == loaded_decoded

    def test_parrot_lightning_encoder_save_load(self, encoder_data, tmp_path):
        """Test save/load cycle for ParrotLightningEncoder."""
        # Create original encoder
        original_encoder = ParrotLightningEncoder(encoder_data["table_config"])
        
        # Test a sequence
        test_sequence = "ACG"
        original_encoded = original_encoder.encode(test_sequence)
        original_decoded = original_encoder.decode(original_encoded)
        
        # Save the encoder
        save_path = tmp_path / "lightning_encoder.pkl"
        original_encoder.save(str(save_path))
        
        # Load the encoder
        loaded_encoder = ParrotLightningEncoder.load(str(save_path))
        
        # Verify the loaded encoder is the correct type
        assert isinstance(loaded_encoder, ParrotLightningEncoder)
        assert loaded_encoder.encoder_type == original_encoder.encoder_type
        assert loaded_encoder.get_alphabet() == original_encoder.get_alphabet()
        assert len(loaded_encoder) == len(original_encoder)
        
        # Test that encoding/decoding works the same
        loaded_encoded = loaded_encoder.encode(test_sequence)
        loaded_decoded = loaded_encoder.decode(loaded_encoded)
        
        assert torch.equal(original_encoded, loaded_encoded)
        assert original_decoded == loaded_decoded

    def test_save_config_json(self, encoder_data, tmp_path):
        """Test saving encoder configuration to JSON."""
        # Test with table encoder
        encoder = TableParrotEncoder(encoder_data["table_config"])
        config_path = tmp_path / "table_config.json"
        
        encoder.save_config(str(config_path))
        
        # Verify the file was created and contains expected content
        assert config_path.exists()
        
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        assert config_data['encoder_class'] == 'TableParrotEncoder'
        assert 'alphabet' in config_data
        assert 'input_size' in config_data

    def test_save_config_lightning_encoder(self, encoder_data, tmp_path):
        """Test saving configuration through ParrotLightningEncoder."""
        encoder = ParrotLightningEncoder(encoder_data["functional_config"])
        config_path = tmp_path / "functional_config.json"
        
        encoder.save_config(str(config_path))
        
        assert config_path.exists()
        
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        assert config_data['encoder_class'] == 'FunctionalParrotEncoder'

    def test_load_nonexistent_file(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Encoder file not found"):
            BaseParrotEncoder.load("nonexistent_file.pkl")

    def test_load_corrupted_file(self, tmp_path):
        """Test that loading corrupted file raises IOError."""
        corrupted_path = tmp_path / "corrupted.pkl"
        corrupted_path.write_text("This is not a valid pickle file")
        
        with pytest.raises(IOError, match="Failed to load encoder"):
            BaseParrotEncoder.load(str(corrupted_path))

    def test_load_invalid_encoder_class(self, tmp_path):
        """Test that loading file with invalid encoder class raises ValueError."""
        import pickle
        
        invalid_path = tmp_path / "invalid.pkl"
        # Create a pickle file with invalid encoder class
        invalid_data = {'encoder_class': 'NonExistentEncoder'}
        
        with open(invalid_path, 'wb') as f:
            pickle.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="Unknown encoder class"):
            BaseParrotEncoder.load(str(invalid_path))

    def test_load_missing_encoder_class(self, tmp_path):
        """Test that loading file without encoder class info raises ValueError."""
        import pickle
        
        invalid_path = tmp_path / "missing_class.pkl"
        # Create a pickle file without encoder class info
        invalid_data = {'some_data': 'value'}
        
        with open(invalid_path, 'wb') as f:
            pickle.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="Invalid encoder file: missing encoder class information"):
            BaseParrotEncoder.load(str(invalid_path))

    def test_save_io_error(self, encoder_data):
        """Test that save operation handles IO errors gracefully."""
        encoder = TableParrotEncoder(encoder_data["table_config"])
        
        # Try to save to an invalid path (directory that doesn't exist and can't be created)
        with pytest.raises(IOError, match="Failed to save encoder"):
            encoder.save("/invalid/path/that/cannot/exist/encoder.pkl")

    def test_round_trip_preserves_state(self, encoder_data, tmp_path):
        """Test that a complete save/load cycle preserves all encoder state."""
        # Test with a functional encoder as it has the most complex state
        original_encoder = FunctionalParrotEncoder(encoder_data["functional_config"])
        
        # Save and load
        save_path = tmp_path / "state_test.pkl"
        original_encoder.save(str(save_path))
        loaded_encoder = BaseParrotEncoder.load(str(save_path))
        
        # Check that all important attributes are preserved
        assert loaded_encoder.alphabet == original_encoder.alphabet
        assert loaded_encoder.input_size == original_encoder.input_size
        assert loaded_encoder.module_path == original_encoder.module_path
        assert loaded_encoder.encode_fn_name == original_encoder.encode_fn_name
        assert loaded_encoder.decode_fn_name == original_encoder.decode_fn_name
        
        # Test that the function callables work the same
        test_seq = "AC"
        original_result = original_encoder.encode(test_seq)
        loaded_result = loaded_encoder.encode(test_seq)
        assert torch.equal(original_result, loaded_result)