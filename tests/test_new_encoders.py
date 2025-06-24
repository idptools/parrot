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