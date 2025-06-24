"""
File containing functions for encoding a string of amino acids into a numeric vector.

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""
# import the hydra stuff
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os
import importlib.util
from typing import Union, List, Dict, Any, Callable, Tuple

import numpy as np
import torch

ONE_HOT = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 
            'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 
            'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19 }

def one_hot(seq):
    """Convert an amino acid sequence to a PyTorch tensor of one-hot vectors

    Each amino acid is represented by a length 20 vector with a single 1 and
    19 0's Inputing a sequence with a nono-canonical amino acid letter will
    cause the program to exit.

    E.g. Glutamic acid (E) is encoded: [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    Parameters
    ----------
    seq : str
            An uppercase sequence of amino acids (single letter code)

    Returns
    -------
    torch.IntTensor
            a PyTorch tensor representing the encoded sequence
    """

    l = len(seq)
    m = np.zeros((l, 20))
    try:
        for i in range(l):
            m[i, ONE_HOT[seq[i]]] = 1
    except:
        error_str = 'Invalid amino acid detected: ' + seq[i]
        raise ValueError(error_str)
    return torch.from_numpy(m)


def rev_one_hot(seq_vectors):
    """Decode a list of one-hot sequence vectors into amino acid sequences

    Parameters
    ----------
    seq_vectors : list of numpy arrays
            A list containing sequence vectors

    Returns
    -------
    list
            Strings of amino acid sequences
    """

    REV_ONE_HOT = 'ACDEFGHIKLMNPQRSTVWY'
    sequences = []

    for seq_vector in seq_vectors:
        seq = []
        for residue in seq_vector:
            seq.append(REV_ONE_HOT[np.argmax(residue)])
        sequences.append("".join(seq))

    return sequences


############  Biophysical scale encoding  #############

# Map each amino acid to a vector of biophysical properties
# 0: Hydrophobicity
# 1: Charge
# 2: pI
# 3: Molecular weight (g/mol)
# 4: Aromatic amino acid
# 5: Capable of hydrogen bonding
# 6: Side chain SASA (measured from ACE-XX-NME dipeptide)
# 7: Backbone SASA (measured from ACE-XX-NME dipeptide)
# 8: Free energy of solvation
BIOPHYSICS = {  'A': [1.8,  0,  6.0,  89.1, 0, 0,  75.8,  76.1,    1.9],
                'C': [2.5,  0,  5.1, 121.2, 0, 0, 115.4,  67.9,   -1.2],
                'D': [-3.5, -1,  2.8, 133.1, 0, 1, 130.3,  71.8, -107.3],
                'E': [-3.5, -1,  3.2, 147.1, 0, 1, 161.8,  68.1, -107.3],
                'F': [2.8,  0,  5.5, 165.2, 1, 0, 209.4,  66.0,   -0.8],
                'G': [-0.4,  0,  6.0,  75.1, 0, 0,   0.0, 115.0,    0.0],
                'H': [-3.2,  1,  7.6, 155.2, 0, 1, 180.8,  67.5,  -52.7],  # Avg of HIP and HIE
                'I': [4.5,  0,  6.0, 131.2, 0, 0, 172.7,  60.3,    2.2],
                'K': [-3.9,  1,  9.7, 146.2, 0, 1, 205.9,  68.7, -100.9],
                'L': [3.8,  0,  6.0, 131.2, 0, 0, 172.0,  64.5,    2.3],
                'M': [1.9,  0,  5.7, 149.2, 0, 0, 184.8,  67.8,   -1.4],
                'N': [-3.5,  0,  5.4, 132.1, 0, 1, 142.7,  66.8,   -9.7],
                'P': [-1.6,  0,  6.3, 115.1, 0, 0, 134.3,  55.8,    2.0],
                'Q': [-3.5,  0,  5.7, 146.2, 0, 1, 173.3,  66.6,   -9.4],
                'R': [-4.5,  1, 10.8, 174.2, 0, 1, 236.5,  66.7, -100.9],
                'S': [-0.8,  0,  5.7, 105.1, 0, 1,  95.9,  72.9,   -5.1],
                'T': [-0.7,  0,  5.6, 119.1, 0, 1, 130.9,  64.1,   -5.0],
                'V': [4.2,  0,  6.0, 117.1, 0, 0, 143.1,  61.7,    2.0],
                'W': [-0.9,  0,  5.9, 204.2, 1, 1, 254.6,  64.3,   -5.9],
                'Y': [-1.3,  0,  5.7, 181.2, 1, 1, 222.5,  71.9,   -6.1]
                }

def biophysics(seq):
    """Convert an amino acid sequence to a PyTorch tensor with biophysical encoding

    Each amino acid is represented by a length 9 vector with each value representing
    a biophysical property. The nine encoded biophysical scales are Kyte-Doolittle
    hydrophobicity, charge, isoelectric point, molecular weight, aromaticity, 
    h-bonding ability, side chain solvent accessible surface area, backbone SASA, and 
    free energy of solvation. Inputing a sequence with a nono-canonical amino acid 
    letter will cause the program to exit.

    E.g. Glutamic acid (E) is: [-3.5, -1,  3.2, 147.1, 0, 1, 161.8,  68.1, -107.3]

    Parameters
    ----------
    seq : str
            An uppercase sequence of amino acids (single letter code)

    Returns
    -------
    torch.FloatTensor
            a PyTorch tensor representing the encoded sequence
    """
    l = len(seq)
    m = np.zeros((l, len(BIOPHYSICS['A'])))
    try:
        for i in range(l):
            m[i] = BIOPHYSICS[seq[i]]
    except:
        error_str = 'Invalid amino acid detected: ' + seq[i]
        raise ValueError(error_str)
    return torch.from_numpy(m)


def rev_biophysics(seq_vectors):
    """Decode a list of biophysically-encoded sequence vectors into amino acid sequences

    Parameters
    ----------
    seq_vectors : list of numpy arrays
            A list containing sequence vectors

    Returns
    -------
    list
            Strings of amino acid sequences
    """

    REV_BIOPHYSICS = {}
    for key, value in BIOPHYSICS.items():
        REV_BIOPHYSICS[str(value[6])] = key

    sequences = []
    for seq_vector in seq_vectors:
        seq = []
        for residue in seq_vector:
            seq.append(REV_BIOPHYSICS[str(residue[6])])
        sequences.append("".join(seq))

    return sequences


################## User-specified encoding ####################

def parse_encode_file(file):
    """Helper function to convert an encoding file into key:value dictionary"""

    with open(file) as f:
        lines = [x.strip().split() for x in f]

    l = len(lines[0]) - 1
    d = {}
    for line in lines:
        d[line[0]] = line[1:]

        if len(line) - 1 != l:
            raise ValueError('Vectors in encoding file do not have same length.')

    return d, l


class UserEncoder():
    """User-specified amino acid-to-vector encoding scheme object

    Attributes
    ----------
    encode_file : str
            A path to a file that describes the encoding scheme
    encode_dict : dict
            A dictionary that maps each amino acid to a numeric vector
    _encoding_dimensions : int
            The length of the encoding vector used for each amino acid
    """

    def __init__(self, encode_file):
        """
        Parameters
        ----------
        encode_file : str
                A path to a file that describes the encoding scheme
        """

        self.encode_file = os.path.abspath(encode_file)
        if not os.path.isfile(self.encode_file):
            raise FileNotFoundError('Encoding file does not exist.')

        self.encode_dict, self.input_size = parse_encode_file(self.encode_file)

    def __len__(self):
        """Get length of encoding scheme"""

        return self.input_size

    def encode(self, seq):
        """Convert an amino acid sequence into this encoding scheme

        Parameters
        ----------
        seq : str
                An uppercase sequence of amino acids (single letter code)

        Returns
        -------
        torch.FloatTensor
                a PyTorch tensor representing the encoded sequence
        """

        l = len(seq)
        m = np.zeros((l, self.input_size))

        try:
            for i in range(l):
                m[i] = self.encode_dict[seq[i]]
        except:
            error_str = 'Invalid amino acid detected: ' + seq[i]
            raise ValueError(error_str)
        return torch.from_numpy(m)

    def decode(self, seq_vectors):
        """Converts a list of sequence vectors back to a list of protein sequences

        Parameters
        ----------
        seq_vectors : list of numpy arrays
                A list containing sequence vectors

        Returns
        -------
        list
                Strings of amino acid sequences
        """

        # Create a reverse of the encode_dict using hashing
        rev_dict = {}
        for key, value in self.encode_dict.items():
            value = np.array(value, dtype=np.float32)
            rev_dict[hash(tuple(value))] = key

        sequences = []
        for seq_vector in seq_vectors:
            seq = []
            for residue in seq_vector:
                seq.append(rev_dict[hash(tuple(residue))])
            sequences.append("".join(seq))

        return sequences
    




"""
Nick's New Implimentation

This section works based on an abstract encoder class. This class is the template for all the 
functions that an encoder needs to be able to interface with the rest of the codebase.

There are two concrete implementation of the base class: TableParrotEncoder and FunctionalParrotEncoder.

The first is for encodings that can be represented by a lookup table. This means the mapping occurs for 
each residue and it never changes. One hot encoding is an example of this type of encoding scheme.

The second is for function based encodings. Basically you will provide the class a python module location
that houses some custom encoding and decoding. This allows for encodings based on ML models, non static encoding schemes, etc.

"""

def _new_parse_encode_file(filepath: str) -> Tuple[Dict[str, List[str]], int]:
    """
    Helper function to convert an encoding file into key:value dictionary.
    This is a self-contained version for the new encoder implementation that
    is more robust than the original.

    Args:
        filepath (str): The path to the encoding definition file (e.g., a TSV).

    Returns:
        Tuple[Dict[str, List[str]], int]: A tuple containing:
            - A dictionary mapping character keys to their string vector representations.
            - The expected length of the encoding vector for each character.
    """
    # Check that the file provided exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Encoding definition file not found: {filepath}")
    with open(filepath, 'r') as f:
        # loop over each line in the file
        # Ignore empty lines and comment lines
        lines = [x.strip().split() for x in f if x.strip() and not x.strip().startswith('#')]

    # Check that the encoding file is not empty
    if not lines:
        raise ValueError(f"Encoding file is empty or contains only comments: {filepath}")

    # Get the first line of data to determine what to expect the data should look like
    first_data_line = lines[0]
    # All lines must have at least a key and one value
    if len(first_data_line) < 2:
        raise ValueError(f"Encoding file lines must have at least one key and one value. Error on line 1: {first_data_line}")
    # Figure out the vector encoding length for the encoder
    expected_vector_len = len(first_data_line) - 1

    # create an encoding dictionary
    encode_dict = {}
    for i, line_parts in enumerate(lines):
        if len(line_parts) - 1 != expected_vector_len:
            raise ValueError(
                f"Inconsistent vector length in encoding file {filepath} at line {i+1}. "
                f"Expected {expected_vector_len} values, got {len(line_parts)-1}."
            )
        key = line_parts[0]
        # Store as strings initially, conversion to float happens in the TableParrotEncoder
        encode_dict[key] = line_parts[1:]
    return encode_dict, expected_vector_len


from abc import ABC, abstractmethod

class BaseParrotEncoder(ABC):
    """
    Abstract Base Class defining the interface for all PARROT sequence encoders.
    All concrete encoder implementations must inherit from this class and
    implement its abstract methods.

    This class ensures a consistent API for encoding and decoding operations,
    regardless of the underlying implementation (e.g., table-based, functional).
    """
    @abstractmethod
    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encodes an amino acid sequence string into a PyTorch tensor.

        Parameters:
            sequence (str): The amino acid sequence to encode.

        Returns:
            torch.Tensor: A tensor of shape (len(sequence), self.input_size)
                          with dtype torch.float32.
        """
        pass

    @abstractmethod
    def decode(self, seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        Decodes one or more sequence vectors back into amino acid sequence strings.

        Parameters:
            seq_vectors (Union[torch.Tensor, np.ndarray, List[np.ndarray]]):
                - A single 2D tensor/array: (seq_len, input_size)
                - A single 3D tensor/array (batch): (batch_size, seq_len, input_size)
                - A list of 2D numpy arrays: each (seq_len, input_size)

        Returns:
            List[str]: A list of decoded sequence strings.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the input_size (dimension of the encoded vector for a single character).
        """
        pass

    @abstractmethod
    def get_alphabet(self) -> set:
        """
        Returns the set of allowed characters for this encoder.
        """
        pass

    @staticmethod
    def _validate_sequence_chars(sequence: str, alphabet: set) -> str:
        """
        Validates characters in the sequence against the defined alphabet.
        Converts sequence to uppercase.

        Args:
            sequence (str): The input amino acid sequence.
            alphabet (set): The set of allowed characters.

        Returns:
            str: The validated (and uppercased) sequence.

        Raises:
            ValueError: If any character in the sequence is not in the alphabet.
        """
        for char_idx, char_val in enumerate(sequence):
            if char_val not in alphabet:
                raise ValueError(f"Invalid character '{char_val}' at position {char_idx} in sequence: {sequence}. \nNot in defined alphabet.")
        return sequence
    

    @staticmethod
    def _process_input_vectors(seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]], input_size: int) -> List[np.ndarray]:
        """
        Helper method to normalize various input vector formats into a list of 2D NumPy arrays.

        Args:
            seq_vectors (Union[torch.Tensor, np.ndarray, List[np.ndarray]]):
                Input vectors which can be a single 2D tensor/array, a 3D batch tensor/array,
                or a list of 2D numpy arrays.
            input_size (int): The expected dimension of the individual character vectors.

        Returns:
            List[np.ndarray]: A list of 2D NumPy arrays, each representing an encoded sequence.

        Raises:
            TypeError: If the input type for `seq_vectors` is unsupported.
            ValueError: If the dimensions of the input vectors are incorrect.
        """
        processed_vectors: List[np.ndarray] = []
        if isinstance(seq_vectors, torch.Tensor):
            # If input is a PyTorch tensor, convert it to a NumPy array.
            # This ensures consistent processing with NumPy operations.
            seq_vectors_np = seq_vectors.cpu().numpy()
        elif isinstance(seq_vectors, np.ndarray):
            # If input is already a NumPy array, use it directly.
            seq_vectors_np = seq_vectors
        elif isinstance(seq_vectors, list):
            # If input is a list, assume it's already a list of NumPy arrays
            # (or convertible to them) and assign directly.
            processed_vectors = seq_vectors
            seq_vectors_np = None # Indicate that no further NumPy conversion is needed for this path
        else:
            # Raise an error for unsupported input types.
            raise TypeError(f"Unsupported type for seq_vectors: {type(seq_vectors)}. Expected torch.Tensor, np.ndarray, or List[np.ndarray].")

        if seq_vectors_np is not None:
            if seq_vectors_np.ndim == 2:
                # If 2D (e.g., a single sequence: seq_len x input_size),
                # add it as a single item to the processed list.
                processed_vectors.append(seq_vectors_np)
            elif seq_vectors_np.ndim == 3:
                # If 3D (e.g., a batch of sequences: batch_size x seq_len x input_size),
                # iterate through the batch dimension and add each 2D sequence.
                for i in range(seq_vectors_np.shape[0]):
                    processed_vectors.append(seq_vectors_np[i, :, :])
            # If 1D, it's likely a single character vector, but decode expects sequence
            else:
                # Raise an error for unsupported NumPy array dimensions.
                raise ValueError(f"Unsupported ndim for seq_vectors: {seq_vectors_np.ndim}. Expected 2 or 3.")
        
        for vec in processed_vectors:
            # Validate each processed vector: ensure it's a NumPy array, 2D,
            # and its second dimension matches the expected input_size.
            if not isinstance(vec, np.ndarray) or vec.ndim != 2 or vec.shape[1] != input_size:
                raise ValueError(f"Invalid vector format. Each must be a 2D NumPy array of shape (seq_len, {input_size}). Got shape {vec.shape if hasattr(vec, 'shape') else 'N/A'}")
        
        return processed_vectors

    




class TableParrotEncoder(BaseParrotEncoder):
    """
    A concrete encoder implementation that uses a lookup table (TSV file)
    to encode and decode amino acid sequences.

    Attributes:
        is_uniquely_decodable (bool): True if the encoding allows unique decoding, False otherwise.
        table_file_path (Optional[str]): Path to the TSV file used for encoding, if provided.
        input_size (int): The dimension of the encoded vector for each character.
        alphabet (set): The set of allowed characters for this encoder.
        _table_encode_dict (Dict[str, List[float]]): Internal dictionary for encoding characters to vectors.
        _table_decode_dict (Dict[int, str]): Internal dictionary for decoding vector hashes back to characters.
    """
    def __init__(self, config: DictConfig):
        # Flag to track if decoding is unique (can be false if multiple chars map to same vector)
        self.is_uniquely_decodable = True
        # Get file path and alphabet from configuration
        table_file_path_str = config.get("table_file_path")
        alphabet_str = config.get("alphabet")

        # Infer whether the table or alphabet was provided and how to deal with that
        # If both are provided it defaults to the table
        if table_file_path_str:
            # Case 1: A table file is provided.
            # Get the absolute path to the tsv file
            self.table_file_path = hydra.utils.to_absolute_path(table_file_path_str)
            raw_encode_dict, self.input_size = _new_parse_encode_file(self.table_file_path)

            if alphabet_str:
                # Alphabet is also provided, use it for validation.
                self.alphabet = set(list(alphabet_str))
            else:
                # Infer alphabet from the keys of the table file.
                self.alphabet = set(raw_encode_dict.keys())
            # Convert string vector values to floats

            self._table_encode_dict = {k: [float(v) for v in val_list] for k, val_list in raw_encode_dict.items()}
            self._validate_table_against_alphabet()

        elif alphabet_str:
            # Case 2: Only an alphabet is provided. Generate a one-hot encoding.
            self.alphabet = set(list(alphabet_str))
            if not self.alphabet:
                raise ValueError("Provided 'alphabet' string cannot be empty.")
            
            # Input size is the size of the alphabet for one-hot encoding
            self.input_size = len(self.alphabet)
            self.table_file_path = None
            
            self._table_encode_dict = {}
            # Create a one-hot encoding table based on the sorted alphabet
            sorted_alphabet = sorted(list(self.alphabet))
            for i, char in enumerate(sorted_alphabet):
                vector = [0.0] * self.input_size
                vector[i] = 1.0
                self._table_encode_dict[char] = vector
        
        else:
            # Case 3: Neither is provided. This is an error.
            raise ValueError("For 'table' encoder, either 'table_file_path' or 'alphabet' must be specified in the configuration.")

        # This is common to both Case 1 and Case 2.
        self._table_decode_dict: Dict[int, str] = self._create_decode_map(self._table_encode_dict)

    def _create_decode_map(self, encode_dict: Dict[str, List[float]]) -> Dict[int, str]:
        """
        Creates a reverse mapping from hashed vector tuples to characters for decoding.

        Args:
            encode_dict (Dict[str, List[float]]): The encoding dictionary.

        Returns:
            Dict[int, str]: A dictionary mapping hash of vector tuples to characters.
        """
        decode_dict = {}
        for char, vector_values in encode_dict.items():
            # Convert list of floats to a tuple for hashing
            key_tuple = tuple(vector_values)
            # Hash the tuple to use as a key for the decode dictionary
            h = hash(key_tuple)
            # Check for duplicate vectors mapping to different characters (non-unique decoding)
            if h in decode_dict and decode_dict[h] != char:
                print(f"Warning: Duplicate vector found for characters '{decode_dict[h]}' and '{char}'. Decoding is not unique.")
                self.is_uniquely_decodable = False
            decode_dict[h] = char
        return decode_dict

    def _validate_table_against_alphabet(self):
        """
        Validates that all characters in the specified alphabet are present in the table
        and warns if table characters are not in the specified alphabet.
        """
        # This validation is only meaningful if a table file was provided.
        if self.table_file_path is None:
            return

        for char_in_alphabet in self.alphabet:
            # Ensure every character in the specified alphabet has an entry in the table
            if char_in_alphabet not in self._table_encode_dict:
                raise ValueError(f"Character '{char_in_alphabet}' from specified alphabet not found in table file '{self.table_file_path}'.")
        for char_in_table in self._table_encode_dict.keys():
            # Warn if characters in the table are not part of the specified alphabet
            if char_in_table not in self.alphabet:
                print(f"Warning: Character '{char_in_table}' from table file '{self.table_file_path}' is not in the specified alphabet. It will not be encodable if it appears in a sequence.")

    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encodes an amino acid sequence string into a PyTorch tensor using the lookup table.

        Args:
            sequence (str): The amino acid sequence to encode.

        Returns:
            torch.Tensor: A tensor of shape (len(sequence), self.input_size)
                          with dtype torch.float32.
        """
        # validate that the sequence is composed of valid characters that are in the alphabet
        validated_sequence = self._validate_sequence_chars(sequence, self.alphabet)
        # for each character find its value in the lookup table
        vectors = [self._table_encode_dict[char] for char in validated_sequence]
        # convert the encoded sequence to a torch tensor and return
        return torch.tensor(vectors, dtype=torch.float32)

    def decode(self, seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        Decodes one or more sequence vectors back into amino acid sequence strings
        using the reverse lookup table.

        Args:
            seq_vectors (Union[torch.Tensor, np.ndarray, List[np.ndarray]]):
                Encoded sequence vectors.

        Returns:
            List[str]: A list of decoded sequence strings.

        Raises:
            ValueError: If a vector cannot be decoded (not found in the reverse map).
        """
        processed_vectors: List[np.ndarray] = BaseParrotEncoder._process_input_vectors(seq_vectors, self.input_size)
        decoded_sequences: List[str] = []
        for vec_np in processed_vectors:
            chars = []
            for i in range(vec_np.shape[0]):
                vector_tuple = tuple(vec_np[i, :].tolist())
                char = self._table_decode_dict.get(hash(vector_tuple))
                if char is None:
                    raise ValueError(f"Cannot decode vector {vec_np[i,:]}: not found in table's reverse map.")
                chars.append(char)
            decoded_sequences.append("".join(chars))
        return decoded_sequences

    def __len__(self) -> int:
        """Returns the input_size (dimension of the encoded vector for a single character)."""
        return self.input_size

    def get_alphabet(self) -> set:
        """Returns the set of allowed characters for this encoder."""
        return self.alphabet


class FunctionalParrotEncoder(BaseParrotEncoder):
    """
    A concrete encoder implementation that uses user-provided Python functions
    for encoding and decoding sequences.

    Attributes:
        alphabet (set): The set of allowed characters for this encoder.
        input_size (int): The dimension of the encoded vector for each character.
        module_path (str): Absolute path to the Python module containing the custom functions.
        encode_fn_name (str): Name of the encoding function within the module.
        decode_fn_name (str): Name of the decoding function within the module.
        _encode_callable (Callable): The loaded encoding function.
        _decode_callable (Callable): The loaded decoding function.
    """
    def __init__(self, config: DictConfig):
        # Extract alphabet from config and convert to a set
        self.alphabet = set(list(config.alphabet))
        # Get input_size (dimension of the encoded vector)
        self.input_size = config.get("input_size") # Renamed from encoding_dimensions
        if not self.input_size or not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError("For 'function' encoder, 'input_size' (a positive integer) must be specified.")
        
        module_path_str = config.get("module_path")
        if not module_path_str:
            raise ValueError("For 'function' encoder, 'module_path' must be specified.")
        self.module_path = hydra.utils.to_absolute_path(module_path_str)
        
        self.encode_fn_name = config.get("encode_function_name")
        if not self.encode_fn_name:
            raise ValueError("For 'function' encoder, 'encode_function_name' must be specified.")
            
        self.decode_fn_name = config.get("decode_function_name")
        if not self.decode_fn_name:
            raise ValueError("For 'function' encoder, 'decode_function_name' must be specified.")

        self._load_functional_encoder()

    def _load_functional_encoder(self):
        """
        Loads the custom Python module and extracts the encode and decode functions.

        Raises:
            FileNotFoundError: If the module file does not exist.
            ImportError: If the module cannot be loaded or executed.
            AttributeError: If the specified functions are not found within the module.
        """
        if not os.path.isfile(self.module_path):
            raise FileNotFoundError(f"Encoder module file not found: {self.module_path}")

        try:
            # Create a module spec from the file path
            spec = importlib.util.spec_from_file_location("custom_encoder_module", self.module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec from {self.module_path}")
            # Create and execute the module
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
        except Exception as e:
            raise ImportError(f"Failed to load module from {self.module_path}: {e}")

        if not hasattr(custom_module, self.encode_fn_name):
            raise AttributeError(f"Encode function '{self.encode_fn_name}' not found in module {self.module_path}.")
        self._encode_callable = getattr(custom_module, self.encode_fn_name)

        if not hasattr(custom_module, self.decode_fn_name):
            raise AttributeError(f"Decode function '{self.decode_fn_name}' not found in module {self.module_path}.")
        self._decode_callable = getattr(custom_module, self.decode_fn_name)

    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encodes an amino acid sequence string using the user-provided encode function.

        Args:
            sequence (str): The amino acid sequence to encode.

        Returns:
            torch.Tensor: A tensor of shape (len(sequence), self.input_size)
                          with dtype torch.float32.

        Raises:
            TypeError: If the encode function returns an unsupported type.
            ValueError: If the shape of the encoded tensor does not match expectations.
        """
        validated_sequence = self._validate_sequence_chars(sequence, self.alphabet)
        encoded_output = self._encode_callable(validated_sequence)
        
        # Ensure the output is a PyTorch tensor
        if not isinstance(encoded_output, torch.Tensor):
            try:
                encoded_output = torch.from_numpy(np.array(encoded_output, dtype=np.float32))
            except Exception as e:
                raise TypeError(f"Encoder function must return a torch.Tensor or a type convertible to it. Got {type(encoded_output)}. Error: {e}")
        
        # Validate the shape of the output tensor
        if encoded_output.shape != (len(validated_sequence), self.input_size):
            raise ValueError(f"Encoded tensor shape mismatch. Expected {(len(validated_sequence), self.input_size)}, got {encoded_output.shape}.")
        
        return encoded_output.float()

    def decode(self, seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        Decodes one or more sequence vectors back into amino acid sequence strings
        using the user-provided decode function.

        Args:
            seq_vectors (Union[torch.Tensor, np.ndarray, List[np.ndarray]]):
                Encoded sequence vectors.

        Returns:
            List[str]: A list of decoded sequence strings.
        """
        # This method reuses the same logic as the Table encoder for processing input vectors
        processed_vectors = BaseParrotEncoder._process_input_vectors(seq_vectors, self.input_size)
        decoded_sequences: List[str] = [self._decode_callable(vec_np) for vec_np in processed_vectors]
        return decoded_sequences

    def __len__(self) -> int:
        """Returns the input_size (dimension of the encoded vector for a single character)."""
        return self.input_size

    def get_alphabet(self) -> set:
        """Returns the set of allowed characters for this encoder."""
        return self.alphabet


class ParrotLightningEncoder:
    """
    A factory/dispatcher class for PARROT sequence encoders.
    It instantiates the appropriate concrete encoder (Table-based or Functional)
    based on the Hydra configuration and delegates all encoding/decoding operations to it.

    This class acts as a unified interface, abstracting away the specifics of
    different encoder implementations.

    Attributes:
        _actual_encoder (BaseParrotEncoder): The concrete encoder instance (TableParrotEncoder or FunctionalParrotEncoder).
    """
    def __init__(self, encoder_cfg: DictConfig):
        """
        Initializes the ParrotLightningEncoder based on the provided Hydra configuration.

        Parameters:
            encoder_cfg (DictConfig): Configuration object specifying the encoder type.
        """
        encoder_type = encoder_cfg.get("type")
        if not encoder_type:
            raise ValueError("Encoder 'type' must be specified in the configuration (e.g., 'table' or 'function').")

        if encoder_type == "table":
            # Instantiate TableParrotEncoder if type is 'table'
            self._actual_encoder: BaseParrotEncoder = TableParrotEncoder(encoder_cfg)
        elif encoder_type == "function":
            # Instantiate FunctionalParrotEncoder if type is 'function'
            self._actual_encoder: BaseParrotEncoder = FunctionalParrotEncoder(encoder_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: '{encoder_type}'. Must be 'table' or 'function'.")

    def encode(self, sequence: str) -> torch.Tensor:
        """Delegates the encode operation to the actual encoder."""
        return self._actual_encoder.encode(sequence)

    def decode(self, seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> List[str]:
        """Delegates the decode operation to the actual encoder."""
        return self._actual_encoder.decode(seq_vectors)

    def __len__(self) -> int:
        """Delegates the length query to the actual encoder."""
        return len(self._actual_encoder)

    def get_alphabet(self) -> set:
        """Delegates the alphabet query to the actual encoder."""
        return self._actual_encoder.get_alphabet()

    @property
    def encoder_type(self) -> str:
        """
        Returns the type of the underlying concrete encoder.

        Returns:
            str: "table", "function", or "unknown".
        """
        if isinstance(self._actual_encoder, TableParrotEncoder):
            return "table"
        elif isinstance(self._actual_encoder, FunctionalParrotEncoder):
            return "function"
        return "unknown"
