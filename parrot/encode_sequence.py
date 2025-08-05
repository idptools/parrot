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
import pickle
import json
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

    TODO: Implement matrix format support
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

    def encode_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Encodes multiple amino acid sequence strings into PyTorch tensors.

        Parameters:
            sequences (List[str]): A list of amino acid sequences to encode.

        Returns:
            List[torch.Tensor]: A list of tensors, each representing an encoded sequence.
        """
        return [self.encode(sequence) for sequence in sequences]

    def encode_sequences_padded(self, sequences: List[str], pad_value: float = 0.0) -> torch.Tensor:
        """
        Encodes multiple amino acid sequences and pads them to the same length.

        Parameters:
            sequences (List[str]): A list of amino acid sequences to encode.
            pad_value (float): The value to use for padding. Default is 0.0.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_seq_len, input_size) for regular encoders
                         or (batch_size, max_seq_len, max_seq_len, input_size) for matrix encoders.
        """
        if not sequences:
            raise ValueError("Cannot encode empty sequence list")
        
        # Encode all sequences
        encoded_sequences = self.encode_sequences(sequences)
        
        # Check if this is a matrix encoder (3D tensors)
        is_matrix_encoder = len(encoded_sequences[0].shape) == 3
        
        if is_matrix_encoder:
            # For matrix encoders: (seq_len, seq_len, input_size)
            max_seq_len = max(tensor.shape[0] for tensor in encoded_sequences)
            input_size = encoded_sequences[0].shape[2]
            batch_size = len(encoded_sequences)
            
            # Create padded tensor: (batch_size, max_seq_len, max_seq_len, input_size)
            padded_tensor = torch.full((batch_size, max_seq_len, max_seq_len, input_size), 
                                     pad_value, dtype=torch.float32)
            
            for i, tensor in enumerate(encoded_sequences):
                seq_len = tensor.shape[0]
                padded_tensor[i, :seq_len, :seq_len, :] = tensor
                
        else:
            # For regular encoders: (seq_len, input_size)
            max_seq_len = max(tensor.shape[0] for tensor in encoded_sequences)
            input_size = encoded_sequences[0].shape[1]
            batch_size = len(encoded_sequences)
            
            # Create padded tensor: (batch_size, max_seq_len, input_size)
            padded_tensor = torch.full((batch_size, max_seq_len, input_size), 
                                     pad_value, dtype=torch.float32)
            
            for i, tensor in enumerate(encoded_sequences):
                seq_len = tensor.shape[0]
                padded_tensor[i, :seq_len, :] = tensor
        
        return padded_tensor

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
    def _process_input_vectors(seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]], input_size: int, is_matrix_format: bool = False) -> List[np.ndarray]:
        """
        Helper method to normalize various input vector formats into a list of NumPy arrays.

        Args:
            seq_vectors (Union[torch.Tensor, np.ndarray, List[np.ndarray]]):
                Input vectors which can be a single tensor/array or a list of arrays.
            input_size (int): The expected dimension of the individual character vectors.
                For matrix encoders, this should be the first dimension of the tensor.
            is_matrix_format (bool): Whether the input is in matrix format (3D tensors).

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing an encoded sequence.
                For regular encoders: each array is 2D (seq_len, input_size)
                For matrix encoders: each array is 3D (seq_len, seq_len, input_size)

        Raises:
            TypeError: If the input type for `seq_vectors` is unsupported.
            ValueError: If the dimensions of the input vectors are incorrect.
        """
        processed_vectors: List[np.ndarray] = []
        seq_vectors_np = None
        
        if isinstance(seq_vectors, torch.Tensor):
            # If input is a PyTorch tensor, convert it to a NumPy array.
            seq_vectors_np = seq_vectors.cpu().numpy()
        elif isinstance(seq_vectors, np.ndarray):
            # If input is already a NumPy array, use it directly.
            seq_vectors_np = seq_vectors
        elif isinstance(seq_vectors, list):
            # If input is a list, assume it's already a list of NumPy arrays
            processed_vectors = seq_vectors
            seq_vectors_np = None # Indicate that no further NumPy conversion is needed for this path
        else:
            # Raise an error for unsupported input types.
            raise TypeError(f"Unsupported type for seq_vectors: {type(seq_vectors)}. Expected torch.Tensor, np.ndarray, or List[np.ndarray].")

        if seq_vectors_np is not None:
            if is_matrix_format:
                # Matrix format handling
                if seq_vectors_np.ndim == 3:
                    # Single matrix: (seq_len, seq_len, input_size)
                    processed_vectors.append(seq_vectors_np)
                elif seq_vectors_np.ndim == 4:
                    # Batch of matrices: (batch_size, seq_len, seq_len, input_size)
                    for i in range(seq_vectors_np.shape[0]):
                        processed_vectors.append(seq_vectors_np[i, :, :, :])
                else:
                    raise ValueError(f"Unsupported ndim for matrix format: {seq_vectors_np.ndim}. Expected 3 or 4.")
            else:
                # Regular format handling
                if seq_vectors_np.ndim == 2:
                    # If 2D (e.g., a single sequence: seq_len x input_size),
                    # add it as a single item to the processed list.
                    processed_vectors.append(seq_vectors_np)
                elif seq_vectors_np.ndim == 3:
                    # If 3D (e.g., a batch of sequences: batch_size x seq_len x input_size),
                    # iterate through the batch dimension and add each 2D sequence.
                    for i in range(seq_vectors_np.shape[0]):
                        processed_vectors.append(seq_vectors_np[i, :, :])
                else:
                    # Raise an error for unsupported NumPy array dimensions.
                    raise ValueError(f"Unsupported ndim for regular format: {seq_vectors_np.ndim}. Expected 2 or 3.")
        
        # Validate processed vectors
        if is_matrix_format:
            for vec in processed_vectors:
                if not isinstance(vec, np.ndarray) or vec.ndim != 3 or vec.shape[2] != input_size:
                    raise ValueError(f"Invalid matrix vector format. Each must be a 3D NumPy array of shape (seq_len, seq_len, {input_size}). Got shape {vec.shape if hasattr(vec, 'shape') else 'N/A'}")
        else:
            for vec in processed_vectors:
                # Validate each processed vector: ensure it's a NumPy array, 2D,
                # and its second dimension matches the expected input_size.
                if not isinstance(vec, np.ndarray) or vec.ndim != 2 or vec.shape[1] != input_size:
                    raise ValueError(f"Invalid vector format. Each must be a 2D NumPy array of shape (seq_len, {input_size}). Got shape {vec.shape if hasattr(vec, 'shape') else 'N/A'}")
        
        return processed_vectors

    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing all the state needed to recreate this encoder.
        Must be implemented by each concrete encoder class.

        Returns:
            Dict[str, Any]: State dictionary containing encoder configuration and data.
        """
        pass

    @classmethod
    @abstractmethod
    def _from_state_dict(cls, state_dict: Dict[str, Any]) -> 'BaseParrotEncoder':
        """
        Creates an encoder instance from a state dictionary.
        Must be implemented by each concrete encoder class.

        Args:
            state_dict (Dict[str, Any]): State dictionary containing encoder configuration and data.

        Returns:
            BaseParrotEncoder: The recreated encoder instance.
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Saves the encoder to a file using pickle serialization.

        Args:
            filepath (str): Path where the encoder should be saved.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            state_dict = self._get_state_dict()
            state_dict['encoder_class'] = self.__class__.__name__
            
            with open(filepath, 'wb') as f:
                pickle.dump(state_dict, f)
        except Exception as e:
            raise IOError(f"Failed to save encoder to {filepath}: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'BaseParrotEncoder':
        """
        Loads an encoder from a file.

        Args:
            filepath (str): Path to the saved encoder file.

        Returns:
            BaseParrotEncoder: The loaded encoder instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            IOError: If the file cannot be read or is corrupted.
            ValueError: If the encoder type is not recognized.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Encoder file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                state_dict = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load encoder from {filepath}: {e}")
        
        encoder_class_name = state_dict.get('encoder_class')
        if not encoder_class_name:
            raise ValueError("Invalid encoder file: missing encoder class information")
        
        # Import the classes here to avoid forward reference issues
        # This assumes all classes are defined in the same module
        current_module = sys.modules[__name__]
        
        try:
            encoder_class = getattr(current_module, encoder_class_name)
        except AttributeError:
            raise ValueError(f"Unknown encoder class: {encoder_class_name}")
        
        return encoder_class._from_state_dict(state_dict)

    def save_config(self, filepath: str) -> None:
        """
        Saves only the configuration of the encoder to a JSON file.
        This can be useful for recreating encoders with the same settings.

        Args:
            filepath (str): Path where the configuration should be saved.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            state_dict = self._get_state_dict()
            # Remove non-serializable items for JSON
            config_dict = {k: v for k, v in state_dict.items() 
                          if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))}
            config_dict['encoder_class'] = self.__class__.__name__
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save encoder config to {filepath}: {e}")

    

class MatrixParrotEncoder(BaseParrotEncoder):
    """
    A concrete encoder implementation that uses a matrix format for encoding and decoding amino acid sequences.

    Matrix encoding transforms a sequence of length N into a matrix of shape NxN, allowing for complex 
    relationships between residues. The encoding scheme is defined by the alphabet and optional gap character.

    On the decoding side, the diagonal of the matrix is used to decode the sequence. The encoding is always 
    symmetric and the diagonal represents the sequence itself. The off-diagonal elements encode relationships
    between residues.

    The gap character (default '*') is useful for multidomain sequences or multiple sequences. It modifies 
    the total number of dimensions in the embedding.

    For vectorial representations:
    - Embedding dimension = alphabet_size^2 (without gap) or alphabet_size^2 + 1 (with gap)
    - Example: 'ACDEFGHIKLMNPQRSTVWY' (20 chars) → 400 dimensions, or 401 with gap

    For numerical representations:
    - Embedding dimension = 1
    - Values range from 1 to alphabet_size*alphabet_size (+ 1 for gap if present)
    - 0's are reserved for padding

    Output tensor shape: (sequence_length, sequence_length, embedding_dimension)

    Attributes:
        alphabet (set): The set of allowed characters for this encoder.
        gap_char (str): The gap character (default '*').
        use_gap (bool): Whether gap character is included.
        encoding_type (str): Either 'vectorial' or 'numerical'.
        input_size (int): The embedding dimension.
        _char_to_idx (Dict[str, int]): Mapping from characters to indices.
        _idx_to_char (Dict[int, str]): Mapping from indices to characters.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the MatrixParrotEncoder.

        Args:
            config (DictConfig): Configuration containing:
                - alphabet (str): Required. String of allowed characters.
                - gap_char (str, optional): Gap character. Default '*'.
                - use_gap (bool, optional): Whether to use gap character. Default True.
                - encoding_type (str, optional): 'vectorial' or 'numerical'. Default 'vectorial'.
        """
        # Extract alphabet from config
        alphabet_str = config.get("alphabet")
        if not alphabet_str:
            raise ValueError("For 'matrix' encoder, 'alphabet' must be specified.")
        # Keep the base alphabet as a list to preserve user-specified order
        # Remove gap character if present to avoid duplication
        alphabet_list = list(alphabet_str)
        self.alphabet = [char for char in alphabet_list if char != config.get("gap_char", "*")]
        
        # Extract gap character settings
        # Future note: will need to ensure that it integrates for this with the Sequence Dataset Object
        # Default is to use a gap character if not specified
        self.gap_char = config.get("gap_char", "*")
        self.use_gap = config.get("use_gap", True)
        
        # Extract encoding type
        # Default is 'vectorial' encoding
        self.encoding_type = config.get("encoding_type", "vectorial")
        if self.encoding_type not in ["vectorial", "numerical"]:
            raise ValueError("encoding_type must be either 'vectorial' or 'numerical'")
        
        # Create character mappings - preserve user-specified order, gap character always goes last
        self._char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self._idx_to_char = {idx: char for char, idx in self._char_to_idx.items()}
        
        # Add gap character as the last index if using gaps
        if self.use_gap:
            gap_idx = len(self.alphabet)
            self._char_to_idx[self.gap_char] = gap_idx
            self._idx_to_char[gap_idx] = self.gap_char
        
        # Calculate input size based on encoding type
        base_alphabet_size = len(self.alphabet)
        total_alphabet_size = base_alphabet_size + (1 if self.use_gap else 0)
        
        if self.encoding_type == "vectorial":
            # For vectorial encoding, we need dimensions for all possible character pairs
            # If using gaps, we add one extra dimension for any pair involving gap character
            if self.use_gap:
                self.input_size = base_alphabet_size * base_alphabet_size + 1
            else:
                self.input_size = base_alphabet_size * base_alphabet_size
        else:  # numerical
            self.input_size = 1

    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encodes an amino acid sequence string into a matrix tensor.

        Args:
            sequence (str): The amino acid sequence to encode.

        Returns:
            torch.Tensor: A tensor of shape (len(sequence), len(sequence), self.input_size)
                          with dtype torch.float32.
        """
        # Validate and preprocess the input sequence
        validated_sequence = self._validate_sequence_chars(sequence, self.get_alphabet())
        seq_len = len(validated_sequence)
        
        if self.encoding_type == "vectorial":
            # Create vectorial encoding matrix
            matrix = torch.zeros((seq_len, seq_len, self.input_size), dtype=torch.float32)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    char_i = validated_sequence[i]
                    char_j = validated_sequence[j]
                    
                    if self.use_gap and (char_i == self.gap_char or char_j == self.gap_char):
                        # Gap character gets special encoding in last dimension
                        matrix[i, j, self.input_size - 1] = 1.0
                    else:
                        # Regular character pair encoding
                        idx_i = self._char_to_idx[char_i]
                        idx_j = self._char_to_idx[char_j]
                        # Calculate position in flattened base_alphabet^2 space
                        base_alphabet_size = len(self.alphabet)
                        encoding_idx = idx_i * base_alphabet_size + idx_j
                        matrix[i, j, encoding_idx] = 1.0
        
        else:  # numerical encoding
            matrix = torch.zeros((seq_len, seq_len, 1), dtype=torch.float32)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    char_i = validated_sequence[i]
                    char_j = validated_sequence[j]
                    
                    if self.use_gap and (char_i == self.gap_char or char_j == self.gap_char):
                        # Gap character gets highest numerical value (alphabet_size² + 1)
                        matrix[i, j, 0] = len(self.alphabet) * len(self.alphabet) + 1
                    else:
                        # Calculate numerical encoding based on character pair
                        idx_i = self._char_to_idx[char_i]
                        idx_j = self._char_to_idx[char_j]
                        # Use the same encoding index as vectorial but as numerical value
                        base_alphabet_size = len(self.alphabet)
                        encoding_value = idx_i * base_alphabet_size + idx_j + 1  # +1 to avoid 0
                        matrix[i, j, 0] = encoding_value
        
        return matrix

    def decode(self, seq_vectors: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> List[str]:
        """
        Decodes one or more matrix tensors back into amino acid sequence strings.
        Uses the diagonal of the matrix to reconstruct the sequence.

        Args:
            seq_vectors: Matrix tensors to decode.

        Returns:
            List[str]: A list of decoded sequence strings.
        """
        # Convert input to consistent format using the updated _process_input_vectors method
        processed_vectors = BaseParrotEncoder._process_input_vectors(seq_vectors, self.input_size, is_matrix_format=True)
        
        # Decode each matrix
        decoded_sequences = []
        for matrix in processed_vectors:
            decoded_sequences.append(self._decode_single_matrix(matrix))
        
        return decoded_sequences

    def _decode_single_matrix(self, matrix: np.ndarray) -> str:
        """
        Decode a single matrix into a sequence string using the diagonal.

        Args:
            matrix (np.ndarray): Matrix of shape (seq_len, seq_len, input_size).

        Returns:
            str: The decoded sequence string.
        """
        if matrix.ndim != 3:
            raise ValueError(f"Matrix must be 3D (seq_len, seq_len, input_size), got {matrix.ndim}D")
        
        seq_len, _, input_size = matrix.shape
        if input_size != self.input_size:
            raise ValueError(f"Matrix input_size {input_size} doesn't match encoder input_size {self.input_size}")
        
        sequence_chars = []
        
        if self.encoding_type == "vectorial":
            base_alphabet_size = len(self.alphabet)
            
            for i in range(seq_len):
                # Extract diagonal element
                diag_vector = matrix[i, i, :]
                
                # Check if it's a gap character (last dimension for gaps)
                if self.use_gap and diag_vector[-1] > 0.5:
                    sequence_chars.append(self.gap_char)
                else:
                    # Find the active dimension
                    active_idx = np.argmax(diag_vector[:-1] if self.use_gap else diag_vector)
                    
                    # Convert back to character indices
                    char_i_idx = active_idx // base_alphabet_size
                    char_j_idx = active_idx % base_alphabet_size
                    
                    # For diagonal elements, char_i should equal char_j
                    # Use char_i_idx as the character index
                    if char_i_idx < len(self._idx_to_char) and char_i_idx == char_j_idx:
                        sequence_chars.append(self._idx_to_char[char_i_idx])
                    else:
                        # Fallback to gap character if index is out of range
                        sequence_chars.append(self.gap_char if self.use_gap else 'X')
        
        else:  # numerical encoding
            for i in range(seq_len):
                # Extract diagonal element
                diag_value = matrix[i, i, 0]
                
                base_alphabet_size = len(self.alphabet)
                gap_threshold = base_alphabet_size * base_alphabet_size + 1
                
                if self.use_gap and diag_value >= gap_threshold:
                    sequence_chars.append(self.gap_char)
                else:
                    # Convert numerical value back to character indices
                    # Since we used idx_i * base_alphabet_size + idx_j + 1
                    # So diag_value = idx_i * base_alphabet_size + idx_j + 1
                    # For diagonal elements, idx_i == idx_j, so diag_value = idx_i * (base_alphabet_size + 1) + 1
                    # Actually, for diagonal: diag_value = idx_i * base_alphabet_size + idx_i + 1 = idx_i * (base_alphabet_size + 1) + 1
                    # So idx_i = (diag_value - 1) / (base_alphabet_size + 1)
                    encoding_idx = int(round(diag_value - 1))
                    char_i_idx = encoding_idx // base_alphabet_size
                    char_j_idx = encoding_idx % base_alphabet_size
                    
                    # For diagonal elements, char_i should equal char_j, so use char_i_idx
                    if char_i_idx == char_j_idx and char_i_idx in self._idx_to_char:
                        sequence_chars.append(self._idx_to_char[char_i_idx])
                    else:
                        sequence_chars.append(self.gap_char if self.use_gap else 'X')
        
        return ''.join(sequence_chars)

    def __len__(self) -> int:
        """Returns the input_size (embedding dimension)."""
        return self.input_size

    def get_alphabet(self) -> set:
        """Returns the set of allowed characters for this encoder."""
        if self.use_gap:
            return set(self.alphabet) | {self.gap_char}
        return set(self.alphabet)

    def _get_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing all the state needed to recreate this encoder."""
        return {
            'alphabet': self.alphabet,
            'gap_char': self.gap_char,
            'use_gap': self.use_gap,
            'encoding_type': self.encoding_type,
            'input_size': self.input_size,
            '_char_to_idx': self._char_to_idx,
            '_idx_to_char': self._idx_to_char
        }

    @classmethod
    def _from_state_dict(cls, state_dict: Dict[str, Any]) -> 'MatrixParrotEncoder':
        """Creates a MatrixParrotEncoder instance from a state dictionary."""
        # Create a DictConfig-like object from the state dict
        from omegaconf import DictConfig
        config = DictConfig({
            'alphabet': ''.join(state_dict['alphabet']),
            'gap_char': state_dict['gap_char'],
            'use_gap': state_dict['use_gap'],
            'encoding_type': state_dict['encoding_type']
        })
        
        # Create new instance
        encoder = cls(config)
        
        # Restore saved state
        encoder.input_size = state_dict['input_size']
        encoder._char_to_idx = state_dict['_char_to_idx']
        encoder._idx_to_char = state_dict['_idx_to_char']
        
        return encoder

    



class TableParrotEncoder(BaseParrotEncoder):
    """
    A concrete encoder implementation that uses a lookup table (TSV file)
    to encode and decode amino acid sequences.

    Individual sequences are encoded as pyTorch tensors using a predefined mapping.
    The dimensions of the encoding always follow: (num_sequences, sequence_length, input_size).
    Where `input_size` is the dimension of the encoded vector for each character.

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
        processed_vectors: List[np.ndarray] = BaseParrotEncoder._process_input_vectors(seq_vectors, self.input_size, is_matrix_format=False)
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

    def _get_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing all the state needed to recreate this encoder."""
        return {
            'is_uniquely_decodable': self.is_uniquely_decodable,
            'table_file_path': self.table_file_path,
            'input_size': self.input_size,
            'alphabet': list(self.alphabet),
            '_table_encode_dict': self._table_encode_dict,
            '_table_decode_dict': self._table_decode_dict
        }

    @classmethod
    def _from_state_dict(cls, state_dict: Dict[str, Any]) -> 'TableParrotEncoder':
        """Creates a TableParrotEncoder instance from a state dictionary."""
        from omegaconf import DictConfig
        
        # Create a minimal config to initialize the encoder
        if state_dict['table_file_path']:
            config = DictConfig({
                'table_file_path': state_dict['table_file_path'],
                'alphabet': ''.join(state_dict['alphabet']) if state_dict['alphabet'] else None
            })
        else:
            config = DictConfig({
                'alphabet': ''.join(state_dict['alphabet'])
            })
        
        # Create new instance
        encoder = cls(config)
        
        # Restore saved state
        encoder.is_uniquely_decodable = state_dict['is_uniquely_decodable']
        encoder.input_size = state_dict['input_size']
        encoder._table_encode_dict = state_dict['_table_encode_dict']
        encoder._table_decode_dict = state_dict['_table_decode_dict']
        
        return encoder


class FunctionalParrotEncoder(BaseParrotEncoder):
    """
    A concrete encoder implementation that uses user-provided Python functions
    for encoding and decoding sequences.

    The dimensions of the encoding always follow: (num_sequences, ..., input_size).
    Where `input_size` is the dimension of the encoded vector for each 'character'.
    This could (num_sequences, sequence_length, input_size) for much like table encoding,
    or (num_sequences, sequence_length, sequence_length, input_size) for matrix style encodings.

    TODO: Implement matrix format support.

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
        processed_vectors = BaseParrotEncoder._process_input_vectors(seq_vectors, self.input_size, is_matrix_format=False)
        decoded_sequences: List[str] = [self._decode_callable(vec_np) for vec_np in processed_vectors]
        return decoded_sequences

    def __len__(self) -> int:
        """Returns the input_size (dimension of the encoded vector for a single character)."""
        return self.input_size

    def get_alphabet(self) -> set:
        """Returns the set of allowed characters for this encoder."""
        return self.alphabet

    def _get_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing all the state needed to recreate this encoder."""
        return {
            'alphabet': list(self.alphabet),
            'input_size': self.input_size,
            'module_path': self.module_path,
            'encode_fn_name': self.encode_fn_name,
            'decode_fn_name': self.decode_fn_name
        }

    @classmethod
    def _from_state_dict(cls, state_dict: Dict[str, Any]) -> 'FunctionalParrotEncoder':
        """Creates a FunctionalParrotEncoder instance from a state dictionary."""
        from omegaconf import DictConfig
        
        config = DictConfig({
            'alphabet': ''.join(state_dict['alphabet']),
            'input_size': state_dict['input_size'],
            'module_path': state_dict['module_path'],
            'encode_function_name': state_dict['encode_fn_name'],
            'decode_function_name': state_dict['decode_fn_name']
        })
        
        return cls(config)


class ParrotLightningEncoder:
    """
    A factory/dispatcher class for PARROT sequence encoders.
    It instantiates the appropriate concrete encoder (Table-based, Functional, or Matrix)
    based on the Hydra configuration and delegates all encoding/decoding operations to it.

    This class acts as a unified interface, abstracting away the specifics of
    different encoder implementations.

    Attributes:
        _actual_encoder (BaseParrotEncoder): The concrete encoder instance (TableParrotEncoder, FunctionalParrotEncoder, or MatrixParrotEncoder).
    """
    def __init__(self, encoder_cfg: DictConfig):
        """
        Initializes the ParrotLightningEncoder based on the provided Hydra configuration.

        Parameters:
            encoder_cfg (DictConfig): Configuration object specifying the encoder type.
        """
        encoder_type = encoder_cfg.get("type")
        if not encoder_type:
            raise ValueError("Encoder 'type' must be specified in the configuration (e.g., 'table', 'function', or 'matrix').")

        if encoder_type == "table":
            # Instantiate TableParrotEncoder if type is 'table'
            self._actual_encoder: BaseParrotEncoder = TableParrotEncoder(encoder_cfg)
        elif encoder_type == "function":
            # Instantiate FunctionalParrotEncoder if type is 'function'
            self._actual_encoder: BaseParrotEncoder = FunctionalParrotEncoder(encoder_cfg)
        elif encoder_type == "matrix":
            # Instantiate MatrixParrotEncoder if type is 'matrix'
            self._actual_encoder: BaseParrotEncoder = MatrixParrotEncoder(encoder_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: '{encoder_type}'. Must be 'table', 'function', or 'matrix'.")

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

    def encode_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Delegates the multi-sequence encode operation to the actual encoder."""
        return self._actual_encoder.encode_sequences(sequences)

    def encode_sequences_padded(self, sequences: List[str], pad_value: float = 0.0) -> torch.Tensor:
        """Delegates the padded multi-sequence encode operation to the actual encoder."""
        return self._actual_encoder.encode_sequences_padded(sequences, pad_value)

    @property
    def encoder_type(self) -> str:
        """
        Returns the type of the underlying concrete encoder.

        Returns:
            str: "table", "function", "matrix", or "unknown".
        """
        if isinstance(self._actual_encoder, TableParrotEncoder):
            return "table"
        elif isinstance(self._actual_encoder, FunctionalParrotEncoder):
            return "function"
        elif isinstance(self._actual_encoder, MatrixParrotEncoder):
            return "matrix"
        return "unknown"

    def save(self, filepath: str) -> None:
        """
        Saves the encoder to a file.

        Args:
            filepath (str): Path where the encoder should be saved.
        """
        self._actual_encoder.save(filepath)

    @staticmethod
    def load(filepath: str) -> 'ParrotLightningEncoder':
        """
        Loads an encoder from a file and wraps it in a ParrotLightningEncoder.

        Args:
            filepath (str): Path to the saved encoder file.

        Returns:
            ParrotLightningEncoder: The loaded encoder wrapped in a ParrotLightningEncoder.
        """
        # Load the underlying encoder
        actual_encoder = BaseParrotEncoder.load(filepath)
        
        # Create a new ParrotLightningEncoder instance
        lightning_encoder = object.__new__(ParrotLightningEncoder)
        lightning_encoder._actual_encoder = actual_encoder
        
        return lightning_encoder

    def save_config(self, filepath: str) -> None:
        """
        Saves only the configuration of the encoder to a JSON file.

        Args:
            filepath (str): Path where the configuration should be saved.
        """
        self._actual_encoder.save_config(filepath)
