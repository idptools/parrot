"""
File containing functions and encoder objects for converting a string of amino
acids (or any other alphabet, via a user encoding file) into numeric vectors.

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import os
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

class LookupTable:
    """Encodes a sequence by looking up every character's vector at once

    Much faster than encoding character by character in Python (about 3x for
    one-hot encoding), which matters because sequences are encoded every time
    they are used: every epoch in training, and for every prediction.

    Attributes
    ----------
    vectors : numpy array
        [number of characters X vector length] float64 table, one row per character
    """

    def __init__(self, table: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        table : dict
            Maps each character to its vector
        """
        characters = list(table)
        self.vectors = np.array([table[character] for character in characters], dtype=np.float64)
        if self.vectors.ndim != 2:
            self.vectors = self.vectors.reshape(len(characters), -1)
        self._row = {character: row for row, character in enumerate(characters)}

        # Characters that are single bytes in Latin-1 (every amino acid letter) are looked up by byte value
        self._byte_row = np.full(256, -1, dtype=np.int64)
        for character, row in self._row.items():
            if len(character) == 1 and ord(character) < 256:
                self._byte_row[ord(character)] = row

    def encode(self, seq: str) -> torch.Tensor:
        """Encode a sequence as a float64 [len(seq) X vector length] tensor

        Raises
        ------
        ValueError
            If the sequence contains a character that is not in the table
        """
        try:
            rows = self._byte_row[np.frombuffer(seq.encode('latin-1'), dtype=np.uint8)]
        except UnicodeEncodeError:
            rows = np.array([self._row.get(character, -1) for character in seq], dtype=np.int64)

        if rows.size and rows.min() < 0:
            raise ValueError('Invalid amino acid detected: ' + seq[int(np.argmax(rows < 0))])
        return torch.from_numpy(self.vectors[rows])


ONE_HOT = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 
            'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 
            'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19 }

_ONE_HOT_TABLE = LookupTable({amino_acid: np.eye(20)[index] for amino_acid, index in ONE_HOT.items()})

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

    return _ONE_HOT_TABLE.encode(seq)


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
    return _BIOPHYSICS_TABLE.encode(seq)


_BIOPHYSICS_TABLE = LookupTable(BIOPHYSICS)


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
    """Convert an encoding file into a character-to-vector dictionary

    Each non-empty line of the file should contain a single character followed
    by the whitespace-separated numbers that make up its encoding vector. Blank
    lines are ignored.

    Parameters
    ----------
    file : str
            Path to the encoding file

    Returns
    -------
    dict
            Maps each character to its encoding vector (a list of floats)
    int
            The length of the encoding vectors

    Raises
    ------
    ValueError
            If the file is empty, if the vectors are not all the same length, or
            if any vector entry is not a number
    """

    with open(file) as f:
        lines = [x.strip().split() for x in f if len(x.strip()) > 0]

    if len(lines) == 0:
        raise ValueError(f'Encoding file {file} is empty.')

    l = len(lines[0]) - 1
    d = {}
    for line in lines:
        if len(line) - 1 != l:
            raise ValueError('Vectors in encoding file do not have same length.')

        try:
            d[line[0]] = [float(value) for value in line[1:]]
        except ValueError:
            raise ValueError(f'Non-numeric value in encoding file for character {line[0]}: {line[1:]}') from None

    return d, l


def check_characters(sequences: Iterable[str], valid: set[str]) -> None:
    """Raise if any sequence contains a character outside `valid`

    Parameters
    ----------
    sequences : iterable of str
        Sequences to check
    valid : set of str
        The characters allowed

    Raises
    ------
    ValueError
        For the first sequence with a character outside `valid`
    """
    for sequence in dict.fromkeys(sequences):
        invalid = set(sequence) - valid
        if invalid:
            shown = sequence if len(sequence) <= 60 else sequence[:57] + '...'
            raise ValueError(f'Invalid amino acid detected: {sorted(invalid)[0]} (in sequence {shown})')


################## Encoder objects ####################
#
# Everything outside this module works with a SequenceEncoder object rather than
# with the functions above, so a new encoding (e.g. a learned embedding) only
# needs a new SequenceEncoder subclass and an entry in get_encoder().

class SequenceEncoder:
    """Base class for objects that convert sequences to numeric vectors

    Attributes
    ----------
    name : str
            Short name of the encoding, stored when a network is saved
    input_size : int
            Length of the vector each residue is encoded as
    """

    name: str = ''
    input_size: int = 0

    def __len__(self) -> int:
        """Get length of encoding scheme"""

        return self.input_size

    @property
    def alphabet(self) -> str:
        """Every character this encoding defines a vector for"""

        raise NotImplementedError

    def encode(self, seq: str) -> torch.Tensor:
        """Convert a sequence into a [len(seq) X input_size] tensor"""

        raise NotImplementedError

    def check_sequences(self, sequences: Iterable[str]) -> None:
        """Raise if any sequence contains a character this encoding cannot encode

        Checking every sequence before training or predicting reports a bad
        residue straight away, rather than when its sequence is first encoded
        (for a test-set sequence, only after training has finished).

        Raises
        ------
        ValueError
            For the first sequence with a character outside the alphabet
        """

        check_characters(sequences, set(self.alphabet))

    def decode(self, seq_vectors: list[np.ndarray]) -> list[str]:
        """Convert a list of encoded sequences back to strings"""

        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        """Return a plain dictionary from which encoder_from_config() can rebuild this encoder"""

        return {'name': self.name}


class OneHotEncoder(SequenceEncoder):
    """One-hot encoding of the 20 canonical amino acids (see one_hot())"""

    name = 'onehot'
    input_size = len(ONE_HOT)

    @property
    def alphabet(self) -> str:
        return ''.join(ONE_HOT)

    def encode(self, seq: str) -> torch.Tensor:
        return one_hot(seq)

    def decode(self, seq_vectors: list[np.ndarray]) -> list[str]:
        return rev_one_hot(seq_vectors)


class BiophysicsEncoder(SequenceEncoder):
    """Encoding of amino acids by nine biophysical scales (see biophysics())"""

    name = 'biophysics'
    input_size = len(BIOPHYSICS['A'])

    @property
    def alphabet(self) -> str:
        return ''.join(BIOPHYSICS)

    def encode(self, seq: str) -> torch.Tensor:
        return biophysics(seq)

    def decode(self, seq_vectors: list[np.ndarray]) -> list[str]:
        return rev_biophysics(seq_vectors)


class UserEncoder(SequenceEncoder):
    """User-specified character-to-vector encoding scheme

    The scheme is read from an encoding file (see parse_encode_file()) or given
    directly as a dictionary, which is how it is restored from a saved network.

    Attributes
    ----------
    encode_file : str or None
            A path to the file that describes the encoding scheme, or None if
            the scheme was given as a dictionary
    encode_dict : dict
            A dictionary that maps each character to a numeric vector
    input_size : int
            The length of the encoding vector used for each character
    """

    name = 'user'

    def __init__(self, encode_file: str | None = None, table: dict[str, list[float]] | None = None) -> None:
        """
        Parameters
        ----------
        encode_file : str, optional
                A path to a file that describes the encoding scheme
        table : dict, optional
                Maps each character to its encoding vector. Exactly one of
                `encode_file` and `table` must be given.
        """

        if (encode_file is None) == (table is None):
            raise ValueError('Provide exactly one of an encoding file or an encoding table.')

        self.encode_file: str | None
        if encode_file is not None:
            self.encode_file = os.path.abspath(encode_file)
            if not os.path.isfile(self.encode_file):
                raise FileNotFoundError('Encoding file does not exist.')
            self.encode_dict, self.input_size = parse_encode_file(self.encode_file)
        else:
            assert table is not None
            self.encode_file = None
            self.encode_dict = {character: [float(value) for value in vector] for character, vector in table.items()}
            vector_lengths = {len(vector) for vector in self.encode_dict.values()}
            if len(vector_lengths) != 1:
                raise ValueError('Vectors in encoding table do not have same length.')
            self.input_size = vector_lengths.pop()

        self._table = LookupTable(self.encode_dict)

    def encode(self, seq: str) -> torch.Tensor:
        """Convert a sequence into this encoding scheme

        Parameters
        ----------
        seq : str
                A sequence made up of characters defined in the encoding scheme

        Returns
        -------
        torch.DoubleTensor
                a [len(seq) X input_size] tensor representing the encoded sequence
        """

        return self._table.encode(seq)

    def decode(self, seq_vectors: list[np.ndarray]) -> list[str]:
        """Converts a list of sequence vectors back to a list of sequences

        Parameters
        ----------
        seq_vectors : list of numpy arrays
                A list containing sequence vectors

        Returns
        -------
        list
                Strings of sequences
        """

        # Create a reverse of the encode_dict using hashing. Both sides are cast to
        # float32 so that vectors decode correctly whether they come straight from
        # encode() (float64) or from a batched FloatTensor (float32).
        rev_dict = {}
        for key, value in self.encode_dict.items():
            value = np.array(value, dtype=np.float32)
            rev_dict[hash(tuple(value))] = key

        sequences = []
        for seq_vector in seq_vectors:
            seq = []
            for residue in seq_vector:
                residue = np.asarray(residue, dtype=np.float32)
                seq.append(rev_dict[hash(tuple(residue))])
            sequences.append("".join(seq))

        return sequences

    @property
    def alphabet(self) -> str:
        return ''.join(self.encode_dict)

    def to_config(self) -> dict[str, Any]:
        """The encoding table is stored in full so a saved network does not depend on the original file"""

        return {'name': self.name, 'table': self.encode_dict}


# Name stored in a saved network's encoding config for a standardized encoding
STANDARDIZED_NAME = 'standardized'

# A feature whose standard deviation across the alphabet is at most this fraction of
# its typical magnitude is treated as constant (it carries no information)
CONSTANT_FEATURE_TOLERANCE = 1e-12


class StandardizedEncoder(SequenceEncoder):
    """Another encoding with every feature standardized (z-scored) across its alphabet

    Each feature (column) of the base encoding's table is shifted and scaled so
    that, across the characters the encoding defines (the 20 amino acids for the
    built-in encodings), it has mean 0 and standard deviation 1:

        standardized value = (value - mean of the feature) / standard deviation of the feature

    with the population standard deviation (the alphabet is the whole set of
    characters, not a sample of it). Features of very different sizes, such as
    molecular weight (~100-200 g/mol) and an aromatic flag (0 or 1), then
    contribute equally to a network's first layer at the start of training, and
    centred inputs make gradient descent better conditioned (LeCun, Bottou, Orr
    & Mueller 1998, "Efficient BackProp", in Neural Networks: Tricks of the
    Trade, LNCS 1524:9-50). The information in the encoding is unchanged: this
    is the same linear transformation for every residue.

    A feature that is the same for every character carries no information; it
    is centred to 0 and left unscaled, with a warning.

    Attributes
    ----------
    base : SequenceEncoder
            The encoding being standardized
    mean, scale : numpy arrays
            Per-feature mean and standard deviation subtracted and divided by
    name : str
            'standardized <base name>', for display
    input_size : int
            Same as the base encoding
    """

    def __init__(self, base: SequenceEncoder, mean: list[float] | None = None,
                 scale: list[float] | None = None) -> None:
        """
        Parameters
        ----------
        base : SequenceEncoder
                The encoding to standardize
        mean, scale : list of float, optional
                Per-feature mean and scale, as stored in a saved network. If
                omitted (the usual case) they are computed from the base
                encoding's table.
        """

        if (mean is None) != (scale is None):
            raise ValueError('Provide both or neither of the standardization mean and scale.')

        self.base = base
        self.name = f'{STANDARDIZED_NAME} {base.name}'
        self.input_size = base.input_size
        if mean is None or scale is None:
            self.mean, self.scale = standardization_parameters(base)
        else:
            self.mean, self.scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)

        if self.mean.shape != (self.input_size,) or self.scale.shape != (self.input_size,):
            raise ValueError(f'Standardization parameters must have {self.input_size} values each.')

    @property
    def alphabet(self) -> str:
        return self.base.alphabet

    def encode(self, seq: str) -> torch.Tensor:
        """Encode with the base encoding, then standardize every feature"""

        raw = self.base.encode(seq).numpy()
        return torch.from_numpy((raw - self.mean) / self.scale)

    def decode(self, seq_vectors: list[np.ndarray]) -> list[str]:
        """Convert encoded sequences back to strings, matching each vector to the nearest character"""

        characters = self.alphabet
        table = np.stack([self.encode(character).numpy()[0] for character in characters])

        sequences = []
        for seq_vector in seq_vectors:
            vectors = np.asarray(seq_vector, dtype=float)
            distances = ((vectors[:, None, :] - table[None, :, :]) ** 2).sum(axis=-1)
            sequences.append(''.join(characters[index] for index in distances.argmin(axis=1)))
        return sequences

    def to_config(self) -> dict[str, Any]:
        """The base encoding and the exact mean and scale are stored, so a saved network is self-contained"""

        return {'name': STANDARDIZED_NAME, 'base': self.base.to_config(),
                'mean': self.mean.tolist(), 'scale': self.scale.tolist()}


def standardization_parameters(encoder: SequenceEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation of an encoding across its alphabet

    Parameters
    ----------
    encoder : SequenceEncoder
            The encoding to standardize

    Returns
    -------
    mean : numpy array
            Mean of each feature across the characters the encoding defines
    scale : numpy array
            Population standard deviation of each feature, or 1 for a feature
            that is constant across the alphabet (with a warning)
    """

    table = np.stack([encoder.encode(character).numpy()[0] for character in encoder.alphabet])
    mean = table.mean(axis=0)
    standard_deviation = table.std(axis=0)

    typical_size = np.maximum(np.abs(table).max(axis=0), 1.0)
    constant = standard_deviation <= CONSTANT_FEATURE_TOLERANCE * typical_size
    if constant.any():
        warnings.warn(f'Encoding feature(s) {np.flatnonzero(constant).tolist()} have the same value for every '
                      'character, so they carry no information; standardizing sets them to 0.', stacklevel=2)

    return mean, np.where(constant, 1.0, standard_deviation)


# Encodings that can be selected by name; anything else is treated as a path to an encoding file
BUILT_IN_ENCODERS: dict[str, type[SequenceEncoder]] = {OneHotEncoder.name: OneHotEncoder,
                                                       BiophysicsEncoder.name: BiophysicsEncoder}


def get_encoder(encoding: str, standardize: bool = False) -> SequenceEncoder:
    """Build the encoder described by a command-line style encoding string

    Parameters
    ----------
    encoding : str
            'onehot', 'biophysics', the name of a protein language model (e.g.
            'esm2_t12_35M_UR50D'; see parrot.plm), or the path to a user
            encoding file
    standardize : bool, optional
            If True, standardize every feature of the encoding across its
            alphabet (see StandardizedEncoder). Not available for protein
            language models.

    Returns
    -------
    SequenceEncoder

    Raises
    ------
    FileNotFoundError
            If `encoding` is not a built-in name and no such file exists
    """

    # Imported here because parrot.plm builds on this module
    from parrot import plm

    if plm.is_plm_encoding(encoding):
        if standardize:
            raise ValueError('--standardize-encoding cannot be used with a protein language model encoding: '
                             'PLM embeddings depend on each residue\'s context, so there is no per-amino-acid '
                             'table to standardize')
        return plm.PLMEncoder(encoding)

    encoder: SequenceEncoder
    if encoding in BUILT_IN_ENCODERS:
        encoder = BUILT_IN_ENCODERS[encoding]()
    elif os.path.isfile(encoding):
        encoder = UserEncoder(encoding)
    else:
        raise FileNotFoundError(f"Encoding file does not exist: '{encoding}' (built-in encodings are "
                                f"{', '.join(BUILT_IN_ENCODERS)}; protein language models are "
                                f"{', '.join(plm.PLM_MODELS)})")

    return StandardizedEncoder(encoder) if standardize else encoder


def encoder_from_config(config: dict[str, Any]) -> SequenceEncoder:
    """Rebuild an encoder from the dictionary written by SequenceEncoder.to_config()

    Parameters
    ----------
    config : dict
            Encoder description, as stored in a saved network

    Returns
    -------
    SequenceEncoder
    """

    from parrot import plm

    if config['name'] == plm.PLM_CONFIG_NAME:
        return plm.encoder_from_plm_config(config)

    if config['name'] == STANDARDIZED_NAME:
        return StandardizedEncoder(encoder_from_config(config['base']), mean=config['mean'], scale=config['scale'])

    if config['name'] == UserEncoder.name:
        return UserEncoder(table=config['table'])

    if config['name'] not in BUILT_IN_ENCODERS:
        raise ValueError(f"Unknown encoding '{config['name']}' in saved network")

    return BUILT_IN_ENCODERS[config['name']]()
