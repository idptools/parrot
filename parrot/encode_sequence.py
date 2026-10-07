"""
Encoding sequences as numbers: one vector per residue.

The built-in encodings are per-amino-acid tables, where every residue of a
given amino acid gets the same vector:

- 'onehot': 20 values, a single 1 marking which amino acid it is;
- 'biophysics': nine biophysical scales (see BIOPHYSICAL_SCALES);
- a user encoding file, which can encode any alphabet (see parse_encode_file());

and any of these can be standardized (see StandardizedEncoder). Protein
language model embeddings, where a residue's vector depends on the whole
sequence, are in plm.py. get_encoder() builds any of them from the name used on
the command line, and encoder_from_config() rebuilds one saved with a network.

one_hot(), biophysics() and UserEncoder are kept from earlier versions of
PARROT, which other packages use.

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from parrot import nardini, plm
from parrot.sequence_encoder import STANDARD_AMINO_ACIDS

# SequenceEncoder and check_characters live in sequence_encoder.py; they are also importable from here, where
# earlier versions of PARROT defined them
from parrot.sequence_encoder import SequenceEncoder as SequenceEncoder
from parrot.sequence_encoder import check_characters as check_characters

# Bytes run from 0 to 255; characters that fit in one byte are looked up by their byte value
NUM_BYTE_VALUES = 256


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
        self._byte_row = np.full(NUM_BYTE_VALUES, -1, dtype=np.int64)
        for character, row in self._row.items():
            if len(character) == 1 and ord(character) < NUM_BYTE_VALUES:
                self._byte_row[ord(character)] = row

    def encode(self, seq: str) -> torch.Tensor:
        """Encode a sequence as a float64 [len(seq) X vector length] tensor

        Parameters
        ----------
        seq : str

        Returns
        -------
        torch.Tensor

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


# .................................................................
# One-hot encoding

# The position of each amino acid's 1 in its one-hot vector
ONE_HOT = {amino_acid: index for index, amino_acid in enumerate(STANDARD_AMINO_ACIDS)}

_ONE_HOT_TABLE = LookupTable({amino_acid: np.eye(len(ONE_HOT))[index] for amino_acid, index in ONE_HOT.items()})


def one_hot(seq: str) -> torch.Tensor:
    """Convert an amino acid sequence to a PyTorch tensor of one-hot vectors

    Each amino acid is represented by a length 20 vector with a single 1 and
    19 0's, in the order of ONE_HOT. E.g. glutamic acid (E) is encoded as
    [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0].

    Parameters
    ----------
    seq : str
        An uppercase sequence of the 20 standard amino acids (single letter code)

    Returns
    -------
    torch.Tensor
        float64 [len(seq) X 20] tensor

    Raises
    ------
    ValueError
        If the sequence contains anything other than the 20 standard amino acids
    """
    return _ONE_HOT_TABLE.encode(seq)


# .................................................................
# Biophysical encoding

@dataclass(frozen=True)
class BiophysicalScale:
    """One of the scales the 'biophysics' encoding describes each amino acid by

    Attributes
    ----------
    name : str
        Short name of the scale
    units : str
        Units of its values
    description : str
        What it measures, and where the values come from
    """

    name: str
    units: str
    description: str


# The nine scales, in the order of the values in BIOPHYSICS. The source of the last three was not recorded
# when the table was added to PARROT.
BIOPHYSICAL_SCALES = (
    BiophysicalScale('hydrophobicity', 'Kyte-Doolittle hydropathy units',
                     'Kyte & Doolittle (1982) hydropathy scale (J Mol Biol 157:105-132)'),
    BiophysicalScale('charge', 'elementary charges',
                     'formal side-chain charge: -1 for D and E, +1 for K and R, and +1 for H (histidine is counted '
                     'as charged)'),
    BiophysicalScale('isoelectric_point', 'pH units', 'isoelectric point of the free amino acid'),
    BiophysicalScale('molecular_weight', 'g/mol', 'molecular weight of the free amino acid'),
    BiophysicalScale('aromatic', '1 or 0', '1 for the aromatic amino acids F, W and Y'),
    BiophysicalScale('hydrogen_bonding', '1 or 0',
                     '1 for side chains that can form hydrogen bonds (D, E, H, K, N, Q, R, S, T, W and Y)'),
    BiophysicalScale('side_chain_sasa', 'square angstroms',
                     'side-chain solvent-accessible surface area, measured in ACE-X-NME dipeptides (source not '
                     'recorded)'),
    BiophysicalScale('backbone_sasa', 'square angstroms',
                     'backbone solvent-accessible surface area, measured in ACE-X-NME dipeptides (source not '
                     'recorded)'),
    BiophysicalScale('solvation_free_energy', 'units not recorded',
                     'free energy of solvation of the side chain (source not recorded)'),
)

# Each amino acid's value on each scale of BIOPHYSICAL_SCALES. The histidine row was recorded as the average
# of its protonated (HIP) and neutral (HIE) forms.
#              hydro- charge  pI    mol.   arom- H-bond side-chain backbone solvation
#              phobic                weight atic          SASA       SASA     free energy
BIOPHYSICS = {'A': [1.8, 0, 6.0, 89.1, 0, 0, 75.8, 76.1, 1.9],
              'C': [2.5, 0, 5.1, 121.2, 0, 0, 115.4, 67.9, -1.2],
              'D': [-3.5, -1, 2.8, 133.1, 0, 1, 130.3, 71.8, -107.3],
              'E': [-3.5, -1, 3.2, 147.1, 0, 1, 161.8, 68.1, -107.3],
              'F': [2.8, 0, 5.5, 165.2, 1, 0, 209.4, 66.0, -0.8],
              'G': [-0.4, 0, 6.0, 75.1, 0, 0, 0.0, 115.0, 0.0],
              'H': [-3.2, 1, 7.6, 155.2, 0, 1, 180.8, 67.5, -52.7],
              'I': [4.5, 0, 6.0, 131.2, 0, 0, 172.7, 60.3, 2.2],
              'K': [-3.9, 1, 9.7, 146.2, 0, 1, 205.9, 68.7, -100.9],
              'L': [3.8, 0, 6.0, 131.2, 0, 0, 172.0, 64.5, 2.3],
              'M': [1.9, 0, 5.7, 149.2, 0, 0, 184.8, 67.8, -1.4],
              'N': [-3.5, 0, 5.4, 132.1, 0, 1, 142.7, 66.8, -9.7],
              'P': [-1.6, 0, 6.3, 115.1, 0, 0, 134.3, 55.8, 2.0],
              'Q': [-3.5, 0, 5.7, 146.2, 0, 1, 173.3, 66.6, -9.4],
              'R': [-4.5, 1, 10.8, 174.2, 0, 1, 236.5, 66.7, -100.9],
              'S': [-0.8, 0, 5.7, 105.1, 0, 1, 95.9, 72.9, -5.1],
              'T': [-0.7, 0, 5.6, 119.1, 0, 1, 130.9, 64.1, -5.0],
              'V': [4.2, 0, 6.0, 117.1, 0, 0, 143.1, 61.7, 2.0],
              'W': [-0.9, 0, 5.9, 204.2, 1, 1, 254.6, 64.3, -5.9],
              'Y': [-1.3, 0, 5.7, 181.2, 1, 1, 222.5, 71.9, -6.1]}

assert all(len(values) == len(BIOPHYSICAL_SCALES) for values in BIOPHYSICS.values())

_BIOPHYSICS_TABLE = LookupTable(BIOPHYSICS)


def biophysics(seq: str) -> torch.Tensor:
    """Convert an amino acid sequence to a PyTorch tensor with biophysical encoding

    Each amino acid is represented by its values on the nine scales of
    BIOPHYSICAL_SCALES: Kyte-Doolittle hydrophobicity, charge, isoelectric
    point, molecular weight, aromaticity, hydrogen-bonding ability, side-chain
    and backbone solvent-accessible surface area, and free energy of solvation.
    E.g. glutamic acid (E) is [-3.5, -1, 3.2, 147.1, 0, 1, 161.8, 68.1, -107.3].

    Parameters
    ----------
    seq : str
        An uppercase sequence of the 20 standard amino acids (single letter code)

    Returns
    -------
    torch.Tensor
        float64 [len(seq) X 9] tensor

    Raises
    ------
    ValueError
        If the sequence contains anything other than the 20 standard amino acids
    """
    return _BIOPHYSICS_TABLE.encode(seq)


# .................................................................
# User-specified encoding

def parse_encode_file(file: str) -> tuple[dict[str, list[float]], int]:
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
        If the file is empty, if the vectors are not all the same length, if
        any vector entry is not a number, or if a character is listed twice
    """

    with open(file) as f:
        lines = [x.strip().split() for x in f if len(x.strip()) > 0]

    if len(lines) == 0:
        raise ValueError(f'Encoding file {file} is empty.')

    vector_length = len(lines[0]) - 1
    table: dict[str, list[float]] = {}
    for line in lines:
        character, values = line[0], line[1:]
        if len(values) != vector_length:
            raise ValueError('Vectors in encoding file do not have same length.')
        if character in table:
            raise ValueError(f"Character '{character}' is listed more than once in encoding file {file}")

        try:
            table[character] = [float(value) for value in values]
        except ValueError:
            raise ValueError(f'Non-numeric value in encoding file for character {character}: {values}') from None

    return table, vector_length


# .................................................................
# Encoder objects (see sequence_encoder.SequenceEncoder)

class OneHotEncoder(SequenceEncoder):
    """One-hot encoding of the 20 standard amino acids (see one_hot())"""

    name = 'onehot'
    input_size = len(ONE_HOT)

    @property
    def alphabet(self) -> str:
        """The 20 standard amino acids."""
        return ''.join(ONE_HOT)

    def encode(self, seq: str) -> torch.Tensor:
        """See one_hot()."""
        return one_hot(seq)


class BiophysicsEncoder(SequenceEncoder):
    """Encoding of amino acids by nine biophysical scales (see biophysics())"""

    name = 'biophysics'
    input_size = len(BIOPHYSICAL_SCALES)

    @property
    def alphabet(self) -> str:
        """The 20 standard amino acids."""
        return ''.join(BIOPHYSICS)

    def encode(self, seq: str) -> torch.Tensor:
        """See biophysics()."""
        return biophysics(seq)


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
                raise FileNotFoundError(f'Encoding file does not exist: {encode_file}')
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
        torch.Tensor
            float64 [len(seq) X input_size] tensor
        """
        return self._table.encode(seq)

    @property
    def alphabet(self) -> str:
        """Every character the encoding file defines."""
        return ''.join(self.encode_dict)

    def to_config(self) -> dict[str, Any]:
        """The encoding table is stored in full so a saved network does not depend on the original file"""
        return {'name': self.name, 'table': self.encode_dict}


# .................................................................
# Standardized encodings

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
        'standardized <base name>', for display (the saved name is 'standardized')
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
        """The base encoding's alphabet."""
        return self.base.alphabet

    def encode(self, seq: str) -> torch.Tensor:
        """Encode with the base encoding, then standardize every feature (float64)."""
        raw = self.base.encode(seq).numpy()
        return torch.from_numpy((raw - self.mean) / self.scale)

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


# .................................................................
# Choosing an encoding

# Encodings that can be selected by name; anything else is treated as a path to an encoding file
BUILT_IN_ENCODERS: dict[str, type[SequenceEncoder]] = {OneHotEncoder.name: OneHotEncoder,
                                                       BiophysicsEncoder.name: BiophysicsEncoder}


def get_encoder(encoding: str, standardize: bool = False, nardini_scrambles: int | None = None) -> SequenceEncoder:
    """Build the encoder described by a command-line style encoding string

    Parameters
    ----------
    encoding : str
        'onehot', 'biophysics', the name of a protein language model (e.g.
        'esm2_t12_35M_UR50D'; see parrot.plm), the path to a user encoding
        file, 'nardini' (NARDINI+ features of the whole sequence; see
        parrot.nardini), or one of the others but a protein language model
        followed by '+nardini' (e.g. 'onehot+nardini')
    standardize : bool, optional
        If True, standardize every feature of the encoding across its
        alphabet (see StandardizedEncoder); for '<encoding>+nardini', of the
        per-residue encoding. Not available for protein language models or
        'nardini' alone.
    nardini_scrambles : int, optional
        NARDINI encodings only: scrambles per sequence for the patterning
        z-scores (default nardini.DEFAULT_NUM_SCRAMBLES)

    Returns
    -------
    SequenceEncoder

    Raises
    ------
    FileNotFoundError
        If `encoding` is not a built-in name and no such file exists
    ValueError
        If the options do not apply to the encoding
    """

    if nardini.is_nardini_encoding(encoding):
        return get_nardini_encoder(encoding, standardize, nardini_scrambles)
    if nardini_scrambles is not None:
        raise ValueError(f"--nardini-scrambles only applies to NARDINI encodings (e.g. 'nardini' or "
                         f"'onehot+nardini'), not '{encoding}'")

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
                                f"{', '.join(BUILT_IN_ENCODERS)} and '{nardini.NARDINI_ENCODING}', which can follow "
                                f"another with '+', e.g. 'onehot+{nardini.NARDINI_ENCODING}'; protein language models "
                                f"are {', '.join(plm.PLM_MODELS)})")

    return StandardizedEncoder(encoder) if standardize else encoder


def get_nardini_encoder(encoding: str, standardize: bool, num_scrambles: int | None) -> nardini.NardiniEncoder:
    """Build the encoder for 'nardini' or '<per-residue encoding>+nardini'

    Parameters
    ----------
    encoding : str
        'nardini', or a per-residue encoding name followed by '+nardini'
    standardize : bool
        Standardize the per-residue encoding (see get_encoder())
    num_scrambles : int or None
        Scrambles per sequence (None for the default)

    Returns
    -------
    NardiniEncoder

    Raises
    ------
    ValueError
        If 'nardini' alone is standardized, or combined with a protein language model
    """
    residue_name = nardini.residue_encoding_name(encoding)
    residue_encoder = None
    if residue_name is None:
        if standardize:
            raise ValueError("--standardize-encoding does not apply to 'nardini' alone: the NARDINI+ features are "
                             'already z-scores, and there is no per-amino-acid table to standardize')
    elif plm.is_plm_encoding(residue_name):
        raise ValueError(f"NARDINI features cannot be combined with a protein language model ('{encoding}'); combine "
                         "them with 'onehot', 'biophysics' or an encoding file")
    else:
        residue_encoder = get_encoder(residue_name, standardize=standardize)

    scrambles = num_scrambles if num_scrambles is not None else nardini.DEFAULT_NUM_SCRAMBLES
    return nardini.NardiniEncoder(residue_encoder, num_scrambles=scrambles)


def encoder_from_config(config: dict[str, Any]) -> SequenceEncoder:
    """Rebuild an encoder from the dictionary written by SequenceEncoder.to_config()

    Parameters
    ----------
    config : dict
        Encoder description, as stored in a saved network

    Returns
    -------
    SequenceEncoder

    Raises
    ------
    ValueError
        If the saved encoding is not one PARROT knows
    """

    if config['name'] == plm.PLM_CONFIG_NAME:
        return plm.encoder_from_plm_config(config)

    if config['name'] == nardini.NARDINI_ENCODING:
        residue_config = config['residue_encoding']
        residue_encoder = encoder_from_config(residue_config) if residue_config is not None else None
        return nardini.encoder_from_nardini_config(config, residue_encoder)

    if config['name'] == STANDARDIZED_NAME:
        return StandardizedEncoder(encoder_from_config(config['base']), mean=config['mean'], scale=config['scale'])

    if config['name'] == UserEncoder.name:
        return UserEncoder(table=config['table'])

    if config['name'] not in BUILT_IN_ENCODERS:
        raise ValueError(f"Unknown encoding '{config['name']}' in saved network")

    return BUILT_IN_ENCODERS[config['name']]()
