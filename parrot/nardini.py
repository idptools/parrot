"""
NARDINI+ sequence features as an encoding.

NARDINI asks whether the residues of an intrinsically disordered region are
arranged non-randomly. It groups the amino acids into eight types and, for each
type and each pair of types, compares how segregated they are along the
sequence with scrambles of that same sequence, giving 36 patterning z-scores.
NARDINI+ adds 54 compositional z-scores, against a reference set of IDRs (here
the human IDRome). PARROT uses SPARROW's implementation (sparrow.nardini).

These describe a whole sequence, while PARROT's encodings give one vector per
residue, so this encoding gives every residue its sequence's 90 features:

- `--encode nardini`: the features alone. With the composition baseline or the
  simple models, which average over residues, this is exactly a model of the
  NARDINI+ vector and the sequence length; order-aware networks see the same
  input at every residue.
- `--encode onehot+nardini` (or any per-residue encoding but a protein language
  model, e.g. biophysics+nardini): each residue's own encoding followed by its
  sequence's features, so a network sees both.

The features describe whole sequences, so only sequence-mapped data can use them.

The patterning z-scores are computed by scrambling each sequence (1,000 times by
default), which takes about 35-100 ms per sequence. So, as for protein language
models, every sequence is computed once before training, in parallel across CPU
cores, and stored in the embedding cache (see plm.EmbeddingCache). SPARROW
derives each sequence's scrambles from the sequence and a fixed seed, which is
saved with the network, so a sequence always gets the same features: when
training and when predicting, and whether or not they came from the cache.

SPARROW is only imported when this encoding is used. Install it with
`pip install git+https://github.com/idptools/sparrow.git`.

References
----------
Cohan, M. C., Shinn, M. K., Lalmansingh, J. M. & Pappu, R. V. Uncovering
non-random binary patterns within sequences of intrinsically disordered
proteins. J. Mol. Biol. 434, 167373 (2022). [NARDINI]

King, M. R. et al. Macromolecular condensation organizes nucleolar sub-phases
and sets up a pH gradient. Cell 187, 1889-1906.e24 (2024). [NARDINI+]

Ruff, K. M., King, M. R. et al. Molecular grammars of predicted intrinsically
disordered regions that span the human proteome. Cell (2025). [the human
IDRome background]

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import os
import time
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from parrot.plm import EmbeddingCache, write_cache_readme
from parrot.sequence_encoder import STANDARD_AMINO_ACIDS, SequenceEncoder
from parrot.tasks import Task

# The encoding's name, on the command line and in saved networks; '<encoding>+nardini' combines it with another
NARDINI_ENCODING = 'nardini'
COMBINATION_SEPARATOR = '+'

# Scrambles per sequence for the patterning z-scores. The NARDINI paper uses 100,000 and SPARROW defaults to 10,000;
# 1,000 is about ten times faster, and SPARROW notes it is usually adequate for low-complexity sequences.
DEFAULT_NUM_SCRAMBLES = 1000

# SPARROW derives each sequence's scrambles from this seed and the sequence, so features never depend on the batch
NARDINI_SEED = 0

# The settings the features are computed with. They are SPARROW's defaults, given explicitly so that the features
# stay the same if SPARROW's defaults change: the reference IDRome for the compositional z-scores (the human IDRome
# of Ruff et al. 2025), the eight residue types, the window size, the fraction of the sequence a residue type must
# exceed for its patterning features to be scored (others are 0), and the published NARDINI patterning statistic.
BACKGROUND = 'human'
TYPESET = '8'
BLOB_SIZE = 5
MIN_FRACTION = 0.10
PATTERNING_BACKEND = 'nardini'

# 36 patterning z-scores (pairs of the eight residue types) and 54 compositional z-scores
NUM_FEATURES = 90

# Sequences shorter than the window cannot be scored
MIN_SEQUENCE_LENGTH = BLOB_SIZE

# Features are stored and used as 32-bit floats (the compositional z-scores reach ~100 for sequences unlike IDRs,
# where 16-bit floats would round them to the nearest ~0.06)
FEATURE_DTYPE = np.float32

# Worker processes for computing features: every CPU core but one, which is left for the rest of the machine
NUM_PROCESSES = max(1, (os.cpu_count() or 1) - 1)

# Below this many sequences, starting worker processes costs more than it saves (SPARROW's own threshold)
MIN_SEQUENCES_FOR_PARALLEL = 16

# Sequences computed between writes to the cache, so an interrupted run keeps most of its work
SEQUENCES_PER_CACHE_WRITE = 2000

SPARROW_INSTALL = 'pip install git+https://github.com/idptools/sparrow.git'


# .................................................................
# Encoding names

def is_nardini_encoding(encoding: str) -> bool:
    """True for 'nardini' and '<per-residue encoding>+nardini'."""
    return encoding == NARDINI_ENCODING or encoding.endswith(COMBINATION_SEPARATOR + NARDINI_ENCODING)


def residue_encoding_name(encoding: str) -> str | None:
    """The per-residue encoding a NARDINI encoding combines with: 'onehot' for 'onehot+nardini', None for 'nardini'."""
    if encoding == NARDINI_ENCODING:
        return None
    return encoding[:-len(COMBINATION_SEPARATOR + NARDINI_ENCODING)]


# .................................................................
# Computing the features with SPARROW

def import_sparrow_nardini() -> Any:
    """Import sparrow.nardini, explaining how to install SPARROW if it is missing

    Returns
    -------
    module
        sparrow.nardini

    Raises
    ------
    ImportError
        If SPARROW is not installed
    """
    try:
        import sparrow.nardini
    except ImportError as error:
        raise ImportError(f'NARDINI encodings need SPARROW, which is not installed: {SPARROW_INSTALL}') from error
    return sparrow.nardini


def sparrow_version() -> str:
    """The installed SPARROW's version."""
    import sparrow

    return str(getattr(sparrow, '__version__', 'unknown'))


def feature_names() -> list[str]:
    """The names of the 90 features, in order: the patterning z-scores, then the compositional z-scores."""
    sparrow_nardini = import_sparrow_nardini()
    names = list(sparrow_nardini.patterning_feature_names(TYPESET)) + list(sparrow_nardini.composition_feature_names())
    assert len(names) == NUM_FEATURES, f'SPARROW gives {len(names)} NARDINI+ features, not {NUM_FEATURES}'
    return names


def compute_features(sequences: Sequence[str], num_scrambles: int, seed: int) -> dict[str, np.ndarray]:
    """Compute the NARDINI+ features of sequences with SPARROW, in parallel if there are many

    Parameters
    ----------
    sequences : list of str
        Unique sequences of the 20 standard amino acids, each at least MIN_SEQUENCE_LENGTH long
    num_scrambles : int
        Scrambles per sequence for the patterning z-scores
    seed : int
        Base seed (each sequence's scrambles are seeded from it and the sequence)

    Returns
    -------
    dict
        Sequence -> [NUM_FEATURES] float32 vector
    """
    sparrow_nardini = import_sparrow_nardini()
    background = sparrow_nardini.load_reference_background(BACKGROUND)
    num_processes = NUM_PROCESSES if len(sequences) >= MIN_SEQUENCES_FOR_PARALLEL else 1

    named = {str(index): sequence for index, sequence in enumerate(sequences)}
    result = sparrow_nardini.nardini_vectors(named, background=background, num_scrambles=num_scrambles, seed=seed,
                                             typeset=TYPESET, blob_size=BLOB_SIZE, min_fraction=MIN_FRACTION,
                                             backend=PATTERNING_BACKEND, n_processes=num_processes)
    zscores = np.asarray(result.zscores, dtype=FEATURE_DTYPE)
    return {named[name]: zscores[row] for row, name in enumerate(result.names)}


# .................................................................
# The encoder

class NardiniEncoder(SequenceEncoder):
    """Give every residue its sequence's NARDINI+ features, optionally after its own per-residue encoding

    Before sequences are encoded in bulk, prepare() computes every sequence not
    yet in the cache. encode() then reads features from memory or the cache,
    or computes them on the spot (as Predictor does for single sequences).
    Features are always 32-bit floats, so a sequence is encoded the same way
    whichever way its features were found.

    Attributes
    ----------
    name : str
        'nardini', or '<per-residue encoding>+nardini'
    input_size : int
        Values per residue: NUM_FEATURES, plus the per-residue encoding's
    residue_encoder : SequenceEncoder or None
        The per-residue encoding each residue's features follow, if any
    num_scrambles : int
        Scrambles per sequence for the patterning z-scores
    seed : int
        Base seed for the scrambles
    trained_with_sparrow : str or None
        For an encoder loaded with a saved network, the SPARROW version its
        features were computed with
    trained_feature_names : list of str or None
        For an encoder loaded with a saved network, the names of its features, in order
    cache : EmbeddingCache or None
        Where features are stored, if anywhere
    """

    def __init__(self, residue_encoder: SequenceEncoder | None = None, num_scrambles: int = DEFAULT_NUM_SCRAMBLES,
                 seed: int = NARDINI_SEED, trained_with_sparrow: str | None = None,
                 trained_feature_names: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        residue_encoder : SequenceEncoder, optional
            A per-residue encoding to combine the features with (not a protein language model)
        num_scrambles : int, optional
            Scrambles per sequence for the patterning z-scores
        seed : int, optional
            Base seed for the scrambles
        trained_with_sparrow : str, optional
            The SPARROW version a saved network's features were computed with
        trained_feature_names : list of str, optional
            The names of a saved network's features, in order

        Raises
        ------
        ValueError
            If num_scrambles is less than 1
        """
        if num_scrambles < 1:
            raise ValueError(f'The number of NARDINI scrambles must be at least 1, got {num_scrambles}')

        self.residue_encoder = residue_encoder
        self.num_scrambles = num_scrambles
        self.seed = seed
        self.trained_with_sparrow = trained_with_sparrow
        self.trained_feature_names = trained_feature_names
        self.name = NARDINI_ENCODING if residue_encoder is None else \
            residue_encoder.name + COMBINATION_SEPARATOR + NARDINI_ENCODING
        self.input_size = NUM_FEATURES + (len(residue_encoder) if residue_encoder is not None else 0)
        self.cache: EmbeddingCache | None = None
        self._features: dict[str, np.ndarray] = {}

    @property
    def alphabet(self) -> str:
        """The 20 standard amino acids (those the per-residue encoding also defines, if combined)"""
        if self.residue_encoder is None:
            return STANDARD_AMINO_ACIDS
        return ''.join(residue for residue in STANDARD_AMINO_ACIDS if residue in self.residue_encoder.alphabet)

    @property
    def cache_name(self) -> str:
        """The name of this encoder's file in the embedding cache, which records every setting the features depend on"""
        return f'{NARDINI_ENCODING}_{BACKGROUND}_{self.num_scrambles}scrambles_seed{self.seed}'

    def to_config(self) -> dict[str, Any]:
        """Everything needed to compute the features of new sequences exactly as the training sequences' were"""
        return {'name': NARDINI_ENCODING,
                'residue_encoding': self.residue_encoder.to_config() if self.residue_encoder is not None else None,
                'num_scrambles': self.num_scrambles, 'seed': self.seed, 'background': BACKGROUND,
                'typeset': TYPESET, 'blob_size': BLOB_SIZE, 'min_fraction': MIN_FRACTION,
                'patterning_backend': PATTERNING_BACKEND, 'feature_names': feature_names(),
                'sparrow_version': self.trained_with_sparrow or sparrow_version()}

    def use_cache(self, directory: str | Path) -> None:
        """Store and read features in `directory` (an embedding cache)."""
        self.cache = EmbeddingCache(directory, self.cache_name, NUM_FEATURES, per_residue=False,
                                    storage_dtype=FEATURE_DTYPE)

    def check_sequences(self, sequences: Iterable[str]) -> None:
        """Raise if any sequence cannot be scored: it has a non-standard residue or is too short

        Parameters
        ----------
        sequences : iterable of str

        Raises
        ------
        ValueError
            For the first problem found
        """
        sequences = list(sequences)
        if self.residue_encoder is not None:
            self.residue_encoder.check_sequences(sequences)

        valid = set(STANDARD_AMINO_ACIDS)
        nonstandard = sorted({residue for sequence in sequences for residue in set(sequence) - valid})
        if nonstandard:
            raise ValueError(f'NARDINI features need the 20 standard amino acids, but the sequences contain '
                             f'{", ".join(nonstandard)}')

        too_short = [sequence for sequence in sequences if len(sequence) < MIN_SEQUENCE_LENGTH]
        if too_short:
            raise ValueError(f'{len(too_short)} sequence(s) are shorter than {MIN_SEQUENCE_LENGTH} residues, the '
                             f'shortest NARDINI can score (e.g. {too_short[0]}). Remove them (from a datafile, e.g. '
                             f'with parrot-preprocess --min-length {MIN_SEQUENCE_LENGTH}).')

    def prepare(self, sequences: Sequence[str], program: str, silent: bool = False) -> None:
        """Compute the features of every sequence that is not yet in the cache, in groups, writing each group

        Parameters
        ----------
        sequences : list of str
            Every sequence that will be encoded
        program : str
            The PARROT command running, recorded in the cache README
        silent : bool, optional
            If True, print nothing

        Raises
        ------
        RuntimeError
            If no cache has been set
        """
        if self.cache is None:
            raise RuntimeError('Call use_cache() before prepare()')
        self.check_sequences(sequences)
        self.check_sparrow_matches_training()

        missing = self.cache.missing(sequences)
        unique = len(set(sequences))
        if not silent:
            print(f'\nNARDINI+ features ({self.num_scrambles:,} scrambles): {unique:,} unique sequences, '
                  f'{unique - len(missing):,} already in the cache at {self.cache.directory}')
            if missing:
                processes = NUM_PROCESSES if len(missing) >= MIN_SEQUENCES_FOR_PARALLEL else 1
                print(f'Computing {len(missing):,} sequences on {processes} CPU core(s)')

        start = time.monotonic()
        for first in range(0, len(missing), SEQUENCES_PER_CACHE_WRITE):
            group = missing[first:first + SEQUENCES_PER_CACHE_WRITE]
            self.cache.write(compute_features(group, self.num_scrambles, self.seed))
            if not silent and len(missing) > SEQUENCES_PER_CACHE_WRITE:
                print_progress(first + len(group), len(missing), time.monotonic() - start)

        write_cache_readme(self.cache.directory, program)

    def check_sparrow_matches_training(self) -> None:
        """For a saved network, check the installed SPARROW computes the features it was trained on

        Raises
        ------
        ValueError
            If SPARROW's features differ (in name or order) from the network's

        Warns
        -----
        UserWarning
            If the network's features were computed with a different SPARROW version
        """
        if self.trained_feature_names is not None and self.trained_feature_names != feature_names():
            raise ValueError("The installed SPARROW's NARDINI+ features differ from those this network was trained "
                             'on, so it cannot encode new sequences for it')
        if self.trained_with_sparrow is not None and self.trained_with_sparrow != sparrow_version():
            warnings.warn(f"This network's NARDINI+ features were computed with SPARROW {self.trained_with_sparrow}, "
                          f'but SPARROW {sparrow_version()} is installed. If its NARDINI implementation has changed '
                          'since, new sequences will be encoded differently from the training sequences.')

    def features(self, sequence: str) -> np.ndarray:
        """A sequence's [NUM_FEATURES] feature vector: from memory, the cache, or computed now

        Parameters
        ----------
        sequence : str

        Returns
        -------
        numpy array of float32
        """
        if sequence in self._features:
            return self._features[sequence]

        if self.cache is not None and not self.cache.missing([sequence]):
            vector = self.cache.get(sequence).astype(FEATURE_DTYPE)
        else:
            self.check_sequences([sequence])
            self.check_sparrow_matches_training()
            computed = compute_features([sequence], self.num_scrambles, self.seed)
            if self.cache is not None:
                self.cache.write(computed)
            vector = computed[sequence]

        self._features[sequence] = vector
        return vector

    def encode(self, seq: str) -> torch.Tensor:
        """The [len(seq) X input_size] float32 encoding: per-residue encoding (if any), then the sequence's features

        Parameters
        ----------
        seq : str

        Returns
        -------
        torch.Tensor
        """
        sequence_features = torch.from_numpy(self.features(seq)).unsqueeze(0).repeat(len(seq), 1)
        if self.residue_encoder is None:
            return sequence_features
        return torch.cat([self.residue_encoder.encode(seq).to(torch.float32), sequence_features], dim=1)


def check_task(encoder: SequenceEncoder, task: Task) -> None:
    """Refuse NARDINI encodings for data other than sequence-mapped data (they describe whole sequences)

    Parameters
    ----------
    encoder : SequenceEncoder
    task : Task

    Raises
    ------
    ValueError
        If a NARDINI encoding is used for residue or matrix data
    """
    if isinstance(encoder, NardiniEncoder) and task.datatype != 'sequence':
        raise ValueError(f"NARDINI+ features describe whole sequences, so --encode {encoder.name} is for "
                         f"sequence-mapped data (-d sequence), not '{task.datatype}' data")


def print_progress(done: int, total: int, elapsed_seconds: float) -> None:
    """Print how many sequences' features have been computed, and roughly how long the rest will take

    Parameters
    ----------
    done, total : int
        Sequences computed so far, and in total
    elapsed_seconds : float
    """
    message = f'  computed {done:,} of {total:,} sequences'
    if elapsed_seconds > 0:
        remaining_minutes = (total - done) * elapsed_seconds / done / 60
        message += f' (about {remaining_minutes:.1f} min left)'
    print(message)


def encoder_from_nardini_config(config: dict[str, Any], residue_encoder: SequenceEncoder | None) -> NardiniEncoder:
    """Rebuild a NardiniEncoder from the config stored in a saved network

    Parameters
    ----------
    config : dict
        Written by NardiniEncoder.to_config()
    residue_encoder : SequenceEncoder or None
        The per-residue encoding, already rebuilt from config['residue_encoding']

    Returns
    -------
    NardiniEncoder

    Raises
    ------
    ValueError
        If the network's features were computed with settings this version of PARROT does not use
    """
    used = {'background': BACKGROUND, 'typeset': TYPESET, 'blob_size': BLOB_SIZE, 'min_fraction': MIN_FRACTION,
            'patterning_backend': PATTERNING_BACKEND}
    differing = [name for name, value in used.items() if config.get(name) != value]
    if differing:
        raise ValueError(f'This network was trained on NARDINI+ features computed with different settings '
                         f'({", ".join(differing)}) from the ones this version of PARROT uses')
    return NardiniEncoder(residue_encoder, num_scrambles=config['num_scrambles'], seed=config['seed'],
                          trained_with_sparrow=config['sparrow_version'],
                          trained_feature_names=list(config['feature_names']))
