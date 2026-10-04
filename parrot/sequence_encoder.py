"""
The interface every sequence encoding implements.

An encoder turns a sequence (a string) into one vector of numbers per residue.
PARROT's encodings are built on this base class: the per-amino-acid tables in
encode_sequence.py (one-hot, biophysical scales, user files, standardized
versions of these) and the protein language models in plm.py. This module
imports nothing else from PARROT, so every encoding module can build on it.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Iterable
from typing import Any

import torch

# The 20 standard amino acids, in the order of their one-hot encoding
STANDARD_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Sequences longer than this are shortened in error messages
MAX_SHOWN_SEQUENCE_LENGTH = 60


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
            shown = sequence
            if len(sequence) > MAX_SHOWN_SEQUENCE_LENGTH:
                shown = sequence[:MAX_SHOWN_SEQUENCE_LENGTH - 3] + '...'
            raise ValueError(f'Invalid amino acid detected: {sorted(invalid)[0]} (in sequence {shown})')


class SequenceEncoder:
    """Base class for objects that convert sequences to numeric vectors

    Everything outside the encoding modules works with SequenceEncoder objects,
    so a new encoding needs a subclass of this class, an entry in
    encode_sequence.get_encoder() (to select it by name) and one in
    encode_sequence.encoder_from_config() (to rebuild it from a saved network).

    Attributes
    ----------
    name : str
        Short name of the encoding, for display (e.g. 'onehot'). What is saved
        with a network is to_config().
    input_size : int
        Length of the vector each residue is encoded as
    """

    name: str = ''
    input_size: int = 0

    def __len__(self) -> int:
        """The length of the vector each residue is encoded as."""
        return self.input_size

    @property
    def alphabet(self) -> str:
        """Every character this encoding defines a vector for"""
        raise NotImplementedError

    def encode(self, seq: str) -> torch.Tensor:
        """Convert a sequence into a [len(seq) X input_size] tensor

        Parameters
        ----------
        seq : str
            The sequence

        Returns
        -------
        torch.Tensor
            float64 for the per-amino-acid tables, float32 for protein language models
        """
        raise NotImplementedError

    def check_sequences(self, sequences: Iterable[str]) -> None:
        """Raise if any sequence contains a character this encoding cannot encode

        Checking every sequence before training or predicting reports a bad
        residue straight away, rather than when its sequence is first encoded
        (for a test-set sequence, only after training has finished).

        Parameters
        ----------
        sequences : iterable of str

        Raises
        ------
        ValueError
            For the first sequence with a character outside the alphabet
        """
        check_characters(sequences, set(self.alphabet))

    def to_config(self) -> dict[str, Any]:
        """Return a plain dictionary from which encode_sequence.encoder_from_config() can rebuild this encoder

        This is what is saved with a network.

        Returns
        -------
        dict
        """
        return {'name': self.name}
