"""
Protein language model (PLM) encodings: per-residue embeddings from ESM-2 or ESM C.

A PLM encoding replaces the fixed per-amino-acid vectors of one-hot or
biophysics encoding with the embedding a pretrained protein language model gives
each residue in the context of its whole sequence. The PLM itself is frozen
(never trained); PARROT trains its own network on the embeddings.

Running the PLM is by far the slowest step, so each sequence is embedded only
once and the result is kept in an on-disk cache (see EmbeddingCache):

- parrot-train and parrot-optimize embed every sequence in the datafile before
  training starts, and every epoch then reads the cached embeddings;
- parrot-predict (and Predictor.predict_batch()) embeds the sequences to
  predict before predicting;
- parrot-preprocess --encode embeds every sequence of its output ahead of time,
  so any number of networks can then be trained on them without embedding again.

Fragments cut out of larger proteins can instead be embedded within their
full-length proteins (--embedding-context; see parrot/embedding_context.py):
each full-length protein is embedded, and each fragment's residues are cut out
of its embedding. These in-context embeddings are kept in a file of their own
in the cache (see InContextEmbeddingCache), so they are never mistaken for
embeddings of sequences on their own.

Two families of models are supported:

- ESM-2 (Lin et al. 2023, Science 379:1123, doi:10.1126/science.ade2574), loaded
  through Hugging Face transformers (the facebook/esm2_* models);
- ESM C (EvolutionaryScale 2024, "ESM Cambrian", evolutionaryscale.ai/blog/esm-cambrian;
  MIT licence), loaded through EvolutionaryScale's esm package.

The model weights are not stored in PARROT's network files: a saved network
records which model and settings produced its embeddings, and the model is
loaded by name. Model libraries and h5py are only imported when a PLM encoding
is used (h5py also for NARDINI encodings, which share the embedding cache);
install them with `pip install idptools-parrot[plm]` (ESM-2) or
`pip install idptools-parrot[esmc]` (ESM C). Do not install Meta's fair-esm in
the same environment as EvolutionaryScale's esm: both use the import name `esm`.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import datetime
import hashlib
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from parrot.devices import full_float32_precision
from parrot.sequence_encoder import SequenceEncoder, check_characters

# The two families of models, which are loaded and run through different packages
ESM2_FAMILY = 'esm2'
ESMC_FAMILY = 'esmc'
PLM_FAMILIES = (ESM2_FAMILY, ESMC_FAMILY)


@dataclass(frozen=True)
class PLMModel:
    """A protein language model PARROT can embed with.

    Attributes
    ----------
    family : str
        'esm2' (loaded through Hugging Face transformers) or 'esmc' (loaded
        through EvolutionaryScale's esm package)
    embedding_size : int
        Values per residue
    max_window : int
        Longest stretch of residues the model was trained on. Longer sequences
        are embedded in overlapping windows of this many residues, and each
        residue's embedding is the average over the windows that contain it.
    """

    family: str
    embedding_size: int
    max_window: int

    def __post_init__(self) -> None:
        """Check the family is one PARROT can load, so a mistake in the model list fails as soon as it is read."""
        if self.family not in PLM_FAMILIES:
            raise ValueError(f"Unknown protein language model family '{self.family}': must be one of {PLM_FAMILIES}")


# ESM-2 was trained on crops of 1,024 tokens: up to 1,022 residues plus the start and
# end tokens (Lin et al. 2023)
ESM2_MAX_WINDOW = 1022

# ESM C was trained with a context of 2,048 tokens in its final training stage
# (EvolutionaryScale 2024, ESM Cambrian blog): up to 2,046 residues plus start and end
ESMC_MAX_WINDOW = 2046

# Every supported model. ESM-2 embedding sizes: Lin et al. 2023, Table S1; ESM C
# widths: the ESM Cambrian blog.
PLM_MODELS = {
    'esm2_t6_8M_UR50D': PLMModel(ESM2_FAMILY, 320, ESM2_MAX_WINDOW),
    'esm2_t12_35M_UR50D': PLMModel(ESM2_FAMILY, 480, ESM2_MAX_WINDOW),
    'esm2_t30_150M_UR50D': PLMModel(ESM2_FAMILY, 640, ESM2_MAX_WINDOW),
    'esm2_t33_650M_UR50D': PLMModel(ESM2_FAMILY, 1280, ESM2_MAX_WINDOW),
    'esm2_t36_3B_UR50D': PLMModel(ESM2_FAMILY, 2560, ESM2_MAX_WINDOW),
    'esm2_t48_15B_UR50D': PLMModel(ESM2_FAMILY, 5120, ESM2_MAX_WINDOW),
    'esmc_300m': PLMModel(ESMC_FAMILY, 960, ESMC_MAX_WINDOW),
    'esmc_600m': PLMModel(ESMC_FAMILY, 1152, ESMC_MAX_WINDOW),
}

# Hugging Face organisation that publishes the ESM-2 models
HUGGINGFACE_ORGANISATION = 'facebook'

# Name stored in a saved network's encoding config for a PLM encoding
PLM_CONFIG_NAME = 'plm'

# Residues shared by consecutive windows of a long sequence
WINDOW_OVERLAP_RESIDUES = 256

# Embeddings are stored (and, for consistency, always rounded) as 16-bit floats, which
# halves the cache size. Their ~3 significant figures are far more precise than the
# networks trained on them need.
STORAGE_DTYPE = np.float16

# Every character ESM-2 and ESM C read as one residue: the uppercase letters except J (the 20
# standard amino acids, X for unknown, and B, Z, U and O). Both read anything else, including
# lowercase letters, as an unknown token, and ESM-2 reads a run of lowercase letters as a single
# token, which would silently misalign the embeddings of every residue after it.
PLM_RESIDUES = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'

# Maximum residues (padding included) the PLM processes at once; bounds GPU memory
MAX_TOKENS_PER_BATCH = 16_000

# When filling a cache, embeddings are written after every this many residues (50 batches' worth), so an
# interrupted run keeps most of its work
RESIDUES_PER_CACHE_WRITE = 50 * MAX_TOKENS_PER_BATCH

# Refuse to start embedding unless the disk has this many times the estimated space free
DISK_SPACE_SAFETY_FACTOR = 1.25

# How often (in seconds) the progress bar of a cache being filled is redrawn: as often as tqdm likes on a terminal,
# but only every half minute when its output goes to a file (e.g. the log of a run started with nohup), so a long run
# does not fill its log with updates
TERMINAL_PROGRESS_INTERVAL_SECONDS = 0.1
LOG_PROGRESS_INTERVAL_SECONDS = 30

# tqdm also redraws, from a monitor thread, any bar not redrawn for this many seconds (its default maxinterval); a bar
# that is redrawn less often than this raises it to its own interval, or the monitor would redraw it in between
TQDM_DEFAULT_MAXINTERVAL_SECONDS = 10

# Progress bars with at least this many units count in thousands and millions (e.g. 5.01M residues), and smaller ones
# in whole units (tqdm's scaling would show 3 sequences as 3.00)
PROGRESS_UNIT_SCALE_FROM = 1000

# Several processes may share a cache (see EmbeddingCache): one waiting for another to finish writing checks again
# this often...
CACHE_POLL_SECONDS = 0.2

# ...for at most this long, since a write of RESIDUES_PER_CACHE_WRITE residues takes seconds, so a lock file still
# there after this long is almost certainly left by a process that stopped while writing
CACHE_WAIT_SECONDS = 600

# Default cache directories, created next to the command's output
TRAIN_CACHE_DIRECTORY = 'parrot_train_embeddings'
PREDICT_CACHE_DIRECTORY = 'parrot_predict_embeddings'

# Embeddings of fragments cut out of their full-length proteins are kept in the cache as <model><suffix>.h5 (see
# InContextEmbeddingCache)
CONTEXT_CACHE_SUFFIX = '.context'

# Recorded in a saved network's encoding config when its sequences were embedded within their full-length proteins
IN_CONTEXT_CONFIG_KEY = 'embedded_in_context'

# Embeddings with their mean over the residues subtracted (--embed-mean-subtraction) differ from the others, so they
# are kept in the cache as <model><suffix>.h5 (and <model><suffix>.context.h5 in context)
MEAN_SUBTRACTED_CACHE_SUFFIX = '.mean_subtracted'

# Recorded in a saved network's encoding config when each embedding had its mean over the residues subtracted
MEAN_SUBTRACTED_CONFIG_KEY = 'mean_subtracted'


@dataclass(frozen=True)
class FragmentContext:
    """Where a fragment sits in the full-length protein it is embedded within (see parrot/embedding_context.py)

    Attributes
    ----------
    full_length_sequence : str
        The full-length protein
    start : int
        0-based position of the fragment's first residue in it
    """

    full_length_sequence: str
    start: int


def is_plm_encoding(encoding: str) -> bool:
    """True if `encoding` names a protein language model PARROT can embed with."""
    return encoding in PLM_MODELS


def human_size(num_bytes: float) -> str:
    """A byte count as a short human-readable string, e.g. '1.4 GB'."""
    for unit in ('bytes', 'KB', 'MB', 'GB'):
        if num_bytes < 1000 or unit == 'GB':
            return f'{num_bytes:,.0f} {unit}' if unit == 'bytes' else f'{num_bytes:,.1f} {unit}'
        num_bytes /= 1000
    raise AssertionError('unreachable')


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    """Start positions of the windows that cover a sequence

    Parameters
    ----------
    length : int
        Sequence length
    window : int
        Maximum residues per window
    overlap : int
        Residues shared by consecutive windows

    Returns
    -------
    list of int
        0-based starts. A sequence no longer than `window` has one window; the
        last window of a longer sequence ends exactly at the sequence end.

    Raises
    ------
    ValueError
        If the sequence is longer than `window` and `overlap` is not between 0 and window - 1
    """
    if length <= window:
        return [0]
    if not 0 <= overlap < window:
        raise ValueError(f'Window overlap must be between 0 and the window size, got {overlap}')

    starts = list(range(0, length - window + 1, window - overlap))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


# .................................................................
# The on-disk cache

class EmbeddingCache:
    """On-disk store of the embeddings one model gives each sequence

    Embeddings live in <directory>/<model name>.h5, an HDF5 file with one
    dataset per sequence, named by the SHA-256 hash of the sequence and grouped
    by the hash's first two characters. A dataset is [length X embedding size]
    for per-residue embeddings (PLMs), or [embedding size] for one vector per
    sequence (NARDINI features; see parrot/nardini.py). Each dataset also
    stores the sequence itself, which is checked on reading. A README.md in the
    directory describes what is there.

    Several processes can use one cache at once (e.g. training runs pointed at
    the same --embedding-cache). Only one writes at a time: a <model>.h5.lock
    file marks a cache that is being written, and another process that wants to
    write waits for it to go. Readers do not lock the file, so a long training
    run reading the cache never stops another process adding to it; instead they
    wait while a write is going on, reopen the file once it has changed, and
    read again anything they were reading while a write began.

    Attributes
    ----------
    directory : Path
        The cache directory
    model_name : str
        What produced the embeddings (a PLM, or NARDINI with its settings)
    embedding_size : int
        Length of each residue's embedding (or of each sequence's vector)
    per_residue : bool
        True if each residue has an embedding, False if each sequence has one vector
    storage_dtype : numpy dtype
        The precision embeddings are stored at
    path : Path
        The HDF5 file
    """

    def __init__(self, directory: str | Path, model_name: str, embedding_size: int, per_residue: bool = True,
                 storage_dtype: Any = STORAGE_DTYPE) -> None:
        """
        Parameters
        ----------
        directory : str or Path
            The cache directory (created when the cache is first written)
        model_name : str
            What produced the embeddings; also the name of the HDF5 file
        embedding_size : int
            Length of each residue's embedding (or of each sequence's vector)
        per_residue : bool, optional
            True if each residue has an embedding, False if each sequence has one vector
        storage_dtype : numpy dtype, optional
            The precision embeddings are stored at (default STORAGE_DTYPE)
        """
        self.directory = Path(directory).resolve()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.per_residue = per_residue
        self.storage_dtype = np.dtype(storage_dtype)
        self.path = self.directory / f'{model_name}.h5'
        self._reader: Any = None
        # The file's (modification time, size) when the reader opened it
        self._reader_state: tuple[int, int] | None = None

    def key(self, sequence: str) -> str:
        """The name an embedding is stored under: '<first two hash characters>/<SHA-256 of the sequence>'."""
        return hash_key(sequence)

    def attributes(self, sequence: str) -> dict[str, Any]:
        """What is stored with a sequence's embedding, and checked when it is read: the sequence itself."""
        return {'sequence': sequence}

    def missing(self, sequences: Iterable[str]) -> list[str]:
        """The unique sequences (in first-seen order) that are not yet in the cache."""
        unique = list(dict.fromkeys(sequences))
        if not self.path.exists():
            return unique
        return self._consistent_read(lambda reader: [sequence for sequence in unique
                                                     if self.key(sequence) not in reader])

    def get(self, sequence: str) -> np.ndarray:
        """The cached embedding of a sequence, at the precision it is stored

        Raises
        ------
        KeyError
            If the sequence is not in the cache
        FileNotFoundError
            If the cache has no file yet
        ValueError
            If the cache holds a different sequence under the same hash
        """
        def read(reader: Any) -> np.ndarray | None:
            """The sequence's embedding from the open file, or None if it is not there."""
            key = self.key(sequence)
            if key not in reader:
                return None
            dataset = reader[key]
            for name, value in self.attributes(sequence).items():
                if dataset.attrs[name] != value:
                    raise ValueError(f'Embedding cache {self.path} holds a different sequence under the same hash')
            return np.asarray(dataset[()])

        embedding = self._consistent_read(read)
        if embedding is None:
            raise KeyError(f'Sequence is not in the embedding cache {self.path}')
        return embedding

    def _consistent_read(self, read: Callable[[Any], Any]) -> Any:
        """Read from the file, again if another process began writing to it during the read

        Parameters
        ----------
        read : callable
            Reads what is needed from an open h5py.File

        Returns
        -------
        Whatever `read` returns
        """
        while True:
            reader = self._open_reader()
            state = self._reader_state
            result = read(reader)
            if not self._lock_path.exists() and self._file_state() == state:
                return result

    def write(self, embeddings: dict[str, np.ndarray]) -> None:
        """Add embeddings (sequence -> [length X embedding size] array, or [embedding size] vector) to the cache

        Sequences already in the cache are skipped. If another process is
        writing to the cache, this waits for it to finish (see wait_for_lock_file).

        Raises
        ------
        RuntimeError
            If another process's lock file is still there after CACHE_WAIT_SECONDS
        ValueError
            If the file holds embeddings from something else, or an embedding has the wrong shape
        """
        import h5py

        self.directory.mkdir(parents=True, exist_ok=True)
        self.close()
        with self._lock(), h5py.File(self.path, 'a') as store:
            if 'model' not in store.attrs:
                store.attrs['model'] = self.model_name
                store.attrs['embedding_size'] = self.embedding_size
                store.attrs['per_residue'] = self.per_residue
            if store.attrs['model'] != self.model_name or store.attrs['embedding_size'] != self.embedding_size:
                raise ValueError(f'{self.path} holds embeddings from a different model')

            added_sequences, added_residues = 0, 0
            for sequence, embedding in embeddings.items():
                key = self.key(sequence)
                if key in store:
                    continue
                expected_shape = (len(sequence), self.embedding_size) if self.per_residue else (self.embedding_size,)
                if embedding.shape != expected_shape:
                    raise ValueError(f'Embedding of shape {embedding.shape} does not match the expected shape '
                                     f'{expected_shape} (a sequence of length {len(sequence)})')
                dataset = store.create_dataset(key, data=embedding.astype(self.storage_dtype))
                for name, value in self.attributes(sequence).items():
                    dataset.attrs[name] = value
                added_sequences += 1
                added_residues += len(sequence)

            store.attrs['num_sequences'] = int(store.attrs.get('num_sequences', 0)) + added_sequences
            store.attrs['num_residues'] = int(store.attrs.get('num_residues', 0)) + added_residues

    def stored_layout(self) -> tuple[int, bool, str]:
        """(values per embedding, whether embeddings are per residue, storage dtype), as recorded in the file."""
        def read(reader: Any) -> tuple[int, bool, str]:
            """The layout from the open file's attributes and its first dataset (if any)."""
            first = next((group[name] for group in reader.values() for name in group), None)
            storage_dtype = first.dtype.name if first is not None else self.storage_dtype.name
            return (int(reader.attrs.get('embedding_size', 0)), bool(reader.attrs.get('per_residue', True)),
                    storage_dtype)

        layout: tuple[int, bool, str] = self._consistent_read(read)
        return layout

    def summary(self) -> tuple[int, int, int]:
        """(number of sequences, number of residues, file size in bytes) of the cache."""
        if not self.path.exists():
            return 0, 0, 0
        counts: tuple[int, int] = self._consistent_read(lambda reader: (int(reader.attrs.get('num_sequences', 0)),
                                                                        int(reader.attrs.get('num_residues', 0))))
        return counts[0], counts[1], self.path.stat().st_size

    def _open_reader(self) -> Any:
        """A read-only handle on the HDF5 file, kept open for repeated reads

        Waits while another process writes to the file, and reopens it if it
        has changed since it was opened (so the handle sees what was added).
        The file is opened without HDF5's own file locking, which would stop
        any other process writing to it for as long as the handle is open.
        """
        import h5py

        wait_for_lock_file(self._lock_path)
        state = self._file_state()
        if self._reader is not None and state != self._reader_state:
            self.close()
        if self._reader is None:
            self._reader = h5py.File(self.path, 'r', locking=False)
            self._reader_state = state
        return self._reader

    def _file_state(self) -> tuple[int, int]:
        """The file's modification time (ns) and size, which change whenever it is written."""
        status = self.path.stat()
        return status.st_mtime_ns, status.st_size

    def close(self) -> None:
        """Close the file handle kept open for reading (it is reopened when next needed)."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._reader_state = None

    @property
    def _lock_path(self) -> Path:
        """The lock file that marks the cache as being written: <model>.h5.lock."""
        return self.path.with_suffix('.h5.lock')

    def _lock(self) -> '_CacheLock':
        """The lock to hold while writing to the cache."""
        return _CacheLock(self._lock_path)


class InContextEmbeddingCache(EmbeddingCache):
    """On-disk store of the embeddings of fragments cut out of their full-length proteins' embeddings

    Embeddings live in <directory>/<cache name>.context.h5 (see
    CONTEXT_CACHE_SUFFIX; the cache name is the model's, marked if the
    embeddings are mean-subtracted, as PLMEncoder.cache_name gives it), apart
    from the embeddings of sequences on their own, which are different. The
    cache is used through each fragment's sequence, like EmbeddingCache, but
    each dataset is named by the SHA-256 hash of the full-length sequence, the
    fragment's start in it and the fragment. So the
    same fragment within two proteins (or at two places in one) has two entries,
    and adding fragments or proteins never makes what is stored wrong. Each
    dataset also stores the fragment, the SHA-256 hash of its full-length
    sequence and its start, which are checked on reading.

    Attributes
    ----------
    contexts : mapping
        Fragment sequence -> its FragmentContext. This is the encoder's own
        dictionary, so contexts the encoder is given later are seen here too.
    """

    def __init__(self, directory: str | Path, model_name: str, embedding_size: int,
                 contexts: Mapping[str, FragmentContext | None]) -> None:
        """
        Parameters
        ----------
        directory : str or Path
            The cache directory (created when the cache is first written)
        model_name : str
            The PLM's cache name (see PLMEncoder.cache_name); the file is named after it, with CONTEXT_CACHE_SUFFIX
        embedding_size : int
            Length of each residue's embedding
        contexts : mapping
            Fragment sequence -> its FragmentContext (every fragment read or written must have one)
        """
        super().__init__(directory, model_name + CONTEXT_CACHE_SUFFIX, embedding_size)
        self.contexts = contexts

    def context(self, sequence: str) -> FragmentContext:
        """The fragment's context

        Raises
        ------
        KeyError
            If the fragment has none (it is embedded on its own, so its embedding belongs in EmbeddingCache)
        """
        context = self.contexts.get(sequence)
        if context is None:
            raise KeyError(f'Sequence {sequence} has no full-length context, so it has no in-context embedding')
        return context

    def key(self, sequence: str) -> str:
        """The name an embedding is stored under: the hash of the full-length sequence, the start and the fragment."""
        context = self.context(sequence)
        return hash_key(f'{context.full_length_sequence}:{context.start}:{sequence}')

    def attributes(self, sequence: str) -> dict[str, Any]:
        """What is stored with a fragment's embedding, and checked when it is read: the fragment and its context."""
        context = self.context(sequence)
        return {'sequence': sequence, 'start': context.start,
                'full_length_sha256': hashlib.sha256(context.full_length_sequence.encode()).hexdigest()}


def hash_key(text: str) -> str:
    """'<first two characters of the SHA-256 hash of text>/<the hash>': the name of an embedding in a cache file."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f'{digest[:2]}/{digest}'


def wait_for_lock_file(path: Path) -> None:
    """Wait while another process holds a cache's lock file (it is writing to the cache)

    Prints a note when it starts waiting.

    Parameters
    ----------
    path : Path
        The lock file

    Raises
    ------
    RuntimeError
        If the lock file is still there after CACHE_WAIT_SECONDS
    """
    waited = 0.0
    while path.exists():
        if waited == 0:
            print(f'Waiting for another PARROT process to finish writing to the embedding cache ({path} exists)')
        if waited >= CACHE_WAIT_SECONDS:
            raise RuntimeError(f'Another PARROT process has been writing to this embedding cache for over '
                               f'{CACHE_WAIT_SECONDS} seconds ({path} exists). If no other process is running, '
                               'delete that lock file and try again.')
        time.sleep(CACHE_POLL_SECONDS)
        waited += CACHE_POLL_SECONDS


class _CacheLock:
    """Exclusive lock file held while a process writes to a cache (another writer waits for it; see
    wait_for_lock_file)."""

    def __init__(self, path: Path) -> None:
        """Lock the cache with the lock file at `path` (taken on entering the with block)."""
        self.path = path

    def __enter__(self) -> None:
        """Create the lock file, once no other process holds it (see wait_for_lock_file for when this gives up)."""
        while True:
            wait_for_lock_file(self.path)
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                continue  # another writer took it first: wait for that one too
        os.write(descriptor, f'pid {os.getpid()}\n'.encode())
        os.close(descriptor)

    def __exit__(self, *exception: Any) -> None:
        """Remove the lock file, whether or not the write succeeded."""
        self.path.unlink()


def write_cache_readme(directory: Path, program: str) -> None:
    """Write (or rewrite) the README.md that explains an embedding cache directory

    Parameters
    ----------
    directory : Path
        The cache directory
    program : str
        The PARROT command that created or last updated the cache
    """
    rows = []
    total_bytes = 0
    for path in sorted(directory.glob('*.h5')):
        cache = EmbeddingCache(directory, path.stem, 0)
        embedding_size, per_residue, _ = cache.stored_layout()
        num_sequences, num_residues, num_bytes = cache.summary()
        cache.close()
        total_bytes += num_bytes
        values = f'{embedding_size} per {"residue" if per_residue else "sequence"}'
        rows.append(f'| `{path.name}` | {path.stem} | {values} | {num_sequences:,} | {num_residues:,} | '
                    f'{human_size(num_bytes)} |')

    updated = datetime.datetime.now().isoformat(timespec='seconds')
    text = f"""# PARROT embedding cache

This directory holds encodings that are slow to compute, for PARROT: per-residue embeddings from a protein language model (PLM), and NARDINI+ features of whole sequences. It was last updated by `{program}`, on {updated}.

Computing these is by far the slowest part of using such an encoding, so every sequence is encoded once and stored here. Training reads the stored values every epoch instead of recomputing them, and any later command that uses the same encoding and cache directory (training another network, or predicting the same sequences again) skips sequences that are already here.

## Contents

| File | Encoding | Values | Sequences | Residues | Size on disk |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Total: {human_size(total_bytes)}.

## Format

Each file is an HDF5 file with one dataset per sequence. Datasets are named by the SHA-256 hash of the sequence (grouped by its first two characters), and each records the sequence itself as an attribute.

- **PLM embeddings** (`<model>.h5`): each dataset has shape [sequence length X values per residue], stored as 16-bit floats. Each value is the model's final-layer representation of that residue (after the model's last layer norm), computed with the whole sequence as context. Sequences longer than the model was trained on ({ESM2_MAX_WINDOW:,} residues for ESM-2, {ESMC_MAX_WINDOW:,} for ESM C) are embedded in overlapping windows of that length, and residues covered by more than one window get the average. The size is about (number of residues) x (values per residue) x 2 bytes.
- **PLM embeddings in context** (`<model>{CONTEXT_CACHE_SUFFIX}.h5`, from `--embedding-context`): the same, but each fragment's embedding was cut out of the embedding of the full-length protein it comes from, so it differs from the fragment's embedding on its own. Datasets are named by the SHA-256 hash of the full-length sequence, the fragment's start in it and the fragment, and each records the fragment, its start (0-based) and the SHA-256 hash of the full-length sequence.
- **Mean-subtracted PLM embeddings** (`<model>{MEAN_SUBTRACTED_CACHE_SUFFIX}.h5`, and `<model>{MEAN_SUBTRACTED_CACHE_SUFFIX}{CONTEXT_CACHE_SUFFIX}.h5` in context, from `--embed-mean-subtraction`): the same, but the mean of each embedding over its residues (its mean pool) was subtracted from every residue's embedding before it was stored. In context, this is the mean over the whole full-length protein, subtracted before the fragment was cut out.
- **NARDINI+ features** (`nardini_*.h5`): each dataset is one vector of NARDINI+ z-scores for the whole sequence, stored as 32-bit floats, computed by SPARROW with the settings in the file name (the compositional background, the number of scrambles, the random seed and the SPARROW version). See PARROT's documentation on NARDINI encodings.

## Reusing or deleting this cache

- **Deleting it is safe.** Nothing here is needed to use a trained network; embeddings are simply recomputed when next needed.
- **To reuse it** (e.g. when training several networks on the same data, or predicting the same sequences again), pass `--embedding-cache <this directory>` to `parrot-train`, `parrot-optimize` or `parrot-predict`. `parrot-preprocess --encode` fills a cache ahead of time.
- Several PARROT commands can use a cache at once: they take turns to write to it, and a `<model>.h5.lock` file marks a cache being written. If one is left behind by a run that crashed, delete it.
"""
    (directory / 'README.md').write_text(text)


# .................................................................
# The encoder

class PLMEncoder(SequenceEncoder):
    """Encode residues by their embeddings from a protein language model (ESM-2 or ESM C)

    Before sequences are encoded in bulk, prepare() embeds every sequence not
    yet in the cache. encode() then reads embeddings from the cache, or runs the
    model on the spot for a sequence that is not there (or for every sequence,
    with no cache; as Predictor.predict() does by default). Embeddings are
    always rounded to float16, so storing one in the cache does not change it.
    The model's output does depend, through float32 rounding, on which other
    sequences it is batched with, so a sequence's embedding alone and its
    embedding in a batch can differ by one float16 step in a small fraction of
    its values.

    Once given contexts (add_contexts(), from --embedding-context), the encoder
    embeds sequences within their full-length proteins: it embeds each
    full-length protein and cuts each fragment's residues out of its embedding.
    Every sequence it encodes from then on must have been given a context (or
    None, to be embedded on its own), and a network trained with it records
    this, so that predictions embed new sequences in the same way.

    With mean_subtracted (--embed-mean-subtraction), the mean of each
    embedding over its residues (its mean pool) is subtracted from every
    residue's embedding before it is stored, so what remains is how each
    residue differs from the sequence as a whole. In context, the mean is that
    of the full-length protein's embedding, subtracted before the fragments are
    cut out of it. These embeddings are kept in their own cache files, and a
    network trained on them records this, so that predictions embed new
    sequences in the same way.

    Attributes
    ----------
    name : str
        The model name, e.g. 'esm2_t12_35M_UR50D'
    model_info : PLMModel
        The model's family, embedding size and maximum window
    input_size : int
        Values per residue
    window, overlap : int
        Window length and overlap used for sequences longer than the window
    device : torch.device
        Device the PLM runs on
    cache : EmbeddingCache or None
        Where embeddings of sequences on their own are stored, if anywhere
    context_cache : InContextEmbeddingCache or None
        Where in-context embeddings are stored, if anywhere (the same directory as `cache`)
    in_context : bool
        True if sequences are embedded within their full-length proteins
    contexts : dict
        Sequence -> its FragmentContext, or None for a sequence embedded on its own
    mean_subtracted : bool
        True if each embedding has its mean over the residues subtracted
    """

    def __init__(self, model_name: str, window: int | None = None,
                 overlap: int = WINDOW_OVERLAP_RESIDUES, mean_subtracted: bool = False) -> None:
        """
        Parameters
        ----------
        model_name : str
            A key of PLM_MODELS, e.g. 'esm2_t12_35M_UR50D' or 'esmc_300m'
        window : int, optional
            Window length for long sequences (default: the model's max_window)
        overlap : int, optional
            Residues shared by consecutive windows (default WINDOW_OVERLAP_RESIDUES)
        mean_subtracted : bool, optional
            Subtract each embedding's mean over its residues from every residue (default False)

        Raises
        ------
        ValueError
            If model_name is not a key of PLM_MODELS
        """
        if not is_plm_encoding(model_name):
            raise ValueError(f"Unknown protein language model '{model_name}': choose from "
                             f"{', '.join(PLM_MODELS)}")
        self.name = model_name
        self.model_info = PLM_MODELS[model_name]
        self.input_size = self.model_info.embedding_size
        self.window = window if window is not None else self.model_info.max_window
        self.overlap = overlap
        self.device = torch.device('cpu')
        self.cache: EmbeddingCache | None = None
        self.context_cache: InContextEmbeddingCache | None = None
        self.in_context = False
        self.contexts: dict[str, FragmentContext | None] = {}
        self.mean_subtracted = mean_subtracted
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def cache_name(self) -> str:
        """The name of this encoder's cache files: the model, marked if the embeddings are mean-subtracted."""
        return self.name + (MEAN_SUBTRACTED_CACHE_SUFFIX if self.mean_subtracted else '')

    @property
    def alphabet(self) -> str:
        """Not defined: a PLM embeds each residue in context (check_sequences() uses PLM_RESIDUES instead)."""
        raise NotImplementedError('A PLM embeds each residue in context, so it has no per-character table')

    def to_config(self) -> dict[str, Any]:
        """Everything needed to embed new sequences in the same way as the training sequences"""
        if self.model_info.family == ESM2_FAMILY:
            source = f'huggingface:{HUGGINGFACE_ORGANISATION}/{self.name}'
            representation = 'final layer after the final layer norm (last_hidden_state), start and end tokens removed'
        else:  # ESM C, the only other family (see PLMModel)
            source = f'evolutionaryscale-esm:{self.name}'
            representation = 'final layer after the final layer norm (ESMCOutput.embeddings), start and end tokens removed'
        config = {'name': PLM_CONFIG_NAME, 'model': self.name, 'source': source, 'representation': representation,
                  'window': self.window, 'overlap': self.overlap, 'storage_dtype': np.dtype(STORAGE_DTYPE).name}
        if self.in_context:
            config[IN_CONTEXT_CONFIG_KEY] = True
        if self.mean_subtracted:
            config[MEAN_SUBTRACTED_CONFIG_KEY] = True
        return config

    def use_device(self, device: torch.device) -> None:
        """Run the PLM on `device` from now on."""
        self.device = device
        if self._model is not None:
            self._model.to(device)

    def use_cache(self, directory: str | Path | None) -> None:
        """Store and read embeddings in `directory` (in files of their own if they are mean-subtracted), or, if it is
        None, in no cache (closing any cache in use, which is left as it is on disk)."""
        for cache in (self.cache, self.context_cache):
            if cache is not None:
                cache.close()
        if directory is None:
            self.cache = None
            self.context_cache = None
            return
        self.cache = EmbeddingCache(directory, self.cache_name, self.input_size)
        self.context_cache = InContextEmbeddingCache(directory, self.cache_name, self.input_size, self.contexts)

    def add_contexts(self, contexts: Mapping[str, FragmentContext | None]) -> None:
        """Embed each of these sequences within its full-length protein (or, where its context is None, on its own)

        From now on, the encoder embeds sequences in context (see the class
        docstring): every sequence it encodes must have been given a context
        here. A sequence given a context again takes the new one.

        Parameters
        ----------
        contexts : mapping
            Sequence -> its FragmentContext, or None (see embedding_context.assign_contexts())
        """
        self.in_context = True
        self.contexts.update(contexts)

    def context_of(self, sequence: str) -> FragmentContext | None:
        """The full-length context a sequence is embedded within, or None if it is embedded on its own

        Raises
        ------
        ValueError
            If the encoder embeds in context but the sequence was not given a context (see add_contexts())
        """
        if not self.in_context:
            return None
        if sequence not in self.contexts:
            raise ValueError(f'No full-length protein was given for the sequence {sequence}. Its embeddings are made '
                             'within the full-length proteins the sequences were cut from (--embedding-context), so '
                             "every sequence needs one (or '-' in the map, to be embedded on its own)")
        return self.contexts[sequence]

    def release_model(self) -> None:
        """Free the PLM (e.g. once every sequence is cached, before training)."""
        self._model = None
        self._tokenizer = None
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def encode(self, seq: str) -> torch.Tensor:
        """The [len(seq) X input_size] embedding of a sequence (from the cache if possible, adding it if missing)

        The sequence is embedded within its full-length protein if it has been given one (see add_contexts()).
        """
        in_context = self.context_of(seq) is not None
        cache = self.context_cache if in_context else self.cache
        embed = self.embed_in_context if in_context else self.embed

        if cache is None:
            embedding = embed([seq])[seq]
        else:
            if cache.missing([seq]):
                cache.write(embed([seq]))
            embedding = cache.get(seq)
        return torch.from_numpy(embedding.astype(np.float32))

    def check_sequences(self, sequences: Iterable[str]) -> None:
        """Raise if any sequence has a character the model cannot read as one residue (see PLM_RESIDUES)."""
        check_characters(sequences, set(PLM_RESIDUES))

    def prepare(self, sequences: Sequence[str], program: str, silent: bool = False) -> None:
        """Embed every sequence that is not yet in the cache, reporting the cache size first

        Sequences given a full-length context (see add_contexts()) are embedded
        within it, into the in-context cache; the others are embedded on their own.

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
            If no cache has been set, the disk does not have room for the cache, or another process's lock file is
            still there after CACHE_WAIT_SECONDS (see wait_for_lock_file())
        ValueError
            If a sequence has a character the model cannot read as one residue, the encoder embeds in context
            and a sequence was not given a context, or a cache file holds embeddings from a different model
        ImportError
            If the packages the model (or the cache) needs are not installed
        """
        if self.cache is None or self.context_cache is None:
            raise RuntimeError('Call use_cache() before prepare()')
        self.check_sequences(sequences)

        unique = list(dict.fromkeys(sequences))
        contexts = {sequence: self.context_of(sequence) for sequence in unique}
        in_context = [sequence for sequence in unique if contexts[sequence] is not None]
        on_own = [sequence for sequence in unique if contexts[sequence] is None]
        missing_on_own = self.cache.missing(on_own)
        missing_in_context = self.context_cache.missing(in_context)
        missing = missing_on_own + missing_in_context

        cached_bytes = self.cache.summary()[2] + self.context_cache.summary()[2]
        new_residues = sum(len(sequence) for sequence in missing)
        new_bytes = new_residues * self.input_size * np.dtype(STORAGE_DTYPE).itemsize

        if not silent:
            placed = (f' ({len(in_context):,} within their full-length proteins, {len(on_own):,} on their own)'
                      if self.in_context else '')
            mean_subtracted = ' (mean-subtracted)' if self.mean_subtracted else ''
            print(f'\nPLM encoding {self.name}{mean_subtracted}: {len(unique):,} unique sequences{placed}, '
                  f'{len(unique) - len(missing):,} already in the embedding cache at {self.cache.directory}')
            if missing:
                print(f'Embedding {len(missing):,} sequences ({new_residues:,} residues x {self.input_size} values x '
                      f'2 bytes): the cache will grow by about {human_size(new_bytes)} to about '
                      f'{human_size(cached_bytes + new_bytes)}')
            else:
                print(f'Every sequence is already embedded (cache size {human_size(cached_bytes)})')
            if missing_in_context:
                proteins = self._fragments_by_protein(missing_in_context)
                print(f'The {len(missing_in_context):,} sequences embedded in context are cut out of the embeddings of '
                      f'{len(proteins):,} full-length proteins ({sum(len(protein) for protein in proteins):,} '
                      'residues), which is what the model runs over')

        if missing:
            self.cache.directory.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.cache.directory).free
            if new_bytes * DISK_SPACE_SAFETY_FACTOR > free:
                raise RuntimeError(f'Not enough disk space for the embedding cache: about {human_size(new_bytes)} '
                                   f'needed, {human_size(free)} free in {self.cache.directory}. Use '
                                   '--embedding-cache (embedding_cache= for Predictor) to put the cache somewhere '
                                   'with more space.')
        if missing_on_own:
            self._embed_into_cache(missing_on_own, silent)
        if missing_in_context:
            self._embed_in_context_into_cache(missing_in_context, silent)

        write_cache_readme(self.cache.directory, program)

    def _embed_into_cache(self, sequences: list[str], silent: bool) -> None:
        """Embed sequences in groups, writing each group to the cache as it finishes

        Parameters
        ----------
        sequences : list of str
            Sequences not yet in the cache
        silent : bool
            If True, print no progress
        """
        assert self.cache is not None

        # Load the model before starting the bar, so the speed it shows is embedding speed
        self._load_model()

        # A group takes minutes to embed, so the bar moves with every batch the model finishes, not every group
        total = window_residues(sequences, self.window, self.overlap)
        with progress_bar(total, 'residues', 'Embedding', silent) as progress:
            for group in cache_write_groups(sequences, RESIDUES_PER_CACHE_WRITE):
                self.cache.write(self.embed(group, progress))

    def _embed_in_context_into_cache(self, fragments: list[str], silent: bool) -> None:
        """Embed fragments within their full-length proteins, in groups of proteins, writing each group as it finishes

        Parameters
        ----------
        fragments : list of str
            Fragments (each with a context) not yet in the in-context cache
        silent : bool
            If True, print no progress
        """
        assert self.context_cache is not None
        self._load_model()

        # The model runs over the full-length proteins, so they are what the bar counts and the groups are made of
        by_protein = self._fragments_by_protein(fragments)
        total = window_residues(by_protein, self.window, self.overlap)
        with progress_bar(total, 'residues', 'Embedding full-length proteins', silent) as progress:
            for proteins in cache_write_groups(list(by_protein), RESIDUES_PER_CACHE_WRITE):
                group = [fragment for protein in proteins for fragment in by_protein[protein]]
                self.context_cache.write(self.embed_in_context(group, progress))

    def embed_in_context(self, fragments: Sequence[str], progress: tqdm | None = None) -> dict[str, np.ndarray]:
        """Embed fragments within their full-length proteins: embed each protein once, and cut out each fragment

        Parameters
        ----------
        fragments : list of str
            Fragments, each given a context (see add_contexts())
        progress : tqdm, optional
            A progress bar to advance by the residues of the full-length proteins (see embed())

        Returns
        -------
        dict
            Fragment -> float16 [length X input_size] embedding, cut out of its full-length protein's embedding
            (which, if mean_subtracted, has had the protein's mean over its residues subtracted; see embed())
        """
        by_protein = self._fragments_by_protein(fragments)
        protein_embeddings = self.embed(list(by_protein), progress)

        embeddings = {}
        for protein, protein_fragments in by_protein.items():
            for fragment in protein_fragments:
                context = self.contexts[fragment]
                assert context is not None
                # A copy, so the full-length protein's embedding can be freed once every fragment is cut out
                embeddings[fragment] = protein_embeddings[protein][context.start:context.start + len(fragment)].copy()
        return embeddings

    def _fragments_by_protein(self, fragments: Iterable[str]) -> dict[str, list[str]]:
        """The unique fragments, grouped by the full-length protein they are embedded within

        Parameters
        ----------
        fragments : iterable of str
            Fragments, each given a context (see add_contexts())

        Returns
        -------
        dict
            Full-length sequence -> its fragments, both in first-seen order

        Raises
        ------
        ValueError
            If a fragment has no context
        """
        by_protein: dict[str, list[str]] = {}
        for fragment in dict.fromkeys(fragments):
            context = self.context_of(fragment)
            if context is None:
                raise ValueError(f'The sequence {fragment} has no full-length context to be embedded within')
            by_protein.setdefault(context.full_length_sequence, []).append(fragment)
        return by_protein

    @full_float32_precision()
    def embed(self, sequences: Sequence[str], progress: tqdm | None = None) -> dict[str, np.ndarray]:
        """Run the PLM over sequences

        Parameters
        ----------
        sequences : list of str
            Sequences to embed (any length)
        progress : tqdm, optional
            A progress bar to advance by the residues of each batch as the model finishes it (so by
            window_residues() over the whole of `sequences`)

        Returns
        -------
        dict
            Sequence -> float16 [length X input_size] embedding; if mean_subtracted, with the embedding's mean over
            its residues subtracted from every residue (a single residue's embedding is then all zeros)

        Raises
        ------
        ValueError
            If a sequence has a character the model cannot read as one residue (see PLM_RESIDUES)
        """
        self.check_sequences(sequences)
        unique = list(dict.fromkeys(sequences))

        # Cut every sequence into the windows that cover it: (sequence, start of the window, window) pieces
        pieces = []
        for sequence in unique:
            for start in window_starts(len(sequence), self.window, self.overlap):
                pieces.append((sequence, start, sequence[start:start + self.window]))

        sums = {sequence: np.zeros((len(sequence), self.input_size), dtype=np.float32) for sequence in unique}
        counts = {sequence: np.zeros((len(sequence), 1), dtype=np.float32) for sequence in unique}

        pieces.sort(key=lambda piece: len(piece[2]))
        for batch in token_batches(pieces, MAX_TOKENS_PER_BATCH):
            outputs = self._run_model([piece[2] for piece in batch])
            for (sequence, start, chunk), output in zip(batch, outputs, strict=True):
                sums[sequence][start:start + len(chunk)] += output
                counts[sequence][start:start + len(chunk)] += 1
            if progress is not None:
                progress.update(sum(len(chunk) for _, _, chunk in batch))

        embeddings = {}
        for sequence in unique:
            embedding = sums[sequence] / counts[sequence]
            if self.mean_subtracted:
                # Subtract the mean pool (at full precision, before rounding for storage)
                embedding -= embedding.mean(axis=0)
            embeddings[sequence] = embedding.astype(STORAGE_DTYPE)
        return embeddings

    def _run_model(self, chunks: list[str]) -> list[np.ndarray]:
        """Final-layer embeddings of a batch of sequences (each at most `window` residues)

        Parameters
        ----------
        chunks : list of str
            The sequences (or windows of longer sequences) to embed together

        Returns
        -------
        list of numpy arrays
            float32 [len(chunk) X input_size] for each chunk, without the start
            and end tokens
        """
        model, tokenizer = self._load_model()
        tokens = tokenizer(chunks, return_tensors='pt', padding=True)['input_ids'].to(self.device)
        with torch.no_grad():
            if self.model_info.family == ESM2_FAMILY:
                hidden = model(input_ids=tokens, attention_mask=tokens != tokenizer.pad_token_id).last_hidden_state
            else:
                # ESM C (the only other family) masks padding itself, from the pad token
                hidden = model(sequence_tokens=tokens).embeddings

        outputs = []
        for i, chunk in enumerate(chunks):
            # Token 0 is the start token; residues follow, then the end token and padding
            outputs.append(hidden[i, 1:len(chunk) + 1].float().cpu().numpy())
        return outputs

    def _load_model(self) -> tuple[Any, Any]:
        """Load the PLM and its tokenizer (once)."""
        if self._model is None:
            if self.model_info.family == ESM2_FAMILY:
                self._model, self._tokenizer = self._load_esm2()
            else:  # ESM C, the only other family (see PLMModel)
                self._model, self._tokenizer = self._load_esmc()
            self._model = self._model.float().to(self.device).eval()
        return self._model, self._tokenizer

    def _load_esm2(self) -> tuple[Any, Any]:
        """An ESM-2 model and tokenizer from Hugging Face."""
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError:
            raise ImportError('ESM-2 encodings need the transformers and h5py packages: '
                              'pip install idptools-parrot[plm]') from None
        identifier = f'{HUGGINGFACE_ORGANISATION}/{self.name}'
        model = EsmModel.from_pretrained(identifier, add_pooling_layer=False)
        if model.config.hidden_size != self.input_size:
            raise ValueError(f'{identifier} has embedding size {model.config.hidden_size}, expected {self.input_size}')
        return model, AutoTokenizer.from_pretrained(identifier)

    def _load_esmc(self) -> tuple[Any, Any]:
        """An ESM C model (and its tokenizer) from EvolutionaryScale's esm package."""
        try:
            from esm.pretrained import LOCAL_MODEL_REGISTRY
        except ImportError:
            raise ImportError("ESM C encodings need EvolutionaryScale's esm package: pip install "
                              "idptools-parrot[esmc]. If Meta's fair-esm is installed, it hides EvolutionaryScale's "
                              'package (both use the import name esm); uninstall fair-esm and reinstall esm.') from None
        # Built on the CPU in float32 and with standard attention: ESMC.from_pretrained would
        # switch a GPU model to bfloat16 (and flash attention), changing the embeddings
        model = LOCAL_MODEL_REGISTRY[self.name](device='cpu', use_flash_attn=False)
        width = model.embed.embedding_dim
        if width != self.input_size:
            raise ValueError(f'{self.name} has embedding size {width}, expected {self.input_size}')
        return model, model.tokenizer


def cache_write_groups(sequences: list[str], residues_per_group: int) -> list[list[str]]:
    """Divide sequences into groups of similar length with about `residues_per_group` residues each

    Grouping sequences of similar length means their batches waste little
    padding; writing each group as it finishes saves progress regularly.

    Parameters
    ----------
    sequences : list of str
    residues_per_group : int
        A group is closed once it has at least this many residues

    Returns
    -------
    list of list of str
        The groups, shortest sequences first
    """
    groups: list[list[str]] = [[]]
    group_residues = 0
    for sequence in sorted(sequences, key=len):
        if group_residues >= residues_per_group:
            groups.append([])
            group_residues = 0
        groups[-1].append(sequence)
        group_residues += len(sequence)
    return [group for group in groups if group]


def window_residues(sequences: Iterable[str], window: int, overlap: int) -> int:
    """The residues the PLM runs over to embed the unique sequences of `sequences`

    Every residue counts once, except where the windows of a sequence longer than `window` overlap (see
    window_starts()), where it counts once per window.

    Parameters
    ----------
    sequences : iterable of str
    window, overlap : int
        Window length and overlap used for sequences longer than the window

    Returns
    -------
    int
    """
    return sum(min(window, len(sequence) - start) for sequence in dict.fromkeys(sequences)
               for start in window_starts(len(sequence), window, overlap))


def progress_bar(total: int, unit: str, description: str, silent: bool) -> tqdm:
    """A tqdm progress bar, on stderr, for filling an embedding cache

    On a terminal the bar is redrawn as it moves. When stderr goes to a file instead, it is redrawn every
    LOG_PROGRESS_INTERVAL_SECONDS, so that a long run's log gets an update every half minute. Totals of at least
    PROGRESS_UNIT_SCALE_FROM are counted in thousands or millions.

    Parameters
    ----------
    total : int
        The number of units to work through
    unit : str
        What is counted (e.g. 'residues')
    description : str
        Shown before the bar
    silent : bool
        If True, show nothing

    Returns
    -------
    tqdm
    """
    interval = TERMINAL_PROGRESS_INTERVAL_SECONDS if sys.stderr.isatty() else LOG_PROGRESS_INTERVAL_SECONDS
    return tqdm(total=total, unit=f' {unit}', unit_scale=total >= PROGRESS_UNIT_SCALE_FROM, desc=description,
                disable=silent, mininterval=interval, maxinterval=max(interval, TQDM_DEFAULT_MAXINTERVAL_SECONDS))


def token_batches(pieces: list[tuple[str, int, str]], max_tokens: int) -> list[list[tuple[str, int, str]]]:
    """Group (sequence, start, chunk) pieces so each padded batch has at most max_tokens residues

    Parameters
    ----------
    pieces : list of (sequence, start, chunk)
        Pieces sorted by chunk length, shortest first
    max_tokens : int
        Maximum of (pieces in a batch) x (longest chunk in the batch)

    Returns
    -------
    list of lists
        The batches, in order (a single chunk longer than max_tokens gets a batch of its own)
    """
    batches: list[list[tuple[str, int, str]]] = []
    batch: list[tuple[str, int, str]] = []
    for piece in pieces:
        # Pieces are sorted, so the newest piece is the batch's longest
        if batch and (len(batch) + 1) * len(piece[2]) > max_tokens:
            batches.append(batch)
            batch = []
        batch.append(piece)
    if batch:
        batches.append(batch)
    return batches


def encoder_from_plm_config(config: dict[str, Any]) -> PLMEncoder:
    """Rebuild a PLMEncoder from the config stored in a saved network

    Parameters
    ----------
    config : dict
        Written by PLMEncoder.to_config()

    Returns
    -------
    PLMEncoder
        With the network's model, windows and mean subtraction, and embedding in context if the network was
        trained so (each sequence's context is then given later; see PLMEncoder.add_contexts())

    Raises
    ------
    ValueError
        If the network was trained on embeddings stored at a precision this version of PARROT does not produce,
        or with a model it does not know
    """
    stored_dtype = config.get('storage_dtype', np.dtype(STORAGE_DTYPE).name)
    if stored_dtype != np.dtype(STORAGE_DTYPE).name:
        raise ValueError(f'This network was trained on {stored_dtype} embeddings, which this version of PARROT '
                         'does not produce')
    # Networks saved before --embed-mean-subtraction existed used embeddings as the model gives them
    encoder = PLMEncoder(config['model'], window=config['window'], overlap=config['overlap'],
                         mean_subtracted=bool(config.get(MEAN_SUBTRACTED_CONFIG_KEY, False)))
    # Networks saved before --embedding-context existed embedded every sequence on its own
    encoder.in_context = bool(config.get(IN_CONTEXT_CONFIG_KEY, False))
    return encoder
