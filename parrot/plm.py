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
- parrot-predict embeds the sequences to predict before predicting.

Two families of models are supported:

- ESM-2 (Lin et al. 2023, Science 379:1123, doi:10.1126/science.ade2574), loaded
  through Hugging Face transformers (the facebook/esm2_* models);
- ESM C (EvolutionaryScale 2024, "ESM Cambrian", evolutionaryscale.ai/blog/esm-cambrian;
  MIT licence), loaded through EvolutionaryScale's esm package.

The model weights are not stored in PARROT's network files: a saved network
records which model and settings produced its embeddings, and the model is
loaded by name. Model libraries and h5py are only imported when a PLM encoding
is used; install them with `pip install idptools-parrot[plm]` (ESM-2) or
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
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

# How often (in seconds) to print progress while embedding
PROGRESS_INTERVAL_SECONDS = 30

# Default cache directories, created next to the command's output
TRAIN_CACHE_DIRECTORY = 'parrot_train_embeddings'
PREDICT_CACHE_DIRECTORY = 'parrot_predict_embeddings'


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
    """On-disk store of per-residue embeddings for one model

    Embeddings live in <directory>/<model name>.h5, an HDF5 file with one
    dataset per sequence ([length X embedding size], float16), named by the
    SHA-256 hash of the sequence and grouped by the hash's first two characters.
    Each dataset also stores the sequence itself, which is checked on reading.
    A README.md in the directory describes what is there.

    Only one process may write to a cache at a time; a <model>.h5.lock file
    marks a cache that is being written.

    Attributes
    ----------
    directory : Path
        The cache directory
    model_name : str
        The PLM whose embeddings are stored
    embedding_size : int
        Length of each residue's embedding
    path : Path
        The HDF5 file
    """

    def __init__(self, directory: str | Path, model_name: str, embedding_size: int) -> None:
        self.directory = Path(directory).resolve()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.path = self.directory / f'{model_name}.h5'
        self._reader: Any = None

    @staticmethod
    def key(sequence: str) -> str:
        """The name an embedding is stored under: '<first two hash characters>/<SHA-256 of the sequence>'."""
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        return f'{digest[:2]}/{digest}'

    def missing(self, sequences: Iterable[str]) -> list[str]:
        """The unique sequences (in first-seen order) that are not yet in the cache."""
        unique = list(dict.fromkeys(sequences))
        if not self.path.exists():
            return unique
        reader = self._open_reader()
        return [sequence for sequence in unique if self.key(sequence) not in reader]

    def get(self, sequence: str) -> np.ndarray:
        """The cached [len(sequence) X embedding size] float16 embedding of a sequence

        Raises
        ------
        KeyError
            If the sequence is not in the cache
        """
        reader = self._open_reader()
        key = self.key(sequence)
        if key not in reader:
            raise KeyError(f'Sequence is not in the embedding cache {self.path}')
        dataset = reader[key]
        if dataset.attrs['sequence'] != sequence:
            raise ValueError(f'Embedding cache {self.path} holds a different sequence under the same hash')
        return np.asarray(dataset[()])

    def write(self, embeddings: dict[str, np.ndarray]) -> None:
        """Add embeddings (sequence -> [length X embedding size] array) to the cache

        Raises
        ------
        RuntimeError
            If another process is writing to the cache
        """
        import h5py

        self.directory.mkdir(parents=True, exist_ok=True)
        self.close()
        with self._lock(), h5py.File(self.path, 'a') as store:
            if 'model' not in store.attrs:
                store.attrs['model'] = self.model_name
                store.attrs['embedding_size'] = self.embedding_size
            if store.attrs['model'] != self.model_name or store.attrs['embedding_size'] != self.embedding_size:
                raise ValueError(f'{self.path} holds embeddings from a different model')

            added_sequences, added_residues = 0, 0
            for sequence, embedding in embeddings.items():
                key = self.key(sequence)
                if key in store:
                    continue
                if embedding.shape != (len(sequence), self.embedding_size):
                    raise ValueError(f'Embedding of shape {embedding.shape} does not match a sequence of length '
                                     f'{len(sequence)} and embedding size {self.embedding_size}')
                dataset = store.create_dataset(key, data=embedding.astype(STORAGE_DTYPE))
                dataset.attrs['sequence'] = sequence
                added_sequences += 1
                added_residues += len(sequence)

            store.attrs['num_sequences'] = int(store.attrs.get('num_sequences', 0)) + added_sequences
            store.attrs['num_residues'] = int(store.attrs.get('num_residues', 0)) + added_residues

    def summary(self) -> tuple[int, int, int]:
        """(number of sequences, number of residues, file size in bytes) of the cache."""
        if not self.path.exists():
            return 0, 0, 0
        reader = self._open_reader()
        return (int(reader.attrs.get('num_sequences', 0)), int(reader.attrs.get('num_residues', 0)),
                self.path.stat().st_size)

    def _open_reader(self) -> Any:
        """A read-only handle on the HDF5 file, kept open for repeated reads."""
        import h5py

        if self._reader is None:
            self._reader = h5py.File(self.path, 'r')
        return self._reader

    def close(self) -> None:
        """Close the file handle kept open for reading (it is reopened when next needed)."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _lock(self) -> '_CacheLock':
        return _CacheLock(self.path.with_suffix('.h5.lock'))


class _CacheLock:
    """Exclusive lock file held while a process writes to a cache."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> None:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f'Another PARROT process is writing to this embedding cache ({self.path} exists). '
                               'If no other process is running, delete that lock file and try again.') from None
        os.write(descriptor, f'pid {os.getpid()}\n'.encode())
        os.close(descriptor)

    def __exit__(self, *exception: Any) -> None:
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
        model_name = path.stem
        known = PLM_MODELS.get(model_name)
        cache = EmbeddingCache(directory, model_name, known.embedding_size if known is not None else 0)
        num_sequences, num_residues, num_bytes = cache.summary()
        cache.close()
        total_bytes += num_bytes
        rows.append(f'| `{path.name}` | {model_name} | {cache.embedding_size} | {num_sequences:,} | {num_residues:,} | '
                    f'{human_size(num_bytes)} |')

    updated = datetime.datetime.now().isoformat(timespec='seconds')
    text = f"""# PARROT protein-language-model embedding cache

This directory holds per-residue embeddings computed by a protein language model (PLM) for PARROT. It was created by `{program}` and last updated on {updated}.

Running the PLM is by far the slowest part of using a PLM encoding, so every sequence is embedded once and stored here. Training reads these embeddings every epoch instead of recomputing them, and any later command that uses the same model and cache directory (training another network, or predicting the same sequences again) skips sequences that are already here.

## Contents

| File | Model | Values per residue | Sequences | Residues | Size on disk |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Total: {human_size(total_bytes)}.

## Format

Each `<model>.h5` file is an HDF5 file with one dataset per sequence, of shape [sequence length X values per residue], stored as 16-bit floats. Datasets are named by the SHA-256 hash of the sequence (grouped by its first two characters), and each records the sequence itself as an attribute. Each value is the model's final-layer representation of that residue (after the model's last layer norm), computed with the whole sequence as context. Sequences longer than the model was trained on ({ESM2_MAX_WINDOW:,} residues for ESM-2, {ESMC_MAX_WINDOW:,} for ESM C) are embedded in overlapping windows of that length, and residues covered by more than one window get the average.

The size is about (number of residues) x (values per residue) x 2 bytes.

## Reusing or deleting this cache

- **Deleting it is safe.** Nothing here is needed to use a trained network; embeddings are simply recomputed when next needed.
- **To reuse it** (e.g. when training several networks on the same data, or predicting the same sequences again), pass `--embedding-cache <this directory>` to `parrot-train`, `parrot-optimize` or `parrot-predict`.
- Only one PARROT command should write to a cache at a time. A `<model>.h5.lock` file marks a cache being written; if one is left behind by a run that crashed, delete it.
"""
    (directory / 'README.md').write_text(text)


# .................................................................
# The encoder

class PLMEncoder(SequenceEncoder):
    """Encode residues by their embeddings from a protein language model (ESM-2 or ESM C)

    Before sequences are encoded in bulk, prepare() embeds every sequence not
    yet in the cache. encode() then reads embeddings from the cache, or, with no
    cache, runs the model on the spot (as Predictor does for single sequences).
    Embeddings are always rounded to float16, so a sequence is encoded the same
    way whether or not it went through the cache.

    Attributes
    ----------
    name : str
        The model name, e.g. 'esm2_t12_35M_UR50D'
    input_size : int
        Values per residue
    window, overlap : int
        Window length and overlap used for sequences longer than the window
    device : torch.device
        Device the PLM runs on
    cache : EmbeddingCache or None
        Where embeddings are stored, if anywhere
    """

    def __init__(self, model_name: str, window: int | None = None,
                 overlap: int = WINDOW_OVERLAP_RESIDUES) -> None:
        """
        Parameters
        ----------
        model_name : str
            A key of PLM_MODELS, e.g. 'esm2_t12_35M_UR50D' or 'esmc_300m'
        window : int, optional
            Window length for long sequences (default: the model's max_window)
        overlap : int, optional
            Residues shared by consecutive windows
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
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def alphabet(self) -> str:
        """Not defined: a PLM embeds each residue in context (check_sequences() uses PLM_RESIDUES instead)."""
        raise NotImplementedError('A PLM embeds each residue in context, so it has no per-character table')

    def to_config(self) -> dict[str, Any]:
        """Everything needed to embed new sequences exactly as the training sequences were"""
        if self.model_info.family == ESM2_FAMILY:
            source = f'huggingface:{HUGGINGFACE_ORGANISATION}/{self.name}'
            representation = 'final layer after the final layer norm (last_hidden_state), start and end tokens removed'
        else:  # ESM C, the only other family (see PLMModel)
            source = f'evolutionaryscale-esm:{self.name}'
            representation = 'final layer after the final layer norm (ESMCOutput.embeddings), start and end tokens removed'
        return {'name': PLM_CONFIG_NAME, 'model': self.name, 'source': source, 'representation': representation,
                'window': self.window, 'overlap': self.overlap, 'storage_dtype': np.dtype(STORAGE_DTYPE).name}

    def use_device(self, device: torch.device) -> None:
        """Run the PLM on `device` from now on."""
        self.device = device
        if self._model is not None:
            self._model.to(device)

    def use_cache(self, directory: str | Path) -> None:
        """Store and read embeddings in `directory`."""
        self.cache = EmbeddingCache(directory, self.name, self.input_size)

    def release_model(self) -> None:
        """Free the PLM (e.g. once every sequence is cached, before training)."""
        self._model = None
        self._tokenizer = None
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def encode(self, seq: str) -> torch.Tensor:
        """The [len(seq) X input_size] embedding of a sequence (from the cache if possible)"""
        if self.cache is None:
            embedding = self.embed([seq])[seq]
        else:
            if self.cache.missing([seq]):
                self.cache.write(self.embed([seq]))
            embedding = self.cache.get(seq)
        return torch.from_numpy(embedding.astype(np.float32))

    def check_sequences(self, sequences: Iterable[str]) -> None:
        """Raise if any sequence has a character the model cannot read as one residue (see PLM_RESIDUES)."""
        check_characters(sequences, set(PLM_RESIDUES))

    def prepare(self, sequences: Sequence[str], program: str, silent: bool = False) -> None:
        """Embed every sequence that is not yet in the cache, reporting the cache size first

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
            If no cache has been set, or the disk does not have room for the cache
        """
        if self.cache is None:
            raise RuntimeError('Call use_cache() before prepare()')
        self.check_sequences(sequences)

        missing = self.cache.missing(sequences)
        cached_sequences, cached_residues, cached_bytes = self.cache.summary()
        new_residues = sum(len(sequence) for sequence in missing)
        new_bytes = new_residues * self.input_size * np.dtype(STORAGE_DTYPE).itemsize
        unique = len(set(sequences))

        if not silent:
            print(f'\nPLM encoding {self.name}: {unique:,} unique sequences, {unique - len(missing):,} already in the '
                  f'embedding cache at {self.cache.directory}')
            if missing:
                print(f'Embedding {len(missing):,} sequences ({new_residues:,} residues x {self.input_size} values x '
                      f'2 bytes): the cache will grow by about {human_size(new_bytes)} to about '
                      f'{human_size(cached_bytes + new_bytes)}')
            else:
                print(f'Every sequence is already embedded (cache size {human_size(cached_bytes)})')

        if missing:
            self.cache.directory.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.cache.directory).free
            if new_bytes * DISK_SPACE_SAFETY_FACTOR > free:
                raise RuntimeError(f'Not enough disk space for the embedding cache: about {human_size(new_bytes)} '
                                   f'needed, {human_size(free)} free in {self.cache.directory}. Use '
                                   '--embedding-cache to put the cache somewhere with more space.')
            self._embed_into_cache(missing, silent)

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
        total_residues = sum(len(sequence) for sequence in sequences)
        done_residues = 0

        # Load the model before starting the clock, so the reported speed is embedding speed
        self._load_model()
        start = last_report = time.monotonic()

        for group in cache_write_groups(sequences, RESIDUES_PER_CACHE_WRITE):
            self.cache.write(self.embed(group))
            done_residues += sum(len(sequence) for sequence in group)

            now = time.monotonic()
            if not silent and (now - last_report > PROGRESS_INTERVAL_SECONDS or done_residues == total_residues):
                print_embedding_progress(done_residues, total_residues, now - start)
                last_report = now

    @full_float32_precision()
    def embed(self, sequences: Sequence[str]) -> dict[str, np.ndarray]:
        """Run the PLM over sequences

        Parameters
        ----------
        sequences : list of str
            Sequences to embed (any length)

        Returns
        -------
        dict
            Sequence -> float16 [length X input_size] embedding
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

        return {sequence: (sums[sequence] / counts[sequence]).astype(STORAGE_DTYPE) for sequence in unique}

    def _run_model(self, chunks: list[str]) -> list[np.ndarray]:
        """Final-layer embeddings of a batch of sequences (each at most `window` residues)

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


def print_embedding_progress(done_residues: int, total_residues: int, elapsed_seconds: float) -> None:
    """Print how many residues have been embedded, how fast, and roughly how long the rest will take

    Parameters
    ----------
    done_residues, total_residues : int
        Residues embedded so far, and in total
    elapsed_seconds : float
        Time spent embedding so far
    """
    message = f'  embedded {done_residues:,} of {total_residues:,} residues'
    if elapsed_seconds > 0:
        rate = done_residues / elapsed_seconds
        remaining_minutes = (total_residues - done_residues) / rate / 60
        message += f' ({rate:,.0f} residues/s, about {remaining_minutes:.1f} min left)'
    print(message)


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
    """Rebuild a PLMEncoder from the config stored in a saved network."""
    stored_dtype = config.get('storage_dtype', np.dtype(STORAGE_DTYPE).name)
    if stored_dtype != np.dtype(STORAGE_DTYPE).name:
        raise ValueError(f'This network was trained on {stored_dtype} embeddings, which this version of PARROT '
                         'does not produce')
    return PLMEncoder(config['model'], window=config['window'], overlap=config['overlap'])
