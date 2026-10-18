'''
Python module for integrating a trained network directly into a Python workflow.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.

'''

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
from scipy.special import softmax  # noqa: F401  (kept importable as py_predictor.softmax)

from parrot import checkpoint, devices, embedding_context, nardini, plm, prediction
from parrot.tasks import VALID_DATATYPES

# What the README of an embedding cache filled by predict_batch() says filled it (see plm.write_cache_readme)
PROGRAM_NAME = 'parrot.Predictor.predict_batch'

# Without an embedding_cache, predict_batch() encodes the sequences of a network trained on protein language model
# embeddings or NARDINI features into a temporary directory with this prefix, deleted once they are predicted
TEMPORARY_CACHE_PREFIX = 'parrot_predict_batch_'

# A per-sequence argument of predict_batch(): a full-length protein (str) or a start (int)
Value = TypeVar('Value', str, int)


class Predictor():
    """Class for integrating a trained PARROT network into a Python workflow

    Usage:
    >>> from parrot import py_predictor
    >>> my_predictor = py_predictor.Predictor('/path/to/saved_network.pt')
    >>> value = my_predictor.predict(AA_sequence)
    >>> values = my_predictor.predict_batch([AA_sequence_1, AA_sequence_2, ...])

    predict_batch() predicts the sequences together, as parrot-predict does:
    every sequence is encoded first (running a protein language model over many
    sequences at once), then the network runs over batches of similar length.
    For more than a handful of sequences, it is much faster. Its predictions
    match predict()'s up to float32 rounding, not exactly: which sequences
    share a batch changes the order of the floating-point operations.

    Networks saved by this version of PARROT record their architecture, task and
    encoding, so only the path is needed. Networks saved by earlier versions
    (which must be one-hot BRNNs) also need `dtype`.

    Predictions run on the CPU unless another device is requested, e.g.
    Predictor(path, device='auto') to use a GPU (CUDA or Apple-silicon MPS) when
    one is available, or Predictor(path, device='cuda', gpu_id=1) to use the
    second NVIDIA GPU.
    The CPU is the default because a GPU only pays off for
    many sequences at once, and MPS cannot be used inside forked worker
    processes (e.g. multiprocessing on macOS with the 'fork' start method).

    A network trained with --embedding-context embeds each sequence within the
    full-length protein it was cut from, so predict() needs that protein too
    (and predict_batch() one per sequence):
    >>> value = my_predictor.predict(fragment, full_length=protein_sequence)
    >>> values = my_predictor.predict_batch(fragments, full_lengths=protein_sequences)

    Attributes
    ----------
    dtype : str
        Data format that the network was trained for: "sequence", "residues" or
        "matrix".
    n_classes : int
        Number of data classes. If 1, the network is a regressor; if >1, a
        classifier with n_classes classes.
    task : str
        "classification" or "regression" (a name kept from earlier versions of
        PARROT; the network's full Task is network.task).
    num_layers : int or None
        The network's num_layers hyperparameter: the number of LSTM layers per
        direction for a BRNN (or of transformer blocks for a transformer); None
        for architectures without one.
    hidden_vector_size : int or None
        The network's hidden_size hyperparameter: the size of the LSTM hidden
        vectors for a BRNN (or the width of each hidden layer for a composition
        network); None for architectures without one.
    network : ParrotArchitecture
        The loaded network.
    encoder : SequenceEncoder
        The encoder the network was trained with.
    device : torch.device
        The device predictions run on.
    in_context : bool
        True if the network was trained with --embedding-context (so each
        sequence is embedded within its full-length protein).
    embed_in_isolation : bool
        True if a sequence predicted without its full-length protein is
        embedded on its own (networks trained with --embedding-context only).
    """

    def __init__(self, saved_weights: str | Path, dtype: str | None = None, device: str = 'cpu',
                 embedding_cache: str | Path | None = None, embed_in_isolation: bool = False,
                 gpu_id: int | None = None) -> None:
        """
        Parameters
        ----------
        saved_weights : str or Path
            Location of the saved network
        dtype : str, optional
            "sequence", "residues" or "matrix". Required for networks saved by
            earlier versions of PARROT (which are "sequence" or "residues"
            networks); for newer networks, if given, it must match the saved
            network.
        device : str, optional
            'cpu' (default), 'cuda', 'mps', or 'auto' (CUDA if available, else
            MPS if available, else the CPU). For networks trained on protein
            language model (PLM) embeddings, the PLM also runs here, so a GPU
            ('auto') is much faster.
        embedding_cache : str or Path, optional
            PLM and NARDINI networks only: a directory in which to store each
            sequence's encoding, so sequences predicted again are not encoded
            again. By default nothing is kept: predict() encodes each sequence
            when it is predicted, and predict_batch() encodes its sequences
            into a temporary cache, deleted once they are predicted.
        embed_in_isolation : bool, optional
            Networks trained with --embedding-context only: embed a sequence
            predicted without its full-length protein on its own, rather than
            raising an error. Its embeddings then differ from those the network
            was trained on, so its prediction may be less accurate.
        gpu_id : int, optional
            Index of the CUDA device to run on, as parrot-predict's --gpu-id.
            Giving one asks for CUDA, so `device` must then be 'cuda' or 'auto'.
            By default, CUDA runs on the current CUDA device.

        Raises
        ------
        ValueError
            If `dtype` is not a valid datatype or does not match the saved
            network, `device` is not a valid choice, `gpu_id` is given with
            `device` 'cpu' or 'mps' or is not a CUDA device on this machine,
            `embedding_cache` is given for a network that uses neither a PLM
            nor a NARDINI encoding, `embed_in_isolation` is given for a network
            trained without --embedding-context, or the file cannot be loaded
            as a PARROT network (see checkpoint.load_network)
        FileNotFoundError
            If `saved_weights` does not exist
        RuntimeError
            If the requested device is not available on this machine
        """

        if dtype is not None and dtype not in VALID_DATATYPES:
            raise ValueError(f"dtype must be one of {', '.join(repr(datatype) for datatype in VALID_DATATYPES)}, "
                             f'got {dtype!r}')

        self.device = devices.resolve_device(device, gpu_id)
        self.network, self.encoder = checkpoint.load_network(str(saved_weights), device=self.device,
                                                             legacy_datatype=dtype)
        network_task = self.network.task

        if isinstance(self.encoder, plm.PLMEncoder):
            self.encoder.use_device(self.device)
            if embedding_cache is not None:
                self.encoder.use_cache(embedding_cache)
        elif isinstance(self.encoder, nardini.NardiniEncoder):
            if embedding_cache is not None:
                self.encoder.use_cache(embedding_cache)
        elif embedding_cache is not None:
            raise ValueError('embedding_cache only applies to networks trained on protein language model embeddings '
                             'or NARDINI features')

        if dtype is not None and dtype != network_task.datatype:
            raise ValueError(f"dtype '{dtype}' does not match the saved network, which is a "
                             f"'{network_task.datatype}' network")

        self.in_context = isinstance(self.encoder, plm.PLMEncoder) and self.encoder.in_context
        if embed_in_isolation and not self.in_context:
            raise ValueError('embed_in_isolation only applies to networks trained with --embedding-context; this '
                             'network was trained on sequences each embedded on its own')
        self.embed_in_isolation = embed_in_isolation

        self.dtype = network_task.datatype
        self.n_classes = network_task.num_classes
        self.task = network_task.problem_type
        self.num_layers = self.network.hyperparameter_values.get('num_layers')
        self.hidden_vector_size = self.network.hyperparameter_values.get('hidden_size')

    def predict(self, seq: str, full_length: str | None = None, start: int | None = None) -> np.ndarray:
        """Use the network to predict values for a single sequence

        Parameters
        ----------
        seq : str
            Sequence (converted to uppercase before encoding)
        full_length : str, optional
            Networks trained with --embedding-context only (and required for
            them, unless the Predictor was created with embed_in_isolation=True):
            the full-length protein `seq` was cut from (converted to
            uppercase). `seq` is embedded within it, as the training sequences
            were.
        start : int, optional
            With `full_length`: the 1-based position of the first residue of
            `seq` in it, needed only if `seq` occurs more than once in it

        Returns
        -------
        np.ndarray
            Regression: a 1D array of one value (sequence networks) or one value
            per residue, as earlier versions of PARROT returned it (parrot-predict
            writes a sequence network's value as a single number instead), or
            the predicted matrix (matrix networks).
            Classification: the probability of each class, with shape
            [num_classes] (sequence networks), [len(seq) X num_classes] (residue
            networks) or [len(seq) X width X num_classes] (matrix networks).

        Raises
        ------
        ValueError
            If `full_length` is missing for a network trained with
            --embedding-context (without embed_in_isolation), or given for one
            trained without it; if `start` is given without `full_length`; or
            if `seq` cannot be placed in `full_length` (see
            embedding_context.locate_fragment())
        """

        seq = seq.upper()
        self._use_contexts({seq: self._context_of(seq, full_length, start)})
        output = prediction.predict_sequences(self.network, [seq], self.encoder, self.device)[seq]
        return self._prediction_from_output(output)

    def predict_batch(self, sequences: Sequence[str], full_lengths: Sequence[str | None] | None = None,
                      starts: Sequence[int | None] | None = None,
                      batch_size: int = prediction.DEFAULT_PREDICTION_BATCH,
                      show_progress: bool = False) -> list[np.ndarray]:
        """Use the network to predict values for many sequences at once

        This predicts the sequences together, as parrot-predict does, which for
        more than a handful of sequences is much faster than calling predict()
        on each. The predictions match predict()'s up to float32 rounding, not
        exactly: which sequences share a batch (and so the batch size) changes
        the order of the floating-point operations, giving differences of the
        order of 1e-7 relative.

        1. Networks trained on protein language model (PLM) embeddings or
           NARDINI features: every sequence is encoded first, the PLM running
           over many sequences at once (and over each full-length protein once,
           for a network trained with --embedding-context) and NARDINI features
           being computed on several CPU cores. The encodings go into the
           Predictor's embedding_cache or, without one, into a temporary cache
           (in the system's temporary directory), deleted once the sequences
           are predicted. Either way, they need about (number of residues) x
           (values per residue) x 2 bytes of disk.
        2. The network runs over batches of `batch_size` sequences of similar
           length. Each distinct sequence is predicted once.

        Parameters
        ----------
        sequences : list of str
            Sequences (each converted to uppercase before encoding)
        full_lengths : list of str or None, optional
            Networks trained with --embedding-context only (and required for
            them, unless the Predictor was created with embed_in_isolation=True):
            the full-length protein each sequence was cut from, one per
            sequence in the same order (converted to uppercase). Each sequence
            is embedded within its protein, as in predict(). An entry of None
            embeds that sequence on its own, which needs embed_in_isolation=True.
        starts : list of int or None, optional
            With `full_lengths`: the 1-based position of each sequence's first
            residue in its full-length protein, needed only for a sequence that
            occurs more than once in it (None for the others)
        batch_size : int, optional
            Number of sequences run through the network at once (default 64).
            It sets speed and memory use (a large matrix network may need a
            smaller batch), and changes the predictions only by float32
            rounding.
        show_progress : bool, optional
            Print what is encoded, with a progress bar, and how many sequences
            have been predicted (default False: print nothing)

        Returns
        -------
        list of np.ndarray
            The prediction for each sequence, in the order given (so a sequence
            given twice gets its prediction twice), each in the form predict()
            returns

        Raises
        ------
        TypeError
            If `sequences`, `full_lengths` or `starts` is a single string rather than a list
        ValueError
            If `batch_size` is less than 1; `full_lengths` or `starts` does not have one entry per sequence; a
            sequence has a residue the network's encoding cannot read; a sequence cannot be embedded as predict()
            would embed it (the message gives its position in `sequences`); or a sequence given more than once is
            given different full-length contexts (its embedding is computed once, so it needs one context)
        RuntimeError
            If the disk does not have room for the encodings (see PLMEncoder.prepare())
        """

        if isinstance(sequences, str):
            raise TypeError('predict_batch() takes a list of sequences; predict() takes a single sequence')
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, got {batch_size}')

        uppercase_sequences = [seq.upper() for seq in sequences]
        if not uppercase_sequences:
            return []

        # Every sequence is checked and placed before anything is encoded or predicted, so a mistake in any of them
        # stops the batch at once
        self.encoder.check_sequences(uppercase_sequences)
        self._use_contexts(self._contexts_of_batch(uppercase_sequences, full_lengths, starts))

        outputs = self._encode_and_predict(uppercase_sequences, batch_size, show_progress)
        return [self._prediction_from_output(outputs[seq]) for seq in uppercase_sequences]

    def _encode_and_predict(self, sequences: list[str], batch_size: int,
                            show_progress: bool) -> dict[str, np.ndarray]:
        """Encode the sequences ahead of time if their encoding is slow to compute (PLM or NARDINI), then predict them

        Parameters
        ----------
        sequences : list of str
            The (uppercase) sequences to predict
        batch_size : int
            Number of sequences run through the network at once
        show_progress : bool
            If True, print what is encoded and how many sequences have been predicted

        Returns
        -------
        dict
            Each distinct sequence -> its raw network output (see prediction.predict_sequences())
        """

        encoder = self.encoder
        if not isinstance(encoder, (plm.PLMEncoder, nardini.NardiniEncoder)):
            return self._run_network(sequences, batch_size, show_progress)

        if encoder.cache is not None:
            encoder.prepare(sequences, PROGRAM_NAME, silent=not show_progress)
            return self._run_network(sequences, batch_size, show_progress)

        # Without an embedding cache, the encodings go into a temporary one: computed in bulk, as parrot-predict
        # computes them, and never all held in memory at once, however many sequences there are
        with tempfile.TemporaryDirectory(prefix=TEMPORARY_CACHE_PREFIX) as directory:
            encoder.use_cache(directory)
            try:
                encoder.prepare(sequences, PROGRAM_NAME, silent=not show_progress)
                return self._run_network(sequences, batch_size, show_progress)
            finally:
                encoder.use_cache(None)

    def _run_network(self, sequences: list[str], batch_size: int, show_progress: bool) -> dict[str, np.ndarray]:
        """Run the network over the sequences in batches (see prediction.predict_sequences())

        Parameters
        ----------
        sequences : list of str
            The (uppercase) sequences to predict
        batch_size : int
            Number of sequences run through the network at once
        show_progress : bool
            If True, print how many sequences have been predicted every DEFAULT_PRINT_FREQUENCY sequences

        Returns
        -------
        dict
            Each distinct sequence -> its raw network output
        """
        print_frequency = prediction.DEFAULT_PRINT_FREQUENCY if show_progress else None
        return prediction.predict_sequences(self.network, sequences, self.encoder, self.device,
                                            batch_size=batch_size, print_frequency=print_frequency)

    def _prediction_from_output(self, output: np.ndarray) -> np.ndarray:
        """The prediction predict() returns, from a sequence's raw network output

        Parameters
        ----------
        output : np.ndarray
            The raw network output for one sequence (see prediction.predict_sequences())

        Returns
        -------
        np.ndarray
            See predict()
        """
        if self.n_classes > 1:
            return prediction.output_to_prediction(output, self.network.task, probabilistic=True)
        if self.network.task.is_matrix:
            return prediction.output_to_prediction(output, self.network.task)

        return np.asarray(output).flatten()

    def _contexts_of_batch(self, sequences: list[str], full_lengths: Sequence[str | None] | None,
                           starts: Sequence[int | None] | None) -> dict[str, plm.FragmentContext | None]:
        """Where each sequence of a batch is embedded: within its full-length protein, or on its own

        Parameters
        ----------
        sequences : list of str
            The (uppercase) sequences to predict
        full_lengths : list of str or None, or None
            The full-length protein of each sequence (see predict_batch())
        starts : list of int or None, or None
            The 1-based start of each sequence in its full-length protein

        Returns
        -------
        dict
            Each distinct sequence -> its FragmentContext, or None to be embedded on its own

        Raises
        ------
        TypeError
            If `full_lengths` or `starts` is a single string
        ValueError
            If `full_lengths` or `starts` does not have one entry per sequence, a sequence cannot be placed (see
            _context_of(); the message gives its position), or a sequence given more than once is given different
            contexts
        """
        self._check_batch_contexts(full_lengths, starts)
        full_length_of_each = one_per_sequence(full_lengths, 'full_lengths', len(sequences))
        start_of_each = one_per_sequence(starts, 'starts', len(sequences))

        contexts: dict[str, plm.FragmentContext | None] = {}
        first_index: dict[str, int] = {}
        for index, (seq, full_length, start) in enumerate(zip(sequences, full_length_of_each, start_of_each,
                                                              strict=True)):
            try:
                context = self._context_of(seq, full_length, start)
            except ValueError as problem:
                raise ValueError(f'sequences[{index}]: {problem}') from None

            if seq not in contexts:
                contexts[seq] = context
                first_index[seq] = index
            elif contexts[seq] != context:
                raise ValueError(f'sequences[{index}] is the same sequence as sequences[{first_index[seq]}] '
                                 f'({embedding_context.shorten(seq)}), but is given a different full-length context. '
                                 'A sequence is embedded once, so it needs the same context everywhere it appears.')
        return contexts

    def _check_batch_contexts(self, full_lengths: Sequence[str | None] | None,
                              starts: Sequence[int | None] | None) -> None:
        """Check predict_batch() was given full-length proteins if, and only if, the network needs them

        Parameters
        ----------
        full_lengths : list of str or None, or None
            The full-length protein of each sequence (see predict_batch())
        starts : list of int or None, or None
            The 1-based start of each sequence in its full-length protein

        Raises
        ------
        ValueError
            If `full_lengths` is missing for a network trained with --embedding-context (without embed_in_isolation),
            or given for one trained without it, or `starts` is given without `full_lengths`
        """
        if starts is not None and full_lengths is None:
            raise ValueError('starts only applies with full_lengths: each is where a sequence starts in its protein')
        if full_lengths is not None and not self.in_context:
            raise ValueError('full_lengths only applies to networks trained with --embedding-context; this network '
                             'was trained on sequences each embedded on its own')
        if full_lengths is None and self.in_context and not self.embed_in_isolation:
            raise ValueError('This network was trained on sequences embedded within their full-length proteins '
                             '(--embedding-context): give the full-length protein each sequence was cut from '
                             '(full_lengths=[...]), or create the Predictor with embed_in_isolation=True to embed them '
                             'on their own (embeddings unlike those the network was trained on, so predictions may '
                             'be less accurate)')

    def _context_of(self, seq: str, full_length: str | None, start: int | None) -> plm.FragmentContext | None:
        """Where `seq` is embedded: within its full-length protein (networks trained with --embedding-context), or
        (None) on its own

        Parameters
        ----------
        seq : str
            The (uppercase) sequence to predict
        full_length : str or None
            The full-length protein it was cut from
        start : int or None
            The 1-based position of its first residue in `full_length`

        Returns
        -------
        FragmentContext or None
            Where in `full_length` the sequence is, or None if it is embedded on its own

        Raises
        ------
        ValueError
            See predict()
        """
        if full_length is None:
            if start is not None:
                raise ValueError('start only applies with full_length: it is where the sequence starts in that protein')
            if self.in_context and not self.embed_in_isolation:
                raise ValueError('This network was trained on sequences embedded within their full-length proteins '
                                 '(--embedding-context): give the full-length protein the sequence was cut from '
                                 '(full_length=...), or create the Predictor with embed_in_isolation=True to embed it '
                                 'on its own (embeddings unlike those the network was trained on, so predictions may '
                                 'be less accurate)')
            return None

        if not self.in_context:
            raise ValueError('full_length only applies to networks trained with --embedding-context; this network '
                             'was trained on sequences each embedded on its own')
        full_length = full_length.upper()
        try:
            position = embedding_context.locate_fragment(seq, full_length, start)
        except embedding_context.ContextProblem as problem:
            raise ValueError(f'The sequence cannot be placed in full_length: {problem}') from None
        return plm.FragmentContext(full_length, position)

    def _use_contexts(self, contexts: Mapping[str, plm.FragmentContext | None]) -> None:
        """Tell the encoder where each sequence is embedded (networks trained with --embedding-context only)

        Parameters
        ----------
        contexts : mapping
            Sequence -> its FragmentContext, or None to be embedded on its own
        """
        if not self.in_context:
            return
        assert isinstance(self.encoder, plm.PLMEncoder)
        self.encoder.add_contexts(contexts)


def one_per_sequence(values: Sequence[Value | None] | None, name: str, num_sequences: int) -> list[Value | None]:
    """One value per sequence of a batch: the list given, checked, or all None if none was given

    Parameters
    ----------
    values : list or None
        The value of each sequence (e.g. its full-length protein), or None
    name : str
        The argument's name, for error messages
    num_sequences : int
        The number of sequences in the batch

    Returns
    -------
    list
        `values` as a list, or `num_sequences` Nones

    Raises
    ------
    TypeError
        If `values` is a single string (whose characters would otherwise be taken as one value per sequence)
    ValueError
        If `values` does not have one entry per sequence
    """
    if values is None:
        return [None] * num_sequences
    if isinstance(values, str):
        raise TypeError(f'{name} takes one value per sequence (a list), not a single string')
    if len(values) != num_sequences:
        raise ValueError(f'{name} has {len(values):,} entries but there are {num_sequences:,} sequences: give one '
                         'per sequence, in the same order (None for a sequence without one)')
    return list(values)
