'''
Python module for integrating a trained network directly into a Python workflow.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.

'''

from pathlib import Path

import numpy as np
from scipy.special import softmax  # noqa: F401  (kept importable as py_predictor.softmax)

from parrot import checkpoint, devices, nardini, plm, prediction


class Predictor():
    """Class for integrating a trained PARROT network into a Python workflow

    Usage:
    >>> from parrot import py_predictor
    >>> my_predictor = py_predictor.Predictor('/path/to/saved_network.pt')
    >>> value = my_predictor.predict(AA_sequence)

    Networks saved by this version of PARROT record their architecture, task and
    encoding, so only the path is needed. Networks saved by earlier versions
    (which must be one-hot BRNNs) also need `dtype`.

    Predictions run on the CPU unless another device is requested, e.g.
    Predictor(path, device='auto') to use a GPU (CUDA or Apple-silicon MPS) when
    one is available. The CPU is the default because a GPU only pays off for
    many sequences at once, and MPS cannot be used inside forked worker
    processes (e.g. multiprocessing on macOS with the 'fork' start method).

    Attributes
    ----------
    dtype : str
        Data format that the network was trained for: "sequence" or "residues".
    n_classes : int
        Number of data classes. If 1, the network is a regressor; if >1, a
        classifier with n_classes classes.
    task : str
        "classification" or "regression" (a name kept from earlier versions of
        PARROT; the network's full Task is network.task).
    num_layers : int or None
        Number of hidden layers, for BRNN networks (None for other architectures).
    hidden_vector_size : int or None
        Size of the hidden vectors, for BRNN networks (None for other architectures).
    network : ParrotArchitecture
        The loaded network.
    encoder : SequenceEncoder
        The encoder the network was trained with.
    device : torch.device
        The device predictions run on.
    """

    def __init__(self, saved_weights: str | Path, dtype: str | None = None, device: str = 'cpu',
                 embedding_cache: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        saved_weights : str or Path
            Location of the saved network
        dtype : str, optional
            "sequence" or "residues". Required for networks saved by earlier
            versions of PARROT; for newer networks, if given, it must match the
            saved network.
        device : str, optional
            'cpu' (default), 'cuda', 'mps', or 'auto' (CUDA if available, else
            MPS if available, else the CPU). For networks trained on protein
            language model (PLM) embeddings, the PLM also runs here, so a GPU
            ('auto') is much faster.
        embedding_cache : str or Path, optional
            PLM and NARDINI networks only: a directory in which to store each
            sequence's encoding, so sequences predicted again are not encoded
            again. By default nothing is stored and each sequence is encoded
            when predicted.
        """

        if dtype is not None and dtype not in ['sequence', 'residues']:
            raise ValueError("dtype must equal 'residues' or 'sequence'")

        self.device = devices.resolve_device(device)
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

        self.dtype = network_task.datatype
        self.n_classes = network_task.num_classes
        self.task = network_task.problem_type
        self.num_layers = self.network.hyperparameter_values.get('num_layers')
        self.hidden_vector_size = self.network.hyperparameter_values.get('hidden_size')

    def predict(self, seq: str) -> np.ndarray:
        """Use the network to predict values for a single sequence

        Parameters
        ----------
        seq : str
            Sequence (converted to uppercase before encoding)

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
        """

        seq = seq.upper()
        output = prediction.predict_sequences(self.network, [seq], self.encoder, self.device)[seq]

        if self.n_classes > 1:
            return prediction.output_to_prediction(output, self.network.task, probabilistic=True)
        if self.network.task.is_matrix:
            return prediction.output_to_prediction(output, self.network.task)

        return np.asarray(output).flatten()
