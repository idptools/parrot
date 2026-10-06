"""
Writing predicted matrices.

Matrices are too large to write one per line of text, as PARROT writes other
predictions, so they are written as a NumPy .npz archive with one array per
sequence, named by its sequence ID: the same format as the matrix files PARROT
reads. Load one with numpy.load(path)[sequence_id].

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import zipfile
from collections import Counter
from collections.abc import Mapping

import numpy as np

# The extension NumPy gives .npz archives (numpy.savez adds it to a name without it)
NPZ_EXTENSION = '.npz'

# Each array of a .npz archive is a .npy file inside it, named <name>.npy
NPY_EXTENSION = '.npy'


def npz_path(path: str) -> str:
    """The path a .npz archive is written to: `path`, with '.npz' added if it does not end with it.

    Parameters
    ----------
    path : str

    Returns
    -------
    str
    """
    return path if path.endswith(NPZ_EXTENSION) else path + NPZ_EXTENSION


def write_matrix_predictions(path: str, predictions: Mapping[str, np.ndarray]) -> str:
    """Write one predicted matrix per sequence to a compressed .npz archive

    Parameters
    ----------
    path : str
        File to write ('.npz' is added if it does not end with it)
    predictions : dict
        Sequence ID -> predicted matrix

    Returns
    -------
    str
        The path written
    """
    # This writes the archive as numpy.savez_compressed does, one .npy member per array. savez_compressed itself takes
    # the names as keyword arguments, so an ID such as 'file' or 'allow_pickle' would be taken for one of its options.
    path = npz_path(path)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, matrix in predictions.items():
            with archive.open(name + NPY_EXTENSION, 'w') as member:
                np.lib.format.write_array(member, np.asarray(matrix), allow_pickle=False)
    return path


def name_matrices(names: list[str], matrices: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Pair matrices with the names they will be saved under, which must be unique and not empty

    Parameters
    ----------
    names : list of str
        Sequence IDs, in the order of `matrices`
    matrices : list of numpy arrays

    Returns
    -------
    dict
        Name -> matrix

    Raises
    ------
    ValueError
        If a name is empty or appears more than once (each names an array of the archive)
    """
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated or '' in names:
        raise ValueError('Each predicted matrix is saved under its sequence ID, so the IDs must be unique and not '
                         f'empty{"; repeated: " + ", ".join(repeated[:5]) if repeated else ""}')
    return dict(zip(names, matrices))
