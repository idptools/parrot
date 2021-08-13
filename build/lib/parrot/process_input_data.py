"""
Module with functions for processing an input datafile into a PyTorch-compatible
format.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from parrot import encode_sequence
from parrot.tools import dataset_warnings


def parse_file(tsvfile, datatype, problem_type, num_classes, excludeSeqID=False, ignoreWarnings=False):
    """Parse a datafile containing sequences and values.

    Each line of of the input tsv file contains a sequence of amino acids, a value
    (or values) corresponding to that sequence, and an optional sequence ID. This 
    file will be parsed into a more convenient list of lists. 

    If excludeSeqID is False, then the format of each line in the file should be:
    <seqID> <sequence> <value(s)>

    If excludeSeqID is True, then the format of each line in the file should be:
    <sequence> <value(s)>

    `value(s)` will either be a single number if `datatype` is 'sequence' or a
    len(sequence) series of whitespace-separated numbers if it is 'residues'.

    If `problem_type` is 'regression', then each value can be any real number. But
    if it is 'classification' then each value should be an integer in the range
    [0-N] where N is the number of classes.

    Parameters
    ----------
    tsvfile : str
            Path to a whitespace-separated datafile
    datatype : str
            Description of the format of the values in `tsvfile`. Providing a string
            other than 'sequence' or 'residues' will produce unintended behavior.
    problem_type : str
            Description of the machine-learning task. Providing a string other than
            'regression' or 'classification' will produce unintended behavior.
    excludeSeqID : bool, optional
            Boolean indicating whether or not each line in `tsvfile` has a sequence ID
            (default is False)
    ignoreWarnings : bool, optional
            If False, assess the structure and balance of the provided dataset with 
            basic heuristics and display warnings for common issues.

    Returns
    -------
    list of lists
            A list representing the entire `tsvfile`. Each inner list corresponds to a
            single line in the file and has the format [seqID, sequence, values].
    """

    with open(tsvfile) as f:
        lines = [line.rstrip().split() for line in f]

    # Add a dummy seqID if none are provided
    if excludeSeqID:
        for line in lines:
            line.insert(0, '')

    try:
        if datatype == 'residues':		# A value for each residue in a sequence
            data = [[x[0], x[1], np.array(x[2:], dtype=np.float)] for x in lines]
        elif datatype == 'sequence':  # A single value per sequence
            data = [[x[0], x[1], float(x[2])] for x in lines]
        else:
            raise ValueError('Invalid datatype. Must be "residues" or "sequence".')
    except:
        raise Exception(f"""Input data is not correctly formatted for datatype '{datatype}'. 
                Make sure your datafile does not have empty lines at the end of the file.""")

    if not ignoreWarnings:
        # Check for identical sequences
        dataset_warnings.check_duplicate_sequences(data)

        # Check for class imbalance
        if problem_type == 'classification':
            dataset_warnings.check_class_imbalance(data)

        # Check for data distribution imbalance
        elif problem_type == 'regression':
            dataset_warnings.check_regression_imbalance(data)

    if problem_type == 'classification':
        if datatype == 'sequence':
            for sample in data:
                sample[2] = int(sample[2])

                # Validate that all of the class labels are valid
                if sample[2] >= num_classes or sample[2] < 0:
                    raise ValueError("Invalid class label: %s" % sample[0])

        else:
            for sample in data:
                sample[2] = list(map(int, sample[2]))
                test = np.array(sample[2])

                if np.any(test < 0) or np.any(test >= num_classes):
                    raise ValueError("Invalid class label: %s" % sample[0])
                elif len(test) != len(sample[1]):
                    raise Exception(
                        'Input not properly formatted. Number of values must be equal to sequence length: %s' % sample[0])
    return data


class SequenceDataset(Dataset):
    """A PyTorch-compatible dataset containing sequences and values

    Stores a collection of sequences as tensors along with their corresponding
    target values. This class is designed to be provided to PyTorch Dataloaders.


    Attributes
    ----------
    data : list of lists
            Each inner list represents a single sequence in the dataset and should
            have the format: [seqID, sequence, value(s)]
    encoding_scheme : str
            Description of how an amino acid sequence should be encoded as a numeric 
            vector. Providing a string other than 'onehot', 'biophysics', or 'user' 
            will produce unintended consequences.
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    """

    def __init__(self, data, subset=np.array([]), encoding_scheme='onehot',
                 encoder=None):
        """
        Parameters
        ----------
        data : list of lists
                Each inner list represents a single sequence in the dataset and should
                have the format: [seqID, sequence, value(s)]
        subset : numpy int array, optional
                Array containing the indices of `data` that are to be a part of this
                dataset. Providing an empty array causes all of `data` to be integrated
                into the dataset (default is empty array).
        encoding_scheme : str
                Description of how an amino acid sequence should be encoded as a numeric 
                vector. Providing a string other than 'onehot', 'biophysics', or 'user' 
                will produce unintended consequences.
        encoder : UserEncoder object, optional
                If encoding_scheme is 'user', encoder should be a UserEncoder object
                that can convert amino acid sequences to numeric vectors. If
                encoding_scheme is not 'user', use None.
        """

        self.encoding_scheme = encoding_scheme
        self.encoder = encoder

        if len(subset) == 0:
            self.data = data
        else:
            all_data = data
            self.data = [all_data[x] for x in subset]

    def __len__(self):
        """Get the number of sequences in this dataset."""

        return len(self.data)

    def __getitem__(self, idx):
        """Get the sequence and values of a specified index.

        Converts a string sequence to an encoded numeric vector.

        Parameters
        ----------
        idx : int, optional
                Index of the desired sequence

        Returns
        -------
        tuple
                a tuple of a name, sequence vector and its corresponding values
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.encoding_scheme == 'onehot':
            sequence_vector = encode_sequence.one_hot(self.data[idx][1])
        elif self.encoding_scheme == 'biophysics':
            sequence_vector = encode_sequence.biophysics(self.data[idx][1])
        elif self.encoding_scheme == 'user':
            sequence_vector = self.encoder.encode(self.data[idx][1])

        name = self.data[idx][0]
        value = self.data[idx][2]
        sample = (name, sequence_vector, value)

        return sample


def seq_class_collate(batch):
    """Collates sequences and their values into a batch

    Transforms a collection of tuples of sequence vectors and values into a single
    tuple by stacking along a newly-created batch dimension. This function is 
    specifically designed for classification problems with sequence-mapped data.

    Parameters
    ----------
    batch : list
            A list of tuples of the form (sequence_vector, target_value(s))

    Returns
    -------
    tuple
            a tuple with concatenated names, sequence_vectors and target_values
    """

    names = [item[0] for item in batch]
    orig_seq_vectors = [item[1] for item in batch]
    orig_targets = [item[2] for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])

    for i, j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    padded_seq_vectors = torch.FloatTensor(padded_seq_vectors)
    targets = torch.LongTensor(orig_targets)

    return (names, padded_seq_vectors, targets)


def seq_regress_collate(batch):
    """Collates sequences and their values into a batch

    Transforms a collection of tuples of sequence vectors and values into a single
    tuple by stacking along a newly-created batch dimension. This function is 
    specifically designed for regression problems with sequence-mapped data.

    Parameters
    ----------
    batch : list
            A list of tuples of the form (sequence_vector, target_value(s))

    Returns
    -------
    tuple
            a tuple with concatenated names, sequence_vectors and target_value
    """

    names = [item[0] for item in batch]
    orig_seq_vectors = [item[1] for item in batch]
    orig_targets = [[item[2]]for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])

    for i, j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    padded_seq_vectors = torch.FloatTensor(padded_seq_vectors)
    targets = torch.FloatTensor(orig_targets)

    return (names, padded_seq_vectors, targets)


def res_class_collate(batch):
    """Collates sequences and their values into a batch

    Transforms a collection of tuples of sequence vectors and values into a single
    tuple by stacking along a newly-created batch dimension. This function is 
    specifically designed for classification problems with residue-mapped data. To
    account for sequences with different lengths, all sequence vectors are zero-
    padded to the length of the longest sequence in the batch

    Parameters
    ----------
    batch : list
            A list of tuples of the form (sequence_vector, target_value(s))

    Returns
    -------
    tuple
            a tuple with concatenated names, sequence_vectors and target_values
    """

    names = [item[0] for item in batch]
    orig_seq_vectors = [item[1] for item in batch]
    orig_targets = [item[2] for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])
    padded_targets = np.zeros([len(orig_targets), longest_seq])

    for i, j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    for i, j in enumerate(orig_targets):
        padded_targets[i][0:len(j)] = j

    padded_seq_vectors = torch.FloatTensor(padded_seq_vectors)
    padded_targets = torch.LongTensor(padded_targets)

    return (names, padded_seq_vectors, padded_targets)


def res_regress_collate(batch):
    """Collates sequences and their values into a batch

    Transforms a collection of tuples of sequence vectors and values into a single
    tuple by stacking along a newly-created batch dimension. This function is 
    specifically designed for regression problems with residue-mapped data. To
    account for sequences with different lengths, all sequence vectors are zero-
    padded to the length of the longest sequence in the batch

    Parameters
    ----------
    batch : list
            A list of tuples of the form (sequence_vector, target_value(s))

    Returns
    -------
    tuple
            a tuple with concatenated names, sequence_vectors and target_values
    """

    names = [item[0] for item in batch]
    orig_seq_vectors = [item[1] for item in batch]
    orig_targets = [item[2] for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])
    padded_targets = np.zeros([len(orig_targets), longest_seq])

    for i, j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    for i, j in enumerate(orig_targets):
        padded_targets[i][0:len(j)] = j

    padded_seq_vectors = torch.FloatTensor(padded_seq_vectors)
    padded_targets = torch.FloatTensor(padded_targets).view(
        (len(padded_targets), len(padded_targets[0]), 1))

    return (names, padded_seq_vectors, padded_targets)


def vector_split(v, fraction):
    """Split a vector randomly by a specified proportion

    Randomly divide the values of a vector into two, non-overlapping smaller 
    vectors. The proportions of the two vectors will be `fraction` and 
    (1 - `fraction`).

    Parameters
    ----------
    v : numpy array
            The vector to divide
    fraction : float
            Size proportion for the returned vectors. Should be in the range [0-1].

    Returns
    -------
    numpy array
            a subset of `v` of length `fraction` * len(v) (rounding up)
    numpy array
            a subset of `v` of length (1-`fraction`) * len(v). 
    """

    segment1 = np.random.choice(v, size=math.ceil(fraction*len(v)), replace=False)
    segment1.sort()
    segment2 = np.setdiff1d(v, segment1, assume_unique=True)
    return segment1, segment2


def read_split_file(split_file):
    """Read in a split_file

    Parameters
    ----------
    split_file : str
            Path to a whitespace-separated splitfile

    Returns
    -------
    numpy int array
            an array of the indices for the training set samples
    numpy int array
            an array of the indices for the validation set samples
    numpy int array
            an array of the indices for the testing set samples
    """

    with open(split_file) as f:
        lines = [line.rstrip().split() for line in f]
        training_samples = np.array([int(i) for i in lines[0]])
        val_samples = np.array([int(i) for i in lines[1]])
        test_samples = np.array([int(i) for i in lines[2]])
    return training_samples, val_samples, test_samples


def split_data(data_file, datatype, problem_type, num_classes, excludeSeqID=False, 
                split_file=None, encoding_scheme='onehot', encoder=None, 
                percent_val=0.15, percent_test=0.15, ignoreWarnings=False,
                save_splits_output=None):
    """Divide a datafile into training, validation, and test datasets

    Takes in a datafile and specification of the data format and the machine
    learning problem, and returns PyTorch-compatible Dataset objects for
    the training, validation and test sets of the data. The user may optionally
    specify how the dataset should be split into these subsets, as well as how
    protein sequences should be encoded as numeric vectors.

    Parameters
    ----------
    data_file : str
            Path to the datafile containing sequences and corresponding values
    datatype : str
            Format of the values within `data_file`. Should be 'sequence' if the
            `data_file` contains a single value per sequence, or 'residues' if
            it contains a value for each residue per sequence.
    problem_type : str
            The machine learning task to be addressed. Should be either 'regression'
            or 'classification'.
    excludeSeqID : bool, optional
            Flag that indicates how `data_file` is formatted. If False (default),
            then each line in the file should begin with a column containing a
            sequence ID. If True, then the datafile will not have this ID column,
            and will begin with the protein sequence.
    split_file : str, optional
            Path to a file containing information on how to divide the data into
            training, validation and test datasets. Default is None, which will
            cause the data to be divided randomly, with proportions based on
            `percent_val` and `percent_test`. If `split_file` is provided it must
            contain 3 lines in the file, corresponding to the training, validation
            and test sets. Each line should have whitespace-separated integer indices
            which correspond to lines in `data_file`.
    encoding_scheme : str, optional
            The method to be used for encoding protein sequences as numeric vectors.
            Currently 'onehot' and 'biophysics' are implemented (default is 'onehot').
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    percent_val : float, optional
            If `split_file` is not provided, the fraction of the data that should be
            randomly assigned to the validation set. Should be in the range [0-1]
            (default is 0.15).
    percent_test : float, optional
            If `split_file` is not provided, the fraction of the data that should be
            randomly assigned to the test set. Should be in the range [0-1] (default
            is 0.15). The proportion of the training set will be calculated by the
            difference between 1 and the sum of `percent_val` and `percent_train`, so
            these should not sum to be greater than 1.
    ignoreWarnings : bool, optional
            If False, assess the structure and balance of the provided dataset with 
            basic heuristics and display warnings for common issues.
    save_splits_output : str, optional
            Location where the train / val / test splits for this run should be saved

    Returns
    -------
    SequenceDataset object
            a dataset containing the training set sequences and values
    SequenceDataset object
            a dataset containing the validation set sequences and values
    SequenceDataset object
            a dataset containing the test set sequences and values
    """

    data = parse_file(data_file, datatype, problem_type, num_classes, 
                    excludeSeqID=excludeSeqID, ignoreWarnings=ignoreWarnings)
    num_samples = len(data)

    if split_file == None:
        percent_train = 1 - percent_val - percent_test

        all_samples = np.arange(num_samples)
        training_samples, val_test_samples = vector_split(all_samples, percent_train)

        # Repeat procedure to split val and test sets
        val_test_fraction = percent_val / (percent_val + percent_test)
        val_samples, test_samples = vector_split(val_test_samples, val_test_fraction)

        # Generate datasets using these random partitions
        train_set = SequenceDataset(data=data, subset=training_samples,
                                    encoding_scheme=encoding_scheme, encoder=encoder)
        val_set = SequenceDataset(data=data, subset=val_samples,
                                  encoding_scheme=encoding_scheme, encoder=encoder)
        test_set = SequenceDataset(data=data, subset=test_samples,
                                   encoding_scheme=encoding_scheme, encoder=encoder)

        if save_splits_output != None:
            # Save train/val/test splits
            with open(save_splits_output, 'w') as out:
                out.write(" ".join(np.sort(training_samples).astype('str')))
                out.write("\n")
                out.write(" ".join(np.sort(val_samples).astype('str')))
                out.write("\n")
                out.write(" ".join(np.sort(test_samples).astype('str')))
                out.write("\n")

    else:
        training_samples, val_samples, test_samples = read_split_file(split_file)

        # Generate datasets using the provided partitions
        train_set = SequenceDataset(data=data, subset=training_samples,
                                    encoding_scheme=encoding_scheme, encoder=encoder)
        val_set = SequenceDataset(data=data, subset=val_samples,
                                  encoding_scheme=encoding_scheme, encoder=encoder)
        test_set = SequenceDataset(data=data, subset=test_samples,
                                   encoding_scheme=encoding_scheme, encoder=encoder)

    return train_set, val_set, test_set


def split_data_cv(data_file, datatype, problem_type, num_classes, excludeSeqID=False,
                  split_file=None, encoding_scheme='onehot', encoder=None,
                  percent_val=0.15, percent_test=0.15, n_folds=5, ignoreWarnings=False,
                  save_splits_output=None):
    """Divide a datafile into training, val, test and 5 cross-val datasets.

    Takes in a datafile and specification of the data format and the machine
    learning problem, and returns PyTorch-compatible Dataset objects for
    the training, validation, test and cross-validation sets of the data. The
    user may optionally specify how the dataset should be split into these 
    subsets, as well as how protein sequences should be encoded as numeric 
    vectors.

    Parameters
    ----------
    data_file : str
            Path to the datafile containing sequences and corresponding values
    datatype : str
            Format of the values within `data_file`. Should be 'sequence' if the
            `data_file` contains a single value per sequence, or 'residues' if
            it contains a value for each residue per sequence.
    problem_type : str
            The machine learning task to be addressed. Should be either 'regression'
            or 'classification'.
    excludeSeqID : bool, optional
            Flag that indicates how `data_file` is formatted. If False (default),
            then each line in the file should begin with a column containing a
            sequence ID. If True, then the datafile will not have this ID column,
            and will begin with the protein sequence.
    split_file : str, optional
            Path to a file containing information on how to divide the data into
            training, validation and test datasets. Default is None, which will
            cause the data to be divided randomly, with proportions based on
            `percent_val` and `percent_test`. If `split_file` is provided it must
            contain 3 lines in the file, corresponding to the training, validation
            and test sets. Each line should have whitespace-separated integer indices
            which correspond to lines in `data_file`.
    encoding_scheme : str, optional
            The method to be used for encoding protein sequences as numeric vectors.
            Currently 'onehot' and 'biophysics' are implemented (default is 'onehot').
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    percent_val : float, optional
            If `split_file` is not provided, the fraction of the data that should be
            randomly assigned to the validation set. Should be in the range [0-1]
            (default is 0.15).
    percent_test : float, optional
            If `split_file` is not provided, the fraction of the data that should be
            randomly assigned to the test set. Should be in the range [0-1] (default
            is 0.15). The proportion of the training set will be calculated by the
            difference between 1 and the sum of `percent_val` and `percent_train`, so
            these should not sum to be greater than 1.
    n_folds : int, optional
            Number of folds for cross-validation (default is 5).
    ignoreWarnings : bool, optional
            If False, assess the structure and balance of the provided dataset with 
            basic heuristics and display warnings for common issues.
    save_splits_output : str, optional
            Location where the train / val / test splits for this run should be saved

    Returns
    -------
    list of tuples of SequenceDataset objects
            a list of tuples of length `n_folds`. Each tuple contains the training
            and validation datasets for one of the cross-val folds.
    SequenceDataset object
            a dataset containing the training set sequences and values
    SequenceDataset object
            a dataset containing the validation set sequences and values
    SequenceDataset object
            a dataset containing the test set sequences and values
    """

    data = parse_file(data_file, datatype, problem_type, num_classes, 
                    excludeSeqID=excludeSeqID, ignoreWarnings=ignoreWarnings)
    n_samples = len(data)

    # Initial step: split into training, val, and test sets
    if split_file == None:
        percent_train = 1 - percent_val - percent_test

        all_samples = np.arange(n_samples)
        training_samples, val_test_samples = vector_split(all_samples, percent_train)

        # Repeat procedure to split val and test sets
        val_test_fraction = percent_val / (percent_val + percent_test)
        val_samples, test_samples = vector_split(val_test_samples, val_test_fraction)

        # Generate datasets using these random partitions
        train_set = SequenceDataset(data=data, subset=training_samples,
                                    encoding_scheme=encoding_scheme, encoder=encoder)
        val_set = SequenceDataset(data=data, subset=val_samples,
                                  encoding_scheme=encoding_scheme, encoder=encoder)
        test_set = SequenceDataset(data=data, subset=test_samples,
                                   encoding_scheme=encoding_scheme, encoder=encoder)

        if save_splits_output != None:
            # Save train/val/test splits
            with open(save_splits_output, 'w') as out:
                out.write(" ".join(np.sort(training_samples).astype('str')))
                out.write("\n")
                out.write(" ".join(np.sort(val_samples).astype('str')))
                out.write("\n")
                out.write(" ".join(np.sort(test_samples).astype('str')))
                out.write("\n")

    # If provided, split datasets according to split_file
    else:
        training_samples, val_samples, test_samples = read_split_file(split_file)

        # Generate datasets using the provided partitions
        train_set = SequenceDataset(data=data, subset=training_samples,
                                    encoding_scheme=encoding_scheme, encoder=encoder)
        val_set = SequenceDataset(data=data, subset=val_samples,
                                  encoding_scheme=encoding_scheme, encoder=encoder)
        test_set = SequenceDataset(data=data, subset=test_samples,
                                   encoding_scheme=encoding_scheme, encoder=encoder)

    # Second step: combine train and val samples, and split evenly into n_folds
    cv_samples = np.append(training_samples, val_samples)
    np.random.shuffle(cv_samples)  # Shuffle train and val to avoid bias

    # Split into n_folds
    cv_samples = np.array_split(cv_samples, n_folds)

    # cv_sets will be a list of tuples: (fold_k_train_dataset, fold_k_test_dataset)
    cv_sets = []
    for i in range(len(cv_samples)):
        cv_test = cv_samples[i]
        cv_train = np.array([], dtype=int)
        for j in range(len(cv_samples)):
            if j != i:
                cv_train = np.append(cv_train, cv_samples[j])
        cv_train.sort()
        cv_test.sort()

        # Tuple of cross val train and test sets
        cv_sets.append((SequenceDataset(data=data, subset=cv_train,
                                        encoding_scheme=encoding_scheme, encoder=encoder),
                        SequenceDataset(data=data, subset=cv_test,
                                        encoding_scheme=encoding_scheme, encoder=encoder)))

    return cv_sets, train_set, val_set, test_set
