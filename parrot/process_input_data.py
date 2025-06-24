# Alternative implementation of process_input_data with memory-efficient loading

import math
import os
import gc

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from parrot import encode_sequence
from parrot.parrot_exceptions import IOExceptionParrot

# .................................................................
# From original implementation of PARROT
#
def read_tsv_raw(tsvfile, delimiter=None):
    """
    Efficiently parses a TSV file, ignoring empty lines and comment lines.
    Parameters
    ----------
    tsvfile : str
        Path to a whitespace-separated datafile.
    delimiter : str or None
        Delimiter for splitting columns. Default is tab.
    Returns
    -------
    generator
        Yields parsed lines as lists of strings.
    """
    # Note: This function parses the data in a memory efficient manner and
    # does not require the program to read the entire file at once (good for large files)

    # opens a file in the default read mode
    with open(tsvfile) as fh:
        # file objects are iterable over the lines in the file
        for line in fh:
            # remove whitespace from the front and end of the line
            stripped = line.strip()
            # check that the string is not empty and check the line does not start with a # - indicating a commment
            if stripped and not stripped.startswith("#"):
                # None defaults to any whitespace
                # pauses and waits for the caller to accept the parsed line value until continuing
                # This is memory efficient as it processes one line and then moves onto the next one
                yield stripped.split(delimiter)

# Note: The parser in this function will have issues if the data is not
# formatted perfectly. This should be made to be a little more forgiving.
# I would like to incorporate the ability to use .csv as well. This would
# require modification to this code.
def __parse_lines(lines, datatype, validate=True):
    """
    Internal function for parsing a set of lines

    Parameters
    ----------
    lines : list
        A list of lists, where the sublists reflect the columns in a tsvfile. Should be the output
        from the read_tsv_raw() function.

    datatype : str
        Identifier that defines the type of data being passed in. Must be either 'residues', 'sequence'

    validate : bool
        If set to true, ensures the number of residue values equals the number of residues
 
    Returns
    -----------
    list
        Returns a parsed list of lists, where each sublist contains the structure
        [id, sequence, <data>]
        where <data> is either a single float (mode=sequence) or a set of floats 

    Raises
    ---------
    Exception
        If an error occurs while parsing the file, the linenumber of the file is printed as well as the
        idenity of the offending line.

    """

    
    # check the datatype is valid
    if datatype not in ['residues','sequence']:
        raise ValueError('Invalid datatype. Must be "residues" or "sequence".')
        
    # parse the lines
    # the conversion from text to numbers could fail if the user provides improper data
    try:
        # here to store the data for each line
        data = [] # splits up data by column [col1,col2,...]
        lc = 0 # counts the lines (rows) in the dataset - only used for the error message

        # A value for each residue in a sequence
        # Note: this does not check that the number of targets matches the number of residues
        if datatype == 'residues':	
            for x in lines:
                # to the number of entries in the dataset
                lc = lc + 1
                
                # Pull the last portion of the dataset out and turns it into a numpy array
                # These should be the target values
                residue_data = np.array(x[2:], dtype=float)

                # Reformats the data as (ID, sequence, target value)
                data.append([x[0], x[1], residue_data])

        # A single value per sequence
        elif datatype == 'sequence':  
            for x in lines:
                # to the number of entries in the dataset
                lc = lc + 1
                # Reformats the data as (ID, sequence, target value)
                data.append([x[0], x[1], float(x[2])])

    # catch any exception and print it.
    except Exception as e:        
        print('Excecption raised on parsing input file...')
        print(e)
        print('')
        raise IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nMake sure your datafile does not have empty lines at the end of the file.\nError on line {lc}:\n{x}")

    # if we want to validate each line - aka check that the length of the sequence matches the number of target values
    if validate:
        # check that the datatype is residues - if not there is nothing to validate
        if datatype == 'residues':
            lc = 0
            for x in data:
                lc = lc + 1
                # check that the lengths match between sequence and targets
                if len(x[1]) != len(x[2]):
                    raise IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nInconsistent number of residue values and residues. Error on line {lc}:\n{x}")
    return data

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

    segment1 = np.random.choice(v, size=math.ceil(fraction * len(v)), replace=False)
    segment1.sort()
    segment2 = np.setdiff1d(v, segment1, assume_unique=True)
    return segment1, segment2

# .................................................................
# .................... New implementations below ..................
# .................................................................

class SequenceDataset(Dataset):
    def __init__(self, filepath : str, encoding_scheme : str = 'onehot', 
                 encoder= None, excludeSeqID : bool = False,
                  datatype : str = 'sequence', delimiter : str = None):
        """
        Initializes a SequenceDataset object. This is used by parrot to handle dataset parsing
        so that models can be easily trained.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset
        encoding_scheme : str
            Encoding scheme to use ('onehot', 'biophysics', 'user')
        encoder : object
            User encoder object (if encoding_scheme='user')
        exludeSeqID : bool
            Whether sequence IDs are excluded from the data file
        datatype : str
            'sequence' or 'residues'
        delimiter : str
            Delimiter for splitting lines (None = any whitespace)
        """
        # set the values of the properties based on the user input
        self.filepath = filepath
        self.encoding_scheme = encoding_scheme
        self.encoder = encoder

        # Validate inputs - check that the path exists
        if not os.path.exists(filepath):
            raise IOExceptionParrot(f"File not found: {filepath}")

        # TODO: Automatically infer the datatype
        self.excludeSeqID = excludeSeqID
        self.datatype = datatype
        self.delimiter = delimiter
        
        # Load and parse all data (fixed approach for reliability)
        self.data = self._load_data()

    def _load_data(self):
        """Load and parse the entire dataset, handling various PARROT formats.
        
        This function should be able to handle a variety of different frameworks.

        """
        # the empty list that will store the data
        data = []
        
        # open the file in read mode
        with open(self.filepath, 'r') as f:
            # loop over each line and get the line number
            for line_num, line in enumerate(f, 1):
                # remove whitespace from the front and end of the line
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # wrap in a try statement to catch errors
                try:
                    # Split by delimiter (None = any whitespace, like original PARROT)
                    parts = line.split(self.delimiter)
                    
                    # Handle excludeSeqID case
                    # Both paths need to end by having a seqID, sequence, values_str
                    # A seqID will be generated for data that does not have it already
                    if self.excludeSeqID:
                        # Format: sequence values...
                        if len(parts) < 2:
                            raise ValueError(f"Insufficient data on line {line_num}")
                        # generate a sequence ID for the sequences
                        seqID = f"seq_{line_num}"  # Generate ID
                        sequence = parts[0]
                        values_str = parts[1:]
                    else:
                        # Format: seqID sequence values...
                        if len(parts) < 3:
                            raise ValueError(f"Insufficient data on line {line_num}")
                        seqID = parts[0]
                        sequence = parts[1]
                        values_str = parts[2:]
                    
                    # Parse values based on datatype
                    if self.datatype == 'sequence':
                        # Single value per sequence
                        if len(values_str) != 1:
                            raise ValueError(f"Expected single value for sequence data on line {line_num}")
                        values = float(values_str[0])
                    elif self.datatype == 'residues':
                        # One value per residue
                        values = np.array([float(v) for v in values_str], dtype=np.float32)
                        if len(values) != len(sequence):
                            raise ValueError(f"Number of values ({len(values)}) doesn't match sequence length ({len(sequence)}) on line {line_num}")
                    else:
                        raise ValueError(f"Invalid datatype: {self.datatype}")
                    
                    data.append((seqID, sequence, values))
                    
                except Exception as e:
                    raise IOExceptionParrot(f"Error parsing line {line_num}: {line}\nError: {str(e)}")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seqID, sequence, values = self.data[idx]

        # Encode sequence
        if self.encoding_scheme == 'onehot':
            sequence_vector = encode_sequence.one_hot(sequence)
        elif self.encoding_scheme == 'biophysics':
            sequence_vector = encode_sequence.biophysics(sequence)
        elif self.encoding_scheme == 'user' and self.encoder:
            sequence_vector = self.encoder.encode(sequence)
        else:
            raise ValueError(f"Unknown encoding scheme: {self.encoding_scheme}")

        return seqID, sequence_vector, values

    def __del__(self):
        # Forces garbage collection if needed (aka - frees up memory by destroying this object)
        gc.collect()    



#----------------
# Collate function for the various modes
#----------------


def seq_regress_collate(batch):
    """Collate function for sequence regression"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    targets = [item[2] for item in batch]  # Single value per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq.clone().detach()

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    return names, padded_seqs, targets_tensor


def seq_class_collate(batch):
    """Collate function for sequence classification"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    targets = [item[2] for item in batch]  # Single class per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq.clone().detach()

    # Convert targets to tensor (integers for classification)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return names, padded_seqs, targets_tensor


def res_regress_collate(batch):
    """Collate function for residue regression"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    target_arrays = [item[2] for item in batch]  # Array of values per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensors
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)
    padded_targets = torch.zeros((len(seq_vectors), max_len), dtype=torch.float32)

    for i, (seq, targets) in enumerate(zip(seq_vectors, target_arrays)):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq.clone().detach()
        padded_targets[i, :seq_len] = torch.tensor(targets, dtype=torch.float32)

    return names, padded_seqs, padded_targets


def res_class_collate(batch):
    """Collate function for residue classification"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    target_arrays = [item[2] for item in batch]  # Array of class labels per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensors
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)
    padded_targets = torch.zeros((len(seq_vectors), max_len), dtype=torch.long)

    for i, (seq, targets) in enumerate(zip(seq_vectors, target_arrays)):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq.clone().detach()
        padded_targets[i, :seq_len] = torch.tensor(targets, dtype=torch.long)

    return names, padded_seqs, padded_targets


def split_dataset_indices(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into training, validation, and test sets based on the 
    specified ratio requested by the user.

    Returns
    -------
    tuple
        train_indices, val_indices, test_indices
    """
    # determine the length of the dataset (number of rows)
    dataset_size = len(dataset)

    # create a list of all the different row indexes (to uniquely identify each row)
    indices = list(range(dataset_size))

    # randomly shuffle the indexes so that we can just pull the first 70% of indexes to get out training set
    np.random.shuffle(indices)

    # determine the number of rows to put in the training and validation set
    # Note: we are aiming to find the last index for each set.
    # the test set is just the rest of the indexes
    train_split = int(np.floor(train_ratio * dataset_size))
    # we add the percentage to find the last row that correspond to having both train and validation indexes pulled
    # this makes life much easier as we can use the integers directly for our indexing below
    val_split = int(np.floor((train_ratio + val_ratio) * dataset_size))

    # pull out the unique indexes for each set from the shuffled set
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    return train_indices, val_indices, test_indices


def initial_data_prep(save_splits_loc, dataset, train_ratio, val_ratio):
    """
    This function preps the data and writes the splits of train, validation, and test to disk.

    Parameters
    ----------
    save_splits_loc : str
        this is the file location to save the splits that are used from the dataset
    dataset : Object
        This is a dataset that you will use to train the model
    train_ratio : float
        this is the ratio to use to train the model
    val_ratio : float
        this is the ratio to use to validate the model

    """
    # function that does initial data prep. Basically,
    # this function will take in a dataset, get the indices
    # then write them out to disk. 
    train_indices, val_indices, test_indices = split_dataset_indices(dataset, train_ratio, val_ratio)
    with open(save_splits_loc, "w") as f:
        f.write(" ".join(str(i) for i in train_indices))
        f.write("\n")
        f.write(" ".join(str(i) for i in val_indices))
        f.write("\n")
        f.write(" ".join(str(i) for i in test_indices))
    f.close()



def read_indices(filepath):
    """
    Read in the indices for train, val, test

    Parameters
    ----------
    filepath : str
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

    with open(filepath) as f:
        lines = f.readlines()

    # Use np.fromstring with explicit dtype and separator (handles deprecation)
    training_samples = np.fromstring(lines[0], dtype=int, sep=" ")
    val_samples = np.fromstring(lines[1], dtype=int, sep=" ")
    test_samples = np.fromstring(lines[2], dtype=int, sep=" ")

    return training_samples, val_samples, test_samples


def parse_file_v2(filepath, datatype='sequence', problem_type='regression', num_classes=1, 
                  excludeSeqID=False, encoding_scheme='onehot', encoder=None, delimiter=None):
    """
    Alternative implementation of parse_file with improved memory handling
    
    Returns a SequenceDataset object instead of raw parsed data
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    datatype : str
        'sequence' or 'residues'
    problem_type : str
        'regression' or 'classification'
    num_classes : int
        Number of classes (for classification)
    excludeSeqID : bool
        Whether sequence IDs are excluded from the file
    encoding_scheme : str
        Encoding scheme ('onehot', 'biophysics', 'user')
    encoder : object
        User encoder object (if encoding_scheme='user')
    delimiter : str
        Delimiter for splitting lines (None = any whitespace)
    """
    
    dataset = SequenceDataset(filepath=filepath, 
                             encoding_scheme=encoding_scheme,
                             encoder=encoder,
                             excludeSeqID=excludeSeqID,
                             datatype=datatype,
                             delimiter=delimiter)
    
    # Validate class labels if classification
    if problem_type == 'classification':
        for i, (seqID, _, values) in enumerate(dataset.data):
            if datatype == 'sequence':
                # Single class label
                if not isinstance(values, (int, float)) or values < 0 or values >= num_classes:
                    raise ValueError(f"Invalid class label {values} for sequence {seqID}. Must be 0 <= label < {num_classes}")
            else:  # residues
                # Array of class labels
                if np.any(values < 0) or np.any(values >= num_classes):
                    raise ValueError(f"Invalid class labels for sequence {seqID}. All labels must be 0 <= label < {num_classes}")
    
    return dataset   

def create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size=32, 
                      distributed=False, num_workers=0, datatype='sequence', problem_type='regression'):
    """
    Create DataLoaders with appropriate collate functions based on data type and problem type
    
    Parameters:
    -----------
    dataset : SequenceDataset
        The dataset to create loaders for
    train_indices, val_indices, test_indices : array-like
        Indices for each split
    batch_size : int
        Batch size for training/validation (test uses batch_size=1)
    distributed : bool
        Whether to use distributed sampling
    num_workers : int
        Number of worker processes for data loading
    datatype : str
        'sequence' or 'residues'
    problem_type : str
        'regression' or 'classification'
    """
    
    # Select appropriate collate function
    if datatype == 'sequence':
        if problem_type == 'regression':
            collate_fn = seq_regress_collate
        else:  # classification
            collate_fn = seq_class_collate
    else:  # residues
        if problem_type == 'regression':
            collate_fn = res_regress_collate
        else:  # classification
            collate_fn = res_class_collate
    
    # Create samplers
    if distributed == False:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_indices, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_indices, shuffle=False)
        test_sampler = torch.utils.data.DistributedSampler(test_indices, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, 
                             collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, 
                           collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, 
                            collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader

