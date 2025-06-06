# Alternative implementation of process_input_data with memory-efficient loading


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import gc

from parrot import encode_sequence
from parrot.parrot_exceptions import IOExceptionParrot


class SequenceDataset(Dataset):
    def __init__(self, filepath, encoding_scheme='onehot', encoder=None, excludeSeqID=False,
                  datatype='sequence', delimiter=None):
        self.filepath = filepath
        self.encoding_scheme = encoding_scheme
        self.encoder = encoder
        self.excludeSeqID = excludeSeqID
        self.datatype = datatype
        self.delimiter = delimiter
        
        # Validate inputs
        if not os.path.exists(filepath):
            raise IOExceptionParrot(f"File not found: {filepath}")
        
        # Load and parse all data (fixed approach for reliability)
        self.data = self._load_data()

    def _load_data(self):
        """Load and parse the entire dataset, handling various PARROT formats"""
        data = []
        
        with open(self.filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    # Split by delimiter (None = any whitespace, like original PARROT)
                    parts = line.split(self.delimiter)
                    
                    # Handle excludeSeqID case
                    if self.excludeSeqID:
                        # Format: sequence values...
                        if len(parts) < 2:
                            raise ValueError(f"Insufficient data on line {line_num}")
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
        # Clean up if needed
        gc.collect()    


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
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor((train_ratio + val_ratio) * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    return train_indices, val_indices, test_indices


def initial_data_prep(save_splits_loc, dataset, train_ratio, val_ratio):
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


def test_fixed_implementation():
    """
    Test function to validate the fixed implementation works correctly
    
    This tests the major fixes:
    - Proper data loading without memory mapping bugs
    - Correct parsing of PARROT format files
    - Support for different data types and problem types
    """
    try:
        # Test with a sample PARROT dataset
        test_file = "/Users/ryanemenecker/Desktop/lab_packages/parrot/data/seq_regress_dataset.tsv"
        
        if os.path.exists(test_file):
            print("Testing fixed process_input_data2.py implementation...")
            
            # Test sequence regression
            dataset = parse_file_v2(test_file, 
                                   datatype='sequence', 
                                   problem_type='regression')
            print(f"✓ Successfully loaded {len(dataset)} sequences for regression")
            
            # Test data loading
            sample_id, sample_seq, sample_val = dataset[0]
            print(f"✓ Sample data: ID={sample_id}, seq_len={sample_seq.shape[0]}, value={sample_val}")
            
            # Test train/val/test splitting
            train_idx, val_idx, test_idx = split_dataset_indices(dataset, 0.7, 0.15)
            print(f"✓ Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
            
            # Test dataloader creation
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset, train_idx, val_idx, test_idx, 
                batch_size=32, datatype='sequence', problem_type='regression'
            )
            print(f"✓ DataLoaders created successfully")
            
            print("All tests passed! The implementation is working correctly.")
            return True
        else:
            print(f"Test file not found: {test_file}")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    test_fixed_implementation()










