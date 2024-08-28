# like process input data but with a 2 at the end. 

# like process_input_data by with a 2 at the end. 
# made to be easily nuked. 

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import mmap
import gc

from parrot import encode_sequence


class SequenceDataset(Dataset):
    def __init__(self, filepath, encoding_scheme='onehot', encoder=None, , dynamic_loading=True):
        self.filepath = filepath
        self.encoding_scheme = encoding_scheme
        self.encoder = encoder
        self.dynamic_loading = dynamic_loading

        # Compute offsets for each line if we plan to dynamically load stuff. 
        if self.dynamic_loading:
            self.offsets = self.compute_offsets(filepath)
            self.length=len(self.offsets)
            # Memory-map the file
            with open(self.filepath, 'r+b') as f:
                self.mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)                

        else:
            with open(self.filepath, 'r') as f:
                self.lines = f.readlines()
                self.length=len(self.lines)

    # for computing offsets
    def compute_offsets(self, filepath):
        offsets = []
        with open(filepath, 'r') as file:
            offset = 0
            for line in file:
                offsets.append(offset)
                offset += len(line.encode('utf-8'))  # Store the byte offset
        return offsets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dynamic_loading:
            # Seek to the line's offset
            self.mmapped_file.seek(self.offsets[idx])
            line = self.mmapped_file.readline().decode('utf-8')
        else:
            line = self.lines[idx]

        # Split the line into components
        seqID, sequence, values = line.strip().split('\t')
        values = np.array([float(value) for value in values.split()], dtype=np.float32)

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
        if self.dynamic_loading:
            # Close the memory-mapped file
            if hasattr(self, 'mmapped_file') and self.mmapped_file:
                self.mmapped_file.close()
                del self.mmapped_file

        else:
            if hasattr(self, 'lines') and self.lines:
                del self.lines

        # Trigger garbage collection
        gc.collect()    


def seq_regress_collate(batch):
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    
    # Determine the maximum length for target values in the batch
    max_target_len = max(item[2].shape[0] for item in batch)
    
    # Pad targets to have the same length
    targets = torch.zeros((len(batch), max_target_len), dtype=torch.float32)
    for i, item in enumerate(batch):
        target_len = item[2].shape[0]
        targets[i, :target_len] = torch.tensor(item[2], dtype=torch.float32)

    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq.clone().detach()

    return names, padded_seqs, targets.unsqueeze(-1)


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

    with open(filepath) as f:
        lines = f.readlines()

    training_samples = np.fromstring(lines[0], dtype=int, sep=" ")
    val_samples = np.fromstring(lines[1], dtype=int, sep=" ")
    test_samples = np.fromstring(lines[2], dtype=int, sep=" ")

    return training_samples, val_samples, test_samples   

def create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size=32, distributed=False, num_workers=0):
    if distributed==False:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    else:
        train_sampler = torch.utils.data.DistributedSampler(train_indices, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_indices, shuffle=False)
        test_sampler = torch.utils.data.DistributedSampler(test_indices, shuffle=False)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=seq_regress_collate,num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=seq_regress_collate,num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, collate_fn=seq_regress_collate,num_workers=num_workers)

    return train_loader, val_loader, test_loader










