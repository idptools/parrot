# like process input data but with a 2 at the end. 

# like process_input_data by with a 2 at the end. 
# made to be easily nuked. 

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from parrot import encode_sequence

class SequenceDataset(Dataset):
    def __init__(self, filepath, encoding_scheme='onehot', encoder=None):
        self.filepath = filepath
        self.encoding_scheme = encoding_scheme
        self.encoder = encoder

    def __len__(self):
        with open(self.filepath, 'r') as file:
            return sum(1 for _ in file)

    def __getitem__(self, idx):
        with open(self.filepath, 'r') as file:
            for i, line in enumerate(file):
                if i == idx:
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

                    print(len(sequence_vector))
                    print(len(values))

                    return seqID, sequence_vector, values



def seq_regress_collate(batch):

    names = [item[0] for item in batch]
    seq_vectors = [torch.tensor(item[1], dtype=torch.float32) for item in batch]
    targets = torch.tensor([item[2] for item in batch], dtype=torch.float32)

    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq

    return names, padded_seqs, targets

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



def create_dataloaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    train_indices, val_indices, test_indices = split_dataset_indices(dataset, train_ratio, val_ratio)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=seq_regress_collate)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=seq_regress_collate)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, collate_fn=seq_regress_collate)

    return train_loader, val_loader, test_loader



