#!/usr/bin/env python

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import math
import encode_sequence

def parse_file(tsvfile, datatype, problem_type, num_classes, excludeSeqID=False):
	'''
	Parse a whitespace-delimited input datafile. "datatype" should equal either
	'residues' or 'sequence' depending on the format of the input data.
	"problem_type" should be either 'regression' or 'classification'.
	'''
	with open(tsvfile) as f:
		lines = [line.rstrip().split() for line in f]

		# Add a dummy seqID if none are provided
		if excludeSeqID:
			for line in lines:
				line.insert(0, '')

		if datatype == 'residues':	# A value for each residue in a sequence
			data = [[x[0], x[1], np.array( x[2:], dtype=np.float)] for x in lines]
		elif datatype == 'sequence':	# A single value per sequence
			data = [[x[0], x[1], float(x[2])] for x in lines]
		else:
			print('Invalid datatype.')

	if problem_type == 'classification':
		if datatype == 'sequence':
			for sample in data:
				sample[2] = int(sample[2])

		else:
			for sample in data:
				sample[2] = list(map(int, sample[2]))
	return data

# Extended PyTorch Dataset class
class SequenceDataset(Dataset):
	def __init__(self, data, subset=np.array([])):
		if len(subset) == 0:
			self.data = data
		else:
			all_data = data
			self.data = [all_data[x] for x in subset]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sequence_vector = encode_sequence.one_hot(self.data[idx][1])
		value = self.data[idx][2]

		sample = (sequence_vector, value)
		return sample

# Collate sequence samples into a batch
def seq_class_collate(batch):
	orig_seq_vectors = [item[0] for item in batch]
	orig_targets = [item[1] for item in batch]

	longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

	padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])

	for i,j in enumerate(orig_seq_vectors):
		padded_seq_vectors[i][0:len(j)] = j

	padded_seq_vectors = torch.IntTensor(padded_seq_vectors)
	targets = torch.LongTensor(orig_targets)

	return (padded_seq_vectors, targets)

# Collate sequence samples into a batch
def seq_regress_collate(batch):
	orig_seq_vectors = [item[0] for item in batch]
	orig_targets = [[item[1]]for item in batch]

	longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

	padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])

	for i,j in enumerate(orig_seq_vectors):
		padded_seq_vectors[i][0:len(j)] = j

	padded_seq_vectors = torch.IntTensor(padded_seq_vectors)
	targets = torch.FloatTensor(orig_targets)

	return (padded_seq_vectors, targets)

# Collate residue samples into a batch, zero-padding end of sequences to the same size
# TODO: keep an eye on class-0 overprediction -- in order to zero-pad variable
# length sequences, empty "pad" residues must be assigned a class (here '0')
def res_class_collate(batch): # TODO: test that this works
    orig_seq_vectors = [item[0] for item in batch]
    orig_targets = [item[1] for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])
    padded_targets = np.zeros([len(orig_targets), longest_seq])

    for i,j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    for i,j in enumerate(orig_targets):
        padded_targets[i][0:len(j)] = j

    padded_seq_vectors = torch.IntTensor(padded_seq_vectors)
    padded_targets = torch.LongTensor(padded_targets)
    return (padded_seq_vectors, padded_targets)

# Collate residue samples into a batch, zero-padding end of sequences to the same size
def res_regress_collate(batch):
    orig_seq_vectors = [item[0] for item in batch]
    orig_targets = [item[1] for item in batch]

    longest_seq = len(max(orig_seq_vectors, key=lambda x: len(x)))

    padded_seq_vectors = np.zeros([len(orig_seq_vectors), longest_seq, len(orig_seq_vectors[0][0])])
    padded_targets = np.zeros([len(orig_targets), longest_seq])

    for i,j in enumerate(orig_seq_vectors):
        padded_seq_vectors[i][0:len(j)] = j

    for i,j in enumerate(orig_targets):
        padded_targets[i][0:len(j)] = j

    padded_seq_vectors = torch.IntTensor(padded_seq_vectors)
    padded_targets = torch.FloatTensor(padded_targets).view((len(padded_targets), len(padded_targets[0]), 1))

    return (padded_seq_vectors, padded_targets)


def vector_split(v, fraction):
	# Split v into two vectors
	segment1 = np.random.choice(v, size=math.ceil(fraction*len(v)), replace=False)
	segment1.sort()
	segment2 = np.setdiff1d(v, segment1, assume_unique=True)
	# return fraction, 1-fraction
	return segment1, segment2

def read_split_file(split_file):
	'''
	The split_file, if provided by the user, specifies which of the data samples are
	to be used as training, validation, and testing data.

	Assume the split_file is in the following format (without the SOF and EOF):

	SOF
	<train_sample1> <train_sample2> <train_sample3> <train_sample4> ...
	<val_sample1> <val_sample2> <val_sample3> <val_sample4>  ...
	<test_sample1> <test_sample2> <test_sample3> <test_sample4> ...
	EOF

	Each sample is specified by the line number in the corresponding data file (e.g. 15)
	'''
	with open(split_file) as f:
		lines = [line.rstrip().split() for line in f]
		training_samples = np.array([int(i) for i in lines[0]])
		val_samples = np.array([int(i) for i in lines[1]])
		test_samples = np.array([int(i) for i in lines[2]])
	return training_samples, val_samples, test_samples

def split_data(data_file, datatype, problem_type, num_classes, excludeSeqID=False, 
						split_file=None, percent_val=0.15, percent_test=0.15):
	data = parse_file(data_file, datatype, problem_type, num_classes, excludeSeqID=excludeSeqID)
	num_samples = len(data)

	if split_file == None:
		percent_train = 1 - percent_val - percent_test

		all_samples = np.arange(num_samples)
		training_samples, val_test_samples = vector_split(all_samples, percent_train)

		# Repeat procedure to split val and test sets
		val_test_fraction = percent_val / (percent_val + percent_test)
		val_samples, test_samples = vector_split(val_test_samples, val_test_fraction)

		# Generate datasets using these random partitions
		train_set = SequenceDataset(data=data, subset=training_samples)
		val_set = SequenceDataset(data=data, subset=val_samples)
		test_set = SequenceDataset(data=data, subset=test_samples)

	else:
		training_samples, val_samples, test_samples = read_split_file(split_file)

		# Generate datasets using the provided partitions
		train_set = SequenceDataset(data=data, subset=training_samples)
		val_set = SequenceDataset(data=data, subset=val_samples)
		test_set = SequenceDataset(data=data, subset=test_samples)

	return train_set, val_set, test_set

def split_data_cv(data_file, datatype, num_folds=5, split_file=None, 
								percent_val=0.15, percent_test=0.15):
	# TODO: write this
	# split as above function, but also split the combined train-val sets into k-folds

	return cv_sets, train_set, val_set, test_set


######### ----------------------------------------------------------- ##########
#------------------------------------------------------------------------------#
