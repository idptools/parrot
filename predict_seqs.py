#!/usr/bin/env python

import numpy as np
import torch
import argparse
import brnn_architecture
import process_input_data as pid
import train_network
import os

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Make predictions with a bi-directional RNN.')
parser.add_argument('seq_file', help='path to tsv file with format: <idx> <sequence> <data>')
parser.add_argument('saved_network', help='path to the trained network weights file')
parser.add_argument('output_file', help='file with sequences and their predicted values')
parser.add_argument('--datatype', metavar='dtype', default='sequence', type=str, required=True,
					help="Required. Format of the input data file, must be 'sequence' or 'residues'")
parser.add_argument('-nc', default=1, type=int, metavar='num_classes', required=True,
					help='Required. Number of output classes, for regression put 1')
parser.add_argument('-hs', default=5, type=int, metavar='hidden_size', 
						help='hidden vector size (def=5)')
parser.add_argument('-nl', default=1, type=int, metavar='num_layers', 
						help='number of layers per direction (def=1)')

args = parser.parse_args()
device = 'cpu'


# Hyper-parameters
hidden_size = args.hs
num_layers = args.nl
dtype = args.datatype
num_classes = args.nc

input_size = 20		# TODO: set to len(encoding_scheme)

###############################################################################
########################      Validate arguments:      ########################
# Ensure that provided data_file exists
seq_file = os.path.abspath(args.seq_file)
if not os.path.isfile(seq_file):
	print('Error: Sequence file does not exist.')
	sys.exit()

# Ensure that the saved weights file exists
saved_weights = os.path.abspath(args.saved_network)
if not os.path.isfile(saved_weights):
	print('Error: Saved weights file does not exist.')
	sys.exit()

# Ensure that output file location is valid
output_file = os.path.abspath(args.output_file)
output_filename = output_file.split('/')[-1]
if not os.path.exists(output_file[:-len(output_filename)]):
	print('Error: Output network directory does not exist.')
	sys.exit()

# Initialize network as classifier or regressor
if num_classes > 1:
	problem_type = 'classification'
elif num_classes == 1:
	problem_type = 'regression'
else:
	print('Error: number of classes must be a positive integer.')
	sys.exit()



# Ensure that hidden size and num layers are both positive ints
if hidden_size < 1:
	print('Error: hidden vector size must be a positive integer.')
	sys.exit()
if num_layers < 1:
	print('Error: number of layers must be a positive integer.')
	sys.exit()

###############################################################################


# Main:
# Initialize network architecture depending on data format
if dtype == 'sequence':
	# Use a many-to-one architecture
	brnn_network = brnn_architecture.BRNN_MtO(input_size, hidden_size, 
									num_layers, num_classes).to(device)
elif dtype == 'residues':
	# Use a many-to-many architecture
	brnn_network = brnn_architecture.BRNN_MtM(input_size, hidden_size, 
									num_layers, num_classes).to(device)
else:
	print("Error: Invalid datatype argument -- must be 'residues' or 'sequence'.")
	sys.exit()

brnn_network.load_state_dict(torch.load(saved_weights, 
				map_location=torch.device(device)))

# Convert sequence file to list of sequences:
with open(seq_file) as f:
	sequences = [line.strip() for line in f]

pred_dict = train_network.test_unlabeled_data(brnn_network, sequences)

if problem_type == 'classification':
	if dtype == 'sequence':
		for seq, values in pred_dict.items():
			pred_dict[seq] = [np.argmax(values[0])]
	else:
		for seq, values in pred_dict.items():
			int_vals = []
			for row in values[0]:
				int_vals.append(np.argmax(row))
			pred_dict[seq] = int_vals
else:
	if dtype == 'sequence':
		for seq, values in pred_dict.items():
			pred_dict[seq] = list(values[0])
	else:
		for seq, values in pred_dict.items():
			pred_dict[seq] = list(values[0].flatten())


with open(output_file, 'w') as f:
	for seq, values in pred_dict.items():
		out_str = seq + ' ' + ' '.join(map(str, values)) + '\n'
		f.write(out_str)


