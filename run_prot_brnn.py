#!/usr/bin/env python

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os
import sys
import argparse
import process_input_data as pid
import brnn_architecture
import train_network
import brnn_plot

"""
Driver script for the prot-brnn module.
"""

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Train and test a bi-directional RNN using entire sequence.')
parser.add_argument('data_file', help='path to tsv file with format: <idx> <sequence> <data>')
parser.add_argument('output_network', help='path for the returned trained network')
parser.add_argument('--datatype', default='residues', type=str, 
					help="format of the input data file: 'sequence' or 'residues'")
parser.add_argument('--split', default='', metavar='split_file', 
			type=str, help="file indicating how to split datafile into training, validation, and testing sets")
parser.add_argument('--stop', default='auto', metavar='stop_condition', 
						type=str, help="training stop condition: either 'auto' (default) or 'iter'")
parser.add_argument('-nc', default=1, type=int, metavar='num_classes', 
					help='number of output classes: for regression, put 1')
parser.add_argument('-hs', default=5, type=int, metavar='hidden_size', 
						help='hidden vector size (def=5)')
parser.add_argument('-nl', default=1, type=int, metavar='num_layers', 
						help='number of layers per direction (def=1)')
parser.add_argument('-b', default=20, type=int, metavar='batch_size', help='(def=20)')
parser.add_argument('-lr', default=0.001, type=float, metavar='learning_rate', help='(def=0.001)')
parser.add_argument('-e', default=25, type=int, metavar='num_epochs', 
						help='number of training epochs (def=5)')

args = parser.parse_args()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = args.hs
num_layers = args.nl
batch_size = args.b
learning_rate = args.lr
num_epochs = args.e
dtype = args.datatype
num_classes = args.nc
split_file = args.split
stop_cond = args.stop

input_size = 20		# TODO: set to len(encoding_scheme)

###############################################################

# TODO: validate input files and validate all arguments
data_file = os.path.abspath(args.data_file)
saved_weights = os.path.abspath(args.output_network)

if split_file != '':
	split_file = os.path.abspath(split_file)
else:
	split_file=None

# Initialize network parameters and architecture
if num_classes > 1:
	problem_type = 'classification'
else:
	problem_type = 'regression'

if dtype == 'sequence':
	brnn_network = brnn_architecture.BRNN_MtO(input_size, hidden_size, 
									num_layers, num_classes).to(device)
	# Set collate function
	if problem_type == 'classification':
		collate_function = pid.seq_class_collate
	else:
		collate_function = pid.seq_regress_collate

elif dtype == 'residues':
	brnn_network = brnn_architecture.BRNN_MtM(input_size, hidden_size, 
									num_layers, num_classes).to(device)
	
	# Set collate function
	if problem_type == 'classification':
		collate_function = pid.res_class_collate
	else:
		collate_function = pid.res_regress_collate

else:
	print('Invalid datatype.')
	sys.exit()


# Split data
train, val, test = pid.split_data(data_file, datatype=dtype, problem_type=problem_type, 
							num_classes=num_classes, split_file=split_file)

# Add data to dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           collate_fn=collate_function,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val,
                                           batch_size=batch_size,
                                           collate_fn=collate_function,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=1,		# Set test batch size to 1
                                          collate_fn=collate_function,
                                          shuffle=False)

# Train network
train_loss, val_loss = train_network.train(brnn_network, train_loader, val_loader, datatype=dtype, 
						problem_type=problem_type, weights_file=saved_weights, stop_condition=stop_cond,
						device=device, learn_rate=learning_rate, batch_size=batch_size, 
						n_epochs=num_epochs)
brnn_plot.training_loss(train_loss, val_loss)

# Test network
#TODO: write this function
train_network.test(brnn_network, test_loader, datatype=dtype, problem_type=problem_type,
		weights_file=saved_weights, num_classes=num_classes, device=device)

