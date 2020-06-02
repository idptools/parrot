#!/usr/bin/env python

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import brnn_plot
import encode_sequence

def train(network, train_loader, val_loader, datatype, problem_type, weights_file,
	stop_condition, device, learn_rate, n_epochs, verbosity=1):
	'''
	problem_type = 'regression' or 'classification'
	stop_condition = 'auto' or 'iter'
	datatype = 'sequence' or 'residues'
	verbose = True, False, or None
	'''
	# Set verbosity level:
	if verbosity > 2:
		verbosity = 2

	# Set optimizer
	optimizer = torch.optim.Adam(network.parameters(), lr=learn_rate)

	# Set loss criteria
	if problem_type == 'regression':
		if datatype == 'residues':
			criterion = nn.MSELoss(reduction='sum')
		elif datatype == 'sequence':
			criterion = nn.L1Loss()	# TODO: L1 or MSE?
	elif problem_type == 'classification':
		if datatype == 'residues':
			criterion = nn.CrossEntropyLoss(reduction='sum') # TODO: double check if this is correct
		elif datatype == 'sequence':
			criterion = nn.CrossEntropyLoss()

	# Train the model - evaluate performance on val set every epoch
	# Only save updated weights to memory if they improve val set performance
	network = network.float()	# TODO: is this necessary?
	total_step = len(train_loader)
	min_val_loss = np.inf
	avg_train_losses = []
	avg_val_losses = []

	if stop_condition == 'auto':
		min_epochs = n_epochs
		# Set to some arbitrarily large number of iterations -- will stop automatically
		n_epochs = 20000000
		last_decrease = 0

	end_training = False
	for epoch in range(n_epochs):	# Main loop

		# Initialize training and testing loss for epoch
		train_loss = 0
		val_loss = 0

		# Iterate over batches
		for i, (vectors, targets) in enumerate(train_loader):
			vectors = vectors.to(device)
			targets = targets.to(device)
		
			# Forward pass
			outputs = network(vectors.float())

			if problem_type == 'regression':
				loss = criterion(outputs, targets.float())
			else:
				if datatype == 'residues':
					outputs = outputs.permute(0, 2, 1)
				loss = criterion(outputs, targets.long())

			train_loss += loss.data.item()
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		for vectors, targets in val_loader:
			vectors = vectors.to(device)
			targets = targets.to(device)

			# Forward pass
			outputs = network(vectors.float())
			if problem_type == 'regression':
				loss = criterion(outputs, targets.float())
			else:
				if datatype == 'residues':
					outputs = outputs.permute(0, 2, 1)
				loss = criterion(outputs, targets.long())

			# Increment test loss
			val_loss += loss.data.item()

		# Avg loss:
		train_loss /= len(train_loader.dataset)
		val_loss /= len(val_loader.dataset)

		signif_decrease = True
		if stop_condition == 'auto' and epoch > min_epochs - 1:
			# Check to see if loss has stopped decreasing
			last_epochs_loss = avg_val_losses[-min_epochs:]

			for loss in last_epochs_loss:
				if val_loss >= loss*0.995:
					signif_decrease = False

			# If network performance has plateaued over the last range of epochs, end training
			if not signif_decrease and epoch - last_decrease > min_epochs:
				end_training = True

		# If test loss is at a minimum
		if val_loss < min_val_loss:
			min_val_loss = val_loss 	# Reset min_val_loss
			last_decrease = epoch
			torch.save(network.state_dict(), weights_file)	# Save model

		# Append losses to lists
		avg_train_losses.append(train_loss)
		avg_val_losses.append(val_loss)

		if verbosity == 2:
			print('Epoch %d\tLoss %.4f' % (epoch, val_loss))
		elif epoch % 5 == 0 and verbosity == 1:
			print('Epoch %d\tLoss %.4f' % (epoch, val_loss))

		# This is placed here to ensure that the best network, even if the performance
		# improvement is marginal, is saved.
		if end_training:
			break

	# Load best-performing network
	network.load_state_dict(torch.load(weights_file)) # TODO: Unnecessary? get rid of this probably

	# Return loss per epoch so that they can be plotted
	return avg_train_losses, avg_val_losses


def test_labeled_data(network, test_loader, datatype, problem_type, 
						weights_file, num_classes, device):
	# Set loss criteria
	if problem_type == 'regression':
		if datatype == 'residues':
			criterion = nn.MSELoss(reduction='sum') # TODO: Is this the best way? 'none'?
		elif datatype == 'sequence':
			criterion = nn.L1Loss()	# TODO: L1 or MSE?
	elif problem_type == 'classification':
		if datatype == 'residues':
			criterion = nn.CrossEntropyLoss(reduction='sum') # TODO: double check if this is correct
		elif datatype == 'sequence':
			criterion = nn.CrossEntropyLoss()

	test_loss = 0
	all_targets = []
	all_outputs = []
	for vectors, targets in test_loader:
		all_targets.append(targets)

		vectors = vectors.to(device)
		targets = targets.to(device)

		# Forward pass
		outputs = network(vectors.float())
		if problem_type == 'regression':
			loss = criterion(outputs, targets.float())
		else:
			if datatype == 'residues':
				outputs = outputs.permute(0, 2, 1)
			loss = criterion(outputs, targets.long())
	
		test_loss += loss.data.item() # Increment test loss
		all_outputs.append(outputs.detach())

	# Calculate 'accuracy' depending on the problem type and datatype
	if problem_type == 'regression':
		if datatype == 'residues':
			# histogram of MSEs?
			brnn_plot.residue_regression_scatterplot(all_targets, all_outputs)

		elif datatype == 'sequence':
			# scatterplot of true vs predicted values
			brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs)

	elif problem_type == 'classification':
		if datatype == 'residues':
			# confusion matrix of all residues
			brnn_plot.res_confusion_matrix(all_targets, all_outputs, num_classes)
		elif datatype == 'sequence':
			# confusion matrix
			brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes)

	# TODO: return training samples and predictions as output file
	return test_loss / len(test_loader)

def test_unlabeled_data(network, sequences, device, encoding_scheme='onehot'):
	'''
	Pass in a list of sequences along with the network with pre-loaded weights.
	Return a dictionary with values mapped to each sequence
	'''
	pred_dict = {}
	for seq in sequences:

		if encoding_scheme == 'onehot':
			seq_vector = encode_sequence.one_hot(seq)
		elif encoding_scheme == 'biophysics':
			seq_vector = encode_sequence.biophysics(seq)
		seq_vector = seq_vector.view(1, len(seq_vector), -1)

		# Forward pass
		outputs = network(seq_vector.float()).detach().numpy()
		pred_dict[seq] = outputs

	return pred_dict

