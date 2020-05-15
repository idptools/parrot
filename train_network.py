#!/usr/bin/env python

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import brnn_plot

# TODO: validate that this function works as intended
def train(network, train_loader, val_loader, datatype, problem_type, weights_file,
	stop_condition, device='cpu', learn_rate=0.001, batch_size=15, n_epochs=25):
	'''
	problem_type = 'regression' or 'classification'
	stop_condition = 'auto' or 'iter'
	datatype = 'sequence' or 'residues'
	'''
	# Set optimizer
	optimizer = torch.optim.Adam(network.parameters(), lr=learn_rate)	# TODO: look into best optimizer

	# Set loss criteria
	if problem_type == 'regression':
		if datatype == 'residues':
			criterion = nn.MSELoss(reduction='sum') # TODO: Is this the best way? 'none'?
		elif datatype == 'sequence':
			criterion = nn.L1Loss()	# TODO: L1 or MSE?
	elif problem_type == 'classification':
		if datatype == 'residues':
			criterion = nn.CrossEntropyLoss(reduction='none') # TODO: double check if this is correct
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

			#print(outputs.shape, targets.shape)
			# TODO: fix uneven shapes for seq-regression

			if problem_type == 'regression':
				loss = criterion(outputs, targets.float())
			else:
				loss = criterion(outputs, targets.long())

			train_loss += loss
			
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
				loss = criterion(outputs, targets.long())

			# Increment test loss
			val_loss += loss

		# Avg loss:
		train_loss /= len(train_loader)		# TODO: double check that len(DataLoader) == num_samples
		val_loss /= len(val_loader)

		signif_decrease = False
		if stop_condition == 'auto' and epoch > min_epochs - 1:
			# Check to see if loss has stopped decreasing
			last_epochs_loss = avg_val_losses[-min_epochs:]
			for loss in last_epochs_loss:
				if val_loss < loss*0.995:
					signif_decrease = True
					last_decrease = epoch

			# If network performance has plateaued over the last 8 epochs, end training
			if not signif_decrease and epoch - last_decrease > min_epochs:
				end_training = True

		# If test loss is at a minimum
		if val_loss < min_val_loss:
			min_val_loss = val_loss 	# Reset min_val_loss
			torch.save(network.state_dict(), weights_file)	# Save model

		# Append losses to lists
		avg_train_losses.append(train_loss)
		avg_val_losses.append(val_loss)

		# This is placed here to ensure that the best network, even if the performance
		# improvement is marginal, is saved.
		if end_training:
			break

	# Load best-performing network
	network.load_state_dict(torch.load(weights_file)) # TODO: Unnecessary? get rid of this probably

	# Return loss per epoch so that they can be plotted
	return avg_train_losses, avg_val_losses


def test(network, test_loader, datatype, problem_type, weights_file, num_classes=2, device='cpu'):
	# Set loss criteria
	if problem_type == 'regression':
		if datatype == 'residues':
			criterion = nn.MSELoss(reduction='sum') # TODO: Is this the best way? 'none'?
		elif datatype == 'sequence':
			criterion = nn.L1Loss()	# TODO: L1 or MSE?
	elif problem_type == 'classification':
		if datatype == 'residues':
			criterion = nn.CrossEntropyLoss(reduction='none') # TODO: double check if this is correct
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
			loss = criterion(outputs, targets.long())
	
		test_loss += loss # Increment test loss
		all_outputs.append(outputs.detach())


	# Calculate 'accuracy' depending on the problem type and datatype
	# TODO: improve this and write these functions in brnn_plot.py
	if problem_type == 'regression':
		if datatype == 'residues':
			# histogram of MSEs?
			brnn_plot.residue_regression_histogram(all_targets, all_outputs)

		elif datatype == 'sequence':
			# scatterplot of true vs predicted values
			brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs)

	elif problem_type == 'classification':
		if datatype == 'residues':
			# confusion matrix of all residues
			brnn_plot.confusion_matrix(all_targets, all_outputs)
			# Anything else?
		elif datatype == 'sequence':
			# confusion matrix
			print(brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes))

	return test_loss

