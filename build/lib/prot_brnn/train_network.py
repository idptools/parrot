"""
File that carries out the core training of the package.

.............................................................................
prot_brnn was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/holehouse-lab/prot-brnn

Licensed under the MIT license. 
"""

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from prot_brnn import brnn_plot
from prot_brnn import encode_sequence

def train(network, train_loader, val_loader, datatype, problem_type, weights_file,
	stop_condition, device, learn_rate, n_epochs, verbosity=1):
	"""Train a BRNN and save the best performing network weights

	Train the network on a training set, and every epoch evaluate its performance on
	a validation set. Save the network weights that acheive the best performance on
	the validation set.

	User must specify the machine learning tast (`problem_type`) and the format of
	the data (`datatype`). Additionally, this function requires the learning rate
	hyperparameter and the number of epochs of training. The other hyperparameters, 
	number of hidden layers and hidden vector size, are implictly included on the 
	the provided network.

	The user may specify if they want to train the network for a set number of
	epochs or until an automatic stopping condition is reached with the argument
	`stop_condition`. Depending on the stopping condition used, the `n_epochs`
	argument will have a different role.

	Parameters
	----------
	network : PyTorch network object
		A BRNN network with the desired architecture
	train_loader : PyTorch DataLoader object
		A DataLoader containing the sequences and targets of the training set
	val_loader : PyTorch DataLoader object
		A DataLoader containing the sequences and targets of the validation set
	datatype : str
		The format of values in the dataset. Should be 'sequence' for datasets
		with a single value (or class label) per sequence, or 'residues' for
		datasets with values (or class labels) for every residue in a sequence.
	problem_type : str
		The machine learning task--should be either 'regression' or
		'classification'.
	weights_file : str
		A path to the location where the best_performing network weights will be
		saved
	stop_condition : str
		Determines when to conclude network training. If 'iter', then the network
		will train for `n_epochs` epochs, then stop. If 'auto' then the network
		will train for at least `n_epochs` epochs, then begin assessing whether
		performance has sufficiently stagnated. If the performance plateaus for
		`n_epochs` consecutive epochs, then training will stop.
	device : str
		Location of where training will take place--should be either 'cpu' or
		'cuda' (GPU). If available, training on GPU is typically much faster.
	learn_rate : float
		Initial learning rate of network training. The training process is
		controlled by the Adam optimization algorithm, so this learning rate
		will tend to decrease as training progresses.
	n_epochs : int
		Number of epochs to train for, or required to have stagnated performance
		for, depending on `stop_condition`.
	verbosity : int, optional
		The degree to which training updates are written to standard out (default
		is 1).

	Returns
	-------
	list
		A list of the average training set losses achieved at each epoch
	list
		A list of the average validation set losses achieved at each epoch
	"""

	# Max verbosity level is 2:
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

	network = network.float()
	total_step = len(train_loader)
	min_val_loss = np.inf
	avg_train_losses = []
	avg_val_losses = []

	if stop_condition == 'auto':
		min_epochs = n_epochs
		# Set to some arbitrarily large number of iterations -- will stop automatically
		n_epochs = 20000000
		last_decrease = 0

	# Train the model - evaluate performance on val set every epoch
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

		# Only save updated weights to memory if they improve val set performance
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

	# Return loss per epoch so that they can be plotted
	return avg_train_losses, avg_val_losses


def test_labeled_data(network, test_loader, datatype, problem_type, 
						weights_file, num_classes, device):
	"""Test a trained BRNN on labeled sequences

	Using the saved weights of a trained network, run a set of sequences through
	the network and evaluate the performancd. Return the average loss per
	sequence and plot the results. Testing a network on previously-unseen data 
	provides a useful estimate of how generalizeable the network's performance is.

	Parameters
	----------
	network : PyTorch network object
		A BRNN network with the desired architecture
	test_loader : PyTorch DataLoader object
		A DataLoader containing the sequences and targets of the test set
	datatype : str
		The format of values in the dataset. Should be 'sequence' for datasets
		with a single value (or class label) per sequence, or 'residues' for
		datasets with values (or class labels) for every residue in a sequence.
	problem_type : str
		The machine learning task--should be either 'regression' or
		'classification'.
	weights_file : str
		A path to the location of the best_performing network weights
	num_classes: int
		Number of data classes. If regression task, put 1.
	device : str
		Location of where testing will take place--should be either 'cpu' or
		'cuda' (GPU). If available, training on GPU is typically much faster.

	Returns
	-------
	float
		The average loss across the entire test set
	"""

	# Load network weights
	network.load_state_dict(torch.load(weights_file))

	# Get output directory for images
	network_filename = weights_file.split('/')[-1]
	output_dir = weights_file[:-len(network_filename)]

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
			brnn_plot.residue_regression_scatterplot(all_targets, all_outputs, output_dir=output_dir)

		elif datatype == 'sequence':
			brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs, output_dir=output_dir)

	elif problem_type == 'classification':
		if datatype == 'residues':
			brnn_plot.res_confusion_matrix(all_targets, all_outputs, num_classes, output_dir=output_dir)

		elif datatype == 'sequence':
			brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes, output_dir=output_dir)

	# TODO: return training samples and predictions as output file?
	return test_loss / len(test_loader.dataset)

def test_unlabeled_data(network, sequences, device, encoding_scheme='onehot', encoder=None):
	"""Test a trained BRNN on unlabeled sequences

	Use a trained network to make predictions on previously-unseen data.

	** Note: Unlike the previous functions, `network` here must have pre-loaded
	weights. **

	Parameters
	----------
	network : PyTorch network object
		A BRNN network with the desired architecture and pre-loaded weights
	sequences : list
		A list of amino acid sequences to test using the network
	device : str
		Location of where testing will take place--should be either 'cpu' or
		'cuda' (GPU). If available, training on GPU is typically much faster.
	encoding_scheme : str
		How amino acid sequences are to be encoded as numeric vectors. Currently,
		'onehot' and 'biophysics' are the implemented options.
	encoder:
		TODO: ...

	Returns
	-------
	dict
		A dictionary containing predictions mapped to sequences
	"""

	pred_dict = {}
	for seq in sequences:
		if encoding_scheme == 'onehot':
			seq_vector = encode_sequence.one_hot(seq)
		elif encoding_scheme == 'biophysics':
			seq_vector = encode_sequence.biophysics(seq)
		elif encoding_scheme == 'user':
			seq_vector = encoder.encode(seq)

		seq_vector = seq_vector.view(1, len(seq_vector), -1)

		# Forward pass
		outputs = network(seq_vector.float()).detach().numpy()
		pred_dict[seq] = outputs

	return pred_dict
