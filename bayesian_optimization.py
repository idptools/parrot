#!/usr/bin/env python

import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import train_network
import brnn_architecture
import math

# TODO: are variance and lengthscale also hyperparameters that I will need to optimize somehow?
# TODO: fix self.std
class BayesianOptimizer(object):
	"""A class for conducting Bayesian Optimization on a PyTorch RNN

	Sets up and runs GPy Bayesian Optimization in order to choose the best-
	performing hyperparameters for a RNN for a given machine learning task. 
	Iteratively change learning rate, hidden vector size, and the number of layers
	in the network, then train and validating using 5-fold cross validation.

	Attributes
	----------
	cv_dataloaders : list of tuples of PyTorch DataLoader objects
		For each of the cross-val folds, a tuple containing a training set
		DataLoader and a validation set DataLoader.
	input_size : int
		Length of the amino acid encoding vectors
	n_epochs : int
		Number of epochs to train for each iteration of the algorithm
	n_classes : int
		Number of classes
	n_folds : int
		Number of cross-validation folds
	problem_type : str
		'classification' or 'regression'
	dtype : str
		'sequence' or 'residues'
	weights_file : str
		Path to which the network weights will be saved during training
	device : str
		'cpu' or 'cuda' depending on system hardware
	verbosity : int
		level of how descriptive the output to console message will be
	bds : list of dicts
		GPy-compatible bounds for each of the hyperparameters to be optimized
	std : float
		Average standard deviation across cross-validation replicates. Used to
		estimate noise inherent to the system.

	Methods
	-------
	compute_cv_loss(hyperparameters)
		Compute the average cross-val loss for a given set of hyperparameters
	eval_cv_brnns(lr, nl, hs)
		Train and test a network with given parameters across all cross-val folds
	optimize()
		Set up and run Bayesian Optimization on the BRNN using GPy
	"""

	def __init__(self, cv_dataloaders, input_size, n_epochs, 
				n_classes, dtype, weights_file, device, verbosity):
		"""
		Parameters
		----------
		cv_dataloaders : list of tuples of PyTorch DataLoader objects
			For each of the cross-val folds, a tuple containing a training set
			DataLoader and a validation set DataLoader.
		input_size : int
			Length of the amino acid encoding vectors
		n_epochs : int
			Number of epochs to train for each iteration of the algorithm
		n_classes : int
			Number of classes
		dtype : str
			'sequence' or 'residues'
		weights_file : str
			Path to which the network weights will be saved during training
		device : str
			'cpu' or 'cuda' depending on system hardware
		verbosity : int
			level of how descriptive the output to console message will be
		"""

		self.cv_loaders = cv_dataloaders
		self.input_size = input_size
		self.n_epochs = n_epochs
		self.n_folds = len(cv_dataloaders)
		self.n_classes = n_classes
		if n_classes > 1:
			self.problem_type = 'classification'
		else:
			self.problem_type = 'regression'

		self.dtype = dtype
		self.weights_file = weights_file
		self.device = device
		self.verbosity = verbosity

		self.bds = [{'name': 'log_learning_rate', 'type': 'continuous', 'domain': (-5, 0)}, # 0.00001-1
					{'name': 'n_layers', 'type': 'discrete', 'domain': tuple(range(1, 16))}, # up to 15
					{'name': 'hidden_size', 'type': 'discrete', 'domain': tuple(range(1, 31))}] # up to 30

		self.std = 0.0

	def compute_cv_loss(self, hyperparameters):
		"""Compute the average cross-val loss for a given set of hyperparameters

		Given N sets of hyperparameters, determine the average cross-validation loss
		for BRNNs trained with these parameters.

		Parameters
		----------
		hyperparameters : numpy float array
			Each row corresponds to a set of hyperparameters, in the order:
			[log_learining_rate, n_layers, hidden_size]

		Returns
		-------
		numpy float array
			a Nx1 numpy array of the average cross-val loss 
			per set of input hyperparameters
		"""

		cv_outputs = np.zeros((len(hyperparameters), self.n_folds))

		for i in range(len(hyperparameters)):
			# lr = hyperparameters[i][0]
			log_lr, nl, hs = hyperparameters[i]

			lr = 10**float(log_lr)
			nl = int(nl)
			hs = int(hs)

			if self.verbosity > 0:
				print('	%.6f	|	  %2d	  |	  %2d' % (lr, nl, hs))

			# Train and validate network with these hyperparams using k-fold CV
			cv_outputs[i] = self.eval_cv_brnns(lr, nl, hs)

		outputs = np.average(cv_outputs, axis=1)

		# Calculate the standard deviation of the losses from each set of cross-vals
		stddevs = np.std(cv_outputs, axis=1)
		self.std = np.average(stddevs) # FIXME
		return outputs


	def eval_cv_brnns(self, lr, nl, hs):
		"""Train and test a network with given parameters across all cross-val folds

		Parameters
		----------
		lr : float
			Learning rate of the network
		nl : int
			Number of hidden layers (for each direction) in the network
		hs : int
			Size of hidden vectors in the network

		Returns
		-------
		numpy float array
			the best validation loss from each fold of cross validation
		"""

		cv_losses = np.zeros(self.n_folds) - 1 # -1 so that it's obvious if something goes wrong

		for k in range(self.n_folds):
			if self.dtype == 'sequence':
				# Use a many-to-one architecture
				brnn_network = brnn_architecture.BRNN_MtO(self.input_size, hs, nl,
										self.n_classes, self.device).to(self.device)
			else:
				# Use a many-to-many architecture
				brnn_network = brnn_architecture.BRNN_MtM(self.input_size, hs, nl,
										self.n_classes, self.device).to(self.device)	

			# Train network with this set of hyperparameters
			train_losses, val_losses = train_network.train(brnn_network, self.cv_loaders[k][0],
										self.cv_loaders[k][1], self.dtype, self.problem_type,
										self.weights_file, stop_condition='iter', device=self.device,
										learn_rate=lr, n_epochs=self.n_epochs, verbosity=0)
			# Take best val loss
			best_val_loss = np.min(val_losses)
			cv_losses[k] = best_val_loss

			if self.verbosity > 1:
				print('[%d/%d] Loss: %.6f' % (k+1, self.n_folds, best_val_loss))

		return cv_losses


	def optimize(self):
		"""Set up and run Bayesian Optimization on the BRNN using GPy

		Returns
		-------
		list
			the best hyperparameters are chosen by Bayesian Optimization. Returned
			in the order: [lr, nl, hs]
		"""

		optimizer = BayesianOptimization(f=self.compute_cv_loss, 
										 domain=self.bds,
										 model_type='GP',
										 acquisition_type ='EI',
										 acquisition_jitter = 0.05, # TODO: what is this?
										 noise_var = self.std**2, # FIXME: Run a few tests to calculate noise a priori?
										 maximize=False)
		# TODO: what should max_iter be set at?
		optimizer.run_optimization(max_iter=80)

		ins = optimizer.get_evaluations()[0]
		outs = optimizer.get_evaluations()[1].flatten()

		if self.verbosity > 0:
			print("\nThe optimal hyperparameters are:\nlr = %.6f\nnl = %d\nhs = %d" % 
						(10**optimizer.x_opt[0], optimizer.x_opt[1], optimizer.x_opt[2]))
			print()

		return optimizer.x_opt
