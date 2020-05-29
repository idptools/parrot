#!/usr/bin/env python

import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import train_network
import brnn_architecture
import math


#### ---------------------------------------- ####
class BayesianOptimizer(object):
	"""

	"""
	def __init__(self, cv_dataloaders, input_size, n_epochs, 
				n_classes, dtype, weights_file, device, verbosity):
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

		# TODO: what does this do? Is this necessary for BayesianOptimization function?
		# Are variance and lengthscale also hyperparameters that I will need to optimize somehow? 

		self.bds = [{'name': 'log_learning_rate', 'type': 'continuous', 'domain': (-5, 0)}, # 0.00001-1
					{'name': 'n_layers', 'type': 'discrete', 'domain': tuple(range(1, 16))}, # up to 15
					{'name': 'hidden_size', 'type': 'discrete', 'domain': tuple(range(1, 31))}] # up to 30

		self.std = 0.0

	def compute_CV_loss(self, hyperparameters):
		'''
		Given a set of hyperparameters (Nxn_folds), determine the average cross-validation loss
		for BRNNs trained with these parameters.

		Returns a (Nx1) numpy array with one network evaluation per row of input hyperparams
		'''
		cv_outputs = np.zeros((len(hyperparameters), self.n_folds))

		for i in range(len(hyperparameters)):
			# lr = hyperparameters[i][0]
			log_lr, nl, hs = hyperparameters[i]

			lr = 10**float(log_lr)
			nl = int(nl)
			hs = int(hs)

			if self.verbosity > 0:
				print('    %.6f    |      %2d      |      %2d' % (lr, nl, hs))

			# Train and validate network with these hyperparams using k-fold CV
			cv_outputs[i] = self.eval_cv_brnns(lr, nl, hs)

		outputs = np.average(cv_outputs, axis=1)
		stddevs = np.std(cv_outputs, axis=1)
		self.std = np.average(stddevs)
		return outputs


	def eval_cv_brnns(self, lr, nl, hs):
		# Best validation losses for all n CVs:
		cv_losses = np.zeros(self.n_folds) - 1 # -1 so that it's obvious if something went wrong

		for k in range(self.n_folds):
			# Initialize network architecture
			if self.dtype == 'sequence':
				# Use a many-to-one architecture
				# TODO: pass input_size depending on encoding scheme
				brnn_network = brnn_architecture.BRNN_MtO(self.input_size, hs, nl,
										self.n_classes, self.device).to(self.device)
			else:	# dtype == 'residues'
				# Use a many-to-many architecture
				brnn_network = brnn_architecture.BRNN_MtM(self.input_size, hs, nl,
										self.n_classes, self.device).to(self.device)	

			# Train using these parameters
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
		optimizer = BayesianOptimization(f=self.compute_CV_loss, 
                                		 domain=self.bds,
                                		 model_type='GP',
                                		 acquisition_type ='EI',
                                		 acquisition_jitter = 0.05, # TODO: what is this?
                                		 noise_var = self.std**2,
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

#### ---------------------------------------- ####
