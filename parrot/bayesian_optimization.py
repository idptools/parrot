"""
This file contains code for conducting Bayesian optimization.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import math

import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization

from parrot import train_network
from parrot import brnn_architecture


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
    max_iterations : int
            Maximum number of iterations to perform the optimization procedure
    silent : bool
            If true, do not print updates to console
    bds : list of dicts
            GPy-compatible bounds for each of the hyperparameters to be optimized
    """

    def __init__(self, cv_dataloaders, input_size, n_epochs, n_classes,
                 dtype, weights_file, max_iterations, device, silent):
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
        max_iterations : int
                Maximum number of iterations to perform the optimization procedure
        silent : bool
                If true, do not print updates to console
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
        self.max_iterations = max_iterations
        self.device = device
        self.silent = silent

        self.bds = [{'name': 'log_learning_rate', 'type': 'continuous', 'domain': (-5, -2)},  # 0.00001-0.01
                    {'name': 'n_layers', 'type': 'discrete', 'domain': tuple(range(1, 6))},  # 1-5
                    {'name': 'hidden_size', 'type': 'discrete', 'domain': tuple(range(5, 51))}]  # 5-50

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

            log_lr, nl, hs = hyperparameters[i]
            lr = 10**float(log_lr)
            nl = int(nl)
            hs = int(hs)

            # Train and validate network with these hyperparams using k-fold CV
            cv_outputs[i] = self.eval_cv_brnns(lr, nl, hs)
            avg = np.average(cv_outputs[i])

            if self.silent is False:
                print('  %.6f	|     %2d       |         %2d           |    %.3f' % (lr, nl, hs, avg))

        outputs = np.average(cv_outputs, axis=1)
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

        cv_losses = np.zeros(self.n_folds) - 1  # -1 so that it's obvious if something goes wrong

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
                                                           learn_rate=lr, n_epochs=self.n_epochs, silent=True)
            # Take best val loss
            best_val_loss = np.min(val_losses)
            cv_losses[k] = best_val_loss

        return cv_losses

    def initial_search(self, x):
        """Calculate loss and estimate noise for an initial set of hyperparameters

        Parameters
        ----------
        x : numpy array
                Array containing initial hyperparameters to test

        Returns
        -------
        numpy array
                Array containing the average losses of the input hyperparameters
        float
                The standard deviation of loss across cross-val folds for the
                input hyperparameters; an estimation of the training noise
        """
        cv_outputs = np.zeros((len(x), self.n_folds))
        y = np.zeros((len(x), 1))

        for i in range(len(x)):

            log_lr, nl, hs = x[i]
            lr = 10**float(log_lr)
            nl = int(nl)
            hs = int(hs)

            # Train and validate network with these hyperparams using k-fold CV
            cv_outputs[i] = self.eval_cv_brnns(lr, nl, hs)

        # Calculate the avg and standard deviation of the losses
        for i in range(len(cv_outputs)):
            y[i] = np.average(cv_outputs[i])
        stddevs = np.std(cv_outputs, axis=1)
        avg_stddev = np.average(stddevs)

        return y, avg_stddev

    def optimize(self):
        """Set up and run Bayesian Optimization on the BRNN using GPy

        Returns
        -------
        list
                the best hyperparameters are chosen by Bayesian Optimization. Returned
                in the order: [lr, nl, hs]
        """

        # Initial hyperparameter search -- used to get noise estimate
        x_init = np.array([[-3.0, 1, 20], [-3.0, 2, 20], [-3.0, 3, 20], [-3.0, 4, 20], [-3.0, 5, 20],
                           [-2.0, 2, 20], [-3.3, 2, 20], [-4.0, 2, 20], [-5.0, 2, 20],
                           [-3.0, 2, 5],  [-3.0, 2, 15], [-3.0, 2, 35], [-3.0, 2, 50]])
        y_init, noise = self.initial_search(x_init)

        if self.silent is False:
            print("\nInitial search results:")
            print("lr\tnl\ths\toutput")
            for i in range(len(x_init)):
                print("%.5f\t%2d\t%2d\t%.4f" % (10**x_init[i][0], x_init[i][1], x_init[i][2], y_init[i][0]))
            print("Noise estimate:", noise)
            print('\n')
            print('Primary optimization:')
            print('--------------------\n')
            print('Learning rate   |   n_layers   |   hidden vector size |  avg CV loss  ')
            print('======================================================================')

        optimizer = BayesianOptimization(f=self.compute_cv_loss,
                                         domain=self.bds,
                                         model_type='GP',
                                         acquisition_type='EI',
                                         acquisition_jitter=0.05,
                                         X=x_init,
                                         Y=y_init,
                                         noise_var=noise,
                                         maximize=False)

        optimizer.run_optimization(max_iter=self.max_iterations)

        ins = optimizer.get_evaluations()[0]
        outs = optimizer.get_evaluations()[1].flatten()

        if self.silent is False:
            print("\nThe optimal hyperparameters are:\nlr = %.5f\nnl = %d\nhs = %d" %
                  (10**optimizer.x_opt[0], optimizer.x_opt[1], optimizer.x_opt[2]))
            print()

        return optimizer.x_opt
