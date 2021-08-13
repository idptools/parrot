"""
Core training module of PARROT

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from parrot import brnn_plot
from parrot import encode_sequence


def train(network, train_loader, val_loader, datatype, problem_type, weights_file,
          stop_condition, device, learn_rate, n_epochs, verbose=False, silent=False):
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
    verbose : bool, optional
            If true, causes training updates to be written every epoch, rather than
            every 5 epochs.
    silent : bool, optional
            If true, causes not training updates to be written to standard out.

    Returns
    -------
    list
            A list of the average training set losses achieved at each epoch
    list
            A list of the average validation set losses achieved at each epoch
    """

    # Set optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learn_rate)

    # Set loss criteria
    if problem_type == 'regression':
        if datatype == 'residues':
            criterion = nn.MSELoss(reduction='sum')
        elif datatype == 'sequence':
            criterion = nn.L1Loss(reduction='sum')
    elif problem_type == 'classification':
        criterion = nn.CrossEntropyLoss(reduction='sum')

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
    for epoch in range(n_epochs):  # Main loop

        # Initialize training and testing loss for epoch
        train_loss = 0
        val_loss = 0

        # Iterate over batches
        for i, (names, vectors, targets) in enumerate(train_loader):
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

        for names, vectors, targets in val_loader:
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

            # Increment val loss
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
            torch.save(network.state_dict(), weights_file)  # Save model

        # Append losses to lists
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        if verbose:
            print('Epoch %d\tLoss %.4f' % (epoch, val_loss))
        elif epoch % 5 == 0 and silent is False:
            print('Epoch %d\tLoss %.4f' % (epoch, val_loss))

        # This is placed here to ensure that the best network, even if the performance
        # improvement is marginal, is saved.
        if end_training:
            break

    # Return loss per epoch so that they can be plotted
    return avg_train_losses, avg_val_losses


def test_labeled_data(network, test_loader, datatype,
                      problem_type, weights_file, num_classes,
                      probabilistic_classification, include_figs, 
                      device, output_file_prefix=''):
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
    probabilistic_classification: bool
            Whether output should be binary labels, or "weights" of each label type.
            This field is only implemented for binary, sequence classification tasks.
    include_figs: bool
            Whether or not matplotlib figures should be generated.
    device : str
            Location of where testing will take place--should be either 'cpu' or
            'cuda' (GPU). If available, training on GPU is typically much faster.
    output_file_prefix : str
            Path and filename prefix to which the test set predictions and plots will be saved. 

    Returns
    -------
    float
            The average loss across the entire test set
    list of lists
            Details of the output predictions for each of the sequences in the test set. Each
            inner list represents a sample in the test set, with the format: [sequence_vector,
            true_value, predicted_value, sequence_ID]
    """

    # Load network weights
    network.load_state_dict(torch.load(weights_file))

    # Get output directory for images
    network_filename = weights_file.split('/')[-1]
    output_dir = weights_file[:-len(network_filename)]

    # Set loss criteria
    if problem_type == 'regression':
        criterion = nn.MSELoss()
    elif problem_type == 'classification':
        criterion = nn.CrossEntropyLoss()

    test_loss = 0
    all_targets = []
    all_outputs = []
    predictions = []
    for names, vectors, targets in test_loader: 	# batch size of 1
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

        test_loss += loss.data.item()  # Increment test loss
        all_outputs.append(outputs.detach())

        # Add to list as: [seq_vector, true value, predicted value, name]
        predictions.append([vectors[0].cpu().numpy(), targets.cpu().numpy()
                           [0], outputs.cpu().detach().numpy(), names[0]])

    # Plot 'accuracy' depending on the problem type and datatype
    if problem_type == 'regression':
        if datatype == 'residues':
            if include_figs:
                brnn_plot.residue_regression_scatterplot(all_targets, all_outputs, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions
            for i in range(len(predictions)):
                predictions[i][2] = predictions[i][2].flatten()
                predictions[i][1] = predictions[i][1].flatten()

        elif datatype == 'sequence':
            if include_figs:
                brnn_plot.sequence_regression_scatterplot(all_targets, all_outputs, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions
            for i in range(len(predictions)):
                predictions[i][2] = predictions[i][2][0][0]
                predictions[i][1] = predictions[i][1][0]

    elif problem_type == 'classification':

        if datatype == 'residues':
            if include_figs:
                brnn_plot.res_confusion_matrix(all_targets, all_outputs, num_classes, 
                                            output_file_prefix=output_file_prefix)

            # Format predictions and assign class predictions
            for i in range(len(predictions)):
                pred_values = []
                for j in range(len(predictions[i][2])):
                    pred_values = np.argmax(predictions[i][2], axis=1)[0]
                predictions[i][2] = np.array(pred_values, dtype=np.int)

        elif datatype == 'sequence':
            if probabilistic_classification:
                # Probabilistic assignment of class predictions
                # Optional implementation for classification tasks
                # e.g. every sequence is assigned probabilities
                # corresponding to each possible class
                pred_probabilities = []
                for i in range(len(predictions)):
                    softmax = np.exp(predictions[i][2][0])
                    probs = softmax / np.sum(softmax)
                    predictions[i][2] = probs
                    pred_probabilities.append(probs)

                # Plot ROC and PR curves
                if include_figs:
                    brnn_plot.plot_roc_curve(all_targets, pred_probabilities, num_classes, 
                                            output_file_prefix=output_file_prefix)
                    brnn_plot.plot_precision_recall_curve(all_targets, pred_probabilities, 
                                            num_classes, output_file_prefix=output_file_prefix)

            else:
                # Absolute assignment of class predictions
                # e.g. every sequence receives an integer class label
                for i in range(len(predictions)):
                    pred_value = np.argmax(predictions[i][2])
                    predictions[i][2] = int(pred_value)

                # Plot confusion matrix (if not in probabilistic classification mode)
                if include_figs:
                    brnn_plot.confusion_matrix(all_targets, all_outputs, num_classes, 
                                                output_file_prefix=output_file_prefix)

    return test_loss / len(test_loader.dataset), predictions


def test_unlabeled_data(network, sequences, device, encoding_scheme='onehot', encoder=None, print_frequency=None):
    """Test a trained BRNN on unlabeled sequences

    Use a trained network to make predictions on previously-unseen data.

    ** 
    Note: Unlike the previous functions, `network` here must have pre-loaded
    weights. 
    **

    Parameters
    ----------
    network : PyTorch network object
            A BRNN network with the desired architecture and pre-loaded weights
    sequences : list
            A list of amino acid sequences to test using the network
    device : str
            Location of where testing will take place--should be either 'cpu' or
            'cuda' (GPU). If available, training on GPU is typically much faster.
    encoding_scheme : str, optional
            How amino acid sequences are to be encoded as numeric vectors. Currently,
            'onehot','biophysics' and 'user' are the implemented options.
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    print_frequency : int
            If provided defines at what sequence interval an update is printed.
            Default = None.
    
    Returns
    -------
    dict
            A dictionary containing predictions mapped to sequences
    """

    pred_dict = {}

    local_count = -1
    total_count = len(sequences)

    for seq in sequences:

        local_count = local_count + 1
        if print_frequency is not None:
            if local_count % print_frequency == 0:
                print(f'On {local_count} of {total_count}')

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
