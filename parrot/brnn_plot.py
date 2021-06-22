"""
Plot training results for regression and classification tasks on both 
sequence-mapped and residue-mapped data.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import numpy as np
import torch
import itertools
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from parrot import encode_sequence


def training_loss(train_loss, val_loss, output_dir=''):
    """Plot training and validation loss per epoch

    Figure is not displayed, but saved to file in current directory with the name
    'train_test.png'.

    Parameters
    ----------
    train_loss : list
            training loss across each epoch
    val_loss : list
            validation loss across each epoch
    output_dir : str, optional
            directory to which the plot will be saved (default is current directory)
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)

    num_epochs = len(train_loss)

    # Loss per epoch
    training_loss, = ax.plot(np.arange(1, num_epochs+1), train_loss, label='Train')
    validation_loss, = ax.plot(np.arange(1, num_epochs+1), val_loss, label='Val')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg loss")
    ax.set_title("Training and testing loss per epoch")
    ax.legend(handles=[training_loss, validation_loss], fontsize=14,
              facecolor='gainsboro', edgecolor='slategray')

    if num_epochs < 21:
        ax.set_xticks(np.arange(2, num_epochs+1, 2))
    elif num_epochs < 66:
        ax.set_xticks(np.arange(5, num_epochs+1, 5))
    elif num_epochs < 151:
        ax.set_xticks(np.arange(10, num_epochs+1, 10))
    else:
        ax.set_xticks(np.arange(50, num_epochs+1, 50))

    plt.savefig(output_dir + 'train_test.png')
    plt.clf()


def sequence_regression_scatterplot(true, predicted, output_dir=''):
    """Create a scatterplot for a sequence-mapped values regression problem

    Figure is displayed to console if possible and saved to file in current 
    directory with the name 'seq_scatter.png'.

    Parameters
    ----------
    true : list of PyTorch FloatTensors
            A list where each item is a [1 x 1] tensor with the true regression value
            of a particular sequence
    predicted : list of PyTorch FloatTensors
            A list where each item is a [1 x 1] tensor with the regression prediction
            for a particular sequence
    output_dir : str, optional
            directory to which the plot will be saved (default is current directory)
    """

    true_list = []
    pred_list = []

    for item in true:
        true_list.append(item.cpu().numpy()[0][0])
    for item in predicted:
        pred_list.append(item.cpu().numpy()[0][0])

    plt.scatter(true_list, pred_list)
    edge_vals = [0.9*min(min(true_list), min(pred_list)),
                 1.1*max(max(true_list), max(pred_list))]
    plt.xlim(edge_vals)
    plt.ylim(edge_vals)
    plt.plot(edge_vals, edge_vals, 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    slope, intercept, r_value, p_value, std_err = linregress(true_list, pred_list)
    plt.title('Testing accuracy: R^2=%.3f' % (r_value**2))
    plt.savefig(output_dir + 'seq_scatter.png')


def residue_regression_scatterplot(true, predicted, output_dir=''):
    """Create a scatterplot for a residue-mapped values regression problem

    Each sequence is plotted with a unique marker-color combination, up to 70
    different sequences.

    Figure is displayed to console if possible and saved to file in current 
    directory with the name 'res_scatter.png'.

    Parameters
    ----------
    true : list of PyTorch FloatTensors
            A list where each item is a [1 x len(sequence)] tensor with the true
            regression values of each residue in a sequence
    predicted : list of PyTorch FloatTensors
            A list where each item is a [1 x len(sequence)] tensor with the 
            regression predictions for each residue in a sequence
    output_dir : str, optional
            directory to which the plot will be saved (default is current directory)
    """

    true_list = []
    pred_list = []

    marker = itertools.cycle(('>', '+', '.', 'o', '*', 'v', 'D'))

    for item in true:
        single_frag = item.cpu().numpy()[0].flatten()
        true_list.append(list(single_frag))
    for item in predicted:
        single_frag = item.cpu().numpy()[0].flatten()
        pred_list.append(list(single_frag))

    for i in range(len(true_list)):
        plt.scatter(true_list[i], pred_list[i], s=6, marker=next(marker))

    plt.figure(1)

    left, right = plt.xlim()
    bottom, top = plt.ylim()
    edge_vals = [min(left, bottom), max(right, top)]
    plt.xlim(edge_vals)
    plt.ylim(edge_vals)
    plt.plot(edge_vals, edge_vals, 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    slope, intercept, r_value, p_value, std_err = linregress(sum(true_list, []), sum(pred_list, []))
    plt.title('Testing accuracy: R^2=%.3f' % (r_value**2))
    plt.savefig(output_dir + 'res_scatter.png')


def confusion_matrix(true_classes, predicted_classes, num_classes, output_dir=''):
    """Create a confusion matrix for a sequence classification problem

    Figure is displayed to console if possible and saved to file in current 
    directory with the name 'seq_CM.png'.

    Parameters
    ----------
    true_classes : list of PyTorch IntTensors
            A list where each item is a [1 x 1] tensor with the true class label of a
            particular sequence
    predicted_classes : list of PyTorch FloatTensors
            A list where each item is a [1 x num_classes] tensor prediction of the
            class label for a particular sequence
    num_classes : int
            Number of distinct data classes
    output_dir : str, optional
            directory to which the plot will be saved (default is current directory)
    """

    cm = np.zeros((num_classes, num_classes))
    for i in range(len(true_classes)):
        cm[true_classes[i][0], np.argmax(predicted_classes[i][0].cpu().numpy())] += 1

    df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16})  # font size
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.title('Test set confusion matrix')
    plt.tight_layout()
    plt.savefig(output_dir + 'seq_CM.png')


def res_confusion_matrix(true_classes, predicted_classes, num_classes, output_dir=''):
    """Create a confusion matrix for a residue classification problem

    Figure is displayed to console if possible and saved to file in current 
    directory with the name 'res_CM.png'.

    Parameters
    ----------
    true_classes : list of PyTorch IntTensors
            A list where each item is a [1 x len(sequence)] tensor with the true class
            label of the residues in a particular sequence
    predicted_classes : list of PyTorch FloatTensors
            A list where each item is a [1 x num_classes x len(sequence)] tensor
            with predictions of the class label for each residue in a particular
            sequence
    num_classes : int
            Number of distinct data classes
    output_dir : str, optional
            directory to which the plot will be saved (default is current directory)
    """

    true_list = []
    pred_list = []

    for item in true_classes:
        single_frag = list(item[0].cpu().numpy().flatten())
        true_list = true_list + single_frag

    for item in predicted_classes:
        single_frag = item[0].permute(1, 0).cpu().numpy()

        for residue in single_frag:
            pred_list.append(np.argmax(residue))

    cm = np.zeros((num_classes, num_classes))
    for i in range(len(true_list)):
        cm[true_list[i], pred_list[i]] += 1

    df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16})  # font size
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.title('Test set confusion matrix')
    plt.tight_layout()
    plt.savefig(output_dir + 'res_CM.png')


def output_predictions_to_file(sequence_data, excludeSeqID, encoding_scheme,
                            probabilistic_class, encoder=None, output_dir=''):
    """Output sequences, their true values, and their predicted values to a file

    Used on the output of the test_unlabeled_data() function in the train_network module in
    order to detail the performance of the trained network on the test set. Produces the
    file "test_set_predictions.tsv" in output_dir. Each pair of lines in this tsvfile
    corresponds to a particular test set sequence, with the first containing the true data
    values, and the second line having the predicted data values.

    Parameters
    ----------
    sequence_data : list of lists
            Details of the output predictions for each of the sequences in the test set. Each
            inner list represents a sample in the test set, with the format: [sequence_vector,
            true_value, predicted_value, sequence_ID]
    excludeSeqID : bool
            Boolean indicating whether or not each line in `tsvfile` has a sequence ID
            (default is False)		
    encoding_scheme : str
            Description of how an amino acid sequence should be encoded as a numeric 
            vector. Providing a string other than 'onehot', 'biophysics', or 'user' 
            will produce unintended consequences.
    probabilistic_class : bool
            Flag indicating if probabilistic classification was specified by the user. If True,
            instead of class labels, predictions will be output as probabilities of each class.
    encoder: UserEncoder object, optional
            If encoding_scheme is 'user', encoder should be a UserEncoder object
            that can convert amino acid sequences to numeric vectors. If
            encoding_scheme is not 'user', use None.
    output_dir : str
            Directory where test set predictions file should be output.
    """

    seq_vectors = []
    true_vals = []
    pred_vals = []
    names = []
    count = 0
    for sequence in sequence_data:
        seq_vector, true_val, pred_val, name = sequence
        seq_vectors.append(seq_vector)
        true_vals.append(true_val)
        pred_vals.append(pred_val)

        if excludeSeqID:
            names.append('test' + str(count))
            count += 1
        else:
            names.append(name)

    # Decode the sequence vectors
    if encoding_scheme == 'onehot':
        sequences = encode_sequence.rev_one_hot(seq_vectors)
    elif encoding_scheme == 'biophysics':
        sequences = encode_sequence.rev_biophysics(seq_vectors)
    else:
        sequences = encoder.decode(seq_vectors)

    # Write to file
    filename = 'test_set_predictions.tsv'

    with open(output_dir + filename, 'w') as tsvfile:
        for i in range(len(names)):

            # Adjust formatting for residues or sequence data
            if isinstance(true_vals[i], np.ndarray):
                true_vals_format = ' '.join(true_vals[i].astype(str))
                pred_vals_format = ' '.join(pred_vals[i].astype(str))
            elif probabilistic_class:
                true_vals_format = true_vals[i]
                pred_vals_format = ' '.join(np.around(pred_vals[i], decimals=4).astype(str))
            else:
                true_vals_format = true_vals[i]
                pred_vals_format = pred_vals[i]

            '''
			Format:
			NAME_TRUE SEQUENCE TRUE_VALUE(S)
			NAME_PRED SEQUENCE PRED_VALUE(S)
			'''
            output_str = "%s_TRUE %s %s\n" % (names[i], sequences[i], true_vals_format)
            output_str = output_str + "%s_PRED %s %s\n" % (names[i], sequences[i], pred_vals_format)

            tsvfile.write(output_str)
