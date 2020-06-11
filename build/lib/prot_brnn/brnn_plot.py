"""
File containing functions for plotting training results.

.............................................................................
prot_brnn was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/holehouse-lab/prot-brnn

Licensed under the MIT license. 
"""

import numpy as np
import torch
import itertools
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

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

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
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
	plt.show()


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
	plt.show()


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
	sn.set(font_scale=1.4) # for label size
	sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16}) # font size
	plt.ylabel('True labels')
	plt.xlabel('Predicted labels')
	plt.title('Test set confusion matrix')
	plt.tight_layout()
	plt.savefig(output_dir + 'seq_CM.png')
	plt.show()


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
	sn.set(font_scale=1.4) # for label size
	sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16}) # font size
	plt.ylabel('True labels')
	plt.xlabel('Predicted labels')
	plt.title('Test set confusion matrix')
	plt.tight_layout()
	plt.savefig(output_dir + 'res_CM.png')
	plt.show()
