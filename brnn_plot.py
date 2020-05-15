#!/usr/bin/env python

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Training / validation loss per epoch
def training_loss(train_loss, val_loss):
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

	plt.show()


# Residue-wise regression: histogram of MSEs
def residue_regression_histogram(true, predicted):
	return 0

# Residue-wise classification: confusion matrix of all residues + other plots?
# Sequence-wise classification: Confusion matrix
def confusion_matrix(true_classes, predicted_classes, num_classes):
	cm = np.zeros((num_classes, num_classes))
	for i in range(len(true_classes)):
		cm[true_classes[i][0], np.argmax(predicted_classes[i][0].numpy())] += 1

	return cm


# Sequence-wise regression: 2D scatterplot of true vs predicted
def sequence_regression_scatterplot(true, predicted):
	true_list = []
	pred_list = []

	for item in true:
		true_list.append(item.numpy()[0][0])
	for item in predicted:
		pred_list.append(item.numpy()[0][0])

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
	plt.show()





