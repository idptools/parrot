#!/usr/bin/env python

import numpy as np
import torch
import itertools
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


# Residue-wise scatterplot: each seq has different patterning 
# (up to ~70 different marker-color combinations)
def residue_regression_scatterplot(true, predicted):
	true_list = []
	pred_list = []

	# TODO: does this cycle for every point? Or every sequence?
	marker = itertools.cycle(('>', '+', '.', 'o', '*', 'v', 'D')) 

	for item in true:
		single_frag = item.numpy()[0].flatten()
		true_list.append(list(single_frag))
	for item in predicted:
		single_frag = item.numpy()[0].flatten()
		pred_list.append(list(single_frag))

	for i in range(len(true_list)):
		plt.scatter(true_list[i], pred_list[i], s=6, marker=next(marker))

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
	plt.show()


# Sequence-wise classification: Confusion matrix
def confusion_matrix(true_classes, predicted_classes, num_classes):
	cm = np.zeros((num_classes, num_classes))
	for i in range(len(true_classes)):
		cm[true_classes[i][0], np.argmax(predicted_classes[i][0].numpy())] += 1

	return cm

def res_confusion_matrix(true_classes, predicted_classes, num_classes):
	#print(true_classes)
	#print()
	true_list = []
	pred_list = []

	for item in true_classes:
		single_frag = list(item[0].numpy().flatten())
		true_list = true_list + single_frag

	for item in predicted_classes:
		single_frag = item[0].permute(1, 0).numpy()

		for residue in single_frag:
			pred_list.append(np.argmax(residue))

	cm = np.zeros((num_classes, num_classes))
	for i in range(len(true_list)):
		cm[true_list[i], pred_list[i]] += 1

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





