#!/usr/bin/env python

import numpy as np
import torch

'''
File containing function(s) for encoding a string of amino acids into a numeric vector.

Although these functions could easily be contained in another file, it is provided here
to allow for easy extensibility. In case the user wishes to define their own encoding
scheme, they can modify the two schemes provided below or add their own. (Additionally,
they must modify the code where the encoding functions are called in <file_name.py>)
'''

############  One-hot encoding  #############
# Map each amino acid to a length 20 vector: E.g. [0 0 0 1 0 ... 0] is E (Glu)
ONE_HOT = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
		   'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}

# Convert an aa sequence to a matrix
def one_hot(seq):
	l = len(seq)
	m = np.zeros((l, 20))		# Return each amino acid as a length=20 vector
	for i in range(l):
		m[i, ONE_HOT[seq[i]]] = 1
	return torch.from_numpy(m)

############  Biophysical scale encoding  #############
# TODO

# Map each amino acid to a vector of ## biophysical properties
## TODO: list the properties here
# Mol. weight
# Hydrophobicity
# Charge
# Solvation surface area
# Pi-system (AKA aromatic)
# ...

BIOPHYSICS1 = {	'A':[101.01, .73, 0.0, 40.2, .01], 	# TODO: actually fill these in
				'C':[], 
				'D':[], 
				'E':[], 
				'F':[], 
				'G':[], 
				'H':[], 
				'I':[], 
				'K':[], 
				'L':[],
		   		'M':[],
		   		'N':[],
		   		'P':[],
		   		'Q':[],
		   		'R':[],
		   		'S':[],
		   		'T':[],
		   		'V':[],
		   		'W':[],
		   		'Y':[]
		   		}

def biophysics1(s):
	l = len(seq)
	m = np.zeros((l, 5))		# Return each amino acid as a length=5 vector
	for i in range(l):
		m[i] = BIOPHYSICS1[seq[i]]
	return torch.from_numpy(m)

##################################################

# Add other encoding schemes here


