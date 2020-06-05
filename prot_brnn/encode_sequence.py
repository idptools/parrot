import numpy as np
import torch
import sys

'''
File containing function(s) for encoding a string of amino acids into a numeric vector.

Although these functions could easily be contained in another file, it is provided here
to allow for easy extensibility. In case the user wishes to define their own encoding
scheme, they can modify the two schemes provided below or add their own. (Additionally,
they must modify the code where the encoding functions are called in <file_name.py>)
'''

############  One-hot encoding  #############
ONE_HOT = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
		   'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}

def one_hot(seq):
	"""Convert an amino acid sequence to a PyTorch tensor of one-hot vectors

	Each amino acid is represented by a length 20 vector with a single `1` and
	19 `0`s. Inputing a sequence with a nono-canonical amino acid letter will
	cause the program to exit.

	E.g. Glutamic acid (E) is encoded: [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

	Parameters
	----------
	seq : str
		An uppercase sequence of amino acids (single letter code)

	Returns
	-------
	torch.IntTensor
		a PyTorch tensor representing the encoded sequence
	"""

	l = len(seq)
	m = np.zeros((l, 20))
	try:
		for i in range(l):
			m[i, ONE_HOT[seq[i]]] = 1
	except:
		print('Error: invalid amino acid detected:', seq[i])
		sys.exit()
	return torch.from_numpy(m)

############  Biophysical scale encoding  #############
# TODO: add more

# Map each amino acid to a vector of biophysical properties (int)
## TODO: list the properties here
# 0: Hydrophobicity (Kyte-Doolitle * 10)
# 1: Charge
# 2: pI * 10
# 3: Molecular weight (Da)

# Others potentially?
# Solvation surface area
# Pi-system (AKA aromatic)
# ...
BIOPHYSICS = {	'A':[18, 0, 60, 89], 
				'C':[25, 0, 51, 121], 
				'D':[-35, -1, 28, 133], 
				'E':[-35, -1, 32, 147], 
				'F':[28, 0, 55, 165], 
				'G':[-4, 0, 60, 75], 
				'H':[-32, 1, 76, 155], 
				'I':[45, 0, 60, 131], 
				'K':[-39, 1, 97, 146], 
				'L':[38, 0, 60, 131],
		   		'M':[19, 0, 57, 149],
		   		'N':[-35, 0, 54, 132],
		   		'P':[-16, 0, 63, 115],
		   		'Q':[-35, 0, 57, 146],
		   		'R':[-45, 1, 108, 174],
		   		'S':[-8, 0, 57, 105],
		   		'T':[-7, 0, 56, 119],
		   		'V':[42, 0, 60, 117],
		   		'W':[-9, 0, 59, 204],
		   		'Y':[-13, 0, 57, 181]
		   		}

def biophysics(seq):
	"""Convert an amino acid sequence to a PyTorch tensor with biophysical encoding

	Each amino acid is represented by a length 4 vector with each value representing
	a biophysical property. The four encoded biophysical scales are Kyte-Doolittle
	hydrophobicity, charge, isoelectric point, and molecular weight. Each value is 
	scaled so that all are integers. Inputing a sequence with a nono-canonical amino
	acid letter will cause the program to exit.

	E.g. Glutamic acid (E) is encoded: [-35, -1, 32, 147]

	Parameters
	----------
	seq : str
		An uppercase sequence of amino acids (single letter code)

	Returns
	-------
	torch.IntTensor
		a PyTorch tensor representing the encoded sequence
	"""
	l = len(seq)
	m = np.zeros((l, len(BIOPHYSICS['A'])))
	try:
		for i in range(l):
			m[i] = BIOPHYSICS[seq[i]]
	except:
		print('Error: invalid amino acid detected:', seq[i])
		sys.exit()
	return torch.from_numpy(m)

##################################################

# Add other encoding schemes here


