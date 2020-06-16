"""
File containing functions for encoding a string of amino acids into a numeric vector.

.............................................................................
prot_brnn was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/holehouse-lab/prot-brnn

Licensed under the MIT license. 
"""

import numpy as np
import torch
import sys
import os

ONE_HOT = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
		   'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}

def one_hot(seq):
	"""Convert an amino acid sequence to a PyTorch tensor of one-hot vectors

	Each amino acid is represented by a length 20 vector with a single 1 and
	19 0's Inputing a sequence with a nono-canonical amino acid letter will
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

# Map each amino acid to a vector of biophysical properties
# 0: Hydrophobicity
# 1: Charge
# 2: pI
# 3: Molecular weight (g/mol)
# 4: Aromatic amino acid
# 5: Capable of hydrogen bonding
# 6: Side chain SASA (measured from ACE-XX-NME dipeptide)
# 7: Backbone SASA (measured from ACE-XX-NME dipeptide)
# 8: Free energy of solvation
BIOPHYSICS = {	'A':[ 1.8,  0,  6.0,  89.1, 0, 0,  75.8,  76.1,    1.9], 
				'C':[ 2.5,  0,  5.1, 121.2, 0, 0, 115.4,  67.9,   -1.2], 
				'D':[-3.5, -1,  2.8, 133.1, 0, 1, 130.3,  71.8, -107.3], 
				'E':[-3.5, -1,  3.2, 147.1, 0, 1, 161.8,  68.1, -107.3], 
				'F':[ 2.8,  0,  5.5, 165.2, 1, 0, 209.4,  66.0,   -0.8], 
				'G':[-0.4,  0,  6.0,  75.1, 0, 0,   0.0, 115.0,    0.0], 
				'H':[-3.2,  1,  7.6, 155.2, 0, 1, 180.8,  67.5,  -52.7], # Avg of HIP and HIE 
				'I':[ 4.5,  0,  6.0, 131.2, 0, 0, 172.7,  60.3,    2.2], 
				'K':[-3.9,  1,  9.7, 146.2, 0, 1, 205.9,  68.7, -100.9], 
				'L':[ 3.8,  0,  6.0, 131.2, 0, 0, 172.0,  64.5,    2.3],
		   		'M':[ 1.9,  0,  5.7, 149.2, 0, 0, 184.8,  67.8,   -1.4],
		   		'N':[-3.5,  0,  5.4, 132.1, 0, 1, 142.7,  66.8,   -9.7],
		   		'P':[-1.6,  0,  6.3, 115.1, 0, 0, 134.3,  55.8,    2.0],
		   		'Q':[-3.5,  0,  5.7, 146.2, 0, 1, 173.3,  66.6,   -9.4],
		   		'R':[-4.5,  1, 10.8, 174.2, 0, 1, 236.5,  66.7, -100.9],
		   		'S':[-0.8,  0,  5.7, 105.1, 0, 1,  95.9,  72.9,   -5.1],
		   		'T':[-0.7,  0,  5.6, 119.1, 0, 1, 130.9,  64.1,   -5.0],
		   		'V':[ 4.2,  0,  6.0, 117.1, 0, 0, 143.1,  61.7,    2.0],
		   		'W':[-0.9,  0,  5.9, 204.2, 1, 1, 254.6,  64.3,   -5.9],
		   		'Y':[-1.3,  0,  5.7, 181.2, 1, 1, 222.5,  71.9,   -6.1]
		   		}

def biophysics(seq):
	"""Convert an amino acid sequence to a PyTorch tensor with biophysical encoding

	Each amino acid is represented by a length 4 vector with each value representing
	a biophysical property. The four encoded biophysical scales are Kyte-Doolittle
	hydrophobicity, charge, isoelectric point, and molecular weight. Inputing a 
	sequence with a nono-canonical amino acid letter will cause the program to exit.

	E.g. Glutamic acid (E) is encoded: [-3.5, -1., 3.2, 147]

	Parameters
	----------
	seq : str
		An uppercase sequence of amino acids (single letter code)

	Returns
	-------
	torch.FloatTensor
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

def parse_encode_file(file):
	"""Helper function to convert an encoding file into key:value dictionary"""

	with open(file) as f:
		lines = [x.strip().split() for x in f]

	l = len(lines[0]) - 1
	d = {}
	for line in lines:
		d[line[0]] = line[1:]

		if len(line) - 1 != l:
			print('Error: encoding file has invalid format.')
			sys.exit()

	return d, l

# TODO: test this
class UserEncoder():
	"""User-specified amino acid-to-vector encoding scheme"""

	def __init__(self, encode_file):
		"""
		Parameters
		----------
		encode_file : str
			A path to a file that describes the encoding scheme
		"""

		self.encode_file = os.path.abspath(encode_file)
		if not os.path.isfile(self.encode_file):
			print('Error: encoding file does not exist.')
			sys.exit()

		self.encode_dict, self.input_size = parse_encode_file(self.encode_file)

	def __len__(self):
		"""Get length of encoding scheme"""

		return self.input_size

	def encode(self, seq):
		"""Convert an amino acid sequence into this encoding scheme"""

		l = len(seq)
		m = np.zeros((l, self.input_size))

		try:
			for i in range(l):
				m[i] = self.encode_dict[seq[i]]
		except:
			print('Error: invalid amino acid detected:', seq[i])
			sys.exit()
		return torch.from_numpy(m)
