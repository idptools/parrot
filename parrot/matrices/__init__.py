"""
Matrix data: one matrix of values per sequence, with one row per residue.

A matrix task ('matrix' datatype) predicts, for each sequence, either a square
map of every pair of residues (such as a contact or distance map) or a matrix
with a fixed number of related, ordered columns per residue (such as bins or
time points). Everything specific to matrix data lives in this package:

- data.py: reading the matrices, checking them against their sequences, and
  padding them into batches

The rest of PARROT (encodings, training, checkpoints, the hyperparameter
search) is shared with sequence and residue data.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""
