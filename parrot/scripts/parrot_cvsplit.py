#!/usr/bin/env python
"""
Usage: $ parrot-cvsplit data_file output_splitfiles <flags>
  
Driver script for creating PARROT-usable "split-files" for K-fold cross-validation
on a given dataset. Assumes datafile is already properly formatted for use in PARROT.
For more information on usage, use the '-h' flag.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import os
import sys

import numpy as np
import argparse

from parrot import process_input_data as pid
from parrot.tools import validate_args

# Parse the command line arguments
def main():
     parser = argparse.ArgumentParser(description='Generate K split-files for K-fold cross-validation')
     parser.add_argument('data_file', help='Path to tsv file with format: <idx> <sequence> <data>')
     parser.add_argument('output_splitfiles', 
          help='Location and name to save the split-files as "<output_splitfiles>_cv#.txt" for each of k-folds')
     parser.add_argument('-k', '--k-folds', default=10, type=int, metavar='K', dest='num_folds',
                         help='number of CV folds (def=10)')
     parser.add_argument('-t', '--training-fraction', default=0.8, type=float, dest='percent_train',
          help='Fraction of the non-test set data to use as training data (def=0.8)')
     args = parser.parse_args()

     # Ensure that provided data file exists, K > 1, and 0 < fraction < 1
     data_file = validate_args.check_file_exists(args.data_file, 'Datafile')
     validate_args.check_between_zero_and_one(args.percent_train, 'Training fraction')
     if args.num_folds <= 1:
          raise ValueError(f'Must have >1 CV folds. Provided: {args.num_folds}.')

     # Extract output directory and output prediction file name
     output = os.path.abspath(args.output_splitfiles)
     prefix, output_dir = validate_args.split_file_and_directory(output)

     # Check total number of sequences
     with open(data_file) as d:
          lines = [line.strip().split() for line in d]
     n_seqs = len(lines)

     # Split dataset randomly into K even chunks
     all_samples = np.arange(n_seqs)
     reference = np.copy(all_samples)
     np.random.shuffle(all_samples)
     test_sets = np.array_split(all_samples, args.num_folds)

     # Get train & validation sets corresponding to each test set
     train_sets = []
     val_sets = []
     for i in range(args.num_folds):
          remainders = test_sets[:i] + test_sets[i+1:]
          train_val_set = np.hstack(remainders)
          train, val = pid.vector_split(train_val_set, args.percent_train)
          train_sets.append(np.sort(train))
          val_sets.append(np.sort(val))

     # Write split-files to specified location
     print('Creating split-files:')
     print('----------------------')
     for i in range(args.num_folds):
          outfile_name = prefix + '_cv' + str(i) + '.txt'
          print(outfile_name)

          with open(outfile_name, 'w') as f:
               f.write(" ".join(train_sets[i].astype(str)) + '\n')
               f.write(" ".join(val_sets[i].astype(str)) + '\n')
               f.write(" ".join(np.sort(test_sets[i]).astype(str)) + '\n')
