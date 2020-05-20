# prot-brnn

## Introduction

This package is designed to be useful for a variety of protein bioinformatic applications. prot-brnn encodes a 
computationally-robust bidirectional recurrent neural network (BRNN) behind an easy-to-use commandline interface. With only an 
input datafile containg sequences and mapped values, the user can automatically train a BRNN for their purposes. This trained
network can then be applied to new, unlabeled data to generate predictions and generate biological hypotheses.

This package can handle regression and classification ML problems, as well as sequence-mapped and residue-mapped input data.

## Usage: (as of May 20, 2020)

There are three primary actions that can be done within the prot-brnn package. Each of these are described in more detail
below.
1. Train a BRNN with user-specified hyperparameters
2. Train a BRNN with unspecified, automatically-determined, optimal hyperparameters
3. Generate predictions on unlabeled sequences using a trained BRNN

### 1. Train BRNN with provided hyperparameters: run_prot_brnn.py

< Describe why you would want to do this. Preliminary data exploration. User has intuition with ML. >  
Format data in the following manner:

seqID1 seq1 seq1data1 <seq1data2> <seq1data3> ... <seq1dataN1>  
seqID2 seq2 seq2data1 <seq2data2> <seq2data3> ... <seq2dataN2>  
...  
seqIDM seqM seqMdata1 <seqMdata2> <seqMdata3> ... <seqMdataNM>
  
Where Ni is the length of sequence i, and M is the total number of labeled sequences. Items must be whitespace-separated.
For sequence-mapped data (i.e. each sequence constitutes a *single datapoint*), each row will only contain three columns. For 
residue-mapped data (i.e. a sequence of N amino acids consists of *N datapoints*), each row will contain N + 2 columns. Please
note that it is not required that sequences are the same length. For example, if Sequence #1 has 12 amino acids and Sequence
#2 has 15 amino acids, then these two rows in the input file will contain 14 and 17 fields respectively.

Optionally, you may use datasets that exclude the first column containing the ID of each sequence. In this case, be sure to use the `--excludeSeqID` flag.

#### Classification problem
If the user wishes to categorize sequences or residues into distinct classes, then the labeled data should be integer class 
labels. For example, if there are 3 classes, then each datapoint should be either a '0', '1', or '2' (with no quote marks). 
For example datasets, look at those provided in the **datasets** folder.

Run the following command to train a network on the classification problem:

    python run_prot_brnn.py <./path/to/dataset.tsv> <./path/to/outputNetwork.pt> --datatype <sequence / residues> --stop iter **-nc <num of classes (int>1)>** -e <num of epochs (int>1)> -nl <num layers in the network (int>1)> -hs <hidden vector size (int>1)> -b <batch size> -lr <learning rate>
  
#### Regression problem
If the user wishes to map each sequence or residue to a continuous real number, then labeled data should be a float. Run using
the following command:

    python run_prot_brnn.py <./path/to/dataset.tsv> <./path/to/outputNetwork.pt> --datatype <sequence / residues> --stop iter **nc 1** -e <num of epochs> -nl <num of layers in the network> -hs <hidden vector size> -b <batch size> -lr <learning rate>

### 2. Train BRNN following hyperparameter optimization -- hyperparameter_optimization.py
Work in progress...

### 3. Generate predictions with trained BRNN -- predict_seqs.py
< Describe sequence format >  
< Generate hypotheses >

## TO DO: 
- Implement automatic hyperparameter search
- Document code better
- Test prot-brnn on a wide range of use-cases
- Package scripts into PyPI module (using cookiecutter)
