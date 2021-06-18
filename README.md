![PARROT_logo_full](https://user-images.githubusercontent.com/54642153/122615183-b274f280-d04d-11eb-88bf-1530d18d310c.png)

# PARROT: Protein Analysis using RecuRrent neural networks On Training data

**PARROT** encodes a computationally-robust bidirectional recurrent neural network (BRNN) behind an easy-to-use commandline interface. PARROT is well-suited for a variety of protein bioinformatics tasks. With only an input datafile containing sequences and mapped values, the user can automatically train a BRNN for whatever purpose. This trained network can then be applied to new, unlabeled data to generate predictions and generate biological hypotheses.

This package can handle regression and classification ML problems, as well as sequence-mapped and residue-mapped input data.

## Installation:

PARROT is available through GitHub or the Python Package Index (PyPI). To install through PyPI, run

	$ pip install idptools-parrot

It is possible that you may experience errors depending on what Python packages are already installed on your machine. If you encounter this, try installing PARROT in a "clean" virtual environment using conda:

	$ conda create --name <env_name> python=3.7
	$ conda activate <env_name>

Then install PARROT with pip.

Alternatively, to clone the GitHub repository and gain the ability to modify a local copy of the code, run:

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

## Usage:

There are three primary commands that can be run within the parrot package. Each of these are briefly described below and
for more information on their usage, visit their individual [documentation pages](https://idptools-parrot.readthedocs.io/en/latest/).

1. Train a BRNN with user-specified hyperparameters
2. Train a BRNN with automatically-determined, optimal hyperparameters
3. Generate predictions on unlabeled sequences using a trained BRNN

### Input data format:

Before data can be integrated into training a BRNN, it must be formatted in the following manner:

	seqID1 seq1 seq1data1 <seq1data2> <seq1data3> ... <seq1dataN1>  
	seqID2 seq2 seq2data1 <seq2data2> <seq2data3> ... <seq2dataN2>  
	.
	.
	.  
	seqIDM seqM seqMdata1 <seqMdata2> <seqMdata3> ... <seqMdataNM>
  
Where Ni is the length of sequence i, and M is the total number of labeled sequences. Items must be whitespace-separated.
For **sequence-mapped data** (i.e. each sequence constitutes a *single datapoint*), each row will only contain three columns.
Note that it is not required that sequences are the same length. For example, if Sequence #1 has 12 amino acids and Sequence #2
has 15 amino acids, then these two rows in the input file will contain 14 and 17 fields respectively.

Optionally, you may use datasets that exclude the first column containing the ID of each sequence. In this case, be sure to 
use the `--excludeSeqID` flag.

**Classification problem:** the labeled data should be integer class labels. For example, if there are 3 classes, then each
datapoint should be either a '0', '1', or '2' (with no quote marks).
  
**Regression problem:** If the user wishes to map each sequence or residue to a continuous real number, then each datapoint 
should be a float

For example datasets, see the TSV files provided in the **data** folder.

### 1. Train BRNN with provided hyperparameters: ``parrot-train``

The ``parrot-train`` command is most useful in the initial stages of data exploration. This command requires the user to 
specify the hyperparameters to train the network, so it may not achieve the optimal results compared to more extensive training
and hyperparameter search. However, if one wishes to quickly train a network for a given task, this command will give a sense
of how effective a BRNN will be. Running ``brnn_train`` on a dataset for a large number of epochs can inform for how many epochs
to train for during the more extensive hyperparameter optimization.

### 2. Optimize hyperparameters and train BRNN: ``parrot-optimize``

The ``parrot-optimize`` command initiates an extensive search for the best-performing network hyperparameters for a given
dataset using Bayesian optimiztion. Three hyperparameters, the learning rate, number of hidden layers, and hidden vector size
can greatly impact network performance and training speed, so it is important to tune these for each particular dataset. This
command will search across hyperparameter space by iteratively training and validating network performance (with 5-fold cross
validation). The best performing hyperparameters will be selected, and used to train a network from scratch as if running
``brnn_train`` with these parameters.

### 3. Generate predictions with trained BRNN: ``parrot-predict``

Once a network has been trained for a particular machine learning task, the user can generate predictions on new sequences
with this network using the ``parrot-predict`` command. The user provides a list of sequences they would like to predict and
the saved network, and a file is outputted with the predictions.

### Copyright

Copyright (c) 2020, Holehouse Lab

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
