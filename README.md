![PARROT_logo_full](https://user-images.githubusercontent.com/54642153/122615183-b274f280-d04d-11eb-88bf-1530d18d310c.png)

# PARROT: Protein Analysis using RecuRrent neural networks On Training data

**PARROT** encodes a computationally-robust bidirectional recurrent neural network (BRNN) behind an easy-to-use commandline interface. PARROT is well-suited for a variety of protein bioinformatics tasks. With only an input data file containing sequences and mapped values, the user can automatically train a network for whatever purpose. This trained network can then be applied to new, unlabeled data to generate predictions and generate biological hypotheses.

This package can handle regression and classification ML problems, as well as sequence-mapped and residue-mapped input data.

## Installation:

PARROT is available through GitHub or the Python Package Index (PyPI). To install the base version of PARROT through PyPI, run

	$ pip install idptools-parrot

It is possible that you may experience errors depending on what Python packages are already installed on your machine. If you encounter this, try installing PARROT in a "clean" virtual environment using conda:

	$ conda create --name <env_name> python=3.9
	$ conda activate <env_name>

Then install PARROT with pip.

Alternatively, to clone the GitHub repository and gain the ability to modify a local copy of the code, run:

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

**JULY 2022 UPDATE**

To mitigate package version dependency issues involving Python, GPy, and PyTorch, new releases of PARROT have separated 
`parrot-optimize` as an optional add-on installation. All of the documentation for using `parrot-optimize` is unchanged.
However, if you wish to use the hyperparameter optimization, there are now slight differences in how you install PARROT.

To install the PARROT that is compatible with `parrot-optimize` install via pip using:

	$ pip install idptools-parrot[optimize]

or

	$ pip install "idptools-parrot[optimize]"

Alternatively if you have the PARROT repository cloned locally you can install using

	$ pip install ".[optimize]"


## Usage:

For detailed information on installation and examples, please visit the [documentation pages](https://idptools-parrot.readthedocs.io/en/latest/).

There are three primary commands that can be run within the parrot package:

1. Train a network with user-specified hyperparameters
2. Train a network with automatically-determined, optimal hyperparameters
3. Generate predictions on unlabeled sequences using a trained network

### Input data format:

Before data can be integrated into training a PARROT network, it must be formatted in the following manner:

	seqID1 seq1 seq1data1 <seq1data2> <seq1data3> ... <seq1dataN1>  
	seqID2 seq2 seq2data1 <seq2data2> <seq2data3> ... <seq2dataN2>  
	.
	.
	.  
	seqIDM seqM seqMdata1 <seqMdata2> <seqMdata3> ... <seqMdataNM>
  
Where Ni is the length of sequence i, and M is the total number of labeled sequences. Items must be whitespace-separated.
For **sequence-mapped data** (i.e. each sequence constitutes a *single datapoint*), each row will only contain three columns.

Note that it is not required that sequences are the same length. For example, if Sequence #1 has 12 amino acids and Sequence #2 has 15 amino acids, these two rows in the input file will contain 14 and 17 fields, respectively.

Optionally, you may use datasets that exclude the first column containing the ID of each sequence. In this case, be sure to use the `--excludeSeqID` flag.

**Classification problem:** the labeled data should be integer class labels. For example, if there are 3 classes, then each datapoint should be either a '0', '1', or '2' (with no quotation marks).
  
**Regression problem:** If the user wishes to map each sequence or residue to a continuous real number, then each datapoint should be a float

For example, datasets, see the TSV files provided in the **/data** folder.

### 1. Train a network with provided hyperparameters: ``parrot-train``

The ``parrot-train`` command is the primary command for training a network with PARROT. By default, users need only to specify their data, where they want to save their output, and some basic information on the type of machine learning problem they are tackling. Beyond this, there are a suite of other options that users can provide to tailor their network to their particular needs. Users can provide specific hyperparameters to craft their network, they can manually specify what samples in their data set are trained on and which are held out as test data, they can choose to output information and figures on their network's performance on the test data, and much more!

### 2. Optimize hyperparameters and train a network: ``parrot-optimize``

The ``parrot-optimize`` command initiates an extensive search for the best-performing network hyperparameters for a given dataset using Bayesian optimization. Three hyperparameters, the learning rate, the number of hidden layers, and hidden vector size can greatly impact network performance and training speed, so it is important to tune these for each particular dataset. This command will search across hyperparameter space by iteratively training and validating network performance (with 5-fold cross-validation). The best-performing hyperparameters will be selected and used to train a network from scratch as if running ``parrot_train`` with these parameters. Note that since this command takes a significant amount of time to run since it involvessearching for the best hyperparameters.

### 3. Generate predictions with a trained network: ``parrot-predict``

Once a network has been trained for a particular machine learning task, the user can generate predictions on new sequences with this network using the ``parrot-predict`` command. The user provides a list of sequences they would like to predict and the saved network, and a file is outputted with the predictions.

### Copyright

Copyright (c) 2020-2023, Holehouse Lab

### Change log 

* Version 1.7.2: Updated Python dependency so PARROT is compatible with Python 3.8, 3.9, and 3.10. 

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
