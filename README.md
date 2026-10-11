![PARROT_logo_full](https://user-images.githubusercontent.com/54642153/122615183-b274f280-d04d-11eb-88bf-1530d18d310c.png)

# PARROT: Protein Analysis using RecuRrent neural networks On Training data

**PARROT** encodes a computationally-robust bidirectional recurrent neural network (BRNN) behind an easy-to-use commandline interface. PARROT is well-suited for a variety of protein bioinformatics tasks. With only an input data file containing sequences and mapped values, the user can automatically train a network for whatever purpose. This trained network can then be applied to new, unlabeled data to generate predictions and generate biological hypotheses.

This package can handle regression and classification ML problems, as well as sequence-mapped and residue-mapped input data.

#### Version 1.7.6 [Ma 2025]
## Installation:

PARROT is available through GitHub or the Python Package Index (PyPI). To install the base version of PARROT through PyPI, run

	$ pip install idptools-parrot

It is possible that you may experience errors depending on what Python packages are already installed on your machine. If you encounter this, try installing PARROT in a "clean" virtual environment using conda:

	$ conda create --name <env_name> python=3.11
	$ conda activate <env_name>

Then install PARROT with pip.

Alternatively, to clone the GitHub repository and gain the ability to modify a local copy of the code, run:

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

**Hyperparameter search:** `parrot-optimize` needs no extra packages. (Earlier versions needed GPy and GPyOpt, installed with `pip install idptools-parrot[optimize]`; that command still works, but installs nothing extra.)

## Usage:

For detailed information on installation and examples, please visit the [documentation pages](https://idptools-parrot.readthedocs.io/en/latest/).

Every command also explains itself: `-h` lists its options, and `--show-examples` prints a guide with worked examples of how to use it, what it does and how it works (e.g. `parrot-train --show-examples`).

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

Optionally, you may use datasets that exclude the first column containing the ID of each sequence. In this case, be sure to use the `--exclude-seq-id` flag.

**Classification problem:** the labeled data should be integer class labels. For example, if there are 3 classes, then each datapoint should be either a '0', '1', or '2' (with no quotation marks).
  
**Regression problem:** If the user wishes to map each sequence or residue to a continuous real number, then each datapoint should be a float

**Missing residue labels:** for residue-mapped data, a residue with no label (e.g. a residue with no experimental measurement) can be written as `nan`. PARROT ignores these residues in training, in performance metrics and in figures, and writes `nan` for their true value in prediction output. Every sequence needs at least one labelled residue, and sequence-mapped data cannot contain missing values.

**Matrix data:** PARROT can also learn a matrix of values per sequence, with one row per residue: a map of every pair of residues (L x L), or N related values per residue (L x N). The datafile then holds `<ID> <sequence>` on each line, and the matrices are in a NumPy `.npz` file, one array per sequence ID (`parrot-train unet2d --input-file data.tsv --output-network network.pt --split kmer --regression --matrix-file matrices.npz`). See the "Matrix data" documentation page.

For example, datasets, see the TSV files provided in the **/data** folder.

### Preparing data: ``parrot-preprocess``

``parrot-preprocess`` turns a PARROT datafile into one ready for training. It can clean sequences, recode class labels, merge or remove duplicate sequences, and split the data into training, validation and test sets. ``--split homology`` uses MMseqs2 (installed separately, e.g. `conda install -c conda-forge -c bioconda mmseqs2`) with stringent criteria: any alignment with E <= 1e-3 covering at least 30% of the shorter sequence, at any identity. It keeps homologous sequences in the same set, then re-searches each set against the others until none shares a homologue with another. Without MMseqs2, ``--split kmer`` follows the same protocol with PARROT's own k-mer search: it needs nothing extra and removes leakage between close relatives (in the yeast proteome, all test-training pairs at >= 90% identity and nearly all at 50-90%), but not between distant homologues. It can also balance the classes of the training set. It writes the processed datafile, a split-file to train on with `parrot-train --split`, and a report:

```
parrot-preprocess data.tsv clean.tsv -d sequence -m classification --deduplicate merge --split homology --class-balance --seed 1
parrot-train brnn --input-file clean.tsv --output-network network.pt -d sequence --classification --num-classes 3 --split clean_split_file.txt
```

### 1. Train a network with provided hyperparameters: ``parrot-train``

The ``parrot-train`` command is the primary command for training a network with PARROT. Users choose a network architecture, then need only to specify their data, where they want to save their output, and some basic information on the type of machine learning problem they are tackling:

```
parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression
```

Each architecture (`brnn`, `cnn`, `unet`, `transformer`, `composition`, the simple models `linear`, `svm`, `random_forest` and `knn`, and `unet2d` for matrix data) is a subcommand with only the options that apply to it: `parrot-train -h` lists them, and `parrot-train cnn -h` shows the options of one. Every input is named: `--input-file` for the datafile, `--output-network` for the network (`.pt` is added if needed), and `--regression` or `--classification --num-classes N` for the task. Next to the network, `<network>_info.txt` records how it was trained, for reproducing it or for a paper's methods. Beyond this, there are a suite of other options that users can provide to tailor their network to their particular needs. Users can provide specific hyperparameters to craft their network, they can manually specify what samples in their data set are trained on and which are held out as test data, they can choose to output information and figures on their network's performance on the test data, and much more!

### 2. Optimize hyperparameters and train a network: ``parrot-optimize``

The ``parrot-optimize`` command searches for the best-performing hyperparameters of an architecture on a given dataset (e.g. ``parrot-optimize cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression``), then trains and tests a network with them, as ``parrot-train`` would. It tunes the learning rate and every hyperparameter the architecture marks as searched (for the BRNN, the number of layers and the hidden vector size), dropping poor configurations after a few epochs (successive halving) and proposing new ones where good results have clustered (TPE). Any hyperparameter given on the command line is fixed rather than searched. It takes considerably longer than ``parrot-train``: the default 50 trials cost about as much as 15-20 training runs.

### 3. Generate predictions with a trained network: ``parrot-predict``

Once a network has been trained for a particular machine learning task, the user can generate predictions on new sequences with this network using the ``parrot-predict`` command. The user provides a list of sequences they would like to predict and the saved network, and a file is outputted with the predictions.

### GPUs (CUDA and Apple silicon)

Every command that trains or predicts takes `--device`, which is `auto` by default. This uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Use `--device cpu`, `--device cuda` or `--device mps` to choose explicitly; asking for a device that is not available is an error. Trained networks are always saved with CPU tensors, so a network trained on any device can be loaded anywhere, and PARROT computes in full float32 precision on every device (on NVIDIA GPUs it turns off TF32 while it runs), so a network gives the same predictions on any device to about 1e-4 or better. In Python, `Predictor(path, device='auto')` runs predictions on a GPU; the default is the CPU, which is usually fastest for a few sequences at a time.

### Encodings

Sequences are encoded with one-hot vectors by default. `--encode biophysics` uses nine biophysical scales instead, and `--encode <file>` uses your own table (see `data/encoding_example.txt`). `--standardize-encoding` rescales every feature of the encoding to mean 0 and standard deviation 1 across the amino acids, so that features measured on very different scales (e.g. molecular weight and an aromatic flag) contribute equally; it is recommended for `biophysics` and most user encodings. The standardization is saved with the network and applied automatically when predicting. See the `parrot-train` documentation for exactly what is done.

`--encode esm2_t12_35M_UR50D` (or another ESM-2 model, which needs `pip install idptools-parrot[plm]`) or `--encode esmc_300m` / `esmc_600m` (EvolutionaryScale's ESM C, which needs `pip install idptools-parrot[esmc]`) encodes each residue by its embedding from a protein language model. Embedding is by far the slowest step, so `parrot-train` and `parrot-predict` embed each sequence once and store the embeddings in a cache directory next to their output (`parrot_train_embeddings` or `parrot_predict_embeddings`, each with a README), after printing how large the cache will be; `--embedding-cache` lets several runs share one cache. See "Protein language model encodings" in the documentation.

For sequence-mapped data, `--encode nardini` gives every residue its sequence's 90 NARDINI+ features (36 patterning z-scores, each against scrambles of the sequence itself, and 54 compositional z-scores against the human IDRome), as computed by SPARROW (`pip install git+https://github.com/idptools/sparrow.git`). Alone, with the `composition` baseline or the simple models, this is a model of the NARDINI+ features and the length; `--encode onehot+nardini` (or `biophysics+nardini`) puts them after each residue's own encoding, for the order-aware networks. The features are computed once, in parallel, into the embedding cache; `--nardini-scrambles` sets the number of scrambles (default 1,000). See the "NARDINI features" documentation page.

### Batching

By default (`--batching length`), training batches group sequences of similar length, re-randomised every epoch, so little computation is spent on padding; for datasets whose sequence lengths vary widely this makes training several-fold faster. Padding never affects a network's output, so this only changes which sequences share a batch. `--batching random` restores the freely shuffled batches of earlier PARROT versions.

### Copyright

Copyright (c) 2020-2025, Holehouse Lab

### Change log 

* Unreleased: Bug-fix release. PARROT now requires Python 3.11 or newer, NumPy 2.3 or newer and PyTorch 2.8 or newer. Zero-padding is now masked during training (the LSTM only sees each sequence's real residues and padded residues no longer contribute to the loss), so predictions no longer depend on how sequences are batched; previously trained networks load and predict exactly as before. Fixed a crash when testing residue-classification networks with NumPy >= 1.24. The reported test loss now uses the same loss function as training/validation. `parrot-predict` writes one output line per input line (keeping every sequence ID) and skips blank/comment lines. Added a `--seed` flag to `parrot-train`, `parrot-optimize`, `parrot-cvsplit` and `parrot-preprocess`. Fixed `--cb-shuffle-off` in `parrot-preprocess` (it previously did the opposite), plus a number of smaller input-validation and plotting fixes. Added an extensive test suite. Restructured PARROT so new network architectures can be added as a single module (see the "Adding a new architecture" documentation page): `parrot-train` and `parrot-optimize` take an `--architecture` flag (default `brnn`), and saved networks now record their architecture, hyperparameters, datatype, number of classes and encoding, so `parrot-predict` and `Predictor` no longer need these to be specified. Networks saved by earlier versions still load, and `BRNN_MtO`/`BRNN_MtM` are unchanged. Added four architectures alongside the BRNN: `cnn` (a dilated residual convolutional network), `unet` (a 1D U-Net adapted from the 2D U-Net on the parrot-lightning branch), `transformer` (a transformer encoder with rotary or sinusoidal positions) and `composition` (an order-independent baseline). By default the CNN and U-Net add a whole-sequence summary to every position, the BRNN, CNN, U-Net and transformer pool residues with learned attention (the BRNN previously used its final hidden states), and the BRNN, CNN, U-Net and transformer give their output layer log(sequence length) on sequence tasks; `--global-context-off`, `--attention-pooling-off` and `--loglength-feedforward-off` turn these off, and networks saved before these options existed load with them off. `parrot-optimize` now accepts architecture hyperparameter flags to fix those hyperparameters instead of searching them. `parrot-optimize` was rewritten without GPy/GPyOpt: it combines successive halving (configurations are dropped after a few epochs unless they are among the best) with model-based sampling (TPE), scores configurations on the validation set by default (`--cv-folds` for cross-validation), searches settings with fixed choices (e.g. kernel widths) as well as numbers, records every trial in `<output>_hyperparameter_trials.tsv`, and is about 15-20 times cheaper than before with its defaults; `--trials` replaces `--max-iter`, which still works. Added simple scikit-learn models for sequence-mapped data (`linear`, `svm`, `random_forest`, `knn`), fitted in one step to composition, length and optionally dipeptide-composition features. `parrot-preprocess` became a complete data-preparation tool: sequence cleaning (`--uppercase`, `--remove-nonstandard`, `--min-length`, `--max-length`), deduplication (`--deduplicate merge|first|drop`), splitting into training, validation and test sets with a matching split-file (`--split homology`, which uses MMseqs2 so that no set contains a homologue of another; `--split kmer`, the same protocol with PARROT's own k-mer search when MMseqs2 is not installed, which finds closer relatives only; or `--split random`; plus `--groups`), class balancing of the training set only, and a report of every step. Class balancing removes the fewest sequences needed for the best balance (sequence data), never makes the balance worse (residue data), and keeps the input order; `--cb-shuffle-off` no longer has any effect. Datafile errors now name the data line and say what is wrong with it. `parrot-preprocess --remove-line-duplicates` keeps every line in its original order (it used to regroup lines that share an ID). `parrot-cvsplit` makes its folds with the same code as `parrot-train`, so a given `--seed` gives different (equally valid) split-files than before. A `--datatype`, `--mode` or `--stop` value that is not one of the allowed choices now stops with a usage message. Figures are drawn with matplotlib alone in a plain style (Arial text that stays editable in vector editors, solid lines, no titles or grid lines), so pandas and seaborn are no longer needed. The unused `decode()`, `rev_one_hot()` and `rev_biophysics()` functions have been removed, and the `parrot` package now exports only its public names. Added matrix data: a matrix of values per sequence with one row per residue, either a square map of every pair of residues (optionally `--symmetric`) or `--matrix-width N` related values per residue, read from a NumPy `.npz` file (`--matrix-file`) with missing cells as NaN, for regression or classification. Matrix data is learnt by the new `unet2d` architecture, a 2D U-Net after the one written by Nick on the parrot-lightning branch. Predicted matrices are written as `.npz` files by `parrot-train`, `parrot-optimize` and `parrot-predict`, with heatmaps of example test sequences in the figures; `parrot-preprocess` cleans, deduplicates and splits matrix datafiles; and `--max-matrix-length` (default 512) refuses sequences whose square maps would not fit in memory. `parrot-train` and `parrot-optimize` now take the architecture as a subcommand (`parrot-train cnn --input-file data.tsv --output-network network.pt -d residues --regression`), and each subcommand offers only the options that apply to that architecture: its own hyperparameters, the training options only for networks trained epoch by epoch, the matrix options only for `unet2d`, and `-d` only where there is a choice of datatype. They name every input, with no positional arguments: `--input-file` and `--output-network` (`.pt` is added if the name does not end with it); `--regression`, or `--classification` with `--num-classes N`, replace `-c`; and `--device cpu` replaces `--force-cpu`. Commands in the earlier form stop with a message saying what to use instead. Every network gets a `<network>_info.txt` describing how it was trained (the exact command and software versions, the data with its checksum, the split, the encoding, the architecture with its number of parameters, the loss, optimizer and epochs, the search for `parrot-optimize`, and the results), in enough detail for a paper's methods section. `--split` is now required, and is either a split-file or one of `parrot-preprocess`'s splitting methods: `mmseqs` or `kmer`, which keep similar sequences in the same set (found by MMseqs2, or by PARROT's own k-mer search) so that no test sequence has a close relative in the training set, or `random`. The methods aim for `--set-fractions`, and a file with the name given is always read first (so a split-file called `random` is still read as a file). `--split random` is closest to the old default, but now keeps identical sequences in the same set. With `--include-figs`, regression networks now get a density plot of predicted against true test values as well as the scatterplot (which becomes a solid block with many values), both labelled with Pearson's r, and residue networks get `_res_profiles.png`, the true and predicted values along six test sequences spanning the range of accuracy, each labelled with its Pearson's r and mean absolute error (or, for classification, its accuracy). `_info.txt` now also records how long training took, the time per epoch, the training speed and the prediction speed (sequences per second). `--probabilistic-classification` now also works for residue classification: each residue's class probabilities are written as one comma-separated token (e.g. `0.91,0.06,0.03`), so tokens still line up with residues, and the ROC and precision-recall curves and their areas pool every labelled residue. Added NARDINI+ sequence features as an encoding for sequence-mapped data (`--encode nardini`, or `onehot+nardini` to follow each residue's own encoding), computed by SPARROW once per sequence, in parallel, into the embedding cache, with the settings saved with the network so new sequences are encoded identically. The commands report mistakes in their input as a one-line message rather than a traceback (`PARROT_TRACEBACK=1` shows the traceback too), start with a line giving PARROT's version, and say more about what they do: `parrot-train` and `parrot-optimize` report how the data were split, number epochs from 1 (always showing the last), say which epoch's network was kept, and end with the test-set metrics and the files written; `parrot-predict` prints its settings only with the new `--verbose`. Error messages name the file or option at fault and how to fix it (e.g. `-c` for out-of-range class labels), and the help of every command was revised for accuracy and consistency. Every command now starts with a header giving PARROT's version and the time, and has a `--show-examples` option that prints a guide to the command: what it does, worked examples, and how it works.
* Version 1.7.6 [May 2025]: Updated the build process to use pyproject.toml and versioningit for version updates, so PARROT is now compatible with Python 3.13+.
* Version 1.7.2: Updated Python dependency so PARROT is compatible with Python 3.8, 3.9, and 3.10. 

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
