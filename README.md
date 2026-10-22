![PARROT_logo_full](https://user-images.githubusercontent.com/54642153/122615183-b274f280-d04d-11eb-88bf-1530d18d310c.png)

# PARROT2: in development

> **PARROT2 (version 2.0.0.dev0) is in active development on the `parrot2` branch and has not yet been released.** `pip install idptools-parrot` installs the latest release, PARROT 1.7.6 (May 2025), and the [online documentation](https://idptools-parrot.readthedocs.io/en/latest/) describes PARROT 1.7. To use PARROT2, install it from this branch (see below). Commands, options, and file formats may still change before it is released.

**PARROT** trains machine-learning models that map protein sequences to values, behind an easy-to-use command-line interface. PARROT is well-suited for a variety of protein bioinformatics tasks. With only an input data file containing sequences and their values, you can train, optimize, and test a network for whatever purpose, then apply it to new, unlabeled sequences to generate predictions and biological hypotheses.

PARROT handles regression and classification problems, with one value per sequence (sequence-mapped data), one per residue (residue-mapped data) or a matrix per sequence (matrix data, e.g. a contact map). The networks can be bidirectional recurrent neural networks (BRNNs, PARROT's original architecture), convolutional networks, U-Nets or transformers, and baselines built on sequence composition (a composition network and simple scikit-learn models) show how much a network gains from residue order. Sequences can be encoded with one-hot vectors, biophysical scales, your own table, protein language model embeddings or NARDINI+ features.

## What's new in PARROT2

PARROT2 is a substantial rewrite of PARROT 1. In brief:

* **Architectures:** `parrot-train` and `parrot-optimize` take the architecture as a subcommand: `brnn`, `cnn`, `unet`, `transformer`, and `composition`; the simple models `linear`, `svm`, `random_forest`, `knn`, and `gbrt`; and `unet2d` for matrix data. You can add new architectures as a single module.
* **Data:** matrix data, missing residue labels (NaN), regression values with uncertainties (`--uncertainties`, which weights each value by its measurement error), and parrot-preprocess, which cleans and deduplicates a data file, splits it into training, validation, and test sets without leakage between them (keeping similar sequences, found with MMseqs2 or PARROT's own k-mer search, in the same set), and balances classes.
* **Encodings:** protein language model embeddings (ESM-2 and ESM-C) and NARDINI+ features, computed once and cached, and standardized encodings.
* **Hyperparameter search:** `parrot-optimize` was rewritten without GPy/GPyOpt, combining successive halving with model-based sampling (TPE); with its defaults it costs about 15-20 times less than before.
* **Training:** Apple-silicon GPUs as well as NVIDIA ones (`--device`, and `--gpu-id` to choose among several NVIDIA GPUs), giving the same predictions on every device to about 1e-4, batching by sequence length, masked padding (a network's output depends on how sequences are batched only through float32 rounding), reproducible runs with `--seed`, and a `<network>_info.txt` describing how each network was trained, in enough detail for a paper's methods section.
* **Saved networks** record their architecture, hyperparameters, task, and encoding, so `parrot-predict` and `Predictor` need nothing else. Networks saved by PARROT 1 still load and predict as before, up to float32 rounding (as they recorded only their weights, they still need their datatype: `-d` for `parrot-predict`, with `--encode` if it was not one-hot, and `dtype` for `Predictor`).
* **Command line:** every input is named (`--input-file`, `--output-network`, `--regression` or `--classification --num-classes N`), and `--split` is required. Commands in PARROT 1 form stop with a message telling you what to use instead.
* **Requirements:** Python 3.11 or newer, NumPy 2.3 or newer, PyTorch 2.8 or newer, scikit-learn 1.4 or newer, and h5py 3.10 or newer.

[CHANGELOG.md](CHANGELOG.md) lists every change.

## Installation:

PARROT2 is installed from the `parrot2` branch on GitHub:

	$ pip install "git+https://github.com/idptools/parrot.git@parrot2"

It is possible that you may experience errors depending on what Python packages are already installed on your machine. If you encounter this, try installing PARROT in a "clean" virtual environment using conda:

	$ conda create --name <env_name> python=3.11
	$ conda activate <env_name>

Then install PARROT with pip.

Alternatively, to clone the GitHub repository and gain the ability to modify a local copy of the code, run:

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ git checkout parrot2
	$ pip install -e .

This installs PARROT in editable mode, so changes you make to the source code take effect without reinstalling.

**Optional extras:** protein language model encodings need the `plm` extra (ESM-2) or the `esmc` extra (ESM C), and NARDINI+ features need SPARROW (`pip install git+https://github.com/idptools/sparrow.git`). From a clone, install an extra with `pip install -e ".[plm]"`; directly from GitHub, use `pip install "idptools-parrot[plm] @ git+https://github.com/idptools/parrot.git@parrot2"`. (`pip install idptools-parrot[plm]` would install PARROT 1.7.5 from PyPI.) `--split homology` (`parrot-preprocess`) and `--split mmseqs` (`parrot-train` and `parrot-optimize`) need MMseqs2, installed separately (e.g. `conda install -c conda-forge -c bioconda mmseqs2`).

**Hyperparameter search:** `parrot-optimize` needs no extra packages. (Earlier versions needed GPy and GPyOpt, installed with `pip install idptools-parrot[optimize]`; that command still works, but installs nothing extra.)

## Usage:

The documentation for PARROT2 is in the `docs` folder of this branch, with detailed information on installation and examples; `docs/README.md` explains how to build it. The [online documentation](https://idptools-parrot.readthedocs.io/en/latest/) describes PARROT 1.7.

Every command also explains itself: `-h` lists its options, and `--show-examples` prints a guide with worked examples of how to use it, what it does, and how it works (e.g., `parrot-train --show-examples`).

There are three primary commands that can be run within the parrot package:

1. Train a network with user-specified hyperparameters
2. Train a network with automatically-determined, optimal hyperparameters
3. Generate predictions on unlabeled sequences using a trained network

Two more commands help prepare data for them: `parrot-preprocess` cleans a data file and splits it into training, validation, and test sets, and parrot-cvsplit writes split files for k-fold cross-validation.

### Input data format:

Before data can be integrated into training a PARROT network, it must be formatted in the following manner:

	seqID1 seq1 seq1data1 <seq1data2> <seq1data3> ... <seq1dataN1>  
	seqID2 seq2 seq2data1 <seq2data2> <seq2data3> ... <seq2dataN2>  
	.
	.
	.  
	seqIDM seqM seqMdata1 <seqMdata2> <seqMdata3> ... <seqMdataNM>

Where Ni is the length of sequence i, and M is the total number of labeled sequences. Items must be whitespace-separated. For **sequence-mapped data** (i.e. each sequence constitutes a *single datapoint*), each row will only contain three columns.

Note that it is not required that sequences are the same length. For example, if Sequence #1 has 12 amino acids and Sequence #2 has 15 amino acids, these two rows in the input file will contain 14 and 17 fields, respectively.

Optionally, you may use datasets that exclude the first column containing the ID of each sequence. In this case, be sure to use the `--exclude-seq-id` flag. Blank lines and lines starting with `#` are ignored.

**Classification problem:** the labeled data should be integer class labels. For example, if there are 3 classes, then each datapoint should be either a '0', '1', or '2' (with no quotation marks).

**Regression problem:** If the user wishes to map each sequence or residue to a continuous real number, then each datapoint should be a float

**Missing residue labels:** for residue-mapped data, a residue with no label (e.g. a residue with no experimental measurement) can be written as `nan`. PARROT ignores these residues in training, in performance metrics and in figures, and writes `nan` for their true value in prediction output. Every sequence needs at least one labelled residue, and sequence-mapped data cannot contain missing values.

**Values with uncertainties:** for regression, each value can come with the standard deviation (SD) of its measurement error, written as `value,SD` (e.g. `3.21,0.15`; a missing residue value is still `nan`), with `--uncertainties`. Training then weights each value by 1/(SD² + s²), where s, the model error, is the error of the network's predictions beyond the values' own errors, fitted on the validation set. s is reported, and the test statistics say how well this account of the errors fits (the reduced chi-squared, the fraction of values within 1 and 2 total SDs, √(SD² + s²), of their prediction, and the noise ceiling). For the mean of n replicates, give the standard error, SD/√n. Every architecture supports it except `knn` and `unet2d`. See the "Values with uncertainties" documentation page.

**Matrix data:** PARROT can also learn a matrix of values per sequence, with one row per residue: a map of every pair of residues (L x L), or N related values per residue (L x N). The datafile then holds `<ID> <sequence>` on each line, and the matrices are in a NumPy `.npz` file, one array per sequence ID (`parrot-train unet2d --input-file data.tsv --output-network network.pt --split kmer --regression --matrix-file matrices.npz`). See the "Matrix data" documentation page.

For example datasets, see the TSV files provided in the **/data** folder.

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

Each architecture (`brnn`, `cnn`, `unet`, `transformer`, `composition`, the simple models `linear`, `svm`, `random_forest`, `knn` and `gbrt`, and `unet2d` for matrix data) is a subcommand with only the options that apply to it: `parrot-train -h` lists them, and `parrot-train cnn -h` shows the options of one. Every input is named: `--input-file` for the datafile, `--output-network` for the network (`.pt` is added if needed), and `--regression` or `--classification --num-classes N` for the task. Next to the network, `<network>_info.txt` records how it was trained, for reproducing it or for a paper's methods. Beyond this, there are a suite of other options that users can provide to tailor their network to their particular needs. Users can provide specific hyperparameters to craft their network, they can manually specify what samples in their data set are trained on and which are held out as test data, they can choose to output information and figures on their network's performance on the test data, and much more!

Neural networks also offer opt-in learning-rate schedules (`cosine`, `exponential`, or validation-driven `plateau`), linear warmup, gradient clipping and finite checks, and bounded recovery that replays a failed epoch at a lower rate. Residue and matrix tasks can optionally weight proteins equally by averaging each protein's labelled losses. Defaults retain a fixed user-specified learning rate and summed labelled losses, with clipping and recovery off. See [Training controls](docs/usage/training_controls.rst) for choosing settings, examples, and the optional training-history and recovery records.

For long runs, `--checkpoint` saves atomic resumable state and `--training-log` flushes epoch records as training proceeds. Continue with `parrot-train --resume <checkpoint>` or `parrot-optimize --resume <checkpoint>`, including partial searches and final CV retraining. Neural regression can select `--regression-loss mae|mse|huber`, with `--huber-delta` for Huber, and opt in to `--optimizer adamw --weight-decay <value>`. Defaults keep the original losses and Adam without weight decay. Optional `--amp` enables guarded CUDA mixed precision on supported GPUs; FP32 remains the default. See [Training controls](docs/usage/training_controls.rst) for resume constraints and hardware requirements.

Test reports now include MAE, RMSE, signed bias and R2, with mean protein errors for residue/matrix tasks. Per-protein and length-stratified metric tables accompany the pooled test statistics; `--no-stats` suppresses all three statistics files.

For numerical failures, optional `--capture-nonfinite failure.pt` saves the first failing batch and its pre-batch training state for replay. `python -m parrot.scripts.diagnose_brnn` runs matched FP32/BF16 BRNN pilots on synthetic proteins or a supplied residue-regression dataset, records batch gradient norms and runtime details, and replays captured failures against packed/unpacked LSTMs. See [Diagnosing numerical failures](docs/usage/numerical_debugging.rst) for commands and interpreting the results.

### 2. Optimize hyperparameters and train a network: ``parrot-optimize``

The ``parrot-optimize`` command searches for the best-performing hyperparameters of an architecture on a given dataset (e.g. ``parrot-optimize cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression``), then trains and tests a network with them, as ``parrot-train`` would. It tunes the initial learning rate (for networks trained epoch by epoch; any chosen schedule, safeguards and recovery policy are fixed throughout the search) and every hyperparameter the architecture marks as searched (for the BRNN, the number of layers and the hidden vector size), dropping poor configurations after a few epochs (successive halving) and proposing new ones where good results have clustered (TPE). Any hyperparameter given on the command line is fixed rather than searched. It takes considerably longer than ``parrot-train``: the default 50 trials cost about as much as 15-20 training runs.

### 3. Generate predictions with a trained network: ``parrot-predict``

Once a network has been trained for a particular machine learning task, the user can generate predictions on new sequences with this network using the ``parrot-predict`` command. The user provides a list of sequences they would like to predict (`<ID> <sequence>` on each line) and the saved network, and a file is outputted with the predictions. The network file records its architecture, task and encoding, so no other options are needed (except, for a network trained with `--embedding-context`, the full-length proteins the sequences were cut from):

```
parrot-predict sequences.txt network.pt predictions.txt
```

In Python, `Predictor` does the same, for one sequence at a time or for a list of them:

```python
from parrot import Predictor

predictor = Predictor('network.pt')
values = predictor.predict('MKVLAAGIVGLLLA')
all_values = predictor.predict_batch(['MKVLAAGIVGLLLA', 'MSDNEEGQSKRRQ'])
```

`predict_batch` predicts the sequences together as `parrot-predict` does, which for more than a handful of sequences is much faster than `predict` on each: every sequence is encoded first (a protein language model runs over many sequences at once), then the network runs over batches of similar length. Its predictions match those of `predict` up to float32 rounding, not exactly, since which sequences share a batch changes the order of the floating-point operations (for a network trained on protein language model embeddings, the model also runs in batches, and a stored 16-bit embedding value can move by one step, so the predictions can differ by more).

### Cross-validation: ``parrot-cvsplit``

``parrot-cvsplit data.tsv cv/splits -k 10`` writes one split-file per fold (`cv/splits_cv0.txt` to `cv/splits_cv9.txt`), each with a different tenth of the data as its test set, to train a network on each fold with `parrot-train --split`. Identical sequences are kept in the same set, but close relatives are not; to score hyperparameters by cross-validation, use `parrot-optimize --cv-folds`.

### GPUs (CUDA and Apple silicon)

Every command that trains, predicts or embeds sequences takes `--device`, which is `auto` by default. This uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Use `--device cpu`, `--device cuda` or `--device mps` to choose explicitly; asking for a device that is not available is an error. On a machine with more than one NVIDIA GPU, `--gpu-id N` chooses which one (numbered from 0). Trained networks are always saved with CPU tensors, so a network trained on any device can be loaded anywhere, and PARROT defaults to full float32 precision on every device (on NVIDIA GPUs it turns off TF32 while it runs), so a network gives the same predictions on any device to about 1e-4 or better. Opt-in `--amp` uses reduced precision for the CUDA training forward pass; evaluation and prediction remain FP32. In Python, `Predictor` also runs on a GPU if there is one; `Predictor(path, device='cpu')` uses the CPU anyway (which can be faster for a few sequences at a time, and is needed inside forked worker processes, where MPS cannot be used), and `Predictor(path, gpu_id=1)` the second NVIDIA GPU, as `--gpu-id 1` does.

### Encodings

Sequences are encoded with one-hot vectors by default. `--encode biophysics` uses nine biophysical scales instead, and `--encode <file>` uses your own table (see `data/encoding_example.txt`). `--standardize-encoding` rescales every feature of the encoding to mean 0 and standard deviation 1 across the amino acids, so that features measured on very different scales (e.g. molecular weight and an aromatic flag) contribute equally; it is recommended for `biophysics` and most user encodings. The standardization is saved with the network and applied automatically when predicting. See the `parrot-train` documentation for exactly what is done.

`--encode esm2_t12_35M_UR50D` (or another ESM-2 model, which needs the `plm` extra) or `--encode esmc_300m` / `esmc_600m` (EvolutionaryScale's ESM C, which needs the `esmc` extra; see Installation) encodes each residue by its embedding from a protein language model. Embedding is by far the slowest step, so `parrot-train` (and `parrot-optimize`) and `parrot-predict` embed each sequence once and store the embeddings in a cache directory next to their output (`parrot_train_embeddings` or `parrot_predict_embeddings`, each with a README), after printing how large the cache will be; `--embedding-cache` lets several runs share one cache, and `parrot-preprocess --encode` fills a cache ahead of time. For fragments cut out of larger proteins (e.g. a peptide-tiling library), `--embedding-context <FASTA file> --embedding-context-map <map>` embeds each fragment within its full-length protein rather than on its own; `--embed-mean-subtraction` subtracts each embedding's mean over its residues before it is stored; and `--embedding-layer` embeds from an earlier layer of the model than the final one. See "Protein language model encodings" in the documentation.

For sequence-mapped data, `--encode nardini` gives every residue its sequence's 90 NARDINI+ features (36 patterning z-scores, each against scrambles of the sequence itself, and 54 compositional z-scores against the human IDRome), as computed by SPARROW (`pip install git+https://github.com/idptools/sparrow.git`; see Installation). Alone, with the `composition` baseline or the simple models, this is a model of the NARDINI+ features and the length; `--encode onehot+nardini` (or `biophysics+nardini`) puts them after each residue's own encoding, for the order-aware networks. The features are computed once, in parallel, into the embedding cache; `--nardini-scrambles` sets the number of scrambles (default 1,000). See the "NARDINI features" documentation page.

### Batching

By default (`--batching length`), training batches group sequences of similar length, re-randomised every epoch, so little computation is spent on padding; for datasets whose sequence lengths vary widely this makes training several-fold faster. Padding is masked, so it affects a network's output only through float32 rounding, and this only changes which sequences share a batch. `--batching random` restores the freely shuffled batches of earlier PARROT versions.

### Copyright

Copyright (c) 2020-2026, Holehouse Lab

### Change log

See [CHANGELOG.md](CHANGELOG.md).

#### Acknowledgements

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
