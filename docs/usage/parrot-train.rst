==============
 parrot-train
==============

``parrot-train`` is the primary command for training a PARROT network. In the most basic usage, the user chooses a network architecture, specifies their data, an output location and how to split the data, and a trained network will be created along with an estimate of network predictive performance on held-out test samples. There are many optional arguments to ``parrot-train`` that allow users to specify network parameters, create helpful output images, and otherwise modify PARROT to meet their needs.

Once PARROT is installed, the user can run ``parrot-train`` from the command line:

.. code-block:: bash
	
	$ parrot-train <architecture> --input-file <datafile> --output-network <name> --split <method or split-file> [-d <datatype>] (--regression | --classification --num-classes <N>) <flags>

Where *architecture* is the network to train (see below), ``--input-file`` gives the path to the whitespace-separated datafile, ``--output-network`` is where the trained network will be saved (``.pt``, PyTorch's convention, is added if the name does not end with it), and ``--split`` says how to divide the data into training, validation and test sets. Every other output file is named after the network. For example:

.. code-block:: bash

	$ parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression -hs 20 -nl 2
	$ parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --classification --num-classes 3 --channels 32
	$ parrot-train knn --input-file data.tsv --output-network network.pt --split kmer --regression --num-neighbors 10

Each architecture is a subcommand, and offers only the options that apply to it: its own hyperparameters, the training options if it is trained epoch by epoch, and the matrix-data options if it learns matrices. ``parrot-train -h`` lists the architectures, and ``parrot-train <architecture> -h`` lists an architecture's options with their defaults.

For residue-mapped data, a residue with no label (e.g. a residue with no experimental measurement) can be written as ``nan`` in the datafile. These residues are ignored in training, in performance metrics and in figures, and their true value is written as ``nan`` in the prediction output. Every sequence needs at least one labelled residue, and sequence-mapped data cannot contain missing values.

**Architectures:**

	*  ``brnn``, ``cnn``, ``unet``, ``transformer`` and ``composition`` : neural networks for sequence or residue data. ``brnn`` is the bidirectional LSTM PARROT has always used.
	*  ``linear``, ``svm``, ``random_forest`` and ``knn`` : simple models for sequence data, fitted in one step, so they have no learning rate, epochs or batch size.
	*  ``unet2d`` : a 2D U-Net for matrix data (see the "Matrix data" page).

Each architecture's hyperparameters, and advice on choosing one, are on the "Architectures" page.

**Commands from earlier versions of PARROT:** earlier versions took the datafile and network as the first two arguments, the number of classes as ``-c`` (1 for regression), and had a ``--force-cpu`` flag. Such commands now stop with a message saying what to use instead: the architecture first, then ``--input-file`` and ``--output-network``; ``--regression``, or ``--classification`` with ``--num-classes``; and ``--device cpu``. Earlier versions also split the data at random unless given a split-file; ``--split`` is now required, and ``--split random`` is closest to that default (it differs only in keeping identical sequences in the same set).

**Required flags:**

	*  ``--input-file`` : The datafile: ``<ID> <sequence> <value(s)>`` on each line (see the example datasets in the **/data** folder).
	*  ``--output-network`` : Where to save the trained network. ``.pt`` is added if the name does not end with it, and every other output file is named after it (e.g. ``network.pt`` gives ``network_info.txt``).
	*  ``--split`` : How to divide the data into training, validation and test sets: one of ``mmseqs``, ``kmer`` or ``random``, or the path to a split-file. The three methods are those of ``parrot-preprocess --split`` (``homology``, ``kmer`` and ``random``), with ``parrot-preprocess``'s default criteria:

		*  ``mmseqs`` : homologous sequences, found by MMseqs2 (which must be installed separately; see "parrot-preprocess"), are kept in the same set, and any sequence still homologous to one in another set is then removed, so that no test sequence has a homologue in the training set. The most thorough choice.
		*  ``kmer`` : the same, with similar sequences found by PARROT's own k-mer search, which needs nothing extra but finds only closer relatives.
		*  ``random`` : sequences are assigned at random, keeping identical sequences in the same set. Close relatives can end up on both sides of the split, which can flatter the test results.
		*  a split-file : three lines, giving the training, validation and test sets, each a whitespace-separated list of 0-indexed line numbers of the datafile (see the **/data** folder for examples, or make one with ``parrot-preprocess --split`` or ``parrot-cvsplit``). Lines in no set are left out, and how many is reported.

	   If a file exists with the name given to ``--split``, it is always read as a split-file, even if the name is also a method's (so a split-file called ``random`` is still read). For other similarity criteria, or to keep groups of records together, make a split-file with ``parrot-preprocess``. The split, and how many sequences it removed, are recorded in ``<network>_info.txt``.
	*  ``--datatype`` / ``-d`` : Describes how values are formatted in the datafile. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. Architectures that support only one datatype (the simple models, which learn 'sequence' data, and ``unet2d``, which learns 'matrix' data) do not need it.
	*  ``--regression`` or ``--classification`` : Whether the values are numbers to predict, or classes labelled 0, 1, 2, ... (one of the two must be given).
	*  ``--num-classes`` : With ``--classification``, the number of classes (at least 2).

**Optional flags, for every architecture:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--show-examples`` : Print a longer guide to ``parrot-train``: examples of how to use it, what it does and how it works.
	*  ``--set-fractions`` : With ``--split mmseqs``, ``kmer`` or ``random``, the target fractions of the data in the training, validation and test sets: three numbers between 0 and 1 that sum to 1 (default 0.7 0.15 0.15). The similarity splits keep related sequences together, so their sets come close to these fractions rather than matching them exactly. A split-file already gives the sets, so the two cannot be combined.
	*  ``--save-splits`` : With ``--split mmseqs``, ``kmer`` or ``random``, save the split as a split-file (``<network>_split_file.txt``, next to the network), to train other networks on exactly the same sets with ``--split``.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics', a protein language model (e.g. 'esm2_t12_35M_UR50D'; see "Protein language model encodings"), or user-specified; and, for sequence data, 'nardini', the NARDINI+ features of the whole sequence, alone or after another encoding (e.g. 'onehot+nardini'; see "NARDINI features"). If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. Example encoding files are provided in the **/data** folder.
	*  ``--embedding-cache`` : Protein language model and NARDINI encodings only. Directory in which to store the embedding (or NARDINI features) of every sequence, so each is computed only once (default: a ``parrot_train_embeddings`` directory next to the network). Point several runs at the same directory to share embeddings. See "Protein language model encodings".
	*  ``--nardini-scrambles`` : NARDINI encodings only. Scrambles of each sequence used for its patterning z-scores (default 1,000; the cost grows in proportion). See "NARDINI features".
	*  ``--standardize-encoding`` : Standardize every feature of the encoding to mean 0 and standard deviation 1 across the amino acids (or other characters) the encoding defines, so that features measured on very different scales contribute equally. Recommended for ``biophysics`` and most user encodings; see "Standardizing an encoding" on the ``parrot-train`` page. The standardization is saved with the network and applied automatically by ``parrot-predict`` and ``Predictor``.
	*  ``--exclude-seq-id`` : Include this flag if the datafile is formatted without sequence IDs as the first column in each row.
	*  ``--include-figs`` : Include this flag to generate images based on network training and performance on test set. Figures will be saved to same directory as the network using same naming convention.
	*  ``--no-stats`` : Include this flag to prevent a "_performance_stats.txt" file from being output.
	*  ``--ignore-warnings`` : By default, PARROT checks your data for a few criteria and prints warnings if it doesn't meet some basic heuristics. Use this flag to silence these warnings (network training occurs unimpeded in either case).
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``).
	*  ``--seed`` : Integer random seed. Providing a seed makes the random train/validation/test split and network training reproducible: re-running on the same machine and device (CPU, CUDA or MPS) gives the same results. By default no seed is set.
	*  ``--verbose`` / ``-v`` : Include this flag to produce more descriptive output to terminal.
	*  ``--silent`` : Include this flag to produce no output to terminal.

**Training flags, for networks trained epoch by epoch (every architecture except the simple models):**

	*  ``--learning-rate`` / ``-lr`` : Learning rate of the network (default is 0.001). Must be a float between 0 and 1.
	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required.
	*  ``--batching`` : How training sequences are grouped into batches: ``length`` (default) or ``random``. ``length`` puts sequences of similar length in the same batch (still re-randomised every epoch), so far less time is spent on padding; for datasets whose lengths vary widely this makes training several-fold faster. ``random`` shuffles freely, as earlier versions of PARROT did. Because padding never affects a network's output, this changes which sequences share a batch during training but nothing else.
	*  ``--epochs`` / ``-e`` : Number of training epochs (default is 100). Has different behavior depending on what is specified by the ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until ``-e`` epochs in a row have not lowered the validation loss by at least 0.5% (relative to the last epoch that did). If using 'auto', be careful not to indicate a large number of epochs, as this can take much longer than is necessary.

**Classification of sequences or residues (every architecture except unet2d):**

	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. For sequence data, each sequence's probabilities are written as one value per class; for residue data, each residue's probabilities are written as one comma-separated token (e.g. ``0.91,0.06,0.03``), so there is still one token per residue. This flag also modifies the output figures and output performance stats.

**Matrix data (unet2d):**

	*  ``--matrix-file`` (required), ``--matrix-width``, ``--symmetric``, ``--max-matrix-length`` : see the "Matrix data" page.

**Standardizing an encoding:**

The ``biophysics`` encoding, and many user-specified encodings, describe each amino acid with features measured on very different scales. In ``biophysics``, molecular weight ranges from 75 to 204 g/mol and side-chain solvent-accessible surface area up to ~250 Å², while the aromatic and hydrogen-bonding features are 0 or 1 and charge is -1, 0 or 1. Neural networks are trained by gradient descent from small random weights, so at the start of training the large-valued features dominate the first layer's inputs and gradients, and the small-valued ones are effectively ignored until the network has learnt to rescale them. This makes training slower and less reliable, and can stop a network from using informative features at all.

``--standardize-encoding`` removes this problem. For every feature (column) of the encoding table, PARROT computes its mean and standard deviation across the characters the encoding defines (the 20 amino acids for the built-in encodings), and then encodes every residue as

    standardized value = (value - mean of the feature) / standard deviation of the feature

so that every feature has mean 0 and standard deviation 1 across the amino acids. This follows the standard advice for training networks by gradient descent: inputs that are centred and on a common scale make optimisation better conditioned (LeCun, Bottou, Orr & Müller 1998, "Efficient BackProp").

What this does and does not change:

* **No information is added or lost.** Every residue is transformed by the same shift and scale, so two amino acids that differ in a feature still differ in it, by the same amount relative to the spread of that feature.
* **The statistics come from the encoding, not from your data.** The mean and standard deviation are taken over the 20 amino acids (each counted once), not over the residues in your datafile. A given encoding is therefore standardized identically for every dataset, and nothing about your training, validation or test sequences is used.
* **Features are not decorrelated.** Some ``biophysics`` features are correlated (e.g. molecular weight and side-chain surface area), and they stay correlated. A network's first layer is linear, so it can learn any combination of the features once they are on a common scale.
* **A feature that is the same for every amino acid** carries no information. It is set to 0, with a warning.
* **Padding is unaffected.** Padded positions in a batch stay zero, and every architecture ignores them.
* **One-hot encoding** is only shifted and rescaled (a present amino acid becomes 4.36 instead of 1, and an absent one -0.23 instead of 0), so standardizing it makes little difference.
* **The scikit-learn models** (``linear``, ``svm``, ``random_forest``, ``knn``) already standardize their own input features, so the option has little effect on them.

The network file records the base encoding and the exact mean and standard deviation of every feature, so ``parrot-predict`` and ``Predictor`` standardize new sequences in exactly the same way, with no flags needed.

**Output:**

After running ``parrot-train``, at least four files will be saved, named after ``--output-network`` (e.g. ``network.pt``). The network itself can be used with ``parrot-predict``: as well as the network weights, this file records the architecture, its hyperparameters, the datatype, the number of classes and the encoding, so none of these need to be given again when making predictions. "_predictions.tsv" contains the true values and predicted values for all of the sequences in the test set, and "_performance_stats.txt" quantifies a variety of performance metrics on the test set.

"_info.txt" describes how the network was trained, in enough detail to reproduce it or to describe it in the methods section of a paper: the exact command, the date and the versions of PARROT, Python and the main libraries, the device and the random seed; the datafile (with its SHA-256 checksum, so the exact data can be identified), the task, the number of sequences in each set and how they were split, the sequence lengths and the values or class balance; the encoding and its settings; the architecture, every hyperparameter, the number of parameters and the network's layers; the loss function, the optimizer and its settings, the batch size, the number of epochs and which epoch's network was kept; for ``parrot-optimize``, the hyperparameter search; the validation and test results; and how long training took (in total and per epoch), how many sequences per second the network trained on, and how many it predicts per second (measured on the test set, on the device used).

Unless ``--silent`` is given, ``parrot-train`` also prints how the data were split, the validation loss as training goes (for the first epoch, every fifth and the last; every epoch with ``--verbose``), which epoch had the lowest validation loss (the network from that epoch is the one saved), the test-set loss and performance metrics, and every file it wrote.

For matrix data, the test-set predictions are instead written to a "_predictions.npz" file, and the figures include heatmaps of example test sequences; see the "Matrix data" page.

If ``--include-figs`` is provided, there will be additional PNG images saved in this directory. The first, with suffix '_train_val_loss.png' displays the network's performance on the training and validation sets over the course of training. The other image(s) describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, there are two images comparing the predicted and true values of the test set sequences, both labelled with Pearson's r: a scatterplot ('_scatterplot.png') and a density plot ('_density.png'), a 2D histogram coloured by how many values fall in each bin (on a log scale, or a linear one if no bin holds 10 values). Predictions that are not finite (NaN or infinite, e.g. from a network whose training diverged) cannot be plotted, so they are left out of both, with a warning; the performance metrics still include them. With many test values (e.g. every residue of thousands of sequences) the scatterplot becomes a solid block, while the density plot still shows where the values lie. If using probabilistic-classification mode, then there will be two output figures: one plotting receiver operator characteristic (ROC) curves and the other plotting precision-recall curves.

For residue data, '_res_profiles.png' also shows the true (black) and predicted (red) values along six test sequences, one panel each. They are chosen to span the range of accuracy, from the most to the least accurately predicted (at evenly spaced quantiles of each sequence's mean absolute error, from sequences of at least 30 residues), and each panel is labelled with its sequence's ID, Pearson's r and the mean absolute error (MAE) between its true and predicted values (or, for classification, its accuracy).
