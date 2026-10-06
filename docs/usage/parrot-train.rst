==============
 parrot-train
==============

``parrot-train`` is the primary command for training a PARROT network. In the most basic usage, the user specifies their data and an output location and a trained bi-directional LSTM network will be created along with an estimate of network predictive performance on randomly chosen test samples. There are many optional arguments to ``parrot-train`` that allow users to specify network parameters, create helpful output images, and otherwise modify PARROT to meet their needs.

Once PARROT is installed, the user can run ``parrot-train`` from the command line:

.. code-block:: bash
	
	$ parrot-train data_file output_network <flags>

Where *data_file* specifies the path to the whitespace-separated datafile and *output_network* is the path to where the final trained network will be saved. It is recommended, but not required, that networks are saved using a ".pt" file extension, following the PyTorch convention.

For residue-mapped data, a residue with no label (e.g. a residue with no experimental measurement) can be written as ``nan`` in *data_file*. These residues are ignored in training, in performance metrics and in figures, and their true value is written as ``nan`` in the prediction output. Every sequence needs at least one labelled residue, and sequence-mapped data cannot contain missing values.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, 'residues' if there are values for every residue in each sequence, or 'matrix' if each sequence has a matrix of values (given in a separate ``--matrix-file``; see the "Matrix data" page). See the example datasets in the **/data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1' (without the quote marks).

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--architecture`` : The model to train: 'brnn' (default; the bidirectional LSTM PARROT has always used), 'cnn', 'unet', 'transformer' or 'composition' (an order-independent baseline), or, for sequence-mapped data only, the simple models 'linear', 'svm', 'random_forest' or 'knn'. For matrix data, 'unet2d' (the default for it). Each architecture has its own hyperparameter flags, described on the "Architectures" page and listed under "architecture hyperparameters" in ``parrot-train --help``; passing a hyperparameter that the chosen architecture does not have is an error.
	*  ``--learning-rate`` / ``-lr`` : Learning rate of the network (default is 0.001). Must be a float between 0 and 1.
	*  ``--num-layers`` / ``-nl`` : (brnn) Number of hidden layers in the network (default is 1). Must be a positive integer.
	*  ``--hidden-size`` / ``-hs`` : (brnn) Size of hidden vectors within the network (default is 10). Must be a positive integer.
	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required.
	*  ``--batching`` : How training sequences are grouped into batches: ``length`` (default) or ``random``. ``length`` puts sequences of similar length in the same batch (still re-randomised every epoch), so far less time is spent on padding; for datasets whose lengths vary widely this makes training several-fold faster. ``random`` shuffles freely, as earlier versions of PARROT did. Because padding never affects a network's output, this changes which sequences share a batch during training but nothing else.
	*  ``--epochs`` / ``-e`` : Number of training epochs (default is 100). Has different behavior depending on what is specified by the ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until performance has plateaued for ``-e`` epochs. If using 'auto', be careful not to indicate a large number of epochs, as this can take much longer than is necessary.
	*  ``--split`` : Path to a "split-file" for manually dividing dataset into training, validation and test sets. The file should contain three lines, corresponding to training, validation and test sets respectively. Each line should have integers separated by whitespace, with the integers specify which sequences/lines in the `datafile` (0-indexed) will belong to which dataset. See **/data** folder for examples. If a split-file is not provided, default behavior is for PARROT to randomly divide data into training, validation and test sets. Random splits put close homologues on both sides of the split; ``parrot-preprocess --split homology`` writes a datafile and split-file in which they are not.
	*  ``--set-fractions`` : Include this flag to manually set the proportions of the data belonging to the training, validation and test sets. This option must be followed by three floats (representing train, validation, and test) between 0 and 1 that cumulatively sum to 1. By default, PARROT uses splits of 70:15:15. Note that the ``--split`` flag overrides these values.
	*  ``--save-splits`` : Include this flag if you would like PARROT to produce a split-file based on its random partitioning of data into training/validation/test sets, which can be useful for replication and/or testing multiple networks on the same data. Output split-file will be saved in the same folder as *output_network* using the same name followed by "_split_file.txt". This flag is overridden if a split-file is manually provided. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics', a protein language model (e.g. 'esm2_t12_35M_UR50D'; see "Protein language model encodings"), or user-specified. If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. Example encoding files are provided in the **/data** folder.
	*  ``--embedding-cache`` : Protein language model encodings only. Directory in which to store the embedding of every sequence, so each is computed only once (default: a ``parrot_train_embeddings`` directory next to *output_network*). Point several runs at the same directory to share embeddings. See "Protein language model encodings".
	*  ``--standardize-encoding`` : Standardize every feature of the encoding to mean 0 and standard deviation 1 across the amino acids (or other characters) the encoding defines, so that features measured on very different scales contribute equally. Recommended for ``biophysics`` and most user encodings; see "Standardizing an encoding" on the ``parrot-train`` page. The standardization is saved with the network and applied automatically by ``parrot-predict`` and ``Predictor``.
	*  ``--matrix-file``, ``--matrix-width``, ``--symmetric``, ``--max-matrix-length`` : Matrix data only; see the "Matrix data" page.
	*  ``--exclude-seq-id`` : Include this flag if the `data_file` is formatted without sequence IDs as the first column in each row.
	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. Currently only implemented for sequence classification. This flag also modifies the output figures and output performance stats. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--include-figs`` : Include this flag to generate images based on network training and performance on test set. Figures will be saved to same directory as specified by *output_network* using same naming convention.
	*  ``--no-stats`` : Include this flag to prevent a "_performance_stats.txt" file from being output.
	*  ``--ignore-warnings`` : By default, PARROT checks your data for a few criteria and prints warnings if it doesn't meet some basic heuristics. Use this flag to silence these warnings (network training occurs unimpeded in either case).
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--force-cpu`` : Same as ``--device cpu`` (kept so that older scripts keep working).
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``).
	*  ``--seed`` : Integer random seed. Providing a seed makes the random train/validation/test split and network training reproducible: re-running on the same machine and device (CPU, CUDA or MPS) gives the same results. By default no seed is set.
	*  ``--verbose`` / ``-v`` : Include this flag to produce more descriptive output to terminal.
	*  ``--silent`` : Include this flag to produce no output to terminal.

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

After running ``parrot-train``, at least three files will be saved to the directory specified by *output_network*. One contains the saved network, which can be used with ``parrot-predict``. As well as the network weights, this file records the architecture, its hyperparameters, the datatype, the number of classes and the encoding, so none of these need to be given again when making predictions. The other file with the suffix "_predictions.tsv" contains the true values and predicted values for all of the sequences in the test set. The final "_performance_stats.txt" file quantifies a variety of performance metrics on the test set. Output text detailing network performance across training is also printed to terminal by default.

For matrix data, the test-set predictions are instead written to a "_predictions.npz" file, and the figures include heatmaps of example test sequences; see the "Matrix data" page.

If ``--include-figs`` is provided, there will be additional PNG images saved in this directory. The first, with suffix '_train_val_loss.png' displays the network's performance on the training and validation sets over the course of training. The other image(s) describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, the image will be a scatterplot comparing the predicted and true values of the test set sequences. If using probabilistic-classification mode, then there will be two output figures: one plotting receiver operator characteristic (ROC) curves and the other plotting precision-recall curves.
