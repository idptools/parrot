=================
 parrot-optimize
=================

``parrot-optimize`` searches for good hyperparameters for a network architecture on your dataset, then trains and tests a network with the best ones, producing the same output as ``parrot-train``. It needs no extra packages.

What is searched: the learning rate (for networks trained by gradient descent), plus every hyperparameter the chosen architecture marks as searched, either over a range of numbers or over a set of choices (for the default ``brnn``: the hidden size, between 5 and 50, and the number of layers, between 1 and 5; for ``cnn`` also the kernel width; see the "Architectures" page). Any architecture hyperparameter given on the command line (e.g. ``parrot-optimize cnn ... --channels 32 --attention-pooling-off``) is fixed at that value and not searched; hyperparameters that are neither searched nor given keep their defaults.

How it searches. Each configuration is scored by its lowest validation loss, exactly as ``parrot-train`` chooses its best epoch. Two ideas keep the search efficient:

* **Successive halving.** Every configuration is first trained for a few epochs. Once enough configurations have reached a stage, the best third of them continue for three times as many epochs, carrying on from where they stopped, up to ``--epochs``. With ``--epochs 30`` the stages are 3, 10 and 30 epochs. Poor configurations are therefore dropped after a few epochs, and most of the training goes to promising ones (Jamieson & Talwalkar 2016; Li et al. 2020).
* **Model-based sampling (TPE).** The first configuration is PARROT's defaults and the next few are spread evenly over the search space. After that, new configurations are suggested where good results have clustered so far: a tree-structured Parzen estimator (Bergstra et al. 2011, as used in BOHB, Falkner et al. 2018) models good and bad configurations separately and proposes the candidate most likely to be good. A third of new configurations are drawn at random, so the whole space keeps being explored.

The best configuration has then already been trained for the full number of epochs on the training set, keeping its best epoch on the validation set, so it is used as the final network and tested on the test set, exactly as ``parrot-train`` would. With ``--cv-folds k``, configurations are instead scored by k-fold cross-validation over the training and validation data (more robust for small datasets, but about k times slower), and the final network is then trained on the training set with the best hyperparameters.

As a rough guide, the default 50 trials cost about as much as 15-20 ``parrot-train`` runs with the same ``--epochs`` (20 trials, about 10). Every trial is recorded as it finishes in ``<output>_hyperparameter_trials.tsv``.

.. code-block:: bash

	$ parrot-optimize <architecture> --input-file <datafile> --output-network <name> --split <method or split-file> [-d <datatype>] (--regression | --classification --num-classes <N>) <flags>

Where *architecture* is the network whose hyperparameters are searched (see the "Architectures" page), ``--input-file`` gives the path to the whitespace-separated datafile, ``--output-network`` is where the final trained network will be saved (``.pt`` is added if the name does not end with it), and ``--split`` says how to divide the data into training, validation and test sets (with ``--cv-folds``, the folds are made from the training and validation sets).

As for ``parrot-train``, each architecture is a subcommand offering only the options that apply to it. ``parrot-optimize <architecture> -h`` lists its hyperparameters with their defaults and the values the search tries. The required flags are the same as ``parrot-train``'s, and commands in the form of earlier versions of PARROT stop with a message saying what to use instead.

**Required flags:**

	*  ``--input-file`` : The datafile: ``<ID> <sequence> <value(s)>`` on each line.
	*  ``--output-network`` : Where to save the final network (``.pt`` is added if the name does not end with it); every other output file is named after it.
	*  ``--split`` : How to divide the data into training, validation and test sets: one of ``mmseqs``, ``kmer`` or ``random``, or the path to a split-file. The three methods are those of ``parrot-preprocess --split`` (``homology``, ``kmer`` and ``random``), with ``parrot-preprocess``'s default criteria:

		*  ``mmseqs`` : homologous sequences, found by MMseqs2 (which must be installed separately; see "parrot-preprocess"), are kept in the same set, and any sequence still homologous to one in another set is then removed, so that no test sequence has a homologue in the training set. The most thorough choice.
		*  ``kmer`` : the same, with similar sequences found by PARROT's own k-mer search, which needs nothing extra but finds only closer relatives.
		*  ``random`` : sequences are assigned at random, keeping identical sequences in the same set. Close relatives can end up on both sides of the split, which can flatter the test results.
		*  a split-file : three lines, giving the training, validation and test sets, each a whitespace-separated list of 0-indexed line numbers of the datafile (see the **/data** folder for examples, or make one with ``parrot-preprocess --split`` or ``parrot-cvsplit``). Lines in no set are left out, and how many is reported.

	   If a file exists with the name given to ``--split``, it is always read as a split-file, even if the name is also a method's (so a split-file called ``random`` is still read). For other similarity criteria, or to keep groups of records together, make a split-file with ``parrot-preprocess``. The split, and how many sequences it removed, are recorded in ``<network>_info.txt``.
	*  ``--datatype`` / ``-d`` : 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. Architectures that support only one datatype (the simple models and ``unet2d``) do not need it.
	*  ``--regression`` or ``--classification`` : Whether the values are numbers to predict, or classes labelled 0, 1, 2, ... (one of the two must be given).
	*  ``--num-classes`` : With ``--classification``, the number of classes (at least 2).

**Optional flags, for every architecture:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--show-examples`` : Print a longer guide to ``parrot-optimize``: examples of how to use it, what it does and how it works.
	*  Architecture hyperparameter flags (e.g. ``--hidden-size``, ``--kernel-size``, ``--attention-pooling-off``) : Fix these hyperparameters instead of searching them.
	*  ``--trials`` : Number of hyperparameter configurations to try (default is 50). More trials explore the space more thoroughly; the cost grows roughly in proportion.
	*  ``--max-iter`` : Same as ``--trials`` (kept so that older scripts keep working).
	*  ``--cv-folds`` : Score each configuration by k-fold cross-validation over the training and validation data instead of on the validation set. Slower (about k times), but more reliable for small datasets.
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
	*  ``--seed`` : Integer random seed. Providing a seed makes the data splits, cross-validation folds and network training reproducible: re-running on the same machine and device (CPU, CUDA or MPS) gives the same results. By default no seed is set.
	*  ``--verbose`` / ``-v`` : Include this flag to produce more descriptive output to terminal.
	*  ``--silent`` : Include this flag to produce no output to terminal.

**Training flags, for networks trained epoch by epoch (every architecture except the simple models):**

	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required.
	*  ``--batching`` : How training sequences are grouped into batches: ``length`` (default) or ``random``. ``length`` puts sequences of similar length in the same batch (still re-randomised every epoch), so far less time is spent on padding; for datasets whose lengths vary widely this makes training several-fold faster. ``random`` shuffles freely, as earlier versions of PARROT did. Because padding never affects a network's output, this changes which sequences share a batch during training but nothing else.
	*  ``--epochs`` / ``-e`` : Number of epochs a configuration is trained for if it reaches the final stage, and so the number of epochs the final network is trained for (default is 100). Choose it as you would for ``parrot-train``.
	*  ``--min-epochs`` : Number of epochs every configuration is first trained for (default: ``--epochs`` divided by the reduction factor squared, i.e. one ninth, at least 1). The stages run from here up to ``--epochs``, each ``--reduction-factor`` times longer than the one before. Raise it if your networks need more epochs before their validation loss says anything about how they will end up.
	*  ``--reduction-factor`` : At each stage the best 1/reduction-factor of the configurations continue, for reduction-factor times as many epochs (default 3).

**Classification of sequences or residues (every architecture except unet2d):**

	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. For sequence data, each sequence's probabilities are written as one value per class; for residue data, each residue's probabilities are written as one comma-separated token (e.g. ``0.91,0.06,0.03``), so there is still one token per residue. This flag also modifies the output figures and output performance stats.

**Matrix data (unet2d):**

	*  ``--matrix-file`` (required), ``--matrix-width``, ``--symmetric``, ``--max-matrix-length`` : as for ``parrot-train`` (see the "Matrix data" page).

**Output:**

``parrot-optimize`` produces the same output as ``parrot-train`` (the network, its "_info.txt" description, which here also describes the search, test-set predictions, performance statistics and, with ``--include-figs``, figures) for the best configuration, plus:

* ``<output>_optimal_hyperparams.txt``: the architecture and the hyperparameters of the final network, one "name: value" pair per line (including the learning rate).
* ``<output>_hyperparameter_trials.tsv``: one row per configuration tried, written as the search goes: the trial number, how it was chosen (``default``, ``initial``, ``model`` or ``random``), whether it was ``completed`` (trained for the full number of epochs), ``stopped`` early or ``failed`` (e.g. a configuration that does not suit the data), the epochs it was trained for, its validation loss at the end of each stage, and its hyperparameters.

Earlier versions of PARROT used Gaussian-process Bayesian optimization (GPyOpt) with 5-fold cross-validation for every configuration. That cost about 325 full training runs with the default settings; the current search needs no extra packages and costs about 15-20.
