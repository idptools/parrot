=================
 parrot-optimize
=================

``parrot-optimize`` searches for good hyperparameters for a network architecture on your dataset, then trains and tests a network with the best ones, producing the same output as ``parrot-train``. It needs no extra packages.

What is searched: the initial learning rate (for networks trained by gradient descent, between 0.00001 and 0.01 on a log scale; ``parrot-optimize`` has no ``--learning-rate``, so it is always searched), plus every hyperparameter the chosen architecture marks as searched, either over a range of numbers or over a set of choices (for ``brnn``: the hidden size, between 5 and 50, and the number of layers, between 1 and 5, but not the dropout; for ``cnn``: the number of channels and of blocks, the kernel width and the dropout; see the "Architectures" page). Any architecture hyperparameter given on the command line (e.g. ``parrot-optimize cnn ... --channels 32 --attention-pooling-off``) is fixed at that value and not searched; hyperparameters that are neither searched nor given keep their defaults. Hyperparameters that have no effect on your data are not searched either, and the output says so: the composition window (``--window-radius``) for sequence data, how pairs of residues are combined (``--pair-features``) for matrices with a fixed number of columns, and dipeptide features for encodings of more than 32 values per residue. A hyperparameter that the rest of a configuration switches off (the SVM's ``--relative-gamma`` with a linear kernel) is set to its default, so configurations that differ only in it are tried once.

How it searches. Each configuration is scored by its lowest validation loss, exactly as ``parrot-train`` chooses its best epoch. Two ideas keep the search efficient:

* **Successive halving.** Every configuration is first trained for a few epochs (``--min-epochs``, by default a ninth of ``--epochs``). Once enough configurations have reached a stage, the best third of them continue for three times as many epochs, carrying on from where they stopped, up to ``--epochs``. With ``--epochs 30`` the stages are 3, 10 and 30 epochs, with ``--epochs 50``, 6, 17 and 50, and with the default 100, 11, 33 and 100. Poor configurations are therefore dropped after a few epochs, and most of the training goes to promising ones (Jamieson & Talwalkar 2016; Li et al. 2020). The simple models (``linear``, ``svm``, ``random_forest`` and ``knn``) are fitted in one step, so each of their configurations is simply fitted and scored once.
* **Model-based sampling (TPE).** The first configuration is PARROT's defaults and the next eight (or two per searched hyperparameter, if that is more) are spread evenly over the search space. After that, new configurations are suggested where good results have clustered so far: a tree-structured Parzen estimator (Bergstra et al. 2011, as used in BOHB, Falkner et al. 2018) models good and bad configurations separately and proposes the candidate most likely to be good. A third of new configurations are drawn at random, so the whole space keeps being explored.

The best configuration has then already been trained for the full number of epochs on the training set, keeping its best epoch on the validation set, so it is used as the final network and tested on the test set, exactly as ``parrot-train`` would. With ``--cv-folds k``, configurations are instead scored by k-fold cross-validation over the training and validation data (more robust for small datasets, but about k times slower), and the final network is then trained on the training set with the best hyperparameters. The folds respect ``--split`` as the sets do: identical sequences are always in the same fold, and with ``--split kmer`` or ``mmseqs`` so are clusters of similar sequences, and each fold is scored only on the held-out sequences that are not similar to any of its training sequences (the output says how many are left out).

As a rough guide, the default 50 trials cost about as much as 15-20 ``parrot-train`` runs with the same ``--epochs`` (20 trials, about 6-8). Every trial is recorded as it finishes in ``<output>_hyperparameter_trials.tsv``. If every hyperparameter is fixed (or not searched), a single network is trained; if the search space has fewer configurations than ``--trials``, the search ends once each has been tried.

.. code-block:: bash

	$ parrot-optimize <architecture> --input-file <datafile> --output-network <name> --split <method or split-file> [-d <datatype>] (--regression | --classification --num-classes <N>) <flags>

Where *architecture* is the network whose hyperparameters are searched (see the "Architectures" page), ``--input-file`` gives the path to the whitespace-separated datafile, ``--output-network`` is where the final trained network will be saved (``.pt`` is added if the name does not end with it), and ``--split`` says how to divide the data into training, validation and test sets (with ``--cv-folds``, the folds are made from the training and validation sets).

As for ``parrot-train``, each architecture is a subcommand offering only the options that apply to it. ``parrot-optimize <architecture> -h`` lists its hyperparameters with their defaults and the values the search tries. The required flags are the same as ``parrot-train``'s, and commands in the form of earlier versions of PARROT stop with a message saying what to use instead.

**Required flags:**

	*  ``--input-file`` : The datafile: ``<ID> <sequence> <value(s)>`` on each line. Blank lines and lines starting with '#' are ignored.
	*  ``--output-network`` : Where to save the final network (``.pt`` is added if the name does not end with it); every other output file is named after it. The directory must already exist.
	*  ``--split`` : How to divide the data into training, validation and test sets: one of ``mmseqs``, ``kmer`` or ``random``, or the path to a split-file. The three methods are those of ``parrot-preprocess --split`` (``homology``, ``kmer`` and ``random``), with ``parrot-preprocess``'s default criteria:

		*  ``mmseqs`` : homologous sequences, found by MMseqs2 (which must be installed separately; see "parrot-preprocess"), are kept in the same set, and any sequence still homologous to one in another set is then removed, so that no test sequence has a homologue in the training set. The most thorough choice. If MMseqs2 cannot be found, the command stops before reading any data.
		*  ``kmer`` : the same, with similar sequences found by PARROT's own k-mer search, which needs nothing extra but finds only closer relatives.
		*  ``random`` : sequences are assigned at random, keeping identical sequences in the same set. Close relatives can end up on both sides of the split, which can flatter the test results.
		*  a split-file : three lines, giving the training, validation and test sets, each a whitespace-separated list of 0-indexed line numbers of the datafile, counting only its data lines, not blank or comment lines (see the **/data** folder for examples, or make one with ``parrot-preprocess --split`` or ``parrot-cvsplit``). Lines in no set are left out, and how many is reported; a line number that is in more than one set, or beyond the end of the datafile, is an error.

	   If a file exists with the name given to ``--split``, it is always read as a split-file, even if the name is also a method's (so a split-file called ``random`` is still read). For other similarity criteria, or to keep groups of records together, make a split-file with ``parrot-preprocess``. The split, and how many sequences it removed, are recorded in ``<network>_info.txt``.
	*  ``--datatype`` / ``-d`` : 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. Architectures that support only one datatype (the simple models and ``unet2d``) do not need it.
	*  ``--regression`` or ``--classification`` : Whether the values are numbers to predict, or classes labelled 0, 1, 2, ... (one of the two must be given).
	*  ``--num-classes`` : With ``--classification``, the number of classes (at least 2). It is an error with ``--regression``.

**Optional flags, for every architecture:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--show-examples`` : Print a longer guide to ``parrot-optimize``: examples of how to use it, what it does and how it works.
	*  Architecture hyperparameter flags (e.g. ``--hidden-size``, ``--kernel-size``, ``--attention-pooling-off``) : Fix these hyperparameters instead of searching them.
	*  ``--trials`` : Number of hyperparameter configurations to try (default is 50; at least 1). More trials explore the space more thoroughly; the cost grows roughly in proportion.
	*  ``--max-iter`` : An older name for ``--trials``, kept so that older scripts keep working (giving both is an error).
	*  ``--cv-folds`` : Score each configuration by k-fold cross-validation over the training and validation data instead of on the validation set (k at least 2). Slower (about k times), but more reliable for small datasets. The folds keep identical sequences together, and with ``--split kmer`` or ``mmseqs`` similar ones too (see above); if that leaves a fold with no held-out sequence to score, the command stops and suggests fewer folds.
	*  ``--set-fractions`` : With ``--split mmseqs``, ``kmer`` or ``random``, the target fractions of the data in the training, validation and test sets: three numbers, each greater than 0 and less than 1, that sum to 1 (default 0.7 0.15 0.15). The similarity splits keep related sequences together, so their sets come close to these fractions rather than matching them exactly. A split-file already gives the sets, so the two cannot be combined.
	*  ``--save-splits`` : With ``--split mmseqs``, ``kmer`` or ``random``, save the split as a split-file (``<network>_split_file.txt``, next to the network), to train other networks on exactly the same sets with ``--split``. It cannot be combined with a split-file.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics', a protein language model (e.g. 'esm2_t12_35M_UR50D'; see "Protein language model encodings"), or user-specified; and, for sequence data, 'nardini', the NARDINI+ features of the whole sequence, alone or after 'onehot', 'biophysics' or a user-specified encoding (e.g. 'onehot+nardini'; not after a protein language model; see "NARDINI features"). If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. Example encoding files are provided in the **/data** folder.
	*  ``--embedding-cache`` : Protein language model and NARDINI encodings only (it is an error with any other encoding). Directory in which to store the embedding (or NARDINI features) of every sequence, so each is computed only once (default: a ``parrot_train_embeddings`` directory next to the network). Point several runs at the same directory to share embeddings. See "Protein language model encodings".
	*  ``--embedding-context`` / ``--embedding-context-map`` : Protein language model encodings only. Embed each sequence within the full-length protein it was cut from, rather than on its own: the FASTA file of full-length proteins, and a map with one ``<sequence ID> <full-length ID> [start]`` line per sequence (``-`` as the full-length ID embeds that sequence on its own). Every sequence is placed in its protein before anything is embedded. The map is matched to the datafile by sequence ID, so they cannot be used with ``--exclude-seq-id``. A table of where each one is embedded is written next to the network (``<network>_embedded_context.tsv``), and the network records that it was trained this way, so ``parrot-predict`` needs the same options for the sequences it predicts (or ``--embed-in-isolation``). See "Embedding fragments within their full-length proteins" on the "Protein language model encodings" page.
	*  ``--embed-mean-subtraction`` : Protein language model encodings only. Subtract the mean of each embedding over its residues (its mean pool) from every residue's embedding before it is stored, so that what the network sees is how each residue differs from the sequence as a whole. With ``--embedding-context``, the mean is that of the whole full-length protein, subtracted before each fragment is cut out. These embeddings are stored in their own files of the cache, and the network records the setting, so ``parrot-predict`` and ``Predictor`` apply it automatically. It cannot be used with the simple models (``linear``, ``svm``, ``random_forest``, ``knn``, ``gbrt``) without ``--embedding-context``, since their features are each sequence's mean embedding, which it makes zero. See "Subtracting each embedding's mean" on the "Protein language model encodings" page.
	*  ``--embedding-layer`` : Protein language model encodings only. The layer of the model whose output is each residue's embedding, from 1 (the first transformer layer) to the model's number of layers (the final layer, the default); negative numbers count back from the final layer (``-1`` is the final layer). The model is run only up to that layer, and an earlier layer's output lacks the final layer norm. These embeddings are stored in their own files of the cache, and the network records the layer, so ``parrot-predict`` and ``Predictor`` use it automatically. See "Choosing the layer" on the "Protein language model encodings" page.
	*  ``--nardini-scrambles`` : NARDINI encodings only. Scrambles of each sequence used for its patterning z-scores (default 1,000; more take longer: 10,000 take about three to four times as long as 1,000). See "NARDINI features".
	*  ``--standardize-encoding`` : Standardize every feature of the encoding to mean 0 and standard deviation 1 across the amino acids (or other characters) the encoding defines, so that features measured on very different scales contribute equally. Recommended for ``biophysics`` and most user encodings; see "Standardizing an encoding" on the ``parrot-train`` page. The standardization is saved with the network and applied automatically by ``parrot-predict`` and ``Predictor``.
	*  ``--exclude-seq-id`` : Include this flag if the datafile is formatted without sequence IDs as the first column in each row. In the test-set predictions the sequences are then named ``test0``, ``test1``, ... (not for matrix data, which needs its sequence IDs).
	*  ``--include-figs`` : Include this flag to generate images based on network training and performance on test set. Figures will be saved to same directory as the network using same naming convention (see "Output" on the ``parrot-train`` page).
	*  ``--no-stats`` : Include this flag to prevent a "_performance_stats.txt" file from being output.
	*  ``--ignore-warnings`` / ``-w`` : By default, PARROT checks your data for a few criteria and prints warnings if it doesn't meet some basic heuristics. Use this flag to silence these warnings (network training occurs unimpeded in either case). ``--silent`` silences them too.
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``). Giving it asks for CUDA, so it is an error with ``--device cpu`` or ``mps``, or on a machine without CUDA.
	*  ``--seed`` : Integer random seed. Providing a seed makes the data splits, cross-validation folds and network training reproducible: re-running on the same machine and device (CPU, CUDA or MPS) gives the same results. By default no seed is set.
	*  ``--verbose`` / ``-v`` : Include this flag to print every setting (and, with ``--cv-folds``, the validation loss after every epoch of the final training; the search itself prints one line per configuration and stage either way).
	*  ``--silent`` : Include this flag to produce no output to terminal.

**Training flags, for networks trained epoch by epoch (every architecture except the simple models):**

Unlike ``parrot-train``, ``parrot-optimize`` has no ``--learning-rate`` (the initial rate is searched) and no ``--stop`` (every configuration is trained for set numbers of epochs).

	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required. PARROT warns if the batch size is above 512, or leaves fewer than 10 batches per epoch.
	*  ``--batching`` : How training sequences are grouped into batches: ``length`` (default) or ``random``. ``length`` puts sequences of similar length in the same batch (still re-randomised every epoch), so far less time is spent on padding; for datasets whose lengths vary widely this makes training several-fold faster. ``random`` shuffles freely, as earlier versions of PARROT did. Padding is masked, so it affects a network's output only through float32 rounding, and this changes which sequences share a batch during training but nothing else.
	*  ``--epochs`` / ``-e`` : Number of epochs a configuration is trained for if it reaches the final stage, and so the number of epochs the final network is trained for (default is 100). Choose it as you would for ``parrot-train``.
	*  ``--min-epochs`` : Number of epochs every configuration is first trained for, the first stage (default: ``--epochs`` divided by the reduction factor squared, i.e. one ninth, rounded and at least 1). The stages run from here up to ``--epochs``: counting back from ``--epochs``, each stage is ``--reduction-factor`` times shorter than the next, with as many stages as fit (for example ``-e 30 --min-epochs 3`` gives 3, 10 and 30; ``-e 30 --min-epochs 5`` gives 5, 10 and 30; and ``-e 10 --min-epochs 4`` gives 4 and 10). Raise it if your networks need more epochs before their validation loss says anything about how they will end up. It must be between 1 and ``--epochs``; equal to ``--epochs``, there is a single stage and every configuration is trained in full.
	*  ``--reduction-factor`` : At each stage the best 1/reduction-factor of the configurations continue, for reduction-factor times as many epochs (default 3; at least 2).

The optional training controls are shared with ``parrot-train``:
``--gradient-clip-norm``, ``--check-finite-gradients``,
``--equal-protein-weighting`` (residue and matrix tasks), ``--lr-schedule``
and its tuning flags, ``--lr-warmup-epochs``, ``--recover-nonfinite``,
``--recovery-lr-factor``, ``--regression-loss``/``--huber-delta``,
``--optimizer``/``--weight-decay`` and opt-in CUDA ``--amp``. Defaults preserve fixed-rate training with summed
labelled losses and no clipping or recovery. The :doc:`parrot-train` reference
lists all flags and defaults; :doc:`training_controls` explains their behavior.

BRNN searches accept ``--amp off`` only (the default). Any enabled AMP mode is
rejected before data loading, including when resuming a BRNN AMP checkpoint.
Start a new FP32 search in that case; see :ref:`parrot-amp`. Other neural
architectures retain opt-in CUDA AMP.

Controls are fixed throughout the search, while the initial learning rate is
searched. Schedules use the full ``--epochs`` horizon and retain their state
across promotions. Recovery budgets also persist across stages, separately
for each fold; final cross-validated retraining has a fresh budget. Trials
that fail the numerical checks and cannot recover are recorded as failed,
and the search continues. Equal-protein weighting applies to validation
scores as well as training and test loss.

``--checkpoint [file]``, ``--checkpoint-every`` and ``--resume file`` also cover
the entire search, including TPE state, partial folds, promotions and final
cross-validated retraining. ``parrot-optimize --resume file`` restores the
original command and actual data partitions. ``--training-log`` continuously
records every successful trial/fold epoch in
``<network>_search_training_history.tsv``; final CV retraining has its own
``_training_history.tsv``. Both features are off by default. See
:ref:`parrot-training-resume` and :ref:`parrot-training-diagnostics`.

**Regression values with uncertainties (every architecture except knn and unet2d):**

	*  ``--uncertainties`` : Every value in the datafile is written as ``value,SD``, where SD is the standard deviation of its measurement error, as for ``parrot-train``. Configurations are then scored by the Gaussian negative log-likelihood of the validation values, with the model error *s* fitted for each configuration (and each cross-validation fold). See the "Values with uncertainties" page.

**Classification of sequences or residues (every architecture except unet2d):**

	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. For sequence data, each sequence's probabilities are written as one value per class; for residue data, each residue's probabilities are written as one comma-separated token (e.g. ``0.9100,0.0600,0.0300``), so there is still one token per residue. Probabilities are written to 4 decimal places. This flag also modifies the output figures and output performance stats (as for ``parrot-train``), but not the search. It is an error with ``--regression``.

**Matrix data (unet2d):**

	*  ``--matrix-file`` (required), ``--matrix-width``, ``--symmetric``, ``--max-matrix-length`` : as for ``parrot-train`` (see the "Matrix data" page).

**Output:**

``parrot-optimize`` produces the same output as ``parrot-train`` (the network, its "_info.txt" description, which here also describes the search, test-set predictions, performance statistics and, with ``--include-figs``, figures) for the best configuration, plus:

* ``<output>_optimal_hyperparams.txt``: the architecture and the hyperparameters of the final network, one "name: value" pair per line (including the initial learning rate).
* ``<output>_hyperparameter_trials.tsv``: one row per configuration tried, written as the search goes: the trial number, how it was chosen (``default``, ``initial``, ``model`` or ``random``), whether it was ``completed`` (trained for the full number of epochs), ``stopped`` early or ``failed`` (e.g. a configuration that does not suit the data, or whose validation loss was never finite; the reason follows), the epochs it was trained for (0 for the simple models), its validation loss at the end of each stage (a single validation loss for the simple models), and its hyperparameters.

When schedules, warmup or recovery are enabled, the successful-epoch
``_training_history.tsv`` and, if needed, ``_recovery_events.json`` are also
written for the winning network, or its final retraining after cross-validation.
They are written on successful completion, rather than for every discarded
trial. Active settings and a rate summary appear in ``_info.txt``; see
:ref:`parrot-training-diagnostics` for file contents.

The best network's test statistics include the same absolute-error and
per-protein summaries as ``parrot-train``. ``_per_protein_metrics.tsv`` and
``_length_metrics.tsv`` describe the held-out test set, rather than the search
folds; ``--no-stats`` suppresses these tables and ``_performance_stats.txt``.

Earlier versions of PARROT used Gaussian-process Bayesian optimization (GPyOpt) with 5-fold cross-validation for every configuration. That cost more than 300 full training runs with the default settings; the current search needs no extra packages and costs about 15-20.
