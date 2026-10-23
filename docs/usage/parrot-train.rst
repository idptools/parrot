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

For regression, each value can also come with the standard deviation of its measurement error, written as ``value,SD`` (with ``--uncertainties``; see below and the "Values with uncertainties" page).

**Architectures:**

	*  ``brnn``, ``cnn``, ``unet``, ``transformer`` and ``composition`` : neural networks for sequence or residue data. ``brnn`` is the bidirectional LSTM PARROT has always used.
	*  ``linear``, ``svm``, ``random_forest``, ``knn`` and ``gbrt`` : simple models for sequence data, fitted in one step, so they have no learning rate, epochs or batch size.
	*  ``unet2d`` : a 2D U-Net for matrix data (see the "Matrix data" page).

Each architecture's hyperparameters, and advice on choosing one, are on the "Architectures" page. Some hyperparameters have no effect on some data (such as ``composition``'s ``--window-radius`` for sequence data, or ``svm``'s ``--relative-gamma`` with ``--kernel linear``). They are still accepted, and if one of these is changed from its default, ``parrot-train`` says so, and "_info.txt" marks it as having no effect. Not every such case is caught yet: ``--pooling``, ``--attention-pooling-off`` and ``--loglength-feedforward-off`` do nothing for residue data, but are not noted.

**Commands from earlier versions of PARROT:** earlier versions took the datafile and network as the first two arguments, the number of classes as ``-c`` (1 for regression), and had a ``--force-cpu`` flag. Such commands now stop with a message saying what to use instead: the architecture first, then ``--input-file`` and ``--output-network``; ``--regression``, or ``--classification`` with ``--num-classes``; and ``--device cpu``. Earlier versions also split the data at random unless given a split-file; ``--split`` is now required, and ``--split random`` is closest to that default (it differs only in keeping identical sequences in the same set).

**Required flags:**

	*  ``--input-file`` : The datafile: ``<ID> <sequence> <value(s)>`` on each line (see the example datasets in the **/data** folder). Blank lines and lines starting with '#' are ignored.
	*  ``--output-network`` : Where to save the trained network. ``.pt`` is added if the name does not end with it, and every other output file is named after it (e.g. ``network.pt`` gives ``network_info.txt``). The directory must already exist.
	*  ``--split`` : How to divide the data into training, validation and test sets: one of ``mmseqs``, ``kmer`` or ``random``, or the path to a split-file. The three methods are those of ``parrot-preprocess --split`` (``homology``, ``kmer`` and ``random``), with ``parrot-preprocess``'s default criteria:

		*  ``mmseqs`` : homologous sequences, found by MMseqs2 (which must be installed separately; see "parrot-preprocess"), are kept in the same set, and any sequence still homologous to one in another set is then removed, so that no test sequence has a homologue in the training set. The most thorough choice. If MMseqs2 cannot be found, the command stops before reading any data.
		*  ``kmer`` : the same, with similar sequences found by PARROT's own k-mer search, which needs nothing extra but finds only closer relatives.
		*  ``random`` : sequences are assigned at random, keeping identical sequences in the same set. Close relatives can end up on both sides of the split, which can flatter the test results.
		*  a split-file : three lines, giving the training, validation and test sets, each a whitespace-separated list of 0-indexed line numbers of the datafile, counting only its data lines, not blank or comment lines (see the **/data** folder for examples, or make one with ``parrot-preprocess --split`` or ``parrot-cvsplit``). Lines in no set are left out, and how many is reported; a line number that is in more than one set, or beyond the end of the datafile, is an error.

	   If a file exists with the name given to ``--split``, it is always read as a split-file, even if the name is also a method's (so a split-file called ``random`` is still read). For other similarity criteria, or to keep groups of records together, make a split-file with ``parrot-preprocess``. The split, and how many sequences it removed, are recorded in ``<network>_info.txt``.
	*  ``--datatype`` / ``-d`` : Describes how values are formatted in the datafile. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. Architectures that support only one datatype (the simple models, which learn 'sequence' data, and ``unet2d``, which learns 'matrix' data) do not need it.
	*  ``--regression`` or ``--classification`` : Whether the values are numbers to predict, or classes labelled 0, 1, 2, ... (one of the two must be given).
	*  ``--num-classes`` : With ``--classification``, the number of classes (at least 2). It is an error with ``--regression``.

**Optional flags, for every architecture:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--show-examples`` : Print a longer guide to ``parrot-train``: examples of how to use it, what it does and how it works.
	*  ``--set-fractions`` : With ``--split mmseqs``, ``kmer`` or ``random``, the target fractions of the data in the training, validation and test sets: three numbers, each greater than 0 and less than 1, that sum to 1 (default 0.7 0.15 0.15). The similarity splits keep related sequences together, so their sets come close to these fractions rather than matching them exactly. A split-file already gives the sets, so the two cannot be combined.
	*  ``--save-splits`` : With ``--split mmseqs``, ``kmer`` or ``random``, save the split as a split-file (``<network>_split_file.txt``, next to the network), to train other networks on exactly the same sets with ``--split``. It cannot be combined with a split-file.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics', a protein language model (e.g. 'esm2_t12_35M_UR50D'; see "Protein language model encodings"), or user-specified; and, for sequence data, 'nardini', the NARDINI+ features of the whole sequence, alone or after 'onehot', 'biophysics' or a user-specified encoding (e.g. 'onehot+nardini'; not after a protein language model; see "NARDINI features"). If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. Example encoding files are provided in the **/data** folder.
	*  ``--embedding-cache`` : Protein language model and NARDINI encodings only (it is an error with any other encoding). Directory in which to store the embedding (or NARDINI features) of every sequence, so each is computed only once (default: a ``parrot_train_embeddings`` directory next to the network). Point several runs at the same directory to share embeddings. See "Protein language model encodings".
	*  ``--embedding-context`` / ``--embedding-context-map`` : Protein language model encodings only. Embed each sequence within the full-length protein it was cut from, rather than on its own: the FASTA file of full-length proteins, and a map with one ``<sequence ID> <full-length ID> [start]`` line per sequence (``-`` as the full-length ID embeds that sequence on its own). Every sequence is placed in its protein before anything is embedded. The map is matched to the datafile by sequence ID, so they cannot be used with ``--exclude-seq-id``. A table of where each one is embedded is written next to the network (``<network>_embedded_context.tsv``), and the network records that it was trained this way, so ``parrot-predict`` needs the same options for the sequences it predicts (or ``--embed-in-isolation``). See "Embedding fragments within their full-length proteins" on the "Protein language model encodings" page.
	*  ``--embed-mean-subtraction`` : Protein language model encodings only. Subtract the mean of each embedding over its residues (its mean pool) from every residue's embedding before it is stored, so that what the network sees is how each residue differs from the sequence as a whole. With ``--embedding-context``, the mean is that of the whole full-length protein, subtracted before each fragment is cut out. These embeddings are stored in their own files of the cache, and the network records the setting, so ``parrot-predict`` and ``Predictor`` apply it automatically. It cannot be used with the simple models (``linear``, ``svm``, ``random_forest``, ``knn``, ``gbrt``) without ``--embedding-context``, since their features are each sequence's mean embedding, which it makes zero. See "Subtracting each embedding's mean" on the "Protein language model encodings" page.
	*  ``--embedding-layer`` : Protein language model encodings only. The layer of the model whose output is each residue's embedding, from 1 (the first transformer layer) to the model's number of layers (the final layer, the default); negative numbers count back from the final layer (``-1`` is the final layer). The model is run only up to that layer, and an earlier layer's output lacks the final layer norm. These embeddings are stored in their own files of the cache, and the network records the layer, so ``parrot-predict`` and ``Predictor`` use it automatically. See "Choosing the layer" on the "Protein language model encodings" page.
	*  ``--nardini-scrambles`` : NARDINI encodings only. Scrambles of each sequence used for its patterning z-scores (default 1,000; more take longer: 10,000 take about three to four times as long as 1,000). See "NARDINI features".
	*  ``--standardize-encoding`` : Standardize every feature of the encoding to mean 0 and standard deviation 1 across the amino acids (or other characters) the encoding defines, so that features measured on very different scales contribute equally. Recommended for ``biophysics`` and most user encodings; see "Standardizing an encoding" below. The standardization is saved with the network and applied automatically by ``parrot-predict`` and ``Predictor``.
	*  ``--exclude-seq-id`` : Include this flag if the datafile is formatted without sequence IDs as the first column in each row. In the test-set predictions the sequences are then named ``test0``, ``test1``, ... (not for matrix data, which needs its sequence IDs).
	*  ``--include-figs`` : Include this flag to generate images based on network training and performance on test set. Figures will be saved to same directory as the network using same naming convention (see "Output" below).
	*  ``--no-stats`` : Include this flag to prevent a "_performance_stats.txt" file from being output.
	*  ``--ignore-warnings`` / ``-w`` : By default, PARROT checks your data for a few criteria (e.g. duplicate sequences, class imbalance, regression values that are not centred on zero, widely spread or skewed, and a batch size that is very large or leaves fewer than 10 batches per epoch) and prints warnings if it doesn't meet some basic heuristics. Use this flag to silence these warnings (network training occurs unimpeded in either case). ``--silent`` silences them too.
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``). Giving it asks for CUDA, so it is an error with ``--device cpu`` or ``mps``, or on a machine without CUDA.
	*  ``--seed`` : Integer random seed. Providing a seed makes the train/validation/test split and network training reproducible: re-running on the same machine and device (CPU, CUDA or MPS) gives the same results. By default no seed is set.
	*  ``--verbose`` / ``-v`` : Include this flag to print every setting before training, and the validation loss after every epoch (rather than every fifth).
	*  ``--silent`` : Include this flag to produce no output to terminal.

**Training flags, for networks trained epoch by epoch (every architecture except the simple models):**

	*  ``--learning-rate`` / ``-lr`` : Initial learning rate of Adam (default is 0.001); fixed unless a schedule, warmup or recovery is enabled (see :doc:`training_controls`). Must be a float between 0 and 1.
	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required. PARROT warns if the batch size is above 512, or leaves fewer than 10 batches per epoch. The test set is also predicted in batches of this size.
	*  ``--batching`` : How training sequences are grouped into batches: ``length`` (default) or ``random``. ``length`` puts sequences of similar length in the same batch (still re-randomised every epoch), so far less time is spent on padding; for datasets whose lengths vary widely this makes training several-fold faster. ``random`` shuffles freely, as earlier versions of PARROT did. Padding is masked, so it affects a network's output only through float32 rounding, and this changes which sequences share a batch during training but nothing else.
	*  ``--epochs`` / ``-e`` : Number of training epochs (default is 100). Has different behavior depending on what is specified by the ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until ``-e`` epochs in a row have not lowered the validation loss by at least 0.5% (relative to the last epoch that did; with ``--uncertainties``, by about 0.005 per labelled value, or in the mean per-protein NLL with ``--equal-protein-weighting``; see :doc:`uncertainties`), so training lasts at least ``-e`` + 1 epochs. If using 'auto', be careful not to indicate a large number of epochs, as this can take much longer than is necessary (PARROT warns if ``-e`` is above 10).

The following controls are off by default, except for the fixed learning-rate
schedule. See :doc:`training_controls` for choosing settings, loss units and
how the options interact.

	*  ``--gradient-clip-norm`` : Positive, finite global L2 clipping threshold; also enables finite checks.
	*  ``--check-finite-gradients`` : Check for nonfinite losses and gradient norms before an update, without clipping.
	*  ``--equal-protein-weighting`` : Average labelled losses within each protein, then across proteins (residue and matrix tasks only). The default sums labelled losses.
	*  ``--lr-schedule`` : ``fixed`` (default), ``cosine``, ``exponential`` or validation-driven ``plateau``.
	*  ``--lr-anneal-epochs`` : Cosine decay duration after warmup (default: full run minus warmup, at least 1); required for cosine with ``--stop auto``.
	*  ``--lr-min`` : Lower bound for schedules, warmup and recovery (default 0); must not exceed the initial rate.
	*  ``--lr-gamma`` : Exponential decay factor per completed decay epoch (default 0.95).
	*  ``--lr-factor`` : Plateau reduction factor (default 0.5).
	*  ``--lr-patience`` : Tolerated epochs without sufficient improvement before plateau reduces the rate (default 5; reduction after the sixth).
	*  ``--lr-threshold`` : Minimum absolute validation-loss improvement for plateau (default 0.0001).
	*  ``--lr-cooldown`` : Cooldown epochs after a plateau reduction (default 0).
	*  ``--lr-warmup-epochs`` : Linear warmup epochs before the chosen schedule (default 0).
	*  ``--recover-nonfinite`` : Restore a clean epoch boundary, reduce the rate and replay the epoch on numerical failure. Off by default; the flag alone allows two recoveries, or give an integer budget.
	*  ``--recovery-lr-factor`` : Rate multiplier for each recovery (default 0.5); requires recovery enabled.
	*  ``--regression-loss`` : ``mae``, ``mse`` or ``huber`` for neural regression; defaults remain MAE for sequence and MSE for residue/matrix tasks. Incompatible with ``--uncertainties``.
	*  ``--huber-delta`` : Positive finite Huber transition in target units (default 1); requires ``--regression-loss huber``.
	*  ``--optimizer`` : ``adam`` (default) or ``adamw``.
	*  ``--weight-decay`` : Finite nonnegative regularization coefficient (default 0 for both optimizers).
	*  ``--amp`` : Opt-in CUDA mixed precision; ``auto`` (also the flag's value when given alone), ``float16`` or ``bfloat16``. Default ``off``. BRNN training accepts ``off`` only. See :ref:`parrot-amp` for architecture restrictions, hardware checks and scaling behavior.
	*  ``--checkpoint [file]`` : Opt in to resumable atomic disk checkpoints (default filename when enabled: ``<network>_training_checkpoint.pt``).
	*  ``--checkpoint-every`` : Successful epochs between checkpoints (default 1); requires checkpointing or resume.
	*  ``--resume file`` : Restore the saved arguments, split and training state. ``parrot-train --resume file`` is sufficient; see :ref:`parrot-training-resume` for allowed changes.
	*  ``--training-log`` : Flush successful-epoch records to ``<network>_training_history.tsv`` during training, including fixed-rate runs. Off by default.

Whichever stop condition is used, the validation loss is measured after every epoch, and the network from the epoch with the lowest validation loss is the one kept. The network file is rewritten every time the validation loss reaches a new low, so if training is interrupted, the best network so far has already been saved.

**Regression values with uncertainties (every architecture except knn and unet2d):**

	*  ``--uncertainties`` : Every value in the datafile is written as ``value,SD``, where SD is the standard deviation of its measurement error (a missing residue value is still ``nan``). Training then minimises the Gaussian negative log-likelihood of the values, weighting each by 1/(SD² + *s*\ ²), where *s*, the model error, is the error of the network's predictions beyond the values' own errors, fitted on the validation set after every epoch. *s* is printed and recorded, and the test-set statistics also say how well this account of the errors fits (the reduced chi-squared, the fraction of values within 1 and 2 total SDs, √(SD² + *s*\ ²), of their prediction, and the noise ceiling). See the "Values with uncertainties" page. It is an error with ``--classification``.

**Classification of sequences or residues (every architecture except unet2d):**

	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. For sequence data, each sequence's probabilities are written as one value per class; for residue data, each residue's probabilities are written as one comma-separated token (e.g. ``0.9100,0.0600,0.0300``), so there is still one token per residue. Probabilities are written to 4 decimal places. This flag also modifies the output figures and output performance stats (see "Output" below). It is an error with ``--regression``.

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
* **The simple models** (``linear``, ``svm``, ``random_forest``, ``knn``, ``gbrt``) already standardize their own input features, so the option has little effect on them.

The network file records the base encoding and the exact mean and standard deviation of every feature, so ``parrot-predict`` and ``Predictor`` standardize new sequences in exactly the same way, with no flags needed.

**Output:**

After running ``parrot-train``, the network, info file, test predictions, performance statistics and two evaluation tables will be saved (only the first three with ``--no-stats``), named after ``--output-network`` (e.g. ``network.pt``). The network itself can be used with ``parrot-predict``: as well as the network weights, this file records the architecture, its hyperparameters, the datatype, the number of classes and the encoding, so none of these need to be given again when making predictions. "_predictions.tsv" contains the true values and predicted values for all of the sequences in the test set, two lines per sequence: ``<ID>_TRUE <sequence> <true value(s)>`` and then ``<ID>_PRED <sequence> <predicted value(s)>`` (missing residue labels are written as ``nan``). "_performance_stats.txt" quantifies a variety of performance metrics on the test set, pooling every labelled residue for residue data: Pearson's and Spearman's correlation coefficients for regression; the Matthews correlation coefficient, the weighted F1 score and the accuracy for classification; and, with ``--probabilistic-classification``, also the areas under the ROC and precision-recall curves. With ``--uncertainties``, "_performance_stats.txt" also gives the reduced chi-squared, the fraction of test values within 1 and 2 total SDs of their prediction, the noise ceiling and the model error *s* (see the "Values with uncertainties" page). With ``--save-splits``, the split is also saved as "_split_file.txt".

"_info.txt" describes how the network was trained, in enough detail to reproduce it or to describe it in the methods section of a paper: the exact command, the date and the versions of PARROT, Python and the main libraries, the device and the random seed; the datafile (with its SHA-256 checksum, so the exact data can be identified), the task, the number of sequences in each set and how they were split, the sequence lengths and the values or class balance; the encoding and its settings; the architecture, every hyperparameter, the number of parameters and the network's layers; the loss function, the optimizer and its settings, the batch size, the number of epochs and which epoch's network was kept; for ``parrot-optimize``, the hyperparameter search; the validation and test results; and how long training took (in total and per epoch), how many sequences per second the network trained on, and how many it predicts per second (measured on the test set, on the device used).

Regression statistics also include MAE, RMSE, signed mean error (prediction minus
measurement) and R2. Residue and matrix tasks additionally report mean protein
MAE, RMSE and signed error, giving each protein's metric equal weight.
``_per_protein_metrics.tsv`` records each test protein's metrics and labelled
count; ``_length_metrics.tsv`` groups them by sequence length (0-49, 50-99,
100-199, 200-399, 400-799 and 800+, omitting empty bins). Missing labels and
symmetric matrix cells use the same masks as pooled metrics. Undefined metrics
are NaN, including R2 for constant targets. ``--no-stats`` suppresses both
tables as well as ``_performance_stats.txt``.

With a learning-rate schedule, warmup or numerical recovery enabled,
``<network>_training_history.tsv`` records each successful epoch's losses,
learning rate and cumulative recovery count. If recoveries occurred,
``<network>_recovery_events.json`` records their reasons and rate reductions.
These files are written on successful completion, and active settings are
recorded in ``_info.txt``. Clipping, finite checks or equal-protein weighting
alone do not create a history file. ``--training-log`` instead writes a richer
epoch history continuously and preserves it on failure. ``--checkpoint`` writes
separate resumable state, including optimizer and random generators, and
``--resume`` continues from it. See :ref:`parrot-training-diagnostics` and
:ref:`parrot-training-resume`.

Unless ``--silent`` is given, ``parrot-train`` also prints how the data were split, the validation loss as training goes (for the first epoch, every fifth and the last; every epoch with ``--verbose``), which epoch had the lowest validation loss (the network from that epoch is the one saved), the test-set loss and performance metrics, and every file it wrote.

For matrix data, the test-set predictions are instead written to a "_predictions.npz" file, and the figures include heatmaps of example test sequences; see the "Matrix data" page.

If ``--include-figs`` is provided, there will be additional PNG images saved in this directory. The first, with suffix '_train_val_loss.png' displays the network's performance on the training and validation sets over the course of training (the simple models, which are fitted in one step, have no such figure). The other image(s) describes the network performance on the held out test set, and will vary depending on the data format and machine learning task; their names include '_seq' for sequence data and '_res' for residue data. If training a network for a classification task, the image will be a confusion matrix ('_seq_CM.png' or '_res_CM.png'). If training for a regression task, there are two images comparing the predicted and true values of the test set sequences, both labelled with Pearson's r: a scatterplot ('_seq_scatterplot.png' or '_res_scatterplot.png') and a density plot ('_seq_density.png' or '_res_density.png'), a 2D histogram coloured by how many values fall in each bin (on a log scale, or a linear one if no bin holds 10 values). Predictions that are not finite (NaN or infinite, e.g. from a network whose training diverged) cannot be plotted, so they are left out of both, with a warning; the performance metrics still include them. With more than 10,000 test values (e.g. every residue of a few dozen sequences), the scatterplot shows 10,000 of them, spread evenly over the test set (one drawn at random from each of 10,000 equal, consecutive slices of the test values, in test-set order), with a warning; more would merge into a solid block. The density plot always shows every value, and shows where they lie. If using probabilistic-classification mode, then instead of the confusion matrix there will be two output figures: one plotting receiver operator characteristic (ROC) curves ('_ROC_curve.png') and the other plotting precision-recall curves ('_PR_curve.png').

For residue data, '_res_profiles.png' also shows the true (black) and predicted (red) values along six test sequences, one panel each. They are chosen to span the range of accuracy, from the most to the least accurately predicted (at evenly spaced quantiles of each sequence's mean absolute error, or for classification its fraction of residues misclassified, chosen from sequences with at least 30 labelled residues if at least six have that many), and each panel is labelled with its sequence's ID, Pearson's r and the mean absolute error (MAE) between its true and predicted values (or, for classification, its accuracy).
