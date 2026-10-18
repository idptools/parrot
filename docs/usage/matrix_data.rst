=============
 Matrix data
=============

As well as one value per sequence or one value per residue, PARROT can learn a **matrix of values for each sequence**, with one row per residue. There are two shapes:

* **Square maps** (L x L for a sequence of L residues): one value for every pair of residues, such as a contact map, a distance map, or the strength of interaction between two residues. Maps can be symmetric (``--symmetric``) or not.
* **Fixed-width matrices** (L x N, with ``--matrix-width N``): N values for every residue, where the N columns are related and ordered (for example a property measured at N temperatures, or a distribution over N bins), rather than N unrelated features.

Matrix data is the ``matrix`` datatype, and is learnt by the ``unet2d`` architecture, a 2D U-Net (see below): ``parrot-train unet2d`` and ``parrot-optimize unet2d``, which need no ``-d``. It works with every per-residue encoding (not NARDINI features, which describe whole sequences), regression and classification, and the ``parrot-train``, ``parrot-optimize``, ``parrot-predict`` and ``parrot-preprocess`` commands.

Input files
-----------

Matrices are too large to write on one line of a text file, so matrix data comes in two files:

* **the datafile**, with a sequence ID and a sequence on each line (and nothing else):

  .. code-block:: text

      protein1 MKVLAAGIVGLLLA
      protein2 MSDEQKRRAEEL

* **the matrix file**, a NumPy ``.npz`` archive holding each sequence's matrix as an array named by its sequence ID. Each matrix has one row per residue and either one column per residue (square maps) or ``--matrix-width`` columns. Write one from Python with:

  .. code-block:: python

      import numpy as np

      # matrices: a dict mapping each sequence ID to its matrix (a 2D numpy array)
      np.savez_compressed('matrices.npz', **matrices)

Sequence IDs are therefore required (``--exclude-seq-id`` cannot be used), and each must be unique. Every sequence in the datafile needs a matrix whose shape matches its length; the matrix file may also hold matrices for sequences that are not in the datafile, so one matrix file can serve several datafiles (for example, the output of ``parrot-preprocess``).

**Missing values.** A cell with no value (e.g. a pair of residues that was not measured) is written as NaN, for regression and for classification. These cells are ignored in training, in the performance metrics and in the figures. Every matrix needs at least one value that is not missing. Infinite values are an error.

**Classification.** Cells hold class labels 0, 1, ..., ``classes - 1`` (stored as whole numbers; NaN for missing). As for other data, give ``--classification`` with ``--num-classes`` (or ``--regression`` for values).

**Symmetric maps.** With ``--symmetric``, every map must be symmetric (cell (i, j) equal to cell (j, i), with NaN in the same places). Each pair of residues is then counted once in the loss and the metrics (the diagonal and upper triangle), and the network's predictions are made exactly symmetric by averaging each cell with its mirror image. ``--symmetric`` only applies to square maps (with ``--matrix-width`` it is an error).

Training
--------

.. code-block:: bash

    $ parrot-train unet2d --input-file data.tsv --output-network network.pt --split kmer --regression --matrix-file matrices.npz --symmetric --include-figs

The flags for matrix data are:

    *  ``--matrix-file`` : the ``.npz`` matrix file (required).
    *  ``--matrix-width`` : the number of columns, for fixed-width matrices. Leave it out for square maps.
    *  ``--symmetric`` : the maps are symmetric (square maps only).
    *  ``--max-matrix-length`` : square maps only; the longest sequence allowed (default 512). A square map takes memory in proportion to the square of the sequence length: a 512-residue map has about 262,000 cells, and the 2D U-Net with its default settings keeps roughly 1-3 kB per cell while training, so one such sequence needs up to about 1 GB. Longer sequences are an error rather than a crash part-way through training; remove them (e.g. ``parrot-preprocess --max-length``), or raise the limit if your GPU has the memory. Lower ``--batch`` as well when sequences are long.

All other ``parrot-train`` flags work as usual, except ``--exclude-seq-id`` (see above) and ``--probabilistic-classification``, which ``unet2d`` does not offer (``Predictor`` gives class probabilities; see "Predicting" below). ``parrot-optimize`` takes the same flags and searches the 2D U-Net's hyperparameters (below), including with ``--cv-folds``.

The unet2d architecture
-----------------------

``unet2d`` is a 2D U-Net, after the one written by Nick on the ``parrot-lightning`` branch. It works in three stages:

1. **Building the input grid.** Each residue's encoding is projected to ``--pair-channels`` features by a learned layer. For square maps, cell (i, j) of the grid holds the features of residues i and j, either side by side (``--pair-features concatenate``, the default) or as every product of one with the other (``outer``; ``pair_channels``\ :sup:`2` features, so keep ``--pair-channels`` small), plus how far apart the two residues are along the sequence, as log(1 + \|i - j\|). For fixed-width matrices, cell (i, k) holds residue i's features and a learned description of column k.
2. **The U-Net.** The encoder halves the grid in both directions ``--depth`` times while doubling the number of channels; the decoder restores it, combining features from the matching encoder level at each step (skip connections). Each output cell can therefore combine fine local detail with context from a large part of the map.
3. **The readout.** A final layer turns each cell's features into its predicted value (or class scores), made symmetric for symmetric maps.

Its hyperparameters are:

	*  ``--pair-channels`` : features each residue is projected to before they are combined into the grid (default 16; searched over 4-32).
	*  ``--pair-features`` : square maps only (it has no effect on fixed-width matrices); ``concatenate`` or ``outer`` (default concatenate; both are searched for square maps).
	*  ``--base-channels`` : channels at full resolution, doubled at each level down (default 16; searched over 8-32).
	*  ``--depth`` : number of times the grid is halved (default 3; searched over 1-4).
	*  ``--kernel-size`` : width of each convolution; one of 3, 5, 7, 9, 11 (default 3; searched over 3, 5 and 7, since a 2D convolution's weights grow with the square of its width: the largest network searched has about 41 million parameters).
	*  ``--dropout`` : dropout probability, at least 0 and below 1 (default 0.0; searched over 0.0-0.5).
	*  ``--upsampling`` : how the decoder doubles the grid, ``transposed`` (a learned transposed convolution) or ``nearest`` (repeating each cell) (default transposed; both are searched).

Compared with Nick's original, the input grid is built from the per-residue encoding (so any of PARROT's per-residue encodings can be used) rather than from a one-hot code of residue pairs, layer normalization replaces batch normalization, padded cells are re-zeroed after every layer, and the number of levels is a hyperparameter. Together these make a sequence's prediction independent of how much padding it is given when batched with longer sequences (up to float32 rounding), and let sequences of any length be used.

Output
------

``parrot-train`` and ``parrot-optimize`` write the same files as for other data, except that:

* the test-set predictions are written to ``<output>_predictions.npz``: one predicted matrix per test sequence, named by its sequence ID (the predicted class of each cell, for classification). Load them with ``np.load('network_predictions.npz')['protein1']``; the true matrices are in your matrix file.
* the performance metrics in ``<output>_performance_stats.txt`` pool every cell of every test sequence (counting each pair of residues once for symmetric maps, and leaving out missing cells).
* with ``--include-figs``, the test-set figures are a scatterplot and a density plot (``_matrix_scatterplot.png`` and ``_matrix_density.png``, labelled with Pearson's r) or a confusion matrix (``_matrix_CM.png``) over those same cells (a scatterplot of more than 10,000 cells shows 10,000 of them spread evenly over the test set, with a warning; the density plot and the metrics always use every cell), and ``_matrix_examples.png`` shows the true and predicted matrices of three test sequences (the shortest, the longest, and one in between) side by side, on the same colour scale.

Predicting
----------

``parrot-predict`` reads the datatype from the saved network and, for a matrix network, writes a ``.npz`` archive (adding ``.npz`` to the output name if it does not end with it) with one predicted matrix per input sequence, named by its sequence ID, or, with ``--exclude-seq-id``, by its position in the file (``1``, ``2``, ...). Sequence IDs must then be unique. From Python, ``Predictor.predict(sequence)`` returns the predicted matrix (for classification, the probability of each class in every cell, with shape [length, width, classes]).

Preprocessing
-------------

``parrot-preprocess -d matrix`` cleans the datafile, removes duplicates and splits it (including homology splits) as for other data. Each sequence keeps its ID, so the original matrix file works with the processed datafile and its split-file unchanged. Give ``--matrix-file`` (and ``--matrix-width`` for fixed-width data) to have the output checked against it. Options that change values (``--recode-classes``, ``--class-balance`` and ``--deduplicate merge``) do not apply to matrix data, whose values are in the matrix file. ``parrot-cvsplit`` works on matrix datafiles as it does on any other.
