================
 parrot-cvsplit
================

``parrot-cvsplit`` is a command for generating PARROT-usable *split-files* for conducting K-fold cross validation on a dataset (where K is specified by the user). This function does not extensively validate the data it's passed (it only reads the sequence on each line, and stops with a message if a line has no sequence, or a number where its sequence should be) and assumes that it is a PARROT-usable datafile. Identical sequences are always put in the same set of every fold (so no test sequence also appears in that fold's training data). Similar but not identical sequences can still end up in different sets; to keep them apart, make the split with ``parrot-preprocess --split homology`` or ``--split kmer``.

Once PARROT is installed, the user can run ``parrot-cvsplit`` from the command line:

.. code-block:: bash

    $ parrot-cvsplit data_file output_splitfiles <flags>

Where *data_file* specifies the path to the whitespace-separated datafile and *output_splitfiles* is the path and name prefix of the output split-files. Files are output as <output_splitfiles>_cv#.txt for # = 0 ... K-1 (a final ``.txt`` on *output_splitfiles* is dropped, so ``splits.txt`` gives ``splits_cv0.txt``, and any other dots are kept). The directory they go in must already exist.

**Optional flags:**

    *  ``--help`` / ``-h`` : Display a help message.
    *  ``--show-examples`` : Print a longer guide to ``parrot-cvsplit``: examples of how to use it, what it does and how it works.
    *  ``--exclude-seq-id`` : The datafile has no sequence IDs: each line starts with the sequence (as for ``parrot-train --exclude-seq-id``). Without it, the second column is taken for the sequence, so a datafile without IDs is refused with a message suggesting this flag.
    *  ``--k-folds`` / ``-k`` : Number of split-files to generate for K-fold cross-validation (default is 10). This term determines the train+validation vs test set split, as for each fold the test set will contain ~1/K of the data, while the training and validation sets combined will contain (K-1)/K. Must be an integer of at least 2, and no more than the number of distinct sequences in the datafile.
    *  ``--seed`` : Integer random seed, to make the generated split-files reproducible. By default no seed is set.
    *  ``--training-fraction`` / ``-t`` : Fraction of the non-test data that should be partitioned into the training set for each fold (default is 0.8). This term determines the train vs validation set split, as for each fold the training set fraction will equal ~((K-1)/K) * *training-fraction* of the data, and the validation set fraction will equal ~((K-1)/K) * (1-*training-fraction*). Must be a number strictly between 0 and 1.

**How the folds are made:**

The sequences are shuffled (repeatably, with ``--seed``) and divided into K folds of near-equal size, keeping identical sequences in the same fold, so every sequence is in exactly one test set. Fold *i* is the test set of split-file *i*, and the rest of the data is divided at random between its training set and its validation set, again keeping identical sequences together. If any set of any fold would be empty (too few sequences for the number of folds and the training fraction), ``parrot-cvsplit`` stops with an error before writing anything.

To score hyperparameters by cross-validation rather than to estimate how well a network does, use ``parrot-optimize --cv-folds`` instead (see "parrot-optimize").

**Output:**

``parrot-cvsplit`` will generate K files to the specified location by *output_splitfiles*, and print the name of each. Each file is a split-file designed to be used with ``parrot-train`` along with the ``--split`` flag: three lines, giving the training, validation and test sets as 0-indexed line numbers of the datafile (skipping blank and comment lines, as ``parrot-train`` does when it reads the datafile). See "Evaluating a Network with Cross-Validation" for a full example.
