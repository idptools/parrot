================
 parrot-cvsplit
================

``parrot-cvsplit`` is a command for generating PARROT-usable *split-files* for conducting K-fold cross validation on a dataset (where K is specified by the user). This function does not extensively validate the data it's passed and assumes that it a PARROT-usable datafile.

Once PARROT is installed, the user can run ``parrot-cvsplit`` from the command line:

.. code-block:: bash
    
    $ parrot-cvsplit data_file output_splitfiles <flags>

Where *data_file* specifies the path to the whitespace-separated datafile and *output_splitfiles* denotes where the output split-files will be saved from this command. Files are output as <output_splitfiles>_cv#.txt for # = 0 ... K-1.

**Optional flags:**

    *  ``--help`` / ``-h`` : Display a help message.
    *  ``--k-folds`` / ``-k`` : Number of split-files to generate for K-fold cross-validation (default is 10). This term determines the train+validation vs test set split, as for each fold the test set will contain ~1/K of the data, while the training and validation sets combined will contain (K-1)/K. Must be a positive integer.
    *  ``--training-fraction`` / ``-t`` : Percent of the non-test data that should be partioned into the training set for each fold (default is 0.8). This term determines the train vs validation set split, as for each fold the training set fraction will equal ~((K-1)/K) * *training-fraction* of the data, and the validation set fraction will equal ~((K-1)/K) * (1-*training-fraction*). Must be a float between 0 and 1.

**Output:**

``parrot-cvsplit`` will generate K files to the specified location by *output_splitfiles*. Each file is a split-file designed to be used with ``parrot-train`` along with the ``--split`` flag.
