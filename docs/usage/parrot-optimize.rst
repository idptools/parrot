=================
 parrot-optimize
=================

``parrot-optimize`` is a command for performing hyperparameter optimization on a given dataset and then training a PARROT network with the optimal hyperparameters. The dataset is divided 5-fold and iteratively retrained and validated using cross-validation to estimate the performance of the network using a given set of hyperparameters. Bayesian optimization ensures that in each iteration, hyperparameters are selected that better sample hyperparameter space and/or are expected to improve performance. After a specified number of iterations, the algorithm returns the best-performing hyperparameters and a new network will be trained on the cumulative cross-validation data, then tested on held out test data. Because of its iterative, cross-validation-based procedure, using ``parrot-optimize`` takes **significantly** longer to train a network than ``parrot-train``, and is not recommended for basic usage and for users running PARROT on computers lacking GPUs. The final trained network will be saved to to memory along with a file specifying the optimal hyperparameters and several estimates of network performance on a held-out test set (identically to ``parrot-train``).

NOTE: there are slightly different instructions for installing optimize-compatible PARROT. See the "Getting Started with PARROT" page for more information. Once PARROT is installed, the user can run ``parrot-optimize`` from the command line:

.. code-block:: bash
	
	$ parrot-optimize data_file output_network <flags>

Where `data_file` specifies the path to the whitespace-separated datafile and `output_network` is the path to where the final trained network will be saved. It is recommended, but not required, that networks are saved using a ".pt" file extension, following the PyTorch convention.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required.
	*  ``--epochs`` / ``-e`` : Number of epochs to train for during each iteration of optimization on each data fold (default is 100).
	*  ``--max-iter`` : Maximum number of iterations the optimization algorithm should run for. Default is 75.
	*  ``--split`` : Path to a "split-file" for manually dividing dataset into training, validation and test sets. The file should contain three lines, corresponding to training, validation and test sets respectively. Each line should have integers separated by whitespace, with the integers specify which sequences/lines in the `datafile` (0-indexed) will belong to which dataset. See **/data** folder for examples. If a split-file is not provided, default behavior is for PARROT to randomly divide data into training, validation and test sets. The cross-validation folds will be divided from the training and validation data.
	*  ``--set-fractions`` : Include this flag to manually set the proportions of the data belonging to the training, validation and test sets. This option must be followed by three floats (representing train, validation, and test) between 0 and 1 that cumulatively sum to 1. By default, PARROT uses splits of 70:15:15. Note that the ``--split`` flag overrides these values.
	*  ``--save-splits`` : Include this flag if you would like PARROT to produce a split-file based on its random partitioning of data into training/validation/test sets, which can be useful for replication and/or testing multiple networks on the same data. Output split-file will be saved in the same folder as *output_network* using the same name followed by "_split_file.txt". This flag is overridden if a split-file is manually provided. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics' or user-specified. If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. Example encoding files are provided in the **/data** folder.
	*  ``--exclude-seq-id`` : Include this flag if the `data_file` is formatted without sequence IDs as the first column in each row.
	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. Currently only implemented for sequence classification. This flag also modifies the output figures and output performance stats. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--include-figs`` : Include this flag to generate images based on network training and performance on test set. Figures will be saved to same directory as specified by *output_network* using same naming convention.
	*  ``--no-stats`` : Include this flag to prevent a "_performance_stats.txt" file from being output.
	*  ``--ignore-warnings`` : By default, PARROT checks your data for a few criteria and prints warnings if it doesn't meet some basic heuristics. Use this flag to silence these warnings (network training occurs unimpeded in either case).
	*  ``--force-cpu`` : Include this flag to force network training on the CPU, even if a GPU is available.
	*  ``--verbose`` / ``-v`` : Include this flag to produce more descriptive output to terminal.
	*  ``--silent`` : Include this flag to produce no output to terminal.

**Output:**

``parrot-optimize`` will produce similar output as ``parrot-train``, with one exception. ``parrot-optimize`` also produces a file "optimal_hyperparams.txt" which specifies which hyperparameters were ultimately chosen for the final network.
