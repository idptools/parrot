brnn_optimize
=============

``brnn_optimize`` is a command for performing hyperparameter optimization on a given dataset. The dataset is divided 5-fold and iteratively retrained and validated using cross validation to estimate the performance of the network using a given set of hyperparameters. Bayesian optimization ensures that in each iteration, hyperparameters are selected that better sample hyperparameter space and/or are expected to improve performance. After 80 iterations, the algorithm will return the best-performing hyperparameters and a new network will be trained on the cumulative cross-validation data, then tested on held out test data. The final trained BRNN will be saved to disk along with a plot that provides a performance estimate.

Once prot_brnn is installed, the user can run ``brnn_optimize`` from the command line:

.. code-block::
	
	$ brnn_optimize data_file output_network <flags>

Where `data_file` specifies the path to the whitespace-separated datafile and `output_network` is the path to where the final trained network will be saved on disk. Of note, the output images from this script will be saved to the same directory as `output_network`.

**Required flags:**

	*  ``--datatype`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``-nc`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``-b`` : Training batch size (default is 32). Must be a positive integer, and ideally should be in the range of 50-500. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance.
	*  ``-e`` : Number of epochs to train for during each iteration of optimization on each data fold (default is 200).
	*  ``--split`` : Path to split_file for manually dividing dataset into training, validation and test sets.
	*  ``--setFractions`` : Include this flag to manually set the proportions of the data belonging to the training, validation and test sets. This option must be followed by three floats (representing train, validation, and test) between 0 and 1 that cumulatively sum to 1.
	*  ``--excludeSeqID`` : Include this flag if the `data_file` is formatted such that it does not have sequence IDs as the first column in each row.
	*  ``--encodeBiophysics`` : Include this flag if you wish to represent each amino acid as a length ## vector representing biophysical properties, rather than a length 20 one-hot vector.
	*  ``--verbose`` / ``-v`` : The level of information that should be printed to console during training. There will be no output if this flag is not included, and maximum output if this flag is included twice or more.

**Output:**

``brnn_optimize`` will produce similar output as ``brnn_train``. Firstly, the saved network weights from the training process will be located at the path provided by `output_network`. Additionally, there will be two PNG images saved to this same directory. The first, called 'train_test.png' displays the network's performance on the training and validation sets over the course of training. The second image describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, the image will be a scatterplot comparing the predicted and true values of the test set sequences.

Output text detailing network performance across training can be printed to console if the ``--verbose`` flag is provided.

.. toctree::
   :maxdepth: 2
   :caption: Contents: