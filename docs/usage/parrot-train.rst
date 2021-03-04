parrot-train
============

``parrot-train`` is a command for training a BRNN with specified parameters. The user specifies the parameters and data and a trained BRNN will be saved to disk along with a plot that provides a performance estimate.

Once PARROT is installed, the user can run ``parrot-train`` from the command line:

.. code-block:: bash
	
	$ parrot-train data_file output_network <flags>

Where `data_file` specifies the path to the whitespace-separated datafile and `output_network` is the path to where the final trained network will be saved on disk. Of note, the output images from this script will be saved to the same directory as `output_network`.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--learning-rate`` / ``-lr`` : Learning rate of the network (default is 0.001). Must be a float between 0 and 1.
	*  ``--num-layers`` / ``-nl`` : Number of hidden layers in the BRNN (default is 1). Must be a positive integer.
	*  ``--hidden-size`` / ``-hs`` : Size of hidden vectors within the BRNN (default is 5). Must be a positive integer.
	*  ``--batch`` / ``-b`` : Training batch size (default is 32). Must be a positive integer, and ideally should be in the range of 50-500. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance.
	*  ``--epochs`` / ``-e`` : Number of training epochs (default is 30). Function varies depending on ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until performance as plateaued for ``-e`` epochs.
	*  ``--split`` : Path to split_file for manually dividing dataset into training, validation and test sets. The file should contain three lines, corresponding to training, validation and test sets respectively. Each line should have integers separated by whitespace, with the integers specify which sequences/lines in the `datafile` (0-indexed) will belong to which dataset.
	*  ``--set-fractions`` : Include this flag to manually set the proportions of the data belonging to the training, validation and test sets. This option must be followed by three floats (representing train, validation, and test) between 0 and 1 that cumulatively sum to 1.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics' or user-specified. If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping.
	*  ``--exclude-seq-id`` : Include this flag if the `data_file` is formatted such that it does not have sequence IDs as the first column in each row.
	*  ``--proportional-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the proportion to which the network perceive the input sample as belonging to class 0 vs class 1. Only implemented for sequence binary classification.
	*  ``--include-figs`` : Include this flag to generate default images based on network training and performance on test set. Figures will be saved to same directory as specified by ``output-network``.
	*  ``--force-cpu`` : Include this flag to force network training on the CPU, even if a GPU is available. Use if your machine has a GPU, but the GPU has insufficient memory.
	*  ``--verbose`` / ``-v`` : The level of information that should be printed to console during training. There will be no output if this flag is not included, and maximum output if this flag is included twice or more.

**Output:**

After running ``parrot-train``, several files will be saved to disk. Firstly, the saved network weights from the training process will be located at the path provided by `output_network`. Additionally, there will be two PNG images saved to this same directory. The first, called 'train_test.png' displays the network's performance on the training and validation sets over the course of training. The second image describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, the image will be a scatterplot comparing the predicted and true values of the test set sequences. Finally, the file "test_set_predictions.tsv" will be saved into this same directory. This file contains both the true values (provided in the original data_file) and predicted values for each of the sequences within the test set.

Output text detailing network performance across training can be printed to console if the ``--verbose`` flag is provided.
