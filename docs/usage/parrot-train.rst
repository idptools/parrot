parrot-train
============

``parrot-train`` is a command for training a BRNN with specified parameters. The user specifies the parameters and data and a trained BRNN will be saved to disk along with a plot that provides a performance estimate.

Once PARROT is installed, the user can run ``parrot-train`` from the command line:

.. code-block::
	
	$ parrot-train data_file output_network <flags>

Where `data_file` specifies the path to the whitespace-separated datafile and `output_network` is the path to where the final trained network will be saved on disk. Of note, the output images from this script will be saved to the same directory as `output_network`.

**Required flags:**

	*  ``--datatype`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``-nc`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``-lr`` : Learning rate of the network (default is 0.001). Must be a float between 0 and 1.
	*  ``-nl`` : Number of hidden layers in the BRNN (default is 1). Must be a positive integer.
	*  ``-hs`` : Size of hidden vectors within the BRNN (default is 5). Must be a positive integer.
	*  ``-b`` : Training batch size (default is 32). Must be a positive integer, and ideally should be in the range of 50-500. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance.
	*  ``-e`` : Number of training epochs (default is 30). Function varies depending on ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until performance as plateaued for ``-e`` epochs.
	*  ``--split`` : Path to split_file for manually dividing dataset into training, validation and test sets. The file should contain three lines, corresponding to training, validation and test sets respectively. Each line should have integers separated by whitespace, with the integers specify which sequences/lines in the `datafile` (0-indexed) will belong to which dataset.
	*  ``--setFractions`` : Include this flag to manually set the proportions of the data belonging to the training, validation and test sets. This option must be followed by three floats (representing train, validation, and test) between 0 and 1 that cumulatively sum to 1.
	*  ``--excludeSeqID`` : Include this flag if the `data_file` is formatted such that it does not have sequence IDs as the first column in each row.
	*  ``--encodeBiophysics`` : Include this flag if you wish to represent each amino acid as a length ## vector representing biophysical properties, rather than a length 20 one-hot vector.
	*  ``--forceCPU`` : Include this flag to force network training on the CPU, even if a GPU is available. Use if your machine has a GPU, but the GPU has insufficient memory.
	*  ``--verbose`` / ``-v`` : The level of information that should be printed to console during training. There will be no output if this flag is not included, and maximum output if this flag is included twice or more.

**Output:**

After running ``parrot-train``, several files will be saved to disk. Firstly, the saved network weights from the training process will be located at the path provided by `output_network`. Additionally, there will be two PNG images saved to this same directory. The first, called 'train_test.png' displays the network's performance on the training and validation sets over the course of training. The second image describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, the image will be a scatterplot comparing the predicted and true values of the test set sequences.

Output text detailing network performance across training can be printed to console if the ``--verbose`` flag is provided.


