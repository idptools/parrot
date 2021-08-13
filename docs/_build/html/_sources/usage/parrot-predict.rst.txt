================
 parrot-predict
================

``parrot-predict`` is a command for making predictions using a trained PARROT network. The ``parrot-train`` and ``parrot-optimize`` commands both output a file with trained network weights, and this trained network can be used by ``parrot-predict`` to make new predictions on unlabeled sequences. The prediction will be output as a text file saved to a specified location. Note that this command will only make predictions for `non-redundant` sequences in the provided file. Currently, users must input the hyperparameters (--num-layers and --hidden-size) they used to train their network originally, but in future versions of PARROT, ``parrot-predict`` will be able to dynamically read in your saved network and automatically detect these hyperparameters.

Once PARROT is installed, the user can run ``parrot-predict`` from the command line:

.. code-block:: bash
	
	$ parrot-predict seq_file saved_network output_file <flags>

Where `seq_file` specifies a file containing a list of sequences. Each line of `seq_file` should have two whitespace-separated columns: a sequence ID and the amino acid sequence. Optionally, the file may also be formatted without the sequence IDs. Two example `seq_file` can be found in the **/data** folder. `saved_network` is the path to where the trained network is saved in memory. `output_file` is the path to where the predictions will be saved as a text file.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--num-layers`` / ``-nl`` : Number of hidden layers in the network (default is 1). Must be a positive integer and must be identical to the number of layers used when the network was trained.
	*  ``--hidden-size`` / ``-hs`` : Size of hidden vectors within the network (default is 10). Must be a positive integer and must be identical to the hidden size used when the network was trained.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics' or user-specified. If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. The encoding scheme used for sequence prediction must be identical to that used for network training.
	*  ``--exclude-seq-id`` : Include this flag if the `seq_file` is formatted without sequence IDs as the first column in each row.
	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. Currently only implemented for sequence classification. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--silent`` : Flag which, if provided, ensures no output is generated to the terminal.
	*  ``--print-frequency`` : Value that defines how often status updates should be printed (in number of sequences predicted. Default=1000

**Output:**

``parrot-predict`` will produce a single text file as output, as well as status updates to the console (if ``--silent`` is not specified). This file will be formatted similarly to the original datafiles used for network training: each row contains a sequence ID (exluded if the flag ``--exclude-seq-id`` is given), an amino acid sequence, and the prediction values for that sequence.
