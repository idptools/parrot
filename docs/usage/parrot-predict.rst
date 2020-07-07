parrot-predict
==============

``parrot-predict`` is a command for making predictions using a trained BRNN. The ``parrot-train`` and ``parrot-optimize`` commands both save a trained network to disk, and this trained network can be applied to unlabeled sequences to make predictions for whatever machine learning task the network was trained on. The prediction will be saved to memory in a text file.

Once PARROT is installed, the user can run ``parrot-predict`` from the command line:

.. code-block:: bash
	
	$ parrot-predict seq_file saved_network output_file <flags>

`seq_file` specifies the path to the file containing the list of sequences. Each line of `seq_file` should have two whitespace-separated columns: a sequence ID and the amino acid sequence. Optionally, the file may also be formatted without the sequence IDs. Two example `seq_file` can be found in the **data** folder. `saved_network` is the path to where the trained network is saved in memory. `output_file` is the path to where the predictions will be saved as a text file.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--num-layers`` / ``-nl`` : Number of hidden layers in the BRNN (default is 1). Must be a positive integer and must be identical to the number of layers used when the network was trained.
	*  ``--hidden-size`` / ``-hs`` : Size of hidden vectors within the BRNN (default is 5). Must be a positive integer and must be identical to the hidden size used when the network was trained.
	*  ``--encode`` : Include this flag to specify the numeric encoding scheme for each amino acid. Available options are 'onehot' (default), 'biophysics' or user-specified. If you wish to manually specify an encoding scheme, provide a path to a text file describing the amino acid to vector mapping. The encoding scheme used for sequence prediction must be identical to that used for network training.
	*  ``--excludeSeqID`` : Include this flag if the `seq_file` is formatted such that it does not have sequence IDs as the first column in each row.

**Output:**

``parrot-predict`` will produce a single text file as output. This file will be formatted similarly to the original datafiles used for network training: each row contains a sequence ID (exluded if ``--excludeSeqID1`` is provided), an amino acid sequence, and the prediction values for that sequence.
