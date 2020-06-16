brnn_predict
============

``brnn_predict`` is a command for making predictions using a trained BRNN. The ``brnn_train`` and ``brnn_optimize`` commands both save a trained network to disk, and this trained network can be applied to unlabeled sequences to make predictions for whatever machine learning task the network was trained on. The prediction will be saved to memory in a text file.

Once prot_brnn is installed, the user can run ``brnn_predict`` from the command line:

.. code-block::
	
	$ brnn_predict seq_file saved_network output_file <flags>

`seq_file` specifies the path to the file containing the list of sequences. Each line of `seq_file` should have two whitespace-separated columns: a sequence ID and the amino acid sequence. Optionally, the file may also be formatted without the sequence IDs. Two example `seq_file` can be found in the **data** folder. `saved_network` is the path to where the trained network is saved in memory. `output_file` is the path to where the predictions will be saved as a text file.

**Required flags:**

	*  ``--datatype`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **data** folder for more information.
	*  ``-nc`` : The number of classes for the machine learning task. If the task is regression, then specify '1'.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``-nl`` : Number of hidden layers in the BRNN (default is 1). Must be a positive integer and must be identical to the number of layers used when the network was trained.
	*  ``-hs`` : Size of hidden vectors within the BRNN (default is 5). Must be a positive integer and must be identical to the hidden size used when the network was trained.
	*  ``--excludeSeqID`` : Include this flag if the `seq_file` is formatted such that it does not have sequence IDs as the first column in each row.
	*  ``--encodeBiophysics`` : Include this flag if you wish to represent each amino acid as a length 4 vector representing biophysical properties, rather than a length 20 one-hot vector. This flag must be included if the network was trained with biophysical encoding.

**Output:**

``brnn_predict`` will produce a single text file as output. This file will be formatted similarly to the original datafiles used for network training: each row contains a sequence ID (exluded if ``--excludeSeqID1`` is provided), an amino acid sequence, and the prediction values for that sequence.
