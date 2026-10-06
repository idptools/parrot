================
 parrot-predict
================

``parrot-predict`` is a command for making predictions using a trained PARROT network. The ``parrot-train`` and ``parrot-optimize`` commands both output a file with trained network weights, and this trained network can be used by ``parrot-predict`` to make new predictions on unlabeled sequences. The prediction will be output as a text file saved to a specified location. Networks saved by this version of PARROT record their architecture, hyperparameters, datatype, number of classes and encoding, so ``parrot-predict`` reads all of these from the saved network. Networks saved by earlier versions of PARROT only contain their weights; for those, ``--datatype`` must be given (and ``--encode``, if the network was not trained with one-hot encoding). The network size is always read from the saved weights.

Once PARROT is installed, the user can run ``parrot-predict`` from the command line:

.. code-block:: bash
	
	$ parrot-predict seq_file saved_network output_file <flags>

Where `seq_file` specifies a file containing a list of sequences. Each line of `seq_file` should have two whitespace-separated columns: a sequence ID and the amino acid sequence. Optionally, the file may also be formatted without the sequence IDs. Blank lines and comment lines (starting with '#') are ignored. Two example `seq_file` can be found in the **/data** folder. `saved_network` is the path to where the trained network is saved in memory. `output_file` is the path to where the predictions will be saved as a text file.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--datatype`` / ``-d`` : 'sequence', 'residues' or 'matrix'. Required only for networks saved by earlier versions of PARROT (which are always 'sequence' or 'residues' networks); otherwise, if given, it is checked against the saved network.
	*  ``--classes`` / ``-c`` : The number of classes (1 for regression). If given, it is checked against the saved network.
	*  ``--encode`` : The encoding the network was trained with ('onehot', 'biophysics' or the path to a user encoding file). Only needed for networks saved by earlier versions of PARROT that were not trained with one-hot encoding; otherwise, if given, it is checked against the saved network.
	*  ``--num-layers`` / ``-nl`` and ``--hidden-size`` / ``-hs`` : No longer needed; accepted (and ignored) so that existing commands keep working.
	*  ``--exclude-seq-id`` : Include this flag if the `seq_file` is formatted without sequence IDs as the first column in each row.
	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. Currently only implemented for sequence classification. (NOTE: This is a new feature, let us know if you run into any issues!)
	*  ``--embedding-cache`` : For networks trained on a protein language model encoding: directory in which to store the embeddings of the sequences being predicted (default: a ``parrot_predict_embeddings`` directory next to *output_file*). Embedding is the slow step of predicting with such a network, so reusing a cache (e.g. one created by ``parrot-train``) avoids re-embedding sequences. See "Protein language model encodings".
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--force-cpu`` : Same as ``--device cpu`` (kept so that older scripts keep working).
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``).
	*  ``--silent`` : Flag which, if provided, ensures no output is generated to the terminal.
	*  ``--print-frequency`` : Value that defines how often status updates should be printed (in number of sequences predicted). Default=1000

**Output:**

``parrot-predict`` will produce a single text file as output, as well as status updates to the console (if ``--silent`` is not specified). This file will be formatted similarly to the original datafiles used for network training: each row contains a sequence ID (exluded if the flag ``--exclude-seq-id`` is given), an amino acid sequence, and the prediction values for that sequence. There is one output row for every sequence in `seq_file`, in the same order, so if the same sequence appears more than once (e.g. under different IDs) it will appear more than once in the output.

For networks trained on matrix data, the predictions are instead written as a NumPy ``.npz`` archive with one predicted matrix per sequence, named by its sequence ID (``.npz`` is added to `output_file` if it does not end with it); see the "Matrix data" page.
