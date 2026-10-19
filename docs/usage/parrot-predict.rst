================
 parrot-predict
================

``parrot-predict`` is a command for making predictions using a trained PARROT network. The ``parrot-train`` and ``parrot-optimize`` commands both output a file with trained network weights, and this trained network can be used by ``parrot-predict`` to make new predictions on unlabeled sequences. The prediction will be output as a text file saved to a specified location. Networks saved by this version of PARROT record their architecture, hyperparameters, datatype, number of classes and encoding, so ``parrot-predict`` reads all of these from the saved network. Networks saved by earlier versions of PARROT only contain their weights; for those, ``--datatype`` must be given (and ``--encode``, if the network was not trained with one-hot encoding). The network size is always read from the saved weights.

Once PARROT is installed, the user can run ``parrot-predict`` from the command line:

.. code-block:: bash
	
	$ parrot-predict seq_file saved_network output_file <flags>

Where `seq_file` specifies a file containing a list of sequences. Each line of `seq_file` should have two whitespace-separated columns: a sequence ID and the amino acid sequence. Optionally, the file may also be formatted without the sequence IDs. Blank lines and comment lines (starting with '#') are ignored. A line with any other number of columns (e.g. a datafile, with values after each sequence) stops the command with a message. Two example `seq_file` can be found in the **/data** folder. `saved_network` is the path to the trained network file (the ``.pt`` file written by ``parrot-train`` or ``parrot-optimize``). `output_file` is the path to where the predictions will be saved as a text file; its directory must already exist.

Every sequence is checked against the network's encoding before anything is predicted, so a residue the encoding cannot read stops the command at once.

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--show-examples`` : Print a longer guide to ``parrot-predict``: examples of how to use it, what it does and how it works.
	*  ``--datatype`` / ``-d`` : 'sequence', 'residues' or 'matrix'. Required only for networks saved by earlier versions of PARROT (which are always 'sequence' or 'residues' networks); otherwise, if given, it is checked against the saved network.
	*  ``--classes`` / ``-c`` : The number of classes (1 for regression). If given, it is checked against the saved network.
	*  ``--encode`` : The encoding the network was trained with ('onehot', 'biophysics', the path to a user encoding file, a protein language model such as 'esm2_t12_35M_UR50D', or one of the NARDINI encodings such as 'nardini' or 'onehot+nardini'). Only needed for networks saved by earlier versions of PARROT that were not trained with one-hot encoding; otherwise, if given, it is checked against the saved network (for a network trained with ``--standardize-encoding``, against the encoding it standardizes).
	*  ``--num-layers`` / ``-nl`` and ``--hidden-size`` / ``-hs`` : No longer needed; accepted (and ignored) so that existing commands keep working.
	*  ``--exclude-seq-id`` : Include this flag if the `seq_file` is formatted without sequence IDs as the first column in each row.
	*  ``--probabilistic-classification`` : Include this flag to output class predictions as continuous values [0-1], based on the probability that the input sample belongs to each class. For sequence data, each sequence's probabilities are written as one value per class; for residue data, each residue's probabilities are written as one comma-separated token (e.g. ``0.9100,0.0600,0.0300``), so there is still one token per residue. Probabilities are written to 4 decimal places. It can be used whether or not the network was trained with it, but only for sequence and residue classification networks (it is an error with a regression or matrix network).
	*  ``--embedding-cache`` : For networks trained on a protein language model encoding or NARDINI features (it is an error with any other encoding): directory in which to store the embeddings (or features) of the sequences being predicted (default: a ``parrot_predict_embeddings`` directory next to *output_file*). Encoding is the slow step of predicting with such a network, so reusing a cache (e.g. one created by ``parrot-train``) avoids encoding sequences again. See "Protein language model encodings" and "NARDINI features".
	*  ``--embedding-context`` / ``--embedding-context-map`` : Required for networks trained with them, unless ``--embed-in-isolation`` is given (and an error with any other network). Embed each sequence within the full-length protein it was cut from, as the training sequences were: the FASTA file of full-length proteins, and a map with one ``<sequence ID> <full-length ID> [start]`` line per sequence, matched to `seq_file` by sequence ID (so they cannot be used with ``--exclude-seq-id``). A table of where each sequence is embedded is written as ``<output>_embedded_context.tsv``. See "Embedding fragments within their full-length proteins" on the "Protein language model encodings" page.
	*  ``--embed-in-isolation`` : For a network trained with ``--embedding-context``, embed each sequence on its own instead (e.g. new sequences with no full-length protein). Their embeddings then differ from those the network was trained on, so predictions may be less accurate. It is an error for any other network, or with ``--embedding-context``.
	*  ``--device`` : Device to run on: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``. ``auto`` uses an NVIDIA GPU (CUDA) if one is available, otherwise an Apple-silicon GPU (MPS) if available, otherwise the CPU. Requesting a device that is not available is an error.
	*  ``--gpu-id`` : Index of the CUDA device to use, for machines with more than one NVIDIA GPU (e.g. ``--gpu-id 1``). Giving it asks for CUDA, so it is an error with ``--device cpu`` or ``mps``, or on a machine without CUDA. In Python, ``Predictor(path, gpu_id=1)`` does the same.
	*  ``--verbose`` / ``-v`` : Print every setting before predicting: the files, the network's datatype, task, encoding, architecture and hyperparameters, and the options in use.
	*  ``--silent`` : Flag which, if provided, ensures no output is generated to the terminal (even with ``--verbose``).
	*  ``--print-frequency`` : Value that defines how often status updates should be printed (in number of sequences predicted; at least 1). Default=1000. Each distinct sequence is predicted once, so if any sequence appears more than once in `seq_file`, the updates count distinct sequences (and say so).
	*  ``--batch`` / ``-b`` : Number of sequences predicted at once (default 64). Padding is masked, so batching changes the predictions only by float32 rounding (which sequences share a batch changes the order of the floating-point operations); mainly it sets speed and memory use, so a network that only fitted in memory with a small ``--batch`` when it was trained (e.g. a large ``unet2d``) may need a small one here too.

**Output:**

``parrot-predict`` will produce a single text file as output, as well as status updates to the console (if ``--silent`` is not specified). This file will be formatted similarly to the original datafiles used for network training: each row contains a sequence ID (excluded if the flag ``--exclude-seq-id`` is given), an amino acid sequence, and the prediction values for that sequence. There is one output row for every sequence in `seq_file`, in the same order, so if the same sequence appears more than once (e.g. under different IDs) it will appear more than once in the output.

With ``--embedding-context``, a table of where each sequence is embedded is also written next to the predictions, as ``<output>_embedded_context.tsv`` (see "Protein language model encodings").

A network trained with ``--embed-mean-subtraction`` records this, so the sequences being predicted are mean-subtracted in the same way automatically: ``parrot-predict`` has no option for it, and their embeddings go in their own files of the embedding cache.

For networks trained on matrix data, the predictions are instead written as a NumPy ``.npz`` archive with one predicted matrix per sequence, named by its sequence ID, or with ``--exclude-seq-id`` by its number among the sequences (``1``, ``2``, ...; blank and comment lines are not counted) (``.npz`` is added to `output_file` if it does not end with it); see the "Matrix data" page.
