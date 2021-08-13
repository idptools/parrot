==============
 parrot-train
==============

``parrot-train`` is the primary command for training a PARROT network. In the most basic usage, the user specifies their data and an output location and a trained bi-directional LSTM network will be created along with an estimate of network predictive performance on randomly chosen test samples. There are many optional arguments to ``parrot-train`` that allow users to specify network parameters, create helpful output images, and otherwise modify PARROT to meet their needs.

Once PARROT is installed, the user can run ``parrot-train`` from the command line:

.. code-block:: bash
	
	$ parrot-train data_file output_network <flags>

Where *data_file* specifies the path to the whitespace-separated datafile and *output_network* is the path to where the final trained network will be saved. It is recommended, but not required, that networks are saved using a ".pt" file extension, following the PyTorch convention.

**Required flags:**

	*  ``--datatype`` / ``-d`` : Describes how values are formatted in `datafile`. Should be 'sequence' if there is a single value per sequence, or 'residues' if there are values for every residue in each sequence. See the example datasets in the **/data** folder for more information.
	*  ``--classes`` / ``-c`` : The number of classes for the machine learning task. If the task is regression, then specify '1' (without the quote marks).

**Optional flags:**

	*  ``--help`` / ``-h`` : Display a help message.
	*  ``--learning-rate`` / ``-lr`` : Learning rate of the network (default is 0.001). Must be a float between 0 and 1.
	*  ``--num-layers`` / ``-nl`` : Number of hidden layers in the network (default is 1). Must be a positive integer.
	*  ``--hidden-size`` / ``-hs`` : Size of hidden vectors within the network (default is 10). Must be a positive integer.
	*  ``--batch`` / ``-b`` : Training minibatch size (default is 32). Must be a positive integer, and for most datasets should be in the range of 8-256. Powers of 2 (64, 128, 256, etc.) are optimized for slightly faster performance, but are not explicitly required.
	*  ``--epochs`` / ``-e`` : Number of training epochs (default is 100). Has different behavior depending on what is specified by the ``--stop`` flag.
	*  ``--stop`` : Stop condition to terminate training. Must be either 'auto' or 'iter' (default is 'iter'). If 'iter', then train for exactly ``-e`` epochs and stop. If 'auto', then train until performance has plateaued for ``-e`` epochs. If using 'auto', be careful not to indicate a large number of epochs, as this can take much longer than is necessary.
	*  ``--split`` : Path to a "split-file" for manually dividing dataset into training, validation and test sets. The file should contain three lines, corresponding to training, validation and test sets respectively. Each line should have integers separated by whitespace, with the integers specify which sequences/lines in the `datafile` (0-indexed) will belong to which dataset. See **/data** folder for examples. If a split-file is not provided, default behavior is for PARROT to randomly divide data into training, validation and test sets.
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

After running ``parrot-train``, at least three files will be saved to the directory specified by *output_network*. One contains the saved network weights from the training process which can be used with ``parrot-predict``. The other file with the suffix "_predictions.tsv" contains the true values and predicted values for all of the sequences in the test set. The final "_performance_stats.txt" file quantifies a variety of performance metrics on the test set. Output text detailing network performance across training is also printed to terminal by default.

If ``--include-figs`` is provided, there will be additional PNG images saved in this directory. The first, with suffix '_train_val_loss.png' displays the network's performance on the training and validation sets over the course of training. The other image(s) describes the network performance on the held out test set, and will vary depending on the data format and machine learning task. If training a network for a classification task, the image will be a confusion matrix. If training for a regression task, the image will be a scatterplot comparing the predicted and true values of the test set sequences. If using probabilistic-classification mode, then there will be two output figures: one plotting receiver operator characteristic (ROC) curves and the other plotting precision-recall curves.
