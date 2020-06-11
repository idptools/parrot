Examples
========

Below are a handful of examples outlining how prot_brnn can be used to train a BRNN on various machine learning tasks. Provided in the **prot_brnn** folder is a the **data** folder which has 4 different example datasets corresponding to the possible combinations of data format (sequence-mapped values and residue-mapped values) and machine learning task (classification and regression). Details on these datasets can be found in the README within the **data** folder. This folder also contains an example list of sequences for ``brnn_predict``.

brnn_train
----------

**Sequence classification:**
In our first example, each of the 300 sequences in *seq_class_dataset.tsv* belongs to one of three classes:

.. code-block::
	
	Frag0 WKHNPKHLRP 0
	Frag1 DLFQDEDDAEEEDFMDDIWDPDS 1
	Frag2 YHFAFTHMPALISQTSKYHYYSASMRG 2
	Frag3 CNRNRNHKLKKFKHKKMGVPRKKRKHWK 0
	.
	.
	.

Let's train a network with ``brnn_train`` that can identify the inherent pattern (described in data/README.md). For starters, let's try to train for 200 epochs on a network with 2 hidden layers, a hidden vector size of 5, a learning rate of 0.001 and a batch size of 32. Note that the paths for the dataset and output network may vary on different machines. Let's also use the ``-v`` flag to get a sense of training.

.. code-block::

	brnn_train datasets/seq_class_dataset.tsv output_dir/network.pt --datatype sequence -nc 3 -nl 2 -hs 5 -lr 0.001 -e 200 -b 32 -v

Training has a stochastic component, so running this multiple times may yield slightly different results. The output should look something like:

.. code-block::

	Epoch 0	Loss 0.0491
	Epoch 5	Loss 0.0486
	Epoch 10	Loss 0.0482
	.
	.
	.
	Epoch 190	Loss 0.0063
	Epoch 195	Loss 0.0063

	Test Loss: 0.1932
	
In **output_dir**, there should also be two PNG files describing the training process and network performance.

.. image:: ../images/train_test.png
  :width: 400

.. image:: ../images/seq_CM.png
  :width: 400

This gives us a general sense of how the network will perform on new data in the future. Overall the network performs well, and the only misclassifications are for sequences in class '2'.

**Sequence regression:**

Using prot_brnn on a machine learning regression task is very similar to classification. In *seq_regress_dataset.tsv*, instead of each sequence being assigned an integer class label, each sequence is represented by a real number.

.. code-block::

	Frag0 EHCWTYIFQMYRIDQTQRVKRGEKPIIYLEPMAR 3.8235294117647056
	Frag1 SDAWVMKFLWDKCGDHFIQYQKPANRWEWVD 3.870967741935484
	Frag2 IYPEQSPDNAWAW 3.076923076923077
	.
	.
	.

The only difference in the ``brnn_train`` command in this regression case (other than the datafile path) is the ``-nc`` argument. Since we are doing regression, we will put '1' here. We could also change the network hyperparameters, but for now let's just use the same as above. Notice that we are using a different output network name so as to not overwrite the previous network.

.. code-block::

	brnn_train datasets/seq_regress_dataset.tsv output_dir/network2.pt --datatype sequence -nc 1 -nl 2 -hs 5 -lr 0.001 -e 200 -b 32 -v

After this command, we see a similar output as before. In this case, in addition to ``train_test.png`` (this overwrites the previous image--if you are using the same output directory for lots of training, it may be wise to rename these files after each run) you will see a scatter plot detailing the predictions on the test set data.

.. image:: ../images/seq_scatter.png
  :width: 400

Not bad!

**Residue classification:**

Now let's try a task where the objective is to classify each residue in a sequence. Unlike before, in *res_class_dataset.tsv* there are multiple values per sequence in the datafile.

.. code-block::

	Frag0 DEDGTEDDMATTK 1 1 1 1 1 1 1 1 1 1 1 1 1
	Frag1 CGSAPSRFVKTCDPDEEDEDDEDE 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
	Frag2 EWYEDDKPFPCPERVPHHKKGHRGGWRAKKNWKV 1 1 1 1 1 1 1 0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	.
	.
	.

Despite this major difference, the ``brnn_train`` command is similar to the above examples. The only difference will be the value we input after the ``--datatype`` flag. Before we put 'sequence', and here we will put 'residues'. Just for fun, we will also switch up our number of layers (``-nl``) and hidden size (``-hs``) hyperparameters.

.. code-block::

	brnn_train datasets/res_class_dataset.tsv output_dir/network3.pt --datatype residues -nc 3 -nl 3 -hs 8 -lr 0.001 -e 200 -b 32 -v

This will save a confusion matrix 'res_CM.png' to the output directory. It's nearly identical to the confusion matrix for sequence classification, although in this case it is for every single residue in all of the sequences in the test set.

.. image:: ../images/res_CM.png
  :width: 400

**Residue regression:**

The final kind of machine learning task that prot_brnn can handle is regression on every residue in a sequence. For this command ``--datatype`` should be set to 'residues' and ``-nc`` should be '1'. In this example I also changed the learning rate hyperparameter ``-lr``.

.. code-block::

	brnn_train prot-brnn/prot_brnn/data/res_regress_dataset.tsv saved_networks/example.pt --datatype residues -nc 1 -nl 3 -hs 8 -lr 0.005 -e 200 -b 32 -v

As in the other regression task, a residue regression task will produce a scatter plot that shows the network's performance on the test set. Each combination of marker shape and color in this scatterplot belong to a single sequence, which may provide some insight on whether the network systematically mis-predicts all sequences, or if there are only a few specific sequences that are outliers.

.. image:: ../images/res_scatter.png
  :width: 400

**Other flags:**

TODO:

-Auto stop condition
-Set fractions
-Manual split file
-Exclude sequence IDs
-Encode biophysics

brnn_optimize
-------------

``brnn_optimize`` will train a network like ``brnn_train``, however this command does not require the user to specify hyperparameters. Instead, it relies upon Bayesian Optimization to automatically select hyperparameters. Although Bayesian Optimization is much more efficient than grid search optimization, it still requires many iterations to converge upon the best hyperparameters. Additionally, this command relies upon 5-fold cross validation for each set of hyperparameters to achieve an accurate estimate of network performance. All together, this means that ``brnn_optimize`` can take over 400x longer to run than ``brnn_train``. It is strongly recommended to only run this command on a machine with a GPU.

Nonetheless, usage for ``brnn_optimize`` is remarkably similar to ``brnn_train``, since many of the flags are identical. As an example, let's run the command on a residue regression dataset:

.. code-block::

	brnn_optimize prot-brnn/prot_brnn/data/res_regress_dataset.tsv saved_networks/cv_example.pt --datatype residues -nc 1 -e 200 -b 32 -vv

Notice how we do not need to specify number of layers, hidden vector size, or learning rate as these are the parameters we are optimizing. Perhaps the most important consideration is the number of epochs. Running the optimization procedure with a large number of epochs is more likely to identify the best performing hyperparameters, however more epochs also means significantly longer run time. It is recommended to play around with your data using ``brnn_train`` with a few different parameters and visualizing 'train_test.png'. Ideally, you should set the number of epochs to be around the point where validation accuracy tends to plateau during training.

Let's break down what is output to console during the optimization procedure:

.. code-block::

	[1/5] Loss: 75.247434
	[2/5] Loss: 75.689319
	[3/5] Loss: 66.811298
	[4/5] Loss: 72.030063
	.
	.
	.
	[3/5] Loss: 1.476518
	[4/5] Loss: 1.395311
	[5/5] Loss: 1.380726

	Initial search results:
	lr	nl	hs	output
	0.00001	 5	10	73.2288
	0.00100	 5	 5	8.7716
	1.00000	 8	20	66.9336
	0.00100	15	 5	52.8299
	0.00100	 3	30	1.4568
	Noise estimate: 3.285178370588926

The first chunk of text details the network performance (for all 5 data folds) during the initial stage of hyperparameter optimization. This stage is used to gather an estimate of the noise (standard deviation across cross-val folds) for future optimization.

.. code-block::

	Primary optimization:
	--------------------

	Learning rate   |   n_layers   |   hidden vector size
	=====================================================
	  0.000630	|      3       |         30
	[1/5] Loss: 1.881410
	[2/5] Loss: 2.010539
	[3/5] Loss: 1.651101
	[4/5] Loss: 1.631336
	[5/5] Loss: 3.060484
	.
	.
	.

	The optimal hyperparameters are:
	lr = 0.004901
	nl = 1
	hs = 29

This long block of text is the main process of optimization. The algorithm automatically selects the learning rate, number of layers and hidden vector size for each iteration. Finally, after the algorithm converges (max 75 iterations), the optimal hyperparameters are determined. These hyperparameters are also saved to a text file called 'optimal_hyperparams.txt' in the output directory.

.. code-block::

	Training with optimal hyperparams:
	Epoch 0	Loss 56.9641
	.
	.
	.

	Test Loss: 0.7732

Lastly, a network is trained on all the data using the optimal hyperparameters. Like in ``brnn_train`` two PNGs are saved to the output directory describing training and performance on the held-out test set.

brnn_predict
------------

Use the trained network from optimize and predict on an example list of sequences (put in /data).

Show input, command (hyperparams MUST be identical), output.


.. toctree::
   :maxdepth: 2
   :caption: Contents: