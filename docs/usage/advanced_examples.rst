====================
 Advanced Examples:
====================

The core usage of PARROT is designed to be as simple as possible so that anyone, regardless of computational expertise, can train a network on their dataset with minimal investment. However, on top of this basic implementation, PARROT has a number of options intended to let experienced users tailor their networks to their particular needs and to facilitate more sophisticated computational workflows.

Advanced ``parrot-train`` options:
----------------------------------

**Automatic determination of number of training epochs with --stop:**

This flag determines the stop condition for network training. Currently, there are two options implemented: either 'iter' or 'auto'. In all of the previous examples we used the default behavior, 'iter', which means that the number we specify for the ``-e`` flag will be the number of iterations that we train the network. Alternatively, using 'auto' means that training will stop automatically once performance on the validation set has plateaued for ``-e`` epochs. Thus, with 'auto' it is recommended to use a smaller number of epochs (10-20) for ``-e`` so training does not extend for a significantly long period of time.

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv stop_example.pt --datatype sequence -c 1 -nl 2 -hs 5 -lr 0.001 -e 10 -b 32 -v --stop auto

.. code-block:: bash

    PARROT with user-specified parameters
    -------------------------------------
    Train on:   cpu
    Datatype:   sequence
    ML Task:    regression
    Learning rate:  0.001000
    Number of layers:   2
    Hidden vector size: 5
    Batch size: 32

    Validation set loss per epoch:

    Epoch 0 Loss 0.1779
    Epoch 1 Loss 0.1752
    Epoch 2 Loss 0.1727
    ...
    Epoch 98    Loss 0.0456
    Epoch 99    Loss 0.0456
    Epoch 100   Loss 0.0456
    Epoch 101   Loss 0.0456
    Epoch 102   Loss 0.0456
    Epoch 103   Loss 0.0456
    Epoch 104   Loss 0.0456
    Epoch 105   Loss 0.0456
    Epoch 106   Loss 0.0456
    Epoch 107   Loss 0.0456
    Epoch 108   Loss 0.0456
    Epoch 109   Loss 0.0456
    Epoch 110   Loss 0.0455
    Epoch 111   Loss 0.0455
    Epoch 112   Loss 0.0455

Training stops here because performance has stopped improving. Worth mentioning: in some cases such as this dataset, 'auto' can actually get stuck in a local minimum well before the network is fully trained. Be mindful of this when using 'auto' stop condition.

You might also notice that in this example, the validation loss is listed for every single epoch instead of every 5. This is simply because the verbose ``-v`` flag was provided.

**Splitting data into train/validation/test sets:**

``--set-fractions``:
This flag allows the user to set the proportions of data that will be a part of the training set, validation set, and test set. By default, the split is 70:15:15. This flag takes three input arguments, between 0 and 1, that must sum to 1.

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv setfractions_network.pt --datatype sequence -c 1 -e 200 --set-fractions 0.5 0.45 0.05

Notice that the output predictions file from this command has fewer datapoints because of the reduced test set. Most likely, the accuracy will be a little worse then the default proportions because the training set is also smaller.

``--split``:
In some cases, users might want precise control over over the training, validation and test set splits of their input data. This flag allows the user to manually specify which subset each sample in their dataset will be assigned. This flag requires an argument that is a path to a *split_file*, which specifically allocates sequences in `datafile` to the different datasets. An example *split_file* is provided in the **/data** folder for reference.

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv manualsplit_network.pt --datatype sequence -c 1 -e 200 --split data/split_file.tsv 

This can especially be useful if you wish to perform k-fold cross-validation on your dataset, as you can prepare k different split_files that each specify a particular 1/kth of your dataset into the test set.

``--save-splits``:
Sometimes, a random partition into training/val/test sets is acceptable, but it is helpful to know for replicability where each sample was assigned. For example, if you are comparing multiple types of machine learning networks, it is best practice to use the same training set for each network. Including this flag causes an additional text file (suffix: "_split_file.txt") to be saved to the output directory. This file is formatted in the same way as a *split_file* for using with the ``--split`` flag.

**Amino acid -> vector encoding:**

"Encoding" in the context of PARROT refers to the process of converting a sequence of amino acids into computer-readable numeric vectors. By default, PARROT utilizes *one-hot* encoding, which represents each amino acid as a vector with 19 zeros and a single 1, where the position of the 1 determines its identity. However, users can change how amino acids are encoded using the ``--encode`` flag. 

In addition to one-hot encoding, encoding using biophysical scales (vector of properties like charge, hydrophobicity, molecular weight, etc.) is also hard-coded into PARROT. Machine learning using biophysical encoding and can be carried out by providing 'biophysics' after this flag.

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv biophysics_network.pt -d sequence -c 1 -nl 2 -hs 10 -e 200 --encode biophysics

More powerfully, PARROT also allows the user to manually specify their own encoding scheme, if they desire. An example encoding file can be found in the **/data** folder. In this case, provide the path to this encoding file following the flag.

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv userencode_network.pt -d sequence -c 1 -nl 2 -hs 10 -e 200 --encode data/encoding_example.txt

With the ``--encode`` flag and a user-provided file, PARROT is even flexible enough to work on nucleotide sequences! To illustrate this, we've included the file "nucleotide_encoding.txt" which can be passed in via this flag to one-hot encode nucleotide sequences. We've also included an example sequence regression dataset (melting temperature prediction) with nucleotide sequences: "nucleotide_dataset.tsv".

.. code-block:: bash

    parrot-train data/nucleotide_dataset.txt nucleotide_network.pt -d sequence -c 1 -nl 2 -hs 10 -e 200 --encode data/nucleotide_encoding.txt

**Probabilistic classification with --probabilistic-classification:**

The standard behavior of "classification" tasks in PARROT is to make predictions of discrete class labels. In reality though, this sort of behavior does not provide any information on the certainty behind these prediction. For example, in a two class problem (classes 0 and 1), if sequence A is deemed to be class 0 with 98% confidence, and sequence B is deemed class 0 with 51% confidence, both of these sequences will appear in the output prediction file as class 0. In some instances, it is useful to provide users a measure of confidence for each of the class predictions that PARROT makes. This can be accomplished with the ``--probabilistic-classification`` flag.

Using this flag is easy and can be used with ``parrot-train``, ``parrot-optimize`` and ``parrot-predict``. For the first two commands, this flag changes how predictions on the test set are output in the "_predictions.tsv" file and changes the figures and performance stats that are output (if specified). For the predict command, it changes how the predictions are outputed. If this flag is combined with ``--include-figs``, it also changes the figure and metrics that are produced for evaluating performance on the test set (see ``parrot-train`` documentation page for more details). Conveniently, this flag can be used in ``parrot-predict`` even if it was not specified during training. As an example, here is the same sequence 3-class classification network making predictions with and without the ``--probabilistic-classification`` flag (default layers and hidden vector size):

.. code-block:: bash

    parrot-predict data/seqfile.txt prob_example.pt discrete.txt -d sequence -c 3

Output:

.. code-block:: bash

    a1 EADDGLYWQQN 2
    b2 RRLKHEEDSTSTSTSTSTQ 0
    c3 YYYGGAFAFAGRM 2
    d4 GGIL 2
    e5 GREPCCMLLYILILAAAQRDESSSSST 2
    f6 PGDEADLGHRSLVWADD 2

.. code-block:: bash

    parrot-predict data/seqfile.txt prob_example.pt probabilistic.txt -d sequence -c 3 --probabilistic-classification

Output:

.. code-block:: bash

    a1 EADDGLYWQQN 0.0527 0.1081 0.8392
    b2 RRLKHEEDSTSTSTSTSTQ 0.9819 0.0034 0.0148
    c3 YYYGGAFAFAGRM 0.0742 0.0098 0.916
    d4 GGIL 0.1509 0.0596 0.7894
    e5 GREPCCMLLYILILAAAQRDESSSSST 0.0465 0.0118 0.9418
    f6 PGDEADLGHRSLVWADD 0.0645 0.2576 0.678

The three numbers following each sequence represent the probability that the sequence belongs to each of the three classes. Notice the numbers in each row sum to 1.

Currently, probabilistic classification is only implemented for *sequence classification* problems. The same principles would work for *residue classification*, however, we have not thought of a convenient way of representing the information in the output files (each sequence has num_classes x seq_len values).

Hyperparameter tuning with ``parrot-optimize``:
-----------------------------------------------

``parrot-optimize`` will train a network like ``parrot-train``, however this command does not require the user to specify hyperparameters. Instead, it relies upon Bayesian Optimization to automatically select hyperparameters. Although Bayesian Optimization is much more efficient than grid search optimization, it still requires many iterations to converge upon the best hyperparameters. Additionally, this command relies upon 5-fold cross validation for each set of hyperparameters to achieve an accurate estimate of network performance. All together, this means that ``parrot-optimize`` can take over 100x longer to run than ``parrot-train``. It is strongly recommended to only run this command on a machine with a GPU.

Nonetheless, usage for ``parrot-optimize`` is remarkably similar to ``parrot-train``, since many of the flags are identical. As an example, let's run the command on a residue regression dataset:

.. code-block:: bash

    parrot-optimize data/res_class_dataset.tsv optimize_example.pt -d residues -c 3 -e 200 --max-iter 20 -b 32 --verbose

Notice how we do not need to specify number of layers, hidden vector size, or learning rate as these are the parameters we are optimizing. Perhaps the most important consideration is the number of epochs. Running the optimization procedure with a large number of epochs is more likely to identify the best performing hyperparameters, however more epochs also means significantly longer run time. **IMPORTANT: I only used 20 iterations and 150 epochs here to speed up the example but it is HIGHLY recommended to use at least the default iterations for normal usage.** It is recommended to play around with your data using ``parrot-train`` with a few different parameters and visualizing the training and validation loss per epoch in order to pick the optimal number of epochs for training. Ideally, you should set the number of epochs to be around the point where validation accuracy tends to plateau during training.

Let's break down what is output to console during the optimization procedure:

.. code-block:: bash

    PARROT with hyperparameter optimization
    ---------------------------------------
    Train on:   cuda
    Datatype:   residues
    ML Task:    classification
    Batch size: 32
    Number of epochs:   200
    Number of optimization iterations:  20


    Initial search results:
    lr  nl  hs  output
    0.00100  1  20  11.6680
    0.00100  2  20  11.2927
    0.00100  3  20  11.0651
    0.00100  4  20  10.9217
    0.00100  5  20  11.2689
    0.01000  2  20  10.7816
    0.00050  2  20  11.6328
    0.00010  2  20  13.6755
    0.00001  2  20  32.7119
    0.00100  2   5  11.2988
    0.00100  2  15  11.1669
    0.00100  2  35  11.2267
    0.00100  2  50  11.0833
    Noise estimate: 0.7594081234203327


The first chunk of text details the network performance (average of 5 data folds) during the initial stage of hyperparameter optimization. This stage is used to gather an estimate of the noise (standard deviation across cross-val folds) for future optimization. The hyperparameters used in the initial search stage are hard-coded into the optimization procedure.

.. code-block:: bash

    Primary optimization:
    --------------------

    Learning rate   |   n_layers   |   hidden vector size |  avg CV loss  
    ======================================================================
      0.010000  |      3       |         20           |    10.593
      0.005001  |      3       |         19           |    10.820
      0.010000  |      4       |         21           |    10.715
      0.005513  |      3       |         21           |    10.852
      0.000744  |      4       |         21           |    11.113
      0.004678  |      5       |         22           |    10.847
      0.008415  |      4       |         22           |    10.550
      0.000954  |      4       |         23           |    11.024
      0.010000  |      3       |         23           |    10.597
      0.010000  |      4       |         24           |    10.559
      0.002181  |      3       |         24           |    10.757
      0.000709  |      4       |         25           |    11.065
      0.001744  |      5       |         24           |    11.281
      0.010000  |      3       |         25           |    10.707
      0.010000  |      2       |         22           |    10.869
      0.010000  |      2       |         24           |    10.758
      0.000822  |      2       |         25           |    11.275
      0.000859  |      2       |         23           |    11.100
      0.010000  |      5       |         26           |    10.817
      0.010000  |      4       |         30           |    10.774

    The optimal hyperparameters are:
    lr = 0.00841
    nl = 4
    hs = 22


This long block of text is the main process of optimization. The algorithm automatically selects the learning rate, number of layers and hidden vector size for each iteration. Finally, after the algorithm runs for 20 iterations (default: 50 iterations), the optimal hyperparameters are determined. These hyperparameters are also saved to a text file called 'optimal_hyperparams.txt' in the output directory. You might notice that the optimization procedure doesn't appear to sample the entire hyperparameter space, but this is due to the fact that we specified to use fewer iterations than normally recommended.

.. code-block:: bash

    Training with optimal hyperparams:
    Epoch 0 Loss 31.7953
    Epoch 1 Loss 30.4627
    Epoch 2 Loss 22.8318
    Epoch 3 Loss 26.4293
    Epoch 4 Loss 17.9814
    Epoch 5 Loss 15.7970
    Epoch 6 Loss 15.0506
    Epoch 7 Loss 13.6761
    Epoch 8 Loss 13.8338
    Epoch 9 Loss 14.3309
    Epoch 10    Loss 13.1378
    ...
    Epoch 396   Loss 40.0893
    Epoch 397   Loss 40.9645
    Epoch 398   Loss 41.5348
    Epoch 399   Loss 41.8932

    Test Loss: 11.1555


Lastly, a network is trained on all the training data using the optimal hyperparameters and tested on the held-out test set. The output produced is analogous to ``parrot-train``.


Integrating trained PARROT networks into Python workflows:
----------------------------------------------------------

We added the option for users to create a predictor object in Python using their trained PARROT network. This option is built-in to the file "py_predictor.py" that is installed with PARROT. Importing PARROT within Python is simple:

.. code-block:: python

    >>> from parrot import py_predictor as ppp

To use a saved network, you need to create a Predictor() object. Initializing this object only requires the path to the saved network weights and specification of whether this network is for sequence or residue prediction.

.. code-block:: python

    >>> my_predictor = ppp.Predictor('/path/to/network.pt', dtype='sequence')

Now we're ready to make predictions! Once a network is loaded, the time to make predictions is negligible, so your predictor can be applied to as many sequences as you want. Just feed in amino acid sequences to the predict() function one at a time and predicted values will be output.

.. code-block:: python

    >>> value = my_predictor.predict('MYTESTAMINACIDSEQ')

Currently, this Python usage is only implemented for networks that were created using standard, one-hot amino acid encoding. In the future, we may add the option to feed in a particular encoding file so that all trained networks can be used in this manner. If this is a feature you'd be interested in, let us know and we can prioritize adding it!
    