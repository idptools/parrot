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

In addition to one-hot encoding, encoding using biophysical scales is also built into PARROT, by providing 'biophysics' after this flag. Each amino acid is described by its value on nine scales (``BIOPHYSICAL_SCALES`` in ``parrot/encode_sequence.py``):

==========================  ============================  ===============================================================
Scale                       Units                         What it measures
==========================  ============================  ===============================================================
hydrophobicity              Kyte-Doolittle units          Kyte & Doolittle (1982) hydropathy scale (J Mol Biol 157:105)
charge                      elementary charges            formal side-chain charge: -1 for D and E, +1 for K, R and H
isoelectric point           pH units                      isoelectric point of the free amino acid
molecular weight            g/mol                         molecular weight of the free amino acid
aromatic                    1 or 0                        1 for F, W and Y
hydrogen bonding            1 or 0                        1 for side chains that can form hydrogen bonds
side-chain SASA             square angstroms              side-chain solvent-accessible surface area in ACE-X-NME
backbone SASA               square angstroms              backbone solvent-accessible surface area in ACE-X-NME
solvation free energy       (not recorded)                free energy of solvation of the side chain
==========================  ============================  ===============================================================

.. code-block:: bash

    parrot-train data/seq_regress_dataset.tsv biophysics_network.pt -d sequence -c 1 -nl 2 -hs 10 -e 200 --encode biophysics

The biophysical features are on very different scales (molecular weight is in the hundreds, while the aromatic flag is 0 or 1), so we recommend adding ``--standardize-encoding``, which rescales every feature to mean 0 and standard deviation 1 across the amino acids (see "Standardizing an encoding" on the ``parrot-train`` page).

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

``parrot-optimize`` trains a network like ``parrot-train``, but chooses the hyperparameters for you: the learning rate and the architecture's main hyperparameters (for the BRNN, the number of layers and the hidden size). It tries many configurations, drops poor ones after a few epochs (successive halving) and suggests new ones where good results have clustered (TPE); see the ``parrot-optimize`` page for details. The default 50 trials cost about as much as 15-20 ``parrot-train`` runs, so a GPU helps.

Usage is almost identical to ``parrot-train``. For example, on the residue classification example dataset:

.. code-block:: bash

    parrot-optimize data/res_class_dataset.tsv optimize_example.pt -d residues -c 3 -e 30 --trials 20 --seed 1

We do not specify the number of layers, the hidden size or the learning rate: these are searched. The most important choice is ``--epochs``, the number of epochs a configuration is trained for if it survives to the end (and so the final network's training): choose it as you would for ``parrot-train``, e.g. by looking at a training and validation loss curve. Here we use 20 trials to keep the example short; the default is 50.

The search first describes what it will try, then reports each configuration's validation loss at the end of each stage it reaches (this output is from an NVIDIA GPU; it took about 40 seconds):

.. code-block:: bash

    Searching 20 configuration(s) of the brnn architecture
      learning_rate: 1e-05 to 0.01, log scale
      hidden_size: 5 to 50
      num_layers: 1 to 5
    Epochs per stage: 3 -> 10 -> 30 (the best 1/3 of each stage continue to the next)
      trial   1 (default)  epochs    3  val loss 20.61    learning_rate=0.001  hidden_size=10  num_layers=1
      trial   2 (initial)  epochs    3  val loss 20.21    learning_rate=0.00146485  hidden_size=24  num_layers=2
      trial   3 (initial)  epochs    3  val loss 18.13    learning_rate=0.00880616  hidden_size=5  num_layers=1
      trial   3 (initial)  epochs   10  val loss 11.97    learning_rate=0.00880616  hidden_size=5  num_layers=1
      trial   4 (initial)  epochs    3  val loss 20.81    learning_rate=3.47937e-05  hidden_size=17  num_layers=5
      trial   5 (initial)  epochs    3  val loss 20.12    learning_rate=0.00297527  hidden_size=14  num_layers=1
      trial   6 (initial)  epochs    3  val loss 20.72    learning_rate=6.72572e-05  hidden_size=31  num_layers=2
      trial   5 (initial)  epochs   10  val loss 13.43    learning_rate=0.00297527  hidden_size=14  num_layers=1
      trial   7 (initial)  epochs    3  val loss 20.77    learning_rate=1.32461e-05  hidden_size=47  num_layers=3
      trial   8 (initial)  epochs    3  val loss 20.86    learning_rate=0.000146358  hidden_size=36  num_layers=5
      trial   9 (initial)  epochs    3  val loss 20.72    learning_rate=0.000471578  hidden_size=42  num_layers=4
      trial   2 (initial)  epochs   10  val loss 11.89    learning_rate=0.00146485  hidden_size=24  num_layers=2
      trial   2 (initial)  epochs   30  val loss 10.05    learning_rate=0.00146485  hidden_size=24  num_layers=2
      trial  10 (model  )  epochs    3  val loss 19.39    learning_rate=0.00839561  hidden_size=5  num_layers=1
      ...
      trial  19 (model  )  epochs    3  val loss 17.01    learning_rate=0.00817644  hidden_size=6  num_layers=1
      trial  19 (model  )  epochs   10  val loss 10.54    learning_rate=0.00817644  hidden_size=6  num_layers=1
      trial  19 (model  )  epochs   30  val loss 10.22    learning_rate=0.00817644  hidden_size=6  num_layers=1
      trial  20 (model  )  epochs    3  val loss 17.06    learning_rate=0.00843449  hidden_size=8  num_layers=1
      trial  20 (model  )  epochs   10  val loss 10.4     learning_rate=0.00843449  hidden_size=8  num_layers=1
      trial  20 (model  )  epochs   30  val loss 9.917    learning_rate=0.00843449  hidden_size=8  num_layers=1
    Best configuration (trial 14, validation loss 9.902):
      learning_rate=0.00864999  hidden_size=5  num_layers=1  loglength_feedforward=True
    Test Loss: 10.8989

With ``--epochs 30`` there are three stages: every configuration is trained for 3 epochs, the best third of them continue to 10 epochs, and the best third of those to 30. Trial 1 is PARROT's default hyperparameters, trials 2-9 are spread evenly over the search space, and from trial 10 the search mostly suggests configurations near the best so far ("model"), with some drawn at random. Configurations that are clearly worse, such as those with very small learning rates, are stopped after 3 epochs. The best configuration has already been trained for the full 30 epochs, so it becomes the final network, which is then tested on the held-out test set exactly as in ``parrot-train``.

As well as the usual ``parrot-train`` output, ``optimize_example_optimal_hyperparams.txt`` lists the chosen hyperparameters, and ``optimize_example_hyperparameter_trials.tsv`` records every trial:

.. code-block:: bash

    trial  origin  status  epochs  val_loss_3_epochs  val_loss_10_epochs  val_loss_30_epochs  learning_rate  hidden_size  num_layers
    1  default  stopped  3  20.6142      0.001  10  1
    2  initial  completed  30  20.2055  11.887  10.0479  0.00146485  24  2
    3  initial  stopped  10  18.1306  11.9684    0.00880616  5  1
    4  initial  stopped  3  20.8106      3.47937e-05  17  5
    ...

Hyperparameters you give on the command line are fixed rather than searched, e.g. ``--architecture cnn --channels 32`` searches everything except the number of channels. For small datasets, ``--cv-folds 5`` scores each configuration by 5-fold cross-validation instead of on the validation set.

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
    