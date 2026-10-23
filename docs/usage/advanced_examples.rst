====================
 Advanced Examples:
====================

The core usage of PARROT is designed to be as simple as possible so that anyone, regardless of computational expertise, can train a network on their dataset with minimal investment. However, on top of this basic implementation, PARROT has a number of options intended to let experienced users tailor their networks to their particular needs and to facilitate more sophisticated computational workflows.

Advanced ``parrot-train`` options:
----------------------------------

**Automatic determination of number of training epochs with --stop:**

This flag determines the stop condition for network training. Currently, there are two options implemented: either 'iter' or 'auto'. In all of the previous examples we used the default behavior, 'iter', which means that the number we specify for the ``-e`` flag will be the number of iterations that we train the network. Alternatively, using 'auto' means that training will stop automatically once ``-e`` epochs in a row have not lowered the validation loss by at least 0.5% (relative to the last epoch that did). Thus, with 'auto' it is recommended to use a smaller number of epochs (10-20) for ``-e`` so training does not extend for a significantly long period of time.

.. code-block:: bash

    parrot-train brnn --input-file data/seq_regress_dataset.tsv --output-network stop_example.pt --split random --datatype sequence --regression -nl 2 -hs 20 -lr 0.001 -e 20 -b 8 -v --stop auto

.. code-block:: bash

     ____      _     ____   ____    ___   _____         ____
    |  _ \    / \   |  _ \ |  _ \  / _ \ |_   _|       |___ \
    | |_) |  / _ \  | |_) || |_) || | | |  | |   _____   __) |
    |  __/  / ___ \ |  _ < |  _ < | |_| |  | |  |_____| / __/
    |_|    /_/   \_\|_| \_\|_| \_\ \___/   |_|         |_____|

    ===========================================================================
     parrot-train  |  PARROT <version>  |  <date> <time>
     Documentation: https://idptools-parrot.readthedocs.io
     For examples of how to use parrot-train, run: parrot-train --show-examples
    ===========================================================================

    Running on: CPU
    Splitting 300 sequences (random, as parrot-preprocess --split random does):
      Assigned clusters to sets: training 210 (70.0%), validation 45 (15.0%), test 45 (15.0%)

    Training brnn: bidirectional LSTM (the original PARROT network)
    ---------------------------------------------------------------
    Data: 210 training, 45 validation and 45 test sequences (regression, one value per sequence)

    Settings:
    ...................................
    device                : cpu
    datatype              : sequence
    task                  : regression
    classes               : 1
    encoding              : onehot
    architecture          : brnn
    hidden_size           : 20
    num_layers            : 2
    dropout               : 0.0
    attention_pooling     : True
    loglength_feedforward : True
    learning_rate         : 0.001
    epochs                : 20
    stop                  : auto
    batch_size            : 8
    batching              : length

    Validation set loss per epoch:
    Epoch 1 Loss 3.0148
    Epoch 2 Loss 1.0526
    Epoch 3 Loss 1.0068
    Epoch 4 Loss 1.0046
    Epoch 5 Loss 1.0020
    Epoch 6 Loss 0.9982
    Epoch 7 Loss 0.9961
    Epoch 8 Loss 0.9963
    Epoch 9 Loss 0.9908
    Epoch 10    Loss 0.9883
    Epoch 11    Loss 0.9725
    Epoch 12    Loss 0.9580
    Epoch 13    Loss 0.9074
    Epoch 14    Loss 0.8447
    Epoch 15    Loss 0.7710
    ...
    Epoch 171   Loss 0.0668
    Epoch 172   Loss 0.0594
    Epoch 173   Loss 0.0769
    Epoch 174   Loss 0.0769
    Epoch 175   Loss 0.0715
    Stopped after 175 epochs, as the validation loss had stopped improving
    Lowest validation loss 0.0542, at epoch 155 of 175: the network from that epoch is kept

    Test set: loss 0.0545, Pearson R 0.998, Spearman R 0.997
    Network saved to /home/user/parrot_examples/stop_example.pt
    Also written there: stop_example_info.txt, stop_example_predictions.tsv, stop_example_performance_stats.txt

Training stops here because the validation loss has not improved for 20 epochs in a row since epoch 155 (the lowest), as the output says, and the network from the epoch with the lowest validation loss is kept.

Worth mentioning: 'auto' can stop in a local minimum well before the network is fully trained. Notice how, in this run, the loss barely changed between epochs 3 and 10, while the network was predicting values close to the average of the dataset, before it started to fall again. With a smaller ``-e`` (or a network that leaves such a plateau more slowly), 'auto' often stops on the plateau: on this dataset, the same command with ``-hs 5 -b 32 -e 10`` usually stops after 35-50 epochs, with a test-set Pearson R of about 0.2 or less. Be mindful of this when using 'auto' stop condition, and check the loss curve (``--include-figs``) to see whether training stopped too early.

You might also notice that in this example, the validation loss is listed for every single epoch instead of every 5, and every setting is listed before training starts. This is simply because the verbose ``-v`` flag was provided.

**Learning-rate schedules and numerical safeguards:**

The default learning rate stays fixed. To anneal it over a known number of
epochs, choose cosine decay, optionally with a few warmup epochs:

.. code-block:: bash

    parrot-train brnn --input-file data/res_regress_dataset.tsv --output-network cosine_example.pt --split data/splitfile.txt -d residues --regression -lr 0.001 -e 50 --lr-schedule cosine --lr-warmup-epochs 5 --lr-min 0.00001 --seed 1

Here warmup uses five epochs, then cosine decay has a 45-epoch duration. Rates
are set at the start of each epoch: the first decay epoch uses 0.001, and the
floor is reached after 45 completed decay epochs. With a longer run, the floor
is held rather than starting another cycle. To choose the duration explicitly,
give ``--lr-anneal-epochs``; this is required with ``--stop auto``.

To adapt the rate when validation improvement stalls, use ``plateau``. It can
be combined with automatic stopping and a bounded recovery budget:

.. code-block:: bash

    parrot-train brnn --input-file data/res_regress_dataset.tsv --output-network plateau_example.pt --split data/splitfile.txt -d residues --regression -lr 0.001 --lr-schedule plateau --lr-patience 3 --lr-factor 0.5 --lr-min 0.000001 --stop auto -e 10 --recover-nonfinite 2 --check-finite-gradients --include-figs --seed 1

After more than three epochs without sufficient validation improvement, the
plateau schedule halves the rate. Its ``--lr-threshold`` is an absolute loss
change (default 0.0001); tune it for the scale of the chosen loss. Automatic
stopping has its own improvement criterion and patience, here ten epochs, which
does not reset on a rate change. Recovery permits at most two rollbacks across
the whole run. Each restores the start of the failed epoch and replays it at a
lower rate; invalid inputs and exhausted budgets stop the run. Recovery does
not resume a process that has crashed.

To choose a clipping threshold, first inspect a short run at a stable rate:

.. code-block:: bash

    parrot-train brnn --input-file data/res_regress_dataset.tsv --output-network gradient_pilot.pt --split data/splitfile.txt -d residues --regression -lr 0.001 -e 5 --check-finite-gradients --verbose --seed 1

This reports the median, 95th percentile and maximum gradient norm before any
clipping. In subsequent pilots, set ``--gradient-clip-norm`` to a positive
finite threshold chosen from those observations, then compare clipping frequency
and validation performance. Recalibrate it if the model, targets, batch size or
loss weighting changes. Clipping implies finite checks; it cannot repair
gradients already containing NaN or infinity.

Schedules, warmup and recovery produce ``<network>_training_history.tsv`` on
successful completion, with losses, the rate used and the cumulative recovery
count for each successful epoch. Any recoveries are also recorded in
``<network>_recovery_events.json``. The saved network is still from the epoch
with the best validation loss. See :doc:`training_controls` for all defaults,
schedule details and recovery limits.

**Choosing residue or protein weighting:**

The default residue loss sums labelled residues, so a protein with more labels
contributes more loss terms. This fits an objective of predicting individual
residues well. To give each protein's mean labelled error equal weight instead:

.. code-block:: bash

    parrot-train brnn --input-file data/res_regress_dataset.tsv --output-network protein_weighted_example.pt --split data/splitfile.txt -d residues --regression --equal-protein-weighting -lr 0.001 -e 50 --seed 1

This changes training, validation and test loss, including which epoch is kept.
It averages labelled losses within each protein before averaging proteins;
missing labels and padding do not count. Equal objective weight does not force
equal gradients. Standard metrics still pool labelled residues. The option
also supports matrix tasks, but not sequence tasks; see
:ref:`parrot-loss-weighting`. Retune the rate and clipping threshold for the new
loss scale before comparing runs.

**Choosing a loss, regularization and resumable records:**

For a robust regression objective, choose Huber loss and its transition in the
units of the targets. AdamW and weight decay are separate opt-in choices:

.. code-block:: bash

    parrot-train brnn --input-file data/res_regress_dataset.tsv --output-network resumable_example.pt --split data/splitfile.txt -d residues --regression -lr 0.001 -e 50 --regression-loss huber --huber-delta 0.5 --optimizer adamw --weight-decay 0.01 --checkpoint --training-log --seed 1

The default losses remain MAE for sequence data and MSE for residue/matrix data;
weight decay defaults to 0 for both Adam and AdamW. Huber is unavailable with
``--uncertainties``, which uses Gaussian NLL. All loss choices use the same
missing-label masks and optional protein weighting during training and evaluation.

The command writes ``resumable_example_training_checkpoint.pt`` at the initial
boundary and after each successful epoch, and flushes epoch losses, rates and
diagnostics to ``resumable_example_training_history.tsv`` as they become
available. To continue after a job interruption:

.. code-block:: bash

    parrot-train --resume resumable_example_training_checkpoint.pt

The saved arguments and actual data split are restored. Inputs must remain
unchanged; loss, batch size, optimizer and schedule changes are rejected. Fixed
iteration runs with a fixed, exponential or plateau schedule can extend the
total duration, for example by appending ``--epochs 100``. Cosine, automatic
stopping and search budgets retain their original planned epochs. See
:ref:`parrot-training-resume` for state coverage and compatibility checks.

BRNN training, including the examples above, requires FP32: omit ``--amp`` or
use ``--amp off``. All enabled AMP modes and saved BRNN AMP runs are rejected.
For other neural architectures, a supported CUDA GPU can use mixed precision:

.. code-block:: bash

    parrot-train cnn --input-file data/res_regress_dataset.tsv --output-network amp_example.pt --split data/splitfile.txt -d residues --regression --device cuda --amp --checkpoint --training-log

Auto selects native BF16 when available, otherwise FP16; explicit
``--amp float16`` and ``--amp bfloat16`` are supported. Loss arithmetic and
validation remain FP32. Unsupported devices fail, and FP16 overflow uses bounded
same-batch scaling retries without silently discarding batches. Compare a
matched-seed FP32 pilot on the target hardware; see :ref:`parrot-amp`.

**Splitting data into train/validation/test sets:**

``--split``:
Every ``parrot-train`` and ``parrot-optimize`` command says how to divide the data into training, validation and test sets. ``--split mmseqs`` and ``--split kmer`` keep similar sequences in the same set (found by MMseqs2, or by PARROT's own k-mer search), so that the test set measures how well the network does on sequences unlike any it was trained on; ``--split random`` assigns sequences at random, keeping only identical sequences together. These are the same methods as ``parrot-preprocess --split homology``, ``kmer`` and ``random``. The example datasets in the **/data** folder were generated at random, so a random split is fine for them; for real proteins, where relatives are common, prefer ``mmseqs`` or ``kmer``.

``--set-fractions``:
This flag allows the user to set the proportions of data that will be a part of the training set, validation set, and test set, when ``--split`` is a method. By default, the split is 70:15:15. This flag takes three input arguments, between 0 and 1, that must sum to 1. The similarity splits keep related sequences together, so their sets come close to these fractions rather than matching them exactly.

.. code-block:: bash

    parrot-train brnn --input-file data/seq_regress_dataset.tsv --output-network setfractions_network.pt --split random --datatype sequence --regression -e 200 --set-fractions 0.5 0.45 0.05

Notice that the output predictions file from this command has fewer datapoints because of the reduced test set. Most likely, the accuracy will be a little worse than with the default proportions because the training set is also smaller.

Split-files:
In some cases, users might want precise control over over the training, validation and test set splits of their input data. Instead of a method, ``--split`` can be given the path to a *split_file*, which specifically allocates sequences in `datafile` to the different datasets. An example *split_file* is provided in the **/data** folder for reference. If a file exists with the name given to ``--split``, it is always read as a split-file, even if its name is also a method's.

.. code-block:: bash

    parrot-train brnn --input-file data/seq_regress_dataset.tsv --output-network manualsplit_network.pt --split data/splitfile.txt --datatype sequence --regression -e 200

This can especially be useful if you wish to perform k-fold cross-validation on your dataset, as you can prepare k different split_files that each specify a particular 1/kth of your dataset into the test set.

``--save-splits``:
When ``--split`` is a method, it is helpful to know for replicability where each sample was assigned. For example, if you are comparing multiple types of machine learning networks, it is best practice to use the same training set for each network. Including this flag causes an additional text file (suffix: "_split_file.txt") to be saved to the output directory. This file is formatted in the same way as a *split_file* for using with the ``--split`` flag.

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
hydrogen bonding            1 or 0                        1 for side chains that can form hydrogen bonds (D, E, H, K, N, Q, R, S, T, W and Y)
side-chain SASA             square angstroms              side-chain solvent-accessible surface area in ACE-X-NME
backbone SASA               square angstroms              backbone solvent-accessible surface area in ACE-X-NME
solvation free energy       (not recorded)                free energy of solvation of the side chain
==========================  ============================  ===============================================================

.. code-block:: bash

    parrot-train brnn --input-file data/seq_regress_dataset.tsv --output-network biophysics_network.pt --split random -d sequence --regression -nl 2 -hs 10 -e 200 --encode biophysics

The biophysical features are on very different scales (molecular weight is in the hundreds, while the aromatic flag is 0 or 1), so we recommend adding ``--standardize-encoding``, which rescales every feature to mean 0 and standard deviation 1 across the amino acids (see "Standardizing an encoding" on the ``parrot-train`` page).

More powerfully, PARROT also allows the user to manually specify their own encoding scheme, if they desire. An example encoding file can be found in the **/data** folder. In this case, provide the path to this encoding file following the flag.

.. code-block:: bash

    parrot-train brnn --input-file data/seq_regress_dataset.tsv --output-network userencode_network.pt --split random -d sequence --regression -nl 2 -hs 10 -e 200 --encode data/encoding_example.txt

With the ``--encode`` flag and a user-provided file, PARROT is even flexible enough to work on nucleotide sequences! To illustrate this, we've included the file "nucleotide_encoding.txt" which can be passed in via this flag to one-hot encode nucleotide sequences. We've also included an example sequence regression dataset (melting temperature prediction) with nucleotide sequences: "nucleotide_dataset.tsv".

.. code-block:: bash

    parrot-train brnn --input-file data/nucleotide_dataset.tsv --output-network nucleotide_network.pt --split random -d sequence --regression -nl 2 -hs 10 -e 200 --encode data/nucleotide_encoding.txt

Finally, residues can be encoded by their embeddings from a protein language model, which reflect each residue's context in the whole sequence (e.g. ``--encode esm2_t12_35M_UR50D``; see :doc:`plm_encodings`), and, for sequence-mapped data, each sequence's NARDINI+ features can be added to every residue (e.g. ``--encode onehot+nardini``; see :doc:`nardini_encoding`). Whatever the encoding, the saved network records it, so ``parrot-predict`` and ``Predictor`` encode new sequences in the same way without being told.

**Probabilistic classification with --probabilistic-classification:**

The standard behavior of "classification" tasks in PARROT is to make predictions of discrete class labels. In reality though, this sort of behavior does not provide any information on the certainty behind these prediction. For example, in a two class problem (classes 0 and 1), if sequence A is deemed to be class 0 with 98% confidence, and sequence B is deemed class 0 with 51% confidence, both of these sequences will appear in the output prediction file as class 0. In some instances, it is useful to provide users a measure of confidence for each of the class predictions that PARROT makes. This can be accomplished with the ``--probabilistic-classification`` flag.

Using this flag is easy and can be used with ``parrot-train``, ``parrot-optimize`` and ``parrot-predict``. For the first two commands, this flag changes how predictions on the test set are output in the "_predictions.tsv" file and changes the performance stats that are output (if specified). For the predict command, it changes how the predictions are outputed. If this flag is combined with ``--include-figs``, it also changes the figures produced for evaluating performance on the test set: ROC and precision-recall curves instead of a confusion matrix (see ``parrot-train`` documentation page for more details). Conveniently, this flag can be used in ``parrot-predict`` even if it was not specified during training. As an example, here is the same sequence 3-class classification network making predictions with and without the ``--probabilistic-classification`` flag (default layers and hidden vector size):

.. code-block:: bash

    parrot-predict data/seqfile.txt prob_example.pt discrete.txt

Output:

.. code-block:: bash

    a1 EADDGLYWQQN 2
    b2 RRLKHEEDSTSTSTSTSTQ 0
    c3 YYYGGAFAFAGRM 2
    d4 GGIL 2
    e5 GREPCCMLLYILILAAAQRDESSSSST 2
    f6 PGDEADLGHRSLVWADD 2

.. code-block:: bash

    parrot-predict data/seqfile.txt prob_example.pt probabilistic.txt --probabilistic-classification

Output:

.. code-block:: bash

    a1 EADDGLYWQQN 0.0527 0.1081 0.8392
    b2 RRLKHEEDSTSTSTSTSTQ 0.9819 0.0034 0.0148
    c3 YYYGGAFAFAGRM 0.0742 0.0098 0.9160
    d4 GGIL 0.1509 0.0596 0.7894
    e5 GREPCCMLLYILILAAAQRDESSSSST 0.0465 0.0118 0.9418
    f6 PGDEADLGHRSLVWADD 0.0645 0.2576 0.6780

The three numbers following each sequence represent the probability that the sequence belongs to each of the three classes. Notice the numbers in each row sum to 1 (up to rounding to four decimal places).

Probabilistic classification also works for *residue classification*, where each sequence has num_classes x seq_len values. So that there is still one value per residue in the output files, each residue's class probabilities are written as one comma-separated token. For example, with the residue classification network from the basic examples:

.. code-block:: bash

    parrot-predict data/seqfile.txt res_class_network.pt res_probabilistic.txt --probabilistic-classification

.. code-block:: bash

    a1 EADDGLYWQQN 0.0388,0.7660,0.1952 0.0390,0.6206,0.3404 0.0118,0.8347,0.1535 ...
    ...

For residue data, the ROC and precision-recall curves and their areas pool every labelled residue of the test set.

Hyperparameter tuning with ``parrot-optimize``:
-----------------------------------------------

``parrot-optimize`` trains a network like ``parrot-train``, but chooses the hyperparameters for you: the initial learning rate and the architecture's main hyperparameters (for the BRNN, the number of layers and the hidden size). It tries many configurations, drops poor ones after a few epochs (successive halving) and suggests new ones where good results have clustered (TPE); see the ``parrot-optimize`` page for details. The default 50 trials cost about as much as 15-20 ``parrot-train`` runs, so a GPU helps.

Usage is almost identical to ``parrot-train``. For example, on the residue classification example dataset:

.. code-block:: bash

    parrot-optimize brnn --input-file data/res_class_dataset.tsv --output-network optimize_example.pt --split random -d residues --classification --num-classes 3 -e 30 --trials 20 --seed 1

We do not specify the number of layers, the hidden size or the learning rate: these are searched. The most important choice is ``--epochs``, the number of epochs a configuration is trained for if it survives to the end (and so the final network's training): choose it as you would for ``parrot-train``, e.g. by looking at a training and validation loss curve. Here we use 20 trials to keep the example short; the default is 50.

The same opt-in training controls apply to every configuration and fold.
For example, fix a cosine schedule and finite checks while searching the
initial learning rate and the BRNN's architecture:

.. code-block:: bash

    parrot-optimize brnn --input-file data/res_class_dataset.tsv --output-network optimize_cosine_example.pt --split random -d residues --classification --num-classes 3 -e 30 --trials 20 --lr-schedule cosine --lr-warmup-epochs 3 --check-finite-gradients --seed 1

The cosine horizon is the full 30-epoch budget minus warmup, even for the
short first stage. Schedule state continues across promotions. Recovery, if
enabled, has a budget across all stages of each fold, with a fresh budget for
final retraining after cross-validation. Policies themselves are fixed rather
than searched; see :doc:`training_controls`.

The same loss, regularization, checkpoint and log options work throughout a
search. For example:

.. code-block:: bash

    parrot-optimize brnn --input-file data/res_regress_dataset.tsv --output-network resumable_search.pt --split data/splitfile.txt -d residues --regression -e 30 --trials 20 --regression-loss huber --optimizer adamw --weight-decay 0.01 --checkpoint --training-log --seed 1

Continue an interrupted search, including a partial fold or final CV retraining,
with ``parrot-optimize --resume resumable_search_training_checkpoint.pt``.
``resumable_search_search_training_history.tsv`` records every successful
trial/fold epoch; final CV retraining has its own epoch log. The held-out test
reports include absolute errors, per-protein metrics and sequence-length bins.

For the original command without a schedule, the search first describes what it will try, then reports each configuration's validation loss at the end of each stage it reaches (this output is from an NVIDIA GPU; it took about 40 seconds):

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
      learning_rate=0.00864999  hidden_size=5  num_layers=1  dropout=0  attention_pooling=True  loglength_feedforward=True
    ...

With ``--epochs 30`` there are three stages: every configuration is trained for 3 epochs, the best third of them continue to 10 epochs, and the best third of those to 30. Trial 1 is PARROT's default hyperparameters, trials 2-9 are spread evenly over the search space, and from trial 10 the search mostly suggests configurations near the best so far ("model"), with some drawn at random. Configurations that are clearly worse, such as those with very small learning rates, are stopped after 3 epochs. The best configuration has already been trained for the full 30 epochs, so it becomes the final network, which is then tested on the held-out test set exactly as in ``parrot-train``.

As well as the usual ``parrot-train`` output, ``optimize_example_optimal_hyperparams.txt`` lists the chosen hyperparameters, and ``optimize_example_hyperparameter_trials.tsv`` records every trial:

.. code-block:: bash

    # parrot-optimize trials for the brnn architecture
    trial  origin  status  epochs  val_loss_3_epochs  val_loss_10_epochs  val_loss_30_epochs  learning_rate  hidden_size  num_layers
    1  default  stopped  3  20.6142      0.001  10  1
    2  initial  completed  30  20.2055  11.887  10.0479  0.00146485  24  2
    3  initial  stopped  10  18.1306  11.9684    0.00880616  5  1
    4  initial  stopped  3  20.8106      3.47937e-05  17  5
    ...

Hyperparameters you give on the command line are fixed rather than searched, e.g. ``parrot-optimize cnn ... --channels 32`` searches everything except the number of channels. For small datasets, ``--cv-folds 5`` scores each configuration by 5-fold cross-validation instead of on the validation set.

Integrating trained PARROT networks into Python workflows:
----------------------------------------------------------

We added the option for users to create a predictor object in Python using their trained PARROT network. This option is built-in to the file "py_predictor.py" that is installed with PARROT, and ``Predictor`` can be imported straight from PARROT:

.. code-block:: python

    >>> from parrot import Predictor

To use a saved network, you need to create a Predictor() object. Initializing this object only requires the path to the saved network: the network file records its architecture, hyperparameters, datatype, number of classes and encoding (including any standardization, protein language model or NARDINI settings), so new sequences are encoded in the same way as the training sequences. (Networks saved by earlier versions of PARROT only record their weights; for those, also give ``dtype='sequence'`` or ``dtype='residues'``. For newer networks, ``dtype``, if given, is checked against the network.)

.. code-block:: python

    >>> my_predictor = Predictor('/path/to/network.pt')

Now we're ready to make predictions! Once a network is loaded, the time to make predictions is small (for networks trained on protein language model embeddings or NARDINI features, encoding each new sequence is the slow step; see below), so your predictor can be applied to as many sequences as you want. Just feed in amino acid sequences to the predict() function one at a time (each is converted to upper case first) and predicted values will be output as a NumPy array.

.. code-block:: python

    >>> value = my_predictor.predict('MYTESTAMINACIDSEQ')

What predict() returns depends on the network: for regression, an array of one value (sequence networks) or one value per residue (residue networks); for classification, the probability of each class, with shape [number of classes] (sequence networks) or [sequence length, number of classes] (residue networks); and for matrix networks, the predicted matrix (see the "Matrix data" page).

To predict many sequences, give them all to predict_batch(), which returns a list of these arrays, one per sequence in the same order:

.. code-block:: python

    >>> values = my_predictor.predict_batch(['MYTESTAMINACIDSEQ', 'MSDNEEGQSKRRQ', 'MKVLAAGIVGLLLA'])

This predicts the sequences together, as ``parrot-predict`` does, which for more than a handful of sequences is much faster than calling predict() on each. Every sequence is encoded first: for a protein language model, the model runs over many sequences at once, and NARDINI+ features are computed on several CPU cores. The network then runs over batches of ``batch_size`` sequences of similar length (default 64; a smaller batch uses less memory, which a large matrix network may need). Each distinct sequence is predicted once. The predictions match predict()'s up to float32 rounding, not exactly: which sequences share a batch, and so ``batch_size``, changes the order of the floating-point operations, giving differences of the order of 1e-7 relative. For a network trained on protein language model embeddings, the model also runs in batches, and its float32 rounding can move a value of the stored (16-bit) embeddings by one step, so the predictions can differ by more. ``show_progress=True`` prints what is being encoded, with a progress bar.

Predictions run on a GPU (CUDA, else Apple-silicon MPS) if one is available and otherwise on the CPU, as ``parrot-predict``'s do; ``Predictor('/path/to/network.pt', device='cpu')`` uses the CPU anyway, which can be faster for a few sequences at a time and is needed inside forked worker processes (where MPS cannot be used), and ``gpu_id=`` chooses which NVIDIA GPU, as ``--gpu-id`` does for ``parrot-predict``. For networks trained on protein language model embeddings or NARDINI features, each new sequence must be encoded first, which is the slow step; ``embedding_cache='<directory>'`` stores these encodings so that sequences predicted again are not encoded again (see :doc:`plm_encodings`). Without it, predict() writes nothing to disk, and predict_batch() encodes its sequences into a temporary cache (in the system's temporary directory) that is deleted once they are predicted. Either way, protein language model embeddings take about (number of residues) x (values per residue) x 2 bytes of disk while they are there (NARDINI+ features far less: 90 values per sequence).

A network trained with ``--embedding-context`` embeds each sequence within the full-length protein it was cut from, so ``predict()`` needs that protein too, and finds the sequence in it (``start=`` gives its 1-based start if it occurs more than once). ``predict_batch()`` takes one full-length protein per sequence (``full_lengths=``, and ``starts=`` if needed), and embeds each protein once, however many of the sequences come from it. ``Predictor('/path/to/network.pt', embed_in_isolation=True)`` embeds a sequence given without its full-length protein (or with None in ``full_lengths``) on its own instead of raising an error.

.. code-block:: python

    >>> value = my_predictor.predict('SPLSWEETKRHADH', full_length='MGLLSQGSPLSWEETKRHADHVRRHGILQF')
    >>> values = my_predictor.predict_batch(['SPLSWEETKRHADH', 'VRRHGILQF'],
    ...                                     full_lengths=['MGLLSQGSPLSWEETKRHADHVRRHGILQF'] * 2)
