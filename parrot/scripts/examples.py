"""
The guide each command prints with --show-examples: what the command does,
examples of how to use it, and how it works.

Every command has a --show-examples option (see add_examples_argument()), which
prints the command's header and guide and exits, like --help, without needing
any other arguments.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import datetime
from collections.abc import Sequence
from typing import Any

from parrot.scripts.arguments import EXAMPLES_FLAG, header

TRAIN_GUIDE = '''
parrot-train: train a network, then test it on sequences it has not seen
=========================================================================

What it does
------------
parrot-train reads a datafile of sequences and the values to predict for them,
splits it into training, validation and test sets, and trains a network of the
architecture you choose on the training set. It keeps the network from the
epoch with the lowest loss on the validation set, saves it, and reports how
well it predicts the test set, which it never saw while training. It also
writes <network>_info.txt, which describes how the network was trained in
enough detail for a paper's methods section.

Examples
--------
Predict one value per sequence (regression) with the original PARROT network,
a bidirectional LSTM:

    parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression

Classify every residue into one of three classes with a convolutional
network, saving figures of the training and of the test-set results:

    parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --classification --num-classes 3 --include-figs

Choose the hyperparameters yourself (each architecture's are listed by
parrot-train <architecture> -h):

    parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression -hs 20 -nl 2 -lr 0.0005 -e 200

A fast baseline that ignores the order of residues: k-nearest neighbours on
sequence composition (these simple models learn one value per sequence, so -d
is not needed):

    parrot-train knn --input-file data.tsv --output-network network.pt --split kmer --regression

Choose how the data are divided into training, validation and test sets
(--split). 'mmseqs' (which needs MMseqs2) and 'kmer' keep similar sequences in
the same set, so that no test sequence has a close relative in the training
set; 'random' does not, so it can flatter the test results. Each aims for the
fractions given by --set-fractions (70:15:15 by default), and --save-splits
keeps the split as a split-file to reuse:

    parrot-train brnn --input-file data.tsv --output-network network.pt --split mmseqs --set-fractions 0.8 0.1 0.1 -d sequence --regression
    parrot-train brnn --input-file data.tsv --output-network network.pt --split random --save-splits -d sequence --regression

Train on a split-file, e.g. from parrot-preprocess, which also cleans the data
and lets you change what counts as similar:

    parrot-preprocess data.tsv clean.tsv -d sequence -m regression --split kmer --kmer-size 3
    parrot-train brnn --input-file clean.tsv --output-network network.pt --split clean_split_file.txt -d sequence --regression

Encode residues with a protein language model, or add each sequence's NARDINI+
features to every residue:

    parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression --encode esm2_t12_35M_UR50D
    parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression --encode onehot+nardini

Learn a matrix for each sequence (e.g. a contact map) with a 2D U-Net:

    parrot-train unet2d --input-file data.tsv --output-network network.pt --split kmer --regression --matrix-file maps.npz --symmetric

Train on values that each come with the standard deviation of their
measurement error, written as 'value,SD' (e.g. 3.21,0.15), so that precise
values count for more than noisy ones:

    parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression --uncertainties

Inspect gradient norms at a stable learning rate before choosing an optional
--gradient-clip-norm threshold (there is no universal threshold):

    parrot-train brnn --input-file data.tsv --output-network gradient_pilot.pt --split kmer -d residues --regression -lr 0.001 -e 5 --check-finite-gradients --verbose

Anneal the initial rate with cosine decay after five warmup epochs. The decay
duration defaults to the full run minus warmup; --lr-anneal-epochs sets it
explicitly and is required with --stop auto:

    parrot-train brnn --input-file data.tsv --output-network cosine_network.pt --split kmer -d residues --regression -lr 0.001 -e 50 --lr-schedule cosine --lr-warmup-epochs 5 --lr-min 0.00001

Reduce the rate when validation loss stalls, and allow at most two numerical
recoveries over the whole run. Recovery restores a clean epoch boundary and
replays the epoch at a lower rate; it cannot resume a crashed process:

    parrot-train brnn --input-file data.tsv --output-network plateau_network.pt --split kmer -d residues --regression -lr 0.001 --lr-schedule plateau --lr-patience 3 --lr-min 0.000001 --stop auto -e 10 --recover-nonfinite 2

Give each protein's mean labelled loss equal weight, instead of the default
sum of residue losses (also supported for matrix tasks). Choose this for an
average-protein objective; longer proteins otherwise contribute more labelled
values. Equal objective weight does not force equal gradients. Retune the rate
and any clipping threshold when the loss weighting changes:

    parrot-train brnn --input-file data.tsv --output-network protein_weighted.pt --split kmer -d residues --regression --equal-protein-weighting

Choose Huber loss, opt in to AdamW with weight decay, and retain a resumable
checkpoint plus continuously flushed epoch records:

    parrot-train brnn --input-file data.tsv --output-network resumable.pt --split kmer -d residues --regression --regression-loss huber --huber-delta 0.5 --optimizer adamw --weight-decay 0.01 --checkpoint --training-log

Resume after an interruption (saved arguments and actual data partitions are
restored; inputs and training policies must remain unchanged):

    parrot-train --resume resumable_training_checkpoint.pt

On a supported CUDA GPU, opt in to mixed precision. Auto prefers native BF16,
else FP16; CPU/MPS are unsupported. Loss arithmetic and validation stay FP32,
and FP16 gradient overflow has bounded same-batch retries without skipped batches:

    parrot-train brnn --input-file data.tsv --output-network amp_network.pt --split kmer -d residues --regression --device cuda --amp --checkpoint --training-log

How it works
------------
1. The datafile (--input-file) has '<ID> <sequence> <value(s)>' on each line:
   one value per sequence (-d sequence) or one per residue (-d residues). For
   matrix data (unet2d) it has '<ID> <sequence>', and each sequence's matrix
   is in --matrix-file. The values are numbers to predict (--regression; with
   --uncertainties, each written as 'value,SD') or classes labelled 0, 1, ...
   (--classification, with --num-classes). Every line is checked first.
2. The data are divided into training, validation and test sets (--split),
   as a split-file says or by one of parrot-preprocess's methods: 'mmseqs' or
   'kmer' put similar sequences (found by MMseqs2, or by PARROT's own k-mer
   search) in the same set and then remove any sequence still similar to one
   in another set; 'random' assigns sequences at random, keeping identical
   ones together. A method aims for --set-fractions (70:15:15 by default). If
   a file has the name given to --split, it is read as a split-file, even if
   the name is also a method's. --seed makes the split, and training,
   repeatable.
3. Each residue is encoded as a vector of numbers (--encode; one-hot by
   default). Protein language model and NARDINI encodings are slow to
   compute, so every sequence is encoded once before training, into a
   parrot_train_embeddings directory next to the network (or
   --embedding-cache). With --embedding-context and --embedding-context-map,
   a protein language model embeds each sequence within the full-length
   protein it was cut from, rather than on its own. --embed-mean-subtraction
   subtracts each embedding's mean over its residues before it is stored.
4. The network is trained for -e epochs (100 by default). After every epoch
   its loss on the validation set is measured, and the network from the epoch
   with the lowest validation loss is the one kept, so training for too long
   does no harm. With --stop auto, training instead ends once -e epochs in a
   row have not lowered the validation loss by at least 0.5%. The simple
   models (linear, svm, random_forest, knn, gbrt) are fitted in one step
   instead.
   With --uncertainties, each value is weighted by 1 / (SD^2 + s^2), where
   the model error s, the error of the predictions beyond the values' own, is
   fitted on the validation set after every epoch (the simple models are
   refitted as it is re-estimated, and knn does not support --uncertainties),
   and --stop auto counts the 0.5% as a gain in likelihood per labelled value
   (about 0.005 in the mean per-protein NLL with --equal-protein-weighting).
   The initial rate (-lr) stays fixed by default. --lr-schedule selects
   fixed, cosine, exponential or validation-driven plateau decay; optional
   --lr-warmup-epochs applies before any schedule. Early-stopping patience
   does not reset on a rate change. Finite checks and clipping are opt in;
   recovery also checks model and optimizer state. Invalid inputs are not
   retried, and exhausted recovery budgets stop the run.
5. The kept network is saved as --output-network (with '.pt' added if the
   name does not end with it); the file records its architecture,
   hyperparameters, task and encoding, so parrot-predict needs only the file.
   It is then tested: predictions for the test set go to
   <network>_predictions.tsv (<network>_predictions.npz for matrix data), and
   performance metrics to <network>_performance_stats.txt (and figures, with
   --include-figs).
6. <network>_info.txt records everything about how the network was trained:
   the exact command, software versions, the data (with a checksum of the
   datafile), the split, the encoding, the architecture and its number of
   parameters, the loss, optimizer, epochs and the epoch kept, and the results.
   With schedules, warmup or recovery, <network>_training_history.tsv records
   successful epochs, losses, rates used and cumulative recoveries. Any
   recoveries also go to <network>_recovery_events.json. These files are
   written on successful completion. Clipping, checks or protein weighting
   alone do not create a history file. --training-log writes richer records
   during training, including fixed-rate runs. --checkpoint separately saves
   full resumable state (model, optimizer, best weights, LR/scaler, stopping,
   recovery, randomness and actual split), every epoch by default; change the
   interval with --checkpoint-every. .pt remains the best inference network.
   Test metrics include MAE, RMSE, signed prediction error and R2, with mean
   protein errors for residue/matrix tasks. _per_protein_metrics.tsv and
   _length_metrics.tsv add counts and stratified results; --no-stats suppresses
   these and _performance_stats.txt.

See also
--------
parrot-train -h                   the architectures you can train
parrot-train <architecture> -h    an architecture's options and hyperparameters
parrot-optimize                   choose the hyperparameters for you
parrot-predict                    predict new sequences with a trained network
'''

OPTIMIZE_GUIDE = '''
parrot-optimize: find good hyperparameters, then train and test a network
=========================================================================

What it does
------------
parrot-optimize searches for good hyperparameters for an architecture on your
data, then trains and tests a network with the best ones. It writes the same
files as parrot-train, plus the chosen hyperparameters and a record of every
configuration it tried.

Examples
--------
Search the BRNN's hidden size and number of layers, and the initial learning rate:

    parrot-optimize brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression

A shorter search for a CNN on residue data, with the number of channels fixed
(any hyperparameter you give is fixed rather than searched):

    parrot-optimize cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --classification --num-classes 3 -e 30 --trials 20 --channels 64

For a small dataset, score each configuration by 5-fold cross-validation
rather than on a single validation set (about 5 times slower):

    parrot-optimize composition --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression --cv-folds 5

Fix a cosine schedule and finite checks for all trials while searching the
initial rate and architecture. Schedule state continues across stages, using
the full --epochs horizon rather than restarting at each promotion:

    parrot-optimize brnn --input-file data.tsv --output-network cosine_search.pt --split kmer -d residues --regression -e 30 --trials 20 --lr-schedule cosine --lr-warmup-epochs 3 --check-finite-gradients

Keep full search checkpoints and continuously record every trial/fold epoch:

    parrot-optimize brnn --input-file data.tsv --output-network resumable_search.pt --split kmer -d residues --regression --regression-loss huber --optimizer adamw --weight-decay 0.01 --checkpoint --training-log

Resume an interrupted search, including a partial fold, promotion or final
cross-validated retraining. Saved arguments and actual folds are restored:

    parrot-optimize --resume resumable_search_training_checkpoint.pt

See what an architecture searches, and over what values:

    parrot-optimize cnn -h

How it works
------------
1. The datafile is read, checked and split exactly as by parrot-train.
2. The first configuration tried is PARROT's defaults, and the next few are
   spread evenly over the search space. After that, new configurations are
   suggested where good results have clustered so far (a tree-structured
   Parzen estimator, TPE), with a third drawn at random so the whole space
   keeps being explored.
3. Successive halving keeps the search cheap: every configuration is first
   trained for a few epochs (--min-epochs, by default a ninth of --epochs), and
   only the best third of each stage continue, for three times as many epochs
   (--reduction-factor, 3 by default), up to --epochs (with -e 30, the stages
   are 3, 10 and 30 epochs). Poor configurations are dropped early. The simple
   models (linear, svm, random_forest, knn, gbrt) are fitted in one step, so
   each of their configurations is simply fitted once. Hyperparameters that
   have no effect on your data (e.g. the composition window for sequence
   data) are not searched.
4. Each configuration is scored by its lowest validation loss, exactly as
   parrot-train chooses its best epoch (with --cv-folds, by its average over
   the folds, which keep identical sequences together, and with --split kmer
   or mmseqs similar ones too).
5. The best configuration's network has already been trained for --epochs on
   the training set, so it is saved and tested on the test set (with
   --cv-folds, a network is first trained on the training set with the best
   configuration). <network>_optimal_hyperparams.txt gives its
   hyperparameters, <network>_hyperparameter_trials.tsv records every trial as
   it finishes, and <network>_info.txt describes how the network was trained,
   including the search. Schedules, warmup, gradient safeguards, residue or
   matrix protein weighting and bounded recovery are shared with parrot-train.
   These policies are fixed, not searched. Recovery budgets persist across
   stages, separately for each fold and for final retraining. A trial that
   cannot recover is marked failed and the search continues. Optional epoch
   history and recovery files describe the winning or final retrained network,
   rather than every trial, and are written on successful completion unless
   --training-log enables continuous records. Then _search_training_history.tsv
   records every successful trial/fold epoch, and final CV retraining has its
   own _training_history.tsv. --checkpoint and --resume also preserve the TPE
   random state, tried configurations, partial folds and promotion progress.
   Loss, optimizer and AMP settings are fixed across all trials and retraining.

The default 50 trials cost about as much as 15-20 parrot-train runs with the
same --epochs, so a GPU helps.

See also
--------
parrot-optimize -h                    the architectures you can search
parrot-optimize <architecture> -h     what is searched, and the search options
'''

PREDICT_GUIDE = '''
parrot-predict: predict new sequences with a trained network
============================================================

What it does
------------
parrot-predict uses a network trained by parrot-train or parrot-optimize to
predict values for new sequences. The network file records how the network was
trained (its architecture, task and encoding), so no other options are needed,
except for a network trained with --embedding-context (see below).

Examples
--------
Predict every sequence in a file of '<ID> <sequence>' lines:

    parrot-predict sequences.txt network.pt predictions.txt

The same, for a file with just one sequence on each line:

    parrot-predict sequences.txt network.pt predictions.txt --exclude-seq-id

For classification, the probability of each class rather than just the most
likely one (for residue networks, each residue's probabilities are written as
one comma-separated token, e.g. 0.9100,0.0600,0.0300):

    parrot-predict sequences.txt network.pt predictions.txt --probabilistic-classification

For a network trained with --embedding-context, the full-length proteins the
sequences were cut from, and which one each was cut from, as for training (or
--embed-in-isolation to embed each on its own, which may make the predictions
less accurate):

    parrot-predict sequences.txt network.pt predictions.txt --embedding-context proteome.fasta --embedding-context-map map.tsv

On the CPU, printing nothing (e.g. in a script):

    parrot-predict sequences.txt network.pt predictions.txt --device cpu --silent

From Python, one sequence at a time, or a list of sequences at once (as
parrot-predict predicts them, and much faster than one at a time):

    from parrot import Predictor
    predictor = Predictor('network.pt')
    values = predictor.predict('MKVLAAGIVGLLLA')
    all_values = predictor.predict_batch(['MKVLAAGIVGLLLA', 'MSDNEEGQSKRRQ'])

How it works
------------
1. The network and the encoding it was trained with are loaded from its file.
   (Networks saved by PARROT 1.x record only their weights, so they need
   --datatype, and --encode if they were not trained on one-hot encoding.)
2. Every sequence is checked first, so a residue the encoding cannot read
   stops the command before anything is predicted. Networks trained on a
   protein language model or NARDINI encoding then encode every sequence once,
   into a parrot_predict_embeddings directory next to the output (or
   --embedding-cache). A network trained with --embedding-context needs the
   same two options (or --embed-in-isolation), so each sequence is embedded
   within its full-length protein, as the training sequences were. A network
   trained with --embed-mean-subtraction subtracts each embedding's mean
   automatically.
3. Sequences are predicted in batches of similar length (up to --batch, 64
   by default), and each unique sequence is predicted once.
4. The output has one line per sequence of the sequence file, in the same
   order (blank lines and lines starting with '#' are skipped): the ID (unless
   --exclude-seq-id), the sequence, and its predicted value(s), one per
   sequence or one per residue. Networks for matrix data write a NumPy .npz
   archive instead, with one matrix per sequence, named by its ID (or 1, 2, ...
   with --exclude-seq-id).
'''

PREPROCESS_GUIDE = '''
parrot-preprocess: get a datafile ready for training
====================================================

What it does
------------
parrot-preprocess turns a datafile into one ready for training. It can clean
sequences, recode class labels, deal with duplicate sequences, split the data
into training, validation and test sets with no leakage between them, and
balance the classes of the training set. It writes the processed datafile, a
matching split-file to train on with parrot-train --split, and a report of
every step. It can also compute a protein language model or NARDINI encoding
of every sequence ahead of time.

Examples
--------
Split regression data so that no test sequence has a close relative in the
training set (needs nothing but PARROT), then train on the split:

    parrot-preprocess data.tsv clean.tsv -d sequence -m regression --split kmer --seed 1
    parrot-train brnn --input-file clean.tsv --output-network network.pt --split clean_split_file.txt -d sequence --regression

The most thorough split, which also finds distant homologues with MMseqs2
(installed separately, e.g. conda install -c conda-forge -c bioconda mmseqs2):

    parrot-preprocess data.tsv clean.tsv -d sequence -m regression --split homology

If separation reaches its cleanup limit, inspect the saved report's removal
totals and remaining set sizes. To allow more rounds with the same similarity
criteria (use the same seed as the earlier run, if it had one):

    parrot-preprocess data.tsv clean_retry.tsv -d sequence -m regression --split homology --max-separation-rounds 25 --seed 42 --verbose

Clean a classification dataset: convert sequences to upper case, remove those
with non-standard residues or fewer than 10 residues, merge duplicate
sequences, split at random and balance the classes of the training set:

    parrot-preprocess data.tsv clean.tsv -d sequence -m classification --uppercase --remove-nonstandard --min-length 10 --deduplicate merge --split random --class-balance

Recode class labels written as text (e.g. 'nuclear', 'cytoplasmic') as the
numbers 0, 1, ... that PARROT needs:

    parrot-preprocess data.tsv clean.tsv -d sequence -m classification --recode-classes

Merge duplicate sequences whose values each come with the standard deviation
of their measurement error ('value,SD'), weighting each copy by 1 / SD^2:

    parrot-preprocess data.tsv clean.tsv -d sequence -m regression --uncertainties --deduplicate merge

Split the data, then compute the ESM-2 embedding of every sequence of the
output ahead of time, so that training on it starts straight away:

    parrot-preprocess data.tsv clean.tsv -d sequence -m regression --split kmer --encode esm2_t33_650M_UR50D --embedding-cache embeddings
    parrot-train cnn --input-file clean.tsv --output-network network.pt --split clean_split_file.txt -d sequence --regression --encode esm2_t33_650M_UR50D --embedding-cache embeddings

Embed each peptide of a tiling library within the full-length protein it was
cut from, rather than on its own. proteome.fasta holds the full-length
proteins, and tile_map.tsv has a '<peptide ID> <protein ID> [start]' line per
peptide. parrot-train then needs the same --encode, --embedding-cache,
--embedding-context and --embedding-context-map:

    parrot-preprocess tiles.tsv clean.tsv -d sequence -m regression --split kmer --encode esm2_t33_650M_UR50D --embedding-cache embeddings --embedding-context proteome.fasta --embedding-context-map tile_map.tsv

How it works
------------
The steps always run in this order, and each is optional:

1. Cleaning (--uppercase, --remove-nonstandard, --min-length, --max-length).
2. Recoding class labels (--recode-classes); the mapping is written to
   <output>_class_mapping.txt.
3. Deduplication (--deduplicate merge, first or drop, and
   --remove-line-duplicates).
4. Splitting (--split homology, kmer or random). Similar sequences are grouped
   and whole groups are assigned to sets, then anything still similar to a
   sequence in another set is removed (keeping test, then validation, then
   training sequences).
5. Balancing the classes of the training set (--class-balance).
6. Encoding every sequence of the output ahead of time (--encode, for protein
   language model and NARDINI encodings), into the embedding cache that
   parrot-train reads with --embedding-cache; with --embedding-context, within
   each sequence's full-length protein (where each one is embedded is written
   to <output>_embedded_context.tsv), and with --embed-mean-subtraction, with
   each embedding's mean over its residues subtracted.

Every step is recorded in <output>_preprocess_report.txt (-v also prints it),
including per-round removals, set retention and example cross-set conflicts.
The report is saved on data-preparation, splitting or encoding failure too,
with the error and available recovery advice. It distinguishes failures before
the data files are written from encoding failures afterwards. Successful runs
check that parrot-train can read the output and, with --encode, record the
options parrot-train needs to use the encodings, ending with COMPLETED.
'''

CVSPLIT_GUIDE = '''
parrot-cvsplit: split-files for k-fold cross-validation
=======================================================

What it does
------------
parrot-cvsplit writes one split-file per fold for k-fold cross-validation.
Each fold's test set is a different tenth (or 1/k) of the data, and the rest
of that fold is divided into training and validation sets.

Examples
--------
Ten folds, then train a network on each of them (in a directory cv, which must
exist):

    mkdir cv
    parrot-cvsplit data.tsv cv/splits -k 10 --seed 1
    parrot-train brnn --input-file data.tsv --output-network cv/network0.pt --split cv/splits_cv0.txt -d sequence --regression
    ...
    parrot-train brnn --input-file data.tsv --output-network cv/network9.pt --split cv/splits_cv9.txt -d sequence --regression

Five folds, training on 90% of each fold's non-test sequences (written as
cv/five_fold_cv0.txt to cv/five_fold_cv4.txt):

    parrot-cvsplit data.tsv cv/five_fold -k 5 -t 0.9

For a datafile without sequence IDs (each line starts with the sequence):

    parrot-cvsplit data.tsv cv/splits --exclude-seq-id

How it works
------------
The sequences are shuffled (repeatably, with --seed) and divided into k folds
of near-equal size, keeping identical sequences in the same fold. Fold i is
the test set of split-file i, and the other sequences are divided between its
training set (-t, 0.8 by default) and its validation set, again keeping
identical sequences together. Each split-file has three lines, giving the
training, validation and test sets as line numbers of the datafile (counting
from 0, and skipping blank and comment lines), as parrot-train --split
expects.

Apart from identical sequences, folds are drawn at random, so close relatives
can end up in different sets; parrot-preprocess --split makes splits that keep
them together. To score hyperparameters by cross-validation, use
parrot-optimize --cv-folds.
'''

# Each command's guide, by the command's name
GUIDES = {
    'parrot-train': TRAIN_GUIDE,
    'parrot-optimize': OPTIMIZE_GUIDE,
    'parrot-predict': PREDICT_GUIDE,
    'parrot-preprocess': PREPROCESS_GUIDE,
    'parrot-cvsplit': CVSPLIT_GUIDE,
}


class ShowExamplesAction(argparse.Action):
    """An option that prints a command's header and guide, then exits (like --help)

    argparse runs an option's action as soon as it reads the option, so the
    guide is shown even if the command's other arguments are missing.
    """

    def __init__(self, option_strings: Sequence[str], dest: str, program_name: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        option_strings, dest
            Passed on by argparse
        program_name : str
            The command whose guide to show, e.g. 'parrot-train'
        """
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)
        self.program_name = program_name

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any,
                 option_string: str | None = None) -> None:
        """Print the header and guide, and exit."""
        print(header(self.program_name, datetime.datetime.now(), examples_hint=False))
        print(GUIDES[self.program_name].strip('\n'))
        parser.exit()


def add_examples_argument(parser: 'argparse._ActionsContainer', program_name: str) -> None:
    """Add --show-examples, which prints the command's guide (see GUIDES) and exits

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the option
    program_name : str
        The command, e.g. 'parrot-train'
    """
    parser.add_argument(EXAMPLES_FLAG, action=ShowExamplesAction, program_name=program_name,
                        help='show examples of how to use this command, what it does and how it works, then exit')
