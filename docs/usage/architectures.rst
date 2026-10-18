===============
 Architectures
===============

PARROT can train several kinds of model on the same data. Each is a subcommand of ``parrot-train`` and ``parrot-optimize`` (e.g. ``parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression``), and every command must name one: to train the network earlier versions of PARROT trained, use ``brnn``. There are two families:

* **neural networks** (``brnn``, ``cnn``, ``unet``, ``transformer``, ``composition``), trained epoch by epoch, which work for all four task types (sequence or residue data, regression or classification);
* **simple models** (``linear``, ``svm``, ``random_forest``, ``knn``) from scikit-learn, fitted in one step to a fixed summary of each sequence, which work for sequence-mapped data only.

Matrix data (a matrix of values per sequence) has its own architecture, ``unet2d``, a 2D U-Net described on the "Matrix data" page.

Every model works with every encoding (NARDINI features, which describe whole sequences, are for sequence data only), and the saved network records which model it is, so ``parrot-predict`` needs no extra flags.

Each architecture has its own hyperparameters, set with the flags listed below (and shown, with their defaults, by ``parrot-train <architecture> -h``). Each subcommand offers only its own architecture's flags, so passing another architecture's is an error. In ``parrot-optimize``, hyperparameters marked as searched (over a range of numbers or a set of choices) are tuned automatically, along with the learning rate; any hyperparameter you set on the command line is fixed at that value instead, and the rest keep their defaults.

Which architecture should I use?
---------------------------------

* **composition** is a baseline, not a model to publish. It cannot use the order of residues (for residue data, within its window), so comparing other architectures against it shows how much of their performance actually comes from sequence order rather than composition and length. It trains in seconds; it is worth running on every new dataset.
* **brnn** is the bidirectional LSTM PARROT has always used. It is compact and works well on small datasets, but processes residues one after another, so it is slower to train on long sequences.
* **cnn** is a dilated residual convolutional network. All positions are computed in parallel, so it trains quickly on a GPU, and stacking blocks lets it see long stretches of sequence. It is a strong choice for residue-level tasks.
* **unet** is a 1D U-Net. It combines fine local detail with broad context by repeatedly halving and then restoring the sequence length, with skip connections between matching resolutions.
* **transformer** is a transformer encoder: every residue attends to every other residue of its sequence, so even one block sees the whole sequence. It is the architecture behind protein language models such as ESM-2, here trained from scratch on your data; transformers usually need more data than the BRNN or CNN to do well.
* **linear**, **svm**, **random_forest** and **knn** are classic machine-learning models on sequence features (composition and length, optionally dipeptides). They fit in about a second and are strong baselines for sequence-level properties, especially with small datasets.

As always, compare architectures on the same data split (save one with ``--save-splits`` and give that split-file to every run with ``--split``) and judge them on the validation set, not the test set.

Whole-sequence context, attention pooling and log(length)
---------------------------------------------------------

Three options are on by default in the networks they apply to, and each has a flag that turns it off:

* **Whole-sequence context** (``cnn``, ``unet``; ``--global-context-off``). A convolution only sees a window of residues (61 for the default CNN), so these networks also add a summary of the whole sequence to every position: the average of the features over the sequence's real residues, passed through a learned layer (as in squeeze-and-excitation and global-context networks; Hu et al. 2018, Cao et al. 2019). The CNN adds it in every block and the U-Net at its deepest level, so every residue sees the whole sequence however long it is. The learned layer starts at zero, so training starts from the same network as without it. The BRNN and transformer already see the whole sequence.
* **Attention pooling** (``brnn``, ``cnn``, ``unet``, ``transformer``; sequence data only; ``--attention-pooling-off``). To turn per-residue features into one vector per sequence, each residue gets a learned score and the features are averaged with softmax weights over the sequence's real residues (Ilse et al. 2018), so the network learns which residues matter. With it off, ``--pooling`` (``mean`` or ``max``) is used instead, or, for the BRNN, the final hidden states of its two directions (the forward state after the last residue and the reverse state after the first), as PARROT's BRNN always did. Those two states must carry everything the network knows about the sequence, and an LSTM's memory of residues fades over hundreds of steps; attention pooling instead lets the LSTM output at every residue contribute directly (a bidirectional LSTM with attention pooling is, for example, the architecture of DeepLoc; Almagro Armenteros et al. 2017).
* **log(length) for the output layer** (``brnn``, ``cnn``, ``unet``, ``transformer``; sequence data only; ``--loglength-feedforward-off``). The final layer sees the log of the sequence length as well as the network's summary of the sequence, as the ``composition`` baseline and the simple models always do. Averaged or attention-pooled features carry no direct information about length, which networks otherwise have to infer; on length-dominated properties such as the radius of gyration, networks without it under-predicted the longest sequences.

The saved network records each option, so ``parrot-predict`` and ``Predictor`` need no flags. Networks saved by earlier versions of PARROT, which did not have these options, are loaded with them off and predict exactly as before.

composition
-----------

Order-independent baseline. For sequence data, the encoded residues are averaged over the sequence (for one-hot encoding this is exactly the amino acid composition) and passed, with the log of the sequence length, through a small fully connected network. For residue data, each residue is described by the composition of a window centred on it.

	*  ``--hidden-size`` / ``-hs`` : width of each hidden layer (default 32; searched over 8-128).
	*  ``--num-hidden-layers`` : number of hidden layers; 0 gives a linear model (default 1; searched over 0-3).
	*  ``--dropout`` : dropout probability, at least 0 and below 1 (default 0.0; searched over 0.0-0.5).
	*  ``--window-radius`` : residue data only; number of residues either side of each residue included in its window (default 7, i.e. 15-residue windows; searched over 1-25).

brnn
----

Bidirectional LSTM followed by a fully connected layer. For sequence data the LSTM output at every residue is combined by attention pooling (or, with ``--attention-pooling-off``, the final hidden states of the forward and reverse directions are used); for residue data the LSTM output at each residue.

	*  ``--hidden-size`` / ``-hs`` : size of the LSTM hidden vectors (default 10; searched over 5-50).
	*  ``--num-layers`` / ``-nl`` : number of LSTM layers per direction (default 1; searched over 1-5).
	*  ``--dropout`` : dropout probability, at least 0 and below 1, applied to the LSTM output (or final hidden states) the output layer sees and, with more than one layer, between layers (default 0.0; not searched, so that ``parrot-optimize`` searches the BRNN as it always has; give it to fix it at that value).
	*  ``--attention-pooling-off`` : sequence data only; summarise the sequence by the final hidden states instead of attention pooling (see above).
	*  ``--loglength-feedforward-off`` : sequence data only; do not give the output layer log(sequence length) (see above).

cnn
---

Dilated residual 1D convolutional network, following the ByteNet design (Kalchbrenner et al. 2016) that CARP showed to be competitive with transformers for protein sequences (Yang et al. 2022). Each block normalizes its input, applies a convolution and adds the result back; the spacing between the residues each convolution looks at doubles from block to block (1, 2, 4, 8, then repeating). With the defaults (4 blocks, width 5) each residue's output depends on the 61 residues centred on it.

	*  ``--channels`` : number of convolutional channels (default 64; searched over 16-128).
	*  ``--num-blocks`` : number of residual convolution blocks (default 4; searched over 1-8).
	*  ``--kernel-size`` : width of each convolution; one of 3, 5, 7, 9, 11 (default 5; searched over all five).
	*  ``--dropout`` : dropout probability, at least 0 and below 1 (default 0.0; searched over 0.0-0.5).
	*  ``--pooling`` : sequence data with attention pooling off; how per-residue features are combined, ``mean`` or ``max`` (default mean).
	*  ``--global-context-off``, ``--attention-pooling-off``, ``--loglength-feedforward-off`` : see above.

unet
----

1D U-Net, adapted from the 2D U-Net written by Nick on the ``parrot-lightning`` branch. The encoder halves the sequence length ``depth`` times while doubling the number of channels; the decoder restores the length, combining features from the matching encoder level at each step. Sequences of any length can be used.

	*  ``--base-channels`` : channels at full resolution, doubled at each level down (default 16; searched over 8-64).
	*  ``--depth`` : number of times the sequence length is halved (default 3; searched over 1-5).
	*  ``--kernel-size`` : width of each convolution; one of 3, 5, 7, 9, 11 (default 3; searched over all five).
	*  ``--dropout`` : dropout probability, at least 0 and below 1 (default 0.0; searched over 0.0-0.5).
	*  ``--upsampling`` : how the decoder doubles the length, ``transposed`` (a learned transposed convolution) or ``nearest`` (repeating each position) (default transposed; both are searched).
	*  ``--pooling`` : sequence data with attention pooling off; ``mean`` or ``max`` (default mean).
	*  ``--global-context-off``, ``--attention-pooling-off``, ``--loglength-feedforward-off`` : see above.

Compared with the 2D original, the network runs along the sequence, uses layer normalization instead of batch normalization, and upsamples with a transposed convolution or nearest-neighbour repetition instead of bilinear interpolation. These changes make each sequence's output independent of how much padding it is given when batched with longer sequences.

transformer
-----------

Transformer encoder (Vaswani et al. 2017). Each residue's encoding is projected to the model width (``--num-heads`` x ``--head-size``, 64 by default) and passed through a stack of pre-norm transformer blocks, each a multi-head self-attention layer followed by a feed-forward network, both with residual connections. Padded residues are masked out of attention, so each sequence's output does not depend on how it is batched. The width is set per head so that any number of heads gives a valid model.

Attention itself ignores residue order, so positions are added by a positional encoding:

* ``rotary`` (default): queries and keys are rotated by an angle proportional to the residue's position (RoFormer, Su et al. 2024), so attention depends on how far apart two residues are. ESM-2 uses this.
* ``sinusoidal``: the fixed sine/cosine position vectors of Vaswani et al. 2017 are added to each residue, giving it its absolute position.
* ``none``: no positions; the network sees each sequence as a set of residues, which makes it an order-free comparison like ``composition``.

Neither encoding has a maximum sequence length. On a GPU, attention uses PyTorch's memory-efficient kernel, so memory grows linearly with sequence length (a batch of 32 sequences of 5,654 residues trains in under 2 GB).

	*  ``--num-layers`` / ``-nl`` : number of transformer blocks (default 2; searched over 1-6).
	*  ``--num-heads`` : number of attention heads (default 4; searched over 1-8).
	*  ``--head-size`` : width of each attention head, at least 2 (positions are encoded in pairs of features); the model width is ``num-heads`` x ``head-size`` (default 16; searched over 8-64).
	*  ``--feedforward-multiplier`` : width of the feed-forward layer in each block, as a multiple of the model width (default 4; searched over 1-4).
	*  ``--dropout`` : dropout probability, at least 0 and below 1 (default 0.0; searched over 0.0-0.5).
	*  ``--positional-encoding`` : ``rotary`` (default), ``sinusoidal`` or ``none`` (``rotary`` and ``sinusoidal`` are searched).
	*  ``--pooling`` : sequence data with attention pooling off; ``mean`` or ``max`` (default mean).
	*  ``--attention-pooling-off``, ``--loglength-feedforward-off`` : see above.

Simple models (sequence data only)
----------------------------------

These models summarize each sequence as a fixed set of features and fit a scikit-learn model to them in a single step, so the learning rate, number of epochs and stop condition are not used (and no loss curve is plotted). The features are:

* the average encoded residue, which for one-hot encoding is the amino acid composition (for biophysics encoding, the average of each biophysical scale);
* the log of the sequence length;
* with ``--features dipeptides``, also the dipeptide composition: the fraction of adjacent residue pairs that are each of the 400 possible dipeptides (for other encodings, the average product of adjacent residues' encodings). This adds 400 features for one-hot encoding, so it needs a reasonably large dataset to avoid overfitting. The number of features grows with the square of the encoding's size, so dipeptides need an encoding of at most 32 values per residue: not a protein language model, whose embeddings have hundreds, nor NARDINI features (90 values). With a wider encoding ``--features dipeptides`` is an error, and ``parrot-optimize`` does not search ``--features``.

Features are standardized before fitting, and so are regression targets (predictions are converted back to the original units). For classification, the models predict class probabilities, so ``--probabilistic-classification`` works as usual.

Every simple model takes ``--features`` (``composition`` (default) or ``dipeptides``; both are searched), plus:

**linear**: ridge regression (regression) or logistic regression (classification).

	*  ``--alpha`` : strength of the L2 penalty on the coefficients; must be greater than 0 (default 1.0; searched over 0.001-1000 on a log scale).

**svm**: support vector machine (scikit-learn's SVR for regression, SVC for classification).

	*  ``--svm-c`` : penalty on training errors (scikit-learn's ``C``); larger values fit the training data more closely; must be greater than 0 (default 1.0; searched over 0.01-1000 on a log scale).
	*  ``--kernel`` : ``rbf`` (default), ``linear`` or ``poly`` (``rbf`` and ``linear`` are searched).
	*  ``--relative-gamma`` : kernel coefficient, as a multiple of scikit-learn's default (about 1 / number of features); larger values give more local, wigglier fits; not used by the linear kernel (default 1.0; searched over 0.01-100 on a log scale).

**random_forest**: random forest of decision trees.

	*  ``--num-trees`` : number of trees (default 200; searched over 50-500).
	*  ``--min-samples-leaf`` : minimum number of training sequences in each leaf; larger values give smoother predictions (default 1; searched over 1-20).

**knn**: k-nearest neighbours; predicts from the most similar training sequences in feature space.

	*  ``--num-neighbors`` : number of neighbours used for each prediction; must not exceed the number of training sequences (default 5; searched over 1-50).
	*  ``--neighbor-weights`` : ``uniform`` (default; each neighbour counts equally) or ``distance`` (closer neighbours count more); both are searched.

Saved simple models contain the fitted scikit-learn model as a Python pickle. Loading a pickle can run arbitrary code, so only load saved models from sources you trust.
