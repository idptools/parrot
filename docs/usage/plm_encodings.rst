Protein language model encodings
================================

Instead of a fixed vector per amino acid (one-hot, biophysics or your own table), PARROT can describe each residue by its embedding from a pretrained protein language model (PLM). The PLM reads the whole sequence, so a residue's embedding reflects its context as well as its identity. The PLM itself is frozen: PARROT trains its own network on the embeddings, exactly as it would on any other encoding, and every architecture works with them.

PARROT supports two families of models:

* **ESM-2** (Lin et al. 2023, *Science* 379:1123), loaded through Hugging Face ``transformers``. Install with ``pip install idptools-parrot[plm]``.
* **ESM C** (EvolutionaryScale's "ESM Cambrian" models, released under the MIT licence), loaded through EvolutionaryScale's ``esm`` package. Install with ``pip install idptools-parrot[esmc]``.

Model weights are downloaded from Hugging Face the first time a model is used and kept in the Hugging Face cache (``~/.cache/huggingface``); they are not stored in PARROT's network files.

Sequences must be written in uppercase one-letter codes. Besides the 20 standard amino acids, the models accept ``X`` (unknown) and ``B``, ``Z``, ``U`` and ``O``. PARROT stops with an error, before embedding anything, if a sequence contains any other character: the models would read it (including any lowercase letter) as an unknown token, which silently gives wrong embeddings. (``parrot-preprocess --uppercase`` converts a datafile; ``Predictor`` converts each sequence to upper case itself.)

.. note::

    Meta's ``fair-esm`` package and EvolutionaryScale's ``esm`` package both install under the import name ``esm`` and overwrite each other's files. PARROT never uses ``fair-esm`` (it loads ESM-2 through ``transformers``), so do not install it in the same environment as ``esm``. If it is already installed, ``pip uninstall fair-esm`` and then reinstall ``esm`` (e.g. ``pip install --force-reinstall --no-deps esm``) to restore the files they share.

.. code-block:: bash

    parrot-train brnn --input-file data.tsv --output-network net.pt --split kmer -d residues --regression --encode esm2_t12_35M_UR50D
    parrot-predict sequences.txt net.pt predictions.txt

Choosing a model
----------------

========================  ===================  ======  ======================  ====================================
``--encode``              Values per residue   Layers  Embedding speed*        Cache size per million residues
========================  ===================  ======  ======================  ====================================
``esm2_t6_8M_UR50D``      320                  6       ~47,000 residues/s      ~0.64 GB
``esm2_t12_35M_UR50D``    480                  12      ~23,000 residues/s      ~0.96 GB
``esm2_t30_150M_UR50D``   640                  30      ~7,300 residues/s       ~1.3 GB
``esm2_t33_650M_UR50D``   1,280                33      ~2,200 residues/s       ~2.6 GB
``esm2_t36_3B_UR50D``     2,560                36      much slower             ~5.1 GB
``esm2_t48_15B_UR50D``    5,120                48      much slower             ~10 GB
``esmc_300m``             960                  30      ~5,400 residues/s       ~1.9 GB
``esmc_600m``             1,152                36      ~2,500 residues/s       ~2.3 GB
========================  ===================  ======  ======================  ====================================

\* Measured with PARROT on an Apple-silicon Mac GPU (MPS), for proteins of median length ~350 residues; an NVIDIA GPU is typically faster, and a CPU much slower.

Larger models usually give more informative embeddings but are slower to embed with, both when training and every time you predict. It is worth comparing a small and a medium model on your own task before committing to a large one.

Embedding is the slow step, so embeddings are cached
----------------------------------------------------

Running the PLM takes far longer than training or running a PARROT network on its output. PARROT therefore embeds every sequence once and stores the result in an embedding cache:

* ``parrot-train`` and ``parrot-optimize`` embed every sequence in the datafile (training, validation and test) before training starts. Every epoch then reads the cached embeddings, so training is as fast as with any other encoding of the same size. By default the cache is a directory called ``parrot_train_embeddings`` next to the output network.
* ``parrot-predict`` embeds the sequences to predict before predicting. By default its cache is a directory called ``parrot_predict_embeddings`` next to the output file.
* ``parrot-preprocess --encode <model>`` embeds every sequence of its output datafile ahead of time, so that later training runs given the same ``--embedding-cache`` start straight away (see "parrot-preprocess").
* ``Predictor`` (in Python) embeds each sequence when it is predicted and stores nothing, unless given ``embedding_cache=<directory>``. ``Predictor.predict_batch()`` embeds all its sequences first, as ``parrot-predict`` does, into ``embedding_cache`` or, without one, into a temporary cache that is deleted once they are predicted.

Before embedding, PARROT prints how many unique sequences there are, how many are already in the cache and how much the cache will grow, and it stops if the disk does not have room (it asks for 25% more free space than the estimate). While embedding, it shows a progress bar (on stderr) with the residues embedded, the speed and the time left. Residues where the windows of a long sequence overlap (see below) count once per window, since the model runs over them more than once. When the output goes to a file rather than a terminal (for example the log of a run started with ``nohup``), the bar is redrawn only every 30 seconds, so the log does not fill with updates. The cache size is about

    (number of residues) x (values per residue) x 2 bytes

because embeddings are stored as 16-bit floats. For example, 155,000 residues (the CheZOD dataset) take ~0.4 GB with ``esm2_t33_650M_UR50D``, and 5 million residues take ~5 GB with ``esm2_t12_35M_UR50D``.

Each cache directory contains a ``README.md`` describing what is in it (models, numbers of sequences and residues, size on disk) and how it is stored. Some practical points:

* **Reuse a cache** by passing ``--embedding-cache <directory>`` to ``parrot-train``, ``parrot-optimize`` or ``parrot-predict``. Sequences already in the cache are not embedded again, so training several networks on the same data with the same model embeds each sequence only once. Point all of them at the same directory.
* **Deleting a cache is safe**: nothing in it is needed to use a trained network, and embeddings are recomputed when next needed.
* **Keep caches out of synced folders** (e.g. Dropbox or iCloud) if they are large, using ``--embedding-cache``.
* **Several commands can share a cache at the same time** (e.g. training runs started together on the same data). Only one writes to it at a time: a ``<model>.h5.lock`` file marks a cache being written, and another command that needs to write waits for it (for up to 10 minutes, after which it stops with an error). Reading never blocks a command that is writing. If a run that crashed leaves a lock file behind, delete it.
* Embeddings are added to the cache in groups as they are computed, so an interrupted run keeps most of its work.
* Caches are only created for PLM encodings and NARDINI features (see "NARDINI features"). ``--embedding-cache`` is an error with any other encoding.
* Fragments embedded within their full-length proteins (see below) are stored in a separate file of the cache.
* Mean-subtracted embeddings (``--embed-mean-subtraction``; see below) are also stored in files of their own.
* So are embeddings from a layer before the final one (``--embedding-layer``; see below).

How residues are embedded
-------------------------

* Each residue's embedding is the PLM's final-layer representation of that residue, after the model's last layer norm (the start and end tokens the model adds are removed). ``--embedding-layer`` uses an earlier layer's output instead (see below).
* ESM-2 was trained on sequences of up to 1,022 residues, and ESM C on up to 2,046. Longer sequences are embedded in overlapping windows of that length (consecutive windows share at least 256 residues, and the last window ends at the end of the sequence), and residues covered by more than one window get the average of their embeddings.
* Models always run in 32-bit precision. (EvolutionaryScale's own loader switches ESM C to 16-bit "bfloat16" on a GPU, which would make embeddings depend on the device.)
* Embeddings are rounded to 16-bit floats, whether or not they go through a cache, so a sequence is encoded at the same precision either way. (The model's own output can differ by float32 rounding with the other sequences it is run with, since padding a batch changes the order of the floating-point operations, and this can move a stored value by one step of the 16-bit rounding.)
* The trained network file records the model, the layer, the windowing, the storage precision and whether the embeddings were made in context or mean-subtracted (see below), so ``parrot-predict`` and ``Predictor`` embed new sequences in the same way as the training sequences.
* The PLM runs on the same device as training or prediction (see ``--device``). Use a GPU if you can: on a CPU, embedding is many times slower.

Embedding fragments within their full-length proteins
-----------------------------------------------------

A PLM embeds each residue in the context of the whole sequence it is given. So the residues of a peptide embedded on its own get different embeddings from the same residues embedded within the protein they come from. Many datasets are libraries of fragments cut out of larger proteins, for example peptides tiled across a proteome. For these, ``--embedding-context`` embeds each full-length protein instead, and cuts each fragment's residues out of its embedding. Whether this helps depends on what was measured: it makes most sense when the measured property belongs to the region within its native protein. If the fragments were measured on their own (for example, each peptide fused to a reporter), embedding them on their own may match the experiment better. It is worth training both ways and comparing.

Which protein each fragment comes from is given explicitly, never guessed, with two files:

* **A FASTA file** (``--embedding-context``) of the full-length proteins. Each protein is named by the first word of its header line (for a UniProt entry, ``sp|P04637|P53_HUMAN``, not ``P04637``). It can be a whole proteome: only the proteins named in the map are kept in memory. Sequences are read as they are, so they must be upper case, with no gaps or ``*`` marking a stop codon.
* **A map** (``--embedding-context-map``) with one line per sequence of the datafile: ``<sequence ID> <full-length ID> [start]``, separated by whitespace. ``start`` is the 1-based position of the fragment's first residue in the full-length protein. It is only needed if the fragment occurs more than once in that protein; otherwise, the fragment is found by searching the protein. A full-length ID of ``-`` says the fragment has no full-length protein, and it is embedded on its own. Blank lines and lines starting with ``#`` (e.g. a header) are skipped.

For example::

    # fragment          protein          start
    P04637_tile0001     sp|P04637|P53_HUMAN  1
    P04637_tile0002     sp|P04637|P53_HUMAN
    designed_peptide_7  -

PARROT checks every sequence before embedding anything, and stops with an error listing the first few problems if any sequence:

* has no line in the map;
* names a protein that is not in the FASTA file;
* does not match its protein at the start given;
* without a start, does not occur in its protein, or occurs more than once.

Only exact matches count, so a fragment carrying a mutation will not be found in the wild-type protein. Embeddings are looked up by sequence, so a sequence must have the same context under every ID it has. Two IDs of identical proteins count as the same context. If the same sequence comes from different proteins under different IDs, PARROT stops; ``parrot-preprocess --deduplicate`` merges or drops repeated sequences. Lines of the map for IDs that are not in the datafile are ignored and counted.

Every command that uses ``--embedding-context`` writes a table of where each sequence was embedded, ``<output>_embedded_context.tsv``, next to its output: the network for ``parrot-train`` and ``parrot-optimize``, the output datafile for ``parrot-preprocess``, and the predictions for ``parrot-predict``. It has a header line, then one tab-separated line per sequence with these columns:

* the sequence's ID and sequence;
* the full-length protein's ID (``-`` for a sequence embedded on its own);
* the residues of the full-length protein it covers (1-based and inclusive, e.g. ``31-60``, or ``-``);
* a note on how it was placed (start given in the map, start found by searching, or embedded on its own).

.. code-block:: bash

    parrot-preprocess tiles.tsv clean.tsv -d sequence -m regression --split kmer --encode esm2_t33_650M_UR50D \
        --embedding-cache embeddings --embedding-context proteome.fasta --embedding-context-map tile_map.tsv
    parrot-train cnn --input-file clean.tsv --output-network net.pt --split clean_split_file.txt -d sequence \
        --regression --encode esm2_t33_650M_UR50D --embedding-cache embeddings \
        --embedding-context proteome.fasta --embedding-context-map tile_map.tsv

Some practical points:

* **Cost.** Each full-length protein is embedded once, however many of its fragments there are, so the time depends on the residues of the proteins rather than of the fragments. For overlapping tiles this can be less than embedding the tiles themselves. The cache stores only the fragments' embeddings, so its size is as before.
* **The cache.** In-context embeddings are stored in their own file of the cache, ``<model>.context.h5``, so they are never mistaken for embeddings of sequences on their own (which stay in ``<model>.h5``). Each is stored under its fragment, its full-length protein and its start. So adding fragments, editing the map or changing the FASTA file never makes what is stored wrong; only new combinations are embedded.
* **Training and predicting.** A network trained with ``--embedding-context`` records this. ``parrot-predict`` then needs ``--embedding-context`` and ``--embedding-context-map`` too, matching the sequences to predict to the map by their IDs (so not with ``--exclude-seq-id``). Without them it stops, unless given ``--embed-in-isolation``, which embeds each sequence on its own (for example, new peptides with no full-length protein). Those embeddings differ from the ones the network was trained on, so predictions may be less accurate. Both options are errors for a network trained without ``--embedding-context``.
* **In Python**, ``Predictor.predict(fragment, full_length=protein)`` embeds the fragment within ``protein``, and ``start=`` gives its 1-based start if it occurs more than once. ``Predictor.predict_batch(fragments, full_lengths=proteins)`` does the same for a list of fragments (one protein per fragment, and ``starts=`` if needed), embedding each protein once. ``Predictor(path, embed_in_isolation=True)`` embeds a sequence predicted without ``full_length`` on its own instead of raising an error.
* **Splitting.** Embedded in context, a fragment's embedding carries information from the rest of its protein, including from neighbouring fragments. Keep all the fragments of a protein in the same set (e.g. with a split-file), or similar fragments in the training and test sets will make the test results look better than they are.

Choosing the layer
------------------

By default, each residue's embedding is the model's final-layer representation of it. ``--embedding-layer N`` uses the output of layer N instead, counting from 1 (the first transformer layer) to the model's number of layers (see the table above). Negative numbers count back from the final layer, so ``-1`` is the final layer and ``-2`` the one before it. The final layer is the one closest to the model's training task, predicting masked amino acids, and for some property-prediction tasks an earlier layer gives better features, so the layer is worth comparing on your own task.

.. code-block:: bash

    parrot-train cnn --input-file clean.tsv --output-network net.pt --split clean_split_file.txt -d residues \
        --regression --encode esm2_t33_650M_UR50D --embedding-cache embeddings --embedding-layer 20

* **Only the layers needed are run.** The model stops after layer N, so an earlier layer is also quicker to embed with.
* **No final layer norm.** The model's final layer norm follows only its final layer, so an earlier layer's output is used as the model computes it. Its values are on a different scale from the final layer's, and in places much larger (over a thousand in ESM C's later layers). Networks see them as they are (``--standardize-encoding`` cannot be used with PLM encodings). PARROT stops with an error if a value is too large to store as a 16-bit float (above 65,504).
* **The cache.** Embeddings from layer N are stored in files of their own, ``<model>.layerN.h5`` (with ``.mean_subtracted`` and ``.context`` added as for the final layer), so they never mix with other layers' embeddings. Giving the final layer (its number, or ``-1``) is the same as leaving the option out, and uses the usual files. To embed ahead of time, give ``parrot-preprocess --encode`` the same option.
* **Training and predicting.** The network records its layer, so ``parrot-predict`` and ``Predictor`` embed new sequences from the same layer automatically.
* ``--embedding-layer`` combines with ``--embedding-context`` and ``--embed-mean-subtraction``.

Subtracting each embedding's mean
---------------------------------

Part of each residue's PLM embedding is shared by every residue of the sequence it was given, and so describes the sequence as a whole rather than the residue. ``--embed-mean-subtraction`` removes that shared part. Every time a sequence is embedded, the mean of its embedding over its residues (its mean pool) is subtracted from every residue's embedding before the embedding is stored, so what is left is how each residue differs from the sequence as a whole.

With ``--embedding-context``, the sequence embedded is the full-length protein. So the mean subtracted is the protein's, and each fragment is cut out afterwards. A fragment's embedding then describes how its residues differ from the rest of their protein, and keeps what distinguishes the fragment within it. A fragment mapped to ``-`` is embedded on its own, so its own mean is subtracted.

.. code-block:: bash

    parrot-train cnn --input-file clean.tsv --output-network net.pt --split clean_split_file.txt -d sequence \
        --regression --encode esm2_t33_650M_UR50D --embedding-cache embeddings --embed-mean-subtraction \
        --embedding-context proteome.fasta --embedding-context-map tile_map.tsv

* **The cache.** Mean-subtracted embeddings are stored in files of their own, ``<model>.mean_subtracted.h5`` (and ``<model>.mean_subtracted.context.h5`` in context), so they never mix with the embeddings the model gives. Using it for the first time means embedding every sequence again. To embed ahead of time, give ``parrot-preprocess --encode`` the same option.
* **Training and predicting.** The network records that its embeddings were mean-subtracted, so ``parrot-predict`` and ``Predictor`` subtract the mean from new sequences' embeddings automatically.
* **The simple models** (``linear``, ``svm``, ``random_forest``, ``knn``, ``gbrt``) use each sequence's mean embedding as their features. Mean subtraction makes that zero for a sequence embedded on its own, so these models only accept ``--embed-mean-subtraction`` with ``--embedding-context``. Even then, any sequence mapped to ``-`` has a mean embedding of zero.
* The mean is subtracted at full precision, before rounding to 16-bit floats, so the stored embedding of a sequence embedded on its own averages to zero only to within that rounding.

Things to know
--------------

* ``--standardize-encoding`` cannot be used with PLM encodings: an embedding depends on the residue's context, so there is no per-amino-acid table to standardize.
* The simple models (``linear``, ``svm``, ``random_forest``, ``knn``, ``gbrt``) use the average embedding over the sequence (plus the log of its length) as their features: the standard "mean-pooled embedding" baseline. Their ``dipeptides`` feature set is not available (it needs an encoding of at most 32 values per residue), because it would need over a hundred thousand features; ``parrot-optimize`` does not search it.
* Protein language model embeddings cannot be combined with NARDINI features.
* Prediction with a PLM network is much slower than with a one-hot network, because every new sequence has to be embedded. For large-scale prediction, choose the smallest model that is accurate enough, use a GPU, and reuse caches when predicting the same sequences again.
* The PLM is never trained (fine-tuned). Fine-tuning would make every epoch as slow as embedding the whole dataset, and is not supported.
