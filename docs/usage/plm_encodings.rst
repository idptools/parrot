Protein language model encodings
================================

Instead of a fixed vector per amino acid (one-hot, biophysics or your own table), PARROT can describe each residue by its embedding from a pretrained protein language model (PLM). The PLM reads the whole sequence, so a residue's embedding reflects its context as well as its identity. The PLM itself is frozen: PARROT trains its own network on the embeddings, exactly as it would on any other encoding, and every architecture works with them.

PARROT supports two families of models:

* **ESM-2** (Lin et al. 2023, *Science* 379:1123), loaded through Hugging Face ``transformers``. Install with ``pip install idptools-parrot[plm]``.
* **ESM C** (EvolutionaryScale's "ESM Cambrian" models, released under the MIT licence), loaded through EvolutionaryScale's ``esm`` package. Install with ``pip install idptools-parrot[esmc]``.

Model weights are downloaded from Hugging Face the first time a model is used and kept in the Hugging Face cache (``~/.cache/huggingface``); they are not stored in PARROT's network files.

Sequences must be written in uppercase one-letter codes. Besides the 20 standard amino acids, the models accept ``X`` (unknown) and ``B``, ``Z``, ``U`` and ``O``. PARROT stops with an error, before embedding anything, if a sequence contains any other character: the models would read it (including any lowercase letter) as an unknown token, which silently gives wrong embeddings.

.. note::

    Meta's ``fair-esm`` package and EvolutionaryScale's ``esm`` package both install under the import name ``esm`` and overwrite each other's files. PARROT never uses ``fair-esm`` (it loads ESM-2 through ``transformers``), so do not install it in the same environment as ``esm``. If it is already installed, ``pip uninstall fair-esm`` and then reinstall ``esm`` (e.g. ``pip install --force-reinstall --no-deps esm``) to restore the files they share.

.. code-block:: bash

    parrot-train data.tsv net.pt -d residues -c 1 --encode esm2_t12_35M_UR50D
    parrot-predict sequences.txt net.pt predictions.txt

Choosing a model
----------------

========================  ===================  ======================  ====================================
``--encode``              Values per residue   Embedding speed*        Cache size per million residues
========================  ===================  ======================  ====================================
``esm2_t6_8M_UR50D``      320                  ~47,000 residues/s      ~0.64 GB
``esm2_t12_35M_UR50D``    480                  ~23,000 residues/s      ~0.96 GB
``esm2_t30_150M_UR50D``   640                  ~7,300 residues/s       ~1.3 GB
``esm2_t33_650M_UR50D``   1,280                ~2,200 residues/s       ~2.6 GB
``esm2_t36_3B_UR50D``     2,560                much slower             ~5.1 GB
``esm2_t48_15B_UR50D``    5,120                much slower             ~10 GB
``esmc_300m``             960                  ~5,400 residues/s       ~1.9 GB
``esmc_600m``             1,152                ~2,500 residues/s       ~2.3 GB
========================  ===================  ======================  ====================================

\* Measured with PARROT on an Apple-silicon Mac GPU (MPS), for proteins of median length ~350 residues; an NVIDIA GPU is typically faster, and a CPU much slower.

Larger models usually give more informative embeddings but are slower to embed with, both when training and every time you predict. It is worth comparing a small and a medium model on your own task before committing to a large one.

Embedding is the slow step, so embeddings are cached
----------------------------------------------------

Running the PLM takes far longer than training or running a PARROT network on its output. PARROT therefore embeds every sequence once and stores the result in an embedding cache:

* ``parrot-train`` and ``parrot-optimize`` embed every sequence in the datafile (training, validation and test) before training starts. Every epoch then reads the cached embeddings, so training is as fast as with any other encoding of the same size. By default the cache is a directory called ``parrot_train_embeddings`` next to the output network.
* ``parrot-predict`` embeds the sequences to predict before predicting. By default its cache is a directory called ``parrot_predict_embeddings`` next to the output file.
* ``Predictor`` (in Python) embeds each sequence when it is predicted and stores nothing, unless given ``embedding_cache=<directory>``.

Before embedding, PARROT prints how many sequences need embedding and how much the cache will grow, and it stops if the disk does not have room. The cache size is about

    (number of residues) x (values per residue) x 2 bytes

because embeddings are stored as 16-bit floats. For example, 155,000 residues (the CheZOD dataset) take ~0.4 GB with ``esm2_t33_650M_UR50D``, and 5 million residues take ~5 GB with ``esm2_t12_35M_UR50D``.

Each cache directory contains a ``README.md`` describing what is in it (models, numbers of sequences and residues, size on disk) and how it is stored. Some practical points:

* **Reuse a cache** by passing ``--embedding-cache <directory>`` to ``parrot-train``, ``parrot-optimize`` or ``parrot-predict``. Sequences already in the cache are not embedded again, so training several networks on the same data with the same model embeds each sequence only once. Point all of them at the same directory.
* **Deleting a cache is safe**: nothing in it is needed to use a trained network, and embeddings are recomputed when next needed.
* **Keep caches out of synced folders** (e.g. Dropbox or iCloud) if they are large, using ``--embedding-cache``.
* Only one PARROT command may write to a cache at a time; a ``<model>.h5.lock`` file marks a cache being written.
* Caches are only created for PLM encodings. ``--embedding-cache`` is an error with any other encoding.

How residues are embedded
-------------------------

* Each residue's embedding is the PLM's final-layer representation of that residue, after the model's last layer norm (the start and end tokens the model adds are removed).
* ESM-2 was trained on sequences of up to 1,022 residues, and ESM C on up to 2,046. Longer sequences are embedded in overlapping windows of that length (consecutive windows share 256 residues), and residues covered by more than one window get the average of their embeddings.
* Models always run in 32-bit precision. (EvolutionaryScale's own loader switches ESM C to 16-bit "bfloat16" on a GPU, which would make embeddings depend on the device.)
* Embeddings are rounded to 16-bit floats, whether or not they go through a cache, so a sequence is always encoded the same way.
* The trained network file records the model, the windowing and the storage precision, so ``parrot-predict`` and ``Predictor`` embed new sequences exactly as the training sequences were.
* The PLM runs on the same device as training or prediction (see ``--device``). Use a GPU if you can: on a CPU, embedding is many times slower.

Things to know
--------------

* ``--standardize-encoding`` cannot be used with PLM encodings: an embedding depends on the residue's context, so there is no per-amino-acid table to standardize.
* The scikit-learn models (``linear``, ``svm``, ``random_forest``, ``knn``) use the average embedding over the sequence (plus the log of its length) as their features: the standard "mean-pooled embedding" baseline. Their ``dipeptides`` feature set is not available, because it would need hundreds of thousands of features.
* Prediction with a PLM network is much slower than with a one-hot network, because every new sequence has to be embedded. For large-scale prediction, choose the smallest model that is accurate enough, use a GPU, and reuse caches when predicting the same sequences again.
* The PLM is never trained (fine-tuned). Fine-tuning would make every epoch as slow as embedding the whole dataset, and is not supported.
