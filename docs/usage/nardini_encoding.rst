NARDINI features
================

NARDINI asks whether the residues of an intrinsically disordered region are arranged non-randomly (Cohan et al. 2022). It groups the amino acids into eight types (polar, hydrophobic, positive, negative, aromatic, alanine, proline and glycine) and, for each type and each pair of types, compares how segregated they are along the sequence with scrambles of that same sequence, giving 36 patterning z-scores. NARDINI+ (King et al. 2024) adds 54 compositional z-scores, which compare what the sequence is made of with the human IDRome (Ruff et al. 2025). PARROT can use these 90 NARDINI+ features, as computed by SPARROW, as an encoding for sequence-mapped data.

The features need SPARROW, and the cache they are stored in (see below) needs ``h5py``:

.. code-block:: bash

    pip install git+https://github.com/idptools/sparrow.git
    pip install idptools-parrot[nardini]

NARDINI+ features describe a whole sequence, while PARROT's encodings give one vector per residue, so every residue is given its sequence's 90 features. There are two ways to use them:

* ``--encode nardini``: the features alone. With the ``composition`` baseline or the simple models (``linear``, ``svm``, ``random_forest``, ``knn``), which average over the residues of a sequence, this gives exactly a model of the NARDINI+ features and the sequence length. The order-aware networks (``brnn``, ``cnn``, ``unet``, ``transformer``) would see the same input at every residue, so use the combined form with them.
* ``--encode onehot+nardini`` (or ``biophysics+nardini``, or ``<encoding file>+nardini``): each residue's own encoding, followed by its sequence's 90 features, so a network sees which residue is where as well as the sequence's NARDINI+ features. ``--standardize-encoding`` standardizes the per-residue part. Protein language model embeddings cannot be combined with NARDINI features.

.. code-block:: bash

    parrot-train svm --input-file data.tsv --output-network net.pt --split kmer --regression --encode nardini
    parrot-train composition --input-file data.tsv --output-network net.pt --split kmer -d sequence --regression --encode nardini
    parrot-train brnn --input-file data.tsv --output-network net.pt --split kmer -d sequence --regression --encode onehot+nardini
    parrot-predict sequences.txt net.pt predictions.txt

NARDINI features are for sequence-mapped data (``-d sequence``) only; PARROT stops with an error if they are used for residue or matrix data.

The features
------------

* **36 patterning z-scores**, one for each of the eight residue types and each pair of them. Each compares the sequence's own arrangement with scrambles of itself, so it measures arrangement only: a positive value means the residues of those types are blockier than the sequence's composition alone would give, and a negative value means they are more evenly interspersed. A type that makes up less than 10% of the sequence is not scored, and its features are 0.
* **54 compositional z-scores**, describing what the sequence is made of (the fraction of each amino acid and of groups of them, charge, hydrophobicity, patches of the same residue and so on), each relative to the human IDRome. These are large for sequences unlike IDRs (around 100 for random sequences of all 20 amino acids). The simple models standardize their input features, so this does not matter for them; the neural networks see the z-scores as they are.

The features are computed with SPARROW's defaults for NARDINI+ (the eight residue types, a window of 5 residues, the 10% threshold and the published NARDINI patterning statistic), which PARROT fixes so that they stay the same if SPARROW's defaults change.

Scrambles, and reproducibility
------------------------------

The patterning z-scores compare each sequence with scrambles of itself. PARROT uses 1,000 scrambles per sequence by default, about ten times fewer than SPARROW's default of 10,000 (the NARDINI paper used 100,000). SPARROW notes that 1,000 is usually adequate for low-complexity sequences; more scrambles make the z-scores less noisy, at a proportional cost. Set the number with ``--nardini-scrambles`` (e.g. ``--nardini-scrambles 10000``).

The scrambles of each sequence are generated from a fixed seed and the sequence itself, so a sequence always gets exactly the same features: whichever other sequences it is computed with, in training and when predicting. The number of scrambles, the seed, the other settings, the feature names and the SPARROW version are saved with the network, and ``parrot-predict`` and ``Predictor`` compute the features of new sequences in exactly the same way. A network whose features were computed with a different SPARROW version still loads, with a warning, and one whose feature names differ from the installed SPARROW's cannot encode new sequences.

Features are computed once and cached
-------------------------------------

Computing the patterning z-scores takes about 35-100 ms per sequence at 1,000 scrambles. So, as for protein language models, ``parrot-train`` and ``parrot-optimize`` compute the features of every sequence before training starts, in parallel on every CPU core but one, and store them in the embedding cache: a ``parrot_train_embeddings`` directory next to the output network, or the directory given with ``--embedding-cache``. The features are stored in a file named after their settings (e.g. ``nardini_human_1000scrambles_seed0_sparrow0.2.2.h5``), as 32-bit floats; a different number of scrambles, or another version of SPARROW, gets a file of its own, so features are never reused from a SPARROW that might compute them differently. Point several runs at the same ``--embedding-cache`` to compute each sequence's features only once; a second run on the same data computes nothing. Deleting the cache is always safe.

``parrot-predict`` computes the features of the sequences to predict before predicting, storing them in a ``parrot_predict_embeddings`` directory (or ``--embedding-cache``). ``Predictor`` computes them as each sequence is predicted, or stores them in ``embedding_cache`` if given.

Sequences NARDINI cannot score
------------------------------

NARDINI needs sequences of the 20 standard amino acids that are at least 5 residues long (the window size). PARROT checks every sequence before computing anything and stops with an error naming the problem; for a datafile, ``parrot-preprocess --remove-nonstandard --min-length 5`` removes such sequences.

References
----------

* Cohan, M. C., Shinn, M. K., Lalmansingh, J. M. & Pappu, R. V. Uncovering non-random binary patterns within sequences of intrinsically disordered proteins. *J. Mol. Biol.* 434, 167373 (2022). [NARDINI]
* King, M. R. et al. Macromolecular condensation organizes nucleolar sub-phases and sets up a pH gradient. *Cell* 187, 1889-1906 (2024). [NARDINI+]
* Ruff, K. M., King, M. R. et al. Molecular grammars of predicted intrinsically disordered regions that span the human proteome. *Cell* (2025). [the human IDRome the compositional z-scores are relative to]
