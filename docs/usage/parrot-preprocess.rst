===================
 parrot-preprocess
===================

``parrot-preprocess`` turns a PARROT datafile into one ready for training. It can clean sequences, recode class labels, deal with duplicate sequences, split the data into training, validation and test sets **without leakage between them**, and balance the classes of the training set. It writes the processed datafile, a matching split-file (if it split the data) to train on with ``parrot-train --split``, and a report of everything it did. It can also compute a protein language model or NARDINI encoding of every sequence of the output ahead of time, so that training starts straight away.

Once PARROT is installed, the user can run ``parrot-preprocess`` from the command line:

.. code-block:: bash

    $ parrot-preprocess data_file output_file -d <datatype> -m <mode> <flags>

Where *data_file* is a standard PARROT datafile and *output_file* is where the processed datafile is written (an existing file is overwritten). Every other output file is named after *output_file*, without its final extension, in the same directory, which must already exist (``clean.tsv`` gives ``clean_split_file.txt`` and ``clean_preprocess_report.txt``). A typical run, which merges duplicate sequences, splits the data so that no set contains a homologue of another set (``--split kmer`` if MMseqs2 is not installed), and balances the classes of the training set, is:

.. code-block:: bash

    $ parrot-preprocess data.tsv clean.tsv -d sequence -m classification \
        --deduplicate merge --split homology --class-balance --seed 1
    $ parrot-train brnn --input-file clean.tsv --output-network network.pt -d sequence --classification --num-classes 3 --split clean_split_file.txt

Steps are always carried out in the order below, and each is optional.

1. **Cleaning** (``--uppercase``, ``--remove-nonstandard``, ``--min-length``, ``--max-length``)
2. **Recoding class labels** (``--recode-classes``)
3. **Deduplication** (``--deduplicate``, ``--remove-line-duplicates``, ``--remove-sequence-duplicates``)
4. **Splitting** into training, validation and test sets (``--split``)
5. **Class balancing** (``--class-balance``): of the training set only, if the data were split
6. **Encoding ahead of time** (``--encode``): after the output is written

**Required flags:**

    *  ``--datatype`` / ``-d`` : 'sequence' (one value per sequence), 'residues' (one value per residue) or 'matrix' (a matrix of values per sequence, held in a separate ``.npz`` file; see below).
    *  ``--mode`` / ``-m`` : 'classification' or 'regression'.

**General flags:**

    *  ``--help`` / ``-h`` : Display a help message.
    *  ``--show-examples`` : Print a longer guide to ``parrot-preprocess``: examples of how to use it, what it does and how it works.
    *  ``--delimiter`` : The character separating columns, if not whitespace. Spaces around each field are ignored. The output is a whitespace-separated PARROT datafile, so no field may contain whitespace (an ID such as ``sp|P1|A protein`` is refused, with a message saying which).
    *  ``--uncertainties`` : Sequence and residue regression only (not matrix data). Every value is written as ``value,SD``, where SD is the standard deviation of its measurement error (a missing residue value is still ``nan``), as for ``parrot-train --uncertainties``; see the "Values with uncertainties" page. The values and SDs are passed through unchanged, except by ``--deduplicate merge`` (below), and the output is checked as data with uncertainties. It cannot be combined with ``--delimiter ,``.
    *  ``--seed`` : Integer random seed, which makes the split and class balancing reproducible. By default no seed is set.
    *  ``--verbose`` / ``-v`` : Print the report as it is written (otherwise only the names of the files written are printed).

Options that only apply to some steps are refused, before any work is done, if that step is not run: for example ``--recode-classes`` and ``--class-balance`` without ``-m classification``, ``--groups`` or ``--set-fractions`` without ``--split``, or ``--kmer-size`` with ``--split homology``. Mistakes in the input stop the command with a one-line message (see "When something goes wrong" on the "Getting Started" page).

Cleaning sequences
------------------

    *  ``--uppercase`` : Convert every sequence to upper case.
    *  ``--remove-nonstandard`` : Remove sequences containing anything other than the 20 standard amino acids.
    *  ``--min-length`` / ``--max-length`` : Remove sequences shorter or longer than these (in residues; at least 1).

These are applied in the order listed, so ``--uppercase --remove-nonstandard`` keeps sequences that were written in lower case. The report says how many sequences each removed (and, for ``--remove-nonstandard``, which characters were found). If cleaning removes every line, nothing is written.

Recoding class labels
---------------------

    *  ``--recode-classes`` : Classification only. Recode class labels of any kind (e.g. 'helix', 'coil' or 5, 10, 20) as PARROT's 0, 1, 2, ... If every label is a number, the classes keep their numeric order, and labels that are the same number (``1`` and ``1.0``) become one class; otherwise the labels are numbered in alphabetical order. Missing labels (``nan``) are kept (only residue data can be trained on with missing labels). The mapping is written to ``<output>_class_mapping.txt``, one ``<label> = <class>`` line per label.

If the class labels are not already 0, 1, 2, ... and ``--recode-classes`` is not given, the check at the end of the report warns that PARROT cannot train on the output and suggests ``--recode-classes``.

Duplicate sequences
-------------------

Identical sequences in different sets are the most blatant form of leakage, and contradictory copies confuse training. ``--deduplicate`` keeps one line per unique sequence (named after the first copy, or, when ``merge`` keeps a sequence's class label, the first copy with a label):

    *  ``merge`` : For regression, the copies' values are averaged (residue by residue for residue data, ignoring ``nan``). With ``--uncertainties``, the average is weighted by 1/SD², and written with its own SD, 1/√Σ(1/SD²) (the standard combination of independent estimates); if a copy has an SD of 0, the merged value is the mean of the exact copies, with SD 0. Every labelled copy must have an SD that is a finite number of at least 0 (as ``parrot-train`` requires), or merging stops with a message naming the sequence. For classification, a sequence is kept once if all its labelled copies have the same label, and removed if they disagree (labels that are the same number, such as ``1`` and ``1.0``, agree, and missing labels, ``nan``, are ignored); for residue data, each residue keeps its label if every labelled copy agrees and otherwise becomes ``nan``, and a sequence left with no labelled residue is removed.
    *  ``first`` : Keep the first copy as it is.
    *  ``drop`` : Remove every copy, since there is no way to know which is right.

The line kept for each sequence takes the place of its first copy, so the lines stay in their original order, and the report says how many sequences were duplicated and what was done with them.

``--remove-sequence-duplicates`` is the same as ``--deduplicate drop`` (so the two cannot be given together). ``--remove-line-duplicates`` removes only lines that are exact copies of an earlier line (same ID, sequence and values), keeping the first copy of each and leaving every line in its original order; the report says how many sequences still appear more than once. It can be combined with ``--deduplicate`` (it runs first), but not with ``--remove-sequence-duplicates``.

Even without ``--deduplicate``, every ``--split`` method always puts identical sequences in the same set.

Splitting into training, validation and test sets
-------------------------------------------------

A test set only measures how well a network generalises if none of its sequences has a close relative in the training set. Otherwise a network can score well by recognising relatives rather than by having learnt anything transferable (see the FAQ "How should I set up my dataset?"). A random split puts homologous sequences on both sides routinely: in a random 70/15/15 split of the yeast proteome, 42% of the test proteins have a homologue in the training set.

    *  ``--split homology`` : Keep homologous sequences in the same set, using `MMseqs2 <https://github.com/soedinglab/MMseqs2>`_ (see below). The most thorough option; MMseqs2 must be installed.
    *  ``--split kmer`` : Keep similar sequences in the same set, using PARROT's own k-mer search. Needs nothing extra, but only finds closer relatives (see below).
    *  ``--split random`` : Split at random, keeping only identical sequences (and ``--groups``) together.
    *  ``--set-fractions`` : Target proportions of the training, validation and test sets (default ``0.7 0.15 0.15``).
    *  ``--groups`` : A file of ``<sequence ID> <group>`` lines (blank lines and lines starting with ``#`` are ignored; records whose ID is not in the file are each a group of their own). Records in the same group are always put in the same set, for example overlapping tiles or fragments of one protein, or measurements of one protein under different conditions. If deduplication removed a copy that belonged to a group, the line that replaced it joins that group too.

The output datafile lists the training lines, then the validation lines, then the test lines, and ``<output>_split_file.txt`` gives the line numbers of each set. ``parrot-train`` and ``parrot-optimize`` can also split the data themselves by the same methods (``--split mmseqs``, ``kmer`` or ``random``), but only with the default criteria and without ``--groups``.

**How the split works.** ``--split homology`` and ``--split kmer`` follow the same protocol and differ only in how similar sequences are found. The criteria are deliberately stringent, so that the test set measures generalisation to genuinely new sequences.

1. *Similarity criterion.* With ``--split homology``, two sequences are homologous if they are identical, or if any MMseqs2 alignment between them has an E-value of at most 10\ :sup:`-3` and covers at least 30% of the shorter sequence (the stretch of the shorter sequence between the alignment's ends, so gaps opposite an insertion in the longer sequence do not count). No sequence-identity threshold is used, so any statistically significant alignment counts, however low its identity. MMseqs2 searches at its most sensitive setting (``-s 7.5``) and considers up to 2,000 targets per query. ``--split kmer`` applies the same E-value and coverage thresholds to its own estimates (see below).
2. *Clustering.* An all-against-all search links similar sequences, which are then clustered around representatives (largest first, as in CD-HIT), so that a chain of weak links cannot merge unrelated families. Groups and identical sequences are always in the same cluster.
3. *Assignment.* Clusters are shared out among the sets, largest first, each to the set furthest below its target fraction.
4. *Separation check.* Each set is searched against the others (test against training, test against validation, validation against training). Any sequence in the lower-priority set (training < validation < test) that is similar to the higher-priority set is removed, together with its group. E-values depend on the size of the database searched, so this is repeated until a round finds nothing (at most five rounds; if the sets are still not separated, ``parrot-preprocess`` stops with an error), and the report then says ``PASSED``, followed by how many lines were removed and the final size of each set. Typically only a few percent of the data are removed.

The report also counts every weaker similarity between sets, up to E = 10\ :sup:`-2` (or up to ``--homology-evalue``, if that is larger), by sequence identity (estimated from the shared k-mers, for ``--split kmer``), so any remaining similarity is visible. With ``--split random``, the sets are not searched against each other, so similar sequences can end up in different sets.

Options for both ``--split homology`` and ``--split kmer``:

    *  ``--homology-evalue`` : Largest E-value of a similarity that links two sequences (default 1e-3; must be positive).
    *  ``--homology-coverage`` : Smallest fraction of the shorter sequence the similarity must cover (default 0.3; above 0 and at most 1).
    *  ``--no-masking`` : Do not mask low-complexity regions, so that similarity in them (common in disordered proteins) also links sequences. This is even more stringent; by default these regions are masked, so that similarity driven only by compositional bias or simple repeats does not count.

Options for ``--split homology`` only:

    *  ``--mmseqs-sensitivity`` : MMseqs2 sensitivity, 1 to 7.5 (default 7.5, the most sensitive).
    *  ``--mmseqs`` : The MMseqs2 executable, if ``mmseqs`` is not on the PATH.
    *  ``--threads`` : Threads for MMseqs2 (default: every CPU).

Options for ``--split kmer`` only:

    *  ``--kmer-size`` : Length of the shared words, 3 to 8 (default 4). 3 finds slightly more distant relatives but is about 20 times slower and more easily misled by chance matches; 5 is about 5 times faster and slightly less sensitive.

**Installing MMseqs2.** MMseqs2 is a separate program, needed only for ``--split homology``. Install it with ``conda install -c conda-forge -c bioconda mmseqs2`` or ``brew install mmseqs2``, or download a static binary from the `MMseqs2 releases <https://github.com/soedinglab/MMseqs2/releases>`_. If it cannot be found, ``parrot-preprocess`` stops with these instructions before doing any work; it never silently falls back to a weaker split. Splitting 1,300 sequences takes about 15 seconds on 20 CPU cores, and the 6,049-protein yeast proteome about a minute.

**The k-mer split (no MMseqs2).** ``--split kmer`` finds similar sequences with a search written in Python and NumPy, so it works wherever PARROT does. It works like the seeding step of BLAST or MMseqs2. Related sequences share short identical words (k-mers, 4 residues by default), and the shared words line up along one alignment diagonal; chance matches between unrelated sequences scatter over many diagonals. For each pair of sequences, shared k-mers are chained into runs along bands of nearby diagonals, which tolerates short insertions and deletions. Each pair is then scored like an MMseqs2 alignment:

    *  *E-value*, from the run whose shared k-mers cover the most residues. Overlapping shared k-mers are one longer stretch of shared sequence, not separate chances (a shared 6-residue stretch holds three 4-mers, but is no likelier to arise by chance than one 4-mer followed by two more identical residues), so a run is scored by the residues its shared k-mers cover. The chance of two residues being identical comes from the two sequences' own amino-acid compositions, so compositionally biased sequences (such as many disordered regions) need more shared sequence to count, and the chance of a run covering as many residues allows for every way chance stretches could make one up. As with MMseqs2, the E-value is the number of equally strong matches a query would be expected to find by chance in the database searched; on random sequences, matches reach a given E-value somewhat less often than it implies, never more.
    *  *Coverage*, from the stretch spanned by the best run and every other run near its diagonal that is itself unlikely by chance, as a fraction of the shorter sequence.

Masking ignores residues in 12-residue windows with less than 1.8 bits of sequence entropy, and k-mers of at most two distinct amino acids (for k = 4). Shared k-mers are counted as distinct positions, so a repeat cannot count one word many times.

Exact shared words cannot detect distant homologues, which share few identical stretches. Compared with the pairs MMseqs2 links in the yeast proteome, ``--split kmer`` links nearly all pairs at 70% identity or more (3,323 of 3,334), 96% at 50-70% and 76% at 40-50%, but only 9% at 30-40%. When the yeast proteome is split each way and the sets are then searched against each other with MMseqs2:

======================  =====================================  ===================================================  ==============
Split                   Test proteins with a homologue         Test-training alignments at >= 90% / 50-90% /        Data removed
                        in the training set                    < 50% identity
======================  =====================================  ===================================================  ==============
``--split random``      42%                                    234 / 335 / 2,129                                    0%
``--split kmer``        26%                                    0 / 10 / 1,209                                       0.7%
``--split homology``    0%                                     0 / 0 / 0                                            4.5%
======================  =====================================  ===================================================  ==============

For the 46,067 disordered regions of the ALBATROSS dataset, the share of test sequences with an MMseqs2 homologue in the training set falls from 5.3% with a random split (137 alignments at >= 90% identity) to 2.3% with ``--split kmer`` (7 at >= 90%, between nearly identical regions, mostly of 14 to 46 residues, too short for the words they share to be significant), and to 0 with ``--split homology``.

So ``--split kmer`` removes nearly all leakage between close relatives, which is the strongest leakage, but not between distant homologues. Use ``--split homology`` whenever MMseqs2 is available. A k-mer split takes about 40 seconds for the yeast proteome and about two minutes for ALBATROSS (with MMseqs2, about a minute and about 25 seconds respectively). The k-mer search runs on one CPU core.

Class balancing
---------------

Classification only. ``--class-balance`` down-samples the more common classes. If the data are split, only the training set is balanced: the validation and test sets keep the data's natural class mix, so reported performance reflects real data. The imbalance is the count of the most common class divided by the count of the least common class (1.0 is perfectly balanced).

    *  ``--cb-balance-threshold`` : Stop once the imbalance is at most this (default 1.4, the largest imbalance PARROT's training commands accept without a warning; between 1 and 5).
    *  ``--cb-max-fraction-removed`` : The largest fraction of the (training) sequences that may be removed (default 0.25; strictly between 0 and 1).
    *  ``--cb-block-size`` : For residue data, how many randomly drawn sequences are considered for each removal (default 100, at least 2; larger makes better choices, more slowly). It does nothing for sequence data, and says so.
    *  ``--cb-shuffle-off`` : No longer has any effect, since balancing always keeps the input order; it is accepted (with a note) so that older commands keep working.

For **sequence** data, every class is capped at the same size. The cap is lowered until the threshold is reached, or as far as the budget allows. This removes the fewest sequences needed for the best balance possible. The least common class is never reduced, and sequences with a missing label are never removed.

For **residue** data, each sequence contains residues of several classes, so sequences are removed one at a time. Each removal is the one, among ``--cb-block-size`` randomly drawn sequences, that improves the balance most. If none of them improves it, every remaining sequence is checked before stopping. A removal that would make the balance worse, or remove a class entirely, is never made. Balancing stops when the threshold is reached, when the budget is used up, or when no single removal improves the balance, and the report says which. The imbalance before balancing and after each removal is written to ``<output>_class_balance_trace.csv``, one ``<step>, <imbalance>`` line each (step 1 is before any removal).

Balancing keeps the input order of the lines. The report gives the count of each class, and the imbalance, before and after.

Matrix data
-----------

For matrix data (``-d matrix``) the datafile holds ``<ID> <sequence>`` on each line, and the matrices are in a separate ``.npz`` file. Cleaning, deduplication (``first`` or ``drop``) and splitting work as for other data, and every sequence keeps its ID, so the original matrix file works with the processed datafile and split-file unchanged. Options that change values (``--recode-classes``, ``--class-balance`` and ``--deduplicate merge``) do not apply, since the values are in the matrix file.

    *  ``--matrix-file`` : The ``.npz`` matrix file. If given, the final check also checks that every sequence of the output has a matrix that fits it.
    *  ``--matrix-width`` : The number of columns of fixed-width matrices (leave it out for square maps); used by that check, so it needs ``--matrix-file``.

See the "Matrix data" page for the file formats.

Encoding ahead of time
----------------------

Protein language model embeddings and NARDINI features are slow to compute, so ``parrot-train``, ``parrot-optimize`` and ``parrot-predict`` compute them for every sequence before they start, and store them in an embedding cache (see "Protein language model encodings"). ``--encode`` does this in ``parrot-preprocess`` instead, for every sequence of the output, in the same way (the same model or features, with the same settings) and into the same kind of cache. Training then finds every sequence already there, so the encoding can be computed once (for example overnight, or on a machine with a GPU) and any number of networks trained on it afterwards.

    *  ``--encode`` : A protein language model (e.g. ``esm2_t33_650M_UR50D``) or, for sequence data, a NARDINI encoding (``nardini`` or ``<encoding>+nardini``, e.g. ``onehot+nardini``). Any other encoding is refused: it is quick to compute, so PARROT computes it as it is needed and never stores it.
    *  ``--embedding-cache`` : The cache directory (default: a ``parrot_train_embeddings`` directory next to the output datafile). Sequences already in it are not computed again.
    *  ``--nardini-scrambles`` : NARDINI encodings only. Scrambles of each sequence for its patterning z-scores (default 1,000). Give ``parrot-train`` the same number, since each number of scrambles has its own file in the cache.
    *  ``--device`` / ``--gpu-id`` : Protein language models only. The device to run the model on, as for ``parrot-train`` (NARDINI features are always computed on the CPU).
    *  ``--embedding-context`` / ``--embedding-context-map`` : Protein language models only. Embed each sequence within the full-length protein it was cut from: the FASTA file of full-length proteins, and a map with one ``<sequence ID> <full-length ID> [start]`` line per sequence (see "Embedding fragments within their full-length proteins" on the "Protein language model encodings" page). Every sequence of the output is placed in its protein before the output is written. The table of where each one is embedded is written as ``<output>_embedded_context.tsv``. Give ``parrot-train`` the same two options, so that it reads these embeddings.
    *  ``--embed-mean-subtraction`` : Protein language models only. Subtract each embedding's mean over its residues before it is stored (with ``--embedding-context``, the full-length protein's mean). These embeddings have their own files in the cache, so give ``parrot-train`` the same option, so that it reads them. See "Subtracting each embedding's mean" on the "Protein language model encodings" page.

These options are refused without ``--encode``. The sequences are checked before the output is written, so a sequence the encoding cannot read (for example one in lower case, which ``--uppercase`` fixes) stops the command with no datafile, split-file or report written (only the files of ``--recode-classes`` and residue ``--class-balance``, which are written as those steps run). Encoding starts once the datafile and split-file are written, so if it stops part of the way, none of the preprocessing is lost, and the cache keeps every encoding already written to it (encodings are written in groups as they are computed, so only the group in progress is lost). The report then gets a final section saying where the cache is and how to use it:

.. code-block:: bash

    $ parrot-preprocess data.tsv clean.tsv -d sequence -m regression --split homology --seed 1 \
        --encode esm2_t33_650M_UR50D --embedding-cache embeddings
    $ parrot-train cnn --input-file clean.tsv --output-network network.pt -d sequence --regression --split clean_split_file.txt \
        --encode esm2_t33_650M_UR50D --embedding-cache embeddings

``parrot-train`` and ``parrot-optimize`` need ``--embedding-cache`` to find the cache, unless the network is saved in the same directory as the output datafile (both default to a ``parrot_train_embeddings`` directory there). ``parrot-predict`` reads the same cache with ``--embedding-cache``, and only computes sequences that are not in it.

Output
------

    *  *output_file* : The processed datafile, with whitespace-separated columns (training, then validation, then test lines, if the data were split). Blank and comment lines of the input are not kept.
    *  ``<output>_split_file.txt`` : With ``--split``, the split-file to train on with ``parrot-train --split``. The datafile and split-file only match each other, so always use them together.
    *  ``<output>_preprocess_report.txt`` : What every step did, starting with the PARROT version, the time and the input file. Its last step is a check that PARROT reads the output (and that the split-file matches it); if it cannot, the report (and the terminal) says why, as a warning rather than an error. With ``--encode``, a final section then says where the encoding is stored and how to use it.
    *  ``<output>_class_mapping.txt`` : With ``--recode-classes``, the label mapping.
    *  ``<output>_class_balance_trace.csv`` : With ``--class-balance`` on residue data, the imbalance before balancing and after each removal.
    *  ``parrot_train_embeddings/`` (or ``--embedding-cache``) : With ``--encode``, the embedding cache, with a ``README.md`` describing what is in it.
    *  ``<output>_embedded_context.tsv`` : With ``--embedding-context``, where each sequence is embedded: its full-length protein and the residues it covers there (see "Protein language model encodings").
