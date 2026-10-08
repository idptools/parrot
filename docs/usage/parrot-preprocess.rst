===================
 parrot-preprocess
===================

``parrot-preprocess`` turns a PARROT datafile into one ready for training. It can clean sequences, recode class labels, deal with duplicate sequences, split the data into training, validation and test sets **without leakage between them**, and balance the classes of the training set. It writes the processed datafile, a matching split-file to train on with ``parrot-train --split``, and a report of everything it did.

Once PARROT is installed, the user can run ``parrot-preprocess`` from the command line:

.. code-block:: bash

    $ parrot-preprocess data_file output_file -d <datatype> -m <mode> <flags>

Where *data_file* is a standard PARROT datafile and *output_file* is where the processed datafile is written. A typical run, which merges duplicate sequences, splits the data so that no set contains a homologue of another set (``--split kmer`` if MMseqs2 is not installed), and balances the classes of the training set, is:

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

**Required flags:**

    *  ``--datatype`` / ``-d`` : 'sequence' (one value per sequence), 'residues' (one value per residue) or 'matrix' (a matrix of values per sequence, held in a separate ``.npz`` file; see below).
    *  ``--mode`` / ``-m`` : 'classification' or 'regression'.

**General flags:**

    *  ``--help`` / ``-h`` : Display a help message.
    *  ``--show-examples`` : Print a longer guide to ``parrot-preprocess``: examples of how to use it, what it does and how it works.
    *  ``--delimiter`` : The character separating columns, if not whitespace.
    *  ``--seed`` : Integer random seed, which makes the split and class balancing reproducible. By default no seed is set.
    *  ``--verbose`` / ``-v`` : Print the report as it is written.

Cleaning sequences
------------------

    *  ``--uppercase`` : Convert every sequence to upper case.
    *  ``--remove-nonstandard`` : Remove sequences containing anything other than the 20 standard amino acids.
    *  ``--min-length`` / ``--max-length`` : Remove sequences shorter or longer than these.

Recoding class labels
---------------------

    *  ``--recode-classes`` : Recode class labels of any kind (e.g. 'helix', 'coil' or 5, 10, 20) as PARROT's 0, 1, 2, ... Numeric labels keep their numeric order, and labels that are the same number (``1`` and ``1.0``) become one class. Missing labels (``nan``) are kept. The mapping is written to ``<output>_class_mapping.txt``.

Duplicate sequences
-------------------

Identical sequences in different sets are the most blatant form of leakage, and contradictory copies confuse training. ``--deduplicate`` keeps one line per unique sequence (named after the first copy):

    *  ``merge`` : For regression, the copies' values are averaged (residue by residue for residue data, ignoring ``nan``). For classification, a sequence is kept once if all copies have the same label, and removed if they disagree; for residue data, each residue keeps its label if every labelled copy agrees and otherwise becomes ``nan``.
    *  ``first`` : Keep the first copy as it is.
    *  ``drop`` : Remove every copy, since there is no way to know which is right.

``--remove-sequence-duplicates`` is the same as ``--deduplicate drop``. ``--remove-line-duplicates`` removes only lines that are exact copies of an earlier line (same ID, sequence and values), keeping the first copy of each and leaving every line in its original order.

Even without ``--deduplicate``, every ``--split`` method always puts identical sequences in the same set.

Splitting into training, validation and test sets
-------------------------------------------------

A test set only measures how well a network generalises if none of its sequences has a close relative in the training set. Otherwise a network can score well by recognising relatives rather than by having learnt anything transferable (see the FAQ "What considerations should I take into account with my dataset?"). ``parrot-train`` splits data at random, so homologous sequences routinely end up on both sides: in a random 70/15/15 split of the yeast proteome, 45% of the test proteins have a homologue in the training set.

    *  ``--split homology`` : Keep homologous sequences in the same set, using `MMseqs2 <https://github.com/soedinglab/MMseqs2>`_ (see below). The most thorough option; MMseqs2 must be installed.
    *  ``--split kmer`` : Keep similar sequences in the same set, using PARROT's own k-mer search. Needs nothing extra, but only finds closer relatives (see below).
    *  ``--split random`` : Split at random, keeping only identical sequences (and ``--groups``) together.
    *  ``--set-fractions`` : Target proportions of the training, validation and test sets (default ``0.7 0.15 0.15``).
    *  ``--groups`` : A file of ``<sequence ID> <group>`` lines. Records in the same group are always put in the same set, for example overlapping tiles or fragments of one protein, or measurements of one protein under different conditions.

The output datafile lists the training lines, then the validation lines, then the test lines, and ``<output>_split_file.txt`` gives the line numbers of each set.

**How the split works.** ``--split homology`` and ``--split kmer`` follow the same protocol and differ only in how similar sequences are found. The criteria are deliberately stringent, so that the test set measures generalisation to genuinely new sequences.

1. *Similarity criterion.* With ``--split homology``, two sequences are homologous if they are identical, or if any MMseqs2 alignment between them has an E-value of at most 10\ :sup:`-3` and covers at least 30% of the shorter sequence. No sequence-identity threshold is used, so any statistically significant alignment counts, however low its identity. MMseqs2 searches at its most sensitive setting (``-s 7.5``) and considers up to 2,000 targets per query. ``--split kmer`` applies the same E-value and coverage thresholds to its own estimates (see below).
2. *Clustering.* An all-against-all search links similar sequences, which are then clustered around representatives (largest first, as in CD-HIT), so that a chain of weak links cannot merge unrelated families. Groups and identical sequences are always in the same cluster.
3. *Assignment.* Clusters are shared out among the sets, largest first, each to the set furthest below its target fraction.
4. *Separation check.* Each set is searched against the others (test against training, test against validation, validation against training). Any sequence in the lower-priority set (training < validation < test) that is similar to the higher-priority set is removed, together with its group. E-values depend on the size of the database searched, so this is repeated until a round finds nothing, and the report ends with ``PASSED``. Typically only a few percent of the data are removed.

The report also lists every weaker similarity between sets, up to E = 10\ :sup:`-2`, grouped by sequence identity, so any remaining similarity is visible.

Options for both ``--split homology`` and ``--split kmer``:

    *  ``--homology-evalue`` : Largest E-value of a similarity that links two sequences (default 1e-3).
    *  ``--homology-coverage`` : Smallest fraction of the shorter sequence the similarity must cover (default 0.3).
    *  ``--no-masking`` : Do not mask low-complexity regions, so that similarity in them (common in disordered proteins) also links sequences. This is even more stringent; by default these regions are masked, so that similarity driven only by compositional bias or simple repeats does not count.

Options for ``--split homology`` only:

    *  ``--mmseqs-sensitivity`` : MMseqs2 sensitivity, 1 to 7.5 (default 7.5, the most sensitive).
    *  ``--mmseqs`` : The MMseqs2 executable, if ``mmseqs`` is not on the PATH.
    *  ``--threads`` : Threads for MMseqs2 (default: every CPU).

Options for ``--split kmer`` only:

    *  ``--kmer-size`` : Length of the shared words, 3 to 8 (default 4). 3 finds slightly more distant relatives but is about 20 times slower and more easily misled by chance matches; 5 is about 5 times faster and slightly less sensitive.

**Installing MMseqs2.** MMseqs2 is a separate program, needed only for ``--split homology``. Install it with ``conda install -c conda-forge -c bioconda mmseqs2`` or ``brew install mmseqs2``, or download a static binary from the `MMseqs2 releases <https://github.com/soedinglab/MMseqs2/releases>`_. If it cannot be found, ``parrot-preprocess`` stops with these instructions before doing any work; it never silently falls back to a weaker split. Splitting 1,300 sequences takes about 15 seconds on 20 CPU cores, and the 6,049-protein yeast proteome about a minute.

**The k-mer split (no MMseqs2).** ``--split kmer`` finds similar sequences with a search written in Python and NumPy, so it works wherever PARROT does. It works like the seeding step of BLAST or MMseqs2. Related sequences share short identical words (k-mers, 4 residues by default), and the shared words line up along one alignment diagonal; chance matches between unrelated sequences scatter over many diagonals. For each pair of sequences, shared k-mers are chained into runs along bands of nearby diagonals, which tolerates short insertions and deletions. Each pair is then scored like an MMseqs2 alignment:

    *  *E-value*, from the run with the most shared k-mers. The chance of a k-mer matching by chance comes from the two sequences' own amino-acid compositions, so compositionally biased sequences (such as many disordered regions) need more shared k-mers to count. As with MMseqs2, it is the number of equally strong matches a query would be expected to find by chance in the database searched.
    *  *Coverage*, from the stretch spanned by the best run and every other run near its diagonal that is itself unlikely by chance, as a fraction of the shorter sequence.

Masking ignores residues in 12-residue windows with less than 1.8 bits of sequence entropy, and k-mers of at most two distinct amino acids (for k = 4). Shared k-mers are counted as distinct positions, so a repeat cannot count one word many times.

Exact shared words cannot detect distant homologues, which share few identical stretches. Compared with the pairs MMseqs2 links in the yeast proteome, ``--split kmer`` links all pairs at 70% identity or more, 97% at 50-70% and 82% at 40-50%, but only 18% at 30-40%. When the yeast proteome is split each way and the sets are then searched against each other with MMseqs2:

======================  =====================================  ===================================================  ==============
Split                   Test proteins with a homologue         Test-training alignments at >= 90% / 50-90% /        Data removed
                        in the training set                    < 50% identity
======================  =====================================  ===================================================  ==============
``--split random``      45%                                    383 / 547 / 2,104                                    0%
``--split kmer``        26%                                    0 / 9 / 1,225                                        1.4%
``--split homology``    0%                                     0 / 0 / 0                                            4.6%
======================  =====================================  ===================================================  ==============

For the 46,067 disordered regions of the ALBATROSS dataset, the share of test sequences with an MMseqs2 homologue in the training set falls from 5.3% with a random split (137 alignments at >= 90% identity) to 1.5% with ``--split kmer`` (none at >= 90%), and to 0 with ``--split homology``.

So ``--split kmer`` removes essentially all leakage between close relatives, which is the strongest leakage, but not between distant homologues. Use ``--split homology`` whenever MMseqs2 is available. A k-mer split takes about a minute for the yeast proteome and about three minutes for ALBATROSS (MMseqs2: about a minute and 25 seconds).

Class balancing
---------------

Classification only. ``--class-balance`` down-samples the more common classes. If the data are split, only the training set is balanced: the validation and test sets keep the data's natural class mix, so reported performance reflects real data. The imbalance is the count of the most common class divided by the count of the least common class (1.0 is perfectly balanced).

    *  ``--cb-balance-threshold`` : Stop once the imbalance is at most this (default 1.4).
    *  ``--cb-max-fraction-removed`` : The largest fraction of the (training) sequences that may be removed (default 0.25).
    *  ``--cb-block-size`` : For residue data, how many randomly drawn sequences are considered for each removal (default 100; larger makes better choices, more slowly).

For **sequence** data, every class is capped at the same size. The cap is lowered until the threshold is reached, or as far as the budget allows. This removes the fewest sequences needed for the best balance possible. The least common class is never reduced, and sequences with a missing label are never removed.

For **residue** data, each sequence contains residues of several classes, so sequences are removed one at a time. Each removal is the one, among ``--cb-block-size`` randomly drawn sequences, that improves the balance most. If none of them improves it, every remaining sequence is checked before stopping. A removal that would make the balance worse, or remove a class entirely, is never made. The imbalance after each removal is written to ``<output>_class_balance_trace.csv``.

Balancing keeps the input order of the lines.

Matrix data
-----------

For matrix data (``-d matrix``) the datafile holds ``<ID> <sequence>`` on each line, and the matrices are in a separate ``.npz`` file. Cleaning, deduplication (``first`` or ``drop``) and splitting work as for other data, and every sequence keeps its ID, so the original matrix file works with the processed datafile and split-file unchanged. Options that change values (``--recode-classes``, ``--class-balance`` and ``--deduplicate merge``) do not apply, since the values are in the matrix file.

    *  ``--matrix-file`` : The ``.npz`` matrix file. If given, the final check also checks that every sequence of the output has a matrix that fits it.
    *  ``--matrix-width`` : The number of columns of fixed-width matrices (leave it out for square maps); used by that check.

See the "Matrix data" page for the file formats.

Output
------

    *  *output_file* : The processed datafile (training, then validation, then test lines, if the data were split).
    *  ``<output>_split_file.txt`` : With ``--split``, the split-file to train on with ``parrot-train --split``. The datafile and split-file only match each other, so always use them together.
    *  ``<output>_preprocess_report.txt`` : What every step did. It ends with a check that PARROT reads the output (and that the split-file matches it).
    *  ``<output>_class_mapping.txt`` : With ``--recode-classes``, the label mapping.
    *  ``<output>_class_balance_trace.csv`` : With ``--class-balance`` on residue data, the imbalance after each removal.
