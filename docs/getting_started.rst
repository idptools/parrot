Getting Started with PARROT
===========================

Installation
------------

PARROT needs Python 3.11 or newer. It is available through GitHub or the Python Package Index (PyPI). To install through PyPI, run

.. code-block:: bash

	$ pip install idptools-parrot

**PARROT2 has not yet been released.** These pages describe PARROT2 (version 2.0.0.dev0), which is developed on the ``parrot2`` branch on GitHub. Until it is released, ``pip install idptools-parrot`` installs the latest release of PARROT 1 (1.7.5), and a clone of the repository gives PARROT 1 unless you run ``git checkout parrot2`` before installing it. To install PARROT2 directly from GitHub, run ``pip install "git+https://github.com/idptools/parrot.git@parrot2"`` (and, for an extra such as ``plm`` below, ``pip install "idptools-parrot[plm] @ git+https://github.com/idptools/parrot.git@parrot2"``).

To clone the GitHub repository and gain the ability to modify a local copy of the code, run

.. code-block:: bash

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

**Hyperparameter search**

``parrot-optimize`` needs no extra packages. (Earlier versions needed GPy and GPyOpt, installed with ``pip install idptools-parrot[optimize]``; that command still works, but installs nothing extra.)

**Homology-aware splitting**

``parrot-preprocess --split homology`` (and ``--split mmseqs`` in ``parrot-train`` and ``parrot-optimize``) needs `MMseqs2 <https://github.com/soedinglab/MMseqs2>`_, a separate program: ``conda install -c conda-forge -c bioconda mmseqs2``, ``brew install mmseqs2``, or a static binary from the MMseqs2 releases page. ``--split kmer`` needs nothing extra.

**Protein language model and NARDINI encodings**

These encodings need packages that are not installed by default:

* ESM-2 embeddings (e.g. ``--encode esm2_t12_35M_UR50D``): ``pip install idptools-parrot[plm]``.
* ESM C embeddings (``--encode esmc_300m`` or ``esmc_600m``): ``pip install idptools-parrot[esmc]``. Do not install Meta's ``fair-esm`` in the same environment, as both use the import name ``esm``.
* NARDINI+ features (``--encode nardini``): ``pip install git+https://github.com/idptools/sparrow.git`` and ``pip install idptools-parrot[nardini]``.

See :doc:`usage/plm_encodings` and :doc:`usage/nardini_encoding`.

Testing
-------

To see if your local installation of PARROT is working properly, first install the "pytest" package:

.. code-block:: bash
	
	$ pip install pytest

Then, you can run the unit tests included in the package by navigating to the top-level directory of the cloned repository and running:

.. code-block:: bash

	$ pytest -v

The tests always run against the source code in the repository you are in. The full suite (over 2,000 tests) takes a few minutes; on a machine with an NVIDIA or Apple-silicon GPU it also checks that every architecture gives the same results on the GPU as on the CPU, up to float32 rounding.

Note that this only works if the package is installed as a repository via GitHub. Installation through PyPI does not include the necessary datafiles to run the tests.

Example datasets
----------------

Example data that can be used with PARROT can be found in the **/data** folder on GitHub, where ``data/README.md`` describes each file. Examples of usage with these datasets can be found in :doc:`usage/basic_examples` and :doc:`usage/advanced_examples`.

Using the commands
------------------

PARROT has five commands:

* ``parrot-preprocess`` gets a datafile ready for training: it cleans it, deals with duplicate sequences, splits it into training, validation and test sets (keeping homologous sequences in the same set, if you choose), balances classes, and can compute protein language model or NARDINI encodings ahead of time (:doc:`usage/parrot-preprocess`).
* ``parrot-train`` trains a network of the architecture you choose, with the hyperparameters you give, and tests it (:doc:`usage/parrot-train`, :doc:`usage/architectures`).
* ``parrot-optimize`` searches for good hyperparameters, then trains and tests a network with them (:doc:`usage/parrot-optimize`).
* ``parrot-predict`` predicts new sequences with a trained network (:doc:`usage/parrot-predict`).
* ``parrot-cvsplit`` writes split-files for k-fold cross-validation (:doc:`usage/parrot-cvsplit`).

Every command starts by printing a header with PARROT's version and the date and time (unless given ``--silent``), which records in any saved output which version produced it. Each command has two options for finding your way around it:

* ``-h`` / ``--help`` lists the command's options (for ``parrot-train`` and ``parrot-optimize``, ``-h`` lists the architectures, and ``parrot-train <architecture> -h`` an architecture's options).
* ``--show-examples`` prints a longer guide: what the command does, worked examples of how to use it, and how it works, step by step.

When something goes wrong
-------------------------

If a command's input has a problem (a file that does not exist, a datafile that does not match the options given, an option that does not apply), the command stops with a short message saying what is wrong, and exits with a non-zero status (2 for a mistake in the command line itself, such as a missing or unknown option, which is shown with the command's usage; 1 otherwise), for example:

.. code-block:: bash

    parrot-train: error: Input file does not exist: data.tsv
    (For the full traceback, run again with PARROT_TRACEBACK=1.)

To also see the full Python traceback (for example, to report a problem), run the command again with the environment variable ``PARROT_TRACEBACK`` set:

.. code-block:: bash

    PARROT_TRACEBACK=1 parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression

Errors that are not a problem with the input are bugs in PARROT; they always show their traceback, and we would be grateful if you reported them on GitHub.
