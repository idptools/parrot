Getting Started with PARROT
===========================

Installation
------------

PARROT is available through GitHub or the Python Package Index (PyPI). To install through PyPI, run

.. code-block:: bash

	$ pip install idptools-parrot

To clone the GitHub repository and gain the ability to modify a local copy of the code, run

.. code-block:: bash

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

**Hyperparameter search**

``parrot-optimize`` needs no extra packages. (Earlier versions needed GPy and GPyOpt, installed with ``pip install idptools-parrot[optimize]``; that command still works, but installs nothing extra.)

**Homology-aware splitting**

``parrot-preprocess --split homology`` needs `MMseqs2 <https://github.com/soedinglab/MMseqs2>`_, a separate program: ``conda install -c conda-forge -c bioconda mmseqs2``, ``brew install mmseqs2``, or a static binary from the MMseqs2 releases page.

Testing
-------

To see if your local installation of PARROT is working properly, first install the "pytest" package:

.. code-block:: bash
	
	$ pip install pytest

Then, you can run the unit tests included in the package by navigating to the top-level directory of the cloned repository and running:

.. code-block:: bash

	$ pytest -v

The tests always run against the source code in the repository you are in. The full suite takes well under a minute on a CPU.

Note that this only works if the package is installed as a repository via GitHub. Installation through PyPI does not include the necessary datafiles to run the tests.

Example datasets
----------------

Example data that can be used with PARROT can be found in the **/data** folder on GitHub. Examples of usage with these datasets can be found in the documentation.

Using the commands
------------------

Every PARROT command (``parrot-train``, ``parrot-optimize``, ``parrot-predict``, ``parrot-preprocess`` and ``parrot-cvsplit``) starts by printing a header with PARROT's version and the date and time, which records in any saved output which version produced it. Each command has two options for finding your way around it:

* ``-h`` / ``--help`` lists the command's options (for ``parrot-train`` and ``parrot-optimize``, ``-h`` lists the architectures, and ``parrot-train <architecture> -h`` an architecture's options).
* ``--show-examples`` prints a longer guide: what the command does, worked examples of how to use it, and how it works, step by step.

When something goes wrong
-------------------------

If a command's input has a problem (a file that does not exist, a datafile that does not match the options given, an option that does not apply), the command stops with a one-line message saying what is wrong, for example:

.. code-block:: bash

    parrot-train: error: Datafile does not exist: data.tsv

To also see the full Python traceback (for example, to report a problem), run the command again with the environment variable ``PARROT_TRACEBACK`` set:

.. code-block:: bash

    PARROT_TRACEBACK=1 parrot-train brnn --input-file data.tsv --output-network network.pt --split kmer -d sequence --regression

Errors that are not a problem with the input are bugs in PARROT; they always show their traceback, and we would be grateful if you reported them on GitHub.
