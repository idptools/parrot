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
