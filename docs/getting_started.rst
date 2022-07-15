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

**JULY 2022 UPDATE**

To mitigate package version dependency issues involving Python, GPy, and PyTorch, new releases of PARROT have separated 
`parrot-optimize` as an optional add-on installation. All of the documentation for using `parrot-optimize` is unchanged.
However, if you wish to use the hyperparameter optimization, there are now slight differences in how you install PARROT.

To install the PARROT that is compatible with `parrot-optimize` install via pip using:

.. code-block:: bash

	$ pip install idptools-parrot[optimize]

or

.. code-block:: bash

	$ pip install "idptools-parrot[optimize]"

Alternatively if you have the PARROT repository cloned locally you can install using

.. code-block:: bash

	$ pip install ".[optimize]"


Testing
-------

To see if your local installation of PARROT is working properly, first install the "pytest" package:

.. code-block:: bash
	
	$ pip install pytest

Then, you can run the unit test included in the package by navigating to the /tests folder within the installation directory and running:

.. code-block:: bash

	$ pytest -v

Note that this only works if the package is installed as a repository via GitHub. Installation through PyPI does not include the necessary datafiles to run the tests.

Example datasets
----------------

Example data that can be used with PARROT can be found in the **/data** folder on GitHub. Examples of usage with these datasets can be found in the documentation.
