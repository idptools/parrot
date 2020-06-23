# Getting Started with PARROT

## Installation

PARROT is available through GitHub or the Python Package Index (PyPI). To install through PyPI, run

	$ pip install idptools-parrot

To clone the GitHub repository and gain the ability to modify a local copy of the code, run

	$ git clone https://github.com/idptools/parrot.git
	$ cd parrot
	$ pip install .

This will install PARROT locally. If you modify the source code in the local repository, be sure to reinstall with pip.

## Testing

To see if your installation of PARROT is working properly, you can run the unit test included in the package by navigating to the parrot/tests folder within the installation directory and running:

	$ pytest -v

Example data can be found in the parrot/data folder on GitHub and information on usage can be found on the Examples page of the documentation.

## Documentation

The documentation for PARROT can be accessed at https://idptools-parrot.readthedocs.io/en/latest/index.html