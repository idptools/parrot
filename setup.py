"""
PARROT
A bidirectional recurrent neural network framework for protein bioinformatics
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='idptools-parrot',
    author='Holehouse Lab',
    author_email='degriffith@wustl.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    # include_package_data=True,

    # Include data files
    # package_data={},

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    scripts=['scripts/parrot-train',
             'scripts/parrot-predict',
             'scripts/parrot-optimize',
             'scripts/parrot-cvsplit',
             'scripts/parrot-preprocess'],

    # Required packages, pulls from pip if needed; do not use for Conda deployment
    install_requires=[
            'cython',
            'torch>=1.8.0',
            'numpy',
            'more-itertools',
            'scipy',
            'scikit-learn',
            'matplotlib',
            'seaborn',
            'pandas'],

    extras_require={
            'optimize': ['GPy', 'GPyOpt'] },


    python_requires=">=3.7,<3.12.0",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,

)
