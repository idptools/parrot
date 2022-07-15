"""
PARROT
A bidirectional recurrent neural network framework for protein bioinformatics
"""

# Add imports here
from .encode_sequence import *
from .process_input_data import *
from .brnn_architecture import *
from .brnn_plot import *
from .train_network import *
from .bayesian_optimization import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_directory(path='../data'):
    return os.path.join(_ROOT, path)
