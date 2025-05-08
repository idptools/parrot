"""
PARROT
A bidirectional recurrent neural network framework for protein bioinformatics
"""

# Add imports here
from parrot.encode_sequence import *
from parrot.process_input_data import *
from parrot.brnn_architecture import *
from parrot.brnn_plot import *
from parrot.train_network import *
from parrot.bayesian_optimization import *

# Generate _version.py if missing and in the Read the Docs environment
if os.getenv("READTHEDOCS") == "True" and not os.path.isfile('../parrot/_version.py'):   
    import versioningit            
    __version__ = versioningit.get_version('../')
else:
    from ._version import __version__


# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_directory(path='../data'):
    return os.path.join(_ROOT, path)
