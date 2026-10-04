"""
PARROT
A framework for training neural networks that map protein sequences to values,
either one value per sequence or one value per residue.

The names below are PARROT's public interface, grouped by what they are for.
Everything else lives in the module that defines it (e.g.
parrot.process_input_data.pad_sequence_vectors).
"""

from parrot.package_version import PARROT_VERSION as __version__

# Encoding sequences as numbers
from parrot.encode_sequence import (BIOPHYSICS, ONE_HOT, BiophysicsEncoder, OneHotEncoder, StandardizedEncoder,
                                    UserEncoder, biophysics, encoder_from_config, get_encoder, one_hot,
                                    parse_encode_file)
from parrot.sequence_encoder import SequenceEncoder

# Reading datafiles, splitting them into sets and batching them
from parrot.process_input_data import (SequenceDataset, cross_validation_folds, load_and_split, make_dataloader,
                                       parse_file, read_split_file, read_tsv_raw, split_data, split_data_cv,
                                       write_split_file)

# The machine-learning task, and the network architectures
from parrot.tasks import Task
from parrot.architectures import (HyperParameter, ParrotArchitecture, available_architectures, build_network,
                                  get_architecture, register_architecture)
from parrot.brnn_architecture import BRNN_MtM, BRNN_MtO

# Training, saving and predicting
from parrot.prediction import predict_labeled, predict_sequences
from parrot.train_network import evaluate_loss, train
from parrot.checkpoint import load_network, save_network
from parrot.py_predictor import Predictor

# Errors PARROT raises for bad input
from parrot.parrot_exceptions import IOExceptionParrot, ParrotException

__all__ = [
    '__version__',
    'BIOPHYSICS', 'ONE_HOT', 'BiophysicsEncoder', 'OneHotEncoder', 'SequenceEncoder', 'StandardizedEncoder',
    'UserEncoder', 'biophysics', 'encoder_from_config', 'get_encoder', 'one_hot', 'parse_encode_file',
    'SequenceDataset', 'cross_validation_folds', 'load_and_split', 'make_dataloader', 'parse_file',
    'read_split_file', 'read_tsv_raw', 'split_data', 'split_data_cv', 'write_split_file',
    'Task', 'HyperParameter', 'ParrotArchitecture', 'available_architectures', 'build_network', 'get_architecture',
    'register_architecture', 'BRNN_MtM', 'BRNN_MtO',
    'evaluate_loss', 'predict_labeled', 'predict_sequences', 'train', 'load_network', 'save_network', 'Predictor',
    'IOExceptionParrot', 'ParrotException',
]
