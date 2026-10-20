"""
Building blocks shared by PARROT's command-line tools: the data loading, and
saving and reporting, that parrot-train and parrot-optimize share (their
command line is built in training_parser.py); the device, embedding-cache and
embedding-context options and the encoding of sequences ahead of time, which
parrot-predict and parrot-preprocess also use (parrot-preprocess also shares
--nardini-scrambles and --embed-mean-subtraction); the check of
--probabilistic-classification, which
parrot-predict also uses; and the datatypes and legacy-option note that
parrot-predict and parrot-preprocess also use.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import csv
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from parrot import devices, embedding_context, figures, nardini, plm, prediction, reporting, train_network
from parrot import process_input_data as pid
from parrot.architectures import ParrotArchitecture, get_architecture
from parrot.architectures.classical import ClassicalModel
from parrot.checkpoint import save_network
from parrot.encode_sequence import get_encoder
from parrot.metrics import METRIC_DECIMALS
from parrot.reproducibility import set_random_seed
from parrot.sequence_encoder import SequenceEncoder
from parrot.scripts import arguments, network_info
from parrot.scripts.data_split import DataSplit, data_split_from_args
from parrot.tasks import Task
from parrot.tools import dataset_warnings

# The datatypes the command-line tools accept
COMMAND_LINE_DATATYPES = ('sequence', 'residues', 'matrix')

# Square maps take memory in proportion to length^2: a 512-residue map has about 262,000 cells, and the 2D U-Net
# with its default settings keeps roughly 1-3 kB per cell while training (an estimate from its layer sizes), so one
# such sequence needs up to about 1 GB. Longer sequences are refused unless --max-matrix-length is raised.
DEFAULT_MAX_MATRIX_LENGTH = 512

# What each datatype means, for describing a run's data
DATATYPE_DESCRIPTIONS = {
    'sequence': 'one value per sequence',
    'residues': 'one value per residue',
    'matrix': 'a matrix per sequence',
}

# Networks are saved with this extension (added to --output-network if it does not end with it)
NETWORK_EXTENSION = '.pt'

# Ends the help of every flag kept only so that commands written for earlier versions of PARROT still work
LEGACY_NOTE = '(kept so that older commands keep working)'

# Batch sizes above this are very large for the datasets PARROT is used with, and may lower accuracy (rule of thumb)
LARGE_BATCH_SIZE = 512

# Fewer batches than this per epoch give the optimizer few updates per epoch (rule of thumb)
MIN_BATCHES_PER_EPOCH = 10


# .................................................................
# Arguments

def check_matrix_arguments(args: argparse.Namespace) -> None:
    """Check the matrix-data arguments of matrix data: a matrix file that exists, and a positive length limit

    Only architectures for matrix data offer these options (see training_parser.py).
    args.matrix_file is replaced by its absolute path.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by training_parser.parse_arguments()

    Raises
    ------
    ValueError
        If no matrix file was given for matrix data, or the length limit is not positive
    FileNotFoundError
        If the matrix file does not exist
    """
    if args.datatype != 'matrix':
        return

    if args.matrix_file is None:
        raise ValueError("Matrix data needs --matrix-file: the .npz file of each sequence's matrix")
    args.matrix_file = arguments.check_file_exists(args.matrix_file, 'Matrix file')
    arguments.check_positive(args.max_matrix_length, '--max-matrix-length')


def check_matrix_lengths(sequences: list[str], max_length: int) -> None:
    """Refuse sequences too long for square maps to fit in memory (see DEFAULT_MAX_MATRIX_LENGTH)

    Parameters
    ----------
    sequences : list of str
        Every sequence of the dataset
    max_length : int
        The longest sequence allowed (--max-matrix-length)

    Raises
    ------
    ValueError
        If any sequence is longer
    """
    too_long = [sequence for sequence in sequences if len(sequence) > max_length]
    if too_long:
        raise ValueError(f'{len(too_long)} sequence(s) are longer than --max-matrix-length ({max_length} residues; the '
                         f'longest has {max(len(sequence) for sequence in too_long)}). Square maps take memory in '
                         'proportion to length^2: remove the longest sequences (e.g. with parrot-preprocess '
                         '--max-length) or, if your GPU has the memory, raise --max-matrix-length.')


def add_embedding_cache_argument(parser: 'argparse._ActionsContainer', default_name: str) -> None:
    """Add --embedding-cache, which only applies to protein language model and NARDINI encodings

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the argument
    default_name : str
        Name of the default cache directory, created next to the command's output
    """

    parser.add_argument('--embedding-cache', dest='embedding_cache', metavar='directory',
                        help='protein language model and NARDINI encodings only: directory in which to store each '
                             f'sequence\'s encoding (default: a "{default_name}" directory next to the output). Point '
                             'several runs at the same directory to encode each sequence only once')


def add_nardini_scrambles_argument(parser: 'argparse._ActionsContainer') -> None:
    """Add --nardini-scrambles, which only applies to NARDINI encodings

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the argument
    """

    parser.add_argument('--nardini-scrambles', dest='nardini_scrambles', type=int, metavar='num_scrambles',
                        help='NARDINI encodings only: scrambles of each sequence used for its patterning z-scores '
                             f'(default {nardini.DEFAULT_NUM_SCRAMBLES:,}; more take longer: 10,000 take about '
                             'three to four times as long as 1,000)')


def add_embedding_context_arguments(parser: 'argparse._ActionsContainer') -> None:
    """Add --embedding-context and --embedding-context-map, which only apply to protein language model encodings

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the arguments
    """

    parser.add_argument('--embedding-context', dest='embedding_context', metavar='fasta_file',
                        help='protein language model encodings only: a FASTA file of the full-length proteins the '
                             'sequences were cut from. Each sequence is then embedded within its full-length protein '
                             '(named in --embedding-context-map), rather than on its own. A network trained with it '
                             'needs it again in parrot-predict (or --embed-in-isolation)')
    parser.add_argument('--embedding-context-map', dest='embedding_context_map', metavar='map_file',
                        help="with --embedding-context: one '<sequence ID> <full-length ID> [start]' line per "
                             'sequence, naming the protein it was cut from (by the first word of its FASTA header '
                             'line). start is the 1-based position of its first residue, needed only if it occurs '
                             f"more than once in the protein; a full-length ID of '{embedding_context.NO_CONTEXT}' "
                             'embeds that sequence on its own')


def add_embed_mean_subtraction_argument(parser: 'argparse._ActionsContainer') -> None:
    """Add --embed-mean-subtraction, which only applies to protein language model encodings

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the argument
    """

    parser.add_argument('--embed-mean-subtraction', dest='embed_mean_subtraction', action='store_true',
                        help="protein language model encodings only: subtract each embedding's mean over its residues "
                             "(its mean pool) from every residue's embedding before it is stored, so that what is "
                             'left is how each residue differs from the sequence as a whole (with '
                             '--embedding-context, from the whole full-length protein). Stored in their own files in '
                             'the embedding cache; recorded in a network trained with it, so parrot-predict applies '
                             'it automatically')


def check_mean_subtraction_applies(args: argparse.Namespace, encoder: SequenceEncoder) -> None:
    """Refuse --embed-mean-subtraction for the scikit-learn models when sequences are embedded on their own

    Their features are each sequence's mean embedding and the log of its length
    (see architectures/classical.py). Mean subtraction makes the mean embedding
    of a sequence embedded on its own zero, so they would learn from its length
    alone. Within a full-length protein (--embedding-context), the protein's
    mean is subtracted instead, and a fragment's mean embedding is how it
    differs from its protein, so that is allowed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed training arguments (with --embedding-context already checked)
    encoder : SequenceEncoder
        The encoding in use

    Raises
    ------
    ValueError
        If the architecture is a scikit-learn model and sequences are mean-subtracted on their own
    """
    if not (isinstance(encoder, plm.PLMEncoder) and encoder.mean_subtracted):
        return
    if args.embedding_context is not None or not issubclass(get_architecture(args.architecture), ClassicalModel):
        return
    raise ValueError(f"--embed-mean-subtraction cannot be used with '{args.architecture}' unless the sequences are "
                     'embedded within their full-length proteins (--embedding-context): its features are each '
                     "sequence's mean embedding, which mean subtraction makes zero")


def check_embedding_context_arguments(args: argparse.Namespace, encoder: SequenceEncoder) -> None:
    """Check --embedding-context and --embedding-context-map before any work is done

    The two files' paths are replaced by their absolute paths.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_embedding_context_arguments()
    encoder : SequenceEncoder
        The encoding in use

    Raises
    ------
    ValueError
        If only one of the two is given, the encoding is not a protein language model, or the sequences have no
        IDs (--exclude-seq-id)
    FileNotFoundError
        If either file does not exist
    """
    if args.embedding_context is None and args.embedding_context_map is None:
        return
    if args.embedding_context is None or args.embedding_context_map is None:
        raise ValueError('--embedding-context and --embedding-context-map go together: the FASTA file of full-length '
                         'proteins, and which of them each sequence was cut from')
    if not isinstance(encoder, plm.PLMEncoder):
        raise ValueError('--embedding-context only applies to protein language model encodings (other encodings do '
                         f"not depend on the rest of the sequence), not '{encoder.name}'")
    if getattr(args, 'exclude_seq_id', False):
        raise ValueError("--embedding-context finds each sequence's full-length protein by its sequence ID, so it "
                         'cannot be used with --exclude-seq-id')
    args.embedding_context = arguments.check_file_exists(args.embedding_context, 'FASTA file of full-length proteins')
    args.embedding_context_map = arguments.check_file_exists(args.embedding_context_map, 'Embedding context map')


def place_in_context(args: argparse.Namespace, encoder: SequenceEncoder, ids: list[str],
                     sequences: list[str]) -> embedding_context.ContextAssignment | None:
    """Give the encoder each sequence's full-length context, if --embedding-context was given

    Parameters
    ----------
    args : argparse.Namespace
        Arguments checked by check_embedding_context_arguments()
    encoder : SequenceEncoder
        The encoding in use (a PLMEncoder if --embedding-context was given)
    ids, sequences : list of str
        Every sequence that will be encoded, and its ID

    Returns
    -------
    ContextAssignment or None
        Where each sequence is embedded (None without --embedding-context)

    Raises
    ------
    ValueError
        If any sequence cannot be placed in its full-length protein (see embedding_context.assign_contexts())
    """
    if args.embedding_context is None:
        return None
    assert isinstance(encoder, plm.PLMEncoder)
    assignment = embedding_context.assign_contexts(ids, sequences, args.embedding_context, args.embedding_context_map)
    encoder.add_contexts(assignment.contexts)
    return assignment


def write_context_table(assignment: embedding_context.ContextAssignment, output_prefix: str, silent: bool) -> str:
    """Write where each sequence is embedded to <output_prefix>_embedded_context.tsv, and say so (unless silent)

    Parameters
    ----------
    assignment : ContextAssignment
    output_prefix : str
        Path and filename prefix of the command's output
    silent : bool
        If True, print nothing

    Returns
    -------
    str
        The file written
    """
    table_file = output_prefix + embedding_context.CONTEXT_TABLE_SUFFIX
    embedding_context.write_context_table(table_file, assignment.rows)
    if not silent:
        print(f'\nEmbedding context: {assignment.describe()}. Where each sequence is embedded is written to '
              f'{table_file}')
    return table_file


def prepare_encoding(encoder: SequenceEncoder, sequences: list[str], cache_directory: str | None,
                     default_directory: str, device: torch.device, program: str, silent: bool) -> None:
    """Encode every sequence a command will need ahead of time, for encodings that are slow to compute

    Protein language model embeddings and NARDINI features are computed once,
    into the embedding cache; every other encoding is computed as it is needed.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoding in use
    sequences : list of str
        Every sequence that will be encoded
    cache_directory : str or None
        The --embedding-cache argument
    default_directory : str
        The cache directory to use if --embedding-cache was not given
    device : torch.device
        Device to run the protein language model on
    program : str
        The command running (recorded in the cache's README)
    silent : bool
        If True, print nothing

    Raises
    ------
    ValueError
        If --embedding-cache was given for an encoding that does not use the cache, or the encoder cannot encode
        the sequences (see PLMEncoder.prepare() and NardiniEncoder.prepare())
    RuntimeError
        If the disk does not have room for the protein language model embeddings
    """
    directory = cache_directory if cache_directory is not None else default_directory
    check_embedding_cache_applies(encoder, cache_directory)

    if isinstance(encoder, nardini.NardiniEncoder):
        encoder.use_cache(directory)
        encoder.prepare(sequences, program, silent=silent)
        return

    if not isinstance(encoder, plm.PLMEncoder):
        return

    encoder.use_device(device)
    encoder.use_cache(directory)
    encoder.prepare(sequences, program, silent=silent)

    # Every sequence is now cached, so the model's memory can be freed
    encoder.release_model()


def add_device_arguments(parser: 'argparse._ActionsContainer') -> None:
    """Add the arguments that choose the device (shared by every command that runs a network or a language model)

    Parameters
    ----------
    parser : argparse.ArgumentParser or argument group
        Where to add the arguments
    """

    parser.add_argument('--device', default='auto', choices=devices.DEVICE_CHOICES,
                        help="device to run on: 'auto' (default) uses a CUDA GPU if available, else an "
                             "Apple-silicon GPU (MPS) if available, else the CPU")
    parser.add_argument('--gpu-id', dest='gpu_id', type=int, metavar='index',
                        help='index of the CUDA device to use, on machines with more than one NVIDIA GPU (this asks '
                             'for CUDA, so it cannot be combined with --device cpu or mps)')
    parser.add_argument('--force-cpu', action=arguments.RetiredOption, replacement='use --device cpu')


def check_probabilistic_classification(probabilistic: bool, task: Task) -> None:
    """Check that class probabilities were only asked for a task that has them (sequence or residue classification)

    Parameters
    ----------
    probabilistic : bool
        Whether --probabilistic-classification was given
    task : Task

    Raises
    ------
    ValueError
        If probabilities were asked for regression, or for matrix data
    """
    if probabilistic and (not task.is_classification or task.is_matrix):
        raise ValueError('Probabilistic classification applies to sequence and residue classification only')


def select_device(args: argparse.Namespace) -> torch.device:
    """Choose the device from the --device and --gpu-id arguments, and say which it is (unless --silent)

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_device_arguments()

    Returns
    -------
    torch.device

    Raises
    ------
    ValueError
        If the arguments are inconsistent (see devices.resolve_device)
    RuntimeError
        If the requested device is not available
    """

    device = devices.resolve_device(args.device, args.gpu_id)
    if not getattr(args, 'silent', False):
        print(f'Running on: {devices.describe_device(device)}')
    return device


# .................................................................
# Setting up and finishing a training run

@dataclass
class TrainingRun:
    """Everything parrot-train and parrot-optimize set up in the same way

    Attributes
    ----------
    program_name : str
        The command running, e.g. 'parrot-train'
    data_file : str
        Absolute path of the datafile
    split_description : str
        How the data were divided into training, validation and test sets
    seed : int or None
        The random seed (None if not seeded)
    saved_split_file : str or None
        The split-file written with --save-splits, if any
    task : Task
        The task being trained
    encoder : SequenceEncoder
        How sequences are encoded
    device : torch.device
        Device to train on
    network_file : str
        Where the trained network is saved
    output_prefix : str
        Path and filename prefix for every other output file
    train_set, val_set, test_set : SequenceDataset
        The data, split into training, validation and test sets
    cv_sets : list of (SequenceDataset, SequenceDataset) or None
        Cross-validation folds of the training + validation data, if requested
    batch_size : int
        Number of sequences per training batch
    batching : str
        How sequences are grouped into batches ('length' or 'random')
    probabilistic : bool
        Report class probabilities rather than class labels
    include_figs : bool
        Save figures
    write_stats : bool
        Write the performance-metrics file
    verbose : bool
        Print more detail
    silent : bool
        Print nothing
    context : ContextAssignment or None
        With --embedding-context, where each sequence is embedded
    context_table : str or None
        With --embedding-context, the table of where each sequence is embedded
    """

    program_name: str
    data_file: str
    split_description: str
    seed: int | None
    saved_split_file: str | None
    task: Task
    encoder: SequenceEncoder
    device: torch.device
    network_file: str
    output_prefix: str
    train_set: pid.SequenceDataset
    val_set: pid.SequenceDataset
    test_set: pid.SequenceDataset
    cv_sets: list[tuple[pid.SequenceDataset, pid.SequenceDataset]] | None
    batch_size: int
    batching: str
    probabilistic: bool
    include_figs: bool
    write_stats: bool
    verbose: bool
    silent: bool
    context: embedding_context.ContextAssignment | None = None
    context_table: str | None = None
    split_state: dict[str, Any] | None = None

    def loader(self, dataset: pid.SequenceDataset, shuffle: bool = False) -> DataLoader:
        """Wrap one of this run's datasets in a DataLoader."""
        return pid.make_dataloader(dataset, self.task, self.batch_size, shuffle=shuffle, batching=self.batching)

    def save(self, network: ParrotArchitecture) -> None:
        """Save a network (with this run's encoder) to the output network file."""
        save_network(self.network_file, network, self.encoder)


def warn_about_batch_size(batch_size: int, num_training_sequences: int) -> None:
    """Warn if the batch size is very large, or large relative to the training set

    Parameters
    ----------
    batch_size : int
        Sequences per batch
    num_training_sequences : int
        Sequences in the training set (or in one cross-validation fold's training set)
    """
    if batch_size > LARGE_BATCH_SIZE:
        dataset_warnings.print_warning(f'WARNING: The batch size ({batch_size}) is very large.',
                                       ['This may lower accuracy; consider a smaller --batch.'])

    batches_per_epoch = math.ceil(num_training_sequences / batch_size)
    if batches_per_epoch < MIN_BATCHES_PER_EPOCH:
        dataset_warnings.print_warning(
            f'WARNING: The batch size ({batch_size}) is large for {num_training_sequences:,} training sequences '
            f'({batches_per_epoch} batches per epoch).',
            ['Few updates per epoch may make training less efficient; consider a smaller --batch.'])


def check_embedding_cache_applies(encoder: SequenceEncoder, cache_directory: str | None) -> None:
    """Check --embedding-cache is only given for an encoding that uses the cache (checked before any work is done)

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoding in use
    cache_directory : str or None
        The --embedding-cache argument

    Raises
    ------
    ValueError
        If --embedding-cache was given for an encoding that does not use the cache
    """
    if cache_directory is not None and not isinstance(encoder, plm.PLMEncoder | nardini.NardiniEncoder):
        raise ValueError('--embedding-cache only applies to protein language model and NARDINI encodings, not '
                         f"'{encoder.name}'")


def check_network_name(name: str) -> None:
    """Check --output-network names a file, not just a directory (checked before any work is done)

    Parameters
    ----------
    name : str
        The --output-network argument

    Raises
    ------
    ValueError
        If it ends with a path separator, or is only the network extension
    """
    if os.path.basename(name) in ('', NETWORK_EXTENSION):
        raise ValueError(f"--output-network {name} names a directory, not a file; give the network's name too, "
                         f'e.g. {os.path.join(name.removesuffix(NETWORK_EXTENSION), "network" + NETWORK_EXTENSION)}')


def network_path(name: str) -> str:
    """The file a network is saved to: `name`, with NETWORK_EXTENSION added if it does not end with it

    Parameters
    ----------
    name : str
        The --output-network argument

    Returns
    -------
    str
    """
    return name if name.endswith(NETWORK_EXTENSION) else name + NETWORK_EXTENSION


def number_of_classes(args: argparse.Namespace) -> int:
    """The number of outputs the task needs: 1 for --regression, --num-classes for --classification

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by training_parser.parse_arguments()

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If --num-classes is missing for --classification, given for --regression, or less than 2
    """
    if args.regression:
        if args.num_classes is not None:
            raise ValueError('--num-classes only applies to --classification (regression has one output)')
        return 1
    if args.num_classes is None:
        raise ValueError('--classification needs --num-classes: the number of classes (labelled 0, 1, ...)')
    if args.num_classes < 2:
        raise ValueError(f'--num-classes must be at least 2 for classification, got {args.num_classes}')
    return int(args.num_classes)


def describe_split(data_split: DataSplit, n_folds: int | None) -> str:
    """How the data are divided into training, validation and test sets, in words

    Parameters
    ----------
    data_split : DataSplit
        The split, once applied to the data
    n_folds : int or None
        Number of cross-validation folds, if configurations are scored by cross-validation

    Returns
    -------
    str
    """
    description = data_split.description()
    if n_folds is not None:
        description += (f'; hyperparameter configurations scored by {n_folds}-fold cross-validation over the training '
                        f'and validation sequences, {data_split.fold_description()}')
    return description


def prepare_training_run(args: argparse.Namespace, program_name: str, n_folds: int | None = None) -> TrainingRun:
    """Validate the shared arguments, then load and split the data

    Protein language model embeddings and NARDINI features of every sequence
    are also computed now, into the embedding cache (see prepare_encoding()),
    within each sequence's full-length protein if --embedding-context was given.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by training_parser.parse_arguments()
    program_name : str
        The command running, e.g. 'parrot-train'
    n_folds : int, optional
        If given, also divide the training + validation data into this many
        cross-validation folds

    Returns
    -------
    TrainingRun

    Raises
    ------
    FileNotFoundError
        If the datafile, matrix file, encoding file, --embedding-context files or output directory does not exist
    ValueError
        If any argument is invalid (including a --split that is neither a file nor a method), the datafile does
        not match the task, or a sequence cannot be placed in its full-length protein (--embedding-context)
    RuntimeError
        If the requested device is not available, --split mmseqs is asked for but MMseqs2 cannot be found, or the
        disk does not have room for the protein language model embeddings
    """

    set_random_seed(args.seed)

    data_file = arguments.check_file_exists(args.input_file, 'Input file')
    check_network_name(args.output_network)
    network_file = os.path.abspath(network_path(args.output_network))
    output_prefix, _ = arguments.output_prefix_and_directory(network_file)

    data_split = data_split_from_args(args)
    save_splits_output = None
    if args.save_splits:
        save_splits_output = output_prefix + '_split_file.txt'

    encoder = get_encoder(args.encode, standardize=args.standardize_encoding, nardini_scrambles=args.nardini_scrambles,
                          mean_subtracted=args.embed_mean_subtraction)
    check_embedding_cache_applies(encoder, args.embedding_cache)
    check_embedding_context_arguments(args, encoder)
    check_mean_subtraction_applies(args, encoder)
    check_matrix_arguments(args)
    if args.uncertainties and not args.regression:
        raise ValueError("--uncertainties applies to regression only: the SDs are of the values' measurement errors")
    task = Task(args.datatype, number_of_classes(args), width=args.matrix_width, symmetric=args.symmetric,
                uncertainties=args.uncertainties, regression_loss=args.regression_loss or 'auto',
                huber_delta=args.huber_delta if args.huber_delta is not None else 1.0)
    nardini.check_task(encoder, task)

    arguments.check_positive(args.epochs, 'Number of epochs')
    arguments.check_positive(args.batch, 'Batch size')
    check_probabilistic_classification(args.probabilistic_classification, task)

    ignore_warnings = args.ignore_warnings or args.silent
    device = select_device(args)
    from parrot.training_precision import resolve_amp
    amp_dtype = resolve_amp(args.amp, device)
    resumed = getattr(args, '_resume_payload', None)
    if resumed is not None:
        if device.type != resumed['random_state']['device_type']:
            raise ValueError('Resume requires the same device type as the checkpoint')
        if str(torch.__version__) != resumed['signature']['torch_version']:
            raise ValueError('Resume requires the same PyTorch version as the checkpoint')
        if resumed['kind'] == 'train':
            controls = [resumed['control_state']]
        else:
            controls = [fold['training_control'] for trial in resumed['search_state']['trials']
                        for fold in trial['folds'] if fold['training_control'] is not None]
        used_dtypes = {control['precision_state']['dtype'] for control in controls
                       if control['precision_state'] is not None}
        if used_dtypes and used_dtypes != {str(amp_dtype)}:
            raise ValueError('Resume requires the same resolved AMP dtype on the selected GPU')

    # The split-file, or the method, is applied as the data are read (see data_split.DataSplit)
    split_arguments: dict[str, Any] = dict(encoder=encoder, exclude_seq_id=args.exclude_seq_id,
                                           assign_sets=data_split, ignore_warnings=ignore_warnings,
                                           save_splits_output=save_splits_output, matrix_file=args.matrix_file)
    cv_sets = None
    if n_folds is None:
        train_set, val_set, test_set = pid.split_data(data_file, task, **split_arguments)
    else:
        cv_sets, train_set, val_set, test_set = pid.split_data_cv(data_file, task, n_folds=n_folds,
                                                                  assign_folds=data_split.cross_validation_folds,
                                                                  **split_arguments)

    # Models fitted in one step see the whole training set at once, so only networks have a batch size to warn about
    if not ignore_warnings and get_architecture(args.architecture).trained_by_gradient_descent:
        warn_about_batch_size(args.batch, len(cv_sets[0][0]) if cv_sets else len(train_set))

    # Check every sequence now, so a bad residue in (say) the test set fails before training rather than after
    every_sequence = train_set.sequences + val_set.sequences + test_set.sequences
    encoder.check_sequences(every_sequence)
    if task.is_square:
        check_matrix_lengths(every_sequence, args.max_matrix_length)
    every_id = train_set.names + val_set.names + test_set.names
    context = place_in_context(args, encoder, every_id, every_sequence)
    context_table = write_context_table(context, output_prefix, args.silent) if context is not None else None
    prepare_encoding(encoder, every_sequence, args.embedding_cache,
                     os.path.join(os.path.dirname(network_file), plm.TRAIN_CACHE_DIRECTORY), device,
                     program_name, args.silent)

    return TrainingRun(program_name=program_name, data_file=data_file,
                       split_description=describe_split(data_split, n_folds), seed=args.seed,
                       saved_split_file=save_splits_output,
                       task=task, encoder=encoder, device=device, network_file=network_file,
                       output_prefix=output_prefix, train_set=train_set, val_set=val_set, test_set=test_set,
                       cv_sets=cv_sets, batch_size=args.batch, batching=args.batching,
                       probabilistic=args.probabilistic_classification,
                       include_figs=args.include_figs, write_stats=not args.no_stats,
                       verbose=args.verbose, silent=args.silent, context=context, context_table=context_table,
                       split_state={'split_file': data_split.split_file, 'method': data_split.method,
                                    'fractions': data_split.fractions, 'criterion': data_split.criterion,
                                    'sets': data_split.assigned_sets,
                                    'folds': data_split.assigned_folds or data_split.saved_folds or []})


def print_run_header(title: str, run: TrainingRun, settings: dict[str, Any]) -> None:
    """Print the heading shown at the start of training: what is trained, on what data (and, if verbose, every setting)

    Parameters
    ----------
    title : str
        First line of the heading
    run : TrainingRun
        The run being started
    settings : dict
        Settings to list in verbose mode, in addition to the device, task and encoding
    """

    if run.silent:
        return

    print()
    print(title)
    print('-' * len(title))
    print(describe_data(run))
    if run.verbose:
        print()
        arguments.print_settings(False, device=run.device, datatype=run.task.datatype, task=run.task.problem_type,
                                 classes=run.task.num_classes, encoding=run.encoder.name, **settings)
    print()


def describe_data(run: TrainingRun) -> str:
    """One line on the data a run trains on: how many sequences each set has, and what is being predicted

    Parameters
    ----------
    run : TrainingRun

    Returns
    -------
    str
    """
    task = run.task
    predicted = DATATYPE_DESCRIPTIONS[task.datatype]
    if task.is_classification:
        predicted = f'classification into {task.num_classes} classes, {predicted}'
    elif task.uncertainties:
        predicted = f'regression with uncertainties, {predicted}'
    else:
        predicted = f'regression, {predicted}'

    description = (f'Data: {len(run.train_set):,} training, {len(run.val_set):,} validation and '
                   f'{len(run.test_set):,} test sequences ({predicted})')
    if run.cv_sets is not None:
        description += (f'; configurations are scored by {len(run.cv_sets)}-fold cross-validation over the training '
                        'and validation sequences')
    return description


def finish_training_run(run: TrainingRun, network: ParrotArchitecture, train_losses: list[float],
                        val_losses: list[float], training: dict[str, Any],
                        search: list[tuple[str, str]] | None = None, other_files: Sequence[str] = (),
                        timing: train_network.TrainingTiming | None = None,
                        search_seconds: float | None = None) -> None:
    """Save the trained network, evaluate and report on the test set, and describe how the network was trained

    Parameters
    ----------
    run : TrainingRun
        The run being finished
    network : ParrotArchitecture
        The trained network (holding its best weights)
    train_losses, val_losses : list of float
        Loss per epoch, as returned by train_network.train()
    training : dict
        The training settings, for the info file: 'learning_rate', 'epochs'
        and 'stop' (networks trained by gradient descent only)
    search : list of (str, str), optional
        parrot-optimize's description of its hyperparameter search, for the info file
    other_files : list of str, optional
        Other files the command wrote, to list with the rest
    timing : TrainingTiming, optional
        How long each epoch of training took, for the info file
    search_seconds : float, optional
        How long parrot-optimize's hyperparameter search took, for the info file
    """

    run.save(network)
    written = list(other_files)
    history = training.get('learning_rate_history')
    if history and not training.get('continuous_training_log'):
        path = run.output_prefix + '_training_history.tsv'
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(history)
        written.append(path)
    if training.get('recovery_events'):
        path = run.output_prefix + '_recovery_events.json'
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(training['recovery_events'], handle, indent=2, allow_nan=False)
            handle.write('\n')
        written.append(path)
    if run.saved_split_file is not None:
        written.append(run.saved_split_file)
    if run.context_table is not None:
        written.append(run.context_table)

    # Models fitted in one step have no loss curve to plot
    if run.include_figs and network.trained_by_gradient_descent:
        written.append(run.output_prefix + '_train_val_loss.png')
        figures.plot_training_loss(train_losses, val_losses, written[-1])

    predicting = time.perf_counter()
    loss_options = {'equal_protein_weighting': True} if training.get('equal_protein_weighting') else {}
    labeled = prediction.predict_labeled(network, run.test_set, run.task, run.device, batch_size=run.batch_size,
                                         **loss_options)
    prediction_seconds = time.perf_counter() - predicting
    metrics, test_files = reporting.report_test_results(labeled, run.task, run.output_prefix,
                                                        probabilistic=run.probabilistic,
                                                        include_figs=run.include_figs, write_stats=run.write_stats)
    written += test_files

    info_file = run.output_prefix + '_info.txt'
    sections = network_info.info_sections(run, network, training, train_losses, val_losses, labeled.loss, metrics,
                                          written, search, timing, prediction_seconds, search_seconds)
    network_info.write_network_info(info_file, run.network_file, sections)
    written.insert(0, info_file)

    if not run.silent:
        print_test_results(labeled.loss, metrics)
        print_written_files(run.network_file, written)


def print_test_results(loss: float, metrics: dict[str, float]) -> None:
    """Print the test-set loss and performance metrics on one line

    Parameters
    ----------
    loss : float
        The test-set loss
    metrics : dict
        Metric name -> value (see metrics.performance_metrics)
    """
    values = [f'loss {loss:.4f}'] + [f'{name} {value:.{METRIC_DECIMALS}f}' for name, value in metrics.items()]
    print(f'\nTest set: {", ".join(values)}')


def print_written_files(network_file: str, other_files: list[str]) -> None:
    """Say where the network was saved, and what else was written next to it

    Parameters
    ----------
    network_file : str
        The saved network
    other_files : list of str
        Every other file written (in the network's directory)
    """
    print(f'Network saved to {network_file}')
    if other_files:
        print(f'Also written there: {", ".join(os.path.basename(path) for path in other_files)}')
