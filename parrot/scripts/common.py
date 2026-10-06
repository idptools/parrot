"""
Building blocks shared by PARROT's command-line tools: the arguments,
data loading, and saving and reporting that parrot-train and parrot-optimize
share, and the device and embedding-cache handling that parrot-predict also
uses.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from parrot import devices, figures, plm, prediction, reporting
from parrot import process_input_data as pid
from parrot.architectures import (DEFAULT_ARCHITECTURE, DEFAULT_MATRIX_ARCHITECTURE, HyperParameter, HyperParameterValue, ParrotArchitecture,
                                  available_architectures, get_architecture, resolve_hyperparameters)
from parrot.checkpoint import save_network
from parrot.encode_sequence import get_encoder
from parrot.sequence_encoder import SequenceEncoder
from parrot.reproducibility import set_random_seed
from parrot.scripts import arguments
from parrot.tasks import Task
from parrot.tools import dataset_warnings

# The datatypes the command-line tools accept
COMMAND_LINE_DATATYPES = ('sequence', 'residues', 'matrix')

# Square maps take memory in proportion to length^2: a 512-residue map has about 262,000 cells, and the 2D U-Net
# with its default settings keeps roughly 1-3 kB per cell while training (an estimate from its layer sizes), so one
# such sequence needs up to about 1 GB. Longer sequences are refused unless --max-matrix-length is raised.
DEFAULT_MAX_MATRIX_LENGTH = 512

# Ends the help of every flag kept only so that commands written for earlier versions of PARROT still work
LEGACY_NOTE = '(kept so that older commands keep working)'

# Batch sizes above this are very large for the datasets PARROT is used with, and may lower accuracy (rule of thumb)
LARGE_BATCH_SIZE = 512

# Fewer batches than this per epoch give the optimizer few updates per epoch (rule of thumb)
MIN_BATCHES_PER_EPOCH = 10


# .................................................................
# Arguments

def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every command that trains a network

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """

    parser.add_argument('data_file', help='path to tsv file with format: <idx> <sequence> <data>')
    parser.add_argument('output_network', help='location to save the trained network')

    parser.add_argument('-d', '--datatype', required=True, choices=COMMAND_LINE_DATATYPES,
                        help="REQUIRED. Format of the input data file, must be 'sequence' or 'residues'")
    parser.add_argument('-c', '--classes', type=int, metavar='num_classes', required=True,
                        help='REQUIRED. Number of output classes, for regression put 1')
    parser.add_argument('--architecture', default=None, choices=command_line_architectures(),
                        help=f'network architecture (default: {DEFAULT_ARCHITECTURE}, or {DEFAULT_MATRIX_ARCHITECTURE} '
                             'for matrix data)')

    parser.add_argument('-b', '--batch', default=32, type=int, metavar='batch_size',
                        help='size of training batch (def=32)')
    parser.add_argument('--batching', default=pid.DEFAULT_BATCHING, choices=pid.BATCHING_CHOICES,
                        help="how training sequences are grouped into batches: 'length' (default) batches sequences "
                             "of similar length together, so little time is spent on padding; 'random' shuffles "
                             "freely (PARROT's original behaviour)")
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='num_epochs',
                        help='number of training epochs (def=100)')

    parser.add_argument('--split', dest='split_file', default=None, metavar='split_file',
                        help="file indicating how to split datafile into training, validation, and test sets")
    parser.add_argument('--set-fractions', nargs=3, default=pid.DEFAULT_SET_FRACTIONS, type=float,
                        dest='set_fractions', metavar=('train', 'val', 'test'),
                        help='proportion of dataset that should be divided into training, validation, and test sets')
    parser.add_argument('--save-splits', dest='save_splits', action='store_true',
                        help='Save a split-file using the random splits from this run')
    parser.add_argument('--encode', default='onehot', type=str, metavar='encoding_scheme',
                        help="'onehot' (default), 'biophysics', a protein language model (e.g. "
                             "'esm2_t12_35M_UR50D'), or a path to a user-created scheme")
    parser.add_argument('--standardize-encoding', dest='standardize_encoding', action='store_true',
                        help='standardize every feature of the encoding to mean 0 and standard deviation 1 across '
                             'the amino acids (or other characters) it defines, so that features on different '
                             'scales contribute equally; saved with the network and applied automatically when '
                             'predicting')
    add_embedding_cache_argument(parser, plm.TRAIN_CACHE_DIRECTORY)
    add_matrix_arguments(parser)
    parser.add_argument('--exclude-seq-id', dest='exclude_seq_id', action='store_true',
                        help='use if data_file lacks sequence IDs in the first column of each line')
    parser.add_argument('--ignore-warnings', '-w', dest='ignore_warnings', action='store_true',
                        help='Do not display warnings for dataset structure')

    parser.add_argument('--probabilistic-classification', dest='probabilistic_classification',
                        action='store_true',
                        help='sequence classification only: report the probability of each class, not just the '
                             'most likely class')
    parser.add_argument('--include-figs', dest='include_figs', action='store_true',
                        help='Generate figures from training results and save to same location as network')
    parser.add_argument('--no-stats', dest='no_stats', action='store_true',
                        help='If passed, do not output a performance stats file.')

    add_device_arguments(parser)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible data splits and network training (default: not seeded)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Flag which, if provided, causes output to terminal to be more descriptive')
    parser.add_argument('--silent', action='store_true',
                        help="Flag which, if provided, ensures no output is generated to the terminal")


def parse_training_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line of a training command, filling in the default architecture for the datatype

    Parameters
    ----------
    parser : argparse.ArgumentParser
        A parser built with add_training_arguments()

    Returns
    -------
    argparse.Namespace
    """
    args = parser.parse_args()
    if args.architecture is None:
        args.architecture = DEFAULT_MATRIX_ARCHITECTURE if args.datatype == 'matrix' else DEFAULT_ARCHITECTURE
    return args


def add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that describe matrix data (-d matrix; see parrot.matrices)

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    group = parser.add_argument_group('matrix data (-d matrix)')
    group.add_argument('--matrix-file', dest='matrix_file', metavar='file.npz',
                       help="the NumPy .npz file holding each sequence's matrix, named by its sequence ID (the datafile "
                            "then has '<ID> <sequence>' on each line)")
    group.add_argument('--matrix-width', dest='matrix_width', type=int, metavar='N',
                       help='every matrix has N (related, ordered) columns (default: one column per residue, i.e. a '
                            'square map of every pair of residues)')
    group.add_argument('--symmetric', action='store_true',
                       help='square maps only: every map is symmetric, so each pair of residues counts once and the '
                            'network predicts symmetric maps')
    group.add_argument('--max-matrix-length', dest='max_matrix_length', type=int, default=DEFAULT_MAX_MATRIX_LENGTH,
                       help='square maps only: refuse sequences longer than this, as memory grows with length^2 '
                            f'(default {DEFAULT_MAX_MATRIX_LENGTH})')


def check_matrix_arguments(args: argparse.Namespace) -> None:
    """Check the matrix-data arguments: given only for matrix data, and a matrix file that exists for it

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_training_arguments()

    Raises
    ------
    ValueError
        If a matrix-data option is given for other data, or no matrix file for matrix data
    FileNotFoundError
        If the matrix file does not exist
    """
    given = [flag for flag, value in (('--matrix-file', args.matrix_file), ('--matrix-width', args.matrix_width),
                                      ('--symmetric', args.symmetric or None)) if value is not None]
    if args.datatype != 'matrix':
        if given:
            raise ValueError(f'{given[0]} only applies to matrix data (-d matrix)')
        return

    if args.matrix_file is None:
        raise ValueError("Matrix data (-d matrix) needs --matrix-file: the .npz file of each sequence's matrix")
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


def add_embedding_cache_argument(parser: argparse.ArgumentParser, default_name: str) -> None:
    """Add --embedding-cache, which only applies to protein language model encodings

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the argument to
    default_name : str
        Name of the default cache directory, created next to the command's output
    """

    parser.add_argument('--embedding-cache', dest='embedding_cache', metavar='directory',
                        help='protein language model encodings only: directory in which to store the per-residue '
                             f'embeddings (default: a "{default_name}" directory next to the output). Point several '
                             'runs at the same directory to embed each sequence only once')


def prepare_plm_embeddings(encoder: SequenceEncoder, sequences: list[str], cache_directory: str | None,
                           default_directory: str, device: torch.device, program: str, silent: bool) -> None:
    """Embed every sequence a command will encode, if the encoding is a protein language model

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
        If --embedding-cache was given for an encoding that is not a protein language model
    """

    if not isinstance(encoder, plm.PLMEncoder):
        if cache_directory is not None:
            raise ValueError('--embedding-cache only applies to protein language model encodings, not '
                             f"'{encoder.name}'")
        return

    encoder.use_device(device)
    encoder.use_cache(cache_directory if cache_directory is not None else default_directory)
    encoder.prepare(sequences, program, silent=silent)

    # Every sequence is now cached, so the model's memory can be freed
    encoder.release_model()


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that choose the device (shared by every command that runs a network)

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """

    parser.add_argument('--device', default='auto', choices=devices.DEVICE_CHOICES,
                        help="device to run on: 'auto' (default) uses a CUDA GPU if available, else an "
                             "Apple-silicon GPU (MPS) if available, else the CPU")
    parser.add_argument('--gpu-id', dest='gpu_id', type=int,
                        help='index of the CUDA device to use, on machines with more than one NVIDIA GPU')
    parser.add_argument('--force-cpu', dest='force_cpu', action='store_true',
                        help=f'same as --device cpu {LEGACY_NOTE}')


def check_probabilistic_classification(probabilistic: bool, task: Task) -> None:
    """Check that class probabilities were only asked for a task that has them (sequence classification)

    Parameters
    ----------
    probabilistic : bool
        Whether --probabilistic-classification was given
    task : Task

    Raises
    ------
    ValueError
        If probabilities were asked for a regression or residue-mapped task
    """
    if probabilistic and (task.per_residue or not task.is_classification):
        raise ValueError('Probabilistic classification only implemented for sequence classification')


def command_line_architectures() -> list[str]:
    """The architectures the command-line tools offer: those for a datatype the tools accept

    Returns
    -------
    list of str
        Names of every registered architecture that supports one of
        COMMAND_LINE_DATATYPES, sorted
    """
    return [name for name in available_architectures()
            if set(get_architecture(name).supported_datatypes) & set(COMMAND_LINE_DATATYPES)]


def hyperparameters_by_name() -> dict[str, list[tuple[str, HyperParameter]]]:
    """Group the hyperparameters of every architecture the command-line tools offer, by name

    Returns
    -------
    dict
        Maps each hyperparameter name to a list of (architecture name,
        HyperParameter) pairs for the architectures that declare it
    """

    grouped: dict[str, list[tuple[str, HyperParameter]]] = {}
    for architecture in command_line_architectures():
        for hyperparameter in get_architecture(architecture).hyperparameters:
            grouped.setdefault(hyperparameter.name, []).append((architecture, hyperparameter))
    return grouped


def add_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one command-line flag for each architecture hyperparameter

    Flags are generated from the hyperparameters every registered architecture
    declares (e.g. --hidden-size/-hs for the BRNN). Architectures that share a
    hyperparameter name share the flag, and its help lists each architecture's
    description and default. An on/off hyperparameter's flag takes no value and
    switches it away from its default (e.g. --attention-pooling-off). Flags
    default to None so that given_hyperparameters() can tell which ones the
    user actually set.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to

    Raises
    ------
    ValueError
        If architectures declare hyperparameters with the same name that cannot
        share a flag (see check_shared_declarations())
    """

    group = parser.add_argument_group('architecture hyperparameters')
    for name, declarations in hyperparameters_by_name().items():
        check_shared_declarations(name, [hyperparameter for _, hyperparameter in declarations])
        first = declarations[0][1]

        # An on/off hyperparameter is a flag that switches it away from its default (e.g. --attention-pooling-off)
        if first.value_type is bool:
            architectures = ', '.join(architecture for architecture, _ in declarations)
            switch = 'turns it off (on by default)' if first.default else 'turns it on (off by default)'
            group.add_argument(first.cli_flag, *first.cli_aliases, dest=name, action='store_const',
                               const=not first.default, default=None,
                               help=f'{architectures}: {first.help}; this flag {switch}')
            continue

        # Restrict the flag to the declared choices only if every architecture declares some;
        # each architecture's own choices are checked again when the network is built
        declared_choices = [hyperparameter.choices for _, hyperparameter in declarations]
        choices = None
        if all(options is not None for options in declared_choices):
            choices = list(dict.fromkeys(choice for options in declared_choices if options for choice in options))

        help_text = '; '.join(f'{architecture}: {hyperparameter.help} (default {hyperparameter.default})'
                              for architecture, hyperparameter in declarations)
        group.add_argument(first.cli_flag, *first.cli_aliases, dest=name, type=first.value_type, default=None,
                           choices=choices, metavar=name, help=help_text)


def check_shared_declarations(name: str, declarations: list[HyperParameter]) -> None:
    """Check that the architectures declaring a hyperparameter name can share one command-line flag

    Parameters
    ----------
    name : str
        The hyperparameter's name
    declarations : list of HyperParameter
        Every architecture's declaration of it

    Raises
    ------
    ValueError
        If the declarations differ in type or in command-line aliases, or (for
        on/off hyperparameters, whose flag switches away from the default) in
        default
    """
    first = declarations[0]
    if any(hyperparameter.value_type is not first.value_type for hyperparameter in declarations):
        raise ValueError(f"Architectures declare hyperparameter '{name}' with different types")
    if any(hyperparameter.cli_aliases != first.cli_aliases for hyperparameter in declarations):
        raise ValueError(f"Architectures declare hyperparameter '{name}' with different command-line aliases")
    if first.value_type is bool and any(hyperparameter.default != first.default for hyperparameter in declarations):
        raise ValueError(f"Architectures declare on/off hyperparameter '{name}' with different defaults")


def given_hyperparameters(args: argparse.Namespace) -> dict[str, HyperParameterValue]:
    """Collect the architecture hyperparameters the user set on the command line

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_training_arguments() and
        add_hyperparameter_arguments()

    Returns
    -------
    dict
        Validated values of only those hyperparameters of args.architecture
        that were given on the command line

    Raises
    ------
    ValueError
        If a hyperparameter flag was given that the chosen architecture does
        not have, or a value is invalid
    """

    declared = {hyperparameter.name: hyperparameter
                for hyperparameter in get_architecture(args.architecture).hyperparameters}

    given = {}
    for name, declarations in hyperparameters_by_name().items():
        value = getattr(args, name)
        if value is None:
            continue
        if name not in declared:
            raise ValueError(f"{declarations[0][1].cli_flag} is not a hyperparameter of the "
                             f"'{args.architecture}' architecture")
        given[name] = declared[name].validate(value)

    return given


def hyperparameters_from_args(args: argparse.Namespace) -> dict[str, HyperParameterValue]:
    """Every hyperparameter of the chosen architecture: the values given on the command line, else defaults

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_training_arguments() and
        add_hyperparameter_arguments()

    Returns
    -------
    dict
        A validated value for every hyperparameter of args.architecture

    Raises
    ------
    ValueError
        If a hyperparameter flag was given that the chosen architecture does
        not have, or a value is invalid
    """

    return resolve_hyperparameters(get_architecture(args.architecture), given_hyperparameters(args))


def select_device(args: argparse.Namespace) -> torch.device:
    """Choose the device from the --device, --force-cpu and --gpu-id arguments

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
        If --force-cpu is combined with a --device other than cpu or auto, or
        the arguments are otherwise inconsistent (see devices.resolve_device)
    RuntimeError
        If the requested device is not available
    """

    requested = args.device
    if args.force_cpu:
        if requested not in ('auto', 'cpu'):
            raise ValueError(f"--force-cpu cannot be combined with --device {requested}")
        requested = 'cpu'

    device = devices.resolve_device(requested, args.gpu_id)
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
    """

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
        dataset_warnings.print_warning('WARNING: Using a very large batch size.',
                                       ['', 'This may decrease model accuracy.', ''])

    if num_training_sequences // batch_size <= MIN_BATCHES_PER_EPOCH:
        dataset_warnings.print_warning(
            'WARNING: Batch size is large relative to the number of samples in the training set.',
            ['', 'This may decrease training efficiency.', ''])


def prepare_training_run(args: argparse.Namespace, n_folds: int | None = None) -> TrainingRun:
    """Validate the shared arguments, then load and split the data

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser built with add_training_arguments()
    n_folds : int, optional
        If given, also divide the training + validation data into this many
        cross-validation folds

    Returns
    -------
    TrainingRun

    Raises
    ------
    FileNotFoundError
        If the datafile, split-file, encoding file or output directory does not exist
    ValueError
        If any argument is invalid, or the datafile does not match the task
    """

    set_random_seed(args.seed)

    data_file = arguments.check_file_exists(args.data_file, 'Datafile')
    network_file = os.path.abspath(args.output_network)
    output_prefix, _ = arguments.output_prefix_and_directory(network_file)

    split_file = None
    if args.split_file is not None:
        split_file = arguments.check_file_exists(args.split_file, 'Split-file')

    save_splits_output = None
    if args.save_splits:
        save_splits_output = output_prefix + '_split_file.txt'

    encoder = get_encoder(args.encode, standardize=args.standardize_encoding)
    check_matrix_arguments(args)
    task = Task(args.datatype, args.classes, width=args.matrix_width, symmetric=args.symmetric)

    arguments.check_positive(args.epochs, 'Number of epochs')
    arguments.check_positive(args.batch, 'Batch size')
    arguments.check_set_fractions(args.set_fractions)
    check_probabilistic_classification(args.probabilistic_classification, task)

    ignore_warnings = args.ignore_warnings or args.silent
    device = select_device(args)

    split_arguments: dict[str, Any] = dict(encoder=encoder, exclude_seq_id=args.exclude_seq_id, split_file=split_file,
                                           percent_val=args.set_fractions[1], percent_test=args.set_fractions[2],
                                           ignore_warnings=ignore_warnings, save_splits_output=save_splits_output,
                                           matrix_file=args.matrix_file)
    cv_sets = None
    if n_folds is None:
        train_set, val_set, test_set = pid.split_data(data_file, task, **split_arguments)
    else:
        cv_sets, train_set, val_set, test_set = pid.split_data_cv(data_file, task, n_folds=n_folds, **split_arguments)

    if not ignore_warnings:
        warn_about_batch_size(args.batch, len(cv_sets[0][0]) if cv_sets else len(train_set))

    # Check every sequence now, so a bad residue in (say) the test set fails before training rather than after
    every_sequence = train_set.sequences + val_set.sequences + test_set.sequences
    encoder.check_sequences(every_sequence)
    if task.is_square:
        check_matrix_lengths(every_sequence, args.max_matrix_length)
    prepare_plm_embeddings(encoder, every_sequence, args.embedding_cache,
                           os.path.join(os.path.dirname(network_file), plm.TRAIN_CACHE_DIRECTORY), device,
                           os.path.basename(sys.argv[0]), args.silent)

    return TrainingRun(task=task, encoder=encoder, device=device, network_file=network_file,
                       output_prefix=output_prefix, train_set=train_set, val_set=val_set, test_set=test_set,
                       cv_sets=cv_sets, batch_size=args.batch, batching=args.batching,
                       probabilistic=args.probabilistic_classification,
                       include_figs=args.include_figs, write_stats=not args.no_stats,
                       verbose=args.verbose, silent=args.silent)


def print_run_header(title: str, run: TrainingRun, settings: dict[str, Any]) -> None:
    """Print the banner shown at the start of training (and, if verbose, the settings)

    Parameters
    ----------
    title : str
        First line of the banner
    run : TrainingRun
        The run being started
    settings : dict
        Settings to list in verbose mode, in addition to the device and task
    """

    if run.silent:
        return

    print()
    print(title)
    print('-' * len(title))
    if run.verbose:
        arguments.print_settings(False, device=run.device, datatype=run.task.datatype, task=run.task.problem_type,
                           encoding=run.encoder.name, **settings)
        print()


def finish_training_run(run: TrainingRun, network: ParrotArchitecture, train_losses: list[float],
                        val_losses: list[float]) -> None:
    """Save the trained network, then evaluate and report on the test set

    Parameters
    ----------
    run : TrainingRun
        The run being finished
    network : ParrotArchitecture
        The trained network (holding its best weights)
    train_losses, val_losses : list of float
        Loss per epoch, as returned by train_network.train()
    """

    run.save(network)

    # Models fitted in one step have no loss curve to plot
    if run.include_figs and network.trained_by_gradient_descent:
        figures.plot_training_loss(train_losses, val_losses, run.output_prefix + '_train_val_loss.png')

    labeled = prediction.predict_labeled(network, run.test_set, run.task, run.device)
    if not run.silent:
        print('\nTest Loss: %.4f' % labeled.loss)

    reporting.report_test_results(labeled, run.task, run.output_prefix, probabilistic=run.probabilistic,
                                  include_figs=run.include_figs, write_stats=run.write_stats)
