"""
The command line of parrot-train and parrot-optimize: one subcommand per
architecture, each offering only the options that apply to it.

    parrot-train cnn data.tsv network.pt -d residues -c 1 --channels 32

Every registered architecture gets a subcommand named after it. Its options are
assembled from sets of options, each defined once below, which a subcommand
inherits according to the kind of architecture it is (argparse calls these
"parent parsers"):

- every architecture: the data and how it is split, its encoding, the output,
  the device and the random seed;
- networks trained by gradient descent: the training loop (epochs, batch size
  and batching), plus what the command adds to it (parrot-train: the learning
  rate and stop condition; parrot-optimize: the successive-halving settings);
- architectures for matrix data: the matrix file and the shape of its matrices;
- architectures for sequence data: reporting class probabilities;
- the architecture's own hyperparameters, one flag each, from its declarations.

The parsed arguments of every subcommand have the same names: an option a
subcommand does not offer takes its default. The code that runs a command
therefore never needs to know which subcommand was used. (For example, models
fitted in one step have no training loop, so they do not offer --epochs; their
arguments carry its default, which they never use.)

Commands written before subcommands existed still work: with no subcommand,
the architecture is the one given by --architecture, or the BRNN (see
with_architecture_first()).

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
from collections.abc import Collection, Sequence
from typing import Any

from parrot import plm
from parrot import process_input_data as pid
from parrot.architectures import (DEFAULT_ARCHITECTURE, HyperParameter, HyperParameterValue, ParrotArchitecture,
                                  available_architectures, get_architecture, resolve_hyperparameters)
from parrot.scripts.common import DEFAULT_MAX_MATRIX_LENGTH, add_device_arguments, add_embedding_cache_argument

# What each datatype means, for the help of -d
DATATYPE_DESCRIPTIONS = {
    'sequence': 'one value per sequence',
    'residues': 'one value per residue',
    'matrix': 'a matrix of values per sequence, read from --matrix-file',
}

# Default size of a training batch, and number of training epochs
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# The flag that chose the architecture before each architecture had its own subcommand
LEGACY_ARCHITECTURE_FLAG = '--architecture'

# Flags asking for help, which may come before any subcommand
HELP_FLAGS = ('-h', '--help')


# .................................................................
# The parser

def build_parser(description: str, training_loop_options: Sequence[argparse.ArgumentParser] = (),
                 command_options: Sequence[argparse.ArgumentParser] = (),
                 describe_search: bool = False) -> argparse.ArgumentParser:
    """The command line of a training command, with one subcommand per registered architecture

    Parameters
    ----------
    description : str
        What the command does, shown in its help and in each subcommand's
    training_loop_options : list of argparse.ArgumentParser, optional
        The command's own options for networks trained by gradient descent,
        offered alongside the shared training-loop options (each built with
        add_help=False)
    command_options : list of argparse.ArgumentParser, optional
        The command's own options for every architecture (each built with
        add_help=False)
    describe_search : bool, optional
        Say in the help of each hyperparameter whether, and over what values,
        parrot-optimize searches it

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest='architecture', required=True, metavar='architecture',
                                     title='architectures',
                                     description=f"Run '{parser.prog} <architecture> -h' to see an architecture's "
                                                 'options. In brackets: the data each one can learn.')
    for architecture in architectures_in_listing_order():
        add_architecture_command(commands, architecture, description, training_loop_options, command_options,
                                 describe_search)
    return parser


def architectures_in_listing_order() -> list[type[ParrotArchitecture]]:
    """Every registered architecture, listed by kind and then alphabetically

    Neural networks for sequence and residue data come first, then models fitted
    in one step, then architectures for matrix data.

    Returns
    -------
    list of ParrotArchitecture subclasses
    """
    def kind(architecture: type[ParrotArchitecture]) -> int:
        if 'matrix' in architecture.supported_datatypes:
            return 2
        if not architecture.trained_by_gradient_descent:
            return 1
        return 0

    architectures = [get_architecture(name) for name in available_architectures()]
    return sorted(architectures, key=lambda architecture: (kind(architecture), architecture.name))


def add_architecture_command(commands: 'argparse._SubParsersAction[argparse.ArgumentParser]',
                             architecture: type[ParrotArchitecture], description: str,
                             training_loop_options: Sequence[argparse.ArgumentParser],
                             command_options: Sequence[argparse.ArgumentParser], describe_search: bool) -> None:
    """Add the subcommand of one architecture, inheriting the option sets that apply to it

    Parameters
    ----------
    commands : argparse subparsers
        The command's subcommands (from add_subparsers())
    architecture : ParrotArchitecture subclass
    description : str
        What the command does
    training_loop_options, command_options : list of argparse.ArgumentParser
        The command's own options (see build_parser())
    describe_search : bool
        Say in each hyperparameter's help whether parrot-optimize searches it
    """
    trained_by_gradient_descent = architecture.trained_by_gradient_descent
    learns_matrices = 'matrix' in architecture.supported_datatypes
    learns_sequence_values = 'sequence' in architecture.supported_datatypes

    # Every option set, in the order the help lists them, and whether this architecture offers it
    option_sets = [
        (True, datatype_options(architecture)),
        (True, data_options()),
        (trained_by_gradient_descent, shared_training_loop_options()),
        *[(trained_by_gradient_descent, options) for options in training_loop_options],
        (True, hyperparameter_options(architecture, describe_search)),
        (learns_matrices, matrix_options()),
        *[(True, options) for options in command_options],
        (learns_sequence_values, class_probability_options()),
        (True, output_options()),
    ]

    command = commands.add_parser(architecture.name, parents=[options for offered, options in option_sets if offered],
                                  help=f'{architecture.description} [{", ".join(architecture.supported_datatypes)}]',
                                  description=f'{description.removesuffix(".")} ({architecture.name}: '
                                              f'{architecture.description}).',
                                  usage=command_usage(architecture))

    # Options this architecture does not offer take their defaults, so every subcommand's arguments have the same names
    for offered, options in option_sets:
        if not offered:
            command.set_defaults(**option_defaults(options))


def command_usage(architecture: type[ParrotArchitecture]) -> str:
    """The usage line of an architecture's subcommand: its required arguments, then '[options]'

    argparse would list every option in the usage line; listing only the
    required ones keeps it readable.

    Parameters
    ----------
    architecture : ParrotArchitecture subclass

    Returns
    -------
    str
    """
    required = ['data_file', 'output_network']
    if len(architecture.supported_datatypes) > 1:
        required.append('-d {' + ','.join(architecture.supported_datatypes) + '}')
    required.append('-c num_classes')
    if 'matrix' in architecture.supported_datatypes:
        required.append('--matrix-file file.npz')
    return '%(prog)s ' + ' '.join(required) + ' [options]'


def option_defaults(options: argparse.ArgumentParser) -> dict[str, Any]:
    """The value each option of a set takes when it is not given

    Parameters
    ----------
    options : argparse.ArgumentParser
        An option set with no required arguments

    Returns
    -------
    dict
        Argument name -> default value
    """
    return vars(options.parse_args([]))


# .................................................................
# The option sets

def datatype_options(architecture: type[ParrotArchitecture]) -> argparse.ArgumentParser:
    """-d and -c: the datatype (inferred when the architecture supports only one) and the number of classes

    Parameters
    ----------
    architecture : ParrotArchitecture subclass

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('data')

    datatypes = architecture.supported_datatypes
    if len(datatypes) == 1:
        only = datatypes[0]
        group.add_argument('-d', '--datatype', choices=datatypes, default=only,
                           help=f"'{only}' ({DATATYPE_DESCRIPTIONS[only]}), the only datatype {architecture.name} "
                                'supports, so this can be left out')
    else:
        meanings = [f"'{datatype}' ({DATATYPE_DESCRIPTIONS[datatype]})" for datatype in datatypes]
        group.add_argument('-d', '--datatype', choices=datatypes, required=True,
                           help=f'REQUIRED. Format of the datafile: {", ".join(meanings[:-1])} or {meanings[-1]}')

    group.add_argument('-c', '--classes', type=int, metavar='num_classes', required=True,
                       help='REQUIRED. Number of output classes, for regression put 1')
    return options


def data_options() -> argparse.ArgumentParser:
    """The datafile and output network, how the data are split, and how sequences are encoded (every architecture)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('data_file', help="the datafile: '<ID> <sequence> <value(s)>' on each line (for matrix "
                                           "data, '<ID> <sequence>')")
    options.add_argument('output_network', help='location to save the trained network')

    group = options.add_argument_group('data')
    group.add_argument('--split', dest='split_file', default=None, metavar='split_file',
                       help='file indicating how to split datafile into training, validation, and test sets')
    group.add_argument('--set-fractions', nargs=3, default=pid.DEFAULT_SET_FRACTIONS, type=float,
                       dest='set_fractions', metavar=('train', 'val', 'test'),
                       help='proportion of dataset that should be divided into training, validation, and test sets')
    group.add_argument('--save-splits', dest='save_splits', action='store_true',
                       help='Save a split-file using the random splits from this run')
    group.add_argument('--exclude-seq-id', dest='exclude_seq_id', action='store_true',
                       help='use if data_file lacks sequence IDs in the first column of each line')
    group.add_argument('--ignore-warnings', '-w', dest='ignore_warnings', action='store_true',
                       help='Do not display warnings for dataset structure')

    group = options.add_argument_group('encoding')
    group.add_argument('--encode', default='onehot', type=str, metavar='encoding_scheme',
                       help="'onehot' (default), 'biophysics', a protein language model (e.g. "
                            "'esm2_t12_35M_UR50D'), or a path to a user-created scheme")
    group.add_argument('--standardize-encoding', dest='standardize_encoding', action='store_true',
                       help='standardize every feature of the encoding to mean 0 and standard deviation 1 across '
                            'the amino acids (or other characters) it defines, so that features on different '
                            'scales contribute equally; saved with the network and applied automatically when '
                            'predicting')
    add_embedding_cache_argument(group, plm.TRAIN_CACHE_DIRECTORY)
    return options


def shared_training_loop_options() -> argparse.ArgumentParser:
    """Epochs, batch size and batching (networks trained by gradient descent)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('training')
    group.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int, metavar='num_epochs',
                       help=f'number of training epochs (def={DEFAULT_EPOCHS})')
    group.add_argument('-b', '--batch', default=DEFAULT_BATCH_SIZE, type=int, metavar='batch_size',
                       help=f'size of training batch (def={DEFAULT_BATCH_SIZE})')
    group.add_argument('--batching', default=pid.DEFAULT_BATCHING, choices=pid.BATCHING_CHOICES,
                       help="how training sequences are grouped into batches: 'length' (default) batches sequences "
                            "of similar length together, so little time is spent on padding; 'random' shuffles "
                            "freely (PARROT's original behaviour)")
    return options


def hyperparameter_options(architecture: type[ParrotArchitecture], describe_search: bool) -> argparse.ArgumentParser:
    """One flag for each of an architecture's hyperparameters, generated from its declarations

    An on/off hyperparameter's flag takes no value and switches it away from its
    default (e.g. --attention-pooling-off). Every flag defaults to None, so
    given_hyperparameters() can tell which ones were given.

    Parameters
    ----------
    architecture : ParrotArchitecture subclass
    describe_search : bool
        Say in each flag's help whether, and over what values, parrot-optimize searches it

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group(
        f'{architecture.name} hyperparameters',
        description='Any hyperparameter given here is fixed at that value; the search tunes the others it '
                    'searches.' if describe_search else None)

    for hyperparameter in architecture.hyperparameters:
        help_text = hyperparameter_help(hyperparameter, describe_search)
        if hyperparameter.value_type is bool:
            group.add_argument(hyperparameter.cli_flag, *hyperparameter.cli_aliases, dest=hyperparameter.name,
                               action='store_const', const=not hyperparameter.default, default=None, help=help_text)
        else:
            group.add_argument(hyperparameter.cli_flag, *hyperparameter.cli_aliases, dest=hyperparameter.name,
                               type=hyperparameter.value_type, choices=hyperparameter.choices, default=None,
                               metavar=hyperparameter.name, help=help_text)
    return options


def hyperparameter_help(hyperparameter: HyperParameter, describe_search: bool) -> str:
    """The help of a hyperparameter's flag: what it is, its default, and (if asked) how it is searched

    Parameters
    ----------
    hyperparameter : HyperParameter
    describe_search : bool

    Returns
    -------
    str
    """
    if hyperparameter.value_type is bool:
        default_state = 'on' if hyperparameter.default else 'off'
        details = [f'{default_state} by default; this flag turns it {"off" if hyperparameter.default else "on"}']
    else:
        details = [f'default {hyperparameter.default}']

    if describe_search:
        details.append(search_description(hyperparameter))
    return f'{hyperparameter.help} ({"; ".join(details)})'


def search_description(hyperparameter: HyperParameter) -> str:
    """Whether, and over what values, parrot-optimize searches a hyperparameter

    Parameters
    ----------
    hyperparameter : HyperParameter

    Returns
    -------
    str
        e.g. 'searched over 5-50', 'searched over 0.001-1000 on a log scale',
        'searched over rbf, linear' or 'not searched'
    """
    if hyperparameter.search_range is not None:
        low, high = hyperparameter.search_range
        scale = ' on a log scale' if hyperparameter.log_scale else ''
        return f'searched over {low:g}-{high:g}{scale}'
    if hyperparameter.search_choices is not None:
        return 'searched over ' + ', '.join(str(choice) for choice in hyperparameter.search_choices)
    return 'not searched'


def matrix_options() -> argparse.ArgumentParser:
    """The matrix file and the shape of its matrices (architectures for matrix data; see parrot.matrices)

    --matrix-file is required, but is checked by common.check_matrix_arguments()
    rather than by argparse, so that this set has a default for every option
    (see option_defaults()).

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('matrix data')
    group.add_argument('--matrix-file', dest='matrix_file', metavar='file.npz',
                       help="REQUIRED. The NumPy .npz file holding each sequence's matrix, named by its sequence ID "
                            "(the datafile then has '<ID> <sequence>' on each line)")
    group.add_argument('--matrix-width', dest='matrix_width', type=int, metavar='N',
                       help='every matrix has N (related, ordered) columns (default: one column per residue, i.e. a '
                            'square map of every pair of residues)')
    group.add_argument('--symmetric', action='store_true',
                       help='square maps only: every map is symmetric, so each pair of residues counts once and the '
                            'network predicts symmetric maps')
    group.add_argument('--max-matrix-length', dest='max_matrix_length', type=int, default=DEFAULT_MAX_MATRIX_LENGTH,
                       help='square maps only: refuse sequences longer than this, as memory grows with length^2 '
                            f'(default {DEFAULT_MAX_MATRIX_LENGTH})')
    return options


def class_probability_options() -> argparse.ArgumentParser:
    """--probabilistic-classification (architectures for sequence data, the only data it applies to)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('output')
    group.add_argument('--probabilistic-classification', dest='probabilistic_classification', action='store_true',
                       help='sequence classification only: report the probability of each class, not just the '
                            'most likely class')
    return options


def output_options() -> argparse.ArgumentParser:
    """What is written and printed, the device, and the random seed (every architecture)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('output')
    group.add_argument('--include-figs', dest='include_figs', action='store_true',
                       help='Generate figures from training results and save to same location as network')
    group.add_argument('--no-stats', dest='no_stats', action='store_true',
                       help='If passed, do not output a performance stats file.')
    group.add_argument('--verbose', '-v', action='store_true',
                       help='Flag which, if provided, causes output to terminal to be more descriptive')
    group.add_argument('--silent', action='store_true',
                       help='Flag which, if provided, ensures no output is generated to the terminal')

    group = options.add_argument_group('device and reproducibility')
    add_device_arguments(group)
    group.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible data splits and network training (default: not seeded)')
    return options


# .................................................................
# Parsing

def parse_arguments(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a training command's arguments, accepting commands written before subcommands existed

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Built by build_parser()
    argv : list of str, optional
        The arguments after the command's name (default: sys.argv[1:])

    Returns
    -------
    argparse.Namespace
        The arguments; `architecture` is the subcommand used
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    architectures = available_architectures()

    # A command with no subcommand starts with an option or its datafile, which must exist; anything else is most
    # likely a misspelt architecture, which would otherwise be read as a datafile and give a confusing error
    first = arguments[0] if arguments else ''
    if first and not first.startswith('-') and first not in architectures and not os.path.exists(first):
        parser.error(f"'{first}' is neither an architecture nor an existing datafile. The architectures are: "
                     f"{', '.join(architectures)}")

    return parser.parse_args(with_architecture_first(arguments, architectures))


def with_architecture_first(arguments: list[str], architectures: Collection[str]) -> list[str]:
    """Rewrite a command written before subcommands existed so that it starts with its architecture

    Before each architecture had its own subcommand, commands looked like
    `parrot-train data.tsv net.pt -d sequence -c 1`, trained a BRNN, and chose
    another architecture with `--architecture cnn` anywhere in the command.
    Such commands are rewritten as `parrot-train brnn data.tsv net.pt ...` (or
    `cnn`). Commands that start with a subcommand, or that only ask for help,
    are left as they are.

    Parameters
    ----------
    arguments : list of str
        The arguments after the command's name
    architectures : collection of str
        The subcommands (registered architecture names)

    Returns
    -------
    list of str
    """
    if not arguments or arguments[0] in architectures or arguments[0] in HELP_FLAGS:
        return arguments

    architecture = DEFAULT_ARCHITECTURE
    remaining = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        # '--architecture cnn' (a trailing '--architecture' with no value is kept, for argparse to reject)
        if argument == LEGACY_ARCHITECTURE_FLAG and index + 1 < len(arguments):
            architecture = arguments[index + 1]
            index += 2
            continue
        # '--architecture=cnn'
        if argument.startswith(LEGACY_ARCHITECTURE_FLAG + '='):
            architecture = argument.split('=', 1)[1]
        else:
            remaining.append(argument)
        index += 1
    return [architecture, *remaining]


def given_hyperparameters(args: argparse.Namespace) -> dict[str, HyperParameterValue]:
    """The hyperparameters of the chosen architecture that were given on the command line, checked

    Parameters
    ----------
    args : argparse.Namespace
        Parsed by parse_arguments()

    Returns
    -------
    dict
        Name -> validated value, for only the hyperparameters that were given

    Raises
    ------
    ValueError
        If a value is not allowed (e.g. out of range)
    """
    given = {}
    for hyperparameter in get_architecture(args.architecture).hyperparameters:
        value = getattr(args, hyperparameter.name)
        if value is not None:
            given[hyperparameter.name] = hyperparameter.validate(value)
    return given


def hyperparameters_from_args(args: argparse.Namespace) -> dict[str, HyperParameterValue]:
    """Every hyperparameter of the chosen architecture: the values given on the command line, else defaults

    Parameters
    ----------
    args : argparse.Namespace
        Parsed by parse_arguments()

    Returns
    -------
    dict
        A validated value for every hyperparameter of args.architecture

    Raises
    ------
    ValueError
        If a value is not allowed
    """
    return resolve_hyperparameters(get_architecture(args.architecture), given_hyperparameters(args))
