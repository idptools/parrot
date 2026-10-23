"""
The command line of parrot-train and parrot-optimize: one subcommand per
architecture, each offering only the options that apply to it.

    parrot-train cnn --input-file data.tsv --output-network network.pt --split kmer -d residues --regression
        --channels 32

Every registered architecture gets a subcommand named after it. Its options are
assembled from sets of options, each defined once below, which a subcommand
inherits according to the kind of architecture it is (argparse calls these
"parent parsers"):

- every architecture: the data and how it is split, its encoding, the output,
  the device and the random seed, plus any options the command adds
  (parrot-optimize: the search's own options);
- networks trained by gradient descent: the training loop (epochs, batch size
  and batching), plus what the command adds to it (parrot-train: the learning
  rate and stop condition; parrot-optimize: the successive-halving settings);
- architectures for matrix data: the matrix file, the shape of its matrices
  and the longest sequence allowed;
- architectures for sequence or residue data: reporting class probabilities,
  and (for those that support it) training on values with uncertainties;
- the architecture's own hyperparameters, one flag each, from its declarations.

Apart from each architecture's own hyperparameters, the parsed arguments of
every subcommand have the same names: an option a subcommand does not offer
takes its default. The code that runs a command therefore never needs to know
which subcommand was used. (For example, models fitted in one step have no
training loop, so they do not offer --epochs; their arguments carry its
default, which they never use.)

Every command names its architecture, input and output, split and task
explicitly:

    parrot-train <architecture> --input-file <datafile> --output-network <name>
        --split <method or split-file> [-d <datatype>]
        (--regression | --classification --num-classes N)

Options PARROT no longer has (-c/--classes, --force-cpu, --architecture) are
recognized, and stop the command with a message saying what replaced them.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import math
import os
import sys
from collections.abc import Sequence
from typing import Any

from parrot import plm, train_network
from parrot import process_input_data as pid
from parrot.architectures import (HyperParameter, HyperParameterValue, ParrotArchitecture, available_architectures,
                                  get_architecture, resolve_hyperparameters)
from parrot.scripts.arguments import EXAMPLES_FLAG, RetiredOption
from parrot.scripts.common import (DEFAULT_MAX_MATRIX_LENGTH, add_device_arguments, add_embedding_cache_argument,
                                   add_embed_mean_subtraction_argument, add_embedding_context_arguments,
                                   add_embedding_layer_argument, add_nardini_scrambles_argument)
from parrot.scripts.data_split import SPLIT_METHODS
from parrot.scripts.examples import add_examples_argument
from parrot.training_control import LR_SCHEDULES, TrainingOptions
from parrot.training_precision import AMP_MODES, validate_amp_architecture

# What each datatype means, for the help of -d
DATATYPE_DESCRIPTIONS = {
    'sequence': 'one value per sequence',
    'residues': 'one value per residue',
    'matrix': 'a matrix of values per sequence, read from --matrix-file',
}

# Default size of a training batch, and number of training epochs
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# Options asking for help or examples, which may come before any subcommand
HELP_FLAGS = ('-h', '--help', EXAMPLES_FLAG)


# .................................................................
# The parser

def build_parser(program_name: str, description: str, training_loop_options: Sequence[argparse.ArgumentParser] = (),
                 command_options: Sequence[argparse.ArgumentParser] = (),
                 describe_search: bool = False) -> argparse.ArgumentParser:
    """The command line of a training command, with one subcommand per registered architecture

    Parameters
    ----------
    program_name : str
        The command's name, e.g. 'parrot-train'
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
    parser = argparse.ArgumentParser(prog=program_name, description=description)
    add_examples_argument(parser, program_name)
    commands = parser.add_subparsers(dest='architecture', required=True, metavar='architecture',
                                     title='architectures',
                                     description=f"Run '{parser.prog} <architecture> -h' to see an architecture's "
                                                 'options. In brackets: the data each one can learn.')
    for architecture in architectures_in_listing_order():
        add_architecture_command(commands, architecture, program_name, description, training_loop_options,
                                 command_options, describe_search)
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
        """0 for neural networks for sequence and residue data, 1 for models fitted in one step, 2 for matrix data."""
        if 'matrix' in architecture.supported_datatypes:
            return 2
        if not architecture.trained_by_gradient_descent:
            return 1
        return 0

    architectures = [get_architecture(name) for name in available_architectures()]
    return sorted(architectures, key=lambda architecture: (kind(architecture), architecture.name))


def add_architecture_command(commands: 'argparse._SubParsersAction[argparse.ArgumentParser]',
                             architecture: type[ParrotArchitecture], program_name: str, description: str,
                             training_loop_options: Sequence[argparse.ArgumentParser],
                             command_options: Sequence[argparse.ArgumentParser], describe_search: bool) -> None:
    """Add the subcommand of one architecture, inheriting the option sets that apply to it

    Parameters
    ----------
    commands : argparse subparsers
        The command's subcommands (from add_subparsers())
    architecture : ParrotArchitecture subclass
    program_name : str
        The command's name, e.g. 'parrot-train'
    description : str
        What the command does
    training_loop_options, command_options : list of argparse.ArgumentParser
        The command's own options (see build_parser())
    describe_search : bool
        Say in each hyperparameter's help whether parrot-optimize searches it
    """
    trained_by_gradient_descent = architecture.trained_by_gradient_descent
    learns_matrices = 'matrix' in architecture.supported_datatypes
    learns_classes_of_sequences_or_residues = bool({'sequence', 'residues'} & set(architecture.supported_datatypes))
    learns_values_with_uncertainties = learns_classes_of_sequences_or_residues and architecture.supports_uncertainties

    # Every option set, in the order the help lists them, and whether this architecture offers it
    option_sets = [
        (True, required_options(architecture)),
        (True, data_options(architecture)),
        (learns_values_with_uncertainties, uncertainty_options()),
        (trained_by_gradient_descent, shared_training_loop_options()),
        *[(trained_by_gradient_descent, options) for options in training_loop_options],
        (True, hyperparameter_options(architecture, describe_search)),
        (learns_matrices, matrix_options()),
        *[(True, options) for options in command_options],
        (learns_classes_of_sequences_or_residues, class_probability_options()),
        (True, output_options()),
        (True, retired_options()),
    ]

    command = commands.add_parser(architecture.name, parents=[options for offered, options in option_sets if offered],
                                  help=f'{architecture.description} [{", ".join(architecture.supported_datatypes)}]',
                                  description=f'{description.removesuffix(".")} ({architecture.name}: '
                                              f'{architecture.description}).',
                                  usage=command_usage(architecture))
    add_examples_argument(command, program_name)

    # Options this architecture does not offer take their defaults, so every subcommand's arguments have the same names
    # (apart from each architecture's own hyperparameters)
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
    required = ['--input-file file', '--output-network name', '--split {' + ','.join(SPLIT_METHODS) + ',file}']
    if len(architecture.supported_datatypes) > 1:
        required.append('-d {' + ','.join(architecture.supported_datatypes) + '}')
    required.append('(--regression | --classification --num-classes N)')
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

def required_options(architecture: type[ParrotArchitecture]) -> argparse.ArgumentParser:
    """The options every command must give: the datafile, the network, the split, the datatype (if there is a choice) and the task

    Parameters
    ----------
    architecture : ParrotArchitecture subclass

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('required')
    group.add_argument('--input-file', dest='input_file', required=True, metavar='file',
                       help="the datafile: '<ID> <sequence> <value(s)>' on each line (for matrix data, "
                            "'<ID> <sequence>')")
    group.add_argument('--output-network', dest='output_network', required=True, metavar='name',
                       help="where to save the trained network ('.pt' is added if the name does not end with it); "
                            'every other output is named after it, including <name>_info.txt, which describes how '
                            'the network was trained')
    group.add_argument('--split', dest='split', required=True, metavar='{' + ','.join(SPLIT_METHODS) + ',file}',
                       help='how to divide the data into training, validation and test sets: a split-file (e.g. '
                            'from parrot-preprocess --split; a file of this name is always read first, even one '
                            "called 'random'), or a method, as parrot-preprocess --split does it: 'mmseqs' keeps "
                            "homologous sequences (found by MMseqs2, installed separately) in the same set, 'kmer' "
                            "keeps similar sequences (found by PARROT's k-mer search) in the same set, and "
                            "'random' assigns sequences at random, keeping identical ones together. A method aims "
                            'for the fractions given by --set-fractions')

    datatypes = architecture.supported_datatypes
    if len(datatypes) > 1:
        meanings = [f"'{datatype}' ({DATATYPE_DESCRIPTIONS[datatype]})" for datatype in datatypes]
        group.add_argument('-d', '--datatype', choices=datatypes, required=True,
                           help=f'format of the datafile: {", ".join(meanings[:-1])} or {meanings[-1]}')

    task = group.add_mutually_exclusive_group(required=True)
    task.add_argument('--regression', action='store_true', help='the values to predict are numbers')
    task.add_argument('--classification', action='store_true',
                      help='the values to predict are classes, labelled 0, 1, ... (give --num-classes)')
    group.add_argument('--num-classes', dest='num_classes', type=int, metavar='N',
                       help='with --classification: the number of classes (at least 2)')
    return options


def data_options(architecture: type[ParrotArchitecture]) -> argparse.ArgumentParser:
    """How the data are read and split, and how sequences are encoded (every architecture)

    Parameters
    ----------
    architecture : ParrotArchitecture subclass

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('data')

    # An architecture that supports a single datatype needs no -d (see required_options() for the others)
    datatypes = architecture.supported_datatypes
    if len(datatypes) == 1:
        only = datatypes[0]
        group.add_argument('-d', '--datatype', choices=datatypes, default=only,
                           help=f"'{only}' ({DATATYPE_DESCRIPTIONS[only]}), the only datatype {architecture.name} "
                                'supports, so this can be left out')
    default_fractions = ' '.join(str(fraction) for fraction in pid.DEFAULT_SET_FRACTIONS)
    group.add_argument('--set-fractions', nargs=3, default=None, type=float, dest='set_fractions',
                       metavar=('train', 'val', 'test'),
                       help=f'with --split {", ".join(SPLIT_METHODS)}: target fractions of the data in the '
                            f'training, validation and test sets (default {default_fractions})')
    group.add_argument('--save-splits', dest='save_splits', action='store_true',
                       help=f'with --split {", ".join(SPLIT_METHODS)}: save the split as a split-file '
                            '(<output>_split_file.txt), to reuse with --split')
    group.add_argument('--exclude-seq-id', dest='exclude_seq_id', action='store_true',
                       help='the datafile has no sequence IDs: each line starts with the sequence')
    group.add_argument('--ignore-warnings', '-w', dest='ignore_warnings', action='store_true',
                       help='do not print warnings about the data (e.g. class imbalance)')

    group = options.add_argument_group('encoding')
    group.add_argument('--encode', default='onehot', type=str, metavar='encoding_scheme',
                       help="'onehot' (default), 'biophysics', a protein language model (e.g. "
                            "'esm2_t12_35M_UR50D'), a path to a user-created scheme, or (sequence data only) "
                            "'nardini', the NARDINI+ features of the whole sequence, alone or after 'onehot', "
                            "'biophysics' or a user-created scheme (e.g. 'onehot+nardini')")
    group.add_argument('--standardize-encoding', dest='standardize_encoding', action='store_true',
                       help='standardize every feature of the encoding to mean 0 and standard deviation 1 across '
                            'the amino acids (or other characters) it defines, so that features on different '
                            'scales contribute equally; saved with the network and applied automatically when '
                            'predicting')
    add_nardini_scrambles_argument(group)
    add_embedding_cache_argument(group, plm.TRAIN_CACHE_DIRECTORY)
    add_embedding_context_arguments(group)
    add_embed_mean_subtraction_argument(group)
    add_embedding_layer_argument(group)
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
                       help=f'number of training epochs (default {DEFAULT_EPOCHS})')
    group.add_argument('-b', '--batch', default=DEFAULT_BATCH_SIZE, type=int, metavar='batch_size',
                       help=f'sequences per training batch (default {DEFAULT_BATCH_SIZE})')
    group.add_argument('--batching', default=pid.DEFAULT_BATCHING, choices=pid.BATCHING_CHOICES,
                       help="how training sequences are grouped into batches: 'length' (default) batches sequences "
                            "of similar length together, so little time is spent on padding; 'random' shuffles "
                            "freely (PARROT's original behaviour)")
    group.add_argument('--gradient-clip-norm', type=float, default=None, metavar='max_norm',
                       help='opt in to global L2 gradient clipping at this positive, finite norm; also stop on '
                            'nonfinite losses or gradients (default: off)')
    group.add_argument('--check-finite-gradients', action='store_true',
                       help='opt in to stopping on nonfinite losses or gradient norms before an update; '
                            'does not clip gradients (default: off)')
    group.add_argument('--equal-protein-weighting', action='store_true',
                       help='opt in to averaging labelled losses within each protein, then across proteins '
                            '(residue and matrix tasks only; default: sum over labelled values)')
    group.add_argument('--recover-nonfinite', type=int, nargs='?', const=2, default=0, metavar='max_recoveries',
                       help='opt in to bounded in-process rollback and epoch replay after numerical failure; '
                            'optional run-wide retry budget (default when enabled: 2; otherwise off)')
    group.add_argument('--recovery-lr-factor', type=float, default=None, metavar='factor',
                       help='multiply LR by this factor after each rollback (default 0.5; needs --recover-nonfinite)')
    group.add_argument('--lr-schedule', choices=LR_SCHEDULES, default='fixed',
                       help='fixed LR (default), cosine or exponential annealing, or validation-driven plateau decay')
    group.add_argument('--lr-anneal-epochs', type=int, default=None, metavar='epochs',
                       help='cosine decay duration after warmup, then hold at --lr-min (default: full run minus '
                            'warmup; required with --stop auto)')
    group.add_argument('--lr-min', type=float, default=None, metavar='learning_rate',
                       help='LR lower bound for scheduling or recovery (default 0)')
    group.add_argument('--lr-gamma', type=float, default=None, metavar='factor',
                       help='exponential schedule: multiply LR by this factor each epoch (default 0.95)')
    group.add_argument('--lr-factor', type=float, default=None, metavar='factor',
                       help='plateau schedule: LR reduction factor (default 0.5)')
    group.add_argument('--lr-patience', type=int, default=None, metavar='epochs',
                       help='plateau schedule: tolerated epochs without improvement before reducing LR (default 5)')
    group.add_argument('--lr-threshold', type=float, default=None, metavar='loss_change',
                       help='plateau schedule: minimum absolute validation-loss improvement (default 0.0001)')
    group.add_argument('--lr-cooldown', type=int, default=None, metavar='epochs',
                       help='plateau schedule: cooldown epochs after a reduction (default 0)')
    group.add_argument('--lr-warmup-epochs', type=int, default=0, metavar='epochs',
                       help='opt in to linear LR warmup to the initial LR before the chosen schedule (default off)')
    group.add_argument('--regression-loss', choices=('mae', 'mse', 'huber'), default=None,
                       help='regression objective (default: MAE for sequence, MSE for residue/matrix); '
                            'not compatible with --uncertainties')
    group.add_argument('--huber-delta', type=float, default=None,
                       help='positive finite Huber transition in target units (default 1; needs --regression-loss huber)')
    group.add_argument('--optimizer', choices=('adam', 'adamw'), default='adam',
                       help='Adam (default) or AdamW; weight decay is configured separately')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='nonnegative optimizer weight decay (default 0, including AdamW)')
    group.add_argument('--amp', choices=AMP_MODES, nargs='?', const='auto', default='off',
                       help='opt in to CUDA mixed precision: auto prefers native BF16, else FP16; '
                            'float16 and bfloat16 select explicitly (default off; CUDA capability >= 7.0); '
                            'BRNN training supports off only')
    group.add_argument('--checkpoint', nargs='?', const='auto', default=None, metavar='file',
                       help='opt in to atomic resumable disk checkpoints (default file: <network>_training_checkpoint.pt)')
    group.add_argument('--checkpoint-every', type=int, default=None, metavar='epochs',
                       help='checkpoint interval in successful epochs (default 1; needs --checkpoint or --resume)')
    group.add_argument('--resume', default=None, metavar='file',
                       help='resume a training checkpoint, restoring saved arguments, data split and training state')
    group.add_argument('--training-log', action='store_true',
                       help='opt in to a flushed epoch-by-epoch <network>_training_history.tsv during training')
    return options


def gradient_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Validated, active gradient options only, shared by training, search and its final retraining."""
    train_network.validate_gradient_clip_norm(args.gradient_clip_norm)
    options: dict[str, Any] = {}
    if args.gradient_clip_norm is not None:
        options['gradient_clip_norm'] = args.gradient_clip_norm
    if args.check_finite_gradients:
        options['check_finite_gradients'] = True
    return options


def training_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Active policies only; reject tuning flags that would have no effect."""
    validate_amp_architecture(args.architecture, args.amp)
    result = gradient_options_from_args(args)
    if args.checkpoint_every is not None and (args.checkpoint_every < 1
                                               or (args.checkpoint is None and args.resume is None)):
        raise ValueError('--checkpoint-every needs --checkpoint or --resume and a positive integer')
    if args.regression_loss is not None and (not args.regression or args.uncertainties):
        raise ValueError('--regression-loss applies to regression without --uncertainties only')
    if args.huber_delta is not None and args.regression_loss != 'huber':
        raise ValueError('--huber-delta requires --regression-loss huber')
    if args.huber_delta is not None and (not math.isfinite(args.huber_delta) or args.huber_delta <= 0):
        raise ValueError('--huber-delta must be positive and finite')
    requirements = {'lr_anneal_epochs': args.lr_schedule == 'cosine',
                    'lr_gamma': args.lr_schedule == 'exponential',
                    'lr_factor': args.lr_schedule == 'plateau', 'lr_patience': args.lr_schedule == 'plateau',
                    'lr_threshold': args.lr_schedule == 'plateau', 'lr_cooldown': args.lr_schedule == 'plateau',
                    'recovery_lr_factor': args.recover_nonfinite > 0,
                    'lr_min': args.lr_schedule != 'fixed' or args.lr_warmup_epochs > 0 or args.recover_nonfinite > 0}
    for name, applies in requirements.items():
        if getattr(args, name) is not None and not applies:
            raise ValueError(f"--{name.replace('_', '-')} does not apply to the chosen training controls")

    def supplied(name: str, default: Any) -> Any:
        value = getattr(args, name)
        return default if value is None else value

    options = TrainingOptions(
        equal_protein_weighting=args.equal_protein_weighting, recover_nonfinite=args.recover_nonfinite,
        recovery_lr_factor=supplied('recovery_lr_factor', 0.5), lr_schedule=args.lr_schedule,
        lr_anneal_epochs=args.lr_anneal_epochs, lr_min=supplied('lr_min', 0.0),
        lr_gamma=supplied('lr_gamma', 0.95), lr_factor=supplied('lr_factor', 0.5),
        lr_patience=supplied('lr_patience', 5), lr_threshold=supplied('lr_threshold', 1e-4),
        lr_cooldown=supplied('lr_cooldown', 0), lr_warmup_epochs=args.lr_warmup_epochs,
        optimizer=args.optimizer, weight_decay=args.weight_decay, amp=args.amp,
    ).resolved(args.epochs, getattr(args, 'stop', 'iter'))
    if options.equal_protein_weighting and args.datatype == 'sequence':
        raise ValueError('Equal-protein weighting applies to residue or matrix tasks, not sequence tasks')
    if options.controls_lr and hasattr(args, 'learning_rate') and options.lr_min > args.learning_rate:
        raise ValueError('--lr-min must not exceed the initial learning rate')
    if options.active:
        result['training_options'] = options
    return result


def active_training_settings(options: dict[str, Any]) -> dict[str, Any]:
    """Flatten active policies for a user-facing header or provenance, omitting inactive options."""
    result = {name: value for name, value in options.items() if name != 'training_options'}
    if 'training_options' in options:
        result.update(options['training_options'].active_settings())
    return result


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
    description = None
    if describe_search:
        description = 'Any hyperparameter given here is fixed at that value; the search tunes the others it searches'
        if architecture.trained_by_gradient_descent:
            description += f', and the learning rate ({search_description(train_network.LEARNING_RATE)})'
        description += '.'

    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group(f'{architecture.name} hyperparameters', description=description)

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
                       metavar='residues',
                       help='square maps only: refuse sequences longer than this, as memory grows with length^2 '
                            f'(default {DEFAULT_MAX_MATRIX_LENGTH})')
    return options


def uncertainty_options() -> argparse.ArgumentParser:
    """--uncertainties (architectures for sequence or residue data that can weight their training values)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('data')
    group.add_argument('--uncertainties', action='store_true',
                       help="regression: every value in the datafile is written as 'value,SD', where SD is the "
                            "standard deviation of its measurement error (a missing residue value is still 'nan'); "
                            'training weights each value by 1 / (SD^2 + s^2), where s, the model error, is the error '
                            "of the predictions beyond the values' own errors, fitted on the validation set")
    return options


def class_probability_options() -> argparse.ArgumentParser:
    """--probabilistic-classification (architectures for sequence or residue data, the data it applies to)

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group('output')
    group.add_argument('--probabilistic-classification', dest='probabilistic_classification', action='store_true',
                       help='classification: report the probability of each class, not just the most likely class '
                            '(for residue data, each residue\'s probabilities are one comma-separated token)')
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
                       help='save figures next to the network: the training and validation loss per epoch (networks '
                            'trained epoch by epoch), and the test-set performance (for regression, a scatterplot '
                            'and a density plot; for classification, a confusion matrix, or with '
                            '--probabilistic-classification ROC and precision-recall curves; for residue data, also '
                            'example profiles of true and predicted values along test sequences, and for matrix '
                            'data, heatmaps of example test sequences)')
    group.add_argument('--no-stats', dest='no_stats', action='store_true',
                       help='do not write the test-set performance statistics file')
    group.add_argument('--verbose', '-v', action='store_true',
                       help='print every setting, and the validation loss after every epoch whenever one network is '
                            'trained epoch by epoch (parrot-train, or parrot-optimize --cv-folds once the best '
                            'configuration is retrained; not the configurations of the search itself)')
    group.add_argument('--silent', action='store_true', help='print nothing')

    group = options.add_argument_group('device and reproducibility')
    add_device_arguments(group)
    group.add_argument('--seed', type=int, default=None, metavar='seed',
                       help='random seed, for a reproducible data split and training (default: not seeded)')
    return options


def retired_options() -> argparse.ArgumentParser:
    """Options PARROT no longer has, hidden from the help: each stops the command with what replaced it

    Returns
    -------
    argparse.ArgumentParser
    """
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('-c', '--classes', action=RetiredOption,
                         replacement='use --regression, or --classification with --num-classes N')
    options.add_argument('--architecture', action=RetiredOption,
                         replacement='the architecture is the subcommand, e.g. parrot-train cnn --input-file ...')
    return options


# .................................................................
# Parsing

def parse_arguments(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a training command's arguments, explaining what to do if they do not start with an architecture

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
    from parrot.scripts.training_artifacts import check_resume_arguments, expand_resume
    arguments, resumed = expand_resume(parser, arguments)
    problem = first_argument_problem(arguments, parser.prog)
    if problem is not None:
        parser.error(problem)
    parsed = parser.parse_args(arguments)
    validate_amp_architecture(parsed.architecture, parsed.amp)
    check_resume_arguments(parsed, resumed)
    return parsed


def first_argument_problem(arguments: Sequence[str], program_name: str) -> str | None:
    """What is wrong with a command that does not start with an architecture (None if nothing is)

    Commands must start with the architecture (or ask for help or examples).
    Otherwise argparse's own message would be unhelpful, so this says what was
    probably meant: an architecture name was misspelt, the architecture was
    left out, or the command was written for an earlier version of PARROT,
    which took the datafile and the network as its first two arguments.

    Parameters
    ----------
    arguments : list of str
        The arguments after the command's name
    program_name : str
        The command, e.g. 'parrot-train'

    Returns
    -------
    str or None
    """
    architectures = available_architectures()
    if not arguments or arguments[0] in architectures or any(argument in HELP_FLAGS for argument in arguments):
        return None

    first = arguments[0]
    example = f'{program_name} brnn --input-file data.tsv --output-network network --split kmer -d sequence --regression'
    if first.startswith('-'):
        return (f'give the architecture first, e.g. {example} ({program_name} -h lists the architectures)')
    if os.path.exists(first):
        return (f'{program_name} now takes the architecture first, then the datafile as --input-file, the network '
                f'as --output-network and how to split the data as --split, e.g. {example} (see {program_name} '
                f'{EXAMPLES_FLAG})')
    return f"'{first}' is not an architecture. The architectures are: {', '.join(architectures)}"


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
