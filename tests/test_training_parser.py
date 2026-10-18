"""
Tests for the command line of parrot-train and parrot-optimize
(parrot/scripts/training_parser.py): one subcommand per architecture, each
offering only the options that apply to it, and commands written before
subcommands existed still working.

These tests only parse arguments; tests/test_cli.py runs the commands.
"""

from types import ModuleType

import pytest

from parrot.architectures import HyperParameter, ParrotArchitecture, available_architectures, get_architecture
from parrot.architectures import registry
from parrot.scripts import parrot_optimize, parrot_train, training_parser

from tests.helpers import data_path

# Every built-in architecture (test-only ones are registered by fixtures, after this is evaluated)
BUILT_IN_ARCHITECTURES = available_architectures()

TRAINING_COMMANDS = [parrot_train, parrot_optimize]


class OffByDefaultTestArchitecture(ParrotArchitecture):
    """A test-only architecture whose attention pooling is off by default (the built-in ones have it on)"""

    name = 'test_attention_off'
    description = 'test-only architecture'
    hyperparameters = (HyperParameter('attention_pooling', bool, default=False, help='pool with attention'),)


def required_arguments(architecture: str) -> list[str]:
    """The fewest arguments an architecture's subcommand accepts (-d only where there is a choice)."""
    datatype = ['-d', 'sequence'] if len(get_architecture(architecture).supported_datatypes) > 1 else []
    return [architecture, '--input-file', 'data.tsv', '--output-network', 'net', '--split', 'random', *datatype,
            '--regression']


def parse(command: ModuleType, arguments: list[str]):
    """Parse a training command's arguments as its main() does."""
    return training_parser.parse_arguments(command.build_parser(), arguments)


def help_text(command: ModuleType, arguments: list[str], capsys) -> str:
    """The help a command prints for the given arguments (which end in -h)."""
    with pytest.raises(SystemExit):
        parse(command, arguments)
    return capsys.readouterr().out


# ..........................................................................................
# One subcommand per architecture, with only its own options

@pytest.mark.parametrize('command', TRAINING_COMMANDS, ids=['train', 'optimize'])
@pytest.mark.parametrize('architecture', BUILT_IN_ARCHITECTURES)
def test_every_architecture_has_a_subcommand(command, architecture):
    args = parse(command, required_arguments(architecture))
    assert args.architecture == architecture
    assert training_parser.given_hyperparameters(args) == {}


@pytest.mark.parametrize('architecture', BUILT_IN_ARCHITECTURES)
def test_subcommands_offer_only_their_own_hyperparameters(architecture):
    own_flags = {hyperparameter.cli_flag for hyperparameter in get_architecture(architecture).hyperparameters}
    other_hyperparameters = {hyperparameter.cli_flag: hyperparameter
                             for name in BUILT_IN_ARCHITECTURES for hyperparameter in get_architecture(name).hyperparameters
                             if hyperparameter.cli_flag not in own_flags}

    parser = parrot_train.build_parser()
    for flag, hyperparameter in other_hyperparameters.items():
        value = [] if hyperparameter.value_type is bool else [str(hyperparameter.default)]
        with pytest.raises(SystemExit):
            parser.parse_args([*required_arguments(architecture), flag, *value])


def test_given_hyperparameters_are_collected_and_the_rest_take_defaults():
    args = parse(parrot_train, [*required_arguments('brnn'), '-hs', '7', '--attention-pooling-off'])
    assert training_parser.given_hyperparameters(args) == {'hidden_size': 7, 'attention_pooling': False}
    assert training_parser.hyperparameters_from_args(args) == {'hidden_size': 7, 'num_layers': 1, 'dropout': 0.0,
                                                               'attention_pooling': False,
                                                               'loglength_feedforward': True}


@pytest.mark.parametrize('dropout', ['1.5', '1'])
def test_out_of_range_hyperparameters_are_rejected(dropout):
    # A dropout of 1 would zero every activation in training, so the network could not learn anything
    args = parse(parrot_train, [*required_arguments('cnn'), '--dropout', dropout])
    with pytest.raises(ValueError, match='less than 1.0'):
        training_parser.given_hyperparameters(args)


def test_hyperparameters_sharing_a_name_keep_their_own_defaults(capsys):
    """The BRNN's hidden size defaults to 10 and the composition baseline's to 32; each help shows its own"""
    brnn_help = help_text(parrot_train, ['brnn', '-h'], capsys)
    composition_help = help_text(parrot_train, ['composition', '-h'], capsys)
    assert 'size of the LSTM hidden vectors (default 10)' in brnn_help
    assert 'width of each hidden layer (default 32)' in composition_help
    assert 'LSTM' not in composition_help


def test_on_off_hyperparameters_may_differ_in_default_between_architectures(monkeypatch):
    """Each subcommand has its own flags, so architectures no longer need to agree on a shared name's default"""
    monkeypatch.setitem(registry._ARCHITECTURES, OffByDefaultTestArchitecture.name, OffByDefaultTestArchitecture)

    turned_on = parse(parrot_train, [*required_arguments('test_attention_off'), '--attention-pooling'])
    turned_off = parse(parrot_train, [*required_arguments('brnn'), '--attention-pooling-off'])
    assert training_parser.given_hyperparameters(turned_on) == {'attention_pooling': True}
    assert training_parser.given_hyperparameters(turned_off) == {'attention_pooling': False}


# ..........................................................................................
# Option sets that only some kinds of architecture inherit

@pytest.mark.parametrize('command, loop_option', [(parrot_train, ['-lr', '0.01']), (parrot_train, ['-e', '5']),
                                                  (parrot_optimize, ['--min-epochs', '2'])],
                         ids=['train-lr', 'train-epochs', 'optimize-min-epochs'])
def test_models_fitted_in_one_step_have_no_training_loop_options(command, loop_option):
    parse(command, [*required_arguments('brnn'), *loop_option])
    with pytest.raises(SystemExit):
        parse(command, [*required_arguments('knn'), *loop_option])


def test_options_a_subcommand_does_not_offer_take_their_defaults():
    """Every subcommand's arguments have the same names, so the commands never need to know which was used"""
    knn = parse(parrot_train, required_arguments('knn'))
    assert (knn.epochs, knn.batch, knn.batching, knn.stop) == (training_parser.DEFAULT_EPOCHS,
                                                               training_parser.DEFAULT_BATCH_SIZE, 'length', 'iter')

    brnn = parse(parrot_train, required_arguments('brnn'))
    assert (brnn.matrix_file, brnn.matrix_width, brnn.symmetric) == (None, None, False)

    unet2d = parse(parrot_train, required_arguments('unet2d'))
    assert unet2d.probabilistic_classification is False


def test_matrix_options_are_offered_to_matrix_architectures_only():
    args = parse(parrot_train, [*required_arguments('unet2d'), '--matrix-file', 'maps.npz', '--symmetric'])
    assert (args.matrix_file, args.symmetric) == ('maps.npz', True)
    with pytest.raises(SystemExit):
        parse(parrot_train, [*required_arguments('brnn'), '--matrix-file', 'maps.npz'])


def test_class_probabilities_are_offered_to_architectures_for_sequence_or_residue_data_only():
    assert parse(parrot_train, [*required_arguments('cnn'), '--probabilistic-classification']).probabilistic_classification
    residue_classes = ['unet', '--input-file', 'data.tsv', '--output-network', 'net', '--split', 'random',
                       '-d', 'residues', '--classification', '--num-classes', '3', '--probabilistic-classification']
    assert parse(parrot_train, residue_classes).probabilistic_classification
    with pytest.raises(SystemExit):
        parse(parrot_train, [*required_arguments('unet2d'), '--probabilistic-classification'])


@pytest.mark.parametrize('command, own_option, other_option', [(parrot_train, ['--stop', 'auto'], ['--trials', '5']),
                                                               (parrot_optimize, ['--trials', '5'], ['--stop', 'auto'])],
                         ids=['train', 'optimize'])
def test_each_command_adds_its_own_options(command, own_option, other_option):
    parse(command, [*required_arguments('brnn'), *own_option])
    with pytest.raises(SystemExit):
        parse(command, [*required_arguments('brnn'), *other_option])


# ..........................................................................................
# The datatype

@pytest.mark.parametrize('architecture, datatype', [('knn', 'sequence'), ('linear', 'sequence'), ('unet2d', 'matrix')])
def test_datatype_is_inferred_when_an_architecture_supports_only_one(architecture, datatype):
    assert parse(parrot_train, required_arguments(architecture)).datatype == datatype
    assert parse(parrot_train, [*required_arguments(architecture), '-d', datatype]).datatype == datatype


def test_datatype_is_required_when_an_architecture_supports_several():
    with pytest.raises(SystemExit):
        parse(parrot_train, ['brnn', '--input-file', 'data.tsv', '--output-network', 'net', '--regression'])


# ..........................................................................................
# The task, and commands that do not start with an architecture

@pytest.mark.parametrize('task_arguments, regression, classification, num_classes', [
    (['--regression'], True, False, None),
    (['--classification', '--num-classes', '3'], False, True, 3),
])
def test_task_options_are_parsed(task_arguments, regression, classification, num_classes):
    arguments = ['brnn', '--input-file', 'data.tsv', '--output-network', 'net', '--split', 'random', '-d', 'sequence',
                 *task_arguments]
    args = parse(parrot_train, arguments)
    assert (args.regression, args.classification, args.num_classes) == (regression, classification, num_classes)


@pytest.mark.parametrize('command', TRAINING_COMMANDS, ids=['train', 'optimize'])
def test_split_is_required(command, capsys):
    arguments = ['brnn', '--input-file', 'data.tsv', '--output-network', 'net', '-d', 'sequence', '--regression']
    with pytest.raises(SystemExit):
        parse(command, arguments)
    assert 'the following arguments are required: --split' in capsys.readouterr().err


@pytest.mark.parametrize('arguments, problem', [
    ([], None),
    (['-h'], None),
    (['cnn', '--input-file', 'data.tsv'], None),
    (['data.tsv', '--show-examples'], None),   # asking for examples always works
    (['--input-file', 'data.tsv'], 'give the architecture first'),
    ([data_path('seq_regress_dataset.tsv'), 'net.pt', '-c', '1'], 'now takes the architecture first'),
    (['cnnn', '--input-file', 'data.tsv'], "'cnnn' is not an architecture"),
], ids=['nothing', 'help', 'architecture', 'examples', 'no-architecture', 'old-style', 'misspelt'])
def test_commands_that_do_not_start_with_an_architecture_are_explained(arguments, problem):
    found = training_parser.first_argument_problem(arguments, 'parrot-train')
    assert (found is None) if problem is None else (problem in found)


# ..........................................................................................
# Help

def test_command_help_lists_architectures_by_kind(capsys):
    """Neural networks for sequence and residue data, then models fitted in one step, then matrix architectures"""
    listing = help_text(parrot_train, ['-h'], capsys)
    positions = {name: listing.index(f'    {name} ') for name in BUILT_IN_ARCHITECTURES}
    assert max(positions[name] for name in ('brnn', 'cnn', 'composition', 'transformer', 'unet')) < \
        min(positions[name] for name in ('knn', 'linear', 'random_forest', 'svm')) and \
        max(positions[name] for name in ('knn', 'linear', 'random_forest', 'svm')) < positions['unet2d']


def test_only_optimize_help_describes_the_search(capsys):
    # argparse wraps help text, so compare with line breaks removed
    optimize_help = ' '.join(help_text(parrot_optimize, ['brnn', '-h'], capsys).split())
    assert 'number of LSTM layers per direction (default 1; searched over 1-5)' in optimize_help
    assert 'searched' not in help_text(parrot_train, ['brnn', '-h'], capsys)


@pytest.mark.parametrize('architecture, usage', [
    ('brnn', 'brnn --input-file file --output-network name --split {mmseqs,kmer,random,file} '
             '-d {sequence,residues} (--regression | --classification --num-classes N) [options]'),
    ('knn', 'knn --input-file file --output-network name --split {mmseqs,kmer,random,file} '
            '(--regression | --classification --num-classes N) [options]'),
    ('unet2d', 'unet2d --input-file file --output-network name --split {mmseqs,kmer,random,file} '
               '(--regression | --classification --num-classes N) --matrix-file file.npz [options]'),
])
def test_usage_lists_the_required_arguments(architecture, usage, capsys):
    assert usage in help_text(parrot_train, [architecture, '-h'], capsys)
