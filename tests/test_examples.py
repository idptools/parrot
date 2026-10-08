"""
Tests for --show-examples (parrot/scripts/examples.py): every command prints its
guide and exits without needing its other arguments, and every example command
in the guides is one the command accepts.
"""

import shlex

import pytest

from parrot.scripts import (examples, parrot_cvsplit, parrot_optimize, parrot_predict, parrot_preprocess, parrot_train,
                            training_parser)

COMMANDS = {
    'parrot-train': parrot_train,
    'parrot-optimize': parrot_optimize,
    'parrot-predict': parrot_predict,
    'parrot-preprocess': parrot_preprocess,
    'parrot-cvsplit': parrot_cvsplit,
}


def show_examples(run_cli, program_name: str, *arguments: str) -> None:
    """Run a command with --show-examples (and any other arguments), expecting it to exit cleanly."""
    with pytest.raises(SystemExit) as exit_info:
        run_cli(COMMANDS[program_name].main, [program_name, *arguments, '--show-examples'])
    assert exit_info.value.code in (0, None)


@pytest.mark.parametrize('program_name', COMMANDS)
def test_every_command_shows_its_guide(run_cli, capsys, program_name):
    show_examples(run_cli, program_name)
    output = capsys.readouterr().out

    assert f' {program_name}  |  PARROT ' in output                      # the header
    assert f'run: {program_name} --show-examples' not in output          # ...without the hint to run this
    assert output.rstrip().endswith(examples.GUIDES[program_name].strip())
    for section in ('What it does', 'Examples', 'How it works'):
        assert f'\n{section}\n' in output


@pytest.mark.parametrize('arguments', [['cnn'], ['brnn', '--input-file', 'data.tsv', '-d', 'sequence'],
                                       ['unet2d', '--regression']],
                         ids=['subcommand', 'partial-command', 'partial-matrix-command'])
def test_train_shows_its_guide_wherever_the_option_is_given(run_cli, capsys, arguments):
    show_examples(run_cli, 'parrot-train', *arguments)
    assert examples.GUIDES['parrot-train'].strip() in capsys.readouterr().out


def example_commands() -> list[tuple[str, list[str]]]:
    """Every example command line in the guides, as (command, arguments)."""
    commands = []
    for guide in examples.GUIDES.values():
        for line in guide.splitlines():
            if not line.startswith('    parrot-'):
                continue
            words = shlex.split(line)
            if '-h' not in words:
                commands.append((words[0], words[1:]))
    return commands


@pytest.mark.parametrize('program_name, arguments', example_commands(),
                         ids=[' '.join(words[:2]) for _, words in example_commands()])
def test_every_example_is_a_valid_command(program_name, arguments):
    """Each example parses with the command's own parser, so the guides stay in step with the options"""
    parser = COMMANDS[program_name].build_parser()
    if program_name in ('parrot-train', 'parrot-optimize'):
        training_parser.parse_arguments(parser, arguments)
    else:
        parser.parse_args(arguments)
