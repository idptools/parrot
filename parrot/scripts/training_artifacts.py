"""CLI preflight and metadata for resumable training and continuous records."""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from parrot.training_persistence import load_training_checkpoint

PATH_ARGUMENTS = ('input_file', 'output_network', 'matrix_file', 'embedding_context', 'embedding_context_map',
                  'embedding_cache')
RESUME_OVERRIDES = {'output_network', 'verbose', 'silent', 'include_figs', 'no_stats', 'checkpoint',
                    'checkpoint_every', 'training_log', 'resume', 'device', 'gpu_id', 'embedding_cache', 'epochs'}


def normalized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = {name: value for name, value in vars(args).items() if not name.startswith('_')}
    for name in (*PATH_ARGUMENTS, 'split', 'encode'):
        value = values.get(name)
        if value is not None and (name in PATH_ARGUMENTS or os.path.isfile(value)):
            values[name] = os.path.abspath(value)
    values['resume'] = None
    return values


def expand_resume(parser: argparse.ArgumentParser, tokens: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    """Restore saved CLI options so '<command> --resume file' is sufficient."""
    if any(flag in tokens for flag in ('-h', '--help', '--show-examples')):
        return tokens, None
    path = None
    for i, token in enumerate(tokens):
        if token == '--resume' and i + 1 < len(tokens):
            path = tokens[i + 1]
        elif token.startswith('--resume='):
            path = token.split('=', 1)[1]
    if path is None:
        return tokens, None
    saved = load_training_checkpoint(path)
    metadata = saved.get('metadata')
    if not isinstance(metadata, dict) or metadata.get('program') != parser.prog:
        raise ValueError('This checkpoint does not contain arguments for ' + parser.prog)
    values = metadata['command_args']
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    architecture = values['architecture']
    if architecture not in commands.choices:
        raise ValueError(f'Checkpoint architecture {architecture!r} is unavailable')
    command = commands.choices[architecture]
    original = [architecture]
    for action in command._actions:
        value = values.get(action.dest)
        # Resume writes back to the file supplied now, even if the original
        # command chose another path. An explicit new --checkpoint is retained
        # in tokens and can still select a different destination.
        if not action.option_strings or value is None or action.dest in (
                'architecture', 'checkpoint', 'resume', 'help', 'show_examples'):
            continue
        flag = action.option_strings[-1]
        if action.nargs == 0:
            if value != action.default:
                original.append(flag)
        elif isinstance(value, (list, tuple)):
            original.extend([flag, *(str(item) for item in value)])
        else:
            original.extend([flag, str(value)])
    if tokens and tokens[0] in commands.choices:
        if tokens[0] != architecture:
            raise ValueError('Resume architecture differs from the checkpoint')
        tokens = tokens[1:]
    return original + tokens, saved


def check_resume_arguments(args: argparse.Namespace, saved: dict[str, Any] | None) -> None:
    """Refuse altered training settings or inputs before preparing data or writing outputs."""
    if saved is None:
        return
    expected = saved['metadata']['command_args']
    current = normalized_arguments(args)
    changed = [name for name in set(expected) | set(current)
               if name not in RESUME_OVERRIDES and current.get(name) != expected.get(name)]
    if changed:
        raise ValueError('Resume arguments changed: ' + ', '.join(sorted(changed)))
    if args.epochs != expected['epochs'] and (getattr(args, 'stop', 'iter') == 'auto' or args.lr_schedule == 'cosine'
                                             or saved['metadata']['program'] == 'parrot-optimize'):
        raise ValueError('Resume must retain the planned epochs for automatic stopping, cosine schedules and searches')
    for path, checksum in saved['metadata']['sources'].items():
        if file_checksum(path) != checksum:
            raise ValueError(f'Resume input changed since the checkpoint: {path}')
    args._resume_payload = saved


def file_checksum(path: str) -> str:
    checksum = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def artifact_paths(args: argparse.Namespace, output_prefix: str) -> tuple[str | None, str | None]:
    if args.checkpoint_every is not None and args.checkpoint is None and args.resume is None:
        raise ValueError('--checkpoint-every requires --checkpoint or --resume')
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        raise ValueError('--checkpoint-every must be a positive integer')
    checkpoint = args.checkpoint
    if checkpoint == 'auto':
        checkpoint = output_prefix + '_training_checkpoint.pt'
    if checkpoint is None and args.resume is not None:
        checkpoint = args.resume
    if checkpoint is not None:
        checkpoint = os.path.abspath(checkpoint)
        reserved = ('.pt', '_training_history.tsv', '_search_training_history.tsv')
        if checkpoint in (os.path.abspath(output_prefix + suffix) for suffix in reserved):
            raise ValueError('Training checkpoint must have a different path from the inference network and epoch log')
        if Path(checkpoint).exists() and (args.resume is None or checkpoint != os.path.abspath(args.resume)):
            raise ValueError(f'Training checkpoint already exists: {checkpoint}; use --resume or choose a new path')
        if not Path(checkpoint).parent.is_dir():
            raise ValueError('Training checkpoint directory must already exist')
    log = output_prefix + '_training_history.tsv' if args.training_log else None
    if log is not None and os.path.exists(log) and args.resume is None:
        raise ValueError(f'Training log already exists: {log}; use --resume or choose a new output')
    return checkpoint, log


def run_metadata(args: argparse.Namespace, run: Any) -> dict[str, Any]:
    sources = {}
    for name in ('input_file', 'matrix_file', 'embedding_context', 'embedding_context_map'):
        value = getattr(args, name, None)
        if value is not None:
            path = os.path.abspath(value)
            sources[path] = file_checksum(path)
    if os.path.isfile(args.encode):
        path = os.path.abspath(args.encode)
        sources[path] = file_checksum(path)
    values = normalized_arguments(args)
    return {'program': run.program_name, 'command_args': values, 'sources': sources,
            'split_state': run.split_state, 'identity': {'sources': sources, 'split_state': run.split_state}}
