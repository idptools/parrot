"""Matched BRNN precision pilots and single-batch replay on the training machine.

Run ``python -m parrot.scripts.diagnose_brnn --help``. This deliberately uses
PARROT's actual loss, AMP, norm checks, clipping and optimizer-update pathway.
"""

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import parrot
from parrot import process_input_data as pid
from parrot import train_network as tn
from parrot.architectures import brnn, build_network
from parrot.devices import full_float32_precision
from parrot.gradient_numerics import gradient_diagnostics, stable_total_norm
from parrot.tasks import Task
from parrot.training_diagnostics import environment
from parrot.training_persistence import restore_random_state
from parrot.training_precision import TrainingPrecision


def json_safe(value):
    """Strict JSON, including clear strings for nonfinite measurements."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def lstm_implementation(implementation, recurrent_float32=False):
    """A diagnostic-only substitution; production BRNN behavior is unchanged."""
    original = brnn.run_lstm

    def run(lstm, vectors, lengths=None):
        def forward():
            inputs = vectors.float() if recurrent_float32 else vectors
            if implementation == 'unpacked' or lengths is None:
                return original(lstm, inputs, lengths)
            packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, (hidden, _) = lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=inputs.shape[1])
            return output, hidden

        if recurrent_float32:
            with torch.autocast(vectors.device.type, enabled=False):
                return forward()
        return forward()

    with patch.object(brnn, 'run_lstm', run):
        yield


def epoch_batches(dataset, task, args, epoch):
    """An independent CPU RNG stream guarantees matching indices across modes."""
    loader = pid.make_dataloader(dataset, task, args.batch, shuffle=True, batching=args.batching)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed + 1000 + epoch)
        batches = list(loader.batch_sampler)
    return batches[:args.max_batches] if args.max_batches else batches


def initial_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_datasets(args, task):
    if args.input_file:
        data, train, validation, _ = pid.load_and_split(
            args.input_file, task, split_file=None if args.split == 'random' else args.split,
            ignore_warnings=True)
        return pid.SequenceDataset(data, train), pid.SequenceDataset(data, validation)
    generator = np.random.default_rng(args.seed)
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    data = []
    for repeat in range(2):
        for length in args.synthetic_lengths:
            sequence = ''.join(generator.choice(alphabet, length))
            values = 0.5 + 0.2 * np.sin(np.arange(length) / 20) + generator.normal(0, 0.01, length)
            data.append([f'synthetic_{repeat}_{length}', sequence, values])
    # Synthetic evaluation is only a numerical smoke check, not a held-out performance estimate.
    dataset = pid.SequenceDataset(data)
    return dataset, dataset


def batch_step(network, optimizer, batch, task, device, precision, clip_norm, capture=None,
               equal_protein_weighting=False):
    statistics = tn.GradientStatistics()
    loss, _ = tn.train_epoch(network, [batch], task, tn.build_loss_criterion(task), optimizer, device,
                              check_finite_gradients=True, check_optimizer_state=True,
                              gradient_clip_norm=clip_norm, gradient_statistics=statistics,
                              precision=precision, capture_nonfinite=capture,
                              equal_protein_weighting=equal_protein_weighting)
    return {'loss_per_protein': float(loss), 'pre_clip_norm': statistics.norms[0],
            'ordinary_norm': statistics.raw_norms[0],
            'norm_reduction_overflow': not math.isfinite(statistics.raw_norms[0]),
            'post_clip_norm': float(stable_total_norm([p.grad for p in network.parameters() if p.grad is not None]))}


def run_pilot(args, mode, train, validation, task):
    seed_all(args.seed)
    label = 'fp32' if mode == 'off' else mode
    result = {'mode': mode, 'implementation': 'unpacked', 'status': 'running', 'completed_batches': 0}
    started = time.perf_counter()
    try:
        precision = None if mode == 'off' else TrainingPrecision(mode, args.device)
    except ValueError as error:
        return {**result, 'status': 'unavailable', 'error': str(error)}
    network = build_network('brnn', 20, task, {'hidden_size': args.hidden_size, 'num_layers': args.num_layers,
                                             'dropout': args.dropout}).to(args.device)
    result['initial_weights_sha256'] = initial_digest(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    capture = str(args.output_dir / f'{label}_failure.pt')
    result['resolved_dtype'] = str(precision.dtype) if precision is not None else 'torch.float32'
    with (args.output_dir / f'{label}_batches.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            for number, indices in enumerate(epoch_batches(train, task, args, epoch), 1):
                batch = pid.collate_batch([train[index] for index in indices], task)
                row = {'epoch': epoch + 1, 'batch': number,
                       'indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                       'length_min': int(batch.lengths.min()), 'length_max': int(batch.lengths.max()),
                       'labelled_residues': int((~torch.isnan(batch.targets[..., 0])).sum())}
                try:
                    row.update(batch_step(network, optimizer, batch, task, args.device, precision,
                                          args.gradient_clip_norm, capture=capture))
                    row['status'] = 'ok'
                except RuntimeError as error:
                    row.update(status='failed', error=str(error), diagnostics=gradient_diagnostics(network))
                    result.update(status='failed', epoch=epoch + 1, batch=number, error=str(error))
                    if Path(capture).exists():
                        result['failure_capture'] = capture
                log.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
                log.flush()
                if row['status'] == 'failed':
                    result['seconds'] = time.perf_counter() - started
                    return result
                result['completed_batches'] += 1
            try:
                val_loss = tn.validation_loss(network, pid.make_dataloader(validation, task, args.batch),
                                              task, args.device)
                if not math.isfinite(val_loss):
                    raise RuntimeError('Nonfinite FP32 validation loss')
            except RuntimeError as error:
                result.update(status='failed', epoch=epoch + 1, error=str(error),
                              seconds=time.perf_counter() - started)
                return result
            print(f'{label}: epoch {epoch + 1}/{args.epochs}, validation loss {val_loss:.6g}', flush=True)
            result['last_validation_loss'] = val_loss
    result.update(status='completed', seconds=time.perf_counter() - started)
    return result


def replay_capture(args, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('parrot_failure_capture') != 1 or saved['network']['architecture'] != 'brnn':
        raise ValueError('Replay requires a PARROT BRNN failure capture')
    task = Task.from_config(saved['task'])
    if not task.per_residue or task.is_classification:
        raise ValueError('This diagnostic replays per-residue BRNN regression captures')
    results = []
    batch = pid.Batch(**saved['batch'])
    for mode in args.modes:
        for implementation, recurrent_float32 in [('unpacked', False), ('packed', False), ('unpacked', True)]:
            if mode == 'off' and recurrent_float32:
                continue
            row = {'capture': str(path), 'mode': mode, 'implementation': implementation,
                   'recurrent_float32': recurrent_float32}
            try:
                precision = None if mode == 'off' else TrainingPrecision(mode, args.device)
            except ValueError as error:
                results.append({**row, 'status': 'unavailable', 'error': str(error)})
                continue
            config = saved['network']
            network = build_network('brnn', config['input_size'], task, config['hyperparameters']).to(args.device)
            network.load_state_dict(saved['model_state'])
            optimizer_class = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW,
                               'SGD': torch.optim.SGD}.get(saved['optimizer'])
            if optimizer_class is None:
                raise ValueError(f"Unsupported replay optimizer: {saved['optimizer']}")
            optimizer = optimizer_class(network.parameters())
            optimizer.load_state_dict(saved['optimizer_state'])
            if precision is not None and saved['precision_state'] is not None \
                    and saved['precision_state']['dtype'] == str(precision.dtype):
                precision.load_state_dict(saved['precision_state'])
            restore_random_state(saved['random_state'], (), args.device)
            try:
                with lstm_implementation(implementation, recurrent_float32):
                    row.update(batch_step(network, optimizer, batch, task, args.device, precision,
                                          saved['gradient_clip_norm'],
                                          equal_protein_weighting=saved['equal_protein_weighting']))
                row['status'] = 'ok'
            except RuntimeError as error:
                row.update(status='failed', error=str(error), diagnostics=gradient_diagnostics(network))
            results.append(row)
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory for diagnostic artifacts')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--modes', nargs='+', default=['off', 'bfloat16'], choices=['off', 'bfloat16', 'float16'])
    parser.add_argument('--input-file', help='per-residue regression TSV; otherwise use synthetic proteins')
    parser.add_argument('--split', help='existing three-line split file, or random (requires --input-file)')
    parser.add_argument('--replay', type=Path, help='replay a saved failure instead of training a pilot')
    parser.add_argument('--hidden-size', '-hs', type=int, default=52)
    parser.add_argument('--num-layers', '-nl', type=int, default=2)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--gradient-clip-norm', type=float, default=None)
    parser.add_argument('--batching', choices=['length', 'random'], default='length')
    parser.add_argument('--max-batches', type=int, default=0, help='cap batches per epoch; 0 means all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--synthetic-lengths', nargs='+', type=int, default=[32, 257, 2049])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.modes) != len(set(args.modes)):
        parser.error('--modes must not contain duplicates')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run this in the remote training environment')
    if args.replay and (args.input_file or args.split):
        parser.error('--replay already contains its data; omit --input-file and --split')
    if bool(args.input_file) != bool(args.split):
        parser.error('give both --input-file and --split, or neither for synthetic smoke tests')
    if min(args.hidden_size, args.num_layers, args.batch, args.epochs, *args.synthetic_lengths) < 1 \
            or args.max_batches < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 \
            or not 0 <= args.dropout < 1:
        parser.error('sizes/epochs/LR must be positive, max-batches nonnegative, and dropout in [0,1)')
    tn.validate_gradient_clip_norm(args.gradient_clip_norm)
    if args.output_dir.exists():
        parser.error('choose a new --output-dir; existing artifacts are never overwritten')
    seed_all(args.seed)
    task = Task('residues', 1)
    datasets = None if args.replay else load_datasets(args, task)
    args.output_dir.mkdir(parents=True)
    with full_float32_precision():
        report = {'environment': environment(args.device), 'parrot_source': parrot.__file__,
                  'working_directory': str(Path.cwd()),
                  'arguments': {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
        write_json(args.output_dir / 'environment.json', report)
        if args.replay:
            report['replay'] = replay_capture(args, args.replay)
        else:
            report['pilots'] = []
            for mode in args.modes:
                result = run_pilot(args, mode, *datasets, task)
                report['pilots'].append(result)
                print(json.dumps(json_safe(result)), flush=True)
                write_json(args.output_dir / 'summary.json', report)
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            report['replay'] = []
            for result in report['pilots']:
                if 'failure_capture' in result:
                    report['replay'].extend(replay_capture(args, Path(result['failure_capture'])))
        write_json(args.output_dir / 'summary.json', report)
    print(f'Diagnostic results: {args.output_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
