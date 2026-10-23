"""Atomic training checkpoints, reproducible random states and flushed epoch records.

Training files use tensors and plain Python values, load with weights_only=True,
and are deliberately distinct from the inference checkpoint in checkpoint.py.
"""

import csv
import hashlib
import json
import os
import pickle
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

FORMAT_KEY = 'parrot_training_format'
FORMAT_VERSION = 1


def atomic_save(value: dict[str, Any], path: str) -> None:
    """Replace a checkpoint only after its new contents have been flushed to disk."""
    target = Path(path).absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '.', delete=False) as handle:
            temporary = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_training_checkpoint(path: str, kind: str | None = None) -> dict[str, Any]:
    """Read a versioned training checkpoint, refusing inference-only files."""
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError(f'Cannot read training checkpoint {path}: {error}') from error
    if not isinstance(value, dict) or value.get(FORMAT_KEY) != FORMAT_VERSION:
        raise ValueError(f'{path} is not a supported PARROT training checkpoint; inference .pt files cannot resume')
    if value.get('kind') not in ('train', 'search') or (kind is not None and value['kind'] != kind):
        raise ValueError(f'{path} is a checkpoint for a different training command')
    if not isinstance(value.get('signature'), dict) or not isinstance(value.get('random_state'), dict):
        raise ValueError(f'{path} has incomplete training state')  # noqa: TRY004 - malformed file contents
    required = ('model_state', 'optimizer_state', 'history', 'control_state', 'epoch', 'stopped', 'reference_loss',
                'last_meaningful_epoch', 'minimum_fall', 'timing', 'records') if value['kind'] == 'train' \
        else ('search_state', 'records')
    if any(name not in value for name in required):
        raise ValueError(f'{path} is missing required training checkpoint fields')
    return value


def loader_generators(loaders: Sequence[DataLoader]) -> list[torch.Generator]:
    """Explicit generators in stable loader/sampler order, shared instances counted once."""
    result: list[torch.Generator] = []
    for loader in loaders:
        if loader.num_workers:
            raise ValueError('Training checkpoint/resume requires DataLoaders with num_workers=0')
        for owner in (loader, loader.sampler, loader.batch_sampler):
            generator = getattr(owner, 'generator', None)
            if isinstance(generator, torch.Generator) and all(generator is not previous for previous in result):
                result.append(generator)
    return result


def capture_random_state(loaders: Sequence[DataLoader], device: str | torch.device) -> dict[str, Any]:
    selected = torch.device(device)
    state = cast(tuple[Any, ...], np.random.get_state())
    accelerator = None
    if selected.type == 'cuda':
        accelerator = torch.cuda.get_rng_state(selected)
    elif selected.type == 'mps':
        accelerator = torch.mps.get_rng_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'accelerator': accelerator, 'device_type': selected.type,
            'generators': [generator.get_state() for generator in loader_generators(loaders)],
            'generator_layout': generator_layout(loaders),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


def restore_random_state(state: dict[str, Any], loaders: Sequence[DataLoader], device: str | torch.device) -> None:
    selected = torch.device(device)
    generators = loader_generators(loaders)
    if selected.type != state['device_type'] or len(generators) != len(state['generators']) \
            or state['generator_layout'] != generator_layout(loaders):
        raise ValueError('Resume requires the same device type and DataLoader generator structure')
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if selected.type == 'cuda':
        torch.cuda.set_rng_state(state['accelerator'], selected)
    elif selected.type == 'mps':
        torch.mps.set_rng_state(state['accelerator'])
    for generator, saved in zip(generators, state['generators']):
        generator.set_state(saved)
    torch.backends.cudnn.deterministic = state['cudnn_deterministic']
    torch.backends.cudnn.benchmark = state['cudnn_benchmark']
    torch.use_deterministic_algorithms(state['deterministic_algorithms'], warn_only=state['deterministic_warn_only'])


def generator_layout(loaders: Sequence[DataLoader]) -> list[int | None]:
    """Record which loader/sampler owners share each explicit generator."""
    generators = loader_generators(loaders)
    return [next((index for index, generator in enumerate(generators)
                  if generator is getattr(owner, 'generator', None)), None)
            for loader in loaders for owner in (loader, loader.sampler, loader.batch_sampler)]


def digest(value: Any) -> str:
    """Stable content identity for plain configuration and NumPy values."""
    def plain(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(f'Unsupported checkpoint configuration value: {type(item).__name__}')
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=plain, separators=(',', ':')).encode()).hexdigest()


def sampler_signature(sampler: Any) -> dict[str, Any]:
    """Fingerprint supported stateless sampling policies without advancing their generators.

    Explicit generator state is saved separately. Exact type checks exclude
    subclasses that can override iteration or carry otherwise untracked state.
    """
    from parrot.process_input_data import LengthBatchSampler

    sampler_type = type(sampler)
    config: dict[str, Any] = {'type': f'{sampler_type.__module__}.{sampler_type.__qualname__}'}
    if sampler_type is SequentialSampler:
        config['num_samples'] = len(sampler)
        return config
    if sampler_type is RandomSampler:
        config.update(replacement=sampler.replacement, num_samples=sampler.num_samples,
                      source_size=len(sampler.data_source))
    elif sampler_type is WeightedRandomSampler:
        config.update(replacement=sampler.replacement, num_samples=sampler.num_samples, weights=digest(sampler.weights))
    elif sampler_type is SubsetRandomSampler:
        config['indices'] = digest(list(sampler.indices))
    elif sampler_type is BatchSampler:
        config.update(batch_size=sampler.batch_size, drop_last=sampler.drop_last,
                      sampler=sampler_signature(sampler.sampler))
    elif sampler_type is LengthBatchSampler:
        config.update(batch_size=sampler.batch_size, shuffle=sampler.shuffle, lengths=digest(sampler.lengths))
    else:
        raise ValueError(f'Training checkpoint/resume does not support sampler {config["type"]}; '
                         'use PARROT make_dataloader or supported PyTorch samplers')
    return config


def loader_signature(loader: DataLoader) -> dict[str, Any]:
    """Fingerprint ordered sequences, labels, encoding and batching without drawing random numbers."""
    dataset = loader.dataset
    if not hasattr(dataset, 'data') or not hasattr(dataset, 'encoder'):
        raise ValueError('Training checkpoint/resume currently requires PARROT SequenceDataset loaders')
    checksum = hashlib.sha256()
    for row in dataset.data:
        checksum.update(digest(row[:2]).encode())
        for values in row[2:]:
            array = np.asarray(values, dtype='<f8')
            checksum.update(str(array.shape).encode())
            checksum.update(array.tobytes())
    return {'data': checksum.hexdigest(), 'encoding': digest(dataset.encoder.to_config()),
            'sampler': sampler_signature(loader.sampler), 'batch_sampler': sampler_signature(loader.batch_sampler),
            'batch_size': loader.batch_size or getattr(loader.batch_sampler, 'batch_size', None),
            'shuffle': getattr(loader.batch_sampler, 'shuffle', None), 'drop_last': loader.drop_last}


def validate_signature(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    differences = [name for name in sorted(set(saved) | set(expected)) if saved.get(name) != expected.get(name)]
    if differences:
        raise ValueError('Resume configuration differs from the checkpoint: ' + ', '.join(differences))


class EpochLog:
    """Write every successful epoch immediately, with safe reconciliation after resume.

    On resume, checkpointed rows replace any trailing rows from work performed
    after the last checkpoint. New runs refuse to truncate an existing log.
    """

    def __init__(self, path: str | None, rows: list[dict[str, Any]] | None = None, resume: bool = False) -> None:
        self.path = path
        self.rows = list(rows or [])
        if path is not None:
            if os.path.exists(path) and not resume:
                raise ValueError(f'Training log already exists: {path}; choose a new output or use --resume')
            if resume:
                self._rewrite()

    def _rewrite(self) -> None:
        assert self.path is not None
        target = Path(self.path).absolute()
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=target.parent,
                                         prefix='.' + target.name + '.', delete=False) as handle:
            temporary = handle.name
            try:
                if self.rows:
                    writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]), delimiter='\t')
                    writer.writeheader()
                    writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        if self.path is not None:
            header = not self.rows
            with open(self.path, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter='\t')
                if header:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
        self.rows.append(dict(row))
