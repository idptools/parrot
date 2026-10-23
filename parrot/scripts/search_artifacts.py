"""Epoch/stage checkpointing and continuous records for a hyperparameter search."""

from dataclasses import asdict
from typing import Any

import torch

from parrot.hyperparameter_search import HyperparameterSearch, Trial
from parrot.training_persistence import (
    FORMAT_KEY,
    FORMAT_VERSION,
    EpochLog,
    atomic_save,
    capture_random_state,
    loader_signature,
    restore_random_state,
    validate_signature,
)
from parrot.training_precision import validate_amp_architecture


class SearchArtifacts:
    """Persist the whole search, including an interrupted fold's latest complete epoch."""

    def __init__(self, searcher: HyperparameterSearch, path: str | None, every: int, log_path: str | None,
                 metadata: dict[str, Any], resumed: dict[str, Any] | None) -> None:
        if resumed is not None:
            saved_signature = resumed['signature']
            validate_amp_architecture(saved_signature.get('architecture', ''),
                                      saved_signature.get('training_options', {}).get('amp', 'off'))
        self.searcher, self.path, self.every, self.metadata = searcher, path, every, metadata
        trainer = searcher.trainer
        self.loaders = tuple(loader for pair in trainer.loaders for loader in pair)
        dimensions = []
        for dimension in searcher.space.dimensions:
            values = asdict(dimension)
            values['value_type'] = dimension.value_type.__name__
            values.pop('help')
            values.pop('cli_aliases')
            dimensions.append(values)
        self.signature = {'architecture': searcher.space.architecture, 'fixed': searcher.space.fixed,
                          'dimensions': dimensions, 'stages': searcher.stages,
                          'trials': searcher.n_trials, 'reduction_factor': searcher.reduction_factor,
                          'task': trainer.task.to_config(), 'input_size': trainer.input_size,
                          'device_type': torch.device(trainer.device).type, 'torch_version': str(torch.__version__),
                          'training_options': asdict(trainer.options), 'gradient_options': trainer.gradient_options,
                          'loaders': [loader_signature(loader) for loader in self.loaders],
                          'metadata': metadata['identity']}
        capture_random_state(self.loaders, trainer.device)
        if resumed is not None:
            if resumed['kind'] != 'search':
                raise ValueError('Expected a hyperparameter-search checkpoint')
            validate_signature(resumed['signature'], self.signature)
            searcher.load_state_dict(resumed['search_state'])
            restore_random_state(resumed['random_state'], self.loaders, trainer.device)
        self.log = EpochLog(log_path, resumed.get('records', []) if resumed else None, resume=resumed is not None)
        previous = searcher.on_progress

        def progress(current: HyperparameterSearch) -> None:
            if previous is not None:
                previous(current)
            self.save()

        searcher.on_progress = progress
        trainer.on_epoch = self.epoch
        if path is not None:
            trainer.on_recovery = self.save
        if path is not None and resumed is None:
            self.save()

    def epoch(self, trial: Trial, fold: int, row: dict[str, Any]) -> None:
        self.log.append(row)
        if len(self.log.rows) % self.every == 0:
            self.save()

    def save(self) -> None:
        if self.path is None:
            return
        atomic_save({FORMAT_KEY: FORMAT_VERSION, 'kind': 'search', 'signature': self.signature,
                     'metadata': self.metadata, 'search_state': self.searcher.state_dict(), 'records': self.log.rows,
                     'random_state': capture_random_state(self.loaders, self.searcher.trainer.device)}, self.path)
