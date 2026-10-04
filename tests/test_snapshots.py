"""
Snapshot tests: behaviour that must not change by accident, pinned against
reference files in tests/test_data/snapshots/.

* Architectures. For every built-in architecture under every combination of its
  on/off options (and every alternative of its text options), the names and
  shapes of the network's parameters and its outputs on a fixed batch. Saved
  networks are loaded by parameter name, so renaming a layer would silently
  break every network saved with it; and a network built from the same seed
  must give the same outputs, so refactoring an architecture cannot quietly
  change what it computes or the order its weights are initialised in.
* Hyperparameter search. The configurations a seeded search tries, in order,
  and what happens to each (scored by a stand-in for training, so this is the
  same on any machine), plus a small search that trains for real.
* Training. The training and validation losses of a short, seeded training run.

If a change is meant to alter any of these (e.g. a new architecture or
hyperparameter, or a deliberate change to training), regenerate the reference
files from the repository root with

    python -m tests.test_snapshots

and say in the change log what changed and why. Never regenerate them to make a
refactor pass: a refactor should leave every snapshot unchanged.
"""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import train_network
from parrot.architectures import available_architectures, get_architecture
from parrot.tasks import Task
from parrot.reproducibility import set_random_seed

from tests.helpers import TEST_SEED, FakeTrainer, encode_batch, make_loaders, make_search, ready_network

SNAPSHOT_DIRECTORY = Path(__file__).parent / 'test_data' / 'snapshots'

# Sequences of the fixed batch every network is run on: a typical one, a single residue (shorter than the
# U-Net's downsampling factor) and the longest, which sets the padded length
SNAPSHOT_SEQUENCES = ['ACDEFGHIK', 'W', 'PQRSTVWYACDEFGHIKLMNPQ']

SNAPSHOT_TASKS = [Task('sequence', 1), Task('sequence', 3), Task('residues', 1), Task('residues', 3),
                  Task('matrix', 1), Task('matrix', 3, width=4)]

# Significant figures stored for each number: enough to catch any real change, few enough that results
# agree between machines whose floating-point arithmetic differs in the last bits
SIGNIFICANT_FIGURES = 7

# Outputs and losses are compared to this relative and absolute tolerance, which allows for the same
# floating-point differences between machines (and accumulates over a few epochs of training)
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 1e-5

# Extra settings worth pinning that are not on/off options or text options
EXTRA_CONFIGURATIONS = [
    ('composition', {'window_radius': 0}),
    ('composition', {'num_hidden_layers': 0}),
    ('brnn', {'num_layers': 2}),
]


# ..........................................................................................
# Computing the snapshots

def rounded(values: Any) -> Any:
    """Round every number in a (nested) list to SIGNIFICANT_FIGURES significant figures."""
    if isinstance(values, list):
        return [rounded(value) for value in values]
    return float(f'{values:.{SIGNIFICANT_FIGURES}g}')


def architecture_configurations() -> list[tuple[str, dict[str, Any]]]:
    """Every configuration of the built-in architectures that the snapshot pins.

    For each architecture: every combination of its on/off options, then each
    alternative of each text option on its own (pooling with attention pooling
    off, since only then is it used), then EXTRA_CONFIGURATIONS.

    Returns
    -------
    list of (str, dict)
        Architecture name and the hyperparameter values that differ from the defaults
    """
    configurations = []
    for name in available_architectures():
        hyperparameters = get_architecture(name).hyperparameters
        switches = [hyperparameter.name for hyperparameter in hyperparameters if hyperparameter.value_type is bool]
        for setting in itertools.product((True, False), repeat=len(switches)):
            configurations.append((name, dict(zip(switches, setting))))

        for hyperparameter in hyperparameters:
            if hyperparameter.value_type is not str:
                continue
            for choice in hyperparameter.choices or ():
                if choice == hyperparameter.default:
                    continue
                values: dict[str, Any] = {hyperparameter.name: choice}
                if hyperparameter.name == 'pooling' and 'attention_pooling' in switches:
                    values['attention_pooling'] = False
                configurations.append((name, values))

    return configurations + EXTRA_CONFIGURATIONS


def case_id(architecture: str, values: dict[str, Any], task: Task) -> str:
    """A readable, unique name for one (configuration, task) case, e.g. 'cnn global_context=False sequence/3'."""
    settings = ' '.join(f'{name}={value}' for name, value in values.items())
    task_name = f'{task.datatype}/{task.num_classes}'
    if task.width is not None:
        task_name += f'/width={task.width}'
    if task.symmetric:
        task_name += '/symmetric'
    return f'{architecture} {settings} {task_name}'.replace('  ', ' ')


def snapshot_network(architecture: str, values: dict[str, Any], task: Task) -> dict[str, Any]:
    """Build one network from a fixed seed and record its parameters and outputs.

    Parameters
    ----------
    architecture : str
        Registry name of the architecture
    values : dict
        Hyperparameter values that differ from the defaults
    task : Task
        The task the network is built for

    Returns
    -------
    dict
        'architecture', 'settings' (`values`) and 'task' (Task.to_config()) of
        the case; 'hyperparameters' (every value the network recorded);
        'parameters' (each state-dict entry's shape, or 'extra' for a fitted
        model's stored estimator) and 'outputs' (for each sequence of
        SNAPSHOT_SEQUENCES, the sequence's output, each real residue's output
        for residue tasks, or each real cell's output for matrix tasks)
    """
    set_random_seed(TEST_SEED)
    network = ready_network(architecture, task, values)
    network.eval()

    parameters = {name: list(tensor.shape) if isinstance(tensor, torch.Tensor) else 'extra'
                  for name, tensor in network.state_dict().items()}

    batch, lengths = encode_batch(SNAPSHOT_SEQUENCES)
    with torch.no_grad():
        output = network(batch, lengths)

    outputs = []
    for index, length in enumerate(lengths.tolist()):
        if task.is_matrix:
            per_sequence = output[index, :length, :length if task.width is None else task.width]
        elif task.per_residue:
            per_sequence = output[index, :length]
        else:
            per_sequence = output[index]
        outputs.append(rounded(per_sequence.tolist()))

    return {'architecture': architecture, 'settings': values, 'task': task.to_config(),
            'hyperparameters': network.hyperparameter_values, 'parameters': parameters, 'outputs': outputs}


def architecture_snapshot() -> dict[str, dict[str, Any]]:
    """Snapshot every configuration of every built-in architecture for every task it supports."""
    snapshot = {}
    for architecture, values in architecture_configurations():
        supported = get_architecture(architecture).supported_datatypes
        for task in SNAPSHOT_TASKS:
            if task.datatype in supported:
                snapshot[case_id(architecture, values, task)] = snapshot_network(architecture, values, task)
    return snapshot


def search_record(search: hs.HyperparameterSearch, best: hs.Trial) -> dict[str, Any]:
    """Everything a finished search did: each trial's values, origin, status and losses, and the best trial."""
    trials = [{'number': trial.number, 'origin': trial.origin, 'status': trial.status_text, 'epochs': trial.epochs,
               'values': trial.values, 'stage_losses': rounded([float(loss) for loss in trial.stage_losses])}
              for trial in search.trials]
    return {'stages': search.stages, 'best': best.number, 'trials': trials}


def search_snapshot() -> dict[str, dict[str, Any]]:
    """Snapshot three seeded searches: a stand-in search, one in which some configurations fail, and a real one.

    Returns
    -------
    dict
        The record (see search_record) of each search, by name
    """
    snapshot = {}

    search = make_search(n_trials=30, max_epochs=27, seed=5)
    snapshot['stand-in'] = search_record(search, search.run())

    search = make_search(n_trials=20, max_epochs=9, seed=2, min_epochs=1,
                         trainer=FakeTrainer(hs.SearchSpace('brnn'), fail_if=lambda values: values['num_layers'] > 2))
    snapshot['stand-in with failures'] = search_record(search, search.run())

    set_random_seed(TEST_SEED)
    space = hs.SearchSpace('brnn')
    trainer = hs.TrialTrainer(space, make_loaders(), 20, Task('sequence', 1), 'cpu')
    search = hs.HyperparameterSearch(space, trainer, 6, 4, rng=np.random.default_rng(0), silent=True)
    snapshot['trained'] = search_record(search, search.run())
    return snapshot


def training_snapshot() -> dict[str, dict[str, list[float]]]:
    """Snapshot the per-epoch losses of short, seeded training runs of a BRNN and a CNN.

    Returns
    -------
    dict
        For each run, by name: 'train_losses' and 'val_losses', one per epoch
    """
    snapshot = {}
    runs = [('brnn 4 epochs', 'brnn', 'iter', 4), ('cnn auto stop', 'cnn', 'auto', 2)]
    for name, architecture, stop_condition, n_epochs in runs:
        set_random_seed(TEST_SEED)
        network = ready_network(architecture, Task('sequence', 1))
        train_loader, val_loader = make_loaders()[0]
        train_losses, val_losses = train_network.train(network, train_loader, val_loader, Task('sequence', 1),
                                                       'cpu', 0.005, n_epochs, stop_condition, silent=True)
        snapshot[name] = {'train_losses': rounded(train_losses), 'val_losses': rounded(val_losses)}
    return snapshot


SNAPSHOTS = {'architectures': architecture_snapshot, 'search': search_snapshot, 'training': training_snapshot}


def load_snapshot(name: str) -> dict[str, Any]:
    """Read one reference snapshot from SNAPSHOT_DIRECTORY."""
    loaded: dict[str, Any] = json.loads((SNAPSHOT_DIRECTORY / f'{name}.json').read_text())
    return loaded


def format_snapshot(snapshot: dict[str, Any]) -> str:
    """Format a snapshot as JSON with one line per entry, so a change shows up in a diff as the entries it affects."""
    entries = [f'{json.dumps(name)}: {json.dumps(value)}' for name, value in snapshot.items()]
    return '{\n' + ',\n'.join(entries) + '\n}\n'


def write_snapshots() -> None:
    """Recompute every snapshot and overwrite the reference files (see the module docstring)."""
    SNAPSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, compute in SNAPSHOTS.items():
        path = SNAPSHOT_DIRECTORY / f'{name}.json'
        path.write_text(format_snapshot(compute()))
        print(f'Wrote {path}')


# ..........................................................................................
# Comparing with the reference

def assert_numbers_match(expected: Any, actual: Any, where: str) -> None:
    """Assert two (nested) lists of numbers have the same shape and agree to the snapshot tolerance."""
    expected_array, actual_array = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    assert expected_array.shape == actual_array.shape, f'{where}: shape {actual_array.shape}, expected {expected_array.shape}'
    np.testing.assert_allclose(actual_array, expected_array, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
                               err_msg=where)


ARCHITECTURE_REFERENCE = load_snapshot('architectures') if (SNAPSHOT_DIRECTORY / 'architectures.json').exists() else {}


def test_snapshot_covers_every_configuration():
    """A new architecture or hyperparameter changes the configurations pinned, so the snapshot must be regenerated"""
    cases = {case_id(architecture, values, task) for architecture, values in architecture_configurations()
             for task in SNAPSHOT_TASKS if task.datatype in get_architecture(architecture).supported_datatypes}
    assert cases == set(ARCHITECTURE_REFERENCE)


@pytest.mark.parametrize('case', sorted(ARCHITECTURE_REFERENCE))
def test_architecture_matches_snapshot(case):
    expected = ARCHITECTURE_REFERENCE[case]
    actual = snapshot_network(expected['architecture'], expected['settings'], Task.from_config(expected['task']))

    # Saved networks are loaded by parameter name and recorded hyperparameter values: these must match exactly
    assert actual['parameters'] == expected['parameters']
    assert actual['hyperparameters'] == expected['hyperparameters']
    for index, (expected_output, actual_output) in enumerate(zip(expected['outputs'], actual['outputs'])):
        assert_numbers_match(expected_output, actual_output, f'{case}, sequence {index}')


def test_search_matches_snapshot():
    expected, actual = load_snapshot('search'), search_snapshot()
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name]['stages'] == expected[name]['stages'], name
        assert actual[name]['best'] == expected[name]['best'], name
        assert len(actual[name]['trials']) == len(expected[name]['trials']), name
        for expected_trial, actual_trial in zip(expected[name]['trials'], actual[name]['trials']):
            where = f"{name}, trial {expected_trial['number']}"
            for field in ('number', 'origin', 'status', 'epochs', 'values'):
                assert actual_trial[field] == pytest.approx(expected_trial[field]), f'{where}: {field}'
            assert_numbers_match(expected_trial['stage_losses'], actual_trial['stage_losses'], where)


def test_training_matches_snapshot():
    expected, actual = load_snapshot('training'), training_snapshot()
    assert set(actual) == set(expected)
    for name in expected:
        for losses in ('train_losses', 'val_losses'):
            assert_numbers_match(expected[name][losses], actual[name][losses], f'{name}: {losses}')


if __name__ == '__main__':
    write_snapshots()
