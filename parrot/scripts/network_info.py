"""
The <network>_info.txt file parrot-train and parrot-optimize write next to every
network: how the network was trained, in enough detail to reproduce it or to
describe it in the methods section of a paper.

The file has one section each for where the network came from (the command,
the software versions and the device), the data, the encoding, the
architecture, the training, the hyperparameter search (parrot-optimize only),
the results on the test set, how long training took and how fast the network
predicts, and the files written. info_sections() gathers them from a finished
run, and write_network_info() writes them.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import datetime
import hashlib
import os
import platform
import shlex
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from parrot import devices, nardini, plm, train_network
from parrot.architectures import ParrotArchitecture
from parrot.metrics import METRIC_DECIMALS
from parrot.package_version import PARROT_VERSION
from parrot.sequence_encoder import SequenceEncoder
from parrot.tasks import IGNORED_CLASS_LABEL, Task
from parrot.train_network import TrainingTiming

if TYPE_CHECKING:
    from parrot.scripts.common import TrainingRun

# Packages whose versions are recorded, when installed (the version of PARROT itself is always recorded)
RECORDED_PACKAGES = ('torch', 'numpy', 'scikit-learn', 'sparrow', 'transformers', 'esm')

# What each datatype is, in words
DATATYPE_DESCRIPTIONS = {
    'sequence': 'sequence-mapped: one value per sequence',
    'residues': 'residue-mapped: one value per residue',
    'matrix': 'matrix: a matrix of values per sequence, with one row per residue',
}

# Encoding settings with more values than this (e.g. a standardization's means) are summarized by their count
MAX_LISTED_VALUES = 10

# How the time a network was trained is shown
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Bytes read at a time when computing a file's checksum
CHECKSUM_CHUNK_BYTES = 1 << 20


class Section(NamedTuple):
    """One section of the info file: a title, 'name : value' entries, and optionally a block of text

    Attributes
    ----------
    title : str
    entries : list of (str, str)
    text : str
        Shown, indented, after the entries (e.g. the network's layers)
    """
    title: str
    entries: list[tuple[str, str]]
    text: str = ''


# .................................................................
# Gathering the sections

def info_sections(run: 'TrainingRun', network: ParrotArchitecture, training: Mapping[str, Any],
                  train_losses: Sequence[float], val_losses: Sequence[float], test_loss: float,
                  metrics: Mapping[str, float], written: Sequence[str],
                  search: list[tuple[str, str]] | None = None, timing: TrainingTiming | None = None,
                  prediction_seconds: float | None = None, search_seconds: float | None = None) -> list[Section]:
    """Every section of the info file, for a finished training run

    Parameters
    ----------
    run : TrainingRun
        The run (data, task, encoding and output files)
    network : ParrotArchitecture
        The trained network
    training : dict
        Training settings: 'learning_rate', 'epochs' and 'stop' (networks trained by gradient descent)
    train_losses, val_losses : list of float
        Average loss per sequence after each epoch (one value each for models fitted in one step)
    test_loss : float
    metrics : dict
        Test-set performance metrics
    written : list of str
        The other files written for this network
    search : list of (str, str), optional
        Description of the hyperparameter search (parrot-optimize)
    timing : TrainingTiming, optional
        How long each epoch of training took (see train_network.train())
    prediction_seconds : float, optional
        How long predicting the test set took
    search_seconds : float, optional
        How long the hyperparameter search took (parrot-optimize)

    Returns
    -------
    list of Section
    """
    sections = [provenance_section(run),
                data_section(run),
                encoding_section(run.encoder),
                architecture_section(network),
                training_section(network, run, training, train_losses, val_losses)]
    if search is not None:
        sections.append(Section('Hyperparameter search', search))
    sections += [results_section(val_losses, test_loss, metrics),
                 timing_section(network, run.device, timing, prediction_seconds, len(run.test_set), search_seconds,
                                embeddings_precomputed=isinstance(run.encoder, (plm.PLMEncoder, nardini.NardiniEncoder))),
                 Section('Files', [('network', run.network_file),
                                   *[('also written', path) for path in written]])]
    return sections


def provenance_section(run: 'TrainingRun') -> Section:
    """Where the network came from: when, the exact command, the software versions and the device

    Parameters
    ----------
    run : TrainingRun

    Returns
    -------
    Section
    """
    command = ' '.join(shlex.quote(argument) for argument in [run.program_name, *sys.argv[1:]])
    entries = [('trained', datetime.datetime.now().strftime(TIME_FORMAT)),
               ('command', command),
               ('PARROT version', PARROT_VERSION),
               ('Python version', platform.python_version())]
    entries += [(f'{package} version', version) for package, version in installed_versions(RECORDED_PACKAGES)]
    entries += [('device', devices.describe_device(run.device)),
                ('precision', 'float32 throughout (TF32 turned off on NVIDIA GPUs)'),
                ('random seed', str(run.seed) if run.seed is not None else 'not set, so training again will '
                                                                           'not give exactly the same network')]
    return Section('Provenance', entries)


def installed_versions(packages: Sequence[str]) -> list[tuple[str, str]]:
    """The version of each of `packages` that is installed (those that are not are left out)

    Parameters
    ----------
    packages : list of str
        Distribution names, e.g. 'scikit-learn'

    Returns
    -------
    list of (str, str)
        (package, version)
    """
    versions = []
    for package in packages:
        try:
            versions.append((package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            continue
    return versions


def data_section(run: 'TrainingRun') -> Section:
    """The data: the file (and its checksum), the task, the split and a summary of the sequences and values

    Parameters
    ----------
    run : TrainingRun

    Returns
    -------
    Section
    """
    task = run.task
    datasets = (run.train_set, run.val_set, run.test_set)
    lengths = np.array([len(sequence) for dataset in datasets for sequence in dataset.sequences])
    total = sum(len(dataset) for dataset in datasets)

    entries = [('input file', run.data_file),
               ('input file SHA-256', file_sha256(run.data_file)),
               ('datatype', DATATYPE_DESCRIPTIONS[task.datatype]),
               ('task', describe_task(task)),
               ('sequences', f'{total:,}: {len(run.train_set):,} training, {len(run.val_set):,} validation, '
                             f'{len(run.test_set):,} test'),
               ('split', run.split_description),
               ('sequence length', f'{lengths.min()}-{lengths.max()} residues (median {np.median(lengths):g})')]
    if task.is_matrix:
        entries.append(('matrices', f'{"square maps" if task.width is None else f"{task.width} columns"}'
                                    f'{", symmetric" if task.symmetric else ""}'))
    entries.append(describe_values(task, [value for dataset in datasets for value in dataset.values]))
    return Section('Data', entries)


def file_sha256(path: str) -> str:
    """The SHA-256 checksum of a file, which identifies exactly which data a network was trained on

    Parameters
    ----------
    path : str

    Returns
    -------
    str
        Hexadecimal digest
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(CHECKSUM_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe_task(task: Task) -> str:
    """'regression', or 'classification into N classes (labelled 0 to N-1)'."""
    if task.is_classification:
        return f'classification into {task.num_classes} classes (labelled 0 to {task.num_classes - 1})'
    return 'regression'


def describe_values(task: Task, values: Sequence[Any]) -> tuple[str, str]:
    """The values to predict, summarized: each class's share, or the mean, spread and range

    Missing labels (nan) are left out and counted.

    Parameters
    ----------
    task : Task
    values : list
        Every sequence's value(s), as stored in the datasets

    Returns
    -------
    (str, str)
        A 'name : value' entry
    """
    pooled = np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])
    unit = 'sequence' if task.datatype == 'sequence' else 'residue' if task.datatype == 'residues' else 'cell'

    if task.is_classification:
        missing = pooled == IGNORED_CLASS_LABEL
        labels = pooled[~missing].astype(int)
        counts = np.bincount(labels, minlength=task.num_classes)
        shares = ', '.join(f'{label}: {count:,} ({100 * count / len(labels):.1f}%)' for label, count in enumerate(counts))
        description = f'per {unit}: {shares}'
    else:
        missing = np.isnan(pooled)
        known = pooled[~missing]
        description = (f'per {unit}: mean {known.mean():.4g}, standard deviation {known.std():.4g}, range '
                       f'{known.min():.4g} to {known.max():.4g}')

    if missing.any():
        description += f'; {int(missing.sum()):,} missing ({unit} labels written as nan, which are ignored)'
    return ('classes' if task.is_classification else 'values', description)


def encoding_section(encoder: SequenceEncoder) -> Section:
    """How residues were encoded: the encoding, the values per residue, and its saved settings

    Parameters
    ----------
    encoder : SequenceEncoder

    Returns
    -------
    Section
    """
    entries = [('encoding', encoder.name), ('values per residue', str(len(encoder)))]
    entries += [(name, value) for name, value in flatten_settings(encoder.to_config()) if name != 'name']
    return Section('Encoding', entries)


def flatten_settings(settings: Mapping[str, Any], prefix: str = '') -> list[tuple[str, str]]:
    """Saved settings as 'name : value' entries: nested settings get dotted names, and long lists and tables their length

    Parameters
    ----------
    settings : dict
    prefix : str, optional
        Prefix for every name (for nested settings)

    Returns
    -------
    list of (str, str)
    """
    entries = []
    for name, value in settings.items():
        if isinstance(value, Mapping):
            if len(value) > MAX_LISTED_VALUES and all(not isinstance(item, Mapping) for item in value.values()):
                entries.append((prefix + name, f'({len(value)} entries)'))
            else:
                entries += flatten_settings(value, f'{prefix}{name}.')
        elif isinstance(value, (list, tuple)) and len(value) > MAX_LISTED_VALUES:
            entries.append((prefix + name, f'({len(value)} values)'))
        else:
            entries.append((prefix + name, str(value)))
    return entries


def architecture_section(network: ParrotArchitecture) -> Section:
    """The architecture: its hyperparameters, its number of parameters, and its layers

    Parameters
    ----------
    network : ParrotArchitecture

    Returns
    -------
    Section
    """
    entries = [('architecture', f'{network.name}: {network.description}')]
    without_effect = network.settings_without_effect()
    entries += [(name, f'{value} (no effect: {without_effect[name]})' if name in without_effect else str(value))
                for name, value in network.hyperparameter_values.items()]

    if not network.trained_by_gradient_descent:
        entries.append(('model', 'fitted in one step with scikit-learn (see below)'))
        return Section('Architecture', entries, text=repr(getattr(network, 'estimator', None)))

    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    entries.append(('parameters', f'{parameters:,} ({trainable:,} trainable)'))
    return Section('Architecture', entries, text=str(network))


def training_section(network: ParrotArchitecture, run: 'TrainingRun', training: Mapping[str, Any],
                     train_losses: Sequence[float], val_losses: Sequence[float]) -> Section:
    """How the network was trained: the loss, the optimizer, the batches, the epochs and which epoch was kept

    Parameters
    ----------
    network : ParrotArchitecture
    run : TrainingRun
    training : dict
        'learning_rate', 'epochs' and 'stop'
    train_losses, val_losses : list of float

    Returns
    -------
    Section
    """
    if not network.trained_by_gradient_descent:
        return Section('Training', [('training', 'fitted in one step to the training set; the validation set is '
                                                 'only used to report the validation loss')])

    best = best_epoch(val_losses)
    epochs = training['epochs']
    stop = (f'after {epochs} epochs' if training['stop'] == 'iter' else
            f'once the validation loss had not improved by at least '
            f'{100 * train_network.AUTO_STOP_RELATIVE_IMPROVEMENT:g}% for {epochs} epochs in a row (--stop auto)')
    entries = [('loss', describe_loss(run.task)),
               ('optimizer', describe_optimizer(training['learning_rate'])),
               ('batch size', f'{run.batch_size} sequences'),
               ('batching', 'sequences of similar length batched together, with new batches, in a new random '
                            'order, every epoch' if run.batching == 'length' else 'random batches every epoch'),
               ('stopping', f'training stopped {stop}'),
               ('epochs trained', str(len(val_losses)))]
    if best is not None:
        entries += [('epoch kept', f'{best + 1}, the epoch with the lowest validation loss (the network after each '
                                   'epoch is evaluated on the validation set)'),
                    ('training loss at that epoch', f'{train_losses[best]:.4f}')]
    return Section('Training', entries)


def best_epoch(val_losses: Sequence[float]) -> int | None:
    """The epoch (counting from 0) with the lowest validation loss, first if tied; None if no loss is finite."""
    losses = np.asarray(val_losses, dtype=float)
    if not np.isfinite(losses).any():
        return None
    return int(np.nanargmin(np.where(np.isfinite(losses), losses, np.nan)))


def describe_loss(task: Task) -> str:
    """The loss function training minimizes (see train_network.build_loss_criterion)."""
    if task.is_classification:
        loss = 'cross-entropy'
    elif task.datatype == 'sequence':
        loss = 'L1 (absolute error)'
    else:
        loss = 'mean squared error'
    where = {'sequence': 'over the sequences', 'residues': 'over every labelled residue',
             'matrix': 'over every labelled cell'}[task.datatype]
    return f'{loss}, summed {where} of a batch; losses are reported as the average per sequence'


def describe_optimizer(learning_rate: float) -> str:
    """The optimizer and its settings: Adam with PyTorch's defaults but for the learning rate."""
    defaults = torch.optim.Adam([torch.zeros(1)], lr=learning_rate).defaults
    beta1, beta2 = defaults['betas']
    return (f'Adam, learning rate {learning_rate:g} (constant), betas ({beta1:g}, {beta2:g}), eps {defaults["eps"]:g}, '
            f'weight decay {defaults["weight_decay"]:g}')


def results_section(val_losses: Sequence[float], test_loss: float, metrics: Mapping[str, float]) -> Section:
    """The network's losses and its performance on the test set

    Parameters
    ----------
    val_losses : list of float
    test_loss : float
    metrics : dict

    Returns
    -------
    Section
    """
    best = best_epoch(val_losses)
    entries = []
    if best is not None:
        entries.append(('validation loss', f'{val_losses[best]:.4f}'))
    entries.append(('test loss', f'{test_loss:.4f}'))
    entries += [(f'test {name}', f'{value:.{METRIC_DECIMALS}f}') for name, value in metrics.items()]
    return Section('Results', entries)


# .................................................................
# Writing the file

def format_duration(seconds: float) -> str:
    """A duration in words, as precise as is useful for its length

    Parameters
    ----------
    seconds : float
        The duration, in seconds (at least 0)

    Returns
    -------
    str
        '0.42s' under a second, '4.2s' under ten seconds, '42s' under a
        minute, '5m 07s' under an hour, and '2h 03m' beyond. Each tier is
        chosen after rounding, so 9.97 s is '10s', not '10.0s'.
    """
    if round(seconds, 2) < 1:
        return f'{seconds:.2f}s'
    if round(seconds, 1) < 10:
        return f'{seconds:.1f}s'
    whole_seconds = round(seconds)
    if whole_seconds < 60:
        return f'{whole_seconds}s'
    minutes, whole_seconds = divmod(whole_seconds, 60)
    if minutes < 60:
        return f'{minutes}m {whole_seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def timing_section(network: ParrotArchitecture, device: torch.device, timing: TrainingTiming | None,
                   prediction_seconds: float | None, test_sequences: int, search_seconds: float | None = None,
                   embeddings_precomputed: bool = False) -> Section:
    """How long training (and any hyperparameter search) took, and how fast the network trains and predicts

    Models fitted in one step (scikit-learn) fit and predict on the CPU
    whatever device was chosen, so their times are given as on the CPU.

    Parameters
    ----------
    network : ParrotArchitecture
        The trained network
    device : torch.device
        The device it was trained and tested on
    timing : TrainingTiming or None
        How long each epoch took (None if the network's training was not timed,
        e.g. the best trial of a hyperparameter search)
    prediction_seconds : float or None
        How long predicting the test set took
    test_sequences : int
        Sequences in the test set
    search_seconds : float, optional
        How long the hyperparameter search took
    embeddings_precomputed : bool, optional
        True if the test sequences' encodings (protein language model or
        NARDINI) were computed before training, so prediction only read them

    Returns
    -------
    Section
    """
    where = devices.describe_device(device) if network.trained_by_gradient_descent else 'CPU (scikit-learn)'
    entries = []
    if search_seconds is not None:
        entries.append(('search time', f'{format_duration(search_seconds)} (every trial, on {where})'))
    if timing is not None and timing.train_seconds:
        if network.trained_by_gradient_descent:
            epochs = len(timing.train_seconds)
            epochs_text = f'{epochs} epoch' if epochs == 1 else f'{epochs} epochs'
            entries += [('training time', f'{format_duration(timing.total_seconds)} for {epochs_text}, on {where}'),
                        ('time per epoch', f'{format_duration(timing.total_seconds / epochs)} on average: '
                                           f'{format_duration(sum(timing.train_seconds) / epochs)} training and '
                                           f'{format_duration(sum(timing.val_seconds) / epochs)} measuring the '
                                           'validation loss'),
                        ('training speed', f'{timing.training_speed:,.0f} sequences per second')]
        else:
            entries.append(('fitting time', f'{format_duration(timing.train_seconds[0])} '
                                            f'({timing.train_sequences[0]:,} sequences), on {where}'))
    if prediction_seconds is not None and prediction_seconds > 0:
        if embeddings_precomputed:
            how = ('from embeddings computed before training; computing the embeddings of new sequences, as '
                   'parrot-predict does, takes much longer')
        else:
            how = 'including reading and encoding them'
        entries.append(('prediction speed', f'{test_sequences / prediction_seconds:,.0f} sequences per second on '
                                            f'{where} (predicting the {test_sequences:,} test sequences, {how})'))
    return Section('Timing', entries)


def write_network_info(path: str, network_file: str, sections: Sequence[Section]) -> None:
    """Write the info file

    Parameters
    ----------
    path : str
        File to write
    network_file : str
        The network it describes
    sections : list of Section
    """
    title = f'PARROT network: {os.path.basename(network_file)}'
    lines = [title, '=' * len(title), '']
    for section in sections:
        lines += [section.title, '-' * len(section.title)]
        width = max((len(name) for name, _ in section.entries), default=0)
        lines += [f'{name:<{width}} : {value}' for name, value in section.entries]
        if section.text:
            lines += ['', *[f'    {line}' for line in section.text.splitlines()]]
        lines.append('')

    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
