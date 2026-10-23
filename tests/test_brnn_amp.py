"""BRNN AMP must fail clearly before data loading, updates or resume writes."""

import copy

import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import train_network as tn
from parrot.architectures import build_network
from parrot.scripts import common, parrot_optimize, parrot_train, training_parser
from parrot.scripts.search_artifacts import SearchArtifacts
from parrot.tasks import Task
from parrot.training_control import TrainingOptions
from parrot.training_persistence import atomic_save, load_training_checkpoint
from parrot.training_precision import TrainingPrecision
from tests.helpers import training_command
from tests.test_gradient_safety import make_loader
from tests.test_training_resume_cli import command, data_file


@pytest.mark.parametrize('module', [parrot_train, parrot_optimize])
@pytest.mark.parametrize('amp', [[], ['auto'], ['float16'], ['bfloat16']])
@pytest.mark.parametrize('datatype,num_classes', [('sequence', 1), ('residues', 1), ('residues', 2)])
def test_cli_rejects_brnn_amp_before_data_or_cuda(tmp_path, monkeypatch, run_cli, module, amp, datatype, num_classes):
    def unexpected(*args, **kwargs):
        pytest.fail('BRNN AMP must be rejected before loading data or querying CUDA')

    monkeypatch.setattr(common, 'prepare_training_run', unexpected)
    monkeypatch.setattr(torch.cuda, 'is_available', unexpected)
    with pytest.raises(ValueError, match='AMP is currently unsupported for BRNN.*--amp off'):
        run_cli(module.main, training_command(module.PROGRAM_NAME, 'brnn', 'missing.tsv', tmp_path / 'network',
                                               datatype, num_classes, '--device', 'cuda', '--amp', *amp))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('module', [parrot_train, parrot_optimize])
@pytest.mark.parametrize('amp', [[], ['--amp', 'off']])
def test_brnn_fp32_default_and_explicit_off_remain_valid(module, amp):
    args = training_parser.parse_arguments(module.build_parser(),
                                           training_command(module.PROGRAM_NAME, 'brnn', 'data.tsv', 'network',
                                                            'residues', 1, *amp)[1:])
    assert args.amp == 'off'
    assert training_parser.training_options_from_args(args) == {}


@pytest.mark.parametrize('mode', ['auto', 'float16', 'bfloat16'])
def test_python_training_and_search_reject_brnn_amp_before_updates(tmp_path, monkeypatch, mode):
    task = Task('residues', 1)
    network = build_network('brnn', 20, task, {'hidden_size': 2})
    before = copy.deepcopy(network.state_dict())
    path = tmp_path / 'checkpoint.pt'
    with pytest.raises(ValueError, match='AMP is currently unsupported for BRNN'):
        tn.train(network, make_loader(), make_loader(), task, 'cuda', .001, 1, silent=True,
                 training_options=TrainingOptions(amp=mode), checkpoint_path=str(path))
    assert not path.exists()
    assert all(torch.equal(before[key], value) for key, value in network.state_dict().items())

    def unexpected(*args, **kwargs):
        pytest.fail('Architecture rejection must precede device resolution')

    monkeypatch.setattr(hs, 'resolve_amp', unexpected)
    with pytest.raises(ValueError, match='AMP is currently unsupported for BRNN'):
        hs.TrialTrainer(hs.SearchSpace('brnn'), [], 20, task, 'cuda', training_options=TrainingOptions(amp=mode))


@pytest.mark.parametrize('entrypoint', ['epoch', 'batch', 'loss'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_level_runtime_cannot_bypass_brnn_restriction(entrypoint, dtype):
    task = Task('residues', 1)
    network = build_network('brnn', 20, task, {'hidden_size': 2})
    optimizer = torch.optim.Adam(network.parameters())
    precision = TrainingPrecision('off', 'cpu')
    precision.dtype = dtype
    batch = next(iter(make_loader()))
    criterion = tn.build_loss_criterion(task)
    with pytest.raises(ValueError, match='AMP is currently unsupported for BRNN'):
        if entrypoint == 'epoch':
            tn.train_epoch(network, [batch], task, criterion, optimizer, 'cpu', precision=precision)
        elif entrypoint == 'batch':
            tn.train_amp_batch(network, batch, task, criterion, optimizer, 'cpu', precision, 1, None, None, False)
        else:
            tn.compute_batch_loss(network, batch, task, criterion, 'cpu', precision=precision)
    assert not optimizer.state


@pytest.mark.parametrize('module', [parrot_train, parrot_optimize])
def test_cli_refuses_old_brnn_amp_checkpoints_even_with_off_override(tmp_path, monkeypatch, run_cli, module):
    output = tmp_path / 'network.pt'
    extra = ['--trials', '1'] if module is parrot_optimize else []
    run_cli(module.main, command(module.PROGRAM_NAME, data_file(tmp_path), output, *extra, '-e', '1'))
    path = tmp_path / 'network_training_checkpoint.pt'
    saved = load_training_checkpoint(str(path))
    original_output = output.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported resume must be rejected before data preparation')

    monkeypatch.setattr(common, 'prepare_training_run', unexpected)
    for mode in ('auto', 'float16', 'bfloat16'):
        saved['metadata']['command_args']['amp'] = mode
        saved['signature']['training_options']['amp'] = mode
        atomic_save(saved, str(path))
        original_checkpoint = path.read_bytes()
        for override in ([], ['--amp', 'off']):
            with pytest.raises(ValueError, match='BRNN AMP checkpoints cannot be resumed'):
                run_cli(module.main, [module.PROGRAM_NAME, '--resume', path, *override])
            assert path.read_bytes() == original_checkpoint
            assert output.read_bytes() == original_output


def test_python_resume_rejects_recorded_amp_before_changing_state(tmp_path):
    task = Task('residues', 1)
    network = build_network('brnn', 20, task, {'hidden_size': 2})
    path = str(tmp_path / 'checkpoint.pt')
    tn.train(network, make_loader(), make_loader(), task, 'cpu', .001, 1, silent=True, checkpoint_path=path)
    saved = load_training_checkpoint(path)
    saved['signature']['training_options']['amp'] = 'bfloat16'
    atomic_save(saved, path)
    before = copy.deepcopy(network.state_dict())
    with pytest.raises(ValueError, match='BRNN AMP checkpoints cannot be resumed'):
        tn.train(network, make_loader(), make_loader(), task, 'cpu', .001, 2, silent=True,
                 checkpoint_path=path, resume_from=path)
    assert all(torch.equal(before[key], value) for key, value in network.state_dict().items())

    trainer = hs.TrialTrainer(hs.SearchSpace('brnn'), [], 20, task, 'cpu')
    search = hs.HyperparameterSearch(trainer.space, trainer, 1, 1, silent=True)
    with pytest.raises(ValueError, match='BRNN AMP checkpoints cannot be resumed'):
        SearchArtifacts(search, None, 1, None, {'identity': {}}, saved)


@pytest.mark.parametrize('architecture', ['cnn', 'unet', 'transformer', 'composition', 'unet2d'])
@pytest.mark.parametrize('mode', ['auto', 'float16', 'bfloat16'])
def test_other_neural_architectures_still_accept_amp_policy(monkeypatch, architecture, mode):
    monkeypatch.setattr(hs, 'resolve_amp', lambda requested, device: torch.bfloat16)
    task = Task('matrix', 1, width=2) if architecture == 'unet2d' else Task('residues', 1)
    trainer = hs.TrialTrainer(hs.SearchSpace(architecture), [], 20, task, 'cuda',
                              training_options=TrainingOptions(amp=mode))
    assert trainer.options.amp == mode
