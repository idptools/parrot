"""Interrupt and resume real commands, including CV final training and search promotions."""

import csv

import pytest
import torch

from parrot import train_network as tn
from parrot.reproducibility import set_random_seed
from parrot.scripts import parrot_optimize, parrot_train
from parrot.training_persistence import load_training_checkpoint


def data_file(tmp_path):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    path = tmp_path / 'values.tsv'
    lines = []
    for i in range(18):
        sequence = (alphabet[i:] + alphabet[:i])[:3 + i % 6]
        values = ' '.join(str((j + i) % 7 / 7) for j in range(len(sequence)))
        lines.append(f'p{i} {sequence} {values}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def command(program, data, output, *extra):
    return [program, 'brnn', '--input-file', data, '--output-network', output, '--split', 'random',
            '-d', 'residues', '--regression', '-b', 4, '-e', 4, '-hs', 4, '-nl', 2, '--dropout', 0.25,
            '--checkpoint', '--training-log', '--silent', '--ignore-warnings', *extra]


def weights(path):
    return torch.load(path, map_location='cpu', weights_only=True)['state_dict']


def read_tsv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def assert_same_weights(first, second):
    for name, value in weights(first).items():
        assert torch.equal(value, weights(second)[name]), name


@pytest.mark.parametrize('program,module', [('parrot-train', parrot_train), ('parrot-optimize', parrot_optimize)])
@pytest.mark.parametrize('suffix', ['.pt', '_training_history.tsv', '_search_training_history.tsv'])
def test_checkpoint_cannot_overwrite_network_or_epoch_logs(tmp_path, run_cli, program, module, suffix):
    output = tmp_path / 'network.pt'
    checkpoint = tmp_path / ('network' + suffix)
    with pytest.raises(ValueError, match='different path'):
        run_cli(module.run, command(program, data_file(tmp_path), output, '--checkpoint', checkpoint))
    assert not checkpoint.exists()


@pytest.mark.parametrize('program,module', [('parrot-train', parrot_train), ('parrot-optimize', parrot_optimize)])
@pytest.mark.parametrize('custom_path', [False, True])
@pytest.mark.parametrize('new_destination', [False, True])
def test_resume_uses_supplied_checkpoint_from_another_directory(
        tmp_path, run_cli, monkeypatch, program, module, custom_path, new_destination):
    from parrot.scripts import search_artifacts
    original, elsewhere = tmp_path / 'original', tmp_path / 'elsewhere'
    original.mkdir()
    elsewhere.mkdir()
    (original / 'checkpoints').mkdir()
    monkeypatch.chdir(original)
    data, output = data_file(original), original / 'network.pt'
    options = ['--trials', 2] if program == 'parrot-optimize' else []
    checkpoint = original / 'network_training_checkpoint.pt'
    if custom_path:
        options += ['--checkpoint', 'checkpoints/state.pt']
        checkpoint = original / 'checkpoints/state.pt'
    saving = tn if program == 'parrot-train' else search_artifacts
    real_save = saving.atomic_save

    def interrupt(payload, path):
        real_save(payload, path)
        if payload['records']:
            raise RuntimeError('job interrupted')

    monkeypatch.setattr(saving, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='job interrupted'):
        run_cli(module.main, command(program, data, output, *options))
    monkeypatch.setattr(saving, 'atomic_save', real_save)
    saved_bytes = checkpoint.read_bytes()
    copied = elsewhere / 'copied.pt'
    copied.write_bytes(saved_bytes)
    monkeypatch.chdir(elsewhere)
    destination = elsewhere / 'continued.pt' if new_destination else copied
    overrides = ['--checkpoint', destination] if new_destination else []
    run_cli(module.main, [program, '--resume', copied, *overrides])
    resumed = load_training_checkpoint(str(destination))
    assert (resumed['epoch'] == 4 if program == 'parrot-train' else resumed['search_state']['phase'] == 'finished')
    assert checkpoint.read_bytes() == saved_bytes  # the original run's checkpoint was not selected
    if new_destination:
        assert copied.read_bytes() == saved_bytes
    assert not (elsewhere / 'checkpoints').exists()


def test_cli_resume_restores_unseeded_split_and_complete_training_args(tmp_path, run_cli, monkeypatch):
    data = data_file(tmp_path)
    reference, output = tmp_path / 'reference.pt', tmp_path / 'interrupted.pt'
    options = ['--regression-loss', 'huber', '--huber-delta', 0.4, '--optimizer', 'adamw', '--weight-decay', 0.02,
               '--lr-schedule', 'cosine', '--lr-warmup-epochs', 1, '--equal-protein-weighting']
    set_random_seed(53)
    run_cli(parrot_train.main, command('parrot-train', data, reference, *options))
    real_save = tn.atomic_save
    def interrupt(payload, path):
        real_save(payload, path)
        if payload['epoch'] == 2:
            raise RuntimeError('job killed')
    monkeypatch.setattr(tn, 'atomic_save', interrupt)
    set_random_seed(53)
    with pytest.raises(RuntimeError, match='job killed'):
        run_cli(parrot_train.main, command('parrot-train', data, output, *options))
    checkpoint = tmp_path / 'interrupted_training_checkpoint.pt'
    assert load_training_checkpoint(str(checkpoint))['metadata']['command_args']['seed'] is None
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    set_random_seed(999)
    run_cli(parrot_train.main, ['parrot-train', '--resume', checkpoint])
    assert_same_weights(reference, output)
    rows = read_tsv(tmp_path / 'interrupted_training_history.tsv')
    assert [int(row['epoch']) for row in rows] == [1, 2, 3, 4]
    assert (tmp_path / 'interrupted_per_protein_metrics.tsv').exists()
    assert (tmp_path / 'interrupted_length_metrics.tsv').exists()


@pytest.mark.parametrize('cv', [False, True])
def test_search_resume_restores_partial_fold_tpe_and_promotions(tmp_path, run_cli, monkeypatch, cv):
    data = data_file(tmp_path)
    reference, output = tmp_path / 'reference.pt', tmp_path / 'interrupted.pt'
    extra = ['--trials', 4, '--min-epochs', 2, '--optimizer', 'adamw', '--weight-decay', 0.01,
             '--regression-loss', 'huber', '--lr-schedule', 'plateau', '--lr-patience', 0]
    if cv:
        extra += ['--cv-folds', 2]
    set_random_seed(127)
    run_cli(parrot_optimize.main, command('parrot-optimize', data, reference, *extra))
    import parrot.scripts.search_artifacts as artifacts
    real_save = artifacts.atomic_save
    def interrupt(payload, path):
        real_save(payload, path)
        rows = payload['records']
        if rows and rows[-1]['trial'] == 2 and rows[-1]['epoch'] == 1:
            raise RuntimeError('job killed')
    monkeypatch.setattr(artifacts, 'atomic_save', interrupt)
    set_random_seed(127)
    with pytest.raises(RuntimeError, match='job killed'):
        run_cli(parrot_optimize.main, command('parrot-optimize', data, output, *extra))
    checkpoint = tmp_path / 'interrupted_training_checkpoint.pt'
    monkeypatch.setattr(artifacts, 'atomic_save', real_save)
    set_random_seed(999)
    run_cli(parrot_optimize.main, ['parrot-optimize', '--resume', checkpoint])
    assert_same_weights(reference, output)
    assert ((tmp_path / 'reference_optimal_hyperparams.txt').read_text()
            == (tmp_path / 'interrupted_optimal_hyperparams.txt').read_text())
    expected = read_tsv(tmp_path / 'reference_search_training_history.tsv')
    actual = read_tsv(tmp_path / 'interrupted_search_training_history.tsv')
    for row in (*expected, *actual):
        del row['seconds']
    assert actual == expected


def test_resume_during_final_cross_validated_retraining(tmp_path, run_cli, monkeypatch):
    data = data_file(tmp_path)
    reference, output = tmp_path / 'reference.pt', tmp_path / 'interrupted.pt'
    extra = ['--trials', 2, '--cv-folds', 2, '--seed', 54]
    run_cli(parrot_optimize.main, command('parrot-optimize', data, reference, *extra))
    real_save = tn.atomic_save
    def interrupt(payload, path):
        real_save(payload, path)
        if payload['epoch'] == 2:
            raise RuntimeError('job killed in final training')
    monkeypatch.setattr(tn, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='final training'):
        run_cli(parrot_optimize.main, command('parrot-optimize', data, output, *extra))
    checkpoint = tmp_path / 'interrupted_training_checkpoint.pt'
    assert load_training_checkpoint(str(checkpoint))['kind'] == 'train'
    monkeypatch.setattr(tn, 'atomic_save', real_save)
    run_cli(parrot_optimize.main, ['parrot-optimize', '--resume', checkpoint])
    assert_same_weights(reference, output)


def test_search_recovery_checkpoint_retains_reduced_rate_and_budget(tmp_path, run_cli, monkeypatch):
    data = data_file(tmp_path)
    output = tmp_path / 'interrupted.pt'
    import parrot.scripts.search_artifacts as artifacts
    real_save, real_epoch = artifacts.atomic_save, tn.train_epoch
    def failed_epoch(*args, **kwargs):
        raise tn.NonFiniteTrainingError('simulated gradient overflow')
    def interrupt(payload, target):
        real_save(payload, target)
        trials = payload['search_state']['trials']
        if trials and trials[0]['folds'] and trials[0]['folds'][0]['training_control']['recoveries_used']:
            raise RuntimeError('interrupted before replay')
    monkeypatch.setattr(tn, 'train_epoch', failed_epoch)
    monkeypatch.setattr(artifacts, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='before replay'):
        run_cli(parrot_optimize.main, command('parrot-optimize', data, output, '--trials', 2,
                                               '--recover-nonfinite', 1, '--seed', 14))
    checkpoint = tmp_path / 'interrupted_training_checkpoint.pt'
    saved = load_training_checkpoint(str(checkpoint))
    progress = saved['search_state']['trials'][0]['folds'][0]
    assert progress['training_control']['recoveries_used'] == 1
    assert progress['training_control']['scheduler_state']['recovery_scale'] == 0.5
    monkeypatch.setattr(tn, 'train_epoch', real_epoch)
    monkeypatch.setattr(artifacts, 'atomic_save', real_save)
    run_cli(parrot_optimize.main, ['parrot-optimize', '--resume', checkpoint])
    first = read_tsv(tmp_path / 'interrupted_search_training_history.tsv')[0]
    assert float(first['learning_rate']) == pytest.approx(0.0005)
    assert int(first['recoveries']) == 1


def test_changed_resume_inputs_and_policies_fail_before_any_artifact_write(tmp_path, run_cli):
    data = data_file(tmp_path)
    output = tmp_path / 'network.pt'
    run_cli(parrot_train.main, command('parrot-train', data, output))
    checkpoint = tmp_path / 'network_training_checkpoint.pt'
    old = {path: path.read_bytes() for path in tmp_path.iterdir() if path != data}
    with pytest.raises(ValueError, match='arguments changed'):
        run_cli(parrot_train.main, ['parrot-train', '--resume', checkpoint, '--weight-decay', 0.1])
    data.write_text(data.read_text().replace('0.0', '4.0', 1))
    with pytest.raises(ValueError, match='input changed'):
        run_cli(parrot_train.main, ['parrot-train', '--resume', checkpoint])
    assert all(path.read_bytes() == content for path, content in old.items())
