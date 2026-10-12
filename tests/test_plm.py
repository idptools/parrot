"""
Tests for protein language model (PLM) encodings (parrot/plm.py).

Most tests replace the PLM with a fake model that gives each amino acid a fixed
random vector, so they are fast and exact. test_real_esm2_model runs the
smallest real ESM-2 model, and is skipped unless its weights are already
downloaded (tests never download anything).
"""

import importlib.util
import subprocess
import sys
import threading
from collections import namedtuple

import numpy as np
import pytest
import torch

from parrot import encode_sequence, plm
from parrot.architectures import build_network
from parrot.checkpoint import load_network
from parrot.py_predictor import Predictor
from parrot.scripts import parrot_predict, parrot_train
from parrot.tasks import Task

from tests.helpers import EXAMPLE_TASKS, FULL_PRECISION, TF32_EVERYWHERE, data_path, tf32_settings, training_command

MODEL = 'esm2_t6_8M_UR50D'
EMBEDDING_SIZE = plm.PLM_MODELS[MODEL].embedding_size
BYTES_PER_VALUE = 2

# The fake model's vector for each amino acid
FAKE_TABLE = {amino_acid: np.random.default_rng(index).normal(size=EMBEDDING_SIZE).astype(np.float32)
              for index, amino_acid in enumerate('ACDEFGHIKLMNPQRSTVWY')}


def fake_run_model(self: plm.PLMEncoder, chunks: list[str]) -> list[np.ndarray]:
    """Stand-in for PLMEncoder._run_model: each residue's vector depends only on its amino acid."""
    fake_run_model.calls += 1
    return [np.stack([FAKE_TABLE[residue] for residue in chunk]) for chunk in chunks]


fake_run_model.calls = 0

# Shape of the value returned by shutil.disk_usage
DiskUsage = namedtuple('DiskUsage', 'total used free')


@pytest.fixture
def fake_plm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real PLM with the fake one (nothing is loaded or downloaded)."""
    fake_run_model.calls = 0
    monkeypatch.setattr(plm.PLMEncoder, '_run_model', fake_run_model)
    monkeypatch.setattr(plm.PLMEncoder, '_load_model', lambda self: (None, None))


def fake_embedding(sequence: str) -> np.ndarray:
    """What the fake PLM should give a sequence, after rounding to the storage type."""
    return np.stack([FAKE_TABLE[residue] for residue in sequence]).astype(plm.STORAGE_DTYPE)


# ..........................................................................................
# Windows and batches

@pytest.mark.parametrize('length', [1, 100, 1022, 1023, 2000, 2044, 5654])
def test_windows_cover_every_residue(length):
    starts = plm.window_starts(length, 1022, 256)
    covered = np.zeros(length, dtype=int)
    for start in starts:
        covered[start:start + 1022] += 1

    assert starts[0] == 0
    assert covered.min() >= 1
    assert starts[-1] + min(1022, length) == length
    assert all(later - earlier <= 1022 - 256 for earlier, later in zip(starts, starts[1:]))


def test_short_sequences_have_one_window():
    assert plm.window_starts(500, 1022, 256) == [0]


def test_cache_write_groups_hold_every_sequence_in_length_order():
    sequences = ['A' * length for length in (5, 1, 8, 3, 2, 9)]
    groups = plm.cache_write_groups(sequences, residues_per_group=10)

    assert [sequence for group in groups for sequence in group] == sorted(sequences, key=len)
    # A group closes as soon as it reaches 10 residues (1 + 2 + 3 + 5), and the last holds what is left (8 + 9)
    assert [sum(len(sequence) for sequence in group) for group in groups] == [11, 17]


def test_token_batches_respect_the_budget():
    pieces = [('s', 0, 'A' * length) for length in sorted([5, 7, 7, 10, 30, 31, 60, 200])]
    batches = plm.token_batches(pieces, max_tokens=64)

    assert [piece for batch in batches for piece in batch] == pieces
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(len(piece[2]) for piece in batch) <= 64


# ..........................................................................................
# Embedding

def test_embed_gives_float16_per_residue_embeddings(fake_plm):
    encoder = plm.PLMEncoder(MODEL)
    embeddings = encoder.embed(['ACDEF', 'KKK', 'ACDEF'])

    assert list(embeddings) == ['ACDEF', 'KKK']
    assert embeddings['ACDEF'].shape == (5, EMBEDDING_SIZE) and embeddings['ACDEF'].dtype == plm.STORAGE_DTYPE
    np.testing.assert_array_equal(embeddings['KKK'], fake_embedding('KKK'))


def test_long_sequences_are_stitched_from_windows(fake_plm):
    sequence = ''.join(np.random.default_rng(1).choice(list(FAKE_TABLE), size=2500))
    encoder = plm.PLMEncoder(MODEL)

    embedding = encoder.embed([sequence])[sequence]

    # The fake model ignores context, so averaging overlapping windows must give back each residue's vector
    np.testing.assert_allclose(embedding.astype(np.float32), fake_embedding(sequence).astype(np.float32),
                               atol=1e-3)


def test_overlapping_windows_are_averaged(monkeypatch):
    # A fake model whose output is each residue's position within its window
    def positions(self, chunks):
        return [np.repeat(np.arange(len(chunk), dtype=np.float32)[:, None], EMBEDDING_SIZE, axis=1)
                for chunk in chunks]
    monkeypatch.setattr(plm.PLMEncoder, '_run_model', positions)
    encoder = plm.PLMEncoder(MODEL, window=10, overlap=4)

    embedding = encoder.embed(['A' * 16])['A' * 16][:, 0].astype(np.float32)

    # Windows start at 0 and 6: residues 6-9 are positions 6-9 of the first window and 0-3 of the second
    np.testing.assert_allclose(embedding[:6], np.arange(6))
    np.testing.assert_allclose(embedding[6:10], (np.arange(6, 10) + np.arange(4)) / 2)
    np.testing.assert_allclose(embedding[10:], np.arange(4, 10))


# ..........................................................................................
# The cache

def test_cache_round_trip(tmp_path):
    cache = plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE)
    assert cache.missing(['ACD', 'EF', 'ACD']) == ['ACD', 'EF']

    cache.write({'ACD': fake_embedding('ACD'), 'EF': fake_embedding('EF')})

    assert cache.missing(['ACD', 'EF', 'GH']) == ['GH']
    np.testing.assert_array_equal(cache.get('ACD'), fake_embedding('ACD'))
    assert cache.summary()[:2] == (2, 5)
    with pytest.raises(KeyError):
        cache.get('GH')


def test_cache_rejects_a_different_model(tmp_path):
    plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE).write({'ACD': fake_embedding('ACD')})
    renamed = tmp_path / 'esm2_t12_35M_UR50D.h5'
    (tmp_path / f'{MODEL}.h5').rename(renamed)

    with pytest.raises(ValueError, match='different model'):
        plm.EmbeddingCache(tmp_path, 'esm2_t12_35M_UR50D', 480).write({'EF': np.zeros((2, 480))})


def test_cache_writer_waits_for_another_writer(tmp_path, capsys):
    lock = tmp_path / f'{MODEL}.h5.lock'
    lock.write_text('pid 1\n')
    threading.Timer(0.5, lock.unlink).start()  # the other writer finishes after half a second

    plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE).write({'ACD': fake_embedding('ACD')})

    assert 'Waiting for another PARROT process' in capsys.readouterr().out
    assert plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE).missing(['ACD']) == []


def test_cache_gives_up_on_a_lock_left_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(plm, 'CACHE_WAIT_SECONDS', 0.3)
    (tmp_path / f'{MODEL}.h5.lock').write_text('pid 1\n')
    with pytest.raises(RuntimeError, match='delete that lock file'):
        plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE).write({'ACD': fake_embedding('ACD')})


# Run in another process: hold the cache open for reading (as a training run does), say so, then wait to be told
# that the main process has written a new embedding, and read it
CACHE_READER_SCRIPT = """
import sys
from parrot import plm
cache = plm.EmbeddingCache(sys.argv[1], sys.argv[2], int(sys.argv[3]))
cache.get('ACD')
print('reading', flush=True)
sys.stdin.readline()
print(cache.get('EF').shape, flush=True)
"""


def test_cache_can_be_written_while_another_process_reads_it(tmp_path):
    # A training run holds the cache open for as long as it trains; another run must still be able to add to it
    cache = plm.EmbeddingCache(tmp_path, MODEL, EMBEDDING_SIZE)
    cache.write({'ACD': fake_embedding('ACD')})
    reader = subprocess.Popen([sys.executable, '-c', CACHE_READER_SCRIPT, str(tmp_path), MODEL, str(EMBEDDING_SIZE)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert reader.stdout.readline().strip() == 'reading'
        cache.write({'EF': fake_embedding('EF')})
        output, _ = reader.communicate('written\n', timeout=60)
    finally:
        reader.kill()
    assert reader.returncode == 0 and output.strip() == f'(2, {EMBEDDING_SIZE})'



def test_prepare_reports_size_and_embeds_each_sequence_once(fake_plm, tmp_path, capsys):
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path / 'cache')
    sequences = ['ACDEFGHIK', 'KLMNPQ', 'ACDEFGHIK']

    encoder.prepare(sequences, 'parrot-train')

    expected_bytes = 15 * EMBEDDING_SIZE * BYTES_PER_VALUE
    assert f'about {plm.human_size(expected_bytes)}' in capsys.readouterr().out
    calls = fake_run_model.calls
    encoder.prepare(sequences, 'parrot-train')
    assert fake_run_model.calls == calls
    assert 'Every sequence is already embedded' in capsys.readouterr().out

    readme = (tmp_path / 'cache' / 'README.md').read_text()
    assert f'| `{MODEL}.h5` | {MODEL} | {EMBEDDING_SIZE} per residue | 2 | 15 |' in readme
    assert 'last updated by `parrot-train`' in readme


def test_prepare_checks_disk_space(fake_plm, tmp_path, monkeypatch):
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path)
    monkeypatch.setattr(plm.shutil, 'disk_usage', lambda path: DiskUsage(total=1000, used=900, free=100))

    with pytest.raises(RuntimeError, match='Not enough disk space'):
        encoder.prepare(['ACDEFGHIK'], 'parrot-train')


def test_encoding_is_the_same_with_and_without_a_cache(fake_plm, tmp_path):
    without_cache = plm.PLMEncoder(MODEL)
    with_cache = plm.PLMEncoder(MODEL)
    with_cache.use_cache(tmp_path)

    first = without_cache.encode('MKWVTF')
    np.testing.assert_array_equal(with_cache.encode('MKWVTF').numpy(), first.numpy())
    np.testing.assert_array_equal(with_cache.encode('MKWVTF').numpy(), first.numpy())  # now read from the cache
    assert first.dtype == torch.float32


# ..........................................................................................
# Choosing and saving the encoding

@pytest.mark.parametrize('sequence', ['mkwvtf', 'MKWvTF', 'MKJWTF', 'MK-WTF', 'MK1WTF'])
def test_residues_the_model_cannot_read_are_rejected(fake_plm, sequence):
    # The tokenizers read these as unknown tokens (and ESM-2 reads a run of lowercase letters as a
    # single token), which would silently give wrong or misaligned embeddings
    with pytest.raises(ValueError, match='Invalid amino acid detected'):
        plm.PLMEncoder(MODEL).embed(['ACDEF', sequence])


def test_prepare_checks_every_sequence_before_embedding(fake_plm, tmp_path):
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path)
    with pytest.raises(ValueError, match='Invalid amino acid detected: x'):
        encoder.prepare(['ACDEF', 'ACDEFGHIK', 'ACDxF'], 'test', silent=True)
    assert fake_run_model.calls == 0
    assert not (tmp_path / f'{MODEL}.h5').exists()


def test_embedding_runs_in_full_precision(fake_plm, tf32_allowed, monkeypatch):
    seen = []
    monkeypatch.setattr(plm.PLMEncoder, '_run_model',
                        lambda self, chunks: seen.append(tf32_settings()) or fake_run_model(self, chunks))
    plm.PLMEncoder(MODEL).embed(['ACDEF', 'GHIKLMN'])
    assert seen and all(settings == FULL_PRECISION for settings in seen)
    assert tf32_settings() == TF32_EVERYWHERE


def test_get_encoder_recognises_plm_names():
    encoder = encode_sequence.get_encoder(MODEL)
    assert isinstance(encoder, plm.PLMEncoder) and len(encoder) == EMBEDDING_SIZE


def test_plm_encodings_cannot_be_standardized():
    with pytest.raises(ValueError, match='cannot be used with a protein language model'):
        encode_sequence.get_encoder(MODEL, standardize=True)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match='Unknown protein language model'):
        plm.PLMEncoder('esm2_t99_1T_UR50D')


@pytest.mark.parametrize('model, size, window', [('esm2_t12_35M_UR50D', 480, 1022), ('esmc_300m', 960, 2046),
                                                 ('esmc_600m', 1152, 2046)])
def test_model_settings(model, size, window):
    encoder = encode_sequence.get_encoder(model)
    assert (encoder.input_size, encoder.window) == (size, window)
    rebuilt = encode_sequence.encoder_from_config(encoder.to_config())
    assert (rebuilt.name, rebuilt.input_size, rebuilt.window) == (model, size, window)


def test_plm_config_round_trip():
    encoder = plm.PLMEncoder(MODEL, window=500, overlap=100)
    rebuilt = encode_sequence.encoder_from_config(encoder.to_config())

    assert isinstance(rebuilt, plm.PLMEncoder)
    assert (rebuilt.name, rebuilt.window, rebuilt.overlap) == (MODEL, 500, 100)


def test_dipeptide_features_need_a_small_encoding():
    with pytest.raises(ValueError, match='dipeptides'):
        build_network('linear', EMBEDDING_SIZE, Task('sequence', 1), {'features': 'dipeptides'})


# ..........................................................................................
# Commands

def train(run_cli, network_file, *extra):
    """Train briefly on the residue-regression example data with the PLM encoding."""
    filename, datatype, num_classes = EXAMPLE_TASKS['res_regress']
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(filename), network_file, datatype,
                                                num_classes, '-e', '2', '--encode', MODEL, '--seed', '1', '--silent',
                                                *extra))


def test_train_and_predict_with_a_plm(fake_plm, run_cli, tmp_path):
    (tmp_path / 'train').mkdir()
    train(run_cli, tmp_path / 'train' / 'net.pt')

    train_cache = tmp_path / 'train' / plm.TRAIN_CACHE_DIRECTORY
    assert (train_cache / f'{MODEL}.h5').is_file() and (train_cache / 'README.md').is_file()
    network, encoder = load_network(str(tmp_path / 'train' / 'net.pt'))
    assert isinstance(encoder, plm.PLMEncoder) and network.input_size == EMBEDDING_SIZE

    # parrot-predict embeds into its own cache next to its output, and reproduces the test-set predictions
    lines = (tmp_path / 'train' / 'net_predictions.tsv').read_text().splitlines()
    sequences = [line.split()[1] for line in lines[1::2]]
    expected = [np.array(line.split()[2:], dtype=float) for line in lines[1::2]]
    seq_file = tmp_path / 'sequences.txt'
    seq_file.write_text(''.join(f's{i} {sequence}\n' for i, sequence in enumerate(sequences)))
    (tmp_path / 'predict').mkdir()
    run_cli(parrot_predict.main, ['parrot-predict', seq_file, tmp_path / 'train' / 'net.pt',
                                  tmp_path / 'predict' / 'out.txt', '--silent'])

    assert (tmp_path / 'predict' / plm.PREDICT_CACHE_DIRECTORY / f'{MODEL}.h5').is_file()
    predicted = [np.array(line.split()[2:], dtype=float) for line in
                 (tmp_path / 'predict' / 'out.txt').read_text().splitlines()]
    for found, wanted in zip(predicted, expected):
        np.testing.assert_allclose(found, wanted, atol=1e-4)

    # Predictor embeds on the fly by default, and agrees
    np.testing.assert_allclose(Predictor(tmp_path / 'train' / 'net.pt').predict(sequences[0]).ravel(), expected[0],
                               atol=1e-4)


def test_embedding_cache_can_be_shared(fake_plm, run_cli, tmp_path):
    shared = tmp_path / 'shared_cache'
    train(run_cli, tmp_path / 'net1.pt', '--embedding-cache', shared)
    calls = fake_run_model.calls
    train(run_cli, tmp_path / 'net2.pt', '--embedding-cache', shared)

    assert fake_run_model.calls == calls  # the second run embedded nothing
    assert not (tmp_path / plm.TRAIN_CACHE_DIRECTORY).exists()


def test_no_cache_for_other_encodings(run_cli, tmp_path):
    filename, datatype, num_classes = EXAMPLE_TASKS['res_regress']
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(filename), tmp_path / 'net.pt',
                                                datatype, num_classes, '-e', '1', '--silent'))
    assert not (tmp_path / plm.TRAIN_CACHE_DIRECTORY).exists()

    with pytest.raises(ValueError, match='only applies to protein language model'):
        run_cli(parrot_train.main, training_command('parrot-train', 'brnn', data_path(filename), tmp_path / 'net2.pt',
                                                    datatype, num_classes, '-e', '1', '--silent', '--embedding-cache',
                                                    tmp_path / 'c'))
    with pytest.raises(ValueError, match='only applies to protein language model'):
        run_cli(parrot_predict.main, ['parrot-predict', data_path('seqfile.txt'), tmp_path / 'net.pt',
                                      tmp_path / 'out.txt', '--silent', '--embedding-cache', tmp_path / 'c'])


# ..........................................................................................
# The real model

def real_model_available() -> bool:
    """True if transformers is installed and the smallest ESM-2 model is already downloaded."""
    if importlib.util.find_spec('transformers') is None:
        return False
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(f'facebook/{MODEL}', 'config.json'), str)


@pytest.mark.skipif(not real_model_available(), reason=f'{MODEL} is not downloaded')
def test_real_esm2_model(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    encoder = plm.PLMEncoder(MODEL)
    long_sequence = 'MKWVTFISLLFLFSSAYS' * 70  # 1,260 residues: two windows

    embeddings = encoder.embed(['ACDEFGHIKLMNPQRSTVWY', long_sequence])

    assert embeddings['ACDEFGHIKLMNPQRSTVWY'].shape == (20, EMBEDDING_SIZE)
    assert embeddings[long_sequence].shape == (1260, EMBEDDING_SIZE)
    assert np.all(np.isfinite(embeddings[long_sequence].astype(np.float32)))

    # Embedding a sequence alone or in a batch with others gives the same result
    alone = encoder.embed(['ACDEFGHIKLMNPQRSTVWY'])['ACDEFGHIKLMNPQRSTVWY'].astype(np.float32)
    np.testing.assert_allclose(alone, embeddings['ACDEFGHIKLMNPQRSTVWY'].astype(np.float32), atol=2e-2)


def esmc_available() -> bool:
    """True if EvolutionaryScale's esm package is installed and ESM C 300M is already downloaded."""
    if importlib.util.find_spec('esm') is None or importlib.util.find_spec('esm.pretrained') is None:
        return False
    from huggingface_hub import try_to_load_from_cache
    weights = try_to_load_from_cache('EvolutionaryScale/esmc-300m-2024-12', 'data/weights/esmc_300m_2024_12_v0.pth')
    return isinstance(weights, str)


@pytest.mark.skipif(not esmc_available(), reason='ESM C 300M is not downloaded')
def test_real_esmc_model(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    encoder = plm.PLMEncoder('esmc_300m')
    short, medium = 'ACDEFGHIKLMNPQRSTVWY', 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF'

    embeddings = encoder.embed([short, medium])

    assert embeddings[short].shape == (20, 960) and embeddings[medium].shape == (len(medium), 960)

    # Embedding in a padded batch matches EvolutionaryScale's own forward pass on each sequence alone
    model, tokenizer = encoder._load_model()
    for sequence in (short, medium):
        tokens = tokenizer([sequence], return_tensors='pt')['input_ids']
        with torch.no_grad():
            alone = model(sequence_tokens=tokens).embeddings[0, 1:len(sequence) + 1].numpy()
        np.testing.assert_allclose(embeddings[sequence].astype(np.float32), alone, rtol=1e-2, atol=2e-2)

