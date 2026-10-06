"""
Tests for NARDINI+ encodings (parrot/nardini.py): the features SPARROW computes
for each sequence, given to every residue, alone ('nardini') or after a
per-residue encoding ('onehot+nardini'); computing them once into the embedding
cache; saving them with networks; and using them from every command.

Skipped if SPARROW is not installed. Features use few scrambles, to keep the
tests fast.
"""

import numpy as np
import pytest
import torch

pytest.importorskip('sparrow.nardini')

from parrot import nardini  # noqa: E402
from parrot import process_input_data as pid  # noqa: E402
from parrot.checkpoint import load_network  # noqa: E402
from parrot.encode_sequence import StandardizedEncoder, encoder_from_config, get_encoder  # noqa: E402
from parrot.py_predictor import Predictor  # noqa: E402
from parrot.scripts import parrot_predict, parrot_train  # noqa: E402
from parrot.tasks import Task  # noqa: E402

from tests.helpers import data_path  # noqa: E402

# Few scrambles keep the tests fast; the features are just as deterministic
QUICK_SCRAMBLES = 50

SEQUENCES = ['MEEEKKKKSPSPSPDEDEDE', 'GSGSGSYGSGSYGQQGSYGS', 'ACDEFGHIKLMNPQRSTVWY']


@pytest.fixture(autouse=True)
def one_process(monkeypatch):
    """Compute features in this process unless a test asks for more (worker start-up dominates for few sequences)."""
    monkeypatch.setattr(nardini, 'NUM_PROCESSES', 1)


def nardini_encoder(encoding: str = 'nardini', standardize: bool = False) -> nardini.NardiniEncoder:
    encoder = get_encoder(encoding, standardize=standardize, nardini_scrambles=QUICK_SCRAMBLES)
    assert isinstance(encoder, nardini.NardiniEncoder)
    return encoder


def write_datafile(write_lines, num_sequences: int = 40) -> str:
    """The first sequences of the sequence-regression example data (all at least 8 residues long)."""
    lines = pid.read_tsv_raw(data_path('seq_regress_dataset.tsv'))[:num_sequences]
    return write_lines('data.tsv', [' '.join(line) for line in lines])


# ..........................................................................................
# Encoding names and building encoders

@pytest.mark.parametrize('encoding, is_nardini, residue_part', [
    ('nardini', True, None), ('onehot+nardini', True, 'onehot'), ('path/to/table.txt+nardini', True, 'path/to/table.txt'),
    ('onehot', False, None), ('nardini_patterning', False, None),
])
def test_encoding_names(encoding, is_nardini, residue_part):
    assert nardini.is_nardini_encoding(encoding) == is_nardini
    if is_nardini:
        assert nardini.residue_encoding_name(encoding) == residue_part


def test_nardini_alone_gives_every_residue_its_sequence_features():
    encoder = nardini_encoder()
    sequence = SEQUENCES[0]
    encoded = encoder.encode(sequence)

    assert encoder.name == 'nardini' and len(encoder) == nardini.NUM_FEATURES
    assert encoded.shape == (len(sequence), nardini.NUM_FEATURES) and encoded.dtype == torch.float32
    expected = nardini.compute_features([sequence], QUICK_SCRAMBLES, nardini.NARDINI_SEED)[sequence]
    assert all(torch.equal(row, torch.from_numpy(expected)) for row in encoded)


@pytest.mark.parametrize('residue_encoding', ['onehot', 'biophysics'])
def test_combined_encodings_put_each_residues_own_encoding_first(residue_encoding):
    encoder = nardini_encoder(f'{residue_encoding}+nardini')
    residue_encoder = get_encoder(residue_encoding)
    sequence = SEQUENCES[1]
    encoded = encoder.encode(sequence)

    assert encoder.name == f'{residue_encoding}+nardini'
    assert len(encoder) == len(residue_encoder) + nardini.NUM_FEATURES
    torch.testing.assert_close(encoded[:, :len(residue_encoder)], residue_encoder.encode(sequence).float())
    torch.testing.assert_close(encoded[:, len(residue_encoder):],
                               nardini_encoder().encode(sequence))


def test_standardizing_a_combined_encoding_standardizes_its_per_residue_part():
    encoder = nardini_encoder('biophysics+nardini', standardize=True)
    assert isinstance(encoder.residue_encoder, StandardizedEncoder)
    assert len(encoder) == len(get_encoder('biophysics')) + nardini.NUM_FEATURES


@pytest.mark.parametrize('encoding, options, message', [
    ('nardini', dict(standardize=True), "does not apply to 'nardini' alone"),
    ('esm2_t6_8M_UR50D+nardini', {}, 'cannot be combined with a protein language model'),
    ('onehot', dict(nardini_scrambles=100), 'only applies to NARDINI encodings'),
    ('nardini', dict(nardini_scrambles=0), 'at least 1'),
])
def test_invalid_nardini_options(encoding, options, message):
    with pytest.raises(ValueError, match=message):
        get_encoder(encoding, **options)


# ..........................................................................................
# The features

def test_features_do_not_depend_on_which_other_sequences_are_computed_with_them():
    together = nardini.compute_features(SEQUENCES, QUICK_SCRAMBLES, nardini.NARDINI_SEED)
    for sequence in SEQUENCES:
        alone = nardini.compute_features([sequence], QUICK_SCRAMBLES, nardini.NARDINI_SEED)[sequence]
        np.testing.assert_array_equal(together[sequence], alone)
        assert together[sequence].shape == (nardini.NUM_FEATURES,) and np.isfinite(together[sequence]).all()


def test_features_computed_in_parallel_equal_those_computed_serially(monkeypatch):
    serial = nardini.compute_features(SEQUENCES, QUICK_SCRAMBLES, nardini.NARDINI_SEED)
    monkeypatch.setattr(nardini, 'NUM_PROCESSES', 2)
    monkeypatch.setattr(nardini, 'MIN_SEQUENCES_FOR_PARALLEL', 2)
    parallel = nardini.compute_features(SEQUENCES, QUICK_SCRAMBLES, nardini.NARDINI_SEED)
    for sequence in SEQUENCES:
        np.testing.assert_array_equal(serial[sequence], parallel[sequence])


@pytest.mark.parametrize('sequences, message', [
    (['MEEK'], 'shorter than 5 residues'),
    (['MEEKXKKEE'], 'need the 20 standard amino acids'),
])
def test_sequences_nardini_cannot_score_are_refused(sequences, message):
    with pytest.raises(ValueError, match=message):
        nardini_encoder().check_sequences(sequences)


def test_nardini_is_for_sequence_data_only():
    nardini.check_task(nardini_encoder(), Task('sequence', 1))
    nardini.check_task(get_encoder('onehot'), Task('residues', 1))
    with pytest.raises(ValueError, match='sequence-mapped data'):
        nardini.check_task(nardini_encoder('onehot+nardini'), Task('residues', 1))


# ..........................................................................................
# The cache

def test_features_are_computed_once_into_the_cache(tmp_path, monkeypatch):
    encoder = nardini_encoder()
    encoder.use_cache(tmp_path)
    encoder.prepare(SEQUENCES, 'test', silent=True)
    computed = [encoder.features(sequence) for sequence in SEQUENCES]

    cache_file = tmp_path / f'nardini_human_{QUICK_SCRAMBLES}scrambles_seed{nardini.NARDINI_SEED}.h5'
    assert cache_file.is_file() and (tmp_path / 'README.md').is_file()

    # A new encoder on the same cache reads every sequence rather than computing it
    def fail(*args, **kwargs):
        raise AssertionError('features were computed again')

    monkeypatch.setattr(nardini, 'compute_features', fail)
    reader = nardini_encoder()
    reader.use_cache(tmp_path)
    reader.prepare(SEQUENCES, 'test', silent=True)
    for sequence, expected in zip(SEQUENCES, computed):
        np.testing.assert_array_equal(reader.features(sequence), expected)


def test_each_number_of_scrambles_has_its_own_cache_file(tmp_path):
    for scrambles in (QUICK_SCRAMBLES, 2 * QUICK_SCRAMBLES):
        encoder = get_encoder('nardini', nardini_scrambles=scrambles)
        encoder.use_cache(tmp_path)
        encoder.prepare(SEQUENCES[:1], 'test', silent=True)
    assert len(list(tmp_path.glob('nardini_*.h5'))) == 2


# ..........................................................................................
# Saved networks

def test_encoders_are_rebuilt_from_their_saved_config():
    for encoding in ('nardini', 'onehot+nardini'):
        encoder = nardini_encoder(encoding)
        rebuilt = encoder_from_config(encoder.to_config())
        assert isinstance(rebuilt, nardini.NardiniEncoder)
        assert (rebuilt.name, rebuilt.num_scrambles, rebuilt.seed) == (encoding, QUICK_SCRAMBLES, nardini.NARDINI_SEED)
        torch.testing.assert_close(rebuilt.encode(SEQUENCES[2]), encoder.encode(SEQUENCES[2]))


def test_saved_networks_check_sparrow_computes_their_features():
    config = nardini_encoder().to_config()

    different_settings = dict(config, background='yeast')
    with pytest.raises(ValueError, match='different settings'):
        encoder_from_config(different_settings)

    different_features = encoder_from_config(dict(config, feature_names=config['feature_names'][::-1]))
    with pytest.raises(ValueError, match='differ from those this network was trained on'):
        different_features.encode(SEQUENCES[0])

    other_version = encoder_from_config(dict(config, sparrow_version='0.0.1'))
    with pytest.warns(UserWarning, match='computed with SPARROW 0.0.1'):
        other_version.encode(SEQUENCES[0])


# ..........................................................................................
# The commands

def train_with_nardini(run_cli, data_file, network_file, architecture, encoding, *arguments):
    datatype = [] if architecture == 'svm' else ['-d', 'sequence']
    epochs = [] if architecture == 'svm' else ['-e', '2']
    run_cli(parrot_train.main, ['parrot-train', architecture, data_file, network_file, *datatype, '-c', '1',
                                '--encode', encoding, '--nardini-scrambles', QUICK_SCRAMBLES, *epochs,
                                '--device', 'cpu', '--silent', *arguments])


@pytest.mark.parametrize('architecture, encoding', [('composition', 'nardini'), ('svm', 'nardini'),
                                                    ('brnn', 'onehot+nardini')])
def test_train_and_predict_with_nardini_features(run_cli, tmp_path, write_lines, architecture, encoding):
    data_file = write_datafile(write_lines)
    train_with_nardini(run_cli, data_file, tmp_path / 'net.pt', architecture, encoding)

    network, encoder = load_network(str(tmp_path / 'net.pt'))
    assert network.name == architecture and encoder.name == encoding
    assert encoder.num_scrambles == QUICK_SCRAMBLES
    assert list((tmp_path / 'parrot_train_embeddings').glob('nardini_*.h5'))

    sequence_file = write_lines('new.txt', [f'new{index} {sequence}' for index, sequence in enumerate(SEQUENCES)])
    run_cli(parrot_predict.main, ['parrot-predict', sequence_file, tmp_path / 'net.pt', tmp_path / 'out.txt',
                                  '--device', 'cpu', '--silent', '--encode', encoding])
    predicted = [float(line.split()[2]) for line in (tmp_path / 'out.txt').read_text().splitlines()]

    # Predictor gives the same predictions as parrot-predict
    predictor = Predictor(tmp_path / 'net.pt')
    np.testing.assert_allclose([predictor.predict(sequence)[0] for sequence in SEQUENCES], predicted, rtol=1e-5)


def test_training_again_reuses_the_cached_features(run_cli, tmp_path, write_lines, capsys):
    data_file = write_datafile(write_lines)
    cache = tmp_path / 'shared_cache'
    for name in ('first', 'second'):
        run_cli(parrot_train.main, ['parrot-train', 'composition', data_file, tmp_path / f'{name}.pt', '-d', 'sequence',
                                    '-c', '1', '--encode', 'nardini', '--nardini-scrambles', QUICK_SCRAMBLES, '-e', '1',
                                    '--device', 'cpu', '--embedding-cache', cache, '--ignore-warnings'])
    output = capsys.readouterr().out
    assert '0 already in the cache' in output and '40 already in the cache' in output


@pytest.mark.parametrize('arguments, message', [
    (['-d', 'residues', '--encode', 'nardini'], 'sequence-mapped data'),
    (['-d', 'sequence', '--nardini-scrambles', '10'], 'only applies to NARDINI encodings'),
])
def test_train_rejects_nardini_misuse(run_cli, tmp_path, arguments, message):
    data_file = data_path('res_regress_dataset.tsv' if 'residues' in arguments else 'seq_regress_dataset.tsv')
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_train.main, ['parrot-train', 'brnn', data_file, tmp_path / 'net.pt', '-c', '1', '-e', '1',
                                    '--device', 'cpu', '--silent', *arguments])


def test_predict_refuses_sequences_nardini_cannot_score(run_cli, tmp_path, write_lines):
    train_with_nardini(run_cli, write_datafile(write_lines), tmp_path / 'net.pt', 'composition', 'nardini')
    sequence_file = write_lines('new.txt', ['short MEEK'])
    with pytest.raises(ValueError, match='shorter than 5 residues'):
        run_cli(parrot_predict.main, ['parrot-predict', sequence_file, tmp_path / 'net.pt', tmp_path / 'out.txt',
                                      '--device', 'cpu', '--silent'])
