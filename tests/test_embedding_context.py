"""
Tests for embedding fragments within their full-length proteins (--embedding-context; parrot/embedding_context.py
and the in-context parts of parrot/plm.py), and for subtracting each embedding's mean over its residues
(--embed-mean-subtraction), alone and in context.

The PLM is replaced by a fake model whose output for a residue depends on its amino acid and on its position in
what the model is given, so a fragment embedded within its full-length protein gets different embeddings from the
same fragment on its own, as with a real model.
"""

import numpy as np
import pytest

from parrot import embedding_context as ec
from parrot import plm
from parrot import process_input_data as pid
from parrot.encode_sequence import get_encoder
from parrot.checkpoint import load_network
from parrot.py_predictor import Predictor
from parrot.scripts import parrot_predict, parrot_preprocess, parrot_train

from tests.helpers import training_command

MODEL = 'esm2_t6_8M_UR50D'
EMBEDDING_SIZE = plm.PLM_MODELS[MODEL].embedding_size

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# The fake model's vector for each amino acid, and how much each position along the input adds to every value
FAKE_TABLE = {amino_acid: np.random.default_rng(index).normal(size=EMBEDDING_SIZE).astype(np.float32)
              for index, amino_acid in enumerate(AMINO_ACIDS)}
FAKE_POSITION_STEP = 0.1

# The synthetic library: proteins tiled into overlapping fragments, as in a peptide-tiling screen
NUM_PROTEINS = 8
PROTEIN_LENGTH = 60
TILE_LENGTH = 12
TILE_STEP = 6


def context_dependent_model(self: plm.PLMEncoder, chunks: list[str]) -> list[np.ndarray]:
    """Stand-in for PLMEncoder._run_model: each residue's vector is its amino acid's plus a term for its position."""
    context_dependent_model.calls += 1
    return [fake_output(chunk) for chunk in chunks]


context_dependent_model.calls = 0


def fake_output(sequence: str) -> np.ndarray:
    """The fake model's float32 output for a sequence (no longer than a window)."""
    positions = np.arange(len(sequence), dtype=np.float32)[:, None] * FAKE_POSITION_STEP
    return np.stack([FAKE_TABLE[residue] for residue in sequence]) + positions


def fake_embedding(sequence: str) -> np.ndarray:
    """What the fake PLM gives a sequence embedded on its own, after rounding to the storage type."""
    return fake_output(sequence).astype(plm.STORAGE_DTYPE)


def fake_mean_subtracted_embedding(sequence: str) -> np.ndarray:
    """What the fake PLM gives a sequence with --embed-mean-subtraction: minus its mean over the residues, rounded."""
    output = fake_output(sequence)
    return (output - output.mean(axis=0)).astype(plm.STORAGE_DTYPE)


@pytest.fixture
def fake_plm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real PLM with the fake one (nothing is loaded or downloaded)."""
    context_dependent_model.calls = 0
    monkeypatch.setattr(plm.PLMEncoder, '_run_model', context_dependent_model)
    monkeypatch.setattr(plm.PLMEncoder, '_load_model', lambda self: (None, None))


def make_library(seed: int = 0) -> dict[str, str]:
    """NUM_PROTEINS random proteins, named P0, P1, ..."""
    rng = np.random.default_rng(seed)
    return {f'P{index}': ''.join(rng.choice(list(AMINO_ACIDS), size=PROTEIN_LENGTH)) for index in range(NUM_PROTEINS)}


def make_tiles(library: dict[str, str]) -> list[tuple[str, str, str, int]]:
    """Every tile of every protein: (tile ID, sequence, protein ID, 0-based start)."""
    tiles = []
    for protein_id, protein in library.items():
        for start in range(0, len(protein) - TILE_LENGTH + 1, TILE_STEP):
            tiles.append((f'{protein_id}_t{start}', protein[start:start + TILE_LENGTH], protein_id, start))
    # Random proteins this long share no 12-residue stretch, so every tile occurs exactly once
    assert len({sequence for _, sequence, _, _ in tiles}) == len(tiles)
    return tiles


def write_fasta(path, library: dict[str, str]) -> None:
    """Write proteins as FASTA, with a description after each ID and sequences wrapped at 25 residues."""
    with open(path, 'w') as handle:
        for protein_id, protein in library.items():
            handle.write(f'>{protein_id} a protein of {len(protein)} residues\n')
            for start in range(0, len(protein), 25):
                handle.write(protein[start:start + 25] + '\n')


@pytest.fixture
def tiled_dataset(tmp_path):
    """A sequence-regression dataset of tiles, the FASTA file of their proteins and the map between them

    Half the map's lines give the start; the others leave it to be found. One extra peptide has no protein ('-').
    """
    library = make_library()
    tiles = make_tiles(library)
    rng = np.random.default_rng(1)
    standalone = ('free_peptide', 'MKWVTFISLLFL')

    data_file = tmp_path / 'tiles.tsv'
    data_file.write_text(''.join(f'{tile_id} {sequence} {rng.normal():.4f}\n' for tile_id, sequence, _, _ in tiles)
                         + f'{standalone[0]} {standalone[1]} 0.5\n')
    library_file = tmp_path / 'proteins.fasta'
    write_fasta(library_file, library)
    map_file = tmp_path / 'map.tsv'
    map_lines = ['# fragment protein start']
    for index, (tile_id, _, protein_id, start) in enumerate(tiles):
        map_lines.append(f'{tile_id} {protein_id} {start + 1}' if index % 2 == 0 else f'{tile_id} {protein_id}')
    map_lines.append(f'{standalone[0]} {ec.NO_CONTEXT}')
    map_file.write_text('\n'.join(map_lines) + '\n')
    return {'data': data_file, 'library': library_file, 'map': map_file, 'proteins': library, 'tiles': tiles,
            'standalone': standalone}


# ..........................................................................................
# Reading the library and the map

def test_library_keeps_the_wanted_proteins_named_by_the_first_word_of_their_header(tmp_path):
    library = make_library()
    write_fasta(tmp_path / 'proteins.fasta', library)

    read = ec.read_library(str(tmp_path / 'proteins.fasta'), {'P1', 'P3', 'not_there'})

    assert read == {'P1': library['P1'], 'P3': library['P3']}


@pytest.mark.parametrize('text, message', [
    ('>A\nACD\n>A\nEFG\n', 'more than one protein called A'),
    ('ACD\n>A\nEFG\n', 'before the first header line'),
    ('>A\nACD\n>\nEFG\n', 'header line with no protein ID'),
    ('>A\n>B\nEFG\n', 'have no sequence'),
    ('\n\n', 'holds no proteins'),
])
def test_library_problems_are_errors(tmp_path, text, message):
    (tmp_path / 'proteins.fasta').write_text(text)
    with pytest.raises(ValueError, match=message):
        ec.read_library(str(tmp_path / 'proteins.fasta'), {'A', 'B'})


def test_map_lines_may_give_a_start(write_lines):
    map_file = write_lines('map.tsv', ['# a header', 'f1 P1', 'f2 P1 7', '', 'f3 -'])

    entries = ec.read_context_map(map_file)

    assert entries == {'f1': ec.MapEntry('P1', None), 'f2': ec.MapEntry('P1', 7), 'f3': ec.MapEntry('-', None)}


@pytest.mark.parametrize('lines, message', [
    (['f1'], 'should be'),
    (['f1 P1 3 extra'], 'should be'),
    (['f1 P1 0'], 'at least 1'),
    (['f1 P1 x'], 'at least 1'),
    (['f1 P1 -2'], 'at least 1'),
    (['f1 - 3'], 'cannot have a start'),
    (['f1 P1', 'f1 P2'], 'more than one line for f1'),
    (['# only a comment'], 'no lines'),
])
def test_map_problems_are_errors(write_lines, lines, message):
    with pytest.raises(ValueError, match=message):
        ec.read_context_map(write_lines('map.tsv', lines))


# ..........................................................................................
# Placing fragments

def test_a_fragment_is_located_by_its_start_or_by_searching():
    protein = 'MKWVTFISLLFLFSSAYS'
    assert ec.locate_fragment('TFIS', protein, 5) == 4
    assert ec.locate_fragment('TFIS', protein, None) == 4


@pytest.mark.parametrize('fragment, start, message', [
    ('TFIS', 6, 'does not match the full-length protein at residues 6-9'),
    ('AYSX', 16, 'runs past the end'),
    ('QQQ', None, 'does not occur'),
    ('LF', None, 'occurs 2 times'),
    ('L', None, 'occurs 3 times'),
])
def test_fragments_that_cannot_be_placed_are_errors(fragment, start, message):
    with pytest.raises(ec.ContextProblem, match=message):
        ec.locate_fragment(fragment, 'MKWVTFISLLFLFSSAYS', start)


def test_overlapping_occurrences_are_all_found():
    assert ec.occurrences('AA', 'AAAB') == [0, 1]


def test_every_fragment_is_assigned_its_context(tiled_dataset):
    lines = pid.read_tsv_raw(str(tiled_dataset['data']))
    ids, sequences = [line[0] for line in lines], [line[1] for line in lines]

    assignment = ec.assign_contexts(ids, sequences, str(tiled_dataset['library']), str(tiled_dataset['map']))

    proteins = tiled_dataset['proteins']
    for (tile_id, sequence, protein_id, start), row in zip(tiled_dataset['tiles'], assignment.rows):
        assert (row.fragment_id, row.full_length_id, row.start) == (tile_id, protein_id, start)
        assert row.positions() == f'{start + 1}-{start + TILE_LENGTH}'
        assert assignment.contexts[sequence] == plm.FragmentContext(proteins[protein_id], start)
    notes = {row.note for row in assignment.rows}
    assert notes == {ec.NOTE_START_GIVEN, ec.NOTE_START_FOUND, ec.NOTE_ON_ITS_OWN}

    free_id, free_sequence = tiled_dataset['standalone']
    assert assignment.rows[-1] == ec.ContextRow(free_id, free_sequence, ec.NO_CONTEXT, None, ec.NOTE_ON_ITS_OWN)
    assert assignment.contexts[free_sequence] is None
    assert assignment.unused_map_lines == 0
    assert f'{len(ids) - 1:,} of the {len(ids):,} sequences are embedded within {NUM_PROTEINS}' in assignment.describe()


def test_map_lines_for_ids_not_in_the_data_are_counted(tiled_dataset):
    lines = pid.read_tsv_raw(str(tiled_dataset['data']))[:5]
    assignment = ec.assign_contexts([line[0] for line in lines], [line[1] for line in lines],
                                    str(tiled_dataset['library']), str(tiled_dataset['map']))

    assert assignment.unused_map_lines == len(tiled_dataset['tiles']) + 1 - 5
    assert 'are for IDs that are not in the data' in assignment.describe()


def test_every_problem_is_counted_and_the_first_few_listed(tmp_path, write_lines):
    write_fasta(tmp_path / 'proteins.fasta', {'P1': 'MKWVTFISLLFLFSSAYS'})
    ids = [f'f{index}' for index in range(8)]
    map_file = write_lines('map.tsv', [f'{fragment_id} P2' for fragment_id in ids[:7]])

    with pytest.raises(ValueError, match='8 of the 8 sequences cannot be placed') as error:
        ec.assign_contexts(ids, ['MKWV'] * 8, str(tmp_path / 'proteins.fasta'), map_file)

    message = str(error.value)
    assert 'f0: its full-length protein P2 is not in' in message
    assert '... and 3 more' in message


def test_a_sequence_needs_the_same_context_under_every_id(tmp_path, write_lines):
    write_fasta(tmp_path / 'proteins.fasta', {'P1': 'MKWVTFISLL', 'P2': 'AAMKWVAA', 'P1_copy': 'MKWVTFISLL'})
    library = str(tmp_path / 'proteins.fasta')

    # Identical proteins under two IDs give the same context
    same = write_lines('same.tsv', ['a P1', 'b P1_copy'])
    assignment = ec.assign_contexts(['a', 'b'], ['MKWV', 'MKWV'], library, same)
    assert assignment.contexts == {'MKWV': plm.FragmentContext('MKWVTFISLL', 0)}

    different = write_lines('different.tsv', ['a P1', 'b P2'])
    with pytest.raises(ValueError, match='different full-length contexts under different IDs'):
        ec.assign_contexts(['a', 'b'], ['MKWV', 'MKWV'], library, different)
    on_its_own = write_lines('on_its_own.tsv', ['a P1', 'b -'])
    with pytest.raises(ValueError, match='different full-length contexts under different IDs'):
        ec.assign_contexts(['a', 'b'], ['MKWV', 'MKWV'], library, on_its_own)


@pytest.mark.parametrize('protein', ['MKWVTFISLL*', 'MKWVtfisLL', 'MKWV-FISLL'])
def test_proteins_the_model_cannot_read_are_errors(tmp_path, write_lines, protein):
    write_fasta(tmp_path / 'proteins.fasta', {'P1': protein})
    with pytest.raises(ValueError, match='cannot read as residues'):
        ec.assign_contexts(['a'], ['MKWV'], str(tmp_path / 'proteins.fasta'), write_lines('map.tsv', ['a P1']))


def test_the_table_has_one_line_per_fragment(tmp_path):
    rows = [ec.ContextRow('a', 'MKWV', 'P1', 4, ec.NOTE_START_FOUND),
            ec.ContextRow('b', 'QQQ', ec.NO_CONTEXT, None, ec.NOTE_ON_ITS_OWN)]
    ec.write_context_table(str(tmp_path / 'table.tsv'), rows)

    lines = [line.split('\t') for line in (tmp_path / 'table.tsv').read_text().splitlines()]
    assert lines == [list(ec.CONTEXT_TABLE_COLUMNS), ['a', 'MKWV', 'P1', '5-8', ec.NOTE_START_FOUND],
                     ['b', 'QQQ', '-', '-', ec.NOTE_ON_ITS_OWN]]


# ..........................................................................................
# Embedding in context

def test_a_fragment_in_context_is_cut_out_of_its_proteins_embedding(fake_plm):
    protein = make_library()['P0']
    fragment = protein[20:32]
    encoder = plm.PLMEncoder(MODEL)
    encoder.add_contexts({fragment: plm.FragmentContext(protein, 20)})

    in_context = encoder.encode(fragment).numpy()

    np.testing.assert_array_equal(in_context, fake_embedding(protein)[20:32].astype(np.float32))
    assert not np.allclose(in_context, fake_embedding(fragment).astype(np.float32))


def test_without_contexts_every_sequence_is_embedded_on_its_own(fake_plm):
    encoder = plm.PLMEncoder(MODEL)
    np.testing.assert_array_equal(encoder.encode('MKWVTF').numpy(), fake_embedding('MKWVTF').astype(np.float32))
    assert 'embedded_in_context' not in encoder.to_config()


def test_once_given_contexts_every_sequence_needs_one(fake_plm):
    encoder = plm.PLMEncoder(MODEL)
    encoder.add_contexts({'MKWV': None})

    np.testing.assert_array_equal(encoder.encode('MKWV').numpy(), fake_embedding('MKWV').astype(np.float32))
    with pytest.raises(ValueError, match='No full-length protein was given'):
        encoder.encode('TFIS')


def test_in_context_embeddings_are_cached_apart_from_embeddings_on_their_own(fake_plm, tmp_path):
    library = make_library()
    tiles = make_tiles(library)
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path)
    encoder.add_contexts({sequence: plm.FragmentContext(library[protein_id], start)
                          for _, sequence, protein_id, start in tiles[:-1]})
    encoder.add_contexts({tiles[-1][1]: None})
    sequences = [sequence for _, sequence, _, _ in tiles]

    encoder.prepare(sequences, 'test', silent=True)

    context_cache = tmp_path / f'{MODEL}{plm.CONTEXT_CACHE_SUFFIX}.h5'
    assert context_cache.is_file() and (tmp_path / f'{MODEL}.h5').is_file()
    assert encoder.context_cache is not None and encoder.cache is not None
    assert encoder.context_cache.summary()[0] == len(tiles) - 1 and encoder.cache.summary()[0] == 1
    for _, sequence, protein_id, start in tiles[:-1]:
        expected = fake_embedding(library[protein_id])[start:start + TILE_LENGTH]
        np.testing.assert_array_equal(encoder.encode(sequence).numpy(), expected.astype(np.float32))
    np.testing.assert_array_equal(encoder.encode(tiles[-1][1]).numpy(), fake_embedding(tiles[-1][1]).astype(np.float32))
    assert 'in context' in (tmp_path / 'README.md').read_text()

    # Everything is cached, so preparing again runs nothing
    calls = context_dependent_model.calls
    encoder.prepare(sequences, 'test', silent=True)
    assert context_dependent_model.calls == calls


def test_each_protein_is_embedded_once_for_all_its_fragments(fake_plm, tmp_path, monkeypatch):
    library = make_library()
    tiles = make_tiles(library)
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path)
    encoder.add_contexts({sequence: plm.FragmentContext(library[protein_id], start)
                          for _, sequence, protein_id, start in tiles})
    embedded = []
    original_embed = plm.PLMEncoder.embed

    def recording_embed(self, sequences, progress=None):
        embedded.extend(sequences)
        return original_embed(self, sequences, progress)

    monkeypatch.setattr(plm.PLMEncoder, 'embed', recording_embed)
    encoder.prepare([sequence for _, sequence, _, _ in tiles], 'test', silent=True)

    assert sorted(embedded) == sorted(library.values())


def test_the_same_fragment_in_two_contexts_has_two_entries(fake_plm, tmp_path):
    fragment = 'MKWVTF'
    first, second = plm.FragmentContext('AAMKWVTFAA', 2), plm.FragmentContext('MKWVTFQQQQ', 0)
    encoder = plm.PLMEncoder(MODEL)
    encoder.use_cache(tmp_path)

    encoder.add_contexts({fragment: first})
    in_first = encoder.encode(fragment).numpy()
    encoder.add_contexts({fragment: second})
    in_second = encoder.encode(fragment).numpy()

    np.testing.assert_array_equal(in_first, fake_embedding('AAMKWVTFAA')[2:8].astype(np.float32))
    np.testing.assert_array_equal(in_second, fake_embedding('MKWVTFQQQQ')[:6].astype(np.float32))
    assert encoder.context_cache is not None and encoder.context_cache.summary()[0] == 2


def test_the_config_records_embedding_in_context(fake_plm):
    encoder = plm.PLMEncoder(MODEL)
    encoder.add_contexts({'MKWV': None})

    config = encoder.to_config()
    assert config[plm.IN_CONTEXT_CONFIG_KEY] is True
    assert plm.encoder_from_plm_config(config).in_context
    assert not plm.encoder_from_plm_config(plm.PLMEncoder(MODEL).to_config()).in_context


# ..........................................................................................
# Commands

def context_flags(dataset: dict) -> list:
    """--embedding-context and --embedding-context-map for a tiled dataset."""
    return ['--embedding-context', dataset['library'], '--embedding-context-map', dataset['map']]


def train(run_cli, dataset, network_file, *extra):
    """Train briefly on a tiled dataset with the PLM encoding."""
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', dataset['data'], network_file, 'sequence', 1,
                                                '-e', '2', '--encode', MODEL, '--seed', '1', '--silent', *extra))


def test_train_and_predict_in_context(fake_plm, run_cli, tmp_path, tiled_dataset):
    (tmp_path / 'train').mkdir()
    network_file = tmp_path / 'train' / 'net.pt'
    train(run_cli, tiled_dataset, network_file, *context_flags(tiled_dataset))

    cache = tmp_path / 'train' / plm.TRAIN_CACHE_DIRECTORY
    assert (cache / f'{MODEL}{plm.CONTEXT_CACHE_SUFFIX}.h5').is_file() and (cache / f'{MODEL}.h5').is_file()
    table = (tmp_path / 'train' / f'net{ec.CONTEXT_TABLE_SUFFIX}').read_text().splitlines()
    assert len(table) == len(tiled_dataset['tiles']) + 2
    _, encoder = load_network(str(network_file))
    assert isinstance(encoder, plm.PLMEncoder) and encoder.in_context
    info = (tmp_path / 'train' / 'net_info.txt').read_text()
    assert 'embedding context map' in info and 'embedded_in_context' in info

    # parrot-predict, given the same context, reproduces the test-set predictions
    lines = [line.split() for line in (tmp_path / 'train' / 'net_predictions.tsv').read_text().splitlines()]
    test_ids = [line[0].removesuffix('_TRUE') for line in lines[::2]]
    expected = [float(line[2]) for line in lines[1::2]]
    sequences = {line[0]: line[1] for line in pid.read_tsv_raw(str(tiled_dataset['data']))}
    seq_file = tmp_path / 'sequences.txt'
    seq_file.write_text(''.join(f'{seq_id} {sequences[seq_id]}\n' for seq_id in test_ids))
    output_file = tmp_path / 'predicted.txt'
    run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, output_file, '--silent',
                                  *context_flags(tiled_dataset)])

    predicted = [float(line.split()[2]) for line in output_file.read_text().splitlines()]
    np.testing.assert_allclose(predicted, expected, atol=1e-4)
    assert (tmp_path / f'predicted{ec.CONTEXT_TABLE_SUFFIX}').is_file()

    # A network trained in context needs the context to predict, unless told to embed each sequence on its own
    with pytest.raises(ValueError, match='trained on sequences embedded within their full-length proteins'):
        run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, tmp_path / 'other.txt', '--silent'])
    run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, tmp_path / 'isolated.txt', '--silent',
                                  '--embed-in-isolation'])
    isolated = [float(line.split()[2]) for line in (tmp_path / 'isolated.txt').read_text().splitlines()]
    assert not np.allclose(isolated, expected, atol=1e-4)
    with pytest.raises(ValueError, match='not both'):
        run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, tmp_path / 'other.txt', '--silent',
                                      '--embed-in-isolation', *context_flags(tiled_dataset)])

    # So does Predictor, given the full-length protein of a test-set tile
    index = next(index for index, seq_id in enumerate(test_ids) if seq_id != tiled_dataset['standalone'][0])
    tile = sequences[test_ids[index]]
    full_length = tiled_dataset['proteins'][test_ids[index].split('_t')[0]]
    predictor = Predictor(network_file)
    np.testing.assert_allclose(predictor.predict(tile, full_length=full_length), [expected[index]], atol=1e-4)
    with pytest.raises(ValueError, match='give the full-length protein'):
        predictor.predict(tile)
    np.testing.assert_allclose(Predictor(network_file, embed_in_isolation=True).predict(tile), [isolated[index]],
                               atol=1e-4)


def test_a_network_trained_without_context_refuses_one(fake_plm, run_cli, tmp_path, tiled_dataset):
    network_file = tmp_path / 'net.pt'
    train(run_cli, tiled_dataset, network_file)
    seq_file = tmp_path / 'sequences.txt'
    seq_file.write_text('P0_t0 ' + tiled_dataset['tiles'][0][1] + '\n')

    for extra in (context_flags(tiled_dataset), ['--embed-in-isolation']):
        with pytest.raises(ValueError, match='only applies to networks trained with --embedding-context'):
            run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, tmp_path / 'out.txt', '--silent',
                                          *extra])
    with pytest.raises(ValueError, match='only applies to networks trained with --embedding-context'):
        Predictor(network_file, embed_in_isolation=True)
    with pytest.raises(ValueError, match='full_length only applies'):
        Predictor(network_file).predict('MKWV', full_length='AAMKWV')


def test_predictor_places_the_sequence_in_its_full_length_protein(fake_plm, run_cli, tmp_path, tiled_dataset):
    network_file = tmp_path / 'net.pt'
    train(run_cli, tiled_dataset, network_file, *context_flags(tiled_dataset))
    predictor = Predictor(network_file)
    protein = tiled_dataset['proteins']['P0']

    by_search = predictor.predict(protein[6:18].lower(), full_length=protein.lower())
    by_start = predictor.predict(protein[6:18], full_length=protein, start=7)
    np.testing.assert_allclose(by_search, by_start)

    with pytest.raises(ValueError, match='cannot be placed in full_length: it does not match'):
        predictor.predict(protein[6:18], full_length=protein, start=8)
    with pytest.raises(ValueError, match='start only applies with full_length'):
        predictor.predict(protein[6:18], start=7)


def test_predict_batch_embeds_each_full_length_protein_once(fake_plm, run_cli, tmp_path, tiled_dataset, monkeypatch):
    network_file = tmp_path / 'net.pt'
    train(run_cli, tiled_dataset, network_file, *context_flags(tiled_dataset))
    library = tiled_dataset['proteins']
    sequences = [sequence for _, sequence, _, _ in tiled_dataset['tiles']]
    full_lengths = [library[protein_id] for _, _, protein_id, _ in tiled_dataset['tiles']]
    predictor = Predictor(network_file)
    embedded = []
    original_embed = plm.PLMEncoder.embed

    def recording_embed(self, sequences, progress=None):
        embedded.extend(sequences)
        return original_embed(self, sequences, progress)

    monkeypatch.setattr(plm.PLMEncoder, 'embed', recording_embed)
    batch = predictor.predict_batch(sequences, full_lengths=full_lengths, batch_size=5)

    assert sorted(embedded) == sorted(library.values())
    for sequence, full_length, predicted in zip(sequences, full_lengths, batch):
        np.testing.assert_allclose(predicted, predictor.predict(sequence, full_length=full_length), rtol=1e-5,
                                   atol=1e-6)


def test_predict_batch_takes_starts_and_sequences_embedded_on_their_own(fake_plm, run_cli, tmp_path,
                                                                        tiled_dataset):
    network_file = tmp_path / 'net.pt'
    train(run_cli, tiled_dataset, network_file, *context_flags(tiled_dataset))
    predictor = Predictor(network_file, embed_in_isolation=True)
    protein = tiled_dataset['proteins']['P0']
    standalone = tiled_dataset['standalone'][1]

    batch = predictor.predict_batch([protein[6:18], standalone], full_lengths=[protein, None], starts=[7, None])

    np.testing.assert_allclose(batch[0], predictor.predict(protein[6:18], full_length=protein, start=7), rtol=1e-5,
                               atol=1e-6)
    np.testing.assert_allclose(batch[1], predictor.predict(standalone), rtol=1e-5, atol=1e-6)


def test_predict_batch_context_problems_are_errors(fake_plm, run_cli, tmp_path, tiled_dataset):
    network_file = tmp_path / 'net.pt'
    train(run_cli, tiled_dataset, network_file, *context_flags(tiled_dataset))
    predictor = Predictor(network_file)
    protein, other = tiled_dataset['proteins']['P0'], tiled_dataset['proteins']['P1']
    fragment, other_fragment = protein[6:18], other[6:18]

    with pytest.raises(ValueError, match='give the full-length protein each sequence was cut from'):
        predictor.predict_batch([fragment])
    with pytest.raises(ValueError, match='full_lengths has 2 entries but there are 1 sequences'):
        predictor.predict_batch([fragment], full_lengths=[protein, protein])
    with pytest.raises(TypeError, match='not a single string'):
        predictor.predict_batch([fragment], full_lengths=protein)
    with pytest.raises(ValueError, match=r'sequences\[1\]: The sequence cannot be placed in full_length'):
        predictor.predict_batch([fragment, other_fragment], full_lengths=[protein, protein])
    with pytest.raises(ValueError, match=r'sequences\[1\]: This network was trained'):
        predictor.predict_batch([fragment, other_fragment], full_lengths=[protein, None])
    with pytest.raises(ValueError, match=r'sequences\[2\] is the same sequence as sequences\[0\]'):
        predictor.predict_batch([fragment, other_fragment, fragment], full_lengths=[protein, other, 'AA' + fragment])


def test_preprocess_embeds_in_context_ahead_of_training(fake_plm, run_cli, tmp_path, tiled_dataset):
    output_file = tmp_path / 'clean.tsv'
    cache = tmp_path / 'cache'
    run_cli(parrot_preprocess.main, ['parrot-preprocess', tiled_dataset['data'], output_file, '-d', 'sequence', '-m',
                                     'regression', '--split', 'random', '--seed', '1', '--encode', MODEL,
                                     '--embedding-cache', cache, *context_flags(tiled_dataset)])

    assert (cache / f'{MODEL}{plm.CONTEXT_CACHE_SUFFIX}.h5').is_file()
    table = (tmp_path / f'clean{ec.CONTEXT_TABLE_SUFFIX}').read_text().splitlines()
    assert len(table) == len(tiled_dataset['tiles']) + 2
    report = (tmp_path / 'clean_preprocess_report.txt').read_text()
    assert f'--embedding-context-map {tiled_dataset["map"]}' in report

    # Training on the output with the same context finds every sequence in the cache, so embeds nothing
    calls = context_dependent_model.calls
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', output_file, tmp_path / 'net.pt', 'sequence', 1,
                                                '--split', tmp_path / 'clean_split_file.txt', '-e', '1', '--encode',
                                                MODEL, '--silent', '--embedding-cache', cache,
                                                *context_flags(tiled_dataset)))
    assert context_dependent_model.calls == calls


def test_preprocess_places_every_sequence_before_writing_anything(fake_plm, run_cli, tmp_path, tiled_dataset,
                                                                  write_lines):
    short_map = write_lines('short_map.tsv', (tiled_dataset['map'].read_text().splitlines())[:-3])
    with pytest.raises(ValueError, match='cannot be placed in their full-length proteins'):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', tiled_dataset['data'], tmp_path / 'out.tsv', '-d',
                                         'sequence', '-m', 'regression', '--encode', MODEL, '--embedding-context',
                                         tiled_dataset['library'], '--embedding-context-map', short_map])

    assert not (tmp_path / 'out.tsv').exists()
    assert not (tmp_path / plm.TRAIN_CACHE_DIRECTORY).exists()
    assert context_dependent_model.calls == 0


@pytest.mark.parametrize('extra, error, message', [
    (['--encode', MODEL, '--embedding-context', 'proteins.fasta'], ValueError, 'go together'),
    (['--encode', 'onehot', '--embedding-context', 'proteins.fasta', '--embedding-context-map', 'map.tsv'],
     ValueError, 'only applies to protein language model'),
    (['--encode', MODEL, '--exclude-seq-id', '--embedding-context', 'proteins.fasta', '--embedding-context-map',
      'map.tsv'], ValueError, 'cannot be used with --exclude-seq-id'),
    (['--encode', MODEL, '--embedding-context', 'missing.fasta', '--embedding-context-map', 'map.tsv'],
     FileNotFoundError, 'FASTA file of full-length proteins does not exist'),
])
def test_training_rejects_misused_context_options(fake_plm, run_cli, tmp_path, tiled_dataset, monkeypatch, extra,
                                                  error, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(error, match=message):
        run_cli(parrot_train.main, training_command('parrot-train', 'brnn', tiled_dataset['data'], tmp_path / 'net.pt',
                                                    'sequence', 1, '-e', '1', '--silent', *extra))
    assert context_dependent_model.calls == 0


@pytest.mark.parametrize('extra, message', [
    (['--embedding-context', 'proteins.fasta', '--embedding-context-map', 'map.tsv'], 'only applies with --encode'),
    (['--encode', 'nardini', '--embedding-context', 'proteins.fasta', '--embedding-context-map', 'map.tsv'],
     'only applies to protein language model'),
])
def test_preprocess_rejects_misused_context_options(run_cli, tmp_path, tiled_dataset, monkeypatch, extra, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', tiled_dataset['data'], tmp_path / 'out.tsv', '-d',
                                         'sequence', '-m', 'regression', *extra])
    assert not (tmp_path / 'out.tsv').exists()


# ..........................................................................................
# Mean subtraction

def test_mean_subtraction_removes_each_embeddings_mean_over_its_residues(fake_plm):
    sequence = 'MKWVTFISLLFL'
    encoder = plm.PLMEncoder(MODEL, mean_subtracted=True)

    embedding = encoder.encode(sequence).numpy()

    np.testing.assert_array_equal(embedding, fake_mean_subtracted_embedding(sequence).astype(np.float32))
    # Only the rounding to float16 keeps the mean from being exactly zero
    assert np.abs(embedding.mean(axis=0)).max() < 1e-2
    assert not np.allclose(embedding, fake_embedding(sequence).astype(np.float32))


def test_in_context_the_full_length_proteins_mean_is_subtracted_before_the_fragment_is_cut_out(fake_plm):
    protein = make_library()['P0']
    fragment = protein[20:32]
    encoder = plm.PLMEncoder(MODEL, mean_subtracted=True)
    encoder.add_contexts({fragment: plm.FragmentContext(protein, 20)})

    in_context = encoder.encode(fragment).numpy()

    np.testing.assert_array_equal(in_context, fake_mean_subtracted_embedding(protein)[20:32].astype(np.float32))
    assert not np.allclose(in_context, fake_mean_subtracted_embedding(fragment).astype(np.float32))


def test_mean_subtracted_embeddings_are_cached_in_their_own_files(fake_plm, tmp_path):
    library = make_library()
    tiles = make_tiles(library)
    contexts = {sequence: plm.FragmentContext(library[protein_id], start) for _, sequence, protein_id, start in tiles}
    contexts['MKWVTFISLLFL'] = None
    sequences = list(contexts)

    as_given = plm.PLMEncoder(MODEL)
    mean_subtracted = plm.PLMEncoder(MODEL, mean_subtracted=True)
    for encoder in (as_given, mean_subtracted):
        encoder.use_cache(tmp_path)
        encoder.add_contexts(contexts)

    as_given.prepare(sequences, 'test', silent=True)
    calls = context_dependent_model.calls
    mean_subtracted.prepare(sequences, 'test', silent=True)

    # The embeddings as the model gives them are no use, so the model runs again
    assert context_dependent_model.calls > calls
    suffix = plm.MEAN_SUBTRACTED_CACHE_SUFFIX
    for name in (MODEL, f'{MODEL}{plm.CONTEXT_CACHE_SUFFIX}', f'{MODEL}{suffix}',
                 f'{MODEL}{suffix}{plm.CONTEXT_CACHE_SUFFIX}'):
        assert (tmp_path / f'{name}.h5').is_file()
    _, sequence, protein_id, start = tiles[3]
    expected = fake_mean_subtracted_embedding(library[protein_id])[start:start + TILE_LENGTH]
    np.testing.assert_array_equal(mean_subtracted.encode(sequence).numpy(), expected.astype(np.float32))
    np.testing.assert_array_equal(mean_subtracted.encode('MKWVTFISLLFL').numpy(),
                                  fake_mean_subtracted_embedding('MKWVTFISLLFL').astype(np.float32))
    assert 'Mean-subtracted PLM embeddings' in (tmp_path / 'README.md').read_text()


def test_the_config_records_mean_subtraction(fake_plm):
    config = plm.PLMEncoder(MODEL, mean_subtracted=True).to_config()

    assert config[plm.MEAN_SUBTRACTED_CONFIG_KEY] is True
    assert plm.encoder_from_plm_config(config).mean_subtracted
    # Networks saved without the setting used the embeddings as the model gives them
    assert plm.MEAN_SUBTRACTED_CONFIG_KEY not in plm.PLMEncoder(MODEL).to_config()
    assert not plm.encoder_from_plm_config(plm.PLMEncoder(MODEL).to_config()).mean_subtracted


@pytest.mark.parametrize('encoding', ['onehot', 'nardini', 'onehot+nardini'])
def test_mean_subtraction_only_applies_to_protein_language_models(encoding):
    with pytest.raises(ValueError, match='only applies to protein language model'):
        get_encoder(encoding, mean_subtracted=True)


def test_train_and_predict_with_mean_subtracted_embeddings_in_context(fake_plm, run_cli, tmp_path, tiled_dataset):
    (tmp_path / 'train').mkdir()
    network_file = tmp_path / 'train' / 'net.pt'
    train(run_cli, tiled_dataset, network_file, '--embed-mean-subtraction', *context_flags(tiled_dataset))

    cache = tmp_path / 'train' / plm.TRAIN_CACHE_DIRECTORY
    suffix = plm.MEAN_SUBTRACTED_CACHE_SUFFIX
    assert (cache / f'{MODEL}{suffix}{plm.CONTEXT_CACHE_SUFFIX}.h5').is_file()
    assert (cache / f'{MODEL}{suffix}.h5').is_file() and not (cache / f'{MODEL}.h5').exists()
    _, encoder = load_network(str(network_file))
    assert isinstance(encoder, plm.PLMEncoder) and encoder.mean_subtracted and encoder.in_context
    assert plm.MEAN_SUBTRACTED_CONFIG_KEY in (tmp_path / 'train' / 'net_info.txt').read_text()

    # parrot-predict subtracts the mean too, without being told, so it reproduces the test-set predictions
    lines = [line.split() for line in (tmp_path / 'train' / 'net_predictions.tsv').read_text().splitlines()]
    test_ids = [line[0].removesuffix('_TRUE') for line in lines[::2]]
    expected = [float(line[2]) for line in lines[1::2]]
    sequences = {line[0]: line[1] for line in pid.read_tsv_raw(str(tiled_dataset['data']))}
    seq_file = tmp_path / 'sequences.txt'
    seq_file.write_text(''.join(f'{seq_id} {sequences[seq_id]}\n' for seq_id in test_ids))
    output_file = tmp_path / 'predicted.txt'
    run_cli(parrot_predict.main, ['parrot-predict', seq_file, network_file, output_file, '--silent', '--encode',
                                  MODEL, *context_flags(tiled_dataset)])

    predicted = [float(line.split()[2]) for line in output_file.read_text().splitlines()]
    np.testing.assert_allclose(predicted, expected, atol=1e-4)
    assert (tmp_path / plm.PREDICT_CACHE_DIRECTORY / f'{MODEL}{suffix}{plm.CONTEXT_CACHE_SUFFIX}.h5').is_file()

    # So does Predictor
    index = next(index for index, seq_id in enumerate(test_ids) if seq_id != tiled_dataset['standalone'][0])
    full_length = tiled_dataset['proteins'][test_ids[index].split('_t')[0]]
    np.testing.assert_allclose(Predictor(network_file).predict(sequences[test_ids[index]], full_length=full_length),
                               [expected[index]], atol=1e-4)


def test_preprocess_embeds_mean_subtracted_ahead_of_training(fake_plm, run_cli, tmp_path, tiled_dataset):
    output_file = tmp_path / 'clean.tsv'
    cache = tmp_path / 'cache'
    run_cli(parrot_preprocess.main, ['parrot-preprocess', tiled_dataset['data'], output_file, '-d', 'sequence', '-m',
                                     'regression', '--split', 'random', '--seed', '1', '--encode', MODEL,
                                     '--embedding-cache', cache, '--embed-mean-subtraction'])

    assert (cache / f'{MODEL}{plm.MEAN_SUBTRACTED_CACHE_SUFFIX}.h5').is_file()
    report = (tmp_path / 'clean_preprocess_report.txt').read_text()
    assert f'--encode {MODEL} --embed-mean-subtraction --embedding-cache' in report
    assert '(mean-subtracted)' in report

    # Training on the output with the same options finds every sequence in the cache, so embeds nothing
    calls = context_dependent_model.calls
    run_cli(parrot_train.main, training_command('parrot-train', 'brnn', output_file, tmp_path / 'net.pt', 'sequence', 1,
                                                '--split', tmp_path / 'clean_split_file.txt', '-e', '1', '--encode',
                                                MODEL, '--silent', '--embedding-cache', cache,
                                                '--embed-mean-subtraction'))
    assert context_dependent_model.calls == calls


@pytest.mark.parametrize('extra, message', [
    (['--embed-mean-subtraction'], 'only applies with --encode'),
    (['--encode', 'nardini', '--embed-mean-subtraction'], 'only applies to protein language model'),
])
def test_preprocess_rejects_misused_mean_subtraction(run_cli, tmp_path, tiled_dataset, extra, message):
    with pytest.raises(ValueError, match=message):
        run_cli(parrot_preprocess.main, ['parrot-preprocess', tiled_dataset['data'], tmp_path / 'out.tsv', '-d',
                                         'sequence', '-m', 'regression', *extra])
    assert not (tmp_path / 'out.tsv').exists()


def test_training_rejects_mean_subtraction_without_a_protein_language_model(run_cli, tmp_path, tiled_dataset):
    with pytest.raises(ValueError, match='only applies to protein language model'):
        run_cli(parrot_train.main, training_command('parrot-train', 'brnn', tiled_dataset['data'], tmp_path / 'net.pt',
                                                    'sequence', 1, '-e', '1', '--silent', '--encode', 'onehot',
                                                    '--embed-mean-subtraction'))


def test_scikit_learn_models_need_context_for_mean_subtraction(fake_plm, run_cli, tmp_path, tiled_dataset):
    # On its own, a mean-subtracted sequence's mean embedding (these models' features) is zero
    with pytest.raises(ValueError, match="cannot be used with 'linear' unless"):
        run_cli(parrot_train.main, training_command('parrot-train', 'linear', tiled_dataset['data'],
                                                    tmp_path / 'net.pt', 'sequence', 1, '--encode', MODEL, '--silent',
                                                    '--embed-mean-subtraction'))
    assert context_dependent_model.calls == 0

    # Within its protein, it is how the fragment differs from the protein
    run_cli(parrot_train.main, training_command('parrot-train', 'linear', tiled_dataset['data'], tmp_path / 'net.pt',
                                                'sequence', 1, '--encode', MODEL, '--silent',
                                                '--embed-mean-subtraction', *context_flags(tiled_dataset)))
    assert (tmp_path / 'net.pt').is_file()
