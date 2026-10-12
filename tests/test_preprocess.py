"""
Tests for parrot-preprocess as a data-preparation tool: cleaning, deduplication,
splitting into training, validation and test sets without leakage
(parrot.tools.splitting, with parrot.tools.mmseqs_search and parrot.tools.kmer_similarity), class balancing of the
training set only, and the files it writes.

The splitting protocol is tested with a fake search, so these tests do not need
MMseqs2; the tests at the end run the real MMseqs2 and are skipped if it is not
installed.
"""

import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from parrot import process_input_data as pid
from parrot.scripts import parrot_preprocess
from parrot.tools import kmer_similarity, mmseqs_search, preproc, splitting

from tests.helpers import data_path

FRACTIONS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def report_lines() -> tuple[preproc.ProcessingReport, list[str]]:
    report = preproc.ProcessingReport()
    return report, report.lines


def run_preprocess(run_cli: Callable[..., Any], input_file: str | Path, output_file: str | Path, datatype: str,
                   mode: str, *extra_arguments: Any) -> None:
    run_cli(parrot_preprocess.main, ['parrot-preprocess', input_file, output_file, '-d', datatype, '-m', mode,
                                     *extra_arguments])


def read_split_output(output_file: Path) -> dict[str, list[list[str]]]:
    """The lines of each set, from a datafile and the split-file written next to it."""
    lines = pid.read_tsv_raw(str(output_file))
    split_file = output_file.with_name(output_file.stem + '_split_file.txt')
    train, val, test = pid.read_split_file(str(split_file), num_samples=len(lines))
    return {name: [lines[i] for i in indices] for name, indices in (('train', train), ('val', val), ('test', test))}


def random_sequence(rng: np.random.Generator, length: int) -> str:
    return ''.join(rng.choice(list(AMINO_ACIDS), length))


def mutate(rng: np.random.Generator, sequence: str, fraction: float) -> str:
    residues = list(sequence)
    for position in rng.choice(len(residues), int(fraction * len(residues)), replace=False):
        residues[position] = rng.choice(list(AMINO_ACIDS))
    return ''.join(residues)


# ..........................................................................................
# Cleaning and deduplication

def test_clean_sequences():
    report, lines = report_lines()
    data = [['a', 'acdef', '1'], ['b', 'ACXEF', '1'], ['c', 'AC', '1'], ['d', 'ACDEFGHIKL', '1'], ['e', 'ACDE', '1']]

    kept = preproc.clean_sequences(data, report, uppercase=True, remove_nonstandard=True, min_length=3, max_length=8)

    assert kept == [['a', 'ACDEF', '1'], ['e', 'ACDE', '1']]
    assert data[0] == ['a', 'acdef', '1']  # the input is not modified
    assert any('(X)' in line for line in lines)


def test_deduplicate_merges_regression_values():
    report, lines = report_lines()
    data = [['a', 'ACD', '1.0'], ['b', 'EFG', '2'], ['c', 'ACD', '3.0'], ['d', 'ACD', 'nan']]
    assert preproc.deduplicate(data, 'merge', 'sequence', False, report) == [['a', 'ACD', '2'], ['b', 'EFG', '2']]


def test_deduplicate_keeps_classification_copies_only_if_they_agree():
    report, lines = report_lines()
    data = [['a', 'ACD', '1'], ['b', 'ACD', '1'], ['c', 'EFG', '0'], ['d', 'EFG', '1'], ['e', 'HIK', '0']]
    assert preproc.deduplicate(data, 'merge', 'sequence', True, report) == [['a', 'ACD', '1'], ['e', 'HIK', '0']]
    assert any('Removed 1 of them' in line for line in lines)


def test_deduplicate_ignores_missing_sequence_labels():
    # 'nan' never equals 'nan', so copies with missing labels were counted as disagreeing and removed
    report, lines = report_lines()
    data = [['a', 'ACD', 'nan'], ['b', 'ACD', '1'], ['c', 'EFG', 'nan'], ['d', 'EFG', 'NaN']]
    assert preproc.deduplicate(data, 'merge', 'sequence', True, report) == [['b', 'ACD', '1'], ['c', 'EFG', 'nan']]
    assert any('Removed 0 of them whose copies had different labels' in line for line in lines)


def test_deduplicate_merges_residue_labels():
    report, _ = report_lines()
    classification = [['a', 'ACD', '0', '1', 'nan'], ['b', 'ACD', '0', '2', '1'], ['c', 'EF', '1', '1']]
    assert preproc.deduplicate(classification, 'merge', 'residues', True, report) == [
        ['a', 'ACD', '0', 'nan', '1'], ['c', 'EF', '1', '1']]

    regression = [['a', 'ACD', '1', '2', 'nan'], ['b', 'ACD', '3', 'nan', 'nan']]
    assert preproc.deduplicate(regression, 'merge', 'residues', False, report) == [['a', 'ACD', '2', '2', 'nan']]


def test_deduplicate_treats_labels_with_the_same_value_as_one_class():
    report, _ = report_lines()
    sequences = [['a', 'ACDEFGHIK', '1'], ['b', 'ACDEFGHIK', '1.0']]
    assert preproc.deduplicate(sequences, 'merge', 'sequence', True, report) == [['a', 'ACDEFGHIK', '1']]
    residues = [['a', 'ACD', 'nan', '0', '1'], ['b', 'ACD', '2.0', '0.0', '1.0']]
    assert preproc.deduplicate(residues, 'merge', 'residues', True, report) == [['a', 'ACD', '2.0', '0', '1']]


def test_deduplicate_removes_residue_sequences_left_with_no_label():
    report, lines = report_lines()
    data = [['a', 'ACDE', '0', '0', '1', '1'], ['b', 'ACDE', '1', '1', '0', '0'], ['c', 'EF', '1', '1']]
    assert preproc.deduplicate(data, 'merge', 'residues', True, report) == [['c', 'EF', '1', '1']]
    assert any('1 sequences left with no labelled residue were removed' in line for line in lines)


@pytest.mark.parametrize('lines, needed', [
    ([['a', 'AC', '0', '1'], ['b', 'D', 'nan']], False),
    ([['a', 'AC', '0', '1.0']], False),
    ([['a', 'AC', '0', '-1']], True),
    ([['a', 'AC', '0', '0.5']], True),
    ([['a', 'AC', 'yes', 'no']], True),
])
def test_labels_need_recoding(lines, needed):
    assert preproc.labels_need_recoding(lines) == needed


@pytest.mark.parametrize('mode, expected', [
    ('first', [['a', 'ACD', '1'], ['b', 'EFG', '0']]),
    ('drop', [['b', 'EFG', '0']]),
])
def test_deduplicate_first_and_drop(mode, expected):
    report, _ = report_lines()
    data = [['a', 'ACD', '1'], ['b', 'EFG', '0'], ['c', 'ACD', '0']]
    assert preproc.deduplicate(data, mode, 'sequence', True, report) == expected


def test_deduplicate_rejects_unknown_modes():
    with pytest.raises(ValueError, match='Unknown deduplication mode'):
        preproc.deduplicate([], 'average', 'sequence', False, preproc.ProcessingReport())


@pytest.mark.parametrize('value, text', [(2.0, '2'), (2.5, '2.5'), (np.nan, 'nan'), (-3.0, '-3')])
def test_format_number(value, text):
    assert preproc.format_number(value) == text


def test_write_dataset_orders_sets_and_writes_a_matching_split_file(tmp_path):
    lines = [['a', 'A', '1'], ['b', 'C', '2'], ['c', 'D', '3'], ['d', 'E', '4']]
    datafile, split_file = tmp_path / 'out.tsv', tmp_path / 'out_split_file.txt'

    preproc.write_dataset(lines, ['test', 'train', 'val', 'train'], str(datafile), str(split_file),
                          preproc.ProcessingReport())

    assert [line[0] for line in pid.read_tsv_raw(str(datafile))] == ['b', 'd', 'c', 'a']
    train, val, test = pid.read_split_file(str(split_file), num_samples=4)
    assert list(train) == [0, 1] and list(val) == [2] and list(test) == [3]


# ..........................................................................................
# The homology protocol, with a fake search

def make_alignment(query: int, target: int, query_span: tuple[int, int], query_length: int,
                   target_span: tuple[int, int], target_length: int, evalue: float, alignment_length: int = 0,
                   linked: bool = False) -> mmseqs_search.Alignment:
    """An MMseqs2 alignment between the spans given (from 1, ends included); by default without gaps."""
    return mmseqs_search.Alignment(query=query, target=target, identity=0.5,
                                   alignment_length=alignment_length or query_span[1] - query_span[0] + 1,
                                   query_start=query_span[0], query_end=query_span[1], query_length=query_length,
                                   target_start=target_span[0], target_end=target_span[1],
                                   target_length=target_length, evalue=evalue, linked=linked)


class FamilySearch:
    """A stand-in for MMseqs2: sequences of the same family (looked up by sequence) align strongly."""

    def __init__(self, families: dict[str, str]) -> None:
        self.families = families

    def describe(self) -> str:
        return 'same family'

    def summarise(self, matches) -> str:
        return f'{len(matches)} matches'

    def __call__(self, queries, targets=None, weaker=False):
        alignments = []
        target_list = queries if targets is None else targets
        for q, query in enumerate(queries):
            for t, target in enumerate(target_list):
                if targets is None and q == t:
                    continue
                if self.families.get(query) is not None and self.families.get(query) == self.families.get(target):
                    alignments.append(make_alignment(q, t, (1, len(query)), len(query), (1, len(target)),
                                                     len(target), 1e-20, linked=True))
        return alignments


class FixedSearch(FamilySearch):
    """Always returns the same alignments."""

    def __init__(self, alignments):
        self.alignments = alignments

    def __call__(self, queries, targets=None, weaker=False):
        return self.alignments


class PairSearch(FamilySearch):
    """Only the pairs of sequences given align (both ways round, as with MMseqs2)."""

    def __init__(self, pairs: set[tuple[str, str]]) -> None:
        self.pairs = pairs | {(second, first) for first, second in pairs}

    def __call__(self, queries, targets=None, weaker=False):
        target_list = queries if targets is None else targets
        return [make_alignment(q, t, (1, len(query)), len(query), (1, len(target)), len(target), 1e-20, linked=True)
                for q, query in enumerate(queries) for t, target in enumerate(target_list)
                if (query, target) in self.pairs]


def make_family_items(n_families: int, members: int) -> tuple[list[splitting.SplitItem], dict[str, str]]:
    items, families = [], {}
    for family in range(n_families):
        for member in range(members):
            sequence = f'F{family}M{member}'
            families[sequence] = f'family{family}'
            items.append(splitting.SplitItem(len(items), sequence, f'line {len(items)}'))
    return items, families


def test_homology_split_keeps_families_together():
    items, families = make_family_items(n_families=40, members=3)
    kept = splitting.similarity_split(items, FRACTIONS, np.random.default_rng(0), lambda line: None,
                                     FamilySearch(families))

    assert len(kept) == len(items)
    sets_of_family = {}
    for item in kept:
        sets_of_family.setdefault(families[item.sequence], set()).add(item.split)
    assert all(len(sets) == 1 for sets in sets_of_family.values())
    counts = Counter(item.split for item in kept)
    assert counts == {'train': 84, 'val': 18, 'test': 18}


def test_representative_clusters_do_not_chain():
    # a-b and b-c are linked, but a and c are not: c must not join a's cluster through b
    clusters = splitting.representative_clusters(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')], {'b': 1, 'a': 2, 'c': 1})
    assert clusters['a'] == clusters['b'] != clusters['c']


def test_enforce_separation_removes_lower_priority_homologues():
    items = [splitting.SplitItem(0, 'X1', 'g0', 'test'), splitting.SplitItem(1, 'X2', 'g1', 'train'),
             splitting.SplitItem(2, 'X3', 'g2', 'val'), splitting.SplitItem(3, 'Y1', 'g3', 'train'),
             splitting.SplitItem(4, 'Y2', 'g4', 'val')]
    families = {'X1': 'x', 'X2': 'x', 'X3': 'x', 'Y1': 'y', 'Y2': 'y'}
    lines = []

    kept = splitting.enforce_separation(items, FamilySearch(families), lines.append)

    # test keeps X1; val loses X3 (homologous to test) but keeps Y2; train loses X2 and Y1
    assert sorted(item.sequence for item in kept) == ['X1', 'Y2']
    assert lines[-1].startswith('PASSED')


def test_enforce_separation_keeps_records_similar_only_to_removed_ones():
    # V (validation) is similar to T (test) and is removed; R (training) is similar only to V, so it stays
    items = [splitting.SplitItem(0, 'T', 'g0', 'test'), splitting.SplitItem(1, 'V', 'g1', 'val'),
             splitting.SplitItem(2, 'R', 'g2', 'train')]
    kept = splitting.enforce_separation(items, PairSearch({('T', 'V'), ('V', 'R')}), lambda line: None)
    assert [item.sequence for item in kept] == ['T', 'R']


def test_similarity_clusters_count_each_similar_pair_once():
    items = [splitting.SplitItem(i, sequence, f'g{i}') for i, sequence in enumerate('ABCDEF')]
    lines = []
    splitting.similarity_clusters(items, PairSearch({('A', 'B'), ('C', 'D'), ('E', 'F')}), lines.append)
    assert lines[0].endswith('3 similar pairs, of which 3 link two groups')


def test_enforce_separation_removes_identical_sequences_even_without_alignments():
    items = [splitting.SplitItem(0, 'SAME', 'g0', 'test'), splitting.SplitItem(1, 'SAME', 'g1', 'train'),
             splitting.SplitItem(2, 'OTHER', 'g2', 'train')]
    kept = splitting.enforce_separation(items, FixedSearch([]), lambda line: None)
    assert [item.index for item in kept] == [0, 2]


def test_enforce_separation_gives_up_after_too_many_rounds():
    items = [splitting.SplitItem(0, 'A', 'g0', 'test'), splitting.SplitItem(1, 'B', 'g1', 'train')]
    always = make_alignment(0, 0, (1, 10), 10, (1, 10), 10, 1e-10, linked=True)
    items += [splitting.SplitItem(i, f'C{i}', f'g{i}', 'train') for i in range(2, 2 + splitting.MAX_SEPARATION_ROUNDS)]
    with pytest.raises(RuntimeError, match='still not separated'):
        splitting.enforce_separation(items, FixedSearch([always]), lambda line: None)


def test_alignment_homology_criteria():
    criteria = mmseqs_search.HomologyCriteria()
    assert make_alignment(0, 1, (1, 30), 100, (51, 80), 200, 1e-4).homologous(criteria)
    assert make_alignment(0, 1, (51, 80), 200, (1, 30), 100, 1e-4).homologous(criteria)  # the target is shorter
    assert not make_alignment(0, 1, (1, 29), 100, (1, 29), 200, 1e-10).homologous(criteria)  # covers < 30%
    assert not make_alignment(0, 1, (1, 100), 100, (1, 100), 200, 2e-3).homologous(criteria)  # E too large


def test_alignment_coverage_leaves_out_gaps_opposite_insertions():
    # 25 residues of the shorter sequence align across an 80-residue insertion in the longer one: 105 columns,
    # but only 25% of the shorter sequence
    alignment = make_alignment(0, 1, (1, 25), 100, (1, 105), 300, 1e-10, alignment_length=105)
    assert alignment.shorter_coverage == 0.25
    assert not alignment.homologous(mmseqs_search.HomologyCriteria())


def test_parse_alignment_reads_every_column():
    alignment = mmseqs_search.parse_alignment('3\t7\t0.512\t120\t5\t118\t130\t11\t126\t140\t2.5E-20\n')
    assert alignment == mmseqs_search.Alignment(query=3, target=7, identity=0.512, alignment_length=120,
                                                query_start=5, query_end=118, query_length=130, target_start=11,
                                                target_end=126, target_length=140, evalue=2.5e-20)


@pytest.mark.parametrize('criteria_class', [mmseqs_search.HomologyCriteria, kmer_similarity.KmerCriteria])
def test_the_separation_check_reports_every_linked_pair_however_loose_the_criterion(criteria_class):
    assert criteria_class().reported_evalue == criteria_class().report_max_evalue
    assert criteria_class(max_evalue=1.0).reported_evalue == 1.0


def test_merge_groups_joins_linked_groups():
    items = [splitting.SplitItem(i, sequence, f'g{i}') for i, sequence in enumerate(['A', 'B', 'A', 'C', 'B'])]
    splitting.merge_groups(items, splitting.identical_sequence_links(items))
    assert items[0].group == items[2].group and items[1].group == items[4].group
    assert len({item.group for item in items}) == 3


def test_random_split_keeps_identical_sequences_together():
    items = [splitting.SplitItem(i, f'S{i % 25}', f'line {i}') for i in range(100)]
    kept = splitting.random_split(items, FRACTIONS, np.random.default_rng(1), lambda line: None)
    sets_of_sequence = {}
    for item in kept:
        sets_of_sequence.setdefault(item.sequence, set()).add(item.split)
    assert all(len(sets) == 1 for sets in sets_of_sequence.values())


def test_assign_clusters_rejects_bad_fractions():
    with pytest.raises(ValueError, match='sum to 1'):
        splitting.assign_clusters([], {}, {'train': 0.5, 'val': 0.1, 'test': 0.1}, np.random.default_rng(0),
                                 lambda line: None)


def test_find_mmseqs_explains_how_to_install_it():
    with pytest.raises(RuntimeError, match='conda install'):
        mmseqs_search.find_mmseqs('/no/such/mmseqs')


# ..........................................................................................
# parrot-preprocess end to end (without MMseqs2)

def test_preprocess_random_split_writes_a_matching_split_file(run_cli, tmp_path):
    output_file = tmp_path / 'out.tsv'
    run_preprocess(run_cli, data_path('seq_regress_dataset.tsv'), output_file, 'sequence', 'regression',
                   '--split', 'random', '--seed', '1')

    sets = read_split_output(output_file)
    assert {name: len(lines) for name, lines in sets.items()} == {'train': 210, 'val': 45, 'test': 45}
    report = (tmp_path / 'out_preprocess_report.txt').read_text()
    assert 'the split-file matches it' in report


def test_preprocess_split_is_reproducible(run_cli, tmp_path):
    for run in range(2):
        run_preprocess(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / f'out{run}.tsv', 'sequence',
                       'regression', '--split', 'random', '--seed', '4')
    assert (tmp_path / 'out0.tsv').read_text() == (tmp_path / 'out1.tsv').read_text()
    assert (tmp_path / 'out0_split_file.txt').read_text() == (tmp_path / 'out1_split_file.txt').read_text()


def test_preprocess_groups_stay_in_one_set(run_cli, tmp_path, write_lines):
    data = pid.read_tsv_raw(data_path('seq_regress_dataset.tsv'))
    groups_file = write_lines('groups.tsv', [f'{line[0]} protein{i // 10}' for i, line in enumerate(data)])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, data_path('seq_regress_dataset.tsv'), output_file, 'sequence', 'regression',
                   '--split', 'random', '--groups', groups_file, '--seed', '2')

    protein_of = {line[0]: i // 10 for i, line in enumerate(data)}
    sets_of_protein = {}
    for name, lines in read_split_output(output_file).items():
        for line in lines:
            sets_of_protein.setdefault(protein_of[line[0]], set()).add(name)
    assert all(len(sets) == 1 for sets in sets_of_protein.values())


@pytest.mark.parametrize('deduplication', [[], ['--deduplicate', 'merge'], ['--deduplicate', 'first']])
def test_preprocess_groups_stay_in_one_set_when_deduplication_removes_a_member(run_cli, tmp_path, write_lines,
                                                                                deduplication):
    # Proteins p and q share an identical first tile; deduplication keeps p's copy, so q's other tile must
    # follow it into p's set
    rng = np.random.default_rng(2)
    lines, groups = [], []
    for i in range(40):
        shared, p_tile, q_tile = (random_sequence(rng, 40) for _ in range(3))
        for tile_id, tile, protein in ((f'p{i}_1', shared, f'P{i}'), (f'p{i}_2', p_tile, f'P{i}'),
                                       (f'q{i}_1', shared, f'Q{i}'), (f'q{i}_2', q_tile, f'Q{i}')):
            lines.append(f'{tile_id} {tile} {rng.normal():.3f}')
            groups.append(f'{tile_id} {protein}')
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, write_lines('in.tsv', lines), output_file, 'sequence', 'regression', '--split',
                   'random', '--groups', write_lines('groups.tsv', groups), '--seed', '1', *deduplication)

    set_of_tile = {line[0]: name for name, set_lines in read_split_output(output_file).items() for line in set_lines}
    assert all(set_of_tile[f'p{i}_2'] == set_of_tile[f'q{i}_2'] for i in range(40))
    report = (tmp_path / 'out_preprocess_report.txt').read_text()
    if deduplication:
        assert '40 lines also belong to the group of a copy of their sequence' in report
    else:
        # Both copies are still there (identical sequences are kept in one set anyway), so nothing was removed
        assert 'deduplication removed' not in report


def test_preprocess_balances_only_the_training_set(run_cli, tmp_path, write_lines):
    rng = np.random.default_rng(0)
    lines = [f'seq{i} {random_sequence(rng, 20)} {int(i % 4 == 0)}' for i in range(400)]
    input_file = write_lines('in.tsv', lines)
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'classification', '--split', 'random',
                   '--class-balance', '--cb-max-fraction-removed', '0.6', '--seed', '3')

    sets = read_split_output(output_file)
    train_labels = Counter(line[2] for line in sets['train'])
    assert train_labels['0'] / train_labels['1'] <= 1.4
    held_out = sets['val'] + sets['test']
    assert len(held_out) == 120  # untouched by balancing
    assert Counter(line[2] for line in held_out)['0'] / Counter(line[2] for line in held_out)['1'] > 2


def test_preprocess_deduplicates_before_splitting(run_cli, tmp_path, write_lines):
    rng = np.random.default_rng(5)
    sequences = [random_sequence(rng, 15) for _ in range(60)]
    input_file = write_lines('in.tsv', [f'seq{i} {sequences[i % 60]} {i % 7}' for i in range(120)])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'regression', '--deduplicate', 'merge',
                   '--split', 'random', '--seed', '1')

    lines = pid.read_tsv_raw(str(output_file))
    assert len(lines) == 60 and len({line[1] for line in lines}) == 60


def test_preprocess_writes_a_report(run_cli, tmp_path, write_lines, capsys):
    input_file = write_lines('in.tsv', ['a acd 1', 'b ACD 2', 'c EFG 3', 'd ACXD 4'])
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'regression', '--uppercase', '--remove-nonstandard',
                   '--deduplicate', 'merge')

    assert pid.read_tsv_raw(str(output_file)) == [['a', 'ACD', '1.5'], ['c', 'EFG', '3']]
    report = (tmp_path / 'out_preprocess_report.txt').read_text()
    assert 'Cleaning sequences' in report and 'Duplicate sequences (merge)' in report
    assert not (tmp_path / 'out_split_file.txt').exists()
    assert 'out_preprocess_report.txt' in capsys.readouterr().out


@pytest.mark.parametrize('extra_arguments, message', [
    (['--groups', 'groups.tsv'], 'only applies to --split'),
    (['--set-fractions', '0.8', '0.1', '0.1'], 'only applies to --split'),
    (['--split', 'random', '--set-fractions', '0.8', '0.1', '0.2'], 'sum to 1'),
    (['--split', 'random', '--set-fractions', '1.0', '0.0', '0.0'], 'between 0 and 1'),
    (['--split', 'homology', '--mmseqs', '/no/such/mmseqs'], 'conda install'),
    (['--split', 'kmer', '--homology-coverage', '1.5'], 'between 0 and 1'),
    (['--split', 'random', '--homology-coverage', '0.5'], 'only applies to --split homology or kmer'),
    (['--split', 'kmer', '--mmseqs', 'mmseqs'], 'only applies to --split homology'),
    (['--split', 'homology', '--kmer-size', '4'], 'only applies to --split kmer'),
    (['--split', 'kmer', '--kmer-size', '2'], 'between 3 and 8'),
    (['--min-length', '10', '--max-length', '5'], 'cannot be larger'),
    (['--remove-sequence-duplicates', '--deduplicate', 'merge'], 'same as --deduplicate drop'),
    (['--split', 'homology', '--threads', '0'], '--threads must be at least 1'),
])
def test_preprocess_rejects_invalid_options(run_cli, tmp_path, extra_arguments, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        run_preprocess(run_cli, data_path('seq_regress_dataset.tsv'), tmp_path / 'out.tsv', 'sequence', 'regression',
                       *extra_arguments)


@pytest.mark.parametrize('lines, datatype, message', [
    (['a ACD'], 'sequence', 'expected <ID> <sequence> <value'),
    (['a ACD 1 2'], 'sequence', 'one value per sequence'),
    (['a ACD 1 2'], 'residues', '2 values for a sequence of 3 residues'),
])
def test_preprocess_checks_the_input_lines(run_cli, tmp_path, write_lines, lines, datatype, message):
    with pytest.raises(ValueError, match=message):
        run_preprocess(run_cli, write_lines('in.tsv', lines), tmp_path / 'out.tsv', datatype, 'regression')


@pytest.mark.parametrize('lines, arguments, message', [
    ([], ['--split', 'random'], 'has no data lines'),
    (['a ACD 1', 'b EFG 2'], ['--min-length', '500', '--split', 'random'],
     'Cleaning removed every line.*shorter than 500 residues'),
    (['a ACD 1', 'b ACD 2'], ['--deduplicate', 'drop', '--class-balance'], 'Removing duplicates removed every line'),
])
def test_preprocess_stops_when_no_lines_remain(run_cli, tmp_path, write_lines, lines, arguments, message):
    mode = 'classification' if '--class-balance' in arguments else 'regression'
    with pytest.raises(ValueError, match=message):
        run_preprocess(run_cli, write_lines('in.tsv', lines), tmp_path / 'out.tsv', 'sequence', mode, *arguments)


def test_preprocess_reads_delimited_fields_without_the_spaces_around_them(run_cli, tmp_path, write_lines):
    input_file = write_lines('in.csv', ['a, ACDE, 1.0', 'b ,EFGH ,2', 'c,KLMN,3'])
    output_file = tmp_path / 'out.tsv'
    run_preprocess(run_cli, input_file, output_file, 'sequence', 'regression', '--delimiter', ',',
                   '--remove-nonstandard')
    assert pid.read_tsv_raw(str(output_file)) == [['a', 'ACDE', '1.0'], ['b', 'EFGH', '2'], ['c', 'KLMN', '3']]


@pytest.mark.parametrize('lines, message', [
    (['sp|P1|A protein one\tACDE\t1.0'], "field containing whitespace \\('sp\\|P1\\|A protein one'\\)"),
    (['\tACDE\t1.0'], 'empty column'),
    (['a\tACDE\t1.0\t'], 'empty column'),
])
def test_preprocess_rejects_delimited_fields_a_datafile_cannot_hold(run_cli, tmp_path, write_lines, lines, message):
    with pytest.raises(ValueError, match=message):
        run_preprocess(run_cli, write_lines('in.tsv', lines), tmp_path / 'out.tsv', 'sequence', 'regression',
                       '--delimiter', '\t')


@pytest.mark.parametrize('arguments, message', [(['--cb-max-fraction-removed', '1'],
                                                 '--cb-max-fraction-removed must be between 0 and 1'),
                                                (['--cb-balance-threshold', '0.5'],
                                                 '--cb-balance-threshold must be between 1'),
                                                (['--cb-block-size', '1'], '--cb-block-size must be at least 2')])
def test_preprocess_checks_class_balancing_options_before_splitting(run_cli, tmp_path, monkeypatch, arguments,
                                                                     message):
    monkeypatch.setattr(splitting, 'random_split', lambda *args: pytest.fail('split anyway'))
    with pytest.raises(Exception, match=message):
        run_preprocess(run_cli, data_path('seq_class_dataset.tsv'), tmp_path / 'out.tsv', 'sequence',
                       'classification', '--split', 'random', '--class-balance', *arguments)


def test_preprocess_class_balancing_says_when_every_label_is_missing(run_cli, tmp_path, write_lines):
    # It used to stop with numpy's "zero-size array to reduction operation" while describing the classes
    input_file = write_lines('in.tsv', ['a MKVLA nan', 'b MKVLC nan', 'c MKVLD nan'])
    with pytest.raises(ValueError, match='found no class labels in the data'):
        run_preprocess(run_cli, input_file, tmp_path / 'out.tsv', 'sequence', 'classification', '--class-balance')


@pytest.mark.parametrize('labels, checked', [
    (['0', '0', '0'], 'Checked: PARROT reads the output as sequence-mapped classification data with 2 classes'),
    (['0', 'inf', '1'], 'WARNING: PARROT cannot train on the output as it is: Data line 2'),
    (['0', 'helix', '1'], 'WARNING: PARROT cannot train on the output as it is: Data line 2'),
])
def test_preprocess_output_check_counts_at_least_two_classes_and_reports_infinite_labels(run_cli, tmp_path,
                                                                                         write_lines, labels, checked):
    # A file of 0s was reported as having "1 classes", and an infinite label crashed with OverflowError
    input_file = write_lines('in.tsv', [f'{name} MKVL{residue} {label}'
                                        for name, residue, label in zip('abc', 'ACD', labels)])
    run_preprocess(run_cli, input_file, tmp_path / 'out.tsv', 'sequence', 'classification')
    assert checked in (tmp_path / 'out_preprocess_report.txt').read_text()


def test_preprocess_warns_when_split_sets_would_be_empty(run_cli, tmp_path, write_lines):
    input_file = write_lines('in.tsv', ['a ACD 1', 'b EFG 2'])
    with pytest.raises(ValueError, match='after splitting'):
        run_preprocess(run_cli, input_file, tmp_path / 'out.tsv', 'sequence', 'regression', '--split', 'random')


# ..........................................................................................
# The k-mer search

def kmer_families(rng: np.random.Generator, n_families: int, members: int, identity: float,
                  length: int = 150) -> tuple[list[str], list[int]]:
    """Families of sequences whose members are, on average, `identity` identical to each other."""
    sequences, family_of = [], []
    # Each member is mutated independently from an ancestor, so is sqrt(identity) identical to it
    replaced = (1 - np.sqrt(identity)) * 20 / 19  # a random replacement is the same residue 1 time in 20
    for family in range(n_families):
        ancestor = random_sequence(rng, length)
        for _ in range(members):
            sequences.append(mutate(rng, ancestor, replaced))
            family_of.append(family)
    return sequences, family_of


def covered_by(starts: set[int], k: int) -> int:
    """Residues covered by k-mers starting at these positions."""
    return len({start + offset for start in starts for offset in range(k)})


def brute_force_best_runs(queries: list[str], criteria: kmer_similarity.KmerCriteria,
                          targets: list[str] | None = None) -> dict[tuple[int, int], tuple[int, int]]:
    """Residues covered and shared k-mers (distinct positions) in each pair's best run, by brute force (see
    find_runs and best_runs)."""
    k, width, gap = criteria.kmer_size, criteria.band_width, criteria.max_gap
    query_table = kmer_similarity.kmer_table(queries, k, criteria.mask_low_complexity)
    target_list = queries if targets is None else targets
    target_table = query_table if targets is None else kmer_similarity.kmer_table(targets, k,
                                                                                   criteria.mask_low_complexity)
    best = {}
    for q in range(len(queries)):
        for t in range(len(target_list)):
            if targets is None and t <= q:
                continue
            in_q, in_t = query_table.sequence == q, target_table.sequence == t
            hits = [(i, j) for i, a in zip(query_table.position[in_q], query_table.codes[in_q])
                    for j, b in zip(target_table.position[in_t], target_table.codes[in_t]) if a == b]
            runs = []
            for shift in (0, width // 2):
                bands = {}
                for i, j in hits:
                    bands.setdefault((i - j + target_table.longest + shift) // width, []).append((i, j))
                for band in bands.values():
                    band.sort()
                    run = [band[0]]
                    for hit in band[1:]:
                        if hit[0] - run[-1][0] > gap:
                            runs.append(run)
                            run = [hit]
                        else:
                            run.append(hit)
                    runs.append(run)
            scores = [(min(covered_by({i for i, _ in run}, k), covered_by({j for _, j in run}, k)),
                       min(len({i for i, _ in run}), len({j for _, j in run}))) for run in runs]
            scores = [score for score in scores if score[1] >= 2]
            if scores:
                best[(q, t)] = max(scores)
    return best


@pytest.mark.parametrize('criteria', [kmer_similarity.KmerCriteria(),
                                      kmer_similarity.KmerCriteria(kmer_size=3, band_width=6, max_gap=10),
                                      kmer_similarity.KmerCriteria(kmer_size=3, mask_low_complexity=False)])
@pytest.mark.parametrize('batch_matches', [10 ** 9, 30])
def test_kmer_search_matches_brute_force(monkeypatch, criteria, batch_matches):
    monkeypatch.setattr(kmer_similarity, 'BATCH_MATCHES', batch_matches)
    rng = np.random.default_rng(0)
    sequences, _ = kmer_families(rng, n_families=6, members=3, identity=0.65, length=60)
    sequences += ['APAPAPAPAKLMNAPAPAPAP', 'APAPAPAPAKLMWAPAPAPAPAPAP', 'QQQQQQQQQQQQQQQQQQ', 'ACDXXACDEF']
    everything = kmer_similarity.KmerCriteria(**{**criteria.__dict__, 'report_max_evalue': 1e300})
    search = kmer_similarity.KmerSearch(everything)

    for queries, targets in ((sequences, None), (sequences[:9], sequences[9:])):
        found = {(match.query, match.target): (match.covered, match.shared)
                 for match in search(queries, targets, weaker=True)}
        assert found == brute_force_best_runs(queries, criteria, targets)


@pytest.mark.parametrize('identity', [0.9, 0.7])
def test_kmer_search_links_close_relatives_and_nothing_else(identity):
    rng = np.random.default_rng(1)
    sequences, family_of = kmer_families(rng, n_families=40, members=2, identity=identity)
    sequences += [random_sequence(rng, 150) for _ in range(100)]
    family_of += list(range(100, 200))

    linked = {(match.query, match.target) for match in kmer_similarity.KmerSearch()(sequences) if match.linked}

    relatives = {(i, j) for i in range(len(sequences)) for j in range(i + 1, len(sequences))
                 if family_of[i] == family_of[j]}
    assert linked == relatives


def test_kmer_search_evalues_are_not_too_significant_for_unrelated_sequences():
    # All against all, about 600 x 0.01 / 2 = 3 pairs of unrelated sequences would reach E <= 0.01 by chance if
    # the E-values were exact (fewer, as they are slightly conservative). Counting overlapping shared k-mers as
    # independent chances made it about 23, with single shared words reported as nearly identical.
    rng = np.random.default_rng(4)
    sequences = [random_sequence(rng, 150) for _ in range(600)]
    criteria = kmer_similarity.KmerCriteria(report_max_evalue=1e-2)

    reported = kmer_similarity.KmerSearch(criteria)(sequences, weaker=True)

    assert len(reported) <= 6
    assert all(match.identity < 0.9 for match in reported)


@pytest.mark.parametrize('residue_match', [0.06, 0.2])
def test_chance_of_a_run_filling_a_short_sequence_is_not_understated(residue_match):
    # Only one 4-residue stretch fits in 12 residues, so covering them all takes 8 more identical residues: p^8.
    # The run's chance of going on (which it cannot) was left out, understating this 4.6-fold at p = 0.2
    chance = kmer_similarity.chance_of_run(np.array([12]), np.array([residue_match]), np.array([12]),
                                           kmer_similarity.KmerCriteria(kmer_size=4))
    assert chance[0] >= residue_match ** 8 * (1 - 1e-12)


def test_kmer_search_with_a_loose_criterion_reports_every_linked_pair():
    rng = np.random.default_rng(6)
    sequences, _ = kmer_families(rng, n_families=20, members=2, identity=0.45)
    search = kmer_similarity.KmerSearch(kmer_similarity.KmerCriteria(max_evalue=10.0, min_coverage=0.0))
    queries, targets = sequences[::2], sequences[1::2]

    linked = {(match.query, match.target) for match in search(queries, targets) if match.linked}
    reported = {(match.query, match.target) for match in search(queries, targets, weaker=True) if match.linked}

    assert linked and reported == linked


def test_kmer_search_measures_coverage_across_an_insertion():
    rng = np.random.default_rng(2)
    original = random_sequence(rng, 300)
    relative = mutate(rng, original, 0.3)
    relative = relative[:150] + random_sequence(rng, 25) + relative[150:]

    (match,) = kmer_similarity.KmerSearch()([original], [relative])

    assert match.linked and match.coverage > 0.9


def test_kmer_search_ignores_a_short_shared_segment():
    # A shared 15-residue tag is significant, but covers too little of either protein to link them
    rng = np.random.default_rng(3)
    tag = 'MASMTGGQQMGRDPN'
    first, second = tag + random_sequence(rng, 135), tag + random_sequence(rng, 135)
    matches = kmer_similarity.KmerSearch()([first], [second], weaker=True)
    assert len(matches) == 1 and matches[0].evalue < 1e-3 and not matches[0].linked


def test_kmer_table_masks_low_complexity_and_nonstandard_residues():
    table = kmer_similarity.kmer_table(['QQQQQQQQQQQQQQQQ', 'SGSGSGSGSGSGSGSGSG', 'ACDEFGHIKX', 'acdefghik'], 4, True)
    assert table.counts.tolist() == [0, 0, 6, 6]
    unmasked = kmer_similarity.kmer_table(['QQQQQQQQQQQQQQQQ'], 4, False)
    assert unmasked.counts.tolist() == [13]


@pytest.mark.parametrize('limit', [1, 5, 12, 1000])
def test_query_batches_cover_every_row_within_the_limit(limit):
    query_sequence = np.array([0, 0, 1, 1, 1, 2, 4, 4])
    hits = np.array([3, 1, 0, 2, 2, 7, 1, 1])
    batches = kmer_similarity.query_batches(query_sequence, hits, limit)

    rows = [row for batch in batches for row in range(batch.start, batch.stop)]
    assert rows == list(range(len(query_sequence)))
    for batch in batches:
        sequences = set(query_sequence[batch].tolist())
        assert hits[batch].sum() <= limit or len(sequences) == 1
        assert batch.start == 0 or query_sequence[batch.start] != query_sequence[batch.start - 1]


def test_preprocess_kmer_split_keeps_families_together(run_cli, tmp_path, write_lines):
    rng = np.random.default_rng(11)
    sequences, family_of = kmer_families(rng, n_families=30, members=4, identity=0.7, length=120)
    lines = [f'fam{family}_{i} {sequence} {rng.normal():.3f}' for i, (sequence, family) in
             enumerate(zip(sequences, family_of))]
    lines += [f'single{i} {random_sequence(rng, 120)} {rng.normal():.3f}' for i in range(40)]
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, write_lines('in.tsv', lines), output_file, 'sequence', 'regression', '--split', 'kmer',
                   '--seed', '1')

    sets_of_family = {}
    for name, set_lines in read_split_output(output_file).items():
        for line in set_lines:
            if line[0].startswith('fam'):
                sets_of_family.setdefault(line[0].split('_')[0], set()).add(name)
    assert len(sets_of_family) == 30
    assert all(len(sets) == 1 for sets in sets_of_family.values())
    report = (tmp_path / 'out_preprocess_report.txt').read_text()
    assert 'PASSED' in report and 'k-mer search' in report


# ..........................................................................................
# With the real MMseqs2

needs_mmseqs = pytest.mark.skipif(shutil.which('mmseqs') is None, reason='MMseqs2 (mmseqs) is not installed')


@needs_mmseqs
def test_preprocess_homology_split_with_mmseqs(run_cli, tmp_path, write_lines):
    # 30 protein families of 4 members each (75% identical to their family's ancestor), and 40 unrelated proteins
    rng = np.random.default_rng(11)
    lines, family_of = [], {}
    for family in range(30):
        ancestor = random_sequence(rng, 120)
        for member in range(4):
            name = f'fam{family}_{member}'
            family_of[name] = family
            lines.append(f'{name} {mutate(rng, ancestor, 0.25)} {rng.normal():.3f}')
    for single in range(40):
        lines.append(f'single{single} {random_sequence(rng, 120)} {rng.normal():.3f}')
    input_file = write_lines('in.tsv', lines)
    output_file = tmp_path / 'out.tsv'

    run_preprocess(run_cli, input_file, output_file, 'sequence', 'regression', '--split', 'homology', '--seed', '1',
                   '--threads', '2')

    sets_of_family = {}
    for name, set_lines in read_split_output(output_file).items():
        for line in set_lines:
            if line[0] in family_of:
                sets_of_family.setdefault(family_of[line[0]], set()).add(name)
    assert len(sets_of_family) == 30
    assert all(len(sets) == 1 for sets in sets_of_family.values())
    report = (tmp_path / 'out_preprocess_report.txt').read_text()
    assert 'PASSED' in report and 'MMseqs2 ' in report


@needs_mmseqs
def test_mmseqs_search_finds_homologues(tmp_path):
    rng = np.random.default_rng(3)
    ancestor = random_sequence(rng, 150)
    sequences = [ancestor, mutate(rng, ancestor, 0.3), random_sequence(rng, 150)]
    search = mmseqs_search.MMseqsSearch(mmseqs_search.HomologyCriteria(), threads=2, workdir=tmp_path)

    alignments = search(sequences)

    assert {(a.query, a.target) for a in alignments if a.linked} == {(0, 1), (1, 0)}
    assert list(tmp_path.iterdir()) == []  # temporary files are removed


@needs_mmseqs
def test_mmseqs_search_with_a_loose_criterion_reports_every_linked_pair(tmp_path):
    rng = np.random.default_rng(7)
    sequences, _ = kmer_families(rng, n_families=10, members=2, identity=0.3)
    sequences += [random_sequence(rng, 150) for _ in range(20)]
    criteria = mmseqs_search.HomologyCriteria(max_evalue=1.0)
    search = mmseqs_search.MMseqsSearch(criteria, threads=2, workdir=tmp_path)
    queries, targets = sequences[::2], sequences[1::2]

    linked = {(a.query, a.target) for a in search(queries, targets) if a.linked}
    reported = {(a.query, a.target) for a in search(queries, targets, weaker=True) if a.linked}

    assert any(a.evalue > criteria.report_max_evalue for a in search(queries, targets) if a.linked)
    assert reported == linked


@needs_mmseqs
def test_mmseqs_search_reads_peptides_of_nucleotide_letters_as_proteins(tmp_path):
    rng = np.random.default_rng(8)
    peptides = [''.join(rng.choice(list('ACGTN'), 80)) for _ in range(10)]
    sequences = peptides + [random_sequence(rng, 80) for _ in range(10)]
    search = mmseqs_search.MMseqsSearch(mmseqs_search.HomologyCriteria(), threads=2, workdir=tmp_path)
    search(sequences)  # without --dbtype 1, MMseqs2 reads the database as nucleotides and fails
