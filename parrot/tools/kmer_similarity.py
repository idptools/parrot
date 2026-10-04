"""
Finding similar protein sequences from shared k-mers, without external programs.

This is the search engine behind parrot-preprocess --split kmer, for when
MMseqs2 is not available. It works like the seeding step of BLAST, FASTA or
MMseqs2: related sequences share short identical words (k-mers), and because
they are related, the shared words line up along one alignment diagonal (the
same offset between their positions in the two sequences). Unrelated sequences
also share k-mers by chance, but scattered over many diagonals.

For each pair of sequences, shared k-mers are collected in bands of nearby
diagonals (KmerCriteria.band_width wide, so short insertions and deletions are
tolerated) and chained into runs along the sequences (a run breaks where two
shared k-mers are more than KmerCriteria.max_gap residues apart). Each pair is
then scored in the same terms as the MMseqs2 criterion:

* E-value, from the run with the most shared k-mers (counted as distinct
  positions, so a repeat cannot count one k-mer many times). If the two
  sequences were unrelated, a k-mer at one position would match a k-mer at
  another with probability p^k, where p is the chance that two residues drawn
  from the two sequences' amino-acid compositions are identical (so
  compositionally biased sequences, such as many disordered regions, need more
  shared k-mers to count). The number of chance matches in a band is then
  Poisson distributed, and the E-value is the probability of at least as many,
  multiplied by the number of bands compared and the number of target sequences:
  as with MMseqs2 or BLAST, the number of matches this strong one query would
  be expected to find by chance in the database searched.
* Coverage: the stretch spanned by the best run together with every other run
  near its diagonal (within KmerCriteria.max_drift) that is itself unlikely by
  chance, as a fraction of the shorter sequence. This approximates the length
  of the alignment, including where its diagonal drifts through insertions and
  deletions or where poorly conserved stretches break the run.

Two sequences are linked if the E-value is at most KmerCriteria.max_evalue and
the coverage at least KmerCriteria.min_coverage. Low-complexity sequence is
ignored by default, as MMseqs2 ignores it, so that compositional bias and
simple repeats alone do not link sequences: residues in any 12-residue window
with less than 1.8 bits of sequence entropy are masked (the SEG program's window,
Wootton & Federhen 1993, Comput Chem 17:149, with a lower threshold than its 2.2
bits, which would mask half of a typical disordered-region dataset), as are
k-mers with at most k // 2 distinct amino acids (QQQQ, SGSG, ...).

The search is exact (every shared k-mer is counted, nothing is sampled) and is
written with NumPy, working through the queries in batches so memory stays
bounded. Exact k-mer matches find close relatives reliably (at least 50%
identity, and most at 40-50%), but not the distant homologues that MMseqs2 finds
with substitution-aware alignments.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
from scipy.special import pdtrc

from parrot.tools.homology import identity_summary

# The 20 standard amino acids; k-mers containing anything else are ignored
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Code for a residue that is not one of the 20 standard amino acids
NOT_AN_AMINO_ACID = 255

# Low-complexity masking: residues in a window this long (as the SEG program)...
LOW_COMPLEXITY_WINDOW = 12

# ...whose sequence entropy is below this many bits are masked. SEG uses 2.2 bits, which masks half the residues
# of the ALBATROSS disordered-region dataset (and 7-9% of yeast or CheZOD proteins) and costs much of the
# k-mer search's sensitivity there; 1.8 bits masks only the most repetitive sequence.
LOW_COMPLEXITY_ENTROPY = 1.8

# Largest number of k-mer matches (pairs of positions sharing a k-mer) handled at once.
# Each needs about 80 bytes of working memory, so this keeps a batch to about 0.8 GB.
BATCH_MATCHES = 10_000_000

# A run of shared k-mers counts towards the extent of a similarity only if runs at least as strong would arise
# by chance, anywhere near the best run's diagonal, with at most this probability
RUN_CHANCE = 0.01

_RESIDUE_CODES = np.full(256, NOT_AN_AMINO_ACID, dtype=np.uint8)
for _code, _letter in enumerate(AMINO_ACIDS):
    _RESIDUE_CODES[ord(_letter)] = _code
    _RESIDUE_CODES[ord(_letter.lower())] = _code


@dataclass(frozen=True)
class KmerCriteria:
    """When two sequences count as similar

    Attributes
    ----------
    kmer_size : int
        Length of the shared words (k). Smaller words find more distant
        relatives but take longer (the work grows roughly 20-fold for each step
        down) and are more easily matched by chance.
    max_evalue : float
        Largest E-value for two sequences to be linked
    min_coverage : float
        Smallest fraction of the shorter sequence the similarity must span
    band_width : int
        Width of a diagonal band: shared k-mers whose offsets differ by less
        than this can be in one run, tolerating short insertions and deletions
    max_gap : int
        Shared k-mers in a band are chained into runs, broken wherever two are
        more than this many residues apart, so stray chance matches elsewhere in
        the band cannot stretch a run
    max_drift : int
        Coverage is measured over every run within this many diagonals of the
        best one (and unlikely by chance), so that alignments whose diagonal
        drifts through insertions and deletions are measured whole
    mask_low_complexity : bool
        Ignore low-complexity sequence (see the module documentation)
    report_max_evalue : float
        The separation check also reports every weaker similarity up to this
        E-value, so any remaining similarity between sets is visible
    """

    kmer_size: int = 4
    max_evalue: float = 1e-3
    min_coverage: float = 0.3
    band_width: int = 16
    max_gap: int = 40
    max_drift: int = 48
    mask_low_complexity: bool = True
    report_max_evalue: float = 1e-2

    def __post_init__(self) -> None:
        if not 3 <= self.kmer_size <= 8:
            raise ValueError(f'The k-mer size must be between 3 and 8, got {self.kmer_size}')
        if self.band_width < 2 or self.max_gap < 1 or self.max_drift < 0:
            raise ValueError('band_width must be at least 2, max_gap at least 1 and max_drift at least 0')

    def describe(self) -> str:
        """The criterion in words, for reports."""
        return (f'identical, or sharing {self.kmer_size}-mers along nearby alignment diagonals with E <= '
                f'{self.max_evalue:g} and spanning >= {100 * self.min_coverage:g}% of the shorter sequence '
                f'(low-complexity sequence {"masked" if self.mask_low_complexity else "not masked"})')


@dataclass(frozen=True)
class KmerMatch:
    """Two sequences sharing k-mers (indices refer to the query and target lists)

    Attributes
    ----------
    query, target : int
    shared : int
        Shared k-mers (distinct positions) in the best run
    coverage : float
        Fraction of the shorter sequence spanned by the similarity
    evalue : float
        Number of matches at least this strong the query would be expected to find by chance
    identity : float
        Rough identity estimate from how densely the best run's shared k-mers
        cover its span (if a fraction f of positions start a shared k-mer,
        f ~ identity^k)
    linked : bool
        True if the pair meets the criterion
    """

    query: int
    target: int
    shared: int
    coverage: float
    evalue: float
    identity: float
    linked: bool


@dataclass
class KmerTable:
    """Every (unmasked) k-mer of a list of sequences, and each sequence's length and composition

    Attributes
    ----------
    codes : numpy.ndarray of int64
        Each k-mer, as a number
    sequence : numpy.ndarray of int64
        Which sequence it comes from
    position : numpy.ndarray of int64
        Where in that sequence it starts
    counts : numpy.ndarray of int64
        Number of (unmasked) k-mers in each sequence
    lengths : numpy.ndarray of int64
        Length of each sequence
    composition : numpy.ndarray of float
        Fraction of each sequence's standard residues that are each amino acid (one row per sequence)
    """

    codes: np.ndarray
    sequence: np.ndarray
    position: np.ndarray
    counts: np.ndarray
    lengths: np.ndarray
    composition: np.ndarray

    @property
    def longest(self) -> int:
        return int(self.lengths.max(initial=0))


def low_complexity_residues(residues: np.ndarray, position: np.ndarray, length_of_residue: np.ndarray,
                            window: int = LOW_COMPLEXITY_WINDOW,
                            min_entropy: float = LOW_COMPLEXITY_ENTROPY) -> np.ndarray:
    """Which residues lie in a window of low sequence entropy (within one sequence)

    Parameters
    ----------
    residues : numpy.ndarray of uint8
        Residue codes of every sequence, one after another
    position : numpy.ndarray of int64
        Position of each residue in its sequence
    length_of_residue : numpy.ndarray of int64
        Length of the sequence each residue belongs to
    window : int
    min_entropy : float
        Windows with less entropy than this (in bits) are low complexity

    Returns
    -------
    numpy.ndarray of bool
        True for masked residues
    """
    n = len(residues)
    n_windows = max(n - window + 1, 0)
    starts = np.flatnonzero(position[:n_windows] + window <= length_of_residue[:n_windows])
    entropy = np.zeros(len(starts))
    for code in range(len(AMINO_ACIDS)):
        cumulative = np.concatenate(([0], np.cumsum(residues == code, dtype=np.int32)))
        fraction = (cumulative[starts + window] - cumulative[starts]) / window
        entropy -= np.where(fraction > 0, fraction * np.log2(np.where(fraction > 0, fraction, 1)), 0)
    low = starts[entropy < min_entropy]
    depth = np.cumsum(np.bincount(low, minlength=n + 1) - np.bincount(low + window, minlength=n + 1))
    return depth[:n] > 0


def kmer_table(sequences: Sequence[str], kmer_size: int, mask_low_complexity: bool) -> KmerTable:
    """Find every k-mer of every sequence (see KmerTable), ordered by sequence and position."""
    lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
    residues = _RESIDUE_CODES[np.frombuffer(''.join(sequences).encode('ascii', 'replace'), dtype=np.uint8)]
    sequence_of = np.repeat(np.arange(len(sequences), dtype=np.int64), lengths)
    starts = np.cumsum(lengths) - lengths
    position = np.arange(len(residues), dtype=np.int64) - np.repeat(starts, lengths)

    standard = residues != NOT_AN_AMINO_ACID
    residue_counts = np.bincount(sequence_of[standard] * len(AMINO_ACIDS) + residues[standard],
                                 minlength=len(sequences) * len(AMINO_ACIDS)).reshape(len(sequences), len(AMINO_ACIDS))
    composition = residue_counts / np.maximum(residue_counts.sum(axis=1, keepdims=True), 1)

    # Residues a k-mer may not contain: anything non-standard, and low-complexity sequence
    excluded = ~standard
    if mask_low_complexity:
        excluded |= low_complexity_residues(residues, position, lengths[sequence_of], LOW_COMPLEXITY_WINDOW,
                                            LOW_COMPLEXITY_ENTROPY)
    excluded_before = np.concatenate(([0], np.cumsum(excluded, dtype=np.int64)))

    n_kmers = max(len(residues) - kmer_size + 1, 0)
    ok = position[:n_kmers] + kmer_size <= lengths[sequence_of[:n_kmers]]
    ok &= excluded_before[kmer_size:kmer_size + n_kmers] == excluded_before[:n_kmers]
    codes = np.zeros(n_kmers, dtype=np.int64)
    for offset in range(kmer_size):
        codes = codes * len(AMINO_ACIDS) + residues[offset:offset + n_kmers]

    if mask_low_complexity:
        distinct = np.ones(n_kmers, dtype=np.int64)
        for later in range(1, kmer_size):
            new = np.ones(n_kmers, dtype=bool)
            for earlier in range(later):
                new &= residues[later:later + n_kmers] != residues[earlier:earlier + n_kmers]
            distinct += new
        ok &= distinct > kmer_size // 2

    kept = np.flatnonzero(ok)
    counts = np.bincount(sequence_of[kept], minlength=len(sequences)).astype(np.int64)
    return KmerTable(codes[kept], sequence_of[kept], position[kept], counts, lengths, composition)


class KmerSearch:
    """Finds similar sequences from shared k-mers (see the module documentation and homology.SimilaritySearch)

    Attributes
    ----------
    criteria : KmerCriteria
    """

    def __init__(self, criteria: KmerCriteria | None = None) -> None:
        self.criteria = criteria or KmerCriteria()

    def describe(self) -> str:
        return f'{self.criteria.describe()}; PARROT k-mer search (no MMseqs2)'

    def summarise(self, matches: Sequence[KmerMatch]) -> str:
        return (f'at E <= {self.criteria.report_max_evalue:g}; estimated '
                + identity_summary(match.identity for match in matches))

    def __call__(self, queries: Sequence[str], targets: Sequence[str] | None = None,
                 weaker: bool = False) -> list[KmerMatch]:
        criteria = self.criteria
        query_table = kmer_table(queries, criteria.kmer_size, criteria.mask_low_complexity)
        target_table = query_table if targets is None else kmer_table(targets, criteria.kmer_size,
                                                                      criteria.mask_low_complexity)
        n_targets = len(queries) if targets is None else len(targets)
        # As for a database search (MMseqs2, BLAST), E-values are per query: the targets are the database
        database_size = len(queries) - 1 if targets is None else len(targets)
        max_evalue = criteria.report_max_evalue if weaker else criteria.max_evalue

        # Index the targets' k-mers by (k-mer, target): each query k-mer's matches are then one contiguous run,
        # and when searching the queries against themselves, only targets after the query are visited
        order = np.lexsort((target_table.sequence, target_table.codes))
        index_keys = target_table.codes[order] * n_targets + target_table.sequence[order]
        index_sequence = target_table.sequence[order]
        index_position = target_table.position[order]

        query_keys = query_table.codes * n_targets
        first = np.searchsorted(index_keys, query_keys + (query_table.sequence if targets is None else 0),
                                side='right' if targets is None else 'left')
        last = np.searchsorted(index_keys, query_keys + n_targets, side='left')
        hits_per_kmer = last - first

        matches: list[KmerMatch] = []
        for batch in query_batches(query_table.sequence, hits_per_kmer):
            rows = np.arange(batch.start, batch.stop)
            repeats = hits_per_kmer[rows]
            total = int(repeats.sum())
            if total == 0:
                continue
            entries = (np.repeat(first[rows] - (np.cumsum(repeats) - repeats), repeats)
                       + np.arange(total, dtype=np.int64))
            lowest_query = int(query_table.sequence[batch.start])
            hits = Hits(pair=(np.repeat(query_table.sequence[rows], repeats) - lowest_query) * n_targets
                        + index_sequence[entries],
                        query_position=np.repeat(query_table.position[rows], repeats),
                        target_position=index_position[entries])
            del entries

            runs = find_runs(hits, criteria.band_width, target_table.longest, criteria.max_gap)
            del hits
            query_index, target_index, shared, evalue, coverage, identity = score_pairs(
                runs, query_table, target_table, n_targets, lowest_query, criteria, database_size)
            reported = evalue <= max_evalue
            linked = (evalue <= criteria.max_evalue) & (coverage >= criteria.min_coverage)
            for values in zip(query_index[reported].tolist(), target_index[reported].tolist(),
                              shared[reported].tolist(), coverage[reported].tolist(), evalue[reported].tolist(),
                              identity[reported].tolist(), linked[reported].tolist()):
                matches.append(KmerMatch(*values))
        return matches


@dataclass
class Hits:
    """Shared k-mers: one entry per pair of positions (in a query and a target) where the same k-mer occurs

    Attributes
    ----------
    pair : numpy.ndarray of int64
        Which pair of sequences
    query_position, target_position : numpy.ndarray of int64
        Where the k-mer starts in each
    """

    pair: np.ndarray
    query_position: np.ndarray
    target_position: np.ndarray


@dataclass
class Runs:
    """Runs of shared k-mers along one diagonal band (see find_runs)

    Attributes
    ----------
    pair : numpy.ndarray of int64
        Which pair of sequences
    size : numpy.ndarray of int64
        Shared k-mers in the run, counted as distinct positions (the smaller of
        the number of distinct query and distinct target positions)
    query_first, query_last : numpy.ndarray of int64
        Query positions of the first and last of them
    target_low, target_high : numpy.ndarray of int64
        Lowest and highest of their target positions
    diagonal : numpy.ndarray of int64
        Offset (query position - target position) of the run's first shared k-mer
    """

    pair: np.ndarray
    size: np.ndarray
    query_first: np.ndarray
    query_last: np.ndarray
    target_low: np.ndarray
    target_high: np.ndarray
    diagonal: np.ndarray

    @property
    def span(self) -> np.ndarray:
        """Longer of the query and target stretches covered (from first to last k-mer start)."""
        return np.maximum(self.query_last - self.query_first, self.target_high - self.target_low)


def find_runs(hits: Hits, band_width: int, longest_target: int, max_gap: int) -> Runs:
    """Every run of at least two shared k-mers (distinct positions) along a diagonal band

    Bands are band_width diagonals wide, and are laid out twice, the second time
    shifted by half a band, so that any run of shared k-mers drifting by up to
    half a band (through insertions or deletions) falls within one band. Within
    a band, shared k-mers are chained in order along the query, and a chain is
    broken wherever two consecutive ones are more than max_gap residues apart, so
    that stray chance matches elsewhere in the band do not stretch a run.
    """
    diagonal = hits.query_position - hits.target_position + longest_target
    n_bands = (diagonal.max(initial=0) + band_width) // band_width + 1
    found = []
    for shift in (0, band_width // 2):
        # Hits arrive ordered by query position within each query, so a stable sort by (pair, band) leaves
        # each band's hits in order along the query
        key = hits.pair * n_bands + (diagonal + shift) // band_width
        order = np.argsort(key, kind='stable')
        key = key[order]
        query_position = hits.query_position[order]
        target_position = hits.target_position[order]
        new_run = np.ones(len(key), dtype=bool)
        new_run[1:] = (key[1:] != key[:-1]) | (query_position[1:] - query_position[:-1] > max_gap)
        run = np.cumsum(new_run) - 1
        starts = np.flatnonzero(new_run)
        ends = np.append(starts[1:], len(key)) - 1

        # Distinct query positions (already in order) and distinct target positions in each run
        new_query = new_run.copy()
        new_query[1:] |= query_position[1:] != query_position[:-1]
        run_target = np.sort(run * (longest_target + 1) + target_position)
        new_target = np.ones(len(key), dtype=bool)
        new_target[1:] = run_target[1:] != run_target[:-1]
        size = np.minimum(np.bincount(run[new_query], minlength=len(starts)),
                          np.bincount(run_target[new_target] // (longest_target + 1), minlength=len(starts)))

        keep = size >= 2
        if not keep.any():
            continue
        starts, ends = starts[keep], ends[keep]
        found.append(Runs(pair=key[starts] // n_bands, size=size[keep], query_first=query_position[starts],
                          query_last=query_position[ends],
                          target_low=reduce_runs(np.minimum, target_position, starts, ends),
                          target_high=reduce_runs(np.maximum, target_position, starts, ends),
                          diagonal=query_position[starts] - target_position[starts]))
    names = [field.name for field in fields(Runs)]
    if not found:
        return Runs(*(np.zeros(0, dtype=np.int64) for _ in names))
    return Runs(*(np.concatenate([getattr(runs, name) for runs in found]) for name in names))


def reduce_runs(function: np.ufunc, values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """function.reduce over values[starts[i]:ends[i] + 1] for each i (runs in order, not overlapping)."""
    # reduceat over the slices [start, end + 1), interleaved with the slices between runs (discarded)
    boundaries = np.empty(2 * len(starts), dtype=np.int64)
    boundaries[0::2] = starts
    boundaries[1::2] = ends + 1
    if boundaries[-1] == len(values):
        boundaries = boundaries[:-1]  # the last run reaches the end of the array
    return function.reduceat(values, boundaries)[0::2]


def score_pairs(runs: Runs, query_table: KmerTable, target_table: KmerTable, n_targets: int, lowest_query: int,
                criteria: KmerCriteria, database_size: int) -> tuple[np.ndarray, ...]:
    """Score every pair of sequences with a run of shared k-mers (see the module documentation)

    Returns
    -------
    (query_index, target_index, shared, evalue, coverage, identity) : numpy.ndarray each
        One entry per pair: the sequences, the shared k-mers of the best run, the
        E-value, the coverage of the shorter sequence, and a rough identity estimate
    """
    k = criteria.kmer_size
    if len(runs.pair) == 0:
        empty = np.zeros(0)
        return empty.astype(np.int64), empty.astype(np.int64), empty.astype(np.int64), empty, empty, empty
    query_of = runs.pair // n_targets + lowest_query
    target_of = runs.pair % n_targets
    # Chance that a k-mer in one sequence matches a k-mer in the other, if they were unrelated
    kmer_match = np.einsum('ij,ij->i', query_table.composition[query_of], target_table.composition[target_of]) ** k
    shorter = np.minimum(query_table.lengths[query_of], target_table.lengths[target_of])

    # Each pair's best run: the most shared k-mers, then the longest span
    order = np.lexsort((-runs.span, -runs.size, runs.pair))
    best = order[np.concatenate(([True], runs.pair[order][1:] != runs.pair[order][:-1]))]
    pairs = runs.pair[best]
    query_index, target_index = query_of[best], target_of[best]
    shared = runs.size[best]

    # E-value: chance of at least this many shared k-mers in one band, times the bands and targets searched
    positions_in_band = criteria.band_width * np.minimum(query_table.counts[query_index],
                                                         target_table.counts[target_index])
    bands_per_pair = 2 * (query_table.lengths[query_index] + target_table.lengths[target_index]) / criteria.band_width
    evalue = pdtrc(shared - 1, positions_in_band * kmer_match[best]) * bands_per_pair * database_size

    # Extent: the best run, and runs near its diagonal that are unlikely by chance anywhere near it
    pair_number = np.searchsorted(pairs, runs.pair)
    near = np.abs(runs.diagonal - runs.diagonal[best][pair_number]) <= criteria.max_drift
    positions_in_stretch = criteria.band_width * (runs.query_last - runs.query_first + criteria.max_gap)
    stretches_near = 2 * (2 * criteria.max_drift / criteria.band_width + 1) * (shorter / criteria.max_gap + 1)
    near &= pdtrc(runs.size - 1, positions_in_stretch * kmer_match) * stretches_near <= RUN_CHANCE
    near[best] = True
    query_low, query_high = runs.query_first[best].copy(), runs.query_last[best].copy()
    target_low, target_high = runs.target_low[best].copy(), runs.target_high[best].copy()
    np.minimum.at(query_low, pair_number[near], runs.query_first[near])
    np.maximum.at(query_high, pair_number[near], runs.query_last[near])
    np.minimum.at(target_low, pair_number[near], runs.target_low[near])
    np.maximum.at(target_high, pair_number[near], runs.target_high[near])

    extent = np.maximum(query_high - query_low, target_high - target_low) + k
    coverage = np.minimum(extent / np.maximum(shorter[best], 1), 1.0)
    density = shared / (runs.query_last[best] - runs.query_first[best] + 1)
    identity = np.minimum(density, 1.0) ** (1 / k)
    return query_index, target_index, shared, evalue, coverage, identity


def query_batches(query_sequence: np.ndarray, hits_per_kmer: np.ndarray, limit: int | None = None) -> list[slice]:
    """Split the query k-mers (ordered by sequence) into runs of whole sequences with at most `limit` matches each.

    A single sequence with more matches than the limit gets a batch of its own.
    """
    limit = BATCH_MATCHES if limit is None else limit
    if len(query_sequence) == 0:
        return []
    # Rows where each sequence's k-mers start (and, last, the end of the table), and the matches before each
    boundaries = np.flatnonzero(np.diff(query_sequence)) + 1
    row_starts = np.concatenate(([0], boundaries, [len(query_sequence)]))
    matches_before = np.concatenate(([0], np.cumsum(hits_per_kmer)))[row_starts]

    batches = []
    start = 0
    n_sequences = len(row_starts) - 1
    while start < n_sequences:
        # The most whole sequences that keep this batch within the limit (always at least one)
        stop = int(np.searchsorted(matches_before, matches_before[start] + limit, side='right')) - 1
        stop = min(max(stop, start + 1), n_sequences)
        batches.append(slice(int(row_starts[start]), int(row_starts[stop])))
        start = stop
    return batches
