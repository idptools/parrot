"""
Finding similar protein sequences from shared k-mers, without external programs.

This is the search engine behind --split kmer (in parrot-preprocess,
parrot-train and parrot-optimize), for when MMseqs2 is not available. It works
like the seeding step of BLAST, FASTA or MMseqs2: related sequences share short
identical words (k-mers), and because they are related, the shared words line
up along one alignment diagonal (the same offset between their positions in the
two sequences). Unrelated sequences also share k-mers by chance, but scattered
over many diagonals.

For each pair of sequences, shared k-mers are collected in bands of nearby
diagonals (KmerCriteria.band_width wide, so short insertions and deletions are
tolerated) and chained into runs along the sequences (a run breaks where two
shared k-mers are more than KmerCriteria.max_gap residues apart). Each pair is
then scored in the same terms as the MMseqs2 criterion:

* E-value, from the run whose shared k-mers cover the most residues (counted
  at distinct positions, so a repeat cannot count one k-mer many times).
  Overlapping shared k-mers are parts of one longer stretch of shared sequence,
  not separate chances: a shared 6-residue stretch holds three 4-mers, but by
  chance it is as likely as one 4-mer followed by two more identical residues.
  If the two sequences were unrelated, each residue would be identical to the
  one it is aligned with with probability p, the chance that two residues drawn
  from the two sequences' amino-acid compositions are identical (so
  compositionally biased sequences, such as many disordered regions, need more
  shared sequence to count). Each pair of k-mer positions would then start a
  stretch of shared sequence with probability at most p^k, each stretch would
  extend beyond k residues while the next residue is also identical, and a run
  would continue while another chance stretch follows in its bands within
  KmerCriteria.max_gap residues (see chance_of_run). The E-value is the
  expected number of chance runs covering at least as many residues, over every
  pair of k-mer positions in the two sequences, multiplied by the number of
  target sequences: as with MMseqs2 or BLAST, the number of matches this strong
  one query would be expected to find by chance in the database searched.
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
bounded. Exact k-mer matches find close relatives reliably (in the yeast
proteome, nearly all the pairs MMseqs2 links at 50% identity or more, and three
quarters of those at 40-50%), but not the distant homologues that MMseqs2 finds
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
from scipy.special import gammaln, nbdtrc, xlogy

from parrot.sequence_encoder import STANDARD_AMINO_ACIDS
from parrot.tools.splitting import Match, identity_summary

# k-mers containing anything other than the 20 standard amino acids are ignored
AMINO_ACIDS = STANDARD_AMINO_ACIDS

# The k-mer sizes allowed: below 3, chance matches swamp real ones; above 8, close relatives are missed
MIN_KMER_SIZE = 3
MAX_KMER_SIZE = 8

# Code for a residue that is not one of the 20 standard amino acids
NOT_AN_AMINO_ACID = 255

# Low-complexity masking: residues in a window this long (as the SEG program)...
LOW_COMPLEXITY_WINDOW = 12

# ...whose sequence entropy is below this many bits are masked. SEG uses 2.2 bits, which masks half the residues
# of the ALBATROSS disordered-region dataset (and 7-9% of yeast or CheZOD proteins) and costs much of the
# k-mer search's sensitivity there; 1.8 bits masks only the most repetitive stretches (still a third of the
# residues of ALBATROSS, whose sequences are rich in repeats, but 2% of CheZOD's).
LOW_COMPLEXITY_ENTROPY = 1.8

# Largest number of k-mer matches (pairs of positions sharing a k-mer) handled at once.
# Each needs about 80 bytes of working memory, so this keeps a batch to about 0.8 GB.
BATCH_MATCHES = 10_000_000

# The chance that a run covers many residues sums over the number of chance stretches of shared sequence in it
# (see chance_of_run). Runs of more stretches than it takes to cover the residues without extending any of them
# grow less likely with every further stretch, so beyond this many more, they are bounded together...
LONGER_RUNS_SUMMED = 4

# ...as are runs of more than this many stretches
MAX_STRETCHES_SUMMED = 64

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
        relatives but take longer (a search with k = 3 takes about 20 times as
        long as with 4, and one with 5 about a fifth as long) and are more
        easily matched by chance.
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
        E-value (or up to max_evalue, if that is larger), so any remaining
        similarity between sets is visible
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
        """Check the settings are usable.

        Raises
        ------
        ValueError
            If the k-mer size is outside [MIN_KMER_SIZE, MAX_KMER_SIZE] or a band setting is too small
        """
        if not MIN_KMER_SIZE <= self.kmer_size <= MAX_KMER_SIZE:
            raise ValueError(f'The k-mer size must be between {MIN_KMER_SIZE} and {MAX_KMER_SIZE}, '
                             f'got {self.kmer_size}')
        if self.band_width < 2 or self.max_gap < 1 or self.max_drift < 0:
            raise ValueError('band_width must be at least 2, max_gap at least 1 and max_drift at least 0')

    @property
    def reported_evalue(self) -> float:
        """Largest E-value of the similarities the separation check reports: every one that meets the criterion,
        and weaker ones up to report_max_evalue."""
        return max(self.report_max_evalue, self.max_evalue)

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
    covered : int
        Residues covered by the shared k-mers of the best run (the fewer of
        those in the query and those in the target)
    shared : int
        Shared k-mers (distinct positions, overlapping or not) in the best run
    coverage : float
        Fraction of the shorter sequence spanned by the similarity
    evalue : float
        Number of matches at least this strong the query would be expected to find by chance
    identity : float
        Rough identity estimate from how densely the best run's shared k-mers
        cover its span (if a fraction f of positions start a shared k-mer,
        f ~ identity^k). A short run is all shared k-mers, so this
        overestimates the identity of short similarities.
    linked : bool
        True if the pair meets the criterion
    """

    query: int
    target: int
    covered: int
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
        """Length of the longest sequence (0 if there are none)."""
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
        Length of the windows, in residues (only windows within one sequence are checked)
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
    """Find every k-mer of every sequence (see KmerTable), ordered by sequence and position

    The sequences are handled as one long array of residue codes, so every
    step works on all of them at once.

    Parameters
    ----------
    sequences : list of str
    kmer_size : int
    mask_low_complexity : bool
        Leave out k-mers in low-complexity sequence (see the module documentation)

    Returns
    -------
    KmerTable
    """
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

    # A k-mer starts at every residue; it is usable if it lies within one sequence and contains no excluded
    # residue (no excluded residue between its start and its end, counted with the cumulative sum above)
    n_kmers = max(len(residues) - kmer_size + 1, 0)
    usable = position[:n_kmers] + kmer_size <= lengths[sequence_of[:n_kmers]]
    usable &= excluded_before[kmer_size:kmer_size + n_kmers] == excluded_before[:n_kmers]

    # Each k-mer as a number: its residue codes as the digits of a base-20 number
    codes = np.zeros(n_kmers, dtype=np.int64)
    for offset in range(kmer_size):
        codes = codes * len(AMINO_ACIDS) + residues[offset:offset + n_kmers]

    if mask_low_complexity:
        # Count the distinct amino acids in each k-mer: a residue adds one if it differs from every earlier one
        distinct = np.ones(n_kmers, dtype=np.int64)
        for later in range(1, kmer_size):
            differs_from_all_earlier = np.ones(n_kmers, dtype=bool)
            for earlier in range(later):
                differs_from_all_earlier &= residues[later:later + n_kmers] != residues[earlier:earlier + n_kmers]
            distinct += differs_from_all_earlier
        usable &= distinct > kmer_size // 2

    kept = np.flatnonzero(usable)
    counts = np.bincount(sequence_of[kept], minlength=len(sequences)).astype(np.int64)
    return KmerTable(codes[kept], sequence_of[kept], position[kept], counts, lengths, composition)


class KmerSearch:
    """Finds similar sequences from shared k-mers (see the module documentation and splitting.SimilaritySearch)

    Attributes
    ----------
    criteria : KmerCriteria
    """

    def __init__(self, criteria: KmerCriteria | None = None) -> None:
        """
        Parameters
        ----------
        criteria : KmerCriteria, optional
            When two sequences count as similar (the defaults if not given)
        """
        self.criteria = criteria or KmerCriteria()

    def describe(self) -> str:
        """The criterion, in words."""
        return f'{self.criteria.describe()}; PARROT k-mer search (no MMseqs2)'

    def summarise(self, matches: Sequence[Match]) -> str:
        """A short description of the strength of the matches found (for the report)."""
        return (f'at E <= {self.criteria.reported_evalue:g}; estimated '
                + identity_summary(match.identity for match in matches))

    def __call__(self, queries: Sequence[str], targets: Sequence[str] | None = None,
                 weaker: bool = False) -> list[KmerMatch]:
        """Search queries against targets (or, with targets=None, against each other, each pair once)

        Parameters
        ----------
        queries : list of str
        targets : list of str, optional
            None to search the queries against themselves, without self-hits
        weaker : bool, optional
            Also return matches up to report_max_evalue that do not meet the
            criterion (linked=False), for the report (see KmerCriteria.reported_evalue)

        Returns
        -------
        list of KmerMatch
            Every pair with an E-value of at most max_evalue (with weaker, of at
            most KmerCriteria.reported_evalue); linked is True for those that
            meet the criterion
        """
        criteria = self.criteria
        self_search = targets is None
        query_table = kmer_table(queries, criteria.kmer_size, criteria.mask_low_complexity)

        # As for a database search (MMseqs2, BLAST), E-values are per query: the targets are the database
        if targets is None:
            target_table, n_targets, database_size = query_table, len(queries), len(queries) - 1
        else:
            target_table = kmer_table(targets, criteria.kmer_size, criteria.mask_low_complexity)
            n_targets, database_size = len(targets), len(targets)
        max_evalue = criteria.reported_evalue if weaker else criteria.max_evalue

        index = TargetIndex.build(target_table, n_targets)
        first, last = index.match_ranges(query_table, self_search)
        hits_per_kmer = last - first

        matches: list[KmerMatch] = []
        for batch in query_batches(query_table.sequence, hits_per_kmer):
            hits = index.hits(query_table, batch, first, hits_per_kmer)
            if hits is None:
                continue
            lowest_query = int(query_table.sequence[batch.start])
            runs = find_runs(hits, criteria.band_width, target_table.longest, criteria.max_gap, criteria.kmer_size)
            del hits
            scores = score_pairs(runs, query_table, target_table, n_targets, lowest_query, criteria, database_size)
            matches += scores.matches(max_evalue, criteria)
        return matches


@dataclass
class TargetIndex:
    """The targets' k-mers, sorted so that every match of one query k-mer is a contiguous range

    Each target k-mer gets the key (k-mer code) x n_targets + (target number):
    sorting by key sorts by k-mer and then by target, so the matches of a k-mer
    are the keys from code x n_targets up to (but not including)
    (code + 1) x n_targets, in target order.

    Attributes
    ----------
    keys : numpy.ndarray of int64
        Sorted keys
    sequence : numpy.ndarray of int64
        The target each key belongs to
    position : numpy.ndarray of int64
        Where in that target the k-mer starts
    n_targets : int
        Number of target sequences
    """

    keys: np.ndarray
    sequence: np.ndarray
    position: np.ndarray
    n_targets: int

    @classmethod
    def build(cls, target_table: KmerTable, n_targets: int) -> 'TargetIndex':
        """Index the k-mers of a table of target sequences.

        Parameters
        ----------
        target_table : KmerTable
        n_targets : int
            Number of target sequences

        Returns
        -------
        TargetIndex
        """
        order = np.lexsort((target_table.sequence, target_table.codes))
        return cls(keys=target_table.codes[order] * n_targets + target_table.sequence[order],
                   sequence=target_table.sequence[order], position=target_table.position[order], n_targets=n_targets)

    def match_ranges(self, query_table: KmerTable, self_search: bool) -> tuple[np.ndarray, np.ndarray]:
        """For every query k-mer, the range of index entries holding the same k-mer

        When the queries are searched against themselves, a query k-mer's range
        starts just after its own sequence (the first key greater than
        code x n_targets + query number), so each pair of sequences is compared
        once and no sequence is compared with itself.

        Parameters
        ----------
        query_table : KmerTable
        self_search : bool
            True if the index holds the queries themselves

        Returns
        -------
        numpy.ndarray of int64
            First entry of each query k-mer's range
        numpy.ndarray of int64
            One past its last entry
        """
        lowest_keys = query_table.codes * self.n_targets
        if self_search:
            first = np.searchsorted(self.keys, lowest_keys + query_table.sequence, side='right')
        else:
            first = np.searchsorted(self.keys, lowest_keys, side='left')
        last = np.searchsorted(self.keys, lowest_keys + self.n_targets, side='left')
        return first, last

    def hits(self, query_table: KmerTable, batch: slice, first: np.ndarray, hits_per_kmer: np.ndarray) -> 'Hits | None':
        """Every shared k-mer of one batch of query k-mers (see query_batches), or None if there are none

        Parameters
        ----------
        query_table : KmerTable
        batch : slice
            Rows of the query table (whole sequences)
        first : numpy.ndarray of int64
            First index entry matching each query k-mer (see match_ranges())
        hits_per_kmer : numpy.ndarray of int64
            Number of index entries matching each query k-mer

        Returns
        -------
        Hits or None
            Pairs are numbered relative to the batch's first query:
            (query - first query of the batch) x n_targets + target
        """
        rows = np.arange(batch.start, batch.stop)
        repeats = hits_per_kmer[rows]
        total = int(repeats.sum())
        if total == 0:
            return None

        # The index entries matching each query k-mer, one after another: row r contributes
        # first[r], first[r] + 1, ..., first[r] + repeats[r] - 1
        entries = (np.repeat(first[rows] - (np.cumsum(repeats) - repeats), repeats)
                   + np.arange(total, dtype=np.int64))
        lowest_query = int(query_table.sequence[batch.start])
        query_number = np.repeat(query_table.sequence[rows], repeats) - lowest_query
        return Hits(pair=query_number * self.n_targets + self.sequence[entries],
                    query_position=np.repeat(query_table.position[rows], repeats),
                    target_position=self.position[entries])


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
    covered : numpy.ndarray of int64
        Residues covered by the run's shared k-mers (the smaller of the number
        covered in the query and in the target)
    query_first, query_last : numpy.ndarray of int64
        Query positions of the first and last of them
    target_low, target_high : numpy.ndarray of int64
        Lowest and highest of their target positions
    diagonal : numpy.ndarray of int64
        Offset (query position - target position) of the run's first shared k-mer
    """

    pair: np.ndarray
    size: np.ndarray
    covered: np.ndarray
    query_first: np.ndarray
    query_last: np.ndarray
    target_low: np.ndarray
    target_high: np.ndarray
    diagonal: np.ndarray

    @property
    def span(self) -> np.ndarray:
        """Longer of the query and target stretches covered (from first to last k-mer start)."""
        return np.maximum(self.query_last - self.query_first, self.target_high - self.target_low)


def find_runs(hits: Hits, band_width: int, longest_target: int, max_gap: int, kmer_size: int) -> Runs:
    """Every run of at least two shared k-mers (distinct positions) along a diagonal band

    Bands are band_width diagonals wide, and are laid out twice, the second time
    shifted by half a band, so that any run of shared k-mers drifting by up to
    half a band (through insertions or deletions) falls within one band. Within
    a band, shared k-mers are chained in order along the query, and a chain is
    broken wherever two consecutive ones are more than max_gap residues apart, so
    that stray chance matches elsewhere in the band do not stretch a run.

    Parameters
    ----------
    hits : Hits
        Every shared k-mer
    band_width : int
        Width of each band, in diagonals
    longest_target : int
        Length of the longest target (used to keep diagonal numbers positive)
    max_gap : int
        Largest gap, in query residues, between consecutive k-mers of a run
    kmer_size : int
        Length of the k-mers (to count the residues they cover)

    Returns
    -------
    Runs
        Those of the first layout, then those of the second (a run that lies
        within a band of each layout is found twice)
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
        target_run = run_target[new_target] // (longest_target + 1)
        size = np.minimum(np.bincount(run[new_query], minlength=len(starts)),
                          np.bincount(target_run, minlength=len(starts)))
        covered = np.minimum(
            covered_residues(run[new_query], query_position[new_query], kmer_size, len(starts)),
            covered_residues(target_run, run_target[new_target] % (longest_target + 1), kmer_size, len(starts)))

        keep = size >= 2
        if not keep.any():
            continue
        starts, ends = starts[keep], ends[keep]
        found.append(Runs(pair=key[starts] // n_bands, size=size[keep], covered=covered[keep],
                          query_first=query_position[starts],
                          query_last=query_position[ends],
                          target_low=reduce_runs(np.minimum, target_position, starts, ends),
                          target_high=reduce_runs(np.maximum, target_position, starts, ends),
                          diagonal=query_position[starts] - target_position[starts]))
    names = [field.name for field in fields(Runs)]
    if not found:
        return Runs(*(np.zeros(0, dtype=np.int64) for _ in names))
    return Runs(*(np.concatenate([getattr(runs, name) for runs in found]) for name in names))


def covered_residues(run: np.ndarray, position: np.ndarray, kmer_size: int, n_runs: int) -> np.ndarray:
    """Residues covered by each run's shared k-mers

    Shared k-mers starting fewer than k residues apart overlap, and together
    form one stretch of shared sequence, from the first one's start to the end
    of the last one.

    Parameters
    ----------
    run : numpy.ndarray of int64
        The run each shared k-mer belongs to, in order of run
    position : numpy.ndarray of int64
        Where each shared k-mer starts (distinct and increasing within a run)
    kmer_size : int
    n_runs : int
        Number of runs

    Returns
    -------
    numpy.ndarray of int64
        One count per run
    """
    if len(run) == 0:
        return np.zeros(n_runs, dtype=np.int64)
    # A new stretch starts at the first k-mer of each run and wherever a k-mer does not overlap the one before
    new_stretch = np.ones(len(run), dtype=bool)
    new_stretch[1:] = (run[1:] != run[:-1]) | (position[1:] - position[:-1] >= kmer_size)
    first = np.flatnonzero(new_stretch)
    last = np.append(first[1:], len(run)) - 1
    stretch_length = position[last] - position[first] + kmer_size
    covered: np.ndarray = np.bincount(run[first], weights=stretch_length, minlength=n_runs).astype(np.int64)
    return covered


def reduce_runs(function: np.ufunc, values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """function.reduce over values[starts[i]:ends[i] + 1] for each i (runs in order, not overlapping)

    Parameters
    ----------
    function : numpy ufunc
        e.g. np.minimum
    values : numpy.ndarray
    starts, ends : numpy.ndarray of int64
        First and last index of each run

    Returns
    -------
    numpy.ndarray
        One value per run
    """
    # reduceat over the slices [start, end + 1), interleaved with the slices between runs (discarded)
    boundaries = np.empty(2 * len(starts), dtype=np.int64)
    boundaries[0::2] = starts
    boundaries[1::2] = ends + 1
    if boundaries[-1] == len(values):
        boundaries = boundaries[:-1]  # the last run reaches the end of the array
    return function.reduceat(values, boundaries)[0::2]


@dataclass
class PairScores:
    """The score of every pair of sequences with a run of shared k-mers (see score_pairs)

    Attributes
    ----------
    query_index, target_index : numpy.ndarray of int64
        The two sequences
    covered : numpy.ndarray of int64
        Residues covered by the shared k-mers of the best run
    shared : numpy.ndarray of int64
        Shared k-mers in the best run
    evalue : numpy.ndarray of float
        E-value
    coverage : numpy.ndarray of float
        Fraction of the shorter sequence spanned by the similarity
    identity : numpy.ndarray of float
        Rough identity estimate
    """

    query_index: np.ndarray
    target_index: np.ndarray
    covered: np.ndarray
    shared: np.ndarray
    evalue: np.ndarray
    coverage: np.ndarray
    identity: np.ndarray

    def matches(self, max_evalue: float, criteria: KmerCriteria) -> list[KmerMatch]:
        """The pairs with an E-value of at most max_evalue, as KmerMatch objects

        Parameters
        ----------
        max_evalue : float
            Largest E-value reported
        criteria : KmerCriteria
            Decides which matches are linked

        Returns
        -------
        list of KmerMatch
        """
        reported = self.evalue <= max_evalue
        linked = (self.evalue <= criteria.max_evalue) & (self.coverage >= criteria.min_coverage)
        columns = [self.query_index, self.target_index, self.covered, self.shared, self.coverage, self.evalue,
                   self.identity, linked]
        return [KmerMatch(*values) for values in zip(*(column[reported].tolist() for column in columns))]


def score_pairs(runs: Runs, query_table: KmerTable, target_table: KmerTable, n_targets: int, lowest_query: int,
                criteria: KmerCriteria, database_size: int) -> PairScores:
    """Score every pair of sequences with a run of shared k-mers (see the module documentation)

    Parameters
    ----------
    runs : Runs
        Runs of shared k-mers of one batch of queries (see find_runs)
    query_table, target_table : KmerTable
    n_targets : int
        Number of target sequences (pairs are numbered query x n_targets + target)
    lowest_query : int
        The batch's first query (pair numbers are relative to it)
    criteria : KmerCriteria
    database_size : int
        Number of sequences each query was searched against

    Returns
    -------
    PairScores
    """
    if len(runs.pair) == 0:
        no_pairs = np.zeros(0)
        no_counts = no_pairs.astype(np.int64)
        return PairScores(no_counts, no_counts, no_counts, no_counts, no_pairs, no_pairs, no_pairs)

    query_of = runs.pair // n_targets + lowest_query
    target_of = runs.pair % n_targets
    # Chance that two residues are identical if the sequences were unrelated: the chance that two residues drawn
    # from the two compositions are identical (the dot product)
    residue_match = np.einsum('ij,ij->i', query_table.composition[query_of], target_table.composition[target_of])

    best = best_runs(runs)
    query_index, target_index = query_of[best], target_of[best]
    covered, shared = runs.covered[best], runs.size[best]
    evalue = pair_evalues(covered, residue_match[best], query_index, target_index, query_table, target_table,
                          criteria, database_size)
    shorter = np.minimum(query_table.lengths[query_of], target_table.lengths[target_of])
    coverage = pair_coverage(runs, best, residue_match, shorter, criteria)

    # If a fraction f of a run's positions start a shared k-mer, f ~ identity^k
    density = shared / (runs.query_last[best] - runs.query_first[best] + 1)
    identity = np.minimum(density, 1.0) ** (1 / criteria.kmer_size)
    return PairScores(query_index, target_index, covered, shared, evalue, coverage, identity)


def best_runs(runs: Runs) -> np.ndarray:
    """Each pair's best run: the one whose shared k-mers cover the most residues, then the one with the most shared
    k-mers, then the longest span

    Parameters
    ----------
    runs : Runs

    Returns
    -------
    numpy.ndarray of int64
        Index of each pair's best run, in order of pair number
    """
    order = np.lexsort((-runs.span, -runs.size, -runs.covered, runs.pair))
    first_of_pair = np.concatenate(([True], runs.pair[order][1:] != runs.pair[order][:-1]))
    best: np.ndarray = order[first_of_pair]
    return best


def pair_evalues(covered: np.ndarray, residue_match: np.ndarray, query_index: np.ndarray, target_index: np.ndarray,
                 query_table: KmerTable, target_table: KmerTable, criteria: KmerCriteria,
                 database_size: int) -> np.ndarray:
    """E-value of each pair's best run: the expected number of chance runs covering at least as many residues,
    over every pair of k-mer positions in the two sequences, times the number of targets searched

    Parameters
    ----------
    covered : numpy.ndarray
        Residues covered by the shared k-mers of each pair's best run
    residue_match : numpy.ndarray
        Chance that two residues of the pair are identical by chance
    query_index, target_index : numpy.ndarray
        The pair's sequences
    query_table, target_table : KmerTable
    criteria : KmerCriteria
    database_size : int
        Number of sequences each query was searched against

    Returns
    -------
    numpy.ndarray of float
    """
    # Every pair of k-mer positions could start a chance stretch of shared sequence, and so a chance run
    position_pairs = query_table.counts[query_index] * target_table.counts[target_index]
    expected_runs = position_pairs * residue_match ** criteria.kmer_size
    shorter = np.minimum(query_table.lengths[query_index], target_table.lengths[target_index])
    evalue: np.ndarray = expected_runs * chance_of_run(covered, residue_match, shorter, criteria) * database_size
    return evalue


def chance_of_run(covered: np.ndarray, residue_match: np.ndarray, shorter: np.ndarray,
                  criteria: KmerCriteria) -> np.ndarray:
    """Chance that a run of shared k-mers starting with a chance stretch of shared sequence covers at least
    `covered` residues

    If two sequences were unrelated, a stretch of k identical residues would
    extend while the next residue is also identical (probability p, the
    residue_match), and the run would continue while another chance stretch
    starts in one of the stretch's bands within max_gap residues along the query.
    The two layouts of bands (see find_runs) hold a stretch's diagonal in two
    bands that together span 1.5 band_width diagonals, so r = 1.5 band_width p^k
    chance stretches start per query residue, and a run found in either layout
    continues with probability at most q = 1 - exp(-r max_gap). A run of n
    stretches covers n k residues plus a negative binomial number of extra
    residues (the identical residues before n differing ones), so

        P(covered >= c) = sum over n of P(n stretches) P(n k + extra >= c)

    where P(n stretches) is at most q^(n - 1) (1 - q) (or q^(n - 1) for the
    most stretches that fit in the shorter sequence, which takes the chance of
    every longer run, as those cannot happen), and also at most the
    expected number of ways the n - 1 further stretches could be placed:
    r^(n - 1) times the number of ways to share out the residues the shorter
    sequence has to spare (n stretches covering c residues, with a differing
    residue between each two, span at least max(c, n k) + n - 1 residues). The
    second bound matters for short sequences, where few runs fit. Runs of more
    than ceil(c / k) + LONGER_RUNS_SUMMED stretches, or of more than
    MAX_STRETCHES_SUMMED, are bounded together: the chance of more than m
    stretches is at most q^m. Every shortcut here (counting k-mer positions
    rather than stretch starts, the smaller of the query's and the target's
    covered residues, and extra residues that would not fit) can only overstate
    the chance.

    Parameters
    ----------
    covered : numpy.ndarray of int64
        Residues covered (at least k)
    residue_match : numpy.ndarray of float
        Chance that two aligned residues are identical, below 1
    shorter : numpy.ndarray of int64
        Length of the shorter of the two sequences
    criteria : KmerCriteria

    Returns
    -------
    numpy.ndarray of float
        One probability per entry
    """
    kmer_size = criteria.kmer_size
    stretch_rate = (criteria.band_width + criteria.band_width // 2) * residue_match ** kmer_size
    continues = -np.expm1(-stretch_rate * criteria.max_gap)

    # Most stretches a run can hold and still fit in the shorter sequence (at least the one it starts with)
    most_stretches = np.maximum(np.minimum(shorter - covered + 1, (shorter + 1) // (kmer_size + 1)), 1)
    # Runs of up to this many stretches are summed one by one, and longer ones that fit are bounded together
    enough_stretches = -(-covered // kmer_size)
    summed_stretches = np.minimum(np.minimum(most_stretches, enough_stretches + LONGER_RUNS_SUMMED),
                                  MAX_STRETCHES_SUMMED)
    chance = np.where(most_stretches > summed_stretches, continues ** summed_stretches, 0.0)
    for stretches in range(1, int(summed_stretches.max(initial=0)) + 1):
        fits = np.flatnonzero(stretches <= summed_stretches)
        enough = covered[fits]
        spare = shorter[fits] - (np.maximum(enough, stretches * kmer_size) + stretches - 1)
        # The most stretches that fit also take the chance that the run would have gone on (it cannot)
        last = stretches == most_stretches[fits]
        chain = continues[fits] ** (stretches - 1) * np.where(last, 1.0, 1 - continues[fits])
        # r^(n - 1) x (spare + n - 1 choose n - 1), the ways to place the further stretches
        placements = np.exp(xlogy(stretches - 1, stretch_rate[fits]) + gammaln(spare + stretches)
                            - gammaln(stretches) - gammaln(spare + 1))
        # nbdtrc(m, n, 1 - p): chance of more than m identical residues before n differing ones (each residue
        # differing with chance 1 - p)
        extra_needed = enough - stretches * kmer_size
        enough_extra = np.where(extra_needed > 0,
                                nbdtrc(np.maximum(extra_needed - 1, 0), stretches, 1 - residue_match[fits]), 1.0)
        chance[fits] += np.minimum(chain, placements) * enough_extra
    return chance


def pair_coverage(runs: Runs, best: np.ndarray, residue_match: np.ndarray, shorter: np.ndarray,
                  criteria: KmerCriteria) -> np.ndarray:
    """Fraction of the shorter sequence spanned by each pair's best run and the runs near its diagonal

    A run near the best one's diagonal (within max_drift) counts only if runs
    at least as strong would arise by chance anywhere near that diagonal with a
    probability of at most RUN_CHANCE (see chance_of_run).

    Parameters
    ----------
    runs : Runs
    best : numpy.ndarray
        Index of each pair's best run (see best_runs)
    residue_match : numpy.ndarray
        For each run, the chance that two residues of its pair are identical by chance
    shorter : numpy.ndarray
        For each run, the length of the shorter sequence of its pair
    criteria : KmerCriteria

    Returns
    -------
    numpy.ndarray of float
        One coverage per pair, at most 1
    """
    # Which pair each run belongs to, as a position in `best`
    pair_number = np.searchsorted(runs.pair[best], runs.pair)

    near = np.abs(runs.diagonal - runs.diagonal[best][pair_number]) <= criteria.max_drift
    # Chance runs could start at any pair of positions on the diagonals near the best run (and on those just
    # beyond, which share its bands)
    candidates = np.flatnonzero(near)
    diagonals_near = 2 * criteria.max_drift + criteria.band_width
    expected_runs = diagonals_near * shorter[candidates] * residue_match[candidates] ** criteria.kmer_size
    chance = chance_of_run(runs.covered[candidates], residue_match[candidates], shorter[candidates], criteria)
    near[candidates] = expected_runs * chance <= RUN_CHANCE
    near[best] = True

    # The extent of every counted run of each pair, in the query and in the target
    query_low, query_high = runs.query_first[best].copy(), runs.query_last[best].copy()
    target_low, target_high = runs.target_low[best].copy(), runs.target_high[best].copy()
    np.minimum.at(query_low, pair_number[near], runs.query_first[near])
    np.maximum.at(query_high, pair_number[near], runs.query_last[near])
    np.minimum.at(target_low, pair_number[near], runs.target_low[near])
    np.maximum.at(target_high, pair_number[near], runs.target_high[near])

    # Spans run from the first to the last k-mer start, so add one k-mer's length
    extent = np.maximum(query_high - query_low, target_high - target_low) + criteria.kmer_size
    coverage: np.ndarray = np.minimum(extent / np.maximum(shorter[best], 1), 1.0)
    return coverage


def query_batches(query_sequence: np.ndarray, hits_per_kmer: np.ndarray, limit: int | None = None) -> list[slice]:
    """Split the query k-mers (ordered by sequence) into runs of whole sequences with at most `limit` matches each

    A single sequence with more matches than the limit gets a batch of its own.

    Parameters
    ----------
    query_sequence : numpy.ndarray of int64
        The sequence each query k-mer belongs to (in order)
    hits_per_kmer : numpy.ndarray of int64
        Number of target k-mers each query k-mer matches
    limit : int, optional
        Most matches per batch (default BATCH_MATCHES)

    Returns
    -------
    list of slice
        Rows of the query k-mer table in each batch
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
