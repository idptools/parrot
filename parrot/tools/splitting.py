"""
Splitting a dataset into training, validation and test sets without leakage.

A test set only measures how well a network generalises if none of its
sequences has a close relative in the training set; otherwise a network can
score well by recognising relatives rather than by having learnt anything
transferable. This module splits a dataset so that no sequence in one set is
similar to a sequence in another. Similarity is found by a search engine:
MMseqs2 (Steinegger & Söding 2017, Nat Biotechnol 35:1026; mmseqs_search.py),
which must be installed separately, or PARROT's own k-mer search
(kmer_similarity.py), which needs nothing extra but only finds closer relatives.
Any object with the SimilaritySearch interface can be used.

The protocol (the one used to build PARROT's benchmark datasets) is deliberately
very stringent:

1. Similarity criterion. With MMseqs2, two sequences are homologous if they are
   identical, or if any alignment between them has an E-value of at most
   HomologyCriteria.max_evalue (1e-3) and covers at least
   HomologyCriteria.min_coverage (30%) of the shorter sequence. No
   sequence-identity threshold is used, so any statistically significant
   alignment counts. Searches use MMseqs2's most sensitive setting by default.
2. Clustering. An all-against-all search links similar records, which are
   clustered around representatives, largest first, as in CD-HIT. Records that
   must stay together (the same --groups entry, or identical sequences) are
   always in the same cluster.
3. Assignment. Clusters are shared out among the sets, largest first, each to the
   set furthest below its target fraction.
4. Separation check. Each set is searched against the others (test against
   training, test against validation, validation against training), and any
   record in the lower-priority set (training < validation < test) similar
   to the higher-priority set is removed together with its group. Because
   E-values depend on the size of the database searched, which removing
   records shrinks, the search is repeated until a round finds nothing.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# The three sets, in the order they are written to a datafile (and to a split-file)
SET_NAMES = ('train', 'val', 'test')

# Names used in reports
SET_LABELS = {'train': 'training', 'val': 'validation', 'test': 'test'}

# The sets have a priority, test > validation > training: when records in two sets are similar, the record in
# the lower-priority set is removed. These are the pairs of sets searched against each other, as
# (higher-priority set, lower-priority set), in the order searched: the validation records to be removed are
# known before the validation set is searched against the training set (see cross_set_matches).
SEPARATED_SET_PAIRS = (('test', 'train'), ('test', 'val'), ('val', 'train'))

# Search the sets against each other at most this many times (see enforce_separation)
MAX_SEPARATION_ROUNDS = 5

# Identity bins for reports: (lowest identity, label), highest first
IDENTITY_BINS = ((0.9, '>=90%'), (0.5, '50-90%'), (0.3, '30-50%'), (0.0, '<30%'))


class Match(Protocol):
    """One similarity found by a search: indices into the query and target lists, its identity, and whether it
    links them (meets the search's criterion)."""

    @property
    def query(self) -> int: ...

    @property
    def target(self) -> int: ...

    @property
    def identity(self) -> float: ...

    @property
    def linked(self) -> bool: ...


class SimilaritySearch(Protocol):
    """A search engine for splitting: finds similar sequences, and describes what it does for the report."""

    def __call__(self, queries: Sequence[str], targets: Sequence[str] | None = None,
                 weaker: bool = False) -> Sequence[Match]:
        """Search queries against targets (None: against themselves, without self-hits; a pair may be found both
        ways round).

        Every match that meets the criterion has linked=True; with weaker=True,
        weaker similarities are returned as well (linked=False), for the report.
        """

    def describe(self) -> str:
        """The criterion, in words."""

    def summarise(self, matches: Sequence[Match]) -> str:
        """A short description of the strength of the matches found (for the report)."""


@dataclass
class SplitItem:
    """One record to be assigned to a set

    Attributes
    ----------
    index : int
        The record's position in the dataset
    sequence : str
    group : str
        Records with the same group are always placed in the same set
    split : str or None
        'train', 'val' or 'test' once assigned (or the name of another set given
        to assign_clusters, such as a cross-validation fold)
    """

    index: int
    sequence: str
    group: str
    split: str | None = None


def identity_summary(identities: Iterable[float]) -> str:
    """Count similarities by sequence identity, for reports (see IDENTITY_BINS).

    Parameters
    ----------
    identities : iterable of float
        Identity of each similarity, from 0 to 1

    Returns
    -------
    str
        e.g. 'identity >=90%: 3, 50-90%: 10, 30-50%: 0, <30%: 1'
    """
    counts = Counter(next(label for threshold, label in IDENTITY_BINS if identity >= threshold)
                     for identity in identities)
    return 'identity ' + ', '.join(f'{label}: {counts[label]:,}' for _, label in IDENTITY_BINS)


# .................................................................
# Clustering

def representative_clusters(keys: Sequence[str], links: Iterable[tuple[str, str]],
                            sizes: dict[str, int]) -> dict[str, int]:
    """Cluster keys around representatives, without chaining

    Keys are visited from largest to smallest (ties broken by name). Each key
    not yet clustered becomes a representative, and every unclustered key
    directly linked to it joins its cluster. Every member of a cluster is
    therefore a direct relative of its representative; unlike connected
    components, a chain of weak links cannot pull unrelated keys together. This
    is the scheme used by CD-HIT (Li & Godzik 2006, Bioinformatics 22:1658),
    with keys ordered by size rather than by sequence length. MMseqs2's default
    clustering (greedy set cover) also clusters around representatives without
    chaining, but picks the key with the most links first.

    Parameters
    ----------
    keys : list of str
        Every key
    links : iterable of (str, str)
        Pairs of directly related keys
    sizes : dict
        Size of each key (e.g. its number of records); larger keys become
        representatives first

    Returns
    -------
    dict
        Maps each key to a cluster number (0, 1, 2, ...)
    """
    neighbours: dict[str, set[str]] = defaultdict(set)
    for first, second in links:
        if first != second:
            neighbours[first].add(second)
            neighbours[second].add(first)

    cluster_of: dict[str, int] = {}
    num_clusters = 0
    for representative in sorted(keys, key=lambda key: (-sizes.get(key, 0), key)):
        if representative in cluster_of:
            continue
        cluster_of[representative] = num_clusters
        for neighbour in sorted(neighbours[representative]):
            cluster_of.setdefault(neighbour, num_clusters)
        num_clusters += 1
    return cluster_of


def identical_sequence_links(items: Sequence[SplitItem]) -> list[tuple[str, str]]:
    """Link the groups of identical sequences (a search can miss them, e.g. when low-complexity regions are masked)."""
    groups_by_sequence: dict[str, set[str]] = defaultdict(set)
    for item in items:
        groups_by_sequence[item.sequence].add(item.group)

    links: list[tuple[str, str]] = []
    for groups in groups_by_sequence.values():
        ordered = sorted(groups)
        links.extend((ordered[0], other) for other in ordered[1:])
    return links


def merge_groups(items: Sequence[SplitItem], links: Iterable[tuple[str, str]]) -> None:
    """Give linked groups one shared group name, in place (connected components: groups must never be split)."""
    parent: dict[str, str] = {item.group: item.group for item in items}

    def root(group: str) -> str:
        """The group's root (its merged name), shortening the path to it on the way."""
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for first, second in links:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)
    for item in items:
        item.group = root(item.group)


def similarity_clusters(items: Sequence[SplitItem], search: SimilaritySearch,
                        report: Callable[[str], None]) -> dict[str, int]:
    """Cluster groups so that similar records usually end up in the same set

    Clustering only keeps similar records together so that little data is lost
    later; the guarantee that no two sets share similar records comes from
    enforce_separation().

    Parameters
    ----------
    items : list of SplitItem
        The records (records in one group always end up in one cluster)
    search : SimilaritySearch
        Finds similar sequences (searched all against all)
    report : callable
        Receives report lines

    Returns
    -------
    dict
        Maps each group to a cluster number
    """
    matches = search([item.sequence for item in items])
    # A search may find a pair both ways round (MMseqs2 does); each pair counts once
    similar_pairs = {(min(match.query, match.target), max(match.query, match.target))
                     for match in matches if match.linked}
    links = [(items[first].group, items[second].group) for first, second in sorted(similar_pairs)]

    records_per_group = Counter(item.group for item in items)
    clusters = representative_clusters(sorted(records_per_group), links, records_per_group)

    records_per_cluster = sorted(Counter(clusters[item.group] for item in items).values(), reverse=True)
    report(f'Searched {len(items):,} sequences all against all: {len(links):,} similar pairs, of which '
           f'{sum(first != second for first, second in links):,} link two groups')
    report(f'{len(records_per_group):,} groups form {len(records_per_cluster):,} clusters; the largest holds '
           f'{records_per_cluster[0]:,} records ({100 * records_per_cluster[0] / len(items):.1f}%), and '
           f'{sum(size == 1 for size in records_per_cluster):,} records are on their own')
    return clusters


# .................................................................
# Assigning sets

def assign_clusters(items: Sequence[SplitItem], clusters: dict[str, int], fractions: dict[str, float],
                    rng: np.random.Generator, report: Callable[[str], None]) -> None:
    """Share clusters out among the sets, largest first, each to the set furthest below its target (in place)

    Parameters
    ----------
    items : list of SplitItem
    clusters : dict
        Cluster of each group
    fractions : dict
        Target fraction of the records for each set, e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15} (any names
        can be used, e.g. for cross-validation folds)
    rng : numpy.random.Generator
        Decides the order in which clusters of the same size are assigned
    report : callable
        Receives report lines

    Raises
    ------
    ValueError
        If the fractions do not sum to 1
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f'Set fractions must sum to 1, got {fractions}')

    members: dict[int, list[SplitItem]] = defaultdict(list)
    for item in items:
        members[clusters[item.group]].append(item)

    cluster_members = list(members.values())
    order = rng.permutation(len(cluster_members))
    cluster_members = sorted((cluster_members[i] for i in order), key=len, reverse=True)

    total = len(items)
    counts = {name: 0 for name in fractions}
    for cluster in cluster_members:
        name = max(fractions, key=lambda split: fractions[split] * total - counts[split])
        for item in cluster:
            item.split = name
        counts[name] += len(cluster)

    report('Assigned clusters to sets: ' + ', '.join(f'{SET_LABELS.get(name, name)} {counts[name]:,} '
                                                     f'({100 * counts[name] / total:.1f}%)' for name in fractions))


def random_split(items: Sequence[SplitItem], fractions: dict[str, float], rng: np.random.Generator,
                 report: Callable[[str], None]) -> list[SplitItem]:
    """Assign groups to sets at random (identical sequences are kept in one group, so never split)

    Parameters
    ----------
    items : list of SplitItem
        The records (their groups and sets are modified in place)
    fractions : dict
        Target fraction of the records for each set
    rng : numpy.random.Generator
        Decides the order in which groups of the same size are assigned
    report : callable
        Receives report lines

    Returns
    -------
    list of SplitItem
        Every item, with its set

    Raises
    ------
    ValueError
        If the fractions do not sum to 1
    """
    merge_groups(items, identical_sequence_links(items))
    clusters = {group: number for number, group in enumerate(sorted({item.group for item in items}))}
    assign_clusters(items, clusters, fractions, rng, report)
    return list(items)


def enforce_separation(items: Sequence[SplitItem], search: SimilaritySearch,
                       report: Callable[[str], None]) -> list[SplitItem]:
    """Search each set against the others, removing records similar to a higher-priority set, until none remain

    Parameters
    ----------
    items : list of SplitItem
        The records, each assigned to 'train', 'val' or 'test'
    search : SimilaritySearch
        Finds similar sequences
    report : callable
        Receives report lines

    Returns
    -------
    list of SplitItem
        The items kept

    Raises
    ------
    RuntimeError
        If the sets are still not separated after MAX_SEPARATION_ROUNDS rounds
    """
    kept = list(items)
    for round_number in range(1, MAX_SEPARATION_ROUNDS + 1):
        offending = cross_set_matches(kept, search, report)
        if not offending:
            report('PASSED: no sequence in one set is identical or similar (by the criterion above) to a sequence '
                   'in another set')
            return kept

        before = len(kept)
        kept = [item for item in kept if item.group not in offending]
        report(f'Round {round_number}: removed {before - len(kept):,} records ({len(offending):,} groups) similar '
               'to a higher-priority set; searching again')

    raise RuntimeError(f'The sets were still not separated after {MAX_SEPARATION_ROUNDS} rounds')


def similar_to(items: Sequence[SplitItem], others: Sequence[SplitItem],
               search: SimilaritySearch) -> list[SplitItem]:
    """The items similar (by the search's criterion) or identical to any of the others

    Parameters
    ----------
    items, others : list of SplitItem
    search : SimilaritySearch

    Returns
    -------
    list of SplitItem
        In the order of items
    """
    if not items or not others:
        return []
    matches = search([item.sequence for item in items], [other.sequence for other in others])
    linked = {match.query for match in matches if match.linked}
    other_sequences = {other.sequence for other in others}
    return [item for position, item in enumerate(items) if position in linked or item.sequence in other_sequences]


def cross_set_matches(items: Sequence[SplitItem], search: SimilaritySearch,
                      report: Callable[[str], None]) -> set[str]:
    """Groups in a lower-priority set that are similar (or identical) to a higher-priority set

    The pairs of sets are searched with the test set first, so validation
    records already to be removed (similar to the test set) are not searched
    against the training set: training records similar only to them can stay.

    Parameters
    ----------
    items : list of SplitItem
        The records, each assigned to 'train', 'val' or 'test'
    search : SimilaritySearch
        Finds similar sequences
    report : callable
        Receives a report line for each pair of sets searched

    Returns
    -------
    set of str
        The groups to remove
    """
    by_set: dict[str, list[SplitItem]] = defaultdict(list)
    for item in items:
        by_set[str(item.split)].append(item)

    offending: set[str] = set()
    for query_set, target_set in SEPARATED_SET_PAIRS:
        queries = [item for item in by_set.get(query_set, []) if item.group not in offending]
        targets = by_set.get(target_set, [])
        if not queries or not targets:
            continue

        matches = search([item.sequence for item in queries], [item.sequence for item in targets], weaker=True)
        linked = [match for match in matches if match.linked]
        offending.update(targets[match.target].group for match in linked)

        query_sequences = {item.sequence for item in queries}
        identical = [item.group for item in targets if item.sequence in query_sequences]
        offending.update(identical)

        report(f'{SET_LABELS[query_set]} vs {SET_LABELS[target_set]}: {len(matches):,} similarities from '
               f'{len({match.query for match in matches}):,} of {len(queries):,} {SET_LABELS[query_set]} sequences '
               f'({search.summarise(matches)}); {len(linked):,} meet the criterion; {len(identical):,} '
               f'{SET_LABELS[target_set]} sequences are identical to a {SET_LABELS[query_set]} sequence')
    return offending


def similarity_split(items: Sequence[SplitItem], fractions: dict[str, float], rng: np.random.Generator,
                     report: Callable[[str], None], search: SimilaritySearch) -> list[SplitItem]:
    """Split items into sets so that no set contains a sequence similar to one in another set

    Parameters
    ----------
    items : list of SplitItem
        The records (their groups and sets are modified in place)
    fractions : dict
        Target fraction of the records for each set
    rng : numpy.random.Generator
    report : callable
        Receives report lines
    search : SimilaritySearch
        Finds similar sequences, e.g. mmseqs_search.MMseqsSearch or kmer_similarity.KmerSearch

    Returns
    -------
    list of SplitItem
        The items kept (records similar to a higher-priority set are removed)

    Raises
    ------
    ValueError
        If the fractions do not sum to 1
    RuntimeError
        If the sets are still not separated after MAX_SEPARATION_ROUNDS rounds
    """
    report(f'Similarity criterion: {search.describe()}')
    merge_groups(items, identical_sequence_links(items))
    clusters = similarity_clusters(items, search, report)
    assign_clusters(items, clusters, fractions, rng, report)
    return enforce_separation(items, search, report)
