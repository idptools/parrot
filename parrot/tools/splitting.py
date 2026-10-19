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
from itertools import islice
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

# Default maximum removal rounds, followed by a final verification search
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
    sequence_id : str or None
        Input ID, when available, for examples in the report
    """

    index: int
    sequence: str
    group: str
    split: str | None = None
    sequence_id: str | None = None


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


def report_retention(items: Sequence[SplitItem], initial_counts: Counter[str],
                     report: Callable[[str], None]) -> None:
    """Report cumulative removals and retention relative to the initially assigned sets."""
    total = initial_counts.total()
    removed = total - len(items)
    report(f'Cumulative removals: {removed:,} of {total:,} records '
           f'({100 * removed / total:.1f}%)' if total else 'Cumulative removals: 0 of 0 records')
    counts = Counter(item.split for item in items)
    parts = []
    for name in SET_NAMES:
        retention = f'{100 * counts[name] / initial_counts[name]:.1f}% retained' if initial_counts[name] else 'n/a'
        parts.append(f'{SET_LABELS[name]} {counts[name]:,} of {initial_counts[name]:,} ({retention})')
    report('Set retention: ' + ', '.join(parts))


def enforce_separation(items: Sequence[SplitItem], search: SimilaritySearch,
                       report: Callable[[str], None], *, max_rounds: int = MAX_SEPARATION_ROUNDS) -> list[SplitItem]:
    """Search each set against the others, removing records similar to a higher-priority set, until none remain

    Parameters
    ----------
    items : list of SplitItem
        The records, each assigned to 'train', 'val' or 'test'
    search : SimilaritySearch
        Finds similar sequences
    report : callable
        Receives report lines
    max_rounds : int
        Maximum removal rounds (at least 1), followed by a final verification
        search. At most max_rounds + 1 cross-set checks are made.

    Returns
    -------
    list of SplitItem
        The items kept

    Raises
    ------
    RuntimeError
        If the sets are still not separated after max_rounds removal rounds
    ValueError
        If max_rounds is less than 1
    """
    if max_rounds < 1:
        raise ValueError(f'max_rounds must be at least 1, got {max_rounds}')
    kept = list(items)
    initial_counts = Counter(str(item.split) for item in items)
    budget_text = f'{max_rounds} removal round' + ('s' if max_rounds != 1 else '')
    report(f'Separation budget: {budget_text}, plus a final verification check')
    report('Retention is relative to the sets assigned after cleaning and deduplication, before separation.')
    report_retention(kept, initial_counts, report)
    for rounds_done in range(max_rounds + 1):
        rounds_text = f'{rounds_done} removal round' + ('s' if rounds_done != 1 else '')
        checks_text = f'{rounds_done + 1} check' + ('s' if rounds_done else '')
        report(f'Separation check {rounds_done + 1} (after {rounds_text})')
        offending = cross_set_matches(kept, search, report)
        if not offending:
            report(f'Separation completed after {rounds_text} and {checks_text}.')
            report_retention(kept, initial_counts, report)
            report('Review the retained set sizes and their label/target distributions before training; '
                   'passing separation only establishes the absence of cross-set matches detected by this search.')
            report('PASSED: no sequence in one set is identical or similar (by the criterion above) to a sequence '
                   'in another set')
            return kept

        if rounds_done == max_rounds:
            pending = Counter(str(item.split) for item in kept if item.group in offending)
            conflicts_text = f'{len(offending):,} conflicting group' + ('s' if len(offending) != 1 else '')
            records_text = f'{pending.total():,} record' + ('s' if pending.total() != 1 else '')
            report(f'FAILED separation: {conflicts_text} remaining; another removal round would '
                   f'remove {records_text} ('
                   + ', '.join(f'{SET_LABELS[name]} {pending[name]:,}' for name in SET_NAMES) + ').')
            report_retention(kept, initial_counts, report)
            report(f'Next step: if the cumulative data loss and remaining set sizes are acceptable, rerun with '
                   f'--max-separation-rounds {max_rounds * 2}. Use the same --seed to compare the same initial '
                   'assignment; adding a seed to an unseeded run changes the assignment. This restarts '
                   'preprocessing; there is no separation checkpoint to resume.')
            report('If too much data is lost, inspect the example pairs and group/cluster sizes above. Check '
                   'that --groups reflects real dependencies, and consider more independent sequences or '
                   'revised --set-fractions. More rounds cannot guarantee sufficiently large sets. Changing '
                   'the similarity thresholds or using a random split changes what leakage is excluded.')
            raise RuntimeError(f'The sets were still not separated after {budget_text} '
                               f'({conflicts_text} remaining). See the separation diagnostics; '
                               'increase --max-separation-rounds if the reported data loss is acceptable.')

        before = len(kept)
        kept = [item for item in kept if item.group not in offending]
        report(f'Round {rounds_done + 1}: removed {before - len(kept):,} records ({len(offending):,} groups) similar '
               f'to a higher-priority set ({100 * (before - len(kept)) / before:.1f}% of records entering this '
               'round); searching again')
        report_retention(kept, initial_counts, report)

    raise AssertionError('Separation loop ended without a result')


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
        shared_sequences = query_sequences.intersection(item.sequence for item in targets)
        linked_queries = {match.query for match in linked}
        linked_queries.update(i for i, item in enumerate(queries) if item.sequence in shared_sequences)
        linked_targets = {match.target for match in linked}
        linked_targets.update(i for i, item in enumerate(targets) if item.sequence in shared_sequences)
        report(f'  Conflicts involve {len(linked_queries):,} of {len(queries):,} '
               f'{SET_LABELS[query_set]} records ({100 * len(linked_queries) / len(queries):.1f}%) and '
               f'{len(linked_targets):,} of {len(targets):,} {SET_LABELS[target_set]} records '
               f'({100 * len(linked_targets) / len(targets):.1f}%); counts include exact duplicates.')
        if linked:
            strengths = [(f'identity {min(match.identity for match in linked):.1%} to '
                          f'{max(match.identity for match in linked):.1%}')]
            # Search engines may provide extra metrics beyond the Match interface.
            evalues = [value for match in linked if (value := getattr(match, 'evalue', None)) is not None]
            coverage = [value for match in linked
                        if (value := getattr(match, 'shorter_coverage', getattr(match, 'coverage', None))) is not None]
            if evalues:
                strengths.append(f'E-value {min(evalues):.3g} to {max(evalues):.3g}')
            if coverage:
                strengths.append(f'shorter-sequence coverage {min(coverage):.1%} to {max(coverage):.1%}')
            report('  Linked match ranges: ' + '; '.join(strengths))
            examples = [(queries[match.query], targets[match.target]) for match in linked[:3]]
        else:
            # Exact copies can be missed by the search, e.g. because of masking.
            query_by_sequence = {item.sequence: item for item in queries}
            examples = list(islice(((query_by_sequence[item.sequence], item) for item in targets
                                    if item.sequence in shared_sequences), 3))
        if examples:
            report('  Example conflicting pairs (up to 3; input IDs or record numbers): ' + '; '.join(
                f'{first.sequence_id or f"record {first.index + 1}"} -> '
                f'{second.sequence_id or f"record {second.index + 1}"}' for first, second in examples))
    return offending


def similarity_split(items: Sequence[SplitItem], fractions: dict[str, float], rng: np.random.Generator,
                     report: Callable[[str], None], search: SimilaritySearch, *,
                     max_rounds: int = MAX_SEPARATION_ROUNDS) -> list[SplitItem]:
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
    max_rounds : int
        Maximum removal rounds, followed by a final verification search

    Returns
    -------
    list of SplitItem
        The items kept (records similar to a higher-priority set are removed)

    Raises
    ------
    ValueError
        If the fractions do not sum to 1
    RuntimeError
        If the sets are still not separated after max_rounds removal rounds
    """
    report(f'Similarity criterion: {search.describe()}')
    merge_groups(items, identical_sequence_links(items))
    clusters = similarity_clusters(items, search, report)
    assign_clusters(items, clusters, fractions, rng, report)
    return enforce_separation(items, search, report, max_rounds=max_rounds)
