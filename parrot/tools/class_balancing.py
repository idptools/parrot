"""
Class balancing for parrot-preprocess --class-balance: removing lines from
classification data so that its classes are more balanced.

The imbalance of a dataset is the count of its most common class divided by
the count of its least common class (1.0 is perfectly balanced). Sequence
classes are balanced by capping every class at the same size
(run_class_balance_sequences); residue classes, where each line holds a mix of
classes, by removing one line at a time, each time the line of a random block
whose removal improves the balance most (run_class_balance_residues).

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np

from parrot.parrot_exceptions import ParrotException
from parrot.reproducibility import random_generator
from parrot.tools.dataset_warnings import BALANCED_CLASS_RATIO
from parrot.tools.preproc import Line, is_missing_label

# Two balance scores closer than this (relative) are treated as equal
SCORE_TOLERANCE = 1e-12

# Imbalances are compared rounded to this many decimal places when choosing between equally good removals, so
# floating-point noise cannot decide the choice
TIE_BREAK_DECIMALS = 9

# Defaults: balance until the classes are as balanced as dataset_warnings expects, removing at most a quarter of
# the lines, and considering 100 lines for each removal of residue balancing
DEFAULT_BALANCE_THRESHOLD = BALANCED_CLASS_RATIO
DEFAULT_MAX_FRACTION_REMOVED = 0.25
DEFAULT_BLOCK_SIZE = 100

# A block must offer at least two lines to choose between
MIN_BLOCK_SIZE = 2

# Balancing to an imbalance above this makes little sense, so it is refused
MAX_BALANCE_THRESHOLD = 5.0


def check_balance_settings(balance_threshold: float, max_fraction_removed: float) -> None:
    """Check the target imbalance and the most data that may be removed

    Parameters
    ----------
    balance_threshold : float
        The imbalance to balance to, between 1 and MAX_BALANCE_THRESHOLD
    max_fraction_removed : float
        The largest fraction of the lines that may be removed, between 0 and 1 (exclusive)

    Raises
    ------
    ParrotException
        If either is out of range
    """
    if not 1.0 <= balance_threshold <= MAX_BALANCE_THRESHOLD:
        raise ParrotException(f'Error during class balancing. The balance threshold must be between 1.0 and '
                              f'{MAX_BALANCE_THRESHOLD:g} (balancing to a larger imbalance makes little sense)')

    if not 0 < max_fraction_removed < 1.0:
        raise ParrotException('Error during class balancing. max_fraction_removed must be between 0 and 1 '
                              '(not inclusive)')


def class_key(label: str) -> float | str:
    """The class a label names: labels that are the same number ('1', '1.0') are the same class.

    Parameters
    ----------
    label : str

    Returns
    -------
    float or str
    """
    try:
        return float(label)
    except ValueError:
        return label


def class_count_matrix(lines: list[Line]) -> tuple[list[Any], np.ndarray]:
    """Count each line's residues in each class (missing labels are not counted).

    Parameters
    ----------
    lines : list
        Parsed datafile lines ([id, sequence, label, label, ...])

    Returns
    -------
    (list, numpy.ndarray)
        The classes, and an integer array with one row per line and one column per class
    """
    keys = [[class_key(label) for label in line[2:] if not is_missing_label(label)] for line in lines]
    classes = sorted({key for line_keys in keys for key in line_keys}, key=lambda key: (isinstance(key, str), key))
    column = {key: i for i, key in enumerate(classes)}
    counts = np.zeros((len(lines), len(classes)), dtype=np.int64)
    for row, line_keys in enumerate(keys):
        for key in line_keys:
            counts[row, column[key]] += 1
    return classes, counts


def balance_scores(class_totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score how imbalanced each row of class totals is.

    Parameters
    ----------
    class_totals : numpy.ndarray
        One row of class totals per possible dataset

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The imbalance (most / least common class; inf if a class would have no
        members left), and a secondary score, the sum over classes of
        (count / smallest count - 1)^2, which also falls when a class other
        than the most common one moves towards the smallest (inf likewise)
    """
    lowest = class_totals.min(axis=1)
    safe_lowest = np.maximum(lowest, 1)[:, None]
    imbalance = np.where(lowest > 0, class_totals.max(axis=1) / safe_lowest[:, 0], np.inf)
    spread = np.where(lowest > 0, ((class_totals / safe_lowest - 1) ** 2).sum(axis=1), np.inf)
    return imbalance, spread


def best_removal(totals: np.ndarray, counts: np.ndarray, candidates: np.ndarray,
                 current: tuple[float, float]) -> int | None:
    """The candidate line whose removal improves the balance most, or None if none improves it.

    A removal improves the balance if it lowers the imbalance, or keeps it the
    same and lowers the secondary score (so ties for the most common class can
    be broken one line at a time).

    Parameters
    ----------
    totals : numpy.ndarray
        Current class totals
    counts : numpy.ndarray
        Class counts of every line (see class_count_matrix)
    candidates : numpy.ndarray
        Lines that may be removed
    current : (float, float)
        Current imbalance and secondary score

    Returns
    -------
    int or None
    """
    imbalance, spread = balance_scores(totals[None, :] - counts[candidates])
    lower = imbalance < current[0] * (1 - SCORE_TOLERANCE)
    level = (imbalance <= current[0] * (1 + SCORE_TOLERANCE)) & (spread < current[1] * (1 - SCORE_TOLERANCE))
    better = np.flatnonzero(lower | level)
    if len(better) == 0:
        return None
    best = better[np.lexsort((spread[better], np.round(imbalance[better], TIE_BREAK_DECIMALS)))[0]]
    return int(candidates[best])


def run_class_balance_residues(lines: list[Line], balance_threshold: float = DEFAULT_BALANCE_THRESHOLD,
                               block_size: int = DEFAULT_BLOCK_SIZE,
                               max_fraction_removed: float = DEFAULT_MAX_FRACTION_REMOVED,
                               rng: np.random.Generator | None = None) -> tuple[list[Line], list[float]]:
    """Remove lines from residue-classification data so its classes are more balanced

    This implements Stochastic Residue Class Balancing (SRCB). Each step draws
    block_size lines at random (with replacement), works out the imbalance
    after removing each of them, and removes the one that improves it most. If
    none of the block improves the balance, every remaining line is checked
    before giving up, and a line that would make the balance worse (or delete a
    class entirely) is never removed. Balancing stops when the imbalance is at
    most balance_threshold, when max_fraction_removed of the lines have been
    removed, or when no single removal improves the balance (see
    residue_balance_stop_reason()). Each step costs O(block_size) unless no
    line of the block improves the balance, so this scales to very large
    datasets; a larger block makes better choices.

    Parameters
    ----------
    lines : list
        Lines of the form [<id>, <sequence>, <class1>, ..., <classn>], with one
        class label per residue
    balance_threshold : float, optional
        Stop once the imbalance (most common class / least common class) is at
        most this
    block_size : int, optional
        Number of lines considered for removal at each step (at least 2)
    max_fraction_removed : float, optional
        The largest fraction of the lines that may be removed (between 0 and 1, exclusive)
    rng : numpy.random.Generator, optional
        Source of randomness. By default one is seeded from NumPy's global
        generator, so reproducibility.set_random_seed() makes balancing reproducible.

    Returns
    -------
    list
        The kept lines, in their input order
    list of float
        The imbalance before balancing and after each removal

    Raises
    ------
    ParrotException
        If a setting is out of range or no line has a class label
    """

    if block_size < MIN_BLOCK_SIZE:
        raise ParrotException(f'Error during class balancing. Block size must be at least {MIN_BLOCK_SIZE}')
    check_balance_settings(balance_threshold, max_fraction_removed)
    if rng is None:
        rng = random_generator()

    classes, counts = class_count_matrix(lines)
    if len(classes) == 0:
        raise ParrotException('Error during class balancing. No class labels found (every label is missing)')

    totals = counts.sum(axis=0)
    imbalance, spread = balance_scores(totals[None, :])
    current = (float(imbalance[0]), float(spread[0]))
    tracker = [current[0]]

    # The most lines that may be removed (0 for very small datasets: max_fraction_removed is a hard limit)
    budget = removal_budget(len(lines), max_fraction_removed)

    # Lines not yet removed, in no particular order; position[i] is where line i is in `remaining`
    remaining = np.arange(len(lines))
    position = np.arange(len(lines))
    n_remaining = len(lines)
    while len(tracker) - 1 < budget and tracker[-1] > balance_threshold:
        block = remaining[rng.integers(0, n_remaining, size=min(block_size, n_remaining))]
        choice = best_removal(totals, counts, block, current)
        if choice is None:
            choice = best_removal(totals, counts, remaining[:n_remaining], current)
            if choice is None:
                break

        # Remove the chosen line: move the last remaining line into its place
        totals = totals - counts[choice]
        last = remaining[n_remaining - 1]
        remaining[position[choice]] = last
        position[last] = position[choice]
        n_remaining -= 1

        imbalance, spread = balance_scores(totals[None, :])
        current = (float(imbalance[0]), float(spread[0]))
        tracker.append(current[0])

    kept = np.sort(remaining[:n_remaining])
    return [lines[i] for i in kept], tracker


def removal_budget(num_lines: int, max_fraction_removed: float) -> int:
    """The most lines class balancing may remove.

    Parameters
    ----------
    num_lines : int
    max_fraction_removed : float

    Returns
    -------
    int
        max_fraction_removed of the lines, rounded down
    """
    return int(num_lines * max_fraction_removed)


def residue_balance_stop_reason(tracker: list[float], balance_threshold: float, budget: int) -> str:
    """Why run_class_balance_residues() stopped, in words

    Parameters
    ----------
    tracker : list of float
        The imbalance before balancing and after each removal
    balance_threshold : float
        The imbalance it balanced to
    budget : int
        The most lines it could remove (see removal_budget())

    Returns
    -------
    str
    """
    if tracker[-1] <= balance_threshold:
        return f'the imbalance reached the target of {balance_threshold:g}'
    if len(tracker) - 1 >= budget:
        return f'the most lines that may be removed ({budget:,}) were removed'
    return 'no single removal improves the balance any further'


def format_class_counts(classes: list[Any], totals: np.ndarray | list[int]) -> str:
    """Class counts as text, e.g. '0: 120, 1: 80'."""
    return ', '.join(f'{format_class(key)}: {int(total):,}' for key, total in zip(classes, totals))


def format_class(key: float | str) -> str:
    """A class key (see class_key) as text: whole numbers without a decimal point."""
    if isinstance(key, float) and key.is_integer():
        return str(int(key))
    return str(key)


def sequence_class_cap(class_counts: list[int], budget: int, balance_threshold: float) -> int:
    """The largest number of lines each class can keep, for sequence class balancing.

    Capping every class at the same size is the best possible balancing of
    sequence classes: the cap is lowered until the most common class is at most
    balance_threshold times the least common, or until the removals would exceed
    the budget, whichever comes first. The least common class is never reduced.

    Parameters
    ----------
    class_counts : list of int
        Lines in each class
    budget : int
        Most lines that may be removed
    balance_threshold : float

    Returns
    -------
    int
    """
    smallest, largest = min(class_counts), max(class_counts)
    target = max(smallest, math.floor(balance_threshold * smallest * (1 + SCORE_TOLERANCE)))
    if target >= largest:
        return largest

    def removals(cap: int) -> int:
        """Lines removed if every class is capped at `cap`."""
        return sum(max(0, count - cap) for count in class_counts)

    if removals(target) <= budget:
        return target

    # removals() falls as the cap rises: find the lowest cap within the budget
    low, high = target, largest
    while high - low > 1:
        middle = (low + high) // 2
        if removals(middle) <= budget:
            high = middle
        else:
            low = middle
    return high


def run_class_balance_sequences(lines: list[Line], balance_threshold: float = DEFAULT_BALANCE_THRESHOLD,
                                max_fraction_removed: float = DEFAULT_MAX_FRACTION_REMOVED,
                                rng: np.random.Generator | None = None) -> list[Line]:
    """Remove lines from sequence-classification data so its classes are more balanced

    Every class is capped at the same size, chosen by sequence_class_cap():
    the cap is lowered until the most common class is at most
    balance_threshold times the least common one, or as far as
    max_fraction_removed allows. This removes the fewest lines needed for the
    best balance possible within the budget. Lines are removed from each class
    over its cap at random; lines with a missing label are never removed.

    Parameters
    ----------
    lines : list
        Lines of the form [<id>, <sequence>, <class>]
    balance_threshold : float, optional
        Stop once the imbalance (most common class / least common class) is at most this
    max_fraction_removed : float, optional
        The largest fraction of the lines that may be removed (between 0 and 1, exclusive)
    rng : numpy.random.Generator, optional
        Source of randomness. By default one is seeded from NumPy's global
        generator, so reproducibility.set_random_seed() makes balancing reproducible.

    Returns
    -------
    list
        The kept lines, in their input order

    Raises
    ------
    ParrotException
        If a setting is out of range or no line has a class label
    """

    check_balance_settings(balance_threshold, max_fraction_removed)
    if rng is None:
        rng = random_generator()

    # Line numbers of each class; lines with a missing label belong to no class and are kept
    members: dict[Any, list[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        if not is_missing_label(line[2]):
            members[class_key(line[2])].append(index)
    if len(members) == 0:
        raise ParrotException('Error during class balancing. No class labels found (every label is missing)')

    classes = sorted(members, key=lambda key: (isinstance(key, str), key))
    before = [len(members[key]) for key in classes]
    cap = sequence_class_cap(before, removal_budget(len(lines), max_fraction_removed), balance_threshold)

    removed: set[int] = set()
    for key in classes:
        excess = len(members[key]) - cap
        if excess > 0:
            removed.update(rng.choice(members[key], size=excess, replace=False).tolist())
    return [line for index, line in enumerate(lines) if index not in removed]
