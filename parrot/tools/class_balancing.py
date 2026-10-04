"""
Module with functions for dealing with automatic class balacing

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""

import datetime
import math
from collections import defaultdict

import numpy as np


from parrot.parrot_exceptions import ParrotException 
from parrot.tools.preproc import is_missing_label

# Two balance scores closer than this (relative) are treated as equal
SCORE_TOLERANCE = 1e-12



def __validate_class_balance_input(balance_threshold, max_fraction_removed):
    """
    Helper function that checks that a balance_threshold and a max_fraction_removed threshold

    Parameters
    -------------
    balance_threshold : float
        Float that reports on the balance score threshold to be used. Expects to be between 1 and 5.0

    max_fraction_removed : float
        Fraction which defines the max fraction of the data that can be removed in the name of 
        class balancing. Expects to be between 0 and 1 not inclusive.


    Returns
    ----------
    None
        No return

    Raises
    ----------
    parrot.parrot_exceptions.ParrotException
        If an invalid exception then raises a ParrotException

    """
    
    if balance_threshold < 1.0 or balance_threshold > 5:
        raise ParrotException('Error during class balancing. The balance threshold must be between 1.0 and 5 - asking to balance with a balance score > 5 does not any sense')

    if max_fraction_removed <= 0 or max_fraction_removed >= 1.0:
        raise ParrotException('Error during class balancing. max_fraction_removed must be between 0 and 1 (not inclusive)')
    

# ..........................................................................................
#
#    
def class_key(label):
    """
    The class a label names: labels that are the same number ('1', '1.0') are the same class.

    Parameters
    ------------
    label : str

    Returns
    ----------
    float or str
    """
    try:
        return float(label)
    except ValueError:
        return label


def class_count_matrix(lines):
    """
    Count each line's residues in each class (missing labels are not counted).

    Parameters
    ------------
    lines : list
        Parsed datafile lines ([id, sequence, label, label, ...])

    Returns
    ----------
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


def balance_scores(class_totals):
    """
    Score how imbalanced each row of class totals is.

    Parameters
    ------------
    class_totals : numpy.ndarray
        One row of class totals per possible dataset

    Returns
    ----------
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


def best_removal(totals, counts, candidates, current):
    """
    The candidate line whose removal improves the balance most, or None if none improves it.

    A removal improves the balance if it lowers the imbalance, or keeps it the
    same and lowers the secondary score (so ties for the most common class can
    be broken one line at a time).

    Parameters
    ------------
    totals : numpy.ndarray
        Current class totals
    counts : numpy.ndarray
        Class counts of every line (see class_count_matrix)
    candidates : numpy.ndarray
        Lines that may be removed
    current : (float, float)
        Current imbalance and secondary score

    Returns
    ----------
    int or None
    """
    imbalance, spread = balance_scores(totals[None, :] - counts[candidates])
    lower = imbalance < current[0] * (1 - SCORE_TOLERANCE)
    level = (imbalance <= current[0] * (1 + SCORE_TOLERANCE)) & (spread < current[1] * (1 - SCORE_TOLERANCE))
    better = np.flatnonzero(lower | level)
    if len(better) == 0:
        return None
    best = better[np.lexsort((spread[better], np.round(imbalance[better], 9)))[0]]
    return int(candidates[best])


def run_class_balance_residues(lines,
                               balance_threshold=1.4,
                               block_size=100,
                               max_fraction_removed=0.25,
                               verbose=True,
                               rng=None):

    """
    Remove lines from residue-classification data so its classes are more balanced.

    This implements Stochastic Residue Class Balancing (SRCB). The imbalance
    score is the count of the most common class divided by the count of the
    least common class (1.0 is perfectly balanced). Each step draws block_size
    lines at random, works out the imbalance after removing each of them, and
    removes the one that improves it most. If none of the block improves the
    balance, every remaining line is checked before giving up, and a line that
    would make the balance worse (or delete a class entirely) is never removed.
    Balancing stops when the imbalance is at most balance_threshold, when
    max_fraction_removed of the lines have been removed, or when no single
    removal improves the balance. Each step costs O(block_size), so this scales
    to very large datasets; a larger block makes better choices.

    Parameters
    -------------
    lines : list
        Lines is a list of lists, where each sublist is of the format [<id>, <sequence>, <class1>, ...,<classn>]. The
        number of class1,class2 data points should match the number of residues in the <sequence>.

    balance_threshold : float
        Stop once the imbalance score (most common class / least common class) is at most this. We use a threshold
        of 1.4 to match the warning thresholds defined in parrot.tools.dataset_warnings. Default = 1.4

    block_size : int
        Number of lines considered for removal at each step (at least 2). Default 100.

    max_fraction_removed : float
        The largest fraction of the lines that may be removed. Must be between 0 and 1 (not inclusive of
        either). Default = 0.25.

    verbose : bool
        Flag which indicates if status and input info should be printed to screen.

    rng : numpy.random.Generator, optional
        Source of randomness. By default one is seeded from NumPy's global generator, so
        parrot.tools.cli.set_random_seed() makes balancing reproducible.

    Returns
    -----------
    (list, list)
        Returns a tuple of two lists:

        [0] - The kept lines, in their input order.

        [1] - The imbalance score before balancing and after each removal.

    """

    # validate input...
    if block_size < 2:
        raise ParrotException('Error during class balancing. Block size must be greater than 1')

    __validate_class_balance_input(balance_threshold, max_fraction_removed)

    if rng is None:
        rng = np.random.default_rng(np.random.randint(2 ** 31))

    classes, counts = class_count_matrix(lines)
    if len(classes) == 0:
        raise ParrotException('Error during class balancing. No class labels found (every label is missing)')

    totals = counts.sum(axis=0)
    imbalance, spread = balance_scores(totals[None, :])
    current = (float(imbalance[0]), float(spread[0]))
    tracker = [current[0]]

    # the most lines that may be removed (0 for very small datasets: max_fraction_removed is a hard limit)
    budget = int(len(lines) * max_fraction_removed)

    if verbose:
        print('-------------------------------------------------------------------')
        print('Performing automatic class balancing for residues (--class-balance)')
        print('-------------------------------------------------------------------')
        print('')
        print(f'   Found {len(classes)} different classes accross {len(lines)} sequences')
        print('   Class breakdown:')
        print('   ' + format_class_counts(classes, totals))
        print(f'   These classes have an imbalance ratio of {current[0]:.4g}')
        print(f'   Max num. of removable sequences : {budget}')
        print(f'   Imbalance threshold             : {balance_threshold}')
        print(f'   Block size                      : {block_size}')
        print(f'   Starting at {datetime.datetime.now()}')

    # Lines not yet removed, in no particular order; position[i] is where line i is in `remaining`
    remaining = np.arange(len(lines))
    position = np.arange(len(lines))
    n_remaining = len(lines)
    stuck = False
    while len(tracker) - 1 < budget and tracker[-1] > balance_threshold:
        block = remaining[rng.integers(0, n_remaining, size=min(block_size, n_remaining))]
        choice = best_removal(totals, counts, block, current)
        if choice is None:
            choice = best_removal(totals, counts, remaining[:n_remaining], current)
            if choice is None:
                stuck = True
                break

        totals = totals - counts[choice]
        last = remaining[n_remaining - 1]
        remaining[position[choice]] = last
        position[last] = position[choice]
        n_remaining -= 1

        imbalance, spread = balance_scores(totals[None, :])
        current = (float(imbalance[0]), float(spread[0]))
        tracker.append(current[0])

    kept = np.sort(remaining[:n_remaining])
    new_lines = [lines[i] for i in kept]

    if verbose:
        removed = len(lines) - len(new_lines)
        if stuck:
            print(f'   Stopped: no single removal improves the balance any further')
        print('   -------------------------------------')
        print(f'   Complete at {datetime.datetime.now()}')
        print(f'   Removed {removed} sequences ({removed / len(lines):.3f} of sequences)')
        print(f'   Updated balance score: {tracker[-1]:.4g}')
        print(f'   Updated class breakdown:')
        print('   ' + format_class_counts(classes, totals))
        print('   -------------------------------------\n')

    return (new_lines, tracker)


def format_class_counts(classes, totals):
    """Class counts as text, e.g. '0: 120, 1: 80'."""
    return ', '.join(f'{format_class(key)}: {int(total):,}' for key, total in zip(classes, totals))


def format_class(key):
    """A class key (see class_key) as text: whole numbers without a decimal point."""
    if isinstance(key, float) and key.is_integer():
        return str(int(key))
    return str(key)


def sequence_class_cap(class_counts, budget, balance_threshold):
    """
    The largest number of lines each class can keep, for sequence class balancing.

    Capping every class at the same size is the best possible balancing of
    sequence classes: the cap is lowered until the most common class is at most
    balance_threshold times the least common, or until the removals would exceed
    the budget, whichever comes first. The least common class is never reduced.

    Parameters
    ------------
    class_counts : list of int
        Lines in each class
    budget : int
        Most lines that may be removed
    balance_threshold : float

    Returns
    ----------
    int
    """
    smallest, largest = min(class_counts), max(class_counts)
    target = max(smallest, math.floor(balance_threshold * smallest * (1 + SCORE_TOLERANCE)))
    if target >= largest:
        return largest

    def removals(cap):
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


def run_class_balance_sequences(lines,
                                balance_threshold=1.4,
                                max_fraction_removed=0.25,
                                verbose=True,
                                rng=None):

    """
    Remove lines from sequence-classification data so its classes are more balanced.

    Every class is capped at the same size, chosen by sequence_class_cap():
    the cap is lowered until the most common class is at most
    balance_threshold times the least common one, or as far as
    max_fraction_removed allows. This removes the fewest lines needed for the
    best balance possible within the budget. Lines are removed from each class
    over its cap at random; lines with a missing label are never removed.

    Parameters
    -------------
    lines : list
        Lines is a list of lists, where each sublist is of the format [<id>, <sequence>, <class>].

    balance_threshold : float
        Stop once the imbalance score (most common class / least common class) is at most this. We use a threshold
        of 1.4 to match the warning thresholds defined in parrot.tools.dataset_warnings. Default = 1.4

    max_fraction_removed : float
        The largest fraction of the lines that may be removed. Must be between 0 and 1 (not inclusive of
        either). Default = 0.25.

    verbose : bool
        Flag which indicates if status and input info should be printed to screen.

    rng : numpy.random.Generator, optional
        Source of randomness. By default one is seeded from NumPy's global generator, so
        parrot.tools.cli.set_random_seed() makes balancing reproducible.

    Returns
    -----------
    list
        The kept lines, in their input order.

    """

    # validate input options (same validation as used for run_class_balance_residues())
    __validate_class_balance_input(balance_threshold, max_fraction_removed)

    if rng is None:
        rng = np.random.default_rng(np.random.randint(2 ** 31))

    # line numbers of each class; lines with a missing label belong to no class and are kept
    members = defaultdict(list)
    for index, line in enumerate(lines):
        if not is_missing_label(line[2]):
            members[class_key(line[2])].append(index)
    if len(members) == 0:
        raise ParrotException('Error during class balancing. No class labels found (every label is missing)')

    classes = sorted(members, key=lambda key: (isinstance(key, str), key))
    before = [len(members[key]) for key in classes]
    budget = int(len(lines) * max_fraction_removed)
    cap = sequence_class_cap(before, budget, balance_threshold)

    removed = set()
    for key in classes:
        excess = len(members[key]) - cap
        if excess > 0:
            removed.update(rng.choice(members[key], size=excess, replace=False).tolist())
    new_lines = [line for index, line in enumerate(lines) if index not in removed]
    after = [min(count, cap) for count in before]

    if verbose:
        print('-------------------------------------------------------------------')
        print('Performing automatic class balancing for sequences (--class-balance)')
        print('-------------------------------------------------------------------')
        print('')
        print(f'   Found {len(classes)} different classes accross {len(lines)} sequences')
        print('   Class breakdown: ' + format_class_counts(classes, before))
        print(f'   These classes have an imbalance ratio of {max(before) / min(before):.4g}')
        print(f'   Max num. of removable sequences: {budget}')
        print(f'   Imbalance threshold            : {balance_threshold}')
        print(f'   Removed {len(removed)} sequences ({len(removed) / len(lines):.3f} of sequences), capping every '
              f'class at {cap}')
        print(f'   Updated balance score: {max(after) / min(after):.4g}')
        print('   Updated class breakdown: ' + format_class_counts(classes, after))
        print('   -------------------------------------\n')

    return new_lines
