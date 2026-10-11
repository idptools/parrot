"""
How parrot-train and parrot-optimize divide the data into training, validation
and test sets (--split, which every training command must give).

--split names a split-file or a method, and a file of that name always comes
first, so a split-file called (say) 'random' is still read as a split-file:

- a split-file: three lines of datafile line numbers, for the training,
  validation and test sets (e.g. from parrot-preprocess --split or
  parrot-cvsplit). Lines in no set are left out, and counted.
- 'random': sequences assigned to sets at random, keeping identical sequences
  in the same set, in the proportions --set-fractions gives.
- 'kmer' or 'mmseqs': similar sequences kept in the same set, then anything
  still similar to a sequence in another set removed, so that no set contains a
  relative of a sequence in another; similarity is found by PARROT's k-mer
  search or by MMseqs2.

The methods are exactly those of parrot-preprocess --split random, kmer and
homology (see tools/splitting.py), with parrot-preprocess's default criteria;
for other criteria, or for groups of records to keep together, make a
split-file with parrot-preprocess.

Cross-validation folds (parrot-optimize --cv-folds) are made from the training
and validation sequences in the same spirit: identical sequences are always in
the same fold, and with 'kmer' or 'mmseqs' so are clusters of similar
sequences, and each fold is scored only on its held-out sequences that are not
similar to any of its training sequences.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np

from parrot import process_input_data as pid
from parrot.reproducibility import random_generator
from parrot.scripts import arguments
from parrot.tools import kmer_similarity, mmseqs_search, splitting

# The methods --split accepts (anything else must be a split-file)
SPLIT_METHODS = ('mmseqs', 'kmer', 'random')

# Each method's name in parrot-preprocess --split
PREPROCESS_METHOD_NAMES = {'mmseqs': 'homology', 'kmer': 'kmer', 'random': 'random'}

# How each method assigns sequences, in words
METHOD_DESCRIPTIONS = {
    'random': 'at random, keeping identical sequences in the same set',
    'kmer': "keeping similar sequences (found by PARROT's k-mer search) in the same set",
    'mmseqs': 'keeping homologous sequences (found by MMseqs2) in the same set',
}


@dataclass
class DataSplit:
    """How the data are divided into sets: a split-file or a method, applied when the data are read

    Passed to process_input_data.split_data() as `assign_sets`, which calls it
    with the parsed data; it then records what it did, for describing the run.

    Attributes
    ----------
    split_file : str or None
        The split-file, if the sets come from one
    method : str or None
        'random', 'kmer' or 'mmseqs', if the sets are made from the data
    fractions : tuple of 3 floats or None
        Target fractions of the training, validation and test sets (methods only)
    silent : bool
        If True, print nothing
    num_sequences : int
        Sequences in the datafile (once applied)
    set_sizes : tuple of 3 ints
        Sequences in the training, validation and test sets (once applied)
    criterion : str
        For 'kmer' and 'mmseqs', what counts as similar
    """
    split_file: str | None
    method: str | None
    fractions: tuple[float, float, float] | None
    silent: bool
    num_sequences: int = 0
    set_sizes: tuple[int, int, int] = (0, 0, 0)
    criterion: str = ''

    def __call__(self, data: list[pid.Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign the parsed data to sets

        Parameters
        ----------
        data : list
            Every parsed line of the datafile

        Returns
        -------
        tuple of numpy int arrays
            (training, validation, test) indices

        Raises
        ------
        ValueError
            If any set is empty, or the split-file does not fit the datafile
        """
        self.num_sequences = len(data)
        if self.split_file is not None:
            sets = pid.read_split_file(self.split_file, len(data))
            context = f'in the split-file {self.split_file}'
        else:
            sets = self.split_by_method(data)
            context = (f'when {len(data):,} sequences are split ({self.method}). Use more data, larger '
                       '--set-fractions or a split-file.')
        pid.check_sets_not_empty(*sets, context)

        training, validation, test = sets
        self.set_sizes = (len(training), len(validation), len(test))
        if self.num_unassigned and not self.silent:
            print(f'Note: {self.num_unassigned:,} of {self.num_sequences:,} sequences are in no set '
                  f'({self.unassigned_reason}), so they are not used')
        return sets

    def split_by_method(self, data: list[pid.Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make the sets from the data, as parrot-preprocess --split does

        Parameters
        ----------
        data : list
            Every parsed line of the datafile

        Returns
        -------
        tuple of numpy int arrays
            (training, validation, test) indices, in datafile order
        """
        assert self.method is not None and self.fractions is not None
        items = [splitting.SplitItem(index, sample[1], f'sequence {index}') for index, sample in enumerate(data)]
        fractions = dict(zip(splitting.SET_NAMES, self.fractions))

        if not self.silent:
            print(f'Splitting {len(data):,} sequences ({self.method}, as parrot-preprocess --split '
                  f'{PREPROCESS_METHOD_NAMES[self.method]} does):')

        def report(line: str) -> None:
            if not self.silent:
                print(f'  {line}')

        if self.method == 'random':
            kept = splitting.random_split(items, fractions, random_generator(), report)
        else:
            search = similarity_search(self.method)
            self.criterion = search.describe()
            kept = splitting.similarity_split(items, fractions, random_generator(), report, search)

        training, validation, test = (np.array(sorted(item.index for item in kept if item.split == name), dtype=int)
                                      for name in splitting.SET_NAMES)
        return training, validation, test

    def cross_validation_folds(self, data: list[pid.Sample], samples: np.ndarray,
                               n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Divide the training and validation sequences into cross-validation folds (see the module documentation)

        Passed to process_input_data.split_data_cv() as `assign_folds`.

        Parameters
        ----------
        data : list
            Every parsed line of the datafile
        samples : numpy int array
            The training and validation sequences' indices
        n_folds : int

        Returns
        -------
        list of (numpy int array, numpy int array)
            For each fold, (training indices, held-out indices to score it on)

        Raises
        ------
        ValueError
            If a fold would have no held-out sequence to score
        """
        if self.method not in ('kmer', 'mmseqs'):
            return pid.identical_sequence_folds(data, samples, n_folds)

        def report(line: str) -> None:
            if not self.silent:
                print(f'  {line}')

        if not self.silent:
            print(f'Dividing the {len(samples):,} training and validation sequences into {n_folds} '
                  'cross-validation folds, keeping similar sequences in the same fold:')
        items = [splitting.SplitItem(int(index), data[index][1], f'sequence {index}') for index in samples]
        splitting.merge_groups(items, splitting.identical_sequence_links(items))
        search = similarity_search(str(self.method))
        clusters = splitting.similarity_clusters(items, search, report)
        names = [f'fold {fold}' for fold in range(1, n_folds + 1)]
        splitting.assign_clusters(items, clusters, {name: 1 / n_folds for name in names}, random_generator(), report)

        folds = []
        for name in names:
            held_out = [item for item in items if item.split == name]
            training = [item for item in items if item.split != name]
            similar = {item.index for item in splitting.similar_to(held_out, training, search)}
            scored = [item.index for item in held_out if item.index not in similar]
            report(f'{name}: {len(similar):,} of its {len(held_out):,} held-out sequences are similar to one of its '
                   'training sequences, so it is not scored on them')
            if not scored:
                raise ValueError(f'Cross-validation {name} has no held-out sequence that is not similar to one of '
                                 'its training sequences; use fewer --cv-folds, or score on the validation set '
                                 '(leave out --cv-folds)')
            folds.append((np.array(sorted(item.index for item in training), dtype=int),
                          np.array(sorted(scored), dtype=int)))
        return folds

    def fold_description(self) -> str:
        """How cross-validation folds are made, in words (see cross_validation_folds)

        Returns
        -------
        str
        """
        if self.method in ('kmer', 'mmseqs'):
            return ('keeping similar sequences in the same fold, and scoring each fold only on held-out sequences '
                    'not similar to its training sequences')
        return 'keeping identical sequences in the same fold'

    @property
    def num_unassigned(self) -> int:
        """Sequences of the datafile in no set."""
        return self.num_sequences - sum(self.set_sizes)

    @property
    def unassigned_reason(self) -> str:
        """Why sequences are in no set."""
        if self.split_file is not None:
            return 'the split-file does not list them'
        return 'they were removed for being similar to a sequence in another set'

    def description(self) -> str:
        """How the data were divided into sets, in words (for the network's info file)

        Returns
        -------
        str
        """
        if self.split_file is not None:
            description = f'from the split-file {self.split_file}'
        else:
            assert self.method is not None and self.fractions is not None
            train, validation, test = self.fractions
            description = (f'{METHOD_DESCRIPTIONS[self.method]} (as parrot-preprocess --split '
                           f'{PREPROCESS_METHOD_NAMES[self.method]}), aiming for {train:g} of the sequences for '
                           f'training, {validation:g} for validation and {test:g} for testing')
            if self.criterion:
                description += f'; similarity criterion: {self.criterion}'
        if self.num_unassigned:
            description += (f'; {self.num_unassigned:,} of {self.num_sequences:,} sequences are in no set '
                            f'({self.unassigned_reason})')
        return description


def similarity_search(method: str) -> splitting.SimilaritySearch:
    """The search engine of a similarity split, with parrot-preprocess's default criteria

    Parameters
    ----------
    method : str
        'kmer' or 'mmseqs'

    Returns
    -------
    SimilaritySearch
    """
    if method == 'mmseqs':
        return mmseqs_search.MMseqsSearch(mmseqs_search.HomologyCriteria())
    return kmer_similarity.KmerSearch(kmer_similarity.KmerCriteria())


def data_split_from_args(args: argparse.Namespace) -> DataSplit:
    """Decide how to divide the data from --split, --set-fractions and --save-splits

    A file named by --split is always used if it exists, even if its name is
    also a method's.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by training_parser.parse_arguments()

    Returns
    -------
    DataSplit

    Raises
    ------
    ValueError
        If --split is neither a file nor a method, or the options do not fit it
    RuntimeError
        If --split mmseqs is asked for but MMseqs2 cannot be found
    """
    value = args.split
    if os.path.isfile(value):
        if args.set_fractions is not None:
            raise ValueError(f'--set-fractions applies to --split {", ".join(SPLIT_METHODS)}; the split-file '
                             f'{value} already gives the sets')
        if args.save_splits:
            raise ValueError(f'--save-splits saves a split made by --split {", ".join(SPLIT_METHODS)}; {value} is '
                             'already a split-file')
        if value in SPLIT_METHODS and not args.silent:
            print(f"Note: --split {value} reads the split-file '{value}', since a file of that name exists")
        return DataSplit(split_file=os.path.abspath(value), method=None, fractions=None, silent=args.silent)

    if value not in SPLIT_METHODS:
        raise ValueError(f"--split {value}: there is no file of that name, and it is not one of "
                         f"{', '.join(SPLIT_METHODS)}")

    fractions = tuple(args.set_fractions) if args.set_fractions is not None else pid.DEFAULT_SET_FRACTIONS
    arguments.check_set_fractions(fractions)
    if value == 'mmseqs':
        mmseqs_search.find_mmseqs()  # fail before reading any data if MMseqs2 is missing
    return DataSplit(split_file=None, method=value, fractions=(fractions[0], fractions[1], fractions[2]),
                     silent=args.silent)
