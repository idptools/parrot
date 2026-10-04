"""
Finding homologous sequences with MMseqs2 (Steinegger & Söding 2017, Nat
Biotechnol 35:1026), the most sensitive search engine for splitting a dataset
without leakage (see splitting.py). MMseqs2 must be installed separately.

Two sequences are homologous if any MMseqs2 alignment between them has an
E-value of at most HomologyCriteria.max_evalue (1e-3) and covers at least
HomologyCriteria.min_coverage (30%) of the shorter sequence. No
sequence-identity threshold is used, so any statistically significant alignment
counts, and searches use MMseqs2's most sensitive setting.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from parrot.tools.splitting import Match, identity_summary

INSTALL_HELP = ('MMseqs2 is needed for --split homology but the `mmseqs` command was not found. Install it with '
                '`conda install -c conda-forge -c bioconda mmseqs2` or `brew install mmseqs2`, or download a static '
                'binary from https://github.com/soedinglab/MMseqs2/releases, then put it on the PATH or pass its '
                'location with --mmseqs. Alternatively, --split kmer needs nothing extra but only finds closer '
                'relatives.')

# How much of MMseqs2's error output is shown when it fails
MAX_ERROR_CHARACTERS = 2000


@dataclass(frozen=True)
class HomologyCriteria:
    """When two sequences count as homologous, and how hard MMseqs2 searches

    Attributes
    ----------
    max_evalue : float
        Largest E-value of an alignment that links two sequences
    min_coverage : float
        Smallest fraction of the shorter sequence the alignment must cover
    sensitivity : float
        MMseqs2 search sensitivity (-s); 7.5 is the most sensitive setting
    max_target_sequences : int
        Targets per query passing MMseqs2's prefilter (--max-seqs; default 300
        in MMseqs2), raised so large families are not truncated
    report_max_evalue : float
        The separation check also reports every weaker alignment up to this
        E-value, so any remaining similarity between sets is visible
    mask_low_complexity : bool
        Whether MMseqs2 masks low-complexity regions (its default). Masking
        avoids alignments driven only by compositional bias; without it,
        similarity in low-complexity regions (common in disordered proteins)
        also links sequences, which is even more stringent.
    """

    max_evalue: float = 1e-3
    min_coverage: float = 0.3
    sensitivity: float = 7.5
    max_target_sequences: int = 2000
    report_max_evalue: float = 1e-2
    mask_low_complexity: bool = True

    def describe(self) -> str:
        """The criteria in words, for reports."""
        return (f'identical, or an MMseqs2 alignment with E <= {self.max_evalue:g} covering >= '
                f'{100 * self.min_coverage:g}% of the shorter sequence (no identity threshold; sensitivity '
                f'{self.sensitivity:g}; up to {self.max_target_sequences:,} targets per query; low-complexity '
                f'masking {"on" if self.mask_low_complexity else "off"})')


@dataclass(frozen=True)
class Alignment:
    """One MMseqs2 alignment between two sequences (indices refer to the query and target lists).

    linked is True if the alignment meets the homology criteria of the search that found it.
    """

    query: int
    target: int
    identity: float
    alignment_length: int
    query_length: int
    target_length: int
    evalue: float
    linked: bool = False

    @property
    def shorter_coverage(self) -> float:
        """Fraction of the shorter of the two sequences covered by the alignment."""
        return self.alignment_length / min(self.query_length, self.target_length)

    def homologous(self, criteria: HomologyCriteria) -> bool:
        """True if the alignment meets the homology criteria."""
        return self.evalue <= criteria.max_evalue and self.shorter_coverage >= criteria.min_coverage


# .................................................................
# Running MMseqs2

def find_mmseqs(path: str | None = None) -> str:
    """The MMseqs2 executable to run

    Parameters
    ----------
    path : str, optional
        Location of the executable; by default `mmseqs` is looked up on the PATH

    Returns
    -------
    str
        Full path of the executable

    Raises
    ------
    RuntimeError
        If it cannot be found, with instructions for installing it
    """
    found = shutil.which(path or 'mmseqs')
    if found is None:
        raise RuntimeError(INSTALL_HELP if path is None else f'MMseqs2 was not found at {path}. ' + INSTALL_HELP)
    return found


def mmseqs_version(executable: str) -> str:
    """The version (or commit) MMseqs2 reports, for the record.

    Parameters
    ----------
    executable : str

    Returns
    -------
    str
    """
    result = subprocess.run([executable, 'version'], capture_output=True, text=True)
    return result.stdout.strip() or 'unknown'


class MMseqsSearch:
    """Finds homologous sequences with MMseqs2 (see splitting.SimilaritySearch)

    Attributes
    ----------
    executable : str
        The MMseqs2 executable
    criteria : HomologyCriteria
    threads : int
    workdir : Path or None
        Where MMseqs2's temporary files go (default: the system's temporary directory)
    """

    def __init__(self, criteria: HomologyCriteria | None = None, executable: str | None = None,
                 threads: int | None = None, workdir: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        criteria : HomologyCriteria, optional
            When two sequences count as homologous (the defaults if not given)
        executable : str, optional
            The MMseqs2 executable (default: mmseqs on the PATH)
        threads : int, optional
            Threads for MMseqs2 (default: every CPU)
        workdir : str or Path, optional
            Where MMseqs2's temporary files go (default: the system's temporary directory)
        """
        self.executable = find_mmseqs(executable)
        self.criteria = criteria or HomologyCriteria()
        self.threads = threads or os.cpu_count() or 1
        self.workdir = None if workdir is None else Path(workdir)

    def describe(self) -> str:
        """The criterion and the MMseqs2 used, in words."""
        return (f'{self.criteria.describe()}; MMseqs2 {self.executable} (version {mmseqs_version(self.executable)}), '
                f'{self.threads} threads')

    def summarise(self, matches: Sequence[Match]) -> str:
        """A short description of the strength of the alignments found (for the report)."""
        return f'at E <= {self.criteria.report_max_evalue:g}; {identity_summary(m.identity for m in matches)}'

    def __call__(self, queries: Sequence[str], targets: Sequence[str] | None = None,
                 weaker: bool = False) -> list[Alignment]:
        """Search queries against targets (or, with targets=None, against each other, without self-hits)

        Parameters
        ----------
        queries : list of str
        targets : list of str, optional
        weaker : bool, optional
            Also return alignments up to report_max_evalue that do not meet the
            criteria (linked=False), for the report

        Returns
        -------
        list of Alignment

        Raises
        ------
        RuntimeError
            If MMseqs2 fails
        """
        max_evalue = self.criteria.report_max_evalue if weaker else self.criteria.max_evalue
        with tempfile.TemporaryDirectory(prefix='parrot_mmseqs_', dir=self.workdir) as directory:
            workdir = Path(directory)
            query_fasta = workdir / 'queries.fasta'
            write_fasta(queries, query_fasta)
            target_fasta = query_fasta
            if targets is not None:
                target_fasta = workdir / 'targets.fasta'
                write_fasta(targets, target_fasta)

            results = workdir / 'results.m8'
            command = [self.executable, 'easy-search', str(query_fasta), str(target_fasta), str(results),
                       str(workdir / 'tmp'), '-s', str(self.criteria.sensitivity), '-e', str(max_evalue),
                       '--max-seqs', str(self.criteria.max_target_sequences),
                       '--mask', '1' if self.criteria.mask_low_complexity else '0',
                       '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue',
                       '--threads', str(self.threads), '-v', '1']
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f'MMseqs2 failed (exit code {completed.returncode}):\n'
                                   f'{completed.stderr[-MAX_ERROR_CHARACTERS:]}')

            alignments = []
            with open(results) as handle:
                for line in handle:
                    query, target, identity, length, query_length, target_length, evalue = line.split()
                    if targets is None and query == target:
                        continue
                    alignment = Alignment(int(query), int(target), float(identity), int(length), int(query_length),
                                          int(target_length), float(evalue))
                    alignments.append(replace(alignment, linked=alignment.homologous(self.criteria)))
        return alignments


def write_fasta(sequences: Sequence[str], path: Path) -> None:
    """Write sequences to a FASTA file, named by their index (0, 1, ...).

    Parameters
    ----------
    sequences : list of str
    path : Path
    """
    with open(path, 'w') as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f'>{index}\n{sequence}\n')
