"""
Embedding fragments within the full-length proteins they were cut from (--embedding-context).

A protein language model embeds each residue in the context of the whole sequence it is given, so the residues of
a peptide embedded on its own get different embeddings from the same residues embedded within the full-length
protein. Many datasets are libraries of fragments cut out of larger proteins (e.g. tiles across a proteome), and
for them it can be better to embed each full-length protein and cut each fragment's residues out of its embedding.

Which protein each fragment comes from is given explicitly, never guessed:

- the library is a FASTA file of the full-length proteins, each named by the first word of its header line;
- the map has one '<fragment ID> <full-length ID> [start]' line per fragment, where start is the 1-based position
  of the fragment's first residue in the full-length protein. Without a start, the fragment must occur exactly
  once in its protein. A full-length ID of '-' says the fragment has no context, and is embedded on its own.

Every problem (a fragment missing from the map, a protein missing from the library, a fragment that does not match
its protein, or one that occurs more than once in it without a start) stops the command before anything is
embedded. Embeddings are looked up by sequence, so a sequence must have the same context under every ID it has.

The embedding itself is done by PLMEncoder (parrot/plm.py), which keeps in-context embeddings in a file of their
own in the embedding cache.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from parrot.plm import PLM_RESIDUES, FragmentContext
from parrot.process_input_data import TEXT_ENCODING, read_tsv_raw
from parrot.sequence_encoder import MAX_SHOWN_SEQUENCE_LENGTH

# The full-length ID that says a fragment has no context, and is embedded on its own
NO_CONTEXT = '-'

# Every line of a FASTA file that starts with this names a new protein
FASTA_HEADER_START = '>'

# A map line holds a fragment ID and a full-length ID, and optionally the fragment's 1-based start
MAP_COLUMNS = (2, 3)

# Error messages list at most this many examples of a problem
MAX_LISTED_PROBLEMS = 5

# The table of where each fragment was placed is written next to the command's output, as <output>_embedded_context.tsv
CONTEXT_TABLE_SUFFIX = '_embedded_context.tsv'
CONTEXT_TABLE_COLUMNS = ('fragment_id', 'fragment_sequence', 'full_length_id', 'full_length_positions', 'notes')

# The notes column of the table: how each fragment was placed
NOTE_ON_ITS_OWN = f"no full-length protein given ('{NO_CONTEXT}'): embedded on its own"
NOTE_START_GIVEN = 'start given in the map'
NOTE_START_FOUND = 'start found by searching the full-length protein'


class ContextProblem(ValueError):
    """A fragment that cannot be placed in its full-length protein (see locate_fragment())."""


@dataclass(frozen=True)
class MapEntry:
    """One line of a context map

    Attributes
    ----------
    full_length_id : str
        The full-length protein the fragment was cut from, or NO_CONTEXT
    start : int or None
        The 1-based position of the fragment's first residue in that protein, if given
    """

    full_length_id: str
    start: int | None


@dataclass(frozen=True)
class ContextRow:
    """How one fragment (one line of the data) is embedded: a row of the table written next to the output

    Attributes
    ----------
    fragment_id : str
    fragment_sequence : str
    full_length_id : str
        The full-length protein it is embedded within, or NO_CONTEXT if it is embedded on its own
    start : int or None
        0-based position of its first residue in the full-length protein (None if embedded on its own)
    note : str
        How it was placed: NOTE_ON_ITS_OWN, NOTE_START_GIVEN or NOTE_START_FOUND
    """

    fragment_id: str
    fragment_sequence: str
    full_length_id: str
    start: int | None
    note: str

    def positions(self) -> str:
        """The residues of the full-length protein the fragment covers, 1-based and inclusive (e.g. '31-60'), or '-'."""
        if self.start is None:
            return NO_CONTEXT
        return f'{self.start + 1}-{self.start + len(self.fragment_sequence)}'


@dataclass
class ContextAssignment:
    """Where every fragment of a dataset is embedded

    Attributes
    ----------
    library_file, map_file : str
        The FASTA file of full-length proteins and the map, as given
    contexts : dict
        Fragment sequence -> its FragmentContext, or None for a fragment embedded on its own (what
        PLMEncoder.add_contexts() takes)
    rows : list of ContextRow
        One per fragment (line of the data), in the data's order
    unused_map_lines : int
        Lines of the map for IDs that are not in the data (e.g. lines parrot-preprocess removed)
    """

    library_file: str
    map_file: str
    contexts: dict[str, FragmentContext | None]
    rows: list[ContextRow]
    unused_map_lines: int

    def describe(self) -> str:
        """How many fragments are embedded within how many proteins, and how many on their own, in one sentence."""
        in_context = [row for row in self.rows if row.start is not None]
        proteins = {row.full_length_id for row in in_context}
        description = (f'{len(in_context):,} of the {len(self.rows):,} sequences are embedded within {len(proteins):,} '
                       f'full-length proteins from {self.library_file}, and {len(self.rows) - len(in_context):,} on '
                       'their own')
        if self.unused_map_lines:
            description += (f'; {self.unused_map_lines:,} lines of the map {self.map_file} are for IDs that are not in '
                            'the data')
        return description


# .................................................................
# Reading the library and the map

def read_library(path: str, wanted: set[str]) -> dict[str, str]:
    """Read the full-length proteins named in `wanted` from a FASTA file

    Each protein is named by the first word of its header line (e.g. 'sp|P04637|P53_HUMAN' for a UniProt
    entry). Sequence lines are joined as they are: nothing is changed to upper case or removed (a protein the
    language model cannot read is reported by assign_contexts()).

    Parameters
    ----------
    path : str
        The FASTA file
    wanted : set of str
        The proteins to keep (only these are held in memory, so the library can be a whole proteome)

    Returns
    -------
    dict
        Protein ID -> sequence, for every wanted ID in the file

    Raises
    ------
    ValueError
        If the file has no proteins, text before its first header line, a header line with no ID, two proteins
        with the same ID, or a wanted protein with no sequence
    """
    sequences: dict[str, list[str]] = {}
    seen: set[str] = set()
    current: list[str] | None = None

    with open(path, encoding=TEXT_ENCODING) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(FASTA_HEADER_START):
                words = line[len(FASTA_HEADER_START):].split()
                if not words:
                    raise ValueError(f'Line {line_number} of {path} is a FASTA header line with no protein ID')
                protein_id = words[0]
                if protein_id in seen:
                    raise ValueError(f'{path} has more than one protein called {protein_id} (the second is on line '
                                     f'{line_number}); every protein needs its own ID')
                seen.add(protein_id)
                current = sequences.setdefault(protein_id, []) if protein_id in wanted else None
            elif not seen:
                raise ValueError(f'{path} is not a FASTA file: line {line_number} comes before the first header line '
                                 f"('{FASTA_HEADER_START}ID ...')")
            elif current is not None:
                current.append(line)

    if not seen:
        raise ValueError(f'{path} holds no proteins: a FASTA file has a header line starting with '
                         f"'{FASTA_HEADER_START}' before each sequence")

    library = {protein_id: ''.join(lines) for protein_id, lines in sequences.items()}
    empty = [protein_id for protein_id, sequence in library.items() if not sequence]
    if empty:
        raise ValueError(f'{len(empty):,} proteins in {path} have no sequence, e.g. {", ".join(empty[:MAX_LISTED_PROBLEMS])}')
    return library


def read_context_map(path: str) -> dict[str, MapEntry]:
    """Read a context map: one '<fragment ID> <full-length ID> [start]' line per fragment

    Blank lines and lines starting with '#' (e.g. a header) are skipped, as in a datafile.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
        Fragment ID -> MapEntry

    Raises
    ------
    ValueError
        If the map is empty, a line has the wrong number of columns, a start is not a whole number of at least 1
        or is given for a fragment with no full-length protein, or a fragment ID is on more than one line
    """
    entries: dict[str, MapEntry] = {}
    for fields in read_tsv_raw(path):
        shown = ' '.join(fields)
        if len(fields) not in MAP_COLUMNS:
            raise ValueError(f"Lines of the map {path} should be '<sequence ID> <full-length ID> [start]', but found: "
                             f'{shown}')

        fragment_id, full_length_id = fields[0], fields[1]
        start = None
        if len(fields) == MAP_COLUMNS[-1]:
            if full_length_id == NO_CONTEXT:
                raise ValueError(f"A sequence with no full-length protein ('{NO_CONTEXT}') cannot have a start, but "
                                 f'the map {path} has: {shown}')
            start = parse_start(fields[2], shown, path)

        if fragment_id in entries:
            raise ValueError(f'The map {path} has more than one line for {fragment_id}: each sequence ID needs '
                             'exactly one')
        entries[fragment_id] = MapEntry(full_length_id, start)

    if not entries:
        raise ValueError(f'The map {path} has no lines')
    return entries


def parse_start(text: str, line: str, path: str) -> int:
    """A map line's start: a whole number of at least 1

    Parameters
    ----------
    text : str
        The start as written
    line : str
        The whole line, for the error message
    path : str
        The map, for the error message

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If it is not a whole number of at least 1
    """
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"A start in the map {path} is not a whole number of at least 1 (the 1-based position of the "
                         f"sequence's first residue in its full-length protein): {line}")
    return int(text)


# .................................................................
# Placing fragments

def shorten(sequence: str) -> str:
    """A sequence as shown in a message: cut to MAX_SHOWN_SEQUENCE_LENGTH characters if it is longer."""
    if len(sequence) <= MAX_SHOWN_SEQUENCE_LENGTH:
        return sequence
    return sequence[:MAX_SHOWN_SEQUENCE_LENGTH - 3] + '...'


def occurrences(fragment: str, protein: str) -> list[int]:
    """Every 0-based position at which `fragment` occurs in `protein`, overlapping occurrences included."""
    positions = []
    position = protein.find(fragment)
    while position != -1:
        positions.append(position)
        position = protein.find(fragment, position + 1)
    return positions


def locate_fragment(fragment: str, protein: str, start: int | None) -> int:
    """Where a fragment sits in its full-length protein

    Parameters
    ----------
    fragment : str
    protein : str
        The full-length protein's sequence
    start : int or None
        The 1-based position of the fragment's first residue, if known. Without it, the fragment must occur
        exactly once in the protein.

    Returns
    -------
    int
        The 0-based position of the fragment's first residue

    Raises
    ------
    ContextProblem
        If the fragment does not match the protein at `start`, or (without a start) does not occur in it or occurs
        more than once
    """
    if start is not None:
        index = start - 1
        end = index + len(fragment)
        if end > len(protein):
            raise ContextProblem(f'from residue {start}, it runs past the end of the full-length protein '
                                 f'({len(protein):,} residues)')
        if protein[index:end] != fragment:
            raise ContextProblem(f'it does not match the full-length protein at residues {start}-{end}, which hold '
                                 f'{shorten(protein[index:end])}')
        return index

    positions = occurrences(fragment, protein)
    if not positions:
        raise ContextProblem('it does not occur in the full-length protein')
    if len(positions) > 1:
        shown = ', '.join(str(position + 1) for position in positions[:MAX_LISTED_PROBLEMS])
        raise ContextProblem(f'it occurs {len(positions)} times in the full-length protein (starting at residues '
                             f'{shown}{", ..." if len(positions) > MAX_LISTED_PROBLEMS else ""}); give its start in '
                             "the map's third column")
    return positions[0]


def assign_contexts(ids: Sequence[str], sequences: Sequence[str], library_file: str,
                    map_file: str) -> ContextAssignment:
    """Find where every fragment of a dataset sits in its full-length protein

    Parameters
    ----------
    ids, sequences : list of str
        Each fragment's ID and sequence (one per line of the data)
    library_file : str
        FASTA file of the full-length proteins
    map_file : str
        The map from fragment IDs to full-length IDs (and starts)

    Returns
    -------
    ContextAssignment

    Raises
    ------
    ValueError
        If either file cannot be read (see read_library() and read_context_map()); if any fragment cannot be placed
        (its ID is not in the map, its protein is not in the library, or locate_fragment() fails); if a sequence
        has different contexts under different IDs; or if a full-length protein has a character a protein language
        model cannot read. Every fragment is checked first, and the message lists the first few of each problem.
    """
    context_map = read_context_map(map_file)
    wanted = {entry.full_length_id for entry in context_map.values()} - {NO_CONTEXT}
    library = read_library(library_file, wanted)

    rows = []
    problems = []
    for fragment_id, fragment in zip(ids, sequences, strict=True):
        try:
            rows.append(place_fragment(fragment_id, fragment, context_map, library, library_file))
        except ContextProblem as problem:
            problems.append(f'{fragment_id}: {problem}')
    if problems:
        raise ValueError(f'{len(problems):,} of the {len(rows) + len(problems):,} sequences cannot be placed in their '
                         f'full-length proteins (with --embedding-context, every sequence ID needs a line in the map '
                         f"{map_file}; give '{NO_CONTEXT}' as the full-length ID of one to embed on its own):\n"
                         + list_problems(problems))

    check_library_characters(rows, library, library_file)
    contexts = contexts_by_sequence(rows, library)
    unused = len(set(context_map) - set(ids))
    return ContextAssignment(library_file, map_file, contexts, rows, unused)


def place_fragment(fragment_id: str, fragment: str, context_map: dict[str, MapEntry], library: dict[str, str],
                   library_file: str) -> ContextRow:
    """Place one fragment in its full-length protein, as the map says

    Parameters
    ----------
    fragment_id, fragment : str
    context_map : dict
        Fragment ID -> MapEntry
    library : dict
        Full-length ID -> sequence
    library_file : str
        For the error message

    Returns
    -------
    ContextRow

    Raises
    ------
    ContextProblem
        If the ID is not in the map, its protein is not in the library, or locate_fragment() fails
    """
    entry = context_map.get(fragment_id)
    if entry is None:
        raise ContextProblem('it is not in the map')
    if entry.full_length_id == NO_CONTEXT:
        return ContextRow(fragment_id, fragment, NO_CONTEXT, None, NOTE_ON_ITS_OWN)

    protein = library.get(entry.full_length_id)
    if protein is None:
        raise ContextProblem(f'its full-length protein {entry.full_length_id} is not in {library_file} (proteins are '
                             'named by the first word of their FASTA header line)')
    try:
        start = locate_fragment(fragment, protein, entry.start)
    except ContextProblem as problem:
        raise ContextProblem(f'in {entry.full_length_id}, {problem}') from None
    note = NOTE_START_GIVEN if entry.start is not None else NOTE_START_FOUND
    return ContextRow(fragment_id, fragment, entry.full_length_id, start, note)


def check_library_characters(rows: list[ContextRow], library: dict[str, str], library_file: str) -> None:
    """Raise if a full-length protein that will be embedded has a character the language model cannot read as a residue

    Parameters
    ----------
    rows : list of ContextRow
    library : dict
        Full-length ID -> sequence
    library_file : str
        For the error message

    Raises
    ------
    ValueError
        Listing the first few such proteins
    """
    valid = set(PLM_RESIDUES)
    problems = []
    for protein_id in dict.fromkeys(row.full_length_id for row in rows if row.start is not None):
        invalid = sorted(set(library[protein_id]) - valid)
        if invalid:
            problems.append(f'{protein_id}: {" ".join(repr(character) for character in invalid)}')
    if problems:
        raise ValueError(f'{len(problems):,} full-length proteins in {library_file} have characters a protein language '
                         'model cannot read as residues (sequences must be upper case, with no gaps or "*" marking '
                         'a stop codon):\n' + list_problems(problems))


def contexts_by_sequence(rows: list[ContextRow], library: dict[str, str]) -> dict[str, FragmentContext | None]:
    """Each fragment sequence's context, checking a sequence has the same one under every ID

    Contexts are compared by the full-length sequence and the start, so two IDs of identical proteins give the
    same context.

    Parameters
    ----------
    rows : list of ContextRow
    library : dict
        Full-length ID -> sequence

    Returns
    -------
    dict
        Fragment sequence -> FragmentContext, or None for one embedded on its own

    Raises
    ------
    ValueError
        If a sequence has different contexts under different IDs (listing the first few)
    """
    contexts: dict[str, FragmentContext | None] = {}
    first_row: dict[str, ContextRow] = {}
    problems = []
    for row in rows:
        context = None if row.start is None else FragmentContext(library[row.full_length_id], row.start)
        if row.fragment_sequence not in contexts:
            contexts[row.fragment_sequence] = context
            first_row[row.fragment_sequence] = row
        elif contexts[row.fragment_sequence] != context:
            first = first_row[row.fragment_sequence]
            problems.append(f'{shorten(row.fragment_sequence)}: {first.fragment_id} is in {first.full_length_id} at '
                            f'{first.positions()}, but {row.fragment_id} is in {row.full_length_id} at '
                            f'{row.positions()}')
    if problems:
        raise ValueError(f'{len(problems):,} sequences have different full-length contexts under different IDs. '
                         'Embeddings are looked up by sequence, so a sequence needs the same context under every ID '
                         '(parrot-preprocess --deduplicate merges or drops repeated sequences):\n'
                         + list_problems(problems))
    return contexts


def list_problems(problems: list[str]) -> str:
    """The first MAX_LISTED_PROBLEMS problems, one per indented line, and how many more there are."""
    listed = '\n'.join(f'  {problem}' for problem in problems[:MAX_LISTED_PROBLEMS])
    if len(problems) > MAX_LISTED_PROBLEMS:
        listed += f'\n  ... and {len(problems) - MAX_LISTED_PROBLEMS:,} more'
    return listed


# .................................................................
# The table written next to the output

def write_context_table(path: str, rows: list[ContextRow]) -> None:
    """Write where each fragment was embedded: one tab-separated line per fragment, after a header line

    The columns are the fragment's ID and sequence, the full-length protein's ID ('-' for a fragment embedded on
    its own), the residues of the full-length protein the fragment covers (1-based and inclusive, e.g. '31-60',
    or '-') and a note on how it was placed.

    Parameters
    ----------
    path : str
    rows : list of ContextRow
    """
    with open(path, 'w') as handle:
        handle.write('\t'.join(CONTEXT_TABLE_COLUMNS) + '\n')
        for row in rows:
            handle.write('\t'.join((row.fragment_id, row.fragment_sequence, row.full_length_id, row.positions(),
                                    row.note)) + '\n')
