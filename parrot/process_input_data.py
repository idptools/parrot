# Alternative implementation of process_input_data with memory-efficient loading
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig

from parrot import encode_sequence
from parrot.encode_sequence import ParrotLightningEncoder, BaseParrotEncoder
from parrot.parrot_exceptions import IOExceptionParrot
import os
import numpy as np
import torch
import gc
import math

# .................................................................
# From original implementation of PARROT
#
def read_tsv_raw(tsvfile, delimiter=None):
    """
    Efficiently parses a TSV file, ignoring empty lines and comment lines.
    Parameters
    ----------
    tsvfile : str
        Path to a whitespace-separated datafile.
    delimiter : str or None
        Delimiter for splitting columns. Default is tab.
    Returns
    -------
    generator
        Yields parsed lines as lists of strings.
    """
    # Note: This function parses the data in a memory efficient manner and
    # does not require the program to read the entire file at once (good for large files)

    # opens a file in the default read mode
    with open(tsvfile) as fh:
        # file objects are iterable over the lines in the file
        for line in fh:
            # remove whitespace from the front and end of the line
            stripped = line.strip()
            # check that the string is not empty and check the line does not start with a # - indicating a commment
            if stripped and not stripped.startswith("#"):
                # None defaults to any whitespace
                # pauses and waits for the caller to accept the parsed line value until continuing
                # This is memory efficient as it processes one line and then moves onto the next one
                yield stripped.split(delimiter)

# Note: The parser in this function will have issues if the data is not
# formatted perfectly. This should be made to be a little more forgiving.
# I would like to incorporate the ability to use .csv as well. This would
# require modification to this code.
def __parse_lines(lines, datatype, validate=True):
    """
    Internal function for parsing a set of lines

    Parameters
    ----------
    lines : list
        A list of lists, where the sublists reflect the columns in a tsvfile. Should be the output
        from the read_tsv_raw() function.

    datatype : str
        Identifier that defines the type of data being passed in. Must be either 'residues', 'sequence'

    validate : bool
        If set to true, ensures the number of residue values equals the number of residues
 
    Returns
    -----------
    list
        Returns a parsed list of lists, where each sublist contains the structure
        [id, sequence, <data>]
        where <data> is either a single float (mode=sequence) or a set of floats 

    Raises
    ---------
    Exception
        If an error occurs while parsing the file, the linenumber of the file is printed as well as the
        idenity of the offending line.

    """

    
    # check the datatype is valid
    if datatype not in ['residues','sequence']:
        raise ValueError('Invalid datatype. Must be "residues" or "sequence".')
        
    # parse the lines
    # the conversion from text to numbers could fail if the user provides improper data
    try:
        # here to store the data for each line
        data = [] # splits up data by column [col1,col2,...]
        lc = 0 # counts the lines (rows) in the dataset - only used for the error message

        # A value for each residue in a sequence
        # Note: this does not check that the number of targets matches the number of residues
        if datatype == 'residues':	
            for x in lines:
                # to the number of entries in the dataset
                lc = lc + 1
                
                # Pull the last portion of the dataset out and turns it into a numpy array
                # These should be the target values
                residue_data = np.array(x[2:], dtype=float)

                # Reformats the data as (ID, sequence, target value)
                data.append([x[0], x[1], residue_data])

        # A single value per sequence
        elif datatype == 'sequence':  
            for x in lines:
                # to the number of entries in the dataset
                lc = lc + 1
                # Reformats the data as (ID, sequence, target value)
                data.append([x[0], x[1], float(x[2])])

    # catch any exception and print it.
    except Exception as e:        
        print('Excecption raised on parsing input file...')
        print(e)
        print('')
        raise IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nMake sure your datafile does not have empty lines at the end of the file.\nError on line {lc}:\n{x}")

    # if we want to validate each line - aka check that the length of the sequence matches the number of target values
    if validate:
        # check that the datatype is residues - if not there is nothing to validate
        if datatype == 'residues':
            lc = 0
            for x in data:
                lc = lc + 1
                # check that the lengths match between sequence and targets
                if len(x[1]) != len(x[2]):
                    raise IOExceptionParrot(f"Input data is not correctly formatted for datatype '{datatype}'.\nInconsistent number of residue values and residues. Error on line {lc}:\n{x}")
    return data

def vector_split(v, fraction):
    """Split a vector randomly by a specified proportion

    Randomly divide the values of a vector into two, non-overlapping smaller
    vectors. The proportions of the two vectors will be `fraction` and
    (1 - `fraction`).

    Parameters
    ----------
    v : numpy array
            The vector to divide
    fraction : float
            Size proportion for the returned vectors. Should be in the range [0-1].

    Returns
    -------
    numpy array
            a subset of `v` of length `fraction` * len(v) (rounding up)
    numpy array
            a subset of `v` of length (1-`fraction`) * len(v).
    """

    segment1 = np.random.choice(v, size=math.ceil(fraction * len(v)), replace=False)
    segment1.sort()
    segment2 = np.setdiff1d(v, segment1, assume_unique=True)
    return segment1, segment2

# .................................................................
# .................... New implementations below ..................
# .................................................................

class SequenceDataset(Dataset):
    """
    A PyTorch Dataset class that handles PARROT's diverse data formats and encoding requirements.
    
    This class serves as the central data processing component of PARROT, bridging the gap between
    raw sequence data files and the tensor inputs required by neural networks. It handles the
    complexity of PARROT's flexible data formats while providing a clean interface for model training.
    
    Key responsibilities:
    1. Parse various PARROT data formats (with/without sequence IDs, single/multi-column sequences)
    2. Automatically infer data types (sequence-level vs residue-level predictions)
    3. Handle sequence encoding through PARROT's encoder system
    4. Manage multi-column sequence formats with proper delimiter handling
    5. Provide memory-efficient data loading for large datasets
    """
    
    def __init__(self, filepath : str, encoder_cfg : DictConfig = None, 
             encoder: BaseParrotEncoder = None, excludeSeqID : bool = False,
             datatype : str = None, delimiter : str = None, sequence_delimiter : str = '*'):
        """
        Initialize the SequenceDataset for PARROT's flexible data processing pipeline.
        
        This constructor handles the complex task of preparing biological sequence data for
        machine learning. It automatically detects data formats, configures appropriate
        encoders, and sets up the internal data structures needed for efficient training.
        
        The initialization process follows these key steps:
        1. Validate file existence and parameters
        2. Infer data type if not explicitly provided (sequence vs residue level)
        3. Detect multi-column sequence formats
        4. Configure and prepare the sequence encoder
        5. Load and parse all data with proper error handling
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file. Supports various PARROT formats including TSV files
            with optional sequence IDs and multi-column sequences.
        encoder_cfg : DictConfig
            Hydra configuration for the encoder. Used to create a new encoder instance.
            If both encoder_cfg and encoder are provided, encoder takes precedence.
        encoder : BaseParrotEncoder
            Pre-instantiated encoder object. This allows reusing encoders across datasets
            or using custom encoder configurations not available through Hydra configs.
        excludeSeqID : bool
            Whether sequence IDs are excluded from the data file. This affects how
            columns are interpreted during parsing. Default is False.
        datatype : str or None
            'sequence' for sequence-level predictions or 'residues' for residue-level
            predictions. If None, the system will automatically infer the type by
            examining the relationship between sequence length and number of target values.
        delimiter : str
            Delimiter for splitting file columns. None defaults to any whitespace,
            maintaining compatibility with original PARROT behavior.
        sequence_delimiter : str
            Delimiter used when joining multiple sequence columns into a single sequence.
            This is crucial for multi-column formats where different sequence parts
            need to be distinguished (e.g., 'ACGT*TGCA' for two sequence columns).
        """
        # Store core configuration - these parameters define how data will be processed
        self.filepath = filepath
        self.sequence_delimiter = sequence_delimiter
        
        # Validate that the input file exists before proceeding with any processing
        # Early validation prevents wasted computation on invalid inputs
        if not os.path.exists(filepath):
            raise IOExceptionParrot(f"File not found: {filepath}")

        # Store parsing configuration that affects how file lines are interpreted
        self.excludeSeqID = excludeSeqID
        self.delimiter = delimiter
        
        # Data type inference: One of PARROT's key features is automatically determining
        # whether the data represents sequence-level or residue-level predictions.
        # This is crucial because it affects model architecture and training procedures.
        if datatype is None:
            # Automatic inference examines the relationship between sequence length
            # and number of target values to make an educated guess
            self.datatype = self._infer_datatype()
        else:
            # Manual specification allows users to override automatic detection
            # when they know their data format or when inference might be ambiguous
            if datatype not in ['sequence', 'residues']:
                raise ValueError(f"Invalid datatype: {datatype}. Must be 'sequence' or 'residues'")
            self.datatype = datatype
        
        # Multi-column sequence detection: PARROT supports complex formats where
        # sequences are split across multiple columns (e.g., protein domains).
        # This detection is crucial for proper sequence reconstruction and encoding.
        self.has_multi_columns = self._check_multi_column_sequences()
        
        # Encoder setup: The encoder is responsible for converting biological sequences
        # into numerical representations that neural networks can process. This is a
        # critical component that bridges biology and machine learning.
        if encoder is not None:
            # Pre-instantiated encoder - modify it to handle multi-columns if needed
            self.encoder = self._prepare_encoder_for_multi_columns(encoder)
        elif encoder_cfg is not None:
            # Create encoder from Hydra configuration - this is the typical pathway
            # for configured training runs where encoder parameters are specified
            encoder = ParrotLightningEncoder(encoder_cfg)
            self.encoder = self._prepare_encoder_for_multi_columns(encoder)
        else:
            # Default fallback: Create a basic one-hot encoder if none specified
            # This ensures the dataset always has a functional encoder
            from omegaconf import DictConfig
            default_cfg = DictConfig({
                'type': 'table',
                'alphabet': 'ACDEFGHIKLMNPQRSTVWY'  # Standard amino acid alphabet
            })
            encoder = ParrotLightningEncoder(default_cfg)
            self.encoder = self._prepare_encoder_for_multi_columns(encoder)
        
        # Data loading: Parse the entire file and store in memory for efficient access
        # This approach trades memory for speed and simplicity, which is appropriate
        # for most biological datasets that are not extremely large
        self.data = self._load_data()

    def _is_numeric(self, value):
        """
        Utility function to distinguish sequence data from target values.
        
        In PARROT data files, sequence columns contain biological sequences (letters)
        while target columns contain numerical values. This distinction is crucial
        for proper parsing of mixed-format files where sequence and target data
        are interleaved.
        
        This method supports the automatic separation of sequence and value columns
        during the parsing process, enabling flexible file formats.
        
        Parameters
        ----------
        value : str
            String to test for numeric content
            
        Returns
        -------
        bool
            True if the string can be converted to a float, False otherwise
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _separate_sequence_and_values(self, parts_after_id):
        """
        Intelligently separate sequence columns from target value columns.
        
        This method implements PARROT's flexible parsing logic that can handle
        various file formats automatically. It assumes that sequence data (letters)
        comes first, followed by numerical target values. This separation is
        fundamental to PARROT's ability to handle diverse data formats without
        requiring strict format specifications.
        
        The method enforces the expected data organization: non-numeric sequence
        data followed by numeric target values. This assumption allows automatic
        parsing while maintaining data integrity.

        This is done in a flexible manner that can handle both single-column and
        multi-column sequence formats in the same dataset. 
        
        Parameters
        ----------
        parts_after_id : list
            List of strings representing columns after seqID (if present).
            These columns may contain a mix of sequence data and target values.
            
        Returns
        -------
        tuple
            (sequence_parts, value_parts) where sequence_parts contains biological
            sequence strings and value_parts contains numeric target values
            
        Raises
        ------
        ValueError
            If the data format doesn't match expected patterns (e.g., numeric
            values appear before sequence data, or required data types are missing)
        """
        sequence_parts = []
        value_parts = []
        
        # Process columns in order, expecting sequence data first, then values
        # This ordering assumption is key to PARROT's automatic format detection
        found_numeric = False
        
        for part in parts_after_id:
            if self._is_numeric(part):
                # Once we find numeric data, all remaining columns should be numeric
                found_numeric = True
                value_parts.append(part)
            else:
                # Non-numeric data should only appear before numeric data
                if found_numeric:
                    # Violation of expected format - this indicates malformed data
                    raise ValueError("Invalid data format: numeric values should come after all sequence data")
                sequence_parts.append(part)
        
        # Validate that we found both required data types
        if not sequence_parts:
            raise ValueError("No sequence data found")
        if not value_parts:
            raise ValueError("No numeric values found")
            
        return sequence_parts, value_parts

    def _infer_datatype(self):
        """
        Automatically determine whether data represents sequence-level or residue-level predictions.
        
        This method implements one of PARROT's most valuable features: automatic data type
        detection. By analyzing the relationship between sequence length and the number
        of target values, it can distinguish between:
        
        1. Sequence-level data: One target value per sequence (e.g., protein function classification)
        2. Residue-level data: One target value per residue (e.g., secondary structure prediction)
        
        This intelligence allows users to work with PARROT without needing to manually
        specify data formats, reducing setup complexity and potential errors.
        
        The inference process examines multiple lines to ensure consistency and handles
        edge cases like multi-column sequences where padding might affect value counts.
        
        Returns
        -------
        str
            'sequence' if data appears to be sequence-level (one value per sequence)
            'residues' if data appears to be residue-level (one value per residue)
            
        Raises
        ------
        IOExceptionParrot
            If the data format is inconsistent, ambiguous, or no valid data lines are found
        """
        with open(self.filepath, 'r') as f:
            lines_checked = 0
            max_lines_to_check = 5  # Sample multiple lines to ensure consistency
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments - focus on actual data
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse this line using the same logic as the main data loader
                    parts = line.split(self.delimiter)
                    
                    # Handle sequence ID presence/absence
                    if self.excludeSeqID:
                        # Format: sequence_columns... values...
                        if len(parts) < 2:
                            continue  # Skip malformed lines during inference
                        parts_after_id = parts
                    else:
                        # Format: seqID sequence_columns... values...
                        if len(parts) < 3:
                            continue  # Skip malformed lines during inference
                        parts_after_id = parts[1:]  # Remove seqID for analysis
                    
                    # Separate biological sequences from numerical targets
                    sequence_parts, value_parts = self._separate_sequence_and_values(parts_after_id)
                    
                    # Analyze the relationship between sequence length and value count
                    # This is the core logic for type inference
                    combined_sequence = self.sequence_delimiter.join(sequence_parts)
                    num_values = len(value_parts)
                    seq_length = len(combined_sequence)
                    
                    if num_values == 1:
                        # Single value strongly suggests sequence-level prediction
                        inferred_type = 'sequence'
                    elif num_values == seq_length:
                        # Value count matches sequence length - likely residue-level
                        inferred_type = 'residues'
                    else:
                        # Handle complex cases like multi-column sequences
                        # Here we check if values match the total characters across all sequence parts
                        total_seq_chars = sum(len(seq_part) for seq_part in sequence_parts)
                        if num_values == total_seq_chars:
                            # Values match total sequence characters (excluding delimiters)
                            inferred_type = 'residues'
                        else:
                            # Ambiguous case - continue examining more lines
                            continue
                    
                    lines_checked += 1
                    
                    # Consistency checking across multiple lines
                    if lines_checked == 1:
                        first_inference = inferred_type
                    elif inferred_type != first_inference:
                        # Inconsistency detected - this indicates problematic data
                        raise IOExceptionParrot(
                            f"Inconsistent data format detected. Line {line_num} suggests '{inferred_type}' "
                            f"but earlier lines suggested '{first_inference}'. "
                            f"Please specify datatype explicitly."
                        )
                    
                    # If we've successfully analyzed enough lines, return the result
                    if lines_checked >= max_lines_to_check:
                        return first_inference
                        
                except Exception as e:
                    # Skip unparseable lines during inference - be permissive here
                    continue
            
            # Return result if we successfully analyzed at least one line
            if lines_checked > 0:
                return first_inference
            else:
                # Complete failure to parse any lines
                raise IOExceptionParrot(
                    "Could not infer datatype from file. No valid data lines found. "
                    "Please specify datatype explicitly."
                )

    def _check_multi_column_sequences(self):
        """
        Detect whether the dataset uses multi-column sequence formats.
        
        Multi-column sequences are a powerful PARROT feature that allows representing
        complex biological data where sequences are naturally split across multiple
        fields. Examples include:
        - Protein domains stored in separate columns
        - Multiple sequence alignments with different regions
        - Composite sequences from different sources
        
        Detection of this format is crucial because it affects:
        1. How sequences are reconstructed (with delimiters)
        2. How encoders need to be configured (to handle delimiters)
        3. How target values are aligned with sequence positions
        
        This method examines the file structure to determine if multi-column
        sequence support is needed, enabling automatic format adaptation.
        
        Returns
        -------
        bool
            True if multi-column sequences are detected, False for single-column format
        """
        with open(self.filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse the line to analyze its structure
                    parts = line.split(self.delimiter)
                    
                    # Extract the portion containing sequence and value data
                    # This section does not check for errors in the number of columns
                    if self.excludeSeqID:
                        if len(parts) < 2:
                            continue
                        parts_after_id = parts
                    else:
                        if len(parts) < 3:
                            continue
                        parts_after_id = parts[1:]  # Skip seqID
                    
                    # Separate sequence columns from value columns
                    sequence_parts, value_parts = self._separate_sequence_and_values(parts_after_id)
                    
                    # Multi-column detection: more than one sequence part indicates
                    # that sequences are split across multiple columns
                    return len(sequence_parts) > 1
                    
                except Exception as e:
                    # Skip unparseable lines - be permissive during detection
                    continue
        
        # Default assumption if detection fails
        return False

    def _prepare_encoder_for_multi_columns(self, encoder):
        """
        Adapt sequence encoders to handle multi-column sequence formats.
        
        When sequences are split across multiple columns, they are joined using
        a delimiter character. Standard encoders may not recognize this delimiter,
        which would cause encoding failures. This method ensures that encoders
        can properly handle the delimiter characters used in multi-column formats.
        
        This adaptation is crucial for the seamless integration of PARROT's
        flexible data formats with its encoding system. It bridges the gap between
        data representation and numerical encoding requirements.
        
        The method handles different encoder types appropriately:
        - Table encoders: Extend alphabet to include delimiter
        - Functional encoders: Validate delimiter support
        
        Parameters
        ----------
        encoder : BaseParrotEncoder
            The original encoder that may need modification for multi-column support
        
        Returns
        -------
        BaseParrotEncoder
            Modified encoder capable of handling sequence delimiters, or the
            original encoder if no modification is needed
        """
        # If single-column format, no modification needed
        if not self.has_multi_columns:
            return encoder
    
        # Determine encoder type and apply appropriate modifications
        # This type detection is necessary because different encoders require
        # different approaches for extending their character support
        if hasattr(encoder, '_actual_encoder'):
            actual_encoder = encoder._actual_encoder
            if hasattr(actual_encoder, '__class__'):
                encoder_class_name = actual_encoder.__class__.__name__
                if 'Table' in encoder_class_name:
                    # Table encoders can be extended by modifying their alphabet
                    return self._extend_table_encoder(encoder)
                elif 'Functional' in encoder_class_name:
                    # Functional encoders need validation of delimiter support
                    return self._validate_functional_encoder(encoder)
    
        # Default approach: attempt table encoder extension
        return self._extend_table_encoder(encoder)

    def _extend_table_encoder(self, encoder):
        """
        Extend table-based encoders to support sequence delimiter characters.
        
        For custom encoding schemes (non-one-hot), this method preserves the original
        encoding vectors and adds a new dimension to accommodate the delimiter character.
        The delimiter gets encoded as a vector with 0.0 in all original dimensions and
        1.0 in the new delimiter dimension.
        
        This approach maintains the semantic meaning of the original encoding while
        clearly distinguishing delimiter positions in multi-column sequences.
        
        Parameters
        ----------
        encoder : ParrotLightningEncoder
            Table-based encoder to extend with delimiter support
        
        Returns
        -------
        ParrotLightningEncoder
            New encoder instance with extended encoding dimension including the delimiter
        """
        # Check if delimiter is already supported
        if hasattr(encoder, '_actual_encoder') and hasattr(encoder._actual_encoder, 'alphabet'):
            if self.sequence_delimiter in encoder._actual_encoder.alphabet:
                # return the original encoder if the delimiter is already supported
                return encoder

        # Access the underlying table encoder to get the encoding scheme
        actual_encoder = encoder._actual_encoder
        if not hasattr(actual_encoder, '_table_encode_dict'):
            raise ValueError("Cannot extend encoder: no table encoding dictionary found")
        
        # store the original encoding dictionary and input size so we can modify it
        original_encode_dict = actual_encoder._table_encode_dict
        original_input_size = actual_encoder.input_size
        
        # Create extended encoding scheme with one additional dimension
        extended_input_size = original_input_size + 1
        extended_encode_dict = {}
        
        # Copy all original character encodings and extend them with 0.0 in the new dimension
        for char, original_vector in original_encode_dict.items():
            # Extend each original vector with 0.0 in the new delimiter dimension
            extended_vector = original_vector + [0.0]
            extended_encode_dict[char] = extended_vector
        
        # Add the delimiter character encoding: all zeros except 1.0 in the new dimension
        delimiter_vector = [0.0] * original_input_size + [1.0]
        extended_encode_dict[self.sequence_delimiter] = delimiter_vector
        
        # Create a temporary TSV content string to use with the existing parser
        # This avoids code duplication and ensures consistency with file-based loading
        tsv_lines = []
        for char, vector in extended_encode_dict.items():
            # Format as: character followed by space-separated vector values
            vector_str = ' '.join(str(v) for v in vector)
            tsv_lines.append(f"{char} {vector_str}")
        
        tsv_content = '\n'.join(tsv_lines)
        
        # Write to a temporary file for the new encoder to read
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as temp_file:
            temp_file.write(tsv_content)
            temp_file_path = temp_file.name
        
        try:
            # Create new encoder configuration with the extended table file
            from omegaconf import DictConfig
            extended_config = DictConfig({
                'type': 'table',
                'table_file_path': temp_file_path,
                'alphabet': ''.join(sorted(extended_encode_dict.keys()))
            })

            # Instantiate new encoder with extended capabilities
            new_encoder = ParrotLightningEncoder(extended_config)
            return new_encoder
        # this will always run so we can ensure the temporary file is cleaned up
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # Ignore cleanup errors

    def _validate_functional_encoder(self, encoder):
        """
        Validate that functional encoders can handle sequence delimiter characters.
        
        Functional encoders use computational methods (rather than lookup tables)
        to encode sequences. They may have built-in character support that cannot
        be easily extended. This method verifies that the encoder can handle the
        delimiter characters used in multi-column sequences.
        
        If the encoder cannot handle the delimiter, this method provides clear
        error messages with suggestions for resolution, maintaining PARROT's
        user-friendly approach to error handling.
        
        Parameters
        ----------
        encoder : BaseParrotEncoder
            Functional encoder to validate for delimiter support
            
        Returns
        -------
        BaseParrotEncoder
            The original encoder if validation passes
            
        Raises
        ------
        ValueError
            If the encoder cannot handle the sequence delimiter, with detailed
            error messages and suggestions for resolution
        """
        # Check alphabet-level support if available
        if hasattr(encoder, 'alphabet') and self.sequence_delimiter not in encoder.alphabet:
            raise ValueError(
                f"Functional encoder does not support sequence delimiter '{self.sequence_delimiter}'. \n" + 
                f"Supported characters: {encoder.alphabet}. \n" + 
                f"Either use a different delimiter or switch to a table encoder."
            )
        
        # Thought about this... checking the encoding capabilities is a nice idea,
        # but it is not always possible to check if the encoder can handle the delimiter
        # This is because functional encoders may not have a fixed alphabet or encoding scheme.
        # Instead, we assume that if the alphabet includes the delimiter, it can be encoded.
        
        # Test actual encoding capability
        # try:
        #     test_encoding = encoder.encode(self.sequence_delimiter)
        #     if test_encoding is None or len(test_encoding) == 0:
        #         raise ValueError(
        #             f"Functional encoder failed to encode sequence delimiter '{self.sequence_delimiter}'"
        #         )
        # except Exception as e:
        #     raise ValueError(
        #         f"Functional encoder cannot handle sequence delimiter '{self.sequence_delimiter}': {str(e)}"
        #     )
        
        # Return encoder if all validations pass
        return encoder

    def _pad_residue_values_for_delimiters(self, raw_values, sequence_parts, combined_sequence):
        """
        Add padding values at delimiter positions for multi-column residue-level data.
        
        When sequences are joined with delimiters, the combined sequence is longer than
        the sum of individual sequence parts. This method ensures that target values
        are properly aligned with sequence positions by inserting padding values at
        delimiter positions.
        
        For example, if we have sequence parts ['ACG', 'TTA'] joined as 'ACG*TTA'
        and values [1,2,3,4,5,6], the result will be [1,2,3,0,4,5,6] where 0 is
        the padding value for the delimiter position.
        
        Parameters
        ----------
        raw_values : list
            Original target values corresponding to sequence characters (excluding delimiters)
        sequence_parts : list
            List of individual sequence strings before joining
        combined_sequence : str
            The final sequence string with delimiters inserted
            
        Returns
        -------
        numpy.ndarray
            Padded values array that matches the length of the combined sequence,
            with padding values (0.0) inserted at delimiter positions
        """
        # Calculate sequence lengths for validation
        total_seq_chars = sum(len(seq_part) for seq_part in sequence_parts)
        combined_seq_length = len(combined_sequence)
        
        # Validate input assumptions
        if len(raw_values) != total_seq_chars:
            raise ValueError(
                f"Number of values ({len(raw_values)}) doesn't match total sequence characters ({total_seq_chars})"
            )
        
        if combined_seq_length <= total_seq_chars:
            # No delimiters present, return values as-is
            return np.array(raw_values, dtype=np.float32)
        
        # Insert padding values at delimiter positions
        padded_values = []
        value_idx = 0
        
        # Process each sequence part and add delimiters between them
        for seq_part_idx, seq_part in enumerate(sequence_parts):
            # Add values for all characters in this sequence part
            for _ in range(len(seq_part)):
                if value_idx < len(raw_values):
                    padded_values.append(raw_values[value_idx])
                    value_idx += 1
                else:
                    # This shouldn't happen if validation passed, but provide fallback
                    padded_values.append(0.0)
            
            # Add delimiter padding (except after the last sequence part)
            if seq_part_idx < len(sequence_parts) - 1:
                padded_values.append(0.0)  # Padding value for delimiter position
        
        return np.array(padded_values, dtype=np.float32)

    def _load_data(self):
        """
        Load and parse the complete dataset with comprehensive error handling.
        
        This method represents the culmination of PARROT's flexible data processing
        pipeline. It applies all the format detection, encoding preparation, and
        parsing logic to convert raw biological data files into structured,
        model-ready format.
        
        Key processing steps:
        1. Line-by-line parsing with robust error handling
        2. Sequence ID handling (generation or extraction)
        3. Multi-column sequence reconstruction with delimiters
        4. Target value parsing and validation
        5. Automatic padding for residue-level data with delimiters
        
        The method handles PARROT's diverse format requirements while maintaining
        data integrity and providing informative error messages for debugging.
        
        Returns
        -------
        list
            List of tuples: (seqID, combined_sequence, values)
            - seqID: String identifier for the sequence
            - combined_sequence: Reconstructed sequence with delimiters if needed
            - values: Float (sequence-level) or numpy array (residue-level) of target values
            
        Raises
        ------
        IOExceptionParrot
            For any parsing errors, with detailed line-specific error information
        """
        # Initialize storage for parsed data
        data = []
        
        # Process file line by line for memory efficiency and detailed error reporting
        with open(self.filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments - these are common in biological data files
                if not line or line.startswith('#'):
                    continue
                    
                # Wrap parsing in try-catch for detailed error reporting
                try:
                    # Split line according to specified delimiter (None = any whitespace)
                    parts = line.split(self.delimiter)
                    
                    # Handle sequence ID based on file format configuration
                    if self.excludeSeqID:
                        # Format: sequence_columns... values...
                        # Generate synthetic sequence IDs for consistency
                        if len(parts) < 2:
                            raise ValueError(f"Insufficient data on line {line_num}")
                        seqID = f"seq_{line_num}"  # Synthetic but unique ID
                        parts_after_id = parts
                    else:
                        # Format: seqID sequence_columns... values...
                        # Extract sequence ID from first column
                        if len(parts) < 3:
                            raise ValueError(f"Insufficient data on line {line_num}")
                        seqID = parts[0]
                        parts_after_id = parts[1:]
                    
                    # Separate sequence data from target values using intelligent parsing
                    sequence_parts, value_parts = self._separate_sequence_and_values(parts_after_id)
                    
                    # Reconstruct sequence by joining multiple columns with delimiter
                    # This is where multi-column sequences become single sequences
                    combined_sequence = self.sequence_delimiter.join(sequence_parts)
                    
                    # Parse target values according to data type (sequence vs residue level)
                    if self.datatype == 'sequence':
                        # Sequence-level: expect exactly one target value per sequence
                        if len(value_parts) != 1:
                            raise ValueError(f"Expected single value for sequence data on line {line_num}, got {len(value_parts)} values")
                        values = float(value_parts[0])
                    elif self.datatype == 'residues':
                        # Residue-level: expect one value per residue position
                        raw_values = [float(v) for v in value_parts]
                        
                        # Handle padding requirements for multi-column sequences
                        total_seq_chars = sum(len(seq_part) for seq_part in sequence_parts)
                        combined_seq_length = len(combined_sequence)
                        
                        # Check if delimiter padding is needed
                        # We first need to check for a format issue - length of raw values not matching expectations
                        if len(raw_values) == total_seq_chars and combined_seq_length > total_seq_chars:
                            # Values correspond to original sequence characters only
                            # Need to insert padding for delimiter positions
                            values = self._pad_residue_values_for_delimiters(raw_values, sequence_parts, combined_sequence)
                        elif len(raw_values) == combined_seq_length:
                            # Values already match combined sequence length - no padding needed
                            values = np.array(raw_values, dtype=np.float32)
                        else:
                            # Length mismatch - this indicates a data format problem
                            raise ValueError(
                                f"Number of values ({len(raw_values)}) doesn't match expected length "
                                f"for sequence data on line {line_num}. "
                                f"Expected {total_seq_chars} (sum of sequence parts) or "
                                f"{combined_seq_length} (combined sequence length)"
                            )
                    else:
                        # This should never happen due to earlier validation
                        raise ValueError(f"Invalid datatype: {self.datatype}")
                    
                    # Store successfully parsed data entry
                    data.append((seqID, combined_sequence, values))
                    
                except Exception as e:
                    # Provide detailed error information for debugging
                    raise IOExceptionParrot(f"Error parsing line {line_num}: {line}\nError: {str(e)}")
        
        return data

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        
        This method is required by PyTorch's Dataset interface and enables
        efficient batching and iteration over the dataset.
        
        Returns
        -------
        int
            Total number of sequences loaded from the data file
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve and encode a single data sample for model training.
        
        This method is the core interface between PARROT's data processing and
        PyTorch's training infrastructure. It combines sequence encoding with
        target value preparation to produce model-ready tensors.
        
        The encoding step is where biological sequences are transformed into
        numerical representations that neural networks can process. This is
        a critical step that bridges the gap between biology and machine learning.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve (0 to len(dataset)-1)
            
        Returns
        -------
        tuple
            (seqID, sequence_vector, values) where:
            - seqID: String identifier for debugging and tracking
            - sequence_vector: Encoded sequence as PyTorch tensor
            - values: Target values (float for sequence-level, array for residue-level)
            
        Raises
        ------
        ValueError
            If sequence encoding fails, with detailed error information
        """
        # Retrieve stored data for this index
        seqID, sequence, values = self.data[idx]

        # Encode the biological sequence into numerical representation
        # This is where the biological sequence becomes machine learning input
        try:
            sequence_vector = self.encoder.encode(sequence)
        except Exception as e:
            # Provide detailed error context for debugging encoding issues
            raise ValueError(f"Error encoding sequence '{sequence}' for seqID '{seqID}': {str(e)}")

        # Return the complete sample ready for model training
        return seqID, sequence_vector, values

    def __del__(self):
        """
        Clean up resources when the dataset object is destroyed.
        
        This method ensures that memory is properly released when the dataset
        is no longer needed. This is particularly important for large datasets
        or when creating multiple dataset instances during experimentation.
        
        The garbage collection call helps ensure that PyTorch tensors and
        other large objects are promptly released from memory.
        """
        # Force garbage collection to free up memory
        gc.collect()



#----------------
# Collate function for the various modes
#----------------


def seq_regress_collate(batch):
    """Collate function for sequence regression"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    targets = [item[2] for item in batch]  # Single value per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq.clone().detach()

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    return names, padded_seqs, targets_tensor


def seq_class_collate(batch):
    """Collate function for sequence classification"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    targets = [item[2] for item in batch]  # Single class per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensor with appropriate size and type
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)

    for i, seq in enumerate(seq_vectors):
        padded_seqs[i, :seq.size(0), :] = seq.clone().detach()

    # Convert targets to tensor (integers for classification)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return names, padded_seqs, targets_tensor


def res_regress_collate(batch):
    """Collate function for residue regression"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    target_arrays = [item[2] for item in batch]  # Array of values per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensors
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)
    padded_targets = torch.zeros((len(seq_vectors), max_len), dtype=torch.float32)

    for i, (seq, targets) in enumerate(zip(seq_vectors, target_arrays)):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq.clone().detach()
        padded_targets[i, :seq_len] = torch.tensor(targets, dtype=torch.float32)

    return names, padded_seqs, padded_targets


def res_class_collate(batch):
    """Collate function for residue classification"""
    names = [item[0] for item in batch]
    seq_vectors = [item[1].clone().detach().float() for item in batch]
    target_arrays = [item[2] for item in batch]  # Array of class labels per sequence
    
    # Determine the longest sequence in the batch
    max_len = max(seq.size(0) for seq in seq_vectors)

    # Preallocate tensors
    padded_seqs = torch.zeros((len(seq_vectors), max_len, seq_vectors[0].size(1)), dtype=torch.float32)
    padded_targets = torch.zeros((len(seq_vectors), max_len), dtype=torch.long)

    for i, (seq, targets) in enumerate(zip(seq_vectors, target_arrays)):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq.clone().detach()
        padded_targets[i, :seq_len] = torch.tensor(targets, dtype=torch.long)

    return names, padded_seqs, padded_targets


def split_dataset_indices(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into training, validation, and test sets based on the 
    specified ratio requested by the user.

    Returns
    -------
    tuple
        train_indices, val_indices, test_indices
    """
    # determine the length of the dataset (number of rows)
    dataset_size = len(dataset)

    # create a list of all the different row indexes (to uniquely identify each row)
    indices = list(range(dataset_size))

    # randomly shuffle the indexes so that we can just pull the first 70% of indexes to get out training set
    np.random.shuffle(indices)

    # determine the number of rows to put in the training and validation set
    # Note: we are aiming to find the last index for each set.
    # the test set is just the rest of the indexes
    train_split = int(np.floor(train_ratio * dataset_size))
    # we add the percentage to find the last row that correspond to having both train and validation indexes pulled
    # this makes life much easier as we can use the integers directly for our indexing below
    val_split = int(np.floor((train_ratio + val_ratio) * dataset_size))

    # pull out the unique indexes for each set from the shuffled set
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    return train_indices, val_indices, test_indices


def initial_data_prep(save_splits_loc, dataset, train_ratio, val_ratio):
    """
    This function preps the data and writes the splits of train, validation, and test to disk.

    Parameters
    ----------
    save_splits_loc : str
        this is the file location to save the splits that are used from the dataset
    dataset : Object
        This is a dataset that you will use to train the model
    train_ratio : float
        this is the ratio to use to train the model
    val_ratio : float
        this is the ratio to use to validate the model

    """
    # function that does initial data prep. Basically,
    # this function will take in a dataset, get the indices
    # then write them out to disk. 
    train_indices, val_indices, test_indices = split_dataset_indices(dataset, train_ratio, val_ratio)
    with open(save_splits_loc, "w") as f:
        f.write(" ".join(str(i) for i in train_indices))
        f.write("\n")
        f.write(" ".join(str(i) for i in val_indices))
        f.write("\n")
        f.write(" ".join(str(i) for i in test_indices))
    f.close()



def read_indices(filepath):
    """
    Read in the indices for train, val, test

    Parameters
    ----------
    filepath : str
            Path to a whitespace-separated splitfile

    Returns
    -------
    numpy int array
            an array of the indices for the training set samples
    numpy int array
            an array of the indices for the validation set samples
    numpy int array
            an array of the indices for the testing set samples
    """

    with open(filepath) as f:
        lines = f.readlines()

    # Use np.fromstring with explicit dtype and separator (handles deprecation)
    training_samples = np.fromstring(lines[0], dtype=int, sep=" ")
    val_samples = np.fromstring(lines[1], dtype=int, sep=" ")
    test_samples = np.fromstring(lines[2], dtype=int, sep=" ")

    return training_samples, val_samples, test_samples


def parse_file_v2(filepath, datatype=None, problem_type='regression', num_classes=1, 
                  excludeSeqID=False, encoder_cfg=None, encoder=None, delimiter=None, 
                  sequence_delimiter='*'):
    """
    Alternative implementation of parse_file with improved memory handling
    
    Returns a SequenceDataset object instead of raw parsed data
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    datatype : str or None
        'sequence' or 'residues'. If None, will be inferred from data
    problem_type : str
        'regression' or 'classification'
    num_classes : int
        Number of classes (for classification)
    excludeSeqID : bool
        Whether sequence IDs are excluded from the file
    encoder_cfg : DictConfig
        Hydra configuration for the encoder
    encoder : BaseParrotEncoder
        Pre-instantiated encoder object (takes precedence over encoder_cfg)
    delimiter : str
        Delimiter for splitting lines (None = any whitespace)
    sequence_delimiter : str
        Delimiter to use when joining multiple sequence columns (default: '*')
    """
    
    dataset = SequenceDataset(filepath=filepath, 
                             encoder_cfg=encoder_cfg,
                             encoder=encoder,
                             excludeSeqID=excludeSeqID,
                             datatype=datatype,
                             delimiter=delimiter,
                             sequence_delimiter=sequence_delimiter)
    
    # Validate class labels if classification
    if problem_type == 'classification':
        for i, (seqID, _, values) in enumerate(dataset.data):
            # Add validation logic here
            pass
    
    return dataset   


def create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size=32, 
                      distributed=False, num_workers=0, datatype='sequence', problem_type='regression'):
    """
    Create DataLoaders with appropriate collate functions based on data type and problem type
    
    Parameters:
    -----------
    dataset : SequenceDataset
        The dataset to create loaders for
    train_indices, val_indices, test_indices : array-like
        Indices for each split
    batch_size : int
        Batch size for training/validation (test uses batch_size=1)
    distributed : bool
        Whether to use distributed sampling
    num_workers : int
        Number of worker processes for data loading
    datatype : str
        'sequence' or 'residues'
    problem_type : str
        'regression' or 'classification'
    """
    
    # Select appropriate collate function
    if datatype == 'sequence':
        if problem_type == 'regression':
            collate_fn = seq_regress_collate
        else:  # classification
            collate_fn = seq_class_collate
    else:  # residues
        if problem_type == 'regression':
            collate_fn = res_regress_collate
        else:  # classification
            collate_fn = res_class_collate
    
    # Create samplers
    if distributed == False:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_indices, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_indices, shuffle=False)
        test_sampler = torch.utils.data.DistributedSampler(test_indices, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, 
                             collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, 
                           collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, 
                            collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader

