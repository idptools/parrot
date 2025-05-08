#!/usr/bin/env python
"""
Usage: $ parrot-preprocess raw_data_file processed_data_file <flags>
  
parrot-preprocess is a text manipulation file that allows for simple conversion of parrot compliant
datafiles to filter, re-define, scale, or regularize input data for a parrot training operation

For more information on usage, use the '-h' flag. 

.............................................................................
parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license. 
"""
import argparse
from parrot import process_input_data
from datetime import datetime
from parrot.tools import cli
from parrot.tools import class_balancing
from parrot.tools import preproc

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#
#
# Parse the command line arguments
def main():

    parser = argparse.ArgumentParser(description='parrot-preprocess is a file manipulation tool for generating input files for parrot-predict')    

    parser.add_argument('input_data_file',   help='path to tsv file with format: <idx> <sequence> <data>. This is the file that will be manipulated in some way')
    parser.add_argument('output_data_file',  help='path to be written to for an output data file. Will overwrite an existing file')
    parser.add_argument('-d', '--datatype', metavar='dtype', type=str, required=True, help="REQUIRED. Format of the input data file, must be 'sequence' or 'residues'")
    parser.add_argument("-m", "--mode", help="Must be either either 'classification' or 'regression'", type=str, required=True, metavar='dtype')
    parser.add_argument("--delimiter", help="If a non-whitespace delimiter was used in an input file, this allows the user to define an alternative character.", type=str)    
    parser.add_argument("-v", "--verbose", help="Defines how loud the output should be...", action='store_true')

    # recode flag
    parser.add_argument("--recode-classes", help="Flag which, if set to true, will read in class data regardless of if it would have been valid in PARROT and recode it in a parrot-compliant format (0,1,2,...[n]) where there are n-1 classes.", action='store_true')

    # duplicate fixing flags
    parser.add_argument("--remove-line-duplicates", help="If flag set to true this will remove all lines that are duplicated in the file", action='store_true')
    parser.add_argument("--remove-sequence-duplicates", help="If flag set to true, this will remove all copies of any sequence that appears more than once in the file", action='store_true')
    
    # class balance 
    parser.add_argument("--class-balance",   help="If this flag is set the output file will be equally balanced in terms of classes assigned to either seuqences or residues", action='store_true')
    parser.add_argument("--cb-block-size", help="If --class-balance used with residues defines the block size used for class balancing (default 100, recommended not to mess with)", type=int, default=100)
    parser.add_argument("--cb-max-fraction-removed", help="If --class-balance used, defines what fraction of the data could be removed in persuit of a balanced dataset. Default = 0.25 ", type=float, default=0.25)
    parser.add_argument("--cb-balance-threshold", help="If --class-balance used, defines the balance ratio (most numerous class / least numerous class) threshold used to say if a dataset is 'balanced'. Default = 1.4", type=float, default=1.4)
    parser.add_argument("--cb-shuffle-off", help="Flag which, if included, means data for class balance sub-sampling is not shuffled for residue-style class-balance. Generally not recommended but may be useful in certain situations. Note this does nothing for sequence class balancing.",  action='store_true')

    args = parser.parse_args()
    
    ## ..........................................................................................
    ## The body of code below does sanity checking on input
    
    cli.validate_args(args.datatype, ['residues','sequence'], "Please pass either 'residues' or 'sequence' with the --datatype flag")
    cli.validate_args(args.mode, ['classification','regression'], "Please passe either 'classification' or 'regression' with the --mode flag")

    if args.recode_classes:
        if args.mode != 'classification':
            raise Exception('If we are trying to recode classes the data mode must be classification')

    if args.remove_line_duplicates:
        if args.remove_sequence_duplicates:
            raise Exception('Cannot request to remove both line duplicates (--remove-line-duplicates) and sequence duplicates (--remove-sequence-duplicates). Must be one or the other')
    
    # read in and parse the TSV file. Note that if no args.delimiter is passed, this passes 'None' which
    # is the default value and splits on any whitespace.
    lines  = process_input_data.read_tsv_raw(args.input_data_file, delimiter=args.delimiter)

    if args.verbose:
        print(f'Found {len(lines)} datapoints in the original input file')

    # how many data points?
    original_data_count = len(lines)

    if args.remove_line_duplicates:
        lines = preproc.remove_line_duplicates(lines, args.verbose)

    if args.remove_sequence_duplicates:
        lines = preproc.remove_sequence_duplicates(lines, args.verbose)

    if args.mode == 'classification':

        if args.recode_classes:
            (lines, mapping) = preproc.recode_classes(lines)

            # write a mapping file out
            with open('class_mapping.txt','w') as fh:
                fh.write('File generated by parrot-preprocess\n\n')
                fh.write(f'File generated at {datetime.now()}\n')
                fh.write(f'Number of classes: {len(mapping)}\n\n')
                for i in mapping:
                    fh.write(f'{i} = {mapping[i]}\n')

        if args.class_balance:

            if args.datatype == 'residues':
                (lines, tracker) = class_balancing.run_class_balance_residues(lines, 
                                                                              balance_threshold=args.cb_balance_threshold,
                                                                              block_size=args.cb_block_size, 
                                                                              max_fraction_removed=args.cb_max_fraction_removed,
                                                                              shuffle_data = args.cb_shuffle_off,
                                                                              verbose=args.verbose)

                with open('class_balance_trace.csv','w') as fh:
                    for x in range(len(tracker)):
                        fh.write(f'{x+1}, {tracker[x]:.3f}\n')
            else:
                if args.cb_shuffle_off:
                    print('Warning: --cb-shuffle-off does nothing for sequence class balancing')
                if args.cb_block_size != 100:
                    print('Warning: --cb-block-size does nothing for sequence class balancing')

                lines = class_balancing.run_class_balance_sequences(lines, 
                                                                    balance_threshold=args.cb_balance_threshold, 
                                                                    max_fraction_removed=args.cb_max_fraction_removed, 
                                                                    verbose=args.verbose)
                                                                              


    # write out
    if args.verbose:
        print(f'Writing out {len(lines)} lines of data (started with {original_data_count})...')

    cli.write_datafile(args.output_data_file, lines)
        
