# PARROT Example Data

This folder contains example datafiles for running the various commands in this package.

## seq_class_dataset.tsv

This datafile is an example sequence classification dataset. For this data, class '0' sequences are randomly generated in a manner that strongly favors basic amino acids. Similarly class '1' sequences are randomly generated in a manner that favors acidic amino acids. Class '2' sequences are generated such that all amino acids have the same probability.

## seq_regress_dataset.tsv

This datafile is an example sequence regression dataset. The target value for each sequence is determined as the "density" of hydrophobic residues within the sequence. Specifically, this number is the count of hydrophobic residues (Ile, Leu, Val, Met, Phe, Tyr, Trp) divided by the length of sequence, multiplied times 10.

## res_class_dataset.tsv

This datafile is an example residue classification dataset. Class assignments are similar to in the sequence classification dataset. However, in this case all sequences are generated using a Hidden Markov Model with three hidden states: basic AA-enriched, acidic AA-enriched, and neutral. There is 6% chance after each amino acid that the generator will switch to one of the other states and an 88% chance it will remain in the current state. Residues are assigned classes based on the hidden state of the model at the time, not based on the actual output amino acid.

## res_regress_dataset.tsv

This datafile is an example residue regression dataset. Each residue has a target value based on the local hydrophobicity within the sequence. Specifically, the value at each residue is the average Kyte-Doolittle hydrophobicity score for that amino acid and the adjacent amino acids.

## seqfile.txt (and seqfile_noIDs.txt)

These files are example sequence files for use in ``parrot-predict``. `seqfile.txt` has sequences and sequence IDs, while `seqfile_noIDs.txt` is just a list of sequences (to be used with the ``--excludeSeqIDs`` flag).

## splitfile.txt

This file is an example splitfile that can be provided with the ``--split`` flag. Each of the three lines contain the integers that assign sequences from the datafile into the training, validation and test sets respectively. This particular file only has integers from 0-299, for intended use with the example datafiles provided here.