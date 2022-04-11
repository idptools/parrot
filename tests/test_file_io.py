import pytest
import parrot
from parrot import process_input_data
from parrot.parrot_exceptions import IOExceptionParrot

datapath = parrot.get_directory()
test_datapath = parrot.get_directory('../tests/test_data')



# ..........................................................................................
#
#
def test_parrot_read_sequence():
    lines = process_input_data.read_tsv_raw(f'{datapath}/res_class_dataset.tsv') 

    # expect this file to have 300 lines in it
    assert len(lines) == 300

    # check we 
    lines2 = process_input_data.read_tsv_raw(f'{datapath}/res_class_dataset.tsv', delimiter=' ' ) 

    # expect this file to have 300 lines in it
    assert len(lines2) == 300



# ..........................................................................................
#
#
def test_parrot_parse_lines():
    lines = process_input_data.read_tsv_raw(f'{datapath}/res_class_dataset.tsv') 

    # test we can convert this to 
    data = process_input_data.__parse_lines(lines, datatype='residues')



# ..........................................................................................
#
#
def test_parrot_parse_lines_error():
    lines = process_input_data.read_tsv_raw(f'{test_datapath}/res_class_dataset_broken_1.tsv') 

    # check that this bad file raises an appropriate exception
    with pytest.raises(IOExceptionParrot):
        data = process_input_data.__parse_lines(lines, datatype='residues')

    lines = process_input_data.read_tsv_raw(f'{test_datapath}/res_class_dataset_broken_2.tsv') 

    # check that this bad file raises an appropriate exception
    with pytest.raises(IOExceptionParrot):
        data = process_input_data.__parse_lines(lines, datatype='residues')

