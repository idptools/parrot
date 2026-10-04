"""
Tests for reading the example datafiles, and broken ones, with parrot.process_input_data.
"""

import pytest

from parrot import process_input_data
from parrot.parrot_exceptions import IOExceptionParrot

from tests.helpers import DATA_DIR, TEST_DATA_DIR


def test_read_example_datafile():
    """The residue classification example has 300 lines, whether split on any whitespace or on spaces"""
    path = f'{DATA_DIR}/res_class_dataset.tsv'
    assert len(process_input_data.read_tsv_raw(path)) == 300
    assert len(process_input_data.read_tsv_raw(path, delimiter=' ')) == 300


def test_parse_example_datafile():
    lines = process_input_data.read_tsv_raw(f'{DATA_DIR}/res_class_dataset.tsv')
    data = process_input_data.parse_lines(lines, datatype='residues')
    assert len(data) == 300 and all(len(sample[1]) == len(sample[2]) for sample in data)


@pytest.mark.parametrize('broken_file', ['res_class_dataset_broken_1.tsv', 'res_class_dataset_broken_2.tsv'])
def test_parse_broken_datafile(broken_file):
    lines = process_input_data.read_tsv_raw(f'{TEST_DATA_DIR}/{broken_file}')
    with pytest.raises(IOExceptionParrot):
        process_input_data.parse_lines(lines, datatype='residues')
