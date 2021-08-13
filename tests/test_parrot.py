"""
Unit and regression tests for the PARROT package. Note that these tests are by no means comprehensive.
For unexplained issues, please reach out on the GitHub page:
https://github.com/idptools/parrot/issues
"""

# Import package, test suite, and other packages as needed
import parrot
import pytest
import sys
import os
import pathlib
import shutil


def test_parrot_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "parrot" in sys.modules


def test_noarg_train_call():
    """Calling parrot-train with no arguments should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-train")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-train"]

    with pytest.raises(SystemExit):
        exec(script)

    script_descriptor.close()


def test_train_invalid_dtype():
    """Calling parrot-train with invalid datatype arg should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-train")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-train", "../data/seq_class_dataset.tsv",
                "../data/output_network.pt", "-d", "gibberish", "-c", "4"]

    with pytest.raises(ValueError):
        exec(script)

    script_descriptor.close()


def test_train_invalid_numclasses():
    """Calling parrot-train with invalid number of classes should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-train")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-train", "../data/seq_class_dataset.tsv",
                "../data/output_network.pt", "-d", "sequence", "-c", "2"]

    with pytest.raises(ValueError):
        exec(script)

    script_descriptor.close()


def test_train_invalid_setFractions():
    """Calling parrot-train with --setFractions not summing to 1 should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-train")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-train", "../data/seq_class_dataset.tsv", "../data/output_network.pt",
                "-d", "sequence", "-c", "3", "--set-fractions", "0.8", "0.15", "0.15"]

    with pytest.raises(ValueError):
        exec(script)

    script_descriptor.close()


def test_noarg_optimize_call():
    """Calling parrot-optimize with no arguments should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-optimize")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-optimize"]

    with pytest.raises(SystemExit):
        exec(script)

    script_descriptor.close()


def test_optimize_invalid_dtype():
    """Calling parrot-optimize with invalid datatype arg should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-optimize")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-optimize", "../data/seq_class_dataset.tsv",
                "../data/output_network.pt", "-d", "gibberish", "-c", "4"]

    with pytest.raises(ValueError):
        exec(script)

    script_descriptor.close()


def test_optimize_invalid_numclasses():
    """Calling parrot-optimize with invalid number of classes should return an error"""
    os.chdir(pathlib.Path(__file__).parent.absolute())
    loc = shutil.which("parrot-optimize")
    script_descriptor = open(os.path.abspath(loc))
    script = script_descriptor.read()
    sys.argv = ["parrot-optimize", "../data/seq_class_dataset.tsv",
                "../data/output_network.pt", "-d", "sequence", "-c", "2"]

    with pytest.raises(ValueError):
        exec(script)

    script_descriptor.close()


def test_split_data():
    """Test that split_data() function is working as intended"""
    from parrot import process_input_data as pid

    data_file = os.path.abspath("../data/seq_class_dataset.tsv")
    train, val, test = pid.split_data(data_file, datatype='sequence',
                                  problem_type='classification', num_classes=3)

    assert (len(train) == 210) and (len(val) == 45) and (len(test) == 45) and (len(train[0]) == 3)


def test_split_data_cv():
    """Test that split_data_cv() function is working as intended"""
    from parrot import process_input_data as pid

    data_file = os.path.abspath("../data/seq_class_dataset.tsv")
    cvs, train, val, test = pid.split_data_cv(data_file, datatype='sequence',
                                  problem_type='classification', num_classes=3)

    assert (len(train) == 210) and (len(val) == 45) and (len(test) == 45) and (
        len(train[0]) == 3) and (len(cvs) == 5) and (len(cvs[0]) == 2)
