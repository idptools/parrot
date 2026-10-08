"""
Unit and regression tests for the PARROT package. Note that these tests are by no means comprehensive.
For unexplained issues, please reach out on the GitHub page:
https://github.com/idptools/parrot/issues
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import parrot
from parrot import process_input_data as pid
from parrot.scripts import parrot_optimize, parrot_train
from parrot.tasks import Task

from tests.helpers import data_path, training_command


def test_parrot_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "parrot" in sys.modules


def test_parrot_has_version():
    """The version is available even when running from a source checkout"""
    assert isinstance(parrot.__version__, str) and len(parrot.__version__) > 0


def test_public_names_are_explicit():
    """Everything parrot exports is listed in __all__, and nothing it merely imports (e.g. numpy) leaks out"""
    assert all(hasattr(parrot, name) for name in parrot.__all__)
    assert not any(hasattr(parrot, name) for name in ('np', 'torch', 'os', 'math', 'Any'))


def test_noarg_train_call(run_cli):
    """Calling parrot-train with no arguments should return an error"""
    with pytest.raises(SystemExit):
        run_cli(parrot_train.main, ["parrot-train"])


def test_train_invalid_dtype(run_cli, tmp_path):
    """Calling parrot-train with invalid datatype arg should stop with a usage message"""
    with pytest.raises(SystemExit):
        run_cli(parrot_train.main, training_command("parrot-train", "brnn", data_path("seq_class_dataset.tsv"),
                                                     tmp_path / "output_network.pt", "gibberish", 4))


def test_train_invalid_numclasses(run_cli, tmp_path):
    """Calling parrot-train with invalid number of classes should return an error"""
    with pytest.raises(ValueError):
        run_cli(parrot_train.main, training_command("parrot-train", "brnn", data_path("seq_class_dataset.tsv"),
                                                     tmp_path / "output_network.pt", "sequence", 2))


def test_train_invalid_set_fractions(run_cli, tmp_path):
    """Calling parrot-train with --setFractions not summing to 1 should return an error"""
    with pytest.raises(ValueError):
        run_cli(parrot_train.main, training_command("parrot-train", "brnn", data_path("seq_class_dataset.tsv"),
                                                    tmp_path / "output_network.pt", "sequence", 3,
                                                    "--set-fractions", "0.8", "0.15", "0.15"))


def test_noarg_optimize_call(run_cli):
    """Calling parrot-optimize with no arguments should return an error"""
    with pytest.raises(SystemExit):
        run_cli(parrot_optimize.main, ["parrot-optimize"])


def test_optimize_invalid_dtype(run_cli, tmp_path):
    """Calling parrot-optimize with invalid datatype arg should stop with a usage message"""
    with pytest.raises(SystemExit):
        run_cli(parrot_optimize.main, training_command("parrot-optimize", "brnn", data_path("seq_class_dataset.tsv"),
                                                        tmp_path / "output_network.pt", "gibberish", 4))


def test_optimize_invalid_numclasses(run_cli, tmp_path):
    """Calling parrot-optimize with invalid number of classes should return an error"""
    with pytest.raises(ValueError):
        run_cli(parrot_optimize.main, training_command("parrot-optimize", "brnn", data_path("seq_class_dataset.tsv"),
                                                        tmp_path / "output_network.pt", "sequence", 2))


def test_split_data():
    """Test that split_data() function is working as intended"""
    train, val, test = pid.split_data(data_path("seq_class_dataset.tsv"), Task('sequence', 3))

    assert (len(train) == 210) and (len(val) == 45) and (len(test) == 45) and (len(train[0]) == 3)


def test_split_data_cv():
    """Test that split_data_cv() function is working as intended"""
    cvs, train, val, test = pid.split_data_cv(data_path("seq_class_dataset.tsv"), Task('sequence', 3))

    assert (len(train) == 210) and (len(val) == 45) and (len(test) == 45) and (
        len(train[0]) == 3) and (len(cvs) == 5) and (len(cvs[0]) == 2)
