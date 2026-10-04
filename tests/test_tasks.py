"""
Tests for parrot.tasks: the Task description of a machine-learning problem.
"""

import numpy as np
import pytest

from parrot.tasks import Task


@pytest.mark.parametrize('datatype, num_classes, problem_type, is_classification, per_residue', [
    ('sequence', 1, 'regression', False, False),
    ('sequence', 3, 'classification', True, False),
    ('residues', 1, 'regression', False, True),
    ('residues', 2, 'classification', True, True),
])
def test_task_properties(datatype, num_classes, problem_type, is_classification, per_residue):
    task = Task(datatype, num_classes)

    assert task.problem_type == problem_type
    assert task.is_classification is is_classification
    assert task.per_residue is per_residue


@pytest.mark.parametrize('datatype', ['residue', 'sequences', 'gibberish', ''])
def test_task_rejects_invalid_datatype(datatype):
    with pytest.raises(ValueError, match='datatype'):
        Task(datatype, 1)


@pytest.mark.parametrize('num_classes', [0, -1, 1.5, True, '3'])
def test_task_rejects_invalid_num_classes(num_classes):
    with pytest.raises(ValueError, match='positive integer'):
        Task('sequence', num_classes)


def test_task_accepts_numpy_integers():
    # e.g. a class count read from the shape of saved weights
    task = Task('sequence', np.int64(3))
    assert task.num_classes == 3 and type(task.num_classes) is int


def test_task_config_round_trip():
    task = Task('residues', 4)
    assert task.to_config() == {'datatype': 'residues', 'num_classes': 4}
    assert Task.from_config(task.to_config()) == task


def test_task_is_immutable_and_hashable():
    task = Task('sequence', 2)
    with pytest.raises(AttributeError):
        task.num_classes = 3
    assert {task: 'ok'}[Task('sequence', 2)] == 'ok'
