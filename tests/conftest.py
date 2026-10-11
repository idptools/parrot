"""
Shared fixtures for the PARROT test suite.

Every test runs against the example datasets in /data (or small files written to
a temporary directory), and every random number generator is re-seeded before
each test so results are reproducible. The whole suite is designed to run on a
CPU in a minute or two.

Device choice 'auto' resolves to the CPU in tests, so results do not depend on
the machine's GPU. Tests marked @pytest.mark.real_devices see the machine's real
CUDA/MPS availability (see test_devices.py).
"""

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import matplotlib

# Never try to open plotting windows during tests
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from parrot import devices
from parrot.architectures import registry
from parrot.reproducibility import set_random_seed
from parrot.scripts import arguments as arguments_module

from tests.helpers import TEST_SEED, LinearTestArchitecture, allow_tf32_everywhere


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom markers used in this suite."""
    config.addinivalue_line('markers', 'real_devices: let the test see the machine\'s real CUDA/MPS devices')


@pytest.fixture(autouse=True)
def cpu_unless_requested(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide GPUs from PARROT's device choice unless the test is marked real_devices."""
    if request.node.get_closest_marker('real_devices') is None:
        monkeypatch.setattr(devices, 'cuda_available', lambda: False)
        monkeypatch.setattr(devices, 'mps_available', lambda: False)


@pytest.fixture(autouse=True)
def seeded_and_clean() -> Iterator[None]:
    """Seed every RNG before each test and close any open figures afterwards."""
    set_random_seed(TEST_SEED)
    yield
    plt.close('all')


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[], Any], list[Any]], Any]:
    """Return a function that runs a PARROT command-line main() with the given arguments.

    The first argument is the program name, exactly as it would appear in sys.argv. Errors in the input propagate
    as exceptions (as with PARROT_TRACEBACK=1), so tests can check them; tests/test_arguments.py checks the messages
    users see instead.
    """
    def _run(main_function: Callable[[], Any], arguments: list[Any]) -> Any:
        monkeypatch.setattr(sys, 'argv', [str(argument) for argument in arguments])
        monkeypatch.setenv(arguments_module.TRACEBACK_VARIABLE, '1')
        return main_function()
    return _run


@pytest.fixture
def write_lines(tmp_path: Path) -> Callable[[str, list[str]], str]:
    """Return a function that writes a list of strings (one per line) to a file in tmp_path."""
    def _write(filename: str, lines: list[str]) -> str:
        path = tmp_path / filename
        path.write_text('\n'.join(lines) + '\n')
        return str(path)
    return _write


@pytest.fixture
def test_architecture(monkeypatch: pytest.MonkeyPatch) -> type[LinearTestArchitecture]:
    """Register the test-only LinearTestArchitecture for the duration of one test."""
    monkeypatch.setitem(registry._ARCHITECTURES, LinearTestArchitecture.name, LinearTestArchitecture)
    return LinearTestArchitecture


@pytest.fixture
def tf32_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow TF32 everywhere for the test (see helpers.allow_tf32_everywhere)."""
    allow_tf32_everywhere(monkeypatch)
