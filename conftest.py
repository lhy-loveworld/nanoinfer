"""Pytest configuration shared across the test suite.

Adds a `device` fixture and a `--device` CLI flag so you can run the same tests
on CPU (default, deterministic, no GPU required) or on your CUDA GPU:

    pytest                       # CPU
    pytest --device cuda         # RTX 5080
    pytest -m "not perf"         # skip benchmarks
"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="torch device to run tests on: cpu | cuda",
    )


@pytest.fixture(scope="session")
def device(request):
    dev = request.config.getoption("--device")
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return torch.device(dev)


@pytest.fixture(autouse=True)
def _determinism():
    """Seed every test so failures are reproducible."""
    torch.manual_seed(0)
    yield
