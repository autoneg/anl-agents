"""The package's torch randomness follows negmas's global seed switch."""

import random

import numpy as np
import pytest

import anl_agents  # noqa: F401  (registers the seeder)

rand = pytest.importorskip("negmas.helpers.rand")
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _restore_entropy():
    """Seeding is process-wide: give the rest of the suite fresh entropy back."""
    yield
    random.seed()
    np.random.seed()
    torch.seed()


def test_seed_all_makes_torch_reproducible():
    rand.seed_all(42)
    first = torch.randn(3).tolist()
    rand.seed_all(42)
    second = torch.randn(3).tolist()
    assert first == second


def test_seed_all_none_is_a_noop():
    assert rand.seed_all(None) is None
