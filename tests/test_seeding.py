"""The package's torch randomness follows negmas's global seed switch."""

import pytest

import anl_agents  # noqa: F401  (registers the seeder)

rand = pytest.importorskip("negmas.helpers.rand")
torch = pytest.importorskip("torch")


def test_seed_all_makes_torch_reproducible():
    rand.seed_all(42)
    first = torch.randn(3).tolist()
    rand.seed_all(42)
    second = torch.randn(3).tolist()
    assert first == second


def test_seed_all_none_is_a_noop():
    assert rand.seed_all(None) is None
