# -*- coding: utf-8 -*-
try:
    from negmas.helpers.rand import register_seeder
except ImportError:  # negmas too old; nothing to hook onto
    register_seeder = None


def _seed_torch(seed: int) -> None:
    """Seed torch's global generator (negmas seeds `random` and `numpy` itself)."""
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # torch missing or unhappy; seeding is best-effort
        pass


if register_seeder is not None:
    register_seeder(_seed_torch)

from .agents import *  # noqa: E402

__all__ = agents.__all__ + ["anl2024", "anl2025", "anl2026", "contrib"]
