from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from negmas.outcomes import Outcome

__all__ = [
    "advantage",
    "kendall_tau_from_ufuns",
    "load_class",
    "enumerate_outcomes",
]


def load_class(path: str):
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a fully-qualified class path, got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def enumerate_outcomes(outcome_space, max_outcomes: int = 512) -> list[Outcome]:
    outcomes = list(outcome_space.enumerate_or_sample(max_outcomes))
    if not outcomes:
        raise ValueError("Scenario has no outcomes")
    return outcomes


def advantage(ufun, agreement: Outcome | None) -> float:
    if agreement is None:
        return 0.0
    mn, mx = ufun.minmax(above_reserve=False)
    rv = float(ufun.reserved_value)
    denom = float(mx) - rv
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (float(ufun(agreement)) - rv) / denom))


def _rank(values: list[float]) -> list[int]:
    indexed = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank
    return ranks


def kendall_tau_from_ufuns(estimated, truth, outcomes: list[Outcome]) -> float:
    from scipy.stats import kendalltau

    if estimated is None:
        return 0.0

    def _eval_estimated(model, outcome: Outcome) -> float:
        try:
            return float(model(outcome))
        except Exception:
            pass
        try:
            return float(model.eval(outcome))
        except Exception:
            pass
        try:
            return float(model.eval_normalized(outcome))
        except Exception:
            return 0.0

    est_vals = [_eval_estimated(estimated, outcome) for outcome in outcomes]
    true_vals = [float(truth(outcome)) for outcome in outcomes]
    tau, _ = kendalltau(est_vals, true_vals, variant="b")
    if tau is None:
        return 0.0
    # Official ANL 2026 normalizes Kendall tau from [-1, 1] to [0, 1].
    return max(0.0, min(1.0, (float(tau) + 1.0) / 2.0))
