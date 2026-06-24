from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations, permutations, product
from math import exp

from negmas import ZeroSumModel
from negmas.gb.components.models.ufun import UFunModel
from negmas.outcomes import Outcome
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun
from negmas.sao.common import ResponseType, SAOState

from .threshold_random import ThresholdRandomInferenceAgent

__all__ = [
    "MywayAgent",
    "MywayV2Agent",
    "MywayOpponentFilterAgent",
    "MywayRejectLateAgent",
    "MywayRejectBothAgent",
    "MywayOpponentFilterRejectLateAgent",
    "MywayOpponentFilterRejectBothAgent",
    "KingAgent",
    "QueenAgent",
    "KingAgentV2",
    "KingV3Agent",
    "QueenAgentV2",
    "YASASHIAgent",
    "YasasiAgent",
    "YasasiV2Agent",
    "YasasiV3Agent",
    "YasasiV4Agent",
    "YasasiV5Agent",
    "MajiKayo",
    "YasasiSwapAgent",
    "YasasiSwap23Agent",
    "YasasiSwap24Agent",
    "YasasiSwap34Agent",
    "YasasiOutcomeSafeSwapAgent",
    "YouSuMiruAgent",
    "YouSiRu",
    "UtilitySwapAgent",
    "mywayAgent",
    "mywayV2",
    "kingAgent",
    "queenAgent",
    "kingAgentV2",
    "kingV3Agent",
    "queenAgentV2",
    "yasashiAgent",
    "yasaiV4Agent",
    "yasasiV4Agent",
    "yasaiV5Agent",
    "yasasiV5Agent",
    "majiKayo",
    "yasasiSwapAgent",
    "yasashiSwapAgent",
    "yasasiSwap23Agent",
    "yasasiSwap24Agent",
    "yasasiSwap34Agent",
    "yasasiiswapAgent",
    "yasasiOutcomeSafeSwapAgent",
    "youSuMiruAgent",
    "YouSiRu",
    "utilityswapAgent",
]


def _swap_ranked_value_order(value_fun, first_rank: int, second_rank: int):
    """Swap two ranked values inside a single issue table.

    Ranks are one-based. If an issue has too few distinct values, the table is
    left as-is.
    """

    mapping = dict(getattr(value_fun, "mapping", {}) or {})
    ordered = sorted(
        mapping.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    first_index = first_rank - 1
    second_index = second_rank - 1
    if first_index < 0 or second_index < 0:
        return TableFun(mapping=mapping)
    if len(ordered) <= max(first_index, second_index):
        return TableFun(mapping=mapping)

    first_value, first_score = ordered[first_index]
    second_value, second_score = ordered[second_index]
    swapped = dict(mapping)
    swapped[first_value] = float(second_score)
    swapped[second_value] = float(first_score)
    return TableFun(mapping=swapped)


def _swap_second_and_third_value_order(value_fun):
    """Swap the 2nd and 3rd ranked values inside a single issue table."""

    return _swap_ranked_value_order(value_fun, 2, 3)


def _build_virtual_utility(ufun, first_rank: int = 2, second_rank: int = 3):
    """Create a virtual utility with two ranks swapped per issue."""

    if ufun is None:
        return None
    values = [
        _swap_ranked_value_order(value_fun, first_rank, second_rank)
        for value_fun in ufun.values
    ]
    virtual = LinearAdditiveUtilityFunction(
        values=values,
        weights=list(getattr(ufun, "weights", [])),
        bias=float(getattr(ufun, "bias", 0.0)),
    )
    for attr in ("reserved_outcome", "_reserved_value", "_owner", "outcome_space"):
        if hasattr(ufun, attr):
            try:
                setattr(virtual, attr, getattr(ufun, attr))
            except Exception:
                pass
    return virtual


def _value_mapping(value_fun) -> dict[object, float]:
    return {value: float(score) for value, score in dict(value_fun.mapping).items()}


def _top_values(mapping: dict[object, float]) -> set[object]:
    if not mapping:
        return set()
    best = max(mapping.values())
    return {value for value, score in mapping.items() if abs(score - best) <= 1e-12}


def _copy_utility_with_issue_swaps(ufun, swaps: tuple[tuple[int, object, object], ...]):
    swap_by_issue = {issue_index: (left, right) for issue_index, left, right in swaps}
    values = []
    for issue_index, value_fun in enumerate(ufun.values):
        mapping = _value_mapping(value_fun)
        swap = swap_by_issue.get(issue_index)
        if swap is not None:
            left, right = swap
            tops = _top_values(mapping)
            if left not in tops and right not in tops and left in mapping and right in mapping:
                mapping = dict(mapping)
                mapping[left], mapping[right] = mapping[right], mapping[left]
        values.append(TableFun(mapping=mapping))
    virtual = LinearAdditiveUtilityFunction(
        values=values,
        weights=list(getattr(ufun, "weights", [])),
        bias=float(getattr(ufun, "bias", 0.0)),
    )
    for attr in ("reserved_outcome", "_reserved_value", "_owner", "outcome_space"):
        if hasattr(ufun, attr):
            try:
                setattr(virtual, attr, getattr(ufun, attr))
            except Exception:
                pass
    return virtual


def _copy_utility_with_issue_reorders(
    ufun, reorders: tuple[tuple[int, dict[object, float]], ...]
):
    reorder_by_issue = {issue_index: reorder for issue_index, reorder in reorders}
    values = []
    for issue_index, value_fun in enumerate(ufun.values):
        mapping = _value_mapping(value_fun)
        reorder = reorder_by_issue.get(issue_index)
        if reorder is not None:
            tops = _top_values(mapping)
            mapping = dict(mapping)
            for value, score in reorder.items():
                if value not in tops and value in mapping:
                    mapping[value] = score
        values.append(TableFun(mapping=mapping))
    virtual = LinearAdditiveUtilityFunction(
        values=values,
        weights=list(getattr(ufun, "weights", [])),
        bias=float(getattr(ufun, "bias", 0.0)),
    )
    for attr in ("reserved_outcome", "_reserved_value", "_owner", "outcome_space"):
        if hasattr(ufun, attr):
            try:
                setattr(virtual, attr, getattr(ufun, attr))
            except Exception:
                pass
    return virtual


def _normalized_outcome_scores(ufun, outcomes: list[Outcome]) -> list[float]:
    mn, mx = ufun.minmax(above_reserve=False)
    denom = float(mx) - float(mn)
    if denom <= 1e-12:
        return [0.5 for _ in outcomes]
    return [
        max(0.0, min(1.0, (float(ufun(outcome)) - float(mn)) / denom))
        for outcome in outcomes
    ]


def _outcome_utility_scores(ufun, outcomes: list[Outcome]) -> list[float]:
    if ufun is None:
        return [0.0 for _ in outcomes]
    return [float(ufun(outcome)) for outcome in outcomes]


def _outcome_safe_swap_candidates(ufun) -> list[list[tuple[int, object, object] | None]]:
    issue_candidates: list[list[tuple[int, object, object] | None]] = []
    for issue_index, value_fun in enumerate(ufun.values):
        mapping = _value_mapping(value_fun)
        tops = _top_values(mapping)
        values = sorted(
            [value for value in mapping if value not in tops],
            key=str,
        )
        candidates: list[tuple[int, object, object] | None] = [None]
        candidates.extend(
            (issue_index, left, right)
            for left, right in combinations(values, 2)
            if abs(mapping[left] - mapping[right]) > 1e-12
        )
        issue_candidates.append(candidates)
    return issue_candidates


def _outcome_reorder_candidates(
    ufun,
) -> list[list[tuple[int, dict[object, float]] | None]]:
    issue_candidates: list[list[tuple[int, dict[object, float]] | None]] = []
    for issue_index, value_fun in enumerate(ufun.values):
        mapping = _value_mapping(value_fun)
        tops = _top_values(mapping)
        values = sorted([value for value in mapping if value not in tops], key=str)
        scores = [mapping[value] for value in values]
        candidates: list[tuple[int, dict[object, float]] | None] = [None]
        seen: set[tuple[float, ...]] = set()
        for permuted_scores in set(permutations(scores)):
            if tuple(permuted_scores) == tuple(scores):
                continue
            if tuple(permuted_scores) in seen:
                continue
            seen.add(tuple(permuted_scores))
            reorder = {
                value: float(score)
                for value, score in zip(values, permuted_scores, strict=True)
                if abs(mapping[value] - float(score)) > 1e-12
            }
            if reorder:
                candidates.append((issue_index, reorder))
        issue_candidates.append(candidates)
    return issue_candidates


def _outcome_safe_swap_metrics(
    before: list[float], after: list[float]
) -> dict[str, float | int]:
    safe_before = [0.5 <= score <= 0.8 for score in before]
    safe_after = [0.5 <= score <= 0.8 for score in after]
    changed = [abs(left - right) > 1e-9 for left, right in zip(before, after, strict=True)]
    safe_changed = [
        was_safe and is_safe and did_change
        for was_safe, is_safe, did_change in zip(
            safe_before, safe_after, changed, strict=True
        )
    ]
    boundary_crossings = sum(
        was_safe != is_safe for was_safe, is_safe in zip(safe_before, safe_after, strict=True)
    )
    high_low_flips = sum(
        (left > 0.8 and right < 0.5) or (left < 0.5 and right > 0.8)
        for left, right in zip(before, after, strict=True)
    )
    safe_deltas = [
        abs(left - right)
        for left, right, did_safe_change in zip(before, after, safe_changed, strict=True)
        if did_safe_change
    ]
    return {
        "safe_before": sum(safe_before),
        "safe_after": sum(safe_after),
        "safe_changed": sum(safe_changed),
        "total_changed": sum(changed),
        "boundary_crossings": boundary_crossings,
        "high_low_flips": high_low_flips,
        "mean_safe_delta": sum(safe_deltas) / len(safe_deltas) if safe_deltas else 0.0,
    }


def _build_outcome_safe_swap_utility(
    ufun, outcomes: list[Outcome]
) -> tuple[object | None, dict[str, object]]:
    if ufun is None or not outcomes:
        return ufun, {}
    issue_candidates = _outcome_reorder_candidates(ufun)
    before = _normalized_outcome_scores(ufun, outcomes)
    best_ufun = ufun
    best_info: dict[str, object] = {
        "swap_count": 0,
        "swaps": [],
        "safe_before": sum(0.5 <= score <= 0.8 for score in before),
        "safe_changed": 0,
        "total_changed": 0,
        "boundary_crossings": 0,
        "high_low_flips": 0,
        "mean_safe_delta": 0.0,
    }
    best_objective = (0, 0, 0, 0, 0.0, 0)
    for selected in product(*issue_candidates):
        reorders = tuple(reorder for reorder in selected if reorder is not None)
        if not reorders:
            continue
        virtual = _copy_utility_with_issue_reorders(ufun, reorders)
        after = _normalized_outcome_scores(virtual, outcomes)
        metrics = _outcome_safe_swap_metrics(before, after)
        objective = (
            -int(metrics["boundary_crossings"]),
            -int(metrics["high_low_flips"]),
            int(metrics["safe_changed"]),
            len(reorders),
            float(metrics["mean_safe_delta"]),
            int(metrics["total_changed"]),
        )
        if objective > best_objective:
            best_objective = objective
            best_ufun = virtual
            best_info = {
                **metrics,
                "changed_issue_count": len(reorders),
                "reorders": [
                    "i"
                    + str(issue_index)
                    + ": "
                    + ", ".join(
                        f"{value!r}->{score:.6f}"
                        for value, score in sorted(
                            reorder.items(), key=lambda item: str(item[0])
                        )
                    )
                    for issue_index, reorder in reorders
                ],
            }
    return best_ufun, best_info


def _low_boundary_swap_metrics(
    before: list[float],
    after: list[float],
    *,
    lower_boundary: float = 0.4,
    upper_boundary: float = 0.45,
    target_delta: float = 0.1,
) -> dict[str, float | int]:
    deltas = [abs(left - right) for left, right in zip(before, after, strict=True)]
    changed = [delta > 1e-9 for delta in deltas]
    changed_deltas = [delta for delta in deltas if delta > 1e-9]

    def layer(score: float) -> int:
        if score < lower_boundary:
            return 0
        if score <= upper_boundary:
            return 1
        return 2

    band_crossings = sum(
        layer(left) != layer(right)
        for left, right in zip(before, after, strict=True)
    )
    low_upward_crossings = sum(
        left < lower_boundary <= right
        for left, right in zip(before, after, strict=True)
    )
    normal_downward_crossings = sum(
        left > upper_boundary and right <= upper_boundary
        for left, right in zip(before, after, strict=True)
    )
    return {
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
        "target_delta": target_delta,
        "changed_outcomes": sum(changed),
        "total_outcomes": len(before),
        "changed_rate": sum(changed) / len(before) if before else 0.0,
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "changed_mean_delta": (
            sum(changed_deltas) / len(changed_deltas) if changed_deltas else 0.0
        ),
        "max_delta": max(deltas) if deltas else 0.0,
        "low_upward_crossings": low_upward_crossings,
        "normal_downward_crossings": normal_downward_crossings,
        "band_crossings": band_crossings,
        "target_delta_error": abs(
            (sum(deltas) / len(deltas) if deltas else 0.0) - target_delta
        ),
    }


def _build_low_boundary_active_swap_utility(
    ufun,
    outcomes: list[Outcome],
    *,
    lower_boundary: float = 0.4,
    upper_boundary: float = 0.45,
    target_delta: float = 0.1,
) -> tuple[object | None, dict[str, object]]:
    if ufun is None or not outcomes:
        return ufun, {}
    issue_candidates = _outcome_reorder_candidates(ufun)
    before = _normalized_outcome_scores(ufun, outcomes)
    best_ufun = ufun
    best_info: dict[str, object] = {
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
        "target_delta": target_delta,
        "changed_issue_count": 0,
        "reorders": [],
        "changed_outcomes": 0,
        "changed_rate": 0.0,
        "mean_delta": 0.0,
        "changed_mean_delta": 0.0,
        "max_delta": 0.0,
        "low_upward_crossings": 0,
        "normal_downward_crossings": 0,
        "band_crossings": 0,
        "target_delta_error": target_delta,
    }
    best_objective = (0, 0, 0, 0, 0.0, 0.0)
    for selected in product(*issue_candidates):
        reorders = tuple(reorder for reorder in selected if reorder is not None)
        if not reorders:
            continue
        virtual = _copy_utility_with_issue_reorders(ufun, reorders)
        after = _normalized_outcome_scores(virtual, outcomes)
        metrics = _low_boundary_swap_metrics(
            before,
            after,
            lower_boundary=lower_boundary,
            upper_boundary=upper_boundary,
            target_delta=target_delta,
        )
        objective = (
            -int(metrics["band_crossings"]),
            int(metrics["changed_outcomes"]),
            len(reorders),
            -float(metrics["target_delta_error"]),
            -float(metrics["max_delta"]),
            float(metrics["changed_rate"]),
        )
        if objective > best_objective:
            best_objective = objective
            best_ufun = virtual
            best_info = {
                **metrics,
                "changed_issue_count": len(reorders),
                "reorders": [
                    "i"
                    + str(issue_index)
                    + ": "
                    + ", ".join(
                        f"{value!r}->{score:.6f}"
                        for value, score in sorted(
                            reorder.items(), key=lambda item: str(item[0])
                        )
                    )
                    for issue_index, reorder in reorders
                ],
            }
    return best_ufun, best_info


def _build_reservation_band_active_swap_utility(
    ufun,
    outcomes: list[Outcome],
    *,
    upper_delta: float = 0.1,
) -> tuple[object | None, dict[str, object]]:
    if ufun is None or not outcomes:
        return ufun, {}
    lower_boundary = float(getattr(ufun, "reserved_value", 0.0) or 0.0)
    upper_boundary = lower_boundary + upper_delta
    issue_candidates = _outcome_reorder_candidates(ufun)
    before = _outcome_utility_scores(ufun, outcomes)
    best_ufun = ufun
    best_info: dict[str, object] = {
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
        "upper_delta": upper_delta,
        "changed_issue_count": 0,
        "reorders": [],
        "changed_outcomes": 0,
        "changed_rate": 0.0,
        "mean_delta": 0.0,
        "changed_mean_delta": 0.0,
        "max_delta": 0.0,
        "low_upward_crossings": 0,
        "normal_downward_crossings": 0,
        "band_crossings": 0,
        "target_delta_error": upper_delta,
    }
    best_objective = (0, 0, 0, 0.0, 0.0, 0)
    for selected in product(*issue_candidates):
        reorders = tuple(reorder for reorder in selected if reorder is not None)
        if not reorders:
            continue
        virtual = _copy_utility_with_issue_reorders(ufun, reorders)
        after = _outcome_utility_scores(virtual, outcomes)
        metrics = _low_boundary_swap_metrics(
            before,
            after,
            lower_boundary=lower_boundary,
            upper_boundary=upper_boundary,
            target_delta=upper_delta,
        )
        objective = (
            -int(metrics["band_crossings"]),
            int(metrics["changed_outcomes"]),
            len(reorders),
            -float(metrics["target_delta_error"]),
            -float(metrics["max_delta"]),
            int(metrics["normal_downward_crossings"]),
        )
        if objective > best_objective:
            best_objective = objective
            best_ufun = virtual
            best_info = {
                **metrics,
                "upper_delta": upper_delta,
                "changed_issue_count": len(reorders),
                "reorders": [
                    "i"
                    + str(issue_index)
                    + ": "
                    + ", ".join(
                        f"{value!r}->{score:.6f}"
                        for value, score in sorted(
                            reorder.items(), key=lambda item: str(item[0])
                        )
                    )
                    for issue_index, reorder in reorders
                ],
            }
    return best_ufun, best_info


class LinearTimeIssueValueModel(UFunModel):
    """Frequency model with linear time weights from 0.9 to 0.6."""

    name = "all_0_5__linear_0_9_to_0_6__sum_normalized__no_cont"

    def __init__(self, *, start: float = 0.9, end: float = 0.6) -> None:
        super().__init__()
        self._start = start
        self._end = end
        self._issue_values: list[list[object]] = []
        self._scores: list[dict[object, float]] = []

    def set_issue_values(self, outcomes: list[Outcome]) -> None:
        if not outcomes:
            self._issue_values = []
            self._scores = []
            return
        issue_count = len(outcomes[0])
        values = [
            sorted({outcome[index] for outcome in outcomes}, key=str)
            for index in range(issue_count)
        ]
        self._issue_values = values
        self._scores = [defaultdict(float) for _ in values]

    def update_from_opponent_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        if offer is None or max_steps <= 0:
            return
        if not self._scores:
            self.set_issue_values([offer])
        score = self._time_score(step, max_steps)
        for issue_index, value in enumerate(offer):
            if issue_index >= len(self._scores):
                continue
            self._scores[issue_index][value] += score

    def penalize_rejected_self_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        _ = offer, step, max_steps

    def eval(self, offer: Outcome) -> float:
        if offer is None or not offer:
            return 0.5
        values = [
            self._issue_value_score(issue_index, value)
            for issue_index, value in enumerate(offer)
        ]
        return sum(values) / len(values) if values else 0.5

    def eval_normalized(self, offer: Outcome) -> float:
        return self.eval(offer)

    def __call__(self, offer: Outcome) -> float:
        return self.eval(offer)

    def _time_score(self, step: int, max_steps: int) -> float:
        progress = max(0.0, min(1.0, step / max(1, max_steps - 1)))
        return self._start + (self._end - self._start) * progress

    def _issue_value_score(self, issue_index: int, value: object) -> float:
        if issue_index >= len(self._scores):
            return 0.5
        value_scores = self._scores[issue_index]
        candidates = self._issue_values[issue_index]
        raw = {candidate: value_scores.get(candidate, 0.0) for candidate in candidates}
        if value not in raw:
            raw[value] = value_scores.get(value, 0.0)
        if not raw:
            return 0.5
        mn = min(raw.values())
        mx = max(raw.values())
        if mx - mn <= 1e-12:
            time_component = 0.5
        else:
            time_component = (raw.get(value, mn) - mn) / (mx - mn)
        return max(0.0, min(1.0, (0.5 + time_component) / 2.0))


class RejectionPenaltyLinearTimeIssueValueModel(LinearTimeIssueValueModel):
    """Linear-time model that lowers values from rejected self offers."""

    def __init__(
        self,
        *,
        first_half_penalty: float,
        second_half_penalty: float,
        start: float = 0.9,
        end: float = 0.6,
    ) -> None:
        super().__init__(start=start, end=end)
        self._first_half_penalty = first_half_penalty
        self._second_half_penalty = second_half_penalty
        self.name = (
            "all_0_5__linear_0_9_to_0_6__sum_normalized__reject_"
            f"{first_half_penalty:.1f}_{second_half_penalty:.1f}"
        )

    def penalize_rejected_self_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        if offer is None or max_steps <= 0:
            return
        if not self._scores:
            self.set_issue_values([offer])
        progress = max(0.0, min(1.0, step / max(1, max_steps - 1)))
        penalty = (
            self._first_half_penalty
            if progress < 0.5
            else self._second_half_penalty
        )
        if penalty <= 0.0:
            return
        for issue_index, value in enumerate(offer):
            if issue_index >= len(self._scores):
                continue
            self._scores[issue_index][value] -= penalty


class PresenceHybridZeroSumIssueValueModel(UFunModel):
    """Zero-sum prior plus observed issue values with presence smoothing."""

    def __init__(
        self,
        *,
        name: str,
        prior: str,
        prior_weight: float,
        start: float = 0.9,
        end: float = 0.6,
    ) -> None:
        super().__init__()
        self.name = name
        self._prior = prior
        self._prior_weight = prior_weight
        self._start = start
        self._end = end
        self._zero_sum = ZeroSumModel()
        self._issue_values: list[list[object]] = []
        self._scores: list[dict[object, float]] = []
        self._presence: list[dict[object, float]] = []

    def on_preferences_changed(self, changes) -> None:
        self._zero_sum.set_negotiator(self.negotiator)
        self._zero_sum.on_preferences_changed(changes)

    def set_issue_values(self, outcomes: list[Outcome]) -> None:
        if not outcomes:
            self._issue_values = []
            self._scores = []
            self._presence = []
            return
        issue_count = len(outcomes[0])
        values = [
            sorted({outcome[index] for outcome in outcomes}, key=str)
            for index in range(issue_count)
        ]
        self._issue_values = values
        self._scores = [defaultdict(float) for _ in values]
        self._presence = [defaultdict(float) for _ in values]

    def update_from_opponent_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        if offer is None or max_steps <= 0:
            return
        if not self._scores:
            self.set_issue_values([offer])
        score = self._time_score(step, max_steps)
        for issue_index, value in enumerate(offer):
            if issue_index >= len(self._scores):
                continue
            self._scores[issue_index][value] += score
            self._presence[issue_index][value] = 1.0

    def penalize_rejected_self_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        _ = offer, step, max_steps

    def penalize_high_confidence_rejected_offer(
        self, offer: Outcome | None, *, penalty: float = 0.25
    ) -> None:
        if offer is None or penalty <= 0.0:
            return
        if not self._scores:
            self.set_issue_values([offer])
        for issue_index, value in enumerate(offer):
            if issue_index >= len(self._scores):
                continue
            self._scores[issue_index][value] -= penalty

    def eval(self, offer: Outcome) -> float:
        observed = self._observed_score(offer)
        prior = self._prior_score(offer)
        return max(
            0.0,
            min(1.0, self._prior_weight * prior + (1.0 - self._prior_weight) * observed),
        )

    def eval_normalized(self, offer: Outcome) -> float:
        return self.eval(offer)

    def __call__(self, offer: Outcome) -> float:
        return self.eval(offer)

    def _prior_score(self, offer: Outcome) -> float:
        if self._prior == "all_0_5":
            return 0.5
        try:
            zero_sum = float(self._zero_sum.eval_normalized(offer))
        except Exception:
            zero_sum = 0.5
        if self._prior == "weak_zero_sum":
            return 0.5 + 0.5 * (zero_sum - 0.5)
        return zero_sum

    def _observed_score(self, offer: Outcome) -> float:
        if offer is None or not offer:
            return 0.5
        values = [
            self._issue_value_score(issue_index, value)
            for issue_index, value in enumerate(offer)
        ]
        return sum(values) / len(values) if values else 0.5

    def _time_score(self, step: int, max_steps: int) -> float:
        progress = max(0.0, min(1.0, step / max(1, max_steps - 1)))
        return self._start + (self._end - self._start) * progress

    def _issue_value_score(self, issue_index: int, value: object) -> float:
        if issue_index >= len(self._scores):
            return 0.5
        candidates = self._issue_values[issue_index]
        weighted_raw = {
            candidate: self._scores[issue_index].get(candidate, 0.0)
            for candidate in candidates
        }
        presence_raw = {
            candidate: self._presence[issue_index].get(candidate, 0.0)
            for candidate in candidates
        }
        if value not in weighted_raw:
            weighted_raw[value] = self._scores[issue_index].get(value, 0.0)
            presence_raw[value] = self._presence[issue_index].get(value, 0.0)
        weighted = self._normalize_value(weighted_raw, value)
        presence = self._normalize_value(presence_raw, value)
        return max(0.0, min(1.0, 0.75 * weighted + 0.25 * presence))

    def _normalize_value(self, raw: dict[object, float], value: object) -> float:
        if not raw:
            return 0.5
        mn = min(raw.values())
        mx = max(raw.values())
        if mx - mn <= 1e-12:
            return 0.5
        return (raw.get(value, mn) - mn) / (mx - mn)


class YasasiOpponentUFunModel(PresenceHybridZeroSumIssueValueModel):
    """Official-facing opponent utility estimate submitted by YasasiV2Agent."""

    name = "yasasi_v2_opponent_ufun__zero_sum_presence_hybrid__prior_0.75"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            prior="zero_sum",
            prior_weight=0.75,
            start=0.9,
            end=0.6,
        )


class YasasiSwapOpponentUFunModel(PresenceHybridZeroSumIssueValueModel):
    """Official-facing opponent utility estimate submitted by Yasasi swap agents."""

    name = "yasasi_swap_opponent_ufun__zero_sum_presence_hybrid__prior_0.75"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            prior="zero_sum",
            prior_weight=0.75,
            start=0.9,
            end=0.6,
        )


class YasasiV4OpponentUFunModel(PresenceHybridZeroSumIssueValueModel):
    """Official-facing opponent utility estimate submitted by YasasiV4Agent."""

    name = "yasasi_v4_opponent_ufun__all_0_5_presence_hybrid__prior_0.75"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            prior="all_0_5",
            prior_weight=0.75,
            start=0.9,
            end=0.6,
        )


class YasasiV5OpponentUFunModel(PresenceHybridZeroSumIssueValueModel):
    """Official-facing opponent utility estimate submitted by YasasiV5Agent."""

    name = "yasasi_v5_opponent_ufun__all_0_5_presence_hybrid__prior_0.75"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            prior="all_0_5",
            prior_weight=0.75,
            start=0.9,
            end=0.6,
        )


class KingV3OpponentUFunModel(PresenceHybridZeroSumIssueValueModel):
    """Official-facing opponent utility estimate submitted by KingV3Agent."""

    name = "king_v3_opponent_ufun__zero_sum_presence_hybrid__prior_0.75"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            prior="zero_sum",
            prior_weight=0.75,
            start=0.9,
            end=0.6,
        )


class MywayAgent(ThresholdRandomInferenceAgent):
    """Threshold-random agent using the best mixed-scenario inference model."""

    model_name = LinearTimeIssueValueModel.name
    selection_name = "select_threshold_0_8_to_0_3_random_no_repeat"
    acceptance_name = "accept_ge_0_8_to_0_4"
    use_opponent_filter = False
    rejection_penalties: tuple[float, float] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = self._make_model()
        self.private_info["opponent_ufun"] = self._model

    def _make_model(self) -> LinearTimeIssueValueModel:
        if self.rejection_penalties is None:
            return LinearTimeIssueValueModel()
        first, second = self.rejection_penalties
        return RejectionPenaltyLinearTimeIssueValueModel(
            first_half_penalty=first,
            second_half_penalty=second,
        )

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._model = self._make_model()
        self._model.set_negotiator(self)
        self._model.on_preferences_changed(changes)
        self._model.set_issue_values(self._outcomes)
        self.model_name = self._model.name
        self.private_info["opponent_ufun"] = self._model
        self._publish_private_info("unknown")

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None
        threshold = self._selection_threshold(state)
        candidates = self._candidates_above_threshold(threshold)
        if not candidates:
            offer = max(self._outcomes, key=lambda outcome: float(self.ufun(outcome)))
        else:
            unused = [
                outcome for outcome in candidates if outcome not in self._used_in_cycle
            ]
            if not unused:
                self._used_in_cycle = set()
                unused = list(candidates)
            if self.use_opponent_filter:
                favorable = [
                    outcome
                    for outcome in unused
                    if self._normalized_self_score(outcome)
                    > float(self._model.eval(outcome))
                ]
                if favorable:
                    unused = favorable
            offer = random.choice(unused)

        self._made_first_offer = True
        self._used_in_cycle.add(offer)
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False
        self._model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_utility = float(self.ufun(offer))
        if offered_utility >= self._acceptance_threshold(state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _selection_threshold(self, state: SAOState) -> float:
        return self._linear_threshold(state, start=0.8, end=0.3)

    def _acceptance_threshold(self, state: SAOState) -> float:
        return self._linear_threshold(state, start=0.8, end=0.4)

    def _threshold(self, state: SAOState) -> float:
        return self._acceptance_threshold(state)

    def _normalized_self_score(self, outcome: Outcome) -> float:
        if self.ufun is None:
            return 0.5
        mn, mx = self.ufun.minmax(above_reserve=False)
        denom = float(mx) - float(mn)
        if denom <= 1e-12:
            return 0.5
        return max(0.0, min(1.0, (float(self.ufun(outcome)) - float(mn)) / denom))

    def _linear_threshold(self, state: SAOState, *, start: float, end: float) -> float:
        relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        if relative_time <= 0.0 and self.nmi is not None and self.nmi.n_steps:
            relative_time = int(state.step) / max(1, int(self.nmi.n_steps) - 1)
        relative_time = max(0.0, min(1.0, relative_time))
        return start + (end - start) * relative_time

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_ufun"] = self._model
        self.private_info["opponent_strategy_analysis"][
            "inference_model"
        ] = self.model_name
        self.private_info["opponent_strategy_analysis"][
            "selection"
        ] = self.selection_name
        self.private_info["opponent_strategy_analysis"][
            "acceptance"
        ] = self.acceptance_name
        self.private_info["active_opponent_model"] = self.model_name + ":" + pattern
        self.private_info["evaluation_opponent_model"] = self.model_name


class MywayV2Agent(MywayAgent):
    """Myway variant using normalized 0.8->0.5 floors and no rejection penalty."""

    selection_name = "select_normalized_threshold_0_8_to_0_5_random_no_repeat"
    acceptance_name = "accept_normalized_ge_0_8_to_0_5"
    rejection_penalties: tuple[float, float] | None = None

    def _selection_threshold(self, state: SAOState) -> float:
        return self._normalized_linear_threshold(state, start=0.8, end=0.5)

    def _acceptance_threshold(self, state: SAOState) -> float:
        return self._normalized_linear_threshold(state, start=0.8, end=0.5)

    def _normalized_linear_threshold(
        self, state: SAOState, *, start: float, end: float
    ) -> float:
        normalized = self._linear_threshold(state, start=start, end=end)
        if self.ufun is None:
            return normalized
        mn, mx = self.ufun.minmax(above_reserve=False)
        denom = float(mx) - float(mn)
        if denom <= 1e-12:
            return float(mn)
        return float(mn) + normalized * denom


class KingAgent(MywayAgent):
    """Top-first agent with late opponent-model-ranked proposal reuse."""

    model_name = "zero_sum__linear_0_9_to_0_6__presence_sum_hybrid__prior_0.75"
    negotiation_model_name = (
        "zero_sum__linear_0.8_to_0.5__presence_0.25__prior_0.55"
    )
    selection_name = (
        "select_top_first_half_then_opp_ranked_own_ge_0_7_reuse_after_10_final_best_opp"
    )
    acceptance_name = "accept_normalized_ge_0_7"
    rejection_penalties: tuple[float, float] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._negotiation_model = self._make_negotiation_model()
        self._opponent_offers: list[Outcome] = []
        self._proposal_steps: dict[Outcome, int] = {}

    def _make_model(self) -> PresenceHybridZeroSumIssueValueModel:
        return PresenceHybridZeroSumIssueValueModel(
            name=self.model_name,
            prior="zero_sum",
            prior_weight=0.75,
        )

    def _make_negotiation_model(self) -> PresenceHybridZeroSumIssueValueModel:
        return PresenceHybridZeroSumIssueValueModel(
            name=self.negotiation_model_name,
            prior="zero_sum",
            prior_weight=0.55,
            start=0.8,
            end=0.5,
        )

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._negotiation_model = self._make_negotiation_model()
        self._negotiation_model.set_negotiator(self)
        self._negotiation_model.on_preferences_changed(changes)
        self._negotiation_model.set_issue_values(self._outcomes)
        self.private_info["opponent_ufun"] = self._model
        self._opponent_offers = []
        self._proposal_steps = {}
        self._publish_private_info("unknown")

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None

        final_opponent_offer = self._best_final_opponent_offer(state)
        if final_opponent_offer is not None:
            offer = final_opponent_offer
        elif self._is_first_half(state):
            offer = self._top_own_offer()
        else:
            offer = self._best_late_offer(state)

        self._made_first_offer = True
        self._used_in_cycle.add(offer)
        self._proposal_steps[offer] = int(state.step)
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            if (
                self._last_self_step >= self._max_steps() / 2
                and self._last_self_offer is not None
                and float(self._negotiation_model.eval(self._last_self_offer)) >= 0.7
            ):
                self._negotiation_model.penalize_high_confidence_rejected_offer(
                    self._last_self_offer
                )
            else:
                self._negotiation_model.penalize_rejected_self_offer(
                    self._last_self_offer, self._last_self_step, self._max_steps()
                )
            self._last_self_offer_pending = False
        self._model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_utility = float(self.ufun(offer))
        if offered_utility >= self._acceptance_threshold(state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _acceptance_threshold(self, state: SAOState) -> float:
        _ = state
        return self._normalized_constant_threshold(0.7)

    def _is_first_half(self, state: SAOState) -> bool:
        return int(state.step) < self._max_steps() / 2

    def _top_own_offer(self) -> Outcome:
        return max(
            self._outcomes,
            key=lambda outcome: (float(self.ufun(outcome)), str(outcome)),
        )

    def _best_final_opponent_offer(self, state: SAOState) -> Outcome | None:
        if int(state.step) < max(0, self._max_steps() - 1):
            return None
        if not self._opponent_offers:
            return None
        return max(
            self._opponent_offers,
            key=lambda outcome: (self._normalized_self_score(outcome), str(outcome)),
        )

    def _best_late_offer(self, state: SAOState) -> Outcome:
        candidates = [
            outcome
            for outcome in self._outcomes
            if self._normalized_self_score(outcome) >= 0.7
        ]
        if not candidates:
            return self._top_own_offer()
        reusable = [
            outcome
            for outcome in candidates
            if self._can_reuse_offer(outcome, int(state.step))
        ]
        if reusable:
            candidates = reusable
        return max(
            candidates,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _can_reuse_offer(self, offer: Outcome, step: int) -> bool:
        previous = self._proposal_steps.get(offer)
        return previous is None or step - previous >= 10

    def _normalized_constant_threshold(self, normalized: float) -> float:
        if self.ufun is None:
            return normalized
        mn, mx = self.ufun.minmax(above_reserve=False)
        denom = float(mx) - float(mn)
        if denom <= 1e-12:
            return float(mn)
        return float(mn) + normalized * denom

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_ufun"] = self._model
        self.private_info["active_opponent_model"] = self.negotiation_model_name
        self.private_info["evaluation_opponent_model"] = self.model_name
        self.private_info["opponent_strategy_analysis"][
            "negotiation_model"
        ] = self.negotiation_model_name


class QueenAgent(KingAgent):
    """King behavior with private opponent inference kept out of submitted scoring."""

    model_name = "hidden_" + KingAgent.model_name

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hide_submitted_model()

    def _hide_submitted_model(self) -> None:
        self.private_info["opponent_ufun"] = None
        self.private_info["evaluation_opponent_model"] = ""
        self.private_info.setdefault("opponent_strategy_analysis", {})[
            "inference_submission"
        ] = "hidden"

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self._hide_submitted_model()


class KingAgentV2(KingAgent):
    """King variant with dynamic threshold and narrow opponent-ranked offers."""

    selection_name = (
        "select_dynamic_threshold_window_0_1_no_repeat_exclude_opponent_offers"
    )
    acceptance_name = "accept_dynamic_normalized_threshold"

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None

        if self._is_opening_phase(state):
            offer = self._top_own_offer()
        else:
            offer = self._best_window_offer(state)

        self._made_first_offer = True
        self._used_in_cycle.add(offer)
        self._proposal_steps[offer] = int(state.step)
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def _acceptance_threshold(self, state: SAOState) -> float:
        if self._is_endgame_phase(state):
            return self._normalized_constant_threshold(0.25)
        return self._normalized_constant_threshold(self._acceptance_dynamic_threshold(state))

    def _is_opening_phase(self, state: SAOState) -> bool:
        return self._progress(state) < 0.2

    def _is_endgame_phase(self, state: SAOState) -> bool:
        return int(state.step) >= max(0, self._configured_max_steps() - 2)

    def _best_window_offer(self, state: SAOState) -> Outcome:
        threshold = self._dynamic_threshold(state)
        upper = threshold + 0.1
        opponent_offers = set(self._opponent_offers)
        candidates = [
            outcome
            for outcome in self._outcomes
            if threshold <= self._normalized_self_score(outcome) <= upper
            and outcome not in opponent_offers
        ]
        if not candidates:
            return self._top_own_offer()

        unused = [
            outcome for outcome in candidates if outcome not in self._used_in_cycle
        ]
        if unused:
            candidates = unused

        return max(
            candidates,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _dynamic_threshold(self, state: SAOState) -> float:
        max_steps = self._configured_max_steps()
        if max_steps <= 0:
            return 1.0
        progress = self._progress(state)
        if progress < 0.2:
            return 1.0
        if progress < 0.9:
            threshold = 0.8 - 0.1 * exp((progress - 0.9) * 4.0)
        else:
            threshold = 0.8 - 0.1 * exp((progress - 0.9) * 11.0)
        return max(0.0, min(1.0, threshold))

    def _acceptance_dynamic_threshold(self, state: SAOState) -> float:
        if self._progress(state) < 0.2:
            return 0.9
        return self._dynamic_threshold(state)

    def _progress(self, state: SAOState) -> float:
        max_steps = self._configured_max_steps()
        if max_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, int(state.step) / max_steps))

    def _configured_max_steps(self) -> int:
        if self.nmi is not None and self.nmi.n_steps is not None:
            return int(self.nmi.n_steps)
        return self._max_steps()

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_strategy_analysis"][
            "selection"
        ] = self.selection_name
        self.private_info["opponent_strategy_analysis"][
            "acceptance"
        ] = self.acceptance_name


class KingV3Agent(KingAgent):
    """King with a relaxed second-half 0.7 -> 0.5 threshold schedule."""

    model_name = KingV3OpponentUFunModel.name
    negotiation_model_name = (
        "king_v3_negotiation__zero_sum_presence_hybrid__0_8_to_0_5__prior_0.55"
    )
    selection_name = "king_v3_top_first_then_opp_ranked_threshold_0_7_to_0_5_until_0_8"
    acceptance_name = "accept_all_offers_above_king_v3_threshold"

    def _make_model(self) -> KingV3OpponentUFunModel:
        return KingV3OpponentUFunModel()

    def _make_negotiation_model(self) -> PresenceHybridZeroSumIssueValueModel:
        return PresenceHybridZeroSumIssueValueModel(
            name=self.negotiation_model_name,
            prior="zero_sum",
            prior_weight=0.55,
            start=0.8,
            end=0.5,
        )

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None

        if self._is_first_half(state):
            offer = self._top_own_offer()
        else:
            offer = self._best_late_offer(state)

        self._made_first_offer = True
        self._used_in_cycle.add(offer)
        self._proposal_steps[offer] = int(state.step)
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def _acceptance_threshold(self, state: SAOState) -> float:
        return self._normalized_constant_threshold(self._scheduled_threshold(state))

    def _best_late_offer(self, state: SAOState) -> Outcome:
        threshold = self._scheduled_threshold(state)
        candidates = [
            outcome
            for outcome in self._outcomes
            if self._normalized_self_score(outcome) >= threshold
        ]
        if not candidates:
            return self._top_own_offer()

        reusable = [
            outcome
            for outcome in candidates
            if self._can_reuse_offer(outcome, int(state.step))
        ]
        if reusable:
            candidates = reusable

        return max(
            candidates,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _scheduled_threshold(self, state: SAOState) -> float:
        progress = self._progress(state)
        if progress < 0.5:
            return 0.7
        if progress < 0.8:
            phase = (progress - 0.5) / 0.3
            return 0.7 + (0.5 - 0.7) * max(0.0, min(1.0, phase))
        return 0.5

    def _progress(self, state: SAOState) -> float:
        max_steps = self._configured_max_steps()
        if max_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, int(state.step) / max_steps))

    def _configured_max_steps(self) -> int:
        if self.nmi is not None and self.nmi.n_steps is not None:
            return int(self.nmi.n_steps)
        return self._max_steps()

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_ufun"] = self._model
        self.private_info["evaluation_opponent_model"] = self._model.name
        self.private_info["active_opponent_model"] = self.negotiation_model_name
        self.private_info["opponent_strategy_analysis"][
            "official_evaluation_model"
        ] = self._model.name
        self.private_info["opponent_strategy_analysis"][
            "selection"
        ] = self.selection_name
        self.private_info["opponent_strategy_analysis"][
            "acceptance"
        ] = self.acceptance_name


class QueenAgentV2(KingAgentV2):
    """KingAgentV2 behavior with opponent inference hidden from submitted scoring."""

    model_name = "hidden_" + KingAgentV2.model_name

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hide_submitted_model()

    def _hide_submitted_model(self) -> None:
        self.private_info["opponent_ufun"] = None
        self.private_info["evaluation_opponent_model"] = ""
        self.private_info.setdefault("opponent_strategy_analysis", {})[
            "inference_submission"
        ] = "hidden"

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self._hide_submitted_model()


class YASASHIAgent(KingAgent):
    """Scheduled descending proposer using King-style opponent inference."""

    selection_name = "yasashi_descending_coverage_bands_then_opponent_friendly"
    acceptance_name = "accept_if_offer_ge_next_self_offer_final_accept"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mid_phase_start_score: float | None = None
        self._self_score_cache: dict[Outcome, float] = {}
        self._sorted_outcomes_by_self_score: list[tuple[float, Outcome]] = []
        self._proposal_value_counts: list[dict[object, int]] = []

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._mid_phase_start_score = None
        self._rebuild_yasashi_caches()

    def _rebuild_yasashi_caches(self) -> None:
        self._self_score_cache = {
            outcome: super(YASASHIAgent, self)._normalized_self_score(outcome)
            for outcome in self._outcomes
        }
        self._sorted_outcomes_by_self_score = sorted(
            ((score, outcome) for outcome, score in self._self_score_cache.items()),
            key=lambda item: (item[0], str(item[1])),
        )
        issue_count = len(self._outcomes[0]) if self._outcomes else 0
        self._proposal_value_counts = [defaultdict(int) for _ in range(issue_count)]

    def _record_yasashi_proposal(self, offer: Outcome, step: int) -> None:
        self._used_in_cycle.add(offer)
        self._proposal_steps[offer] = step
        if not self._proposal_value_counts:
            self._proposal_value_counts = [defaultdict(int) for _ in range(len(offer))]
        for issue_index, value in enumerate(offer):
            if issue_index < len(self._proposal_value_counts):
                self._proposal_value_counts[issue_index][value] += 1

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None

        offer = self._select_yasashi_offer(state)
        self._made_first_offer = True
        self._record_yasashi_proposal(offer, int(state.step))
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        if self._is_final_response(state):
            return ResponseType.ACCEPT_OFFER

        next_offer = self._select_yasashi_offer(state)
        offered_score = self._normalized_self_score(offer)
        next_score = self._normalized_self_score(next_offer)
        if offered_score >= next_score:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _select_yasashi_offer(self, state: SAOState) -> Outcome:
        progress = self._progress(state)
        if progress <= 0.05:
            return self._top_own_offer()
        if progress <= 0.5:
            return self._early_staged_offer(state)
        if progress <= 0.8:
            if self._mid_phase_start_score is None and self._last_self_offer is not None:
                self._mid_phase_start_score = self._normalized_self_score(
                    self._last_self_offer
                )
            start = self._mid_phase_start_score if self._mid_phase_start_score else 0.8
            return self._band_offer(state, start_t=0.5, end_t=0.8, start=start, end=0.6)
        return self._opponent_friendly_offer(state)

    def _early_staged_offer(self, state: SAOState) -> Outcome:
        progress = self._progress(state)
        phase_t = max(0.0, min(1.0, (progress - 0.05) / (0.5 - 0.05)))
        scheduled_target = 1.0 + (0.8 - 1.0) * phase_t
        stages = (
            (0.8, 1.0, scheduled_target),
            (0.7, 0.8, 0.8),
            (0.6, 0.7, 0.7),
        )
        for lower, upper, target in stages:
            offer = self._coverage_offer_in_score_band(
                lower=lower,
                upper=upper,
                target=target,
            )
            if offer is not None:
                return offer
        return self._fallback_descending_offer()

    def _band_offer(
        self,
        state: SAOState,
        *,
        start_t: float,
        end_t: float,
        start: float,
        end: float,
    ) -> Outcome:
        progress = self._progress(state)
        phase_t = max(0.0, min(1.0, (progress - start_t) / max(1e-12, end_t - start_t)))
        target = start + (end - start) * phase_t
        upper = self._last_self_score_limit()
        candidates = self._fresh_candidates(max_score=upper)
        if not candidates:
            return self._fallback_descending_offer()

        offer = self._coverage_offer_in_score_band(
            lower=end,
            upper=max(start, upper),
            target=target,
        )
        if offer is not None:
            return offer
        return self._fallback_descending_offer()

    def _coverage_offer_in_score_band(
        self,
        *,
        lower: float,
        upper: float,
        target: float,
    ) -> Outcome | None:
        max_score = min(upper, self._last_self_score_limit())
        pool = [
            outcome
            for outcome in self._fresh_candidates(max_score=max_score)
            if lower <= self._normalized_self_score(outcome) <= max_score + 1e-12
        ]
        if not pool:
            return None
        target_pool = [
            outcome for outcome in pool if self._normalized_self_score(outcome) <= target
        ]
        if target_pool:
            pool = target_pool
        return max(
            pool,
            key=lambda outcome: (
                self._coverage_score(outcome),
                self._yasashi_distance(self._last_self_offer, outcome),
                -abs(self._normalized_self_score(outcome) - target),
                float(self._negotiation_model.eval(outcome)),
                -self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _opponent_friendly_offer(self, state: SAOState) -> Outcome:
        _ = state
        upper = self._last_self_score_limit()
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=upper)
            if self._normalized_self_score(outcome) >= 0.4
            and self._normalized_self_score(outcome)
            > float(self._negotiation_model.eval(outcome))
        ]
        if not candidates:
            return self._random_own_score_offer(min_score=0.4, max_score=1.0)
        return max(
            candidates,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _coverage_score(self, outcome: Outcome) -> float:
        score = 0.0
        for issue_index, value in enumerate(outcome):
            count = (
                self._proposal_value_counts[issue_index].get(value, 0)
                if issue_index < len(self._proposal_value_counts)
                else 0
            )
            score += 1.0 / (1.0 + count)
        return score

    def _yasashi_distance(self, reference: Outcome | None, candidate: Outcome) -> float:
        if reference is None:
            return 0.0
        return sum(
            1.0 for left, right in zip(reference, candidate, strict=True) if left != right
        )

    def _random_own_score_offer(self, *, min_score: float, max_score: float) -> Outcome:
        upper = min(max_score, self._last_self_score_limit())
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=upper)
            if min_score <= self._normalized_self_score(outcome) <= max_score
        ]
        if not candidates:
            excluded = set(self._opponent_offers)
            excluded.update(self._proposal_steps)
            candidates = [
                outcome
                for outcome in self._outcomes
                if outcome not in excluded
                and min_score <= self._normalized_self_score(outcome) <= max_score
            ]
        if not candidates:
            return self._fallback_descending_offer()
        return random.choice(sorted(candidates, key=str))

    def _fresh_candidates(self, *, max_score: float) -> list[Outcome]:
        excluded = set(self._opponent_offers)
        excluded.update(self._proposal_steps)
        return [
            outcome
            for score, outcome in self._sorted_outcomes_by_self_score
            if outcome not in excluded
            and score <= max_score + 1e-12
        ]

    def _fallback_descending_offer(self) -> Outcome:
        upper = self._last_self_score_limit()
        candidates = [
            outcome
            for outcome in self._outcomes
            if self._normalized_self_score(outcome) <= upper + 1e-12
            and outcome not in set(self._opponent_offers)
        ]
        if not candidates:
            candidates = list(self._outcomes)
        return max(
            candidates,
            key=lambda outcome: (
                self._normalized_self_score(outcome),
                float(self._negotiation_model.eval(outcome)),
                str(outcome),
            ),
        )

    def _last_self_score_limit(self) -> float:
        if self._last_self_offer is None or self._progress_from_step(self._last_self_step) <= 0.05:
            return 1.0
        return self._normalized_self_score(self._last_self_offer) - 1e-12

    def _normalized_self_score(self, outcome: Outcome) -> float:
        cached = self._self_score_cache.get(outcome)
        if cached is not None:
            return cached
        return super()._normalized_self_score(outcome)

    def _is_final_response(self, state: SAOState) -> bool:
        return int(state.step) >= max(0, self._configured_max_steps() - 1)

    def _progress(self, state: SAOState) -> float:
        return self._progress_from_step(int(state.step))

    def _progress_from_step(self, step: int) -> float:
        max_steps = self._configured_max_steps()
        if max_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, step / max_steps))

    def _configured_max_steps(self) -> int:
        if self.nmi is not None and self.nmi.n_steps is not None:
            return int(self.nmi.n_steps)
        return self._max_steps()

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_strategy_analysis"][
            "selection"
        ] = self.selection_name
        self.private_info["opponent_strategy_analysis"][
            "acceptance"
        ] = self.acceptance_name


class YasasiAgent(YASASHIAgent):
    """PascalCase submission alias for YASASHIAgent."""


class YasasiV2Agent(YASASHIAgent):
    """YASASHI without final unconditional acceptance and with a late 0.45 floor."""

    selection_name = (
        "yasashi_v2_descending_coverage_bands_then_opponent_friendly_floor_0_45"
    )
    acceptance_name = (
        "accept_if_offer_ge_next_self_offer_late_floor_0_45_no_final_accept_all"
    )
    late_floor = 0.45

    def _make_model(self) -> YasasiOpponentUFunModel:
        return YasasiOpponentUFunModel()

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_ufun"] = self._model
        self.private_info["evaluation_opponent_model"] = self._model.name
        self.private_info["opponent_strategy_analysis"][
            "official_evaluation_model"
        ] = self._model.name

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(offer, int(state.step), self._max_steps())
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        next_offer = self._select_yasashi_offer(state)
        offered_score = self._normalized_self_score(offer)
        next_score = self._normalized_self_score(next_offer)
        threshold = max(next_score, self._late_acceptance_floor(state))
        if offered_score >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _late_acceptance_floor(self, state: SAOState) -> float:
        if self._progress(state) >= 0.8:
            return self.late_floor
        return 0.0

    def _opponent_friendly_offer(self, state: SAOState) -> Outcome:
        _ = state
        upper = self._last_self_score_limit()
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=upper)
            if self.late_floor <= self._normalized_self_score(outcome)
            and self._normalized_self_score(outcome)
            > float(self._negotiation_model.eval(outcome))
        ]
        if not candidates:
            return self._late_floor_fallback_offer(max_score=upper)
        return max(
            candidates,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def _late_floor_fallback_offer(self, *, max_score: float) -> Outcome:
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=max_score)
            if self._normalized_self_score(outcome) >= self.late_floor
        ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if self._normalized_self_score(outcome) >= self.late_floor
            ]
        if not candidates:
            return self._top_own_offer()
        return max(
            candidates,
            key=lambda outcome: (
                self._normalized_self_score(outcome),
                float(self._negotiation_model.eval(outcome)),
                str(outcome),
            ),
        )


class YasasiV3Agent(YasasiV2Agent):
    """YASASHI V3 with an 85% phase split and reservation-safe endgame."""

    selection_name = "yasashi_v3_to_0_65_until_0_85_then_best_self_over_opp_unseen"
    acceptance_name = "accept_next_offer_until_0_85_then_accept_above_reservation"
    mid_floor = 0.65
    endgame_start = 0.85

    def _select_yasashi_offer(self, state: SAOState) -> Outcome:
        progress = self._progress(state)
        if progress >= self.endgame_start:
            return self._endgame_best_unseen_offer(state)
        if progress <= 0.05:
            return self._top_own_offer()
        if progress <= 0.5:
            return self._early_staged_offer(state)

        if self._mid_phase_start_score is None and self._last_self_offer is not None:
            self._mid_phase_start_score = self._normalized_self_score(
                self._last_self_offer
            )
        start = self._mid_phase_start_score if self._mid_phase_start_score else 0.8
        return self._band_offer(
            state,
            start_t=0.5,
            end_t=self.endgame_start,
            start=start,
            end=self.mid_floor,
        )

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(offer, int(state.step), self._max_steps())
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_utility = float(self.ufun(offer))
        if self._progress(state) >= self.endgame_start:
            if self._is_above_reservation(offered_utility):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        next_offer = self._select_yasashi_offer(state)
        offered_score = self._normalized_self_score(offer)
        next_score = self._normalized_self_score(next_offer)
        if offered_score >= next_score:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _endgame_best_unseen_offer(self, state: SAOState) -> Outcome:
        _ = state
        candidates = [
            outcome
            for outcome in self._outcomes
            if outcome not in self._proposal_steps
            and self._is_above_reservation(float(self.ufun(outcome)))
            and self._normalized_self_score(outcome)
            > float(self._negotiation_model.eval(outcome))
        ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if self._is_above_reservation(float(self.ufun(outcome)))
                and self._normalized_self_score(outcome)
                > float(self._negotiation_model.eval(outcome))
            ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if outcome not in self._proposal_steps
                and self._is_above_reservation(float(self.ufun(outcome)))
            ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if self._is_above_reservation(float(self.ufun(outcome)))
            ]
        if not candidates:
            return self._top_own_offer()
        return max(
            candidates,
            key=lambda outcome: (
                self._normalized_self_score(outcome),
                -float(self._negotiation_model.eval(outcome)),
                str(outcome),
            ),
        )

    def _is_above_reservation(self, utility: float) -> bool:
        if self.ufun is None:
            return False
        reserved = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        return utility >= reserved


class YasasiSwapAgent(YASASHIAgent):
    """YASASHI agent that evaluates self utility through a virtual rank swap.

    The negotiation schedule is the same as YASASHIAgent, but every self-score
    comparison uses a virtual utility where each issue swaps the 2nd and 3rd
    ranked values. The top-ranked value remains unchanged.
    """

    selection_name = (
        "yasashi_virtual_rank_swap_descending_coverage_bands_then_opponent_friendly"
    )
    acceptance_name = (
        "accept_if_virtual_rank_swap_offer_ge_next_self_offer_final_accept"
    )
    swap_ranks = (2, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._virtual_ufun = None

    def _make_model(self) -> YasasiSwapOpponentUFunModel:
        return YasasiSwapOpponentUFunModel()

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._virtual_ufun = _build_virtual_utility(self.ufun, *self.swap_ranks)

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_ufun"] = self._model
        self.private_info["evaluation_opponent_model"] = self._model.name
        self.private_info["opponent_strategy_analysis"][
            "official_evaluation_model"
        ] = self._model.name

    def _selection_ufun(self):
        return self._virtual_ufun if self._virtual_ufun is not None else self.ufun

    def _top_own_offer(self) -> Outcome:
        score_ufun = self._selection_ufun()
        if score_ufun is None:
            return super()._top_own_offer()
        return max(
            self._outcomes,
            key=lambda outcome: (float(score_ufun(outcome)), str(outcome)),
        )

    def _normalized_self_score(self, outcome: Outcome) -> float:
        score_ufun = self._selection_ufun()
        if score_ufun is None:
            return 0.5
        mn, mx = score_ufun.minmax(above_reserve=False)
        denom = float(mx) - float(mn)
        if denom <= 1e-12:
            return 0.5
        return max(0.0, min(1.0, (float(score_ufun(outcome)) - float(mn)) / denom))


class YasasiSwap23Agent(YasasiSwapAgent):
    """YASASHI swap variant that swaps ranks 2 and 3."""

    selection_name = (
        "yasashi_virtual_rank_swap_2_3_descending_coverage_bands_then_opponent_friendly"
    )
    acceptance_name = "accept_if_virtual_rank_swap_2_3_offer_ge_next_self_offer"
    swap_ranks = (2, 3)


class YasasiSwap24Agent(YasasiSwapAgent):
    """YASASHI swap variant that swaps ranks 2 and 4."""

    selection_name = (
        "yasashi_virtual_rank_swap_2_4_descending_coverage_bands_then_opponent_friendly"
    )
    acceptance_name = "accept_if_virtual_rank_swap_2_4_offer_ge_next_self_offer"
    swap_ranks = (2, 4)


class YasasiSwap34Agent(YasasiSwapAgent):
    """YASASHI swap variant that swaps ranks 3 and 4."""

    selection_name = (
        "yasashi_virtual_rank_swap_3_4_descending_coverage_bands_then_opponent_friendly"
    )
    acceptance_name = "accept_if_virtual_rank_swap_3_4_offer_ge_next_self_offer"
    swap_ranks = (3, 4)


class YasasiOutcomeSafeSwapAgent(YasasiSwapAgent):
    """YASASHI with active reorders that preserve the 0.4-0.45 buffer band."""

    selection_name = (
        "yasashi_buffer_band_reorder_0_4_to_0_45_target_delta_0_1"
    )
    acceptance_name = "accept_if_buffer_band_reorder_offer_ge_next_self_offer"
    lower_boundary = 0.4
    upper_boundary = 0.45
    target_delta = 0.1

    def on_preferences_changed(self, changes):
        YASASHIAgent.on_preferences_changed(self, changes)
        self._virtual_ufun, self._safe_swap_info = (
            _build_low_boundary_active_swap_utility(
                self.ufun,
                self._outcomes,
                lower_boundary=self.lower_boundary,
                upper_boundary=self.upper_boundary,
                target_delta=self.target_delta,
            )
        )
        self.private_info.setdefault("opponent_strategy_analysis", {})[
            "utility_swap"
        ] = self._safe_swap_info

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_strategy_analysis"][
            "utility_swap"
        ] = getattr(self, "_safe_swap_info", {})


class YasasiV4Agent(YasasiSwapAgent):
    """YASASHI V4 with reservation-band virtual utility and late coverage."""

    model_name = "yasasi_v4_opponent_ufun__all_0_5_presence_hybrid__prior_0.75"
    negotiation_model_name = (
        "all_0_5__linear_0.8_to_0.5__presence_0.25__prior_0.55"
    )
    selection_name = (
        "yasasi_v4_reservation_band_swap_1_0_to_0_7_then_opp_0_4_coverage"
    )
    acceptance_name = "accept_ge_0_7_until_0_8_then_accept_ge_next_self_offer"
    reservation_band_delta = 0.1
    mid_end = 0.8
    mid_floor = 0.7
    late_opponent_floor = 0.4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._safe_swap_info: dict[str, object] = {}

    def _make_model(self) -> YasasiV4OpponentUFunModel:
        return YasasiV4OpponentUFunModel()

    def _make_negotiation_model(self) -> PresenceHybridZeroSumIssueValueModel:
        return PresenceHybridZeroSumIssueValueModel(
            name=self.negotiation_model_name,
            prior="all_0_5",
            prior_weight=0.55,
            start=0.8,
            end=0.5,
        )

    def on_preferences_changed(self, changes):
        YASASHIAgent.on_preferences_changed(self, changes)
        self._virtual_ufun, self._safe_swap_info = (
            _build_reservation_band_active_swap_utility(
                self.ufun,
                self._outcomes,
                upper_delta=self.reservation_band_delta,
            )
        )
        self._rebuild_virtual_score_cache()
        self.private_info.setdefault("opponent_strategy_analysis", {})[
            "utility_swap"
        ] = self._safe_swap_info

    def _rebuild_virtual_score_cache(self) -> None:
        self._self_score_cache = {
            outcome: self._normalized_self_score(outcome) for outcome in self._outcomes
        }
        self._sorted_outcomes_by_self_score = sorted(
            ((score, outcome) for outcome, score in self._self_score_cache.items()),
            key=lambda item: (item[0], str(item[1])),
        )
        issue_count = len(self._outcomes[0]) if self._outcomes else 0
        self._proposal_value_counts = [defaultdict(int) for _ in range(issue_count)]

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_strategy_analysis"][
            "utility_swap"
        ] = getattr(self, "_safe_swap_info", {})

    def _select_yasashi_offer(self, state: SAOState) -> Outcome:
        progress = self._progress(state)
        if progress <= 0.05:
            return self._top_own_offer()
        if progress <= self.mid_end:
            return self._band_offer(
                state,
                start_t=0.05,
                end_t=self.mid_end,
                start=1.0,
                end=self.mid_floor,
            )
        return self._late_coverage_offer(state)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_score = self._normalized_self_score(offer)
        if self._progress(state) <= self.mid_end:
            if offered_score >= self.mid_floor:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        next_offer = self._select_yasashi_offer(state)
        next_score = self._normalized_self_score(next_offer)
        if offered_score >= next_score:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _late_coverage_offer(self, state: SAOState) -> Outcome:
        candidates = self._late_ranked_candidates()
        if not candidates:
            return self._top_own_offer()

        max_steps = self._configured_max_steps()
        start_step = max(0, int(self.mid_end * max_steps))
        endgame_steps = max(1, max_steps - start_step)
        step_index = max(0, int(state.step) - start_step)
        if len(candidates) <= 1 or endgame_steps <= 1:
            target_index = 0
        else:
            phase = max(0.0, min(1.0, step_index / max(1, endgame_steps - 1)))
            target_index = min(
                len(candidates) - 1,
                int(round(phase * (len(candidates) - 1))),
            )

        unused = [
            (index, outcome)
            for index, outcome in enumerate(candidates)
            if outcome not in self._proposal_steps
        ]
        if unused:
            return min(
                unused,
                key=lambda item: (abs(item[0] - target_index), item[0], str(item[1])),
            )[1]
        return candidates[target_index]

    def _late_ranked_candidates(self) -> list[Outcome]:
        reservation = self._reservation_utility()
        candidates = [
            outcome
            for outcome in self._outcomes
            if float(self.ufun(outcome)) >= reservation
            and float(self._negotiation_model.eval(outcome)) >= self.late_opponent_floor
        ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if float(self.ufun(outcome)) >= reservation
            ]
        return sorted(
            candidates,
            key=lambda outcome: (
                -self._normalized_self_score(outcome),
                -float(self._negotiation_model.eval(outcome)),
                str(outcome),
            ),
        )

    def _reservation_utility(self) -> float:
        if self.ufun is None:
            return 0.0
        return float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)


class MajiKayo(YASASHIAgent):
    """YASASHI V5 with all-0.5 inference and no utility swap."""

    model_name = "yasasi_v5_opponent_ufun__all_0_5_presence_hybrid__prior_0.75"
    negotiation_model_name = model_name
    selection_name = (
        "majikayo_real_utility_top_twice_then_descending_best_of_four_opponent_score"
    )
    acceptance_name = "accept_if_real_score_ge_next_self_offer"
    opponent_choice_start = 0.5
    top_offer_repeats = 2
    lookahead_count = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._majikayo_proposal_count = 0

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._majikayo_proposal_count = 0

    def propose(self, state: SAOState, dest: str | None = None):
        offer = super().propose(state, dest)
        if offer is not None:
            self._majikayo_proposal_count += 1
        return offer

    def _make_model(self) -> YasasiV5OpponentUFunModel:
        return YasasiV5OpponentUFunModel()

    def _make_negotiation_model(self) -> YasasiV5OpponentUFunModel:
        return YasasiV5OpponentUFunModel()

    def _select_yasashi_offer(self, state: SAOState) -> Outcome:
        if self._majikayo_proposal_count < self.top_offer_repeats:
            return self._top_own_offer()
        candidates = self._descending_real_utility_candidates()
        if not candidates:
            return self._reservation_random_offer()

        if self._progress(state) < self.opponent_choice_start:
            return candidates[0]

        shortlist = candidates[: self.lookahead_count]
        return max(
            shortlist,
            key=lambda outcome: (
                float(self._negotiation_model.eval(outcome)),
                self._normalized_self_score(outcome),
                str(outcome),
            ),
        )

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(offer, int(state.step), self._max_steps())
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        next_offer = self._select_yasashi_offer(state)
        if self._normalized_self_score(offer) >= self._normalized_self_score(next_offer):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _descending_real_utility_candidates(self) -> list[Outcome]:
        reservation = self._reservation_utility()
        limit = self._last_self_score_limit()
        candidates = [
            outcome
            for outcome in self._outcomes
            if outcome not in self._proposal_steps
            and float(self.ufun(outcome)) >= reservation
            and self._normalized_self_score(outcome) < limit
        ]
        return sorted(
            candidates,
            key=lambda outcome: (
                -self._normalized_self_score(outcome),
                -float(self.ufun(outcome)),
                str(outcome),
            ),
        )

    def _last_self_score_limit(self) -> float:
        if self._last_self_offer is None:
            return 1.0 + 1e-12
        return self._normalized_self_score(self._last_self_offer) - 1e-12

    def _reservation_random_offer(self) -> Outcome:
        reservation = self._reservation_utility()
        candidates = [
            outcome
            for outcome in self._outcomes
            if float(self.ufun(outcome)) >= reservation
            and self._normalized_self_score(outcome) < self._last_self_score_limit()
        ]
        if not candidates:
            candidates = [
                outcome
                for outcome in self._outcomes
                if float(self.ufun(outcome)) >= reservation
            ]
        if not candidates:
            return self._top_own_offer()
        return random.choice(sorted(candidates, key=str))

    def _reservation_utility(self) -> float:
        if self.ufun is None:
            return 0.0
        return float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)


class YouSuMiruAgent(YASASHIAgent):
    """YASASHI variant that changes late behavior based on opponent concession."""

    selection_name = "yousumiru_watch_concession_then_branch"
    acceptance_name = "accept_next_offer_or_0_7_when_no_concession_final_accept"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opponent_self_score_history: list[tuple[int, float]] = []

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._negotiation_model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(offer, int(state.step), self._max_steps())
        self._negotiation_model.update_from_opponent_offer(
            offer, int(state.step), self._max_steps()
        )
        self._opponent_offers.append(offer)
        self._opponent_self_score_history.append(
            (int(state.step), self._normalized_self_score(offer))
        )
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        if self._is_final_response(state):
            return ResponseType.ACCEPT_OFFER

        offered_score = self._normalized_self_score(offer)
        progress = self._progress(state)
        if 0.5 <= progress <= 0.8 and not self._opponent_looks_conceding():
            if offered_score >= 0.7:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        next_offer = self._select_yasashi_offer(state)
        next_score = self._normalized_self_score(next_offer)
        if offered_score >= next_score:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _select_yasashi_offer(self, state: SAOState) -> Outcome:
        progress = self._progress(state)
        if progress <= 0.5:
            return super()._select_yasashi_offer(state)
        if self._opponent_looks_conceding():
            return super()._select_yasashi_offer(state)
        if progress <= 0.8:
            return self._random_nonconcession_mid_offer()
        return self._super_concession_offer()

    def _opponent_looks_conceding(self) -> bool:
        history = self._opponent_self_score_history[-10:]
        if len(history) < 3:
            return False
        steps = [float(step) for step, _score in history]
        scores = [score for _step, score in history]
        improvement = scores[-1] - scores[0]
        positive_moves = sum(
            1 for previous, current in zip(scores, scores[1:], strict=False)
            if current > previous + 1e-6
        )
        positive_ratio = positive_moves / max(1, len(scores) - 1)
        correlation = self._score_time_correlation(steps, scores)
        self.private_info.setdefault("opponent_strategy_analysis", {})[
            "yousumiru_concession"
        ] = {
            "improvement": improvement,
            "positive_ratio": positive_ratio,
            "correlation": correlation,
        }
        return improvement >= 0.03 and correlation >= 0.2 or positive_ratio >= 0.6

    @staticmethod
    def _score_time_correlation(steps: list[float], scores: list[float]) -> float:
        if len(steps) != len(scores) or len(steps) < 2:
            return 0.0
        mean_step = sum(steps) / len(steps)
        mean_score = sum(scores) / len(scores)
        numerator = sum(
            (step - mean_step) * (score - mean_score)
            for step, score in zip(steps, scores, strict=True)
        )
        step_var = sum((step - mean_step) ** 2 for step in steps)
        score_var = sum((score - mean_score) ** 2 for score in scores)
        denom = (step_var * score_var) ** 0.5
        if denom <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, numerator / denom))

    def _random_nonconcession_mid_offer(self) -> Outcome:
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=1.0)
            if self._normalized_self_score(outcome) >= 0.7
        ]
        if not candidates:
            return self._random_own_score_offer(min_score=0.7, max_score=1.0)
        return random.choice(sorted(candidates, key=str))

    def _super_concession_offer(self) -> Outcome:
        candidates = [
            outcome
            for outcome in self._fresh_candidates(max_score=1.0)
            if self._normalized_self_score(outcome) >= 0.2
            and float(self._negotiation_model.eval(outcome)) >= 0.8
        ]
        if candidates:
            return max(
                candidates,
                key=lambda outcome: (
                    float(self._negotiation_model.eval(outcome)),
                    self._normalized_self_score(outcome),
                    str(outcome),
                ),
            )
        return self._random_own_score_offer(min_score=0.2, max_score=1.0)

    def _publish_private_info(self, pattern: str) -> None:
        super()._publish_private_info(pattern)
        self.private_info["opponent_strategy_analysis"][
            "selection"
        ] = self.selection_name
        self.private_info["opponent_strategy_analysis"][
            "acceptance"
        ] = self.acceptance_name


YouSiRu = YouSuMiruAgent


class UtilitySwapAgent(MywayAgent):
    """Myway agent that scores its own outcomes using a virtual rank swap.

    For every issue, the 2nd and 3rd ranked issue values are swapped before the
    agent evaluates, filters, and accepts offers. The top-ranked value remains
    unchanged, so normalized score 1.0 is preserved.
    """

    model_name = LinearTimeIssueValueModel.name
    selection_name = "select_threshold_0_8_to_0_3_virtual_rank_swap_no_repeat"
    acceptance_name = "accept_ge_virtual_rank_swap_0_8_to_0_4"
    rejection_penalties: tuple[float, float] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._virtual_ufun = None

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._virtual_ufun = _build_virtual_utility(self.ufun)

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        score_ufun = self._selection_ufun()
        if score_ufun is None or not self._outcomes:
            return None

        threshold = self._selection_threshold(state)
        candidates = self._candidates_above_threshold(threshold)
        if not candidates:
            offer = max(self._outcomes, key=lambda outcome: float(score_ufun(outcome)))
        else:
            unused = [
                outcome for outcome in candidates if outcome not in self._used_in_cycle
            ]
            if not unused:
                self._used_in_cycle = set()
                unused = list(candidates)
            if self.use_opponent_filter:
                favorable = [
                    outcome
                    for outcome in unused
                    if self._normalized_self_score(outcome)
                    > float(self._model.eval(outcome))
                ]
                if favorable:
                    unused = favorable
            offer = random.choice(unused)

        self._made_first_offer = True
        self._used_in_cycle.add(offer)
        self._last_self_offer = offer
        self._last_self_step = int(state.step)
        self._last_self_offer_pending = True
        return offer

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        score_ufun = self._selection_ufun()
        if offer is None or score_ufun is None:
            return ResponseType.REJECT_OFFER

        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, self._max_steps()
            )
            self._last_self_offer_pending = False
        self._model.update_from_opponent_offer(offer, int(state.step), self._max_steps())
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_utility = float(score_ufun(offer))
        if offered_utility >= self._acceptance_threshold(state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _selection_ufun(self):
        return self._virtual_ufun if self._virtual_ufun is not None else self.ufun

    def _normalized_self_score(self, outcome: Outcome) -> float:
        score_ufun = self._selection_ufun()
        if score_ufun is None:
            return 0.5
        mn, mx = score_ufun.minmax(above_reserve=False)
        denom = float(mx) - float(mn)
        if denom <= 1e-12:
            return 0.5
        return max(0.0, min(1.0, (float(score_ufun(outcome)) - float(mn)) / denom))

    def _candidates_above_threshold(self, threshold: float) -> list[Outcome]:
        score_ufun = self._selection_ufun()
        if score_ufun is None:
            return []
        return [
            outcome
            for outcome in self._outcomes
            if float(score_ufun(outcome)) >= threshold
        ]


class MywayOpponentFilterAgent(MywayAgent):
    """Myway with random choice filtered by self-score > opponent-model score."""

    selection_name = "select_threshold_0_8_to_0_3_random_no_repeat_self_gt_opp"
    use_opponent_filter = True


class MywayRejectLateAgent(MywayAgent):
    """Myway with rejected self offers penalized only in the second half."""

    rejection_penalties = (0.0, 0.5)


class MywayRejectBothAgent(MywayAgent):
    """Myway with rejected self offers penalized in both halves."""

    rejection_penalties = (0.2, 0.6)


class MywayOpponentFilterRejectLateAgent(MywayOpponentFilterAgent):
    """Opponent-filtered Myway with second-half rejection penalty."""

    rejection_penalties = (0.0, 0.5)


class MywayOpponentFilterRejectBothAgent(MywayOpponentFilterAgent):
    """Opponent-filtered Myway with rejection penalties in both halves."""

    rejection_penalties = (0.2, 0.6)


mywayAgent = MywayAgent
mywayV2 = MywayV2Agent
kingAgent = KingAgent
queenAgent = QueenAgent
kingAgentV2 = KingAgentV2
kingV3Agent = KingV3Agent
queenAgentV2 = QueenAgentV2
yasashiAgent = YASASHIAgent
yasasiAgent = YasasiAgent
yasasiV3Agent = YasasiV3Agent
yasasiV4Agent = YasasiV4Agent
yasaiV4Agent = YasasiV4Agent
YasasiV5Agent = MajiKayo
yasasiV5Agent = MajiKayo
yasaiV5Agent = MajiKayo
majiKayo = MajiKayo
yasasiSwapAgent = YasasiSwapAgent
yasashiSwapAgent = YasasiSwapAgent
yasasiSwap23Agent = YasasiSwap23Agent
yasasiSwap24Agent = YasasiSwap24Agent
yasasiSwap34Agent = YasasiSwap34Agent
yasasiiswapAgent = YasasiOutcomeSafeSwapAgent
yasasiOutcomeSafeSwapAgent = YasasiOutcomeSafeSwapAgent
youSuMiruAgent = YouSuMiruAgent
utilityswapAgent = UtilitySwapAgent
