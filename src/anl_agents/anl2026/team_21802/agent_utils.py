from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any, Iterable

from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.preferences.value_fun import TableFun
from negmas.sao import SAOCallNegotiator, SAOState


def normalize_mapping(mapping: dict[Any, float], invert: bool = False) -> dict[Any, float]:
    if not mapping:
        return {}
    items = list(mapping.items())
    values = [float(v) for _, v in items]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        base = {k: 0.5 for k, _ in items}
    else:
        base = {k: (float(v) - lo) / (hi - lo) for k, v in items}
    if invert:
        return {k: 1.0 - v for k, v in base.items()}
    return base


def normalize_weights(values: list[float]) -> list[float]:
    if not values:
        return []
    shifted = [max(1e-6, float(v)) for v in values]
    s = sum(shifted)
    return [v / s for v in shifted]


def build_additive_model(
    issues: tuple[Any, ...],
    value_tables: list[dict[Any, float]],
    weights: list[float],
    outcome_space: Any,
    reserved_value: float = 0.0,
) -> U:
    return U(
        values=tuple(TableFun(table) for table in value_tables),
        weights=tuple(normalize_weights(weights)),
        reserved_value=reserved_value,
        outcome_space=outcome_space,
    )


class ToolkitNegotiator(SAOCallNegotiator):
    def __init__(
        self,
        *args,
        sample_limit: int = 40_000,
        ranking_pool_size: int = 300,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_limit = sample_limit
        self.ranking_pool_size = ranking_pool_size
        self._rng = random.Random(seed)
        self._issues: tuple[Any, ...] = tuple()
        self.rational_outcomes: tuple[Outcome, ...] = tuple()
        self._rational_utils: tuple[float, ...] = tuple()
        self._utility_by_outcome: dict[Outcome, float] = {}
        self._normalized_rank: dict[Outcome, float] = {}
        self._negative_rational_utils: tuple[float, ...] = tuple()
        self._sent_offers: list[Outcome] = []
        self._received_offers: list[Outcome] = []
        self._last_offer_from_opponent: Outcome | None = None
        self._best_received_utility = float("-inf")
        self._self_value_means: list[dict[Any, float]] = []
        self._sent_value_counts: list[dict[Any, int]] = []

    def on_preferences_changed(self, changes):
        _ = changes
        if self.ufun is None or self.nmi is None or self.nmi.outcome_space is None:
            return

        self._issues = tuple(self.nmi.outcome_space.issues)
        outcomes = tuple(
            self.nmi.outcome_space.enumerate_or_sample(max_cardinality=self.sample_limit)
        )
        utility_of = self.ufun
        reserved_value = float(utility_of.reserved_value)
        self._utility_by_outcome = {o: float(utility_of(o)) for o in outcomes}
        rational_pairs = [
            (u, o) for o, u in self._utility_by_outcome.items() if u > reserved_value
        ]
        rational_pairs.sort(reverse=True)
        if not rational_pairs and outcomes:
            best = max(outcomes, key=self._utility_by_outcome.__getitem__)
            rational_pairs = [(self._utility_by_outcome[best], best)]
        self.rational_outcomes = tuple(o for _, o in rational_pairs)
        self._rational_utils = tuple(u for u, _ in rational_pairs)
        self._negative_rational_utils = tuple(-u for u in self._rational_utils)

        n = max(1, len(self.rational_outcomes) - 1)
        self._normalized_rank = {
            outcome: index / n for index, outcome in enumerate(self.rational_outcomes)
        }
        self._sent_offers = []
        self._received_offers = []
        self._last_offer_from_opponent = None
        self._best_received_utility = float("-inf")
        self._self_value_means = self._compute_self_value_means()
        self._sent_value_counts = [{value: 0 for value in issue.all} for issue in self._issues]
        self.initialize_strategy()

    def initialize_strategy(self) -> None:
        """Hook for subclasses."""

    def outcome_values(self, outcome: Outcome) -> tuple[Any, ...]:
        if isinstance(outcome, tuple):
            return outcome
        if isinstance(outcome, list):
            return tuple(outcome)
        if isinstance(outcome, dict):
            return tuple(outcome[issue.name] for issue in self._issues)
        if isinstance(outcome, Iterable):
            return tuple(outcome)
        return (outcome,)

    def reservation(self) -> float:
        assert self.ufun is not None
        return float(self.ufun.reserved_value)

    def best_utility(self) -> float:
        return self._rational_utils[0] if self._rational_utils else self.reservation()

    def utility_span(self) -> float:
        return max(1e-9, self.best_utility() - self.reservation())

    def normalized_self_utility(self, outcome: Outcome) -> float:
        return max(
            0.0,
            min(
                1.0,
                (self._utility_by_outcome[outcome] - self.reservation()) / self.utility_span(),
            ),
        )

    def select_candidates(
        self,
        target_utility: float,
        lower_slack: float,
        upper_slack: float,
    ) -> list[Outcome]:
        if not self.rational_outcomes:
            return []
        lo = max(self.reservation(), target_utility - self.utility_span() * lower_slack)
        hi = min(self.best_utility(), target_utility + self.utility_span() * upper_slack)
        left = bisect_left(self._negative_rational_utils, -hi)
        right = bisect_right(self._negative_rational_utils, -lo)
        pool = list(self.rational_outcomes[left:right])
        if len(pool) < 16:
            pool = list(self.rational_outcomes[:right])[: self.ranking_pool_size]
        if len(pool) > self.ranking_pool_size:
            stride = max(1, len(pool) // self.ranking_pool_size)
            pool = pool[::stride][: self.ranking_pool_size]
        return pool

    def target_utility(
        self,
        relative_time: float,
        early: float = 0.985,
        middle: float = 0.88,
        late: float = 0.70,
        floor: float = 0.12,
    ) -> float:
        t = relative_time
        if t < 0.15:
            normalized = early
        elif t < 0.50:
            x = (t - 0.15) / 0.35
            normalized = early - (early - middle) * (x**1.5)
        elif t < 0.85:
            x = (t - 0.50) / 0.35
            normalized = middle - (middle - late) * (x**1.15)
        else:
            x = min(1.0, (t - 0.85) / 0.15)
            normalized = late - (late - floor) * (x**0.85)
        normalized = min(0.995, max(floor, normalized))
        return self.reservation() + self.utility_span() * normalized

    def diversity_score(self, outcome: Outcome) -> float:
        if not self._sent_offers:
            return 1.0
        values = self.outcome_values(outcome)
        recent = self._sent_offers[-4:]
        mismatch = []
        for prev in recent:
            pv = self.outcome_values(prev)
            mismatch.append(
                sum(1 for a, b in zip(values, pv) if a != b) / max(1, len(values))
            )
        novelty = [
            1.0 - self._sent_value_counts[i].get(value, 0) / max(1, len(self._sent_offers))
            for i, value in enumerate(values)
        ]
        return 0.55 * (sum(novelty) / len(novelty)) + 0.45 * (
            sum(mismatch) / max(1, len(mismatch))
        )

    def reveal_penalty(self, outcome: Outcome, relative_time: float) -> float:
        if not self._sent_offers:
            repetition = 0.0
        else:
            repetition = sum(
                self._sent_value_counts[i].get(value, 0) / max(1, len(self._sent_offers))
                for i, value in enumerate(self.outcome_values(outcome))
            ) / max(1, len(self._issues))
        rank_pressure = 1.0 - self._normalized_rank.get(outcome, 1.0)
        early_weight = max(0.0, 1.0 - 1.2 * relative_time)
        return 0.5 * repetition + 0.5 * rank_pressure * early_weight

    def reciprocity_score(self, outcome: Outcome) -> float:
        if self._last_offer_from_opponent is None:
            return 0.5
        ov = self.outcome_values(outcome)
        pv = self.outcome_values(self._last_offer_from_opponent)
        same = sum(1 for a, b in zip(ov, pv) if a == b)
        return same / max(1, len(ov))

    def should_accept(self, state: SAOState, next_offer_utility: float, slack: float) -> bool:
        assert self.ufun is not None
        offer = state.current_offer
        if offer is None:
            return False
        utility = float(self.ufun(offer))
        threshold = max(
            self.reservation(),
            min(self.target_utility(state.relative_time), next_offer_utility) - slack,
        )
        if utility >= threshold:
            return True
        if state.relative_time >= 0.97 and utility >= self.reservation() + 0.01 * self.utility_span():
            return True
        return False

    def register_received_offer(self, offer: Outcome) -> None:
        self._received_offers.append(offer)
        if self.ufun is not None:
            self._best_received_utility = max(self._best_received_utility, float(self.ufun(offer)))
        self._last_offer_from_opponent = offer

    def register_sent_offer(self, offer: Outcome) -> None:
        self._sent_offers.append(offer)
        values = self.outcome_values(offer)
        for i, value in enumerate(values):
            self._sent_value_counts[i][value] = self._sent_value_counts[i].get(value, 0) + 1

    def _compute_self_value_means(self) -> list[dict[Any, float]]:
        if not self.rational_outcomes:
            return []
        per_issue: list[dict[Any, list[float]]] = []
        for issue in self._issues:
            per_issue.append({value: [] for value in issue.all})
        cap = min(len(self.rational_outcomes), 3000)
        for outcome in self.rational_outcomes[:cap]:
            utility = self._utility_by_outcome[outcome]
            for i, value in enumerate(self.outcome_values(outcome)):
                per_issue[i][value].append(utility)
        means: list[dict[Any, float]] = []
        for table in per_issue:
            means.append(
                {
                    value: (sum(vals) / len(vals) if vals else self.reservation())
                    for value, vals in table.items()
                }
            )
        return means
