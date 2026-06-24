from __future__ import annotations

import math
import os
import random
from collections import Counter
from typing import Any, Iterable

from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import ResponseType, SAOCallNegotiator, SAOResponse, SAOState

try:
    from examples.boa import BOANeg as _BaseNegotiator
except Exception:
    _BaseNegotiator = SAOCallNegotiator


class MiyaDreamBeliefSafeNegotiator(_BaseNegotiator):
    """ANL 2026 submission negotiator.

    The class exposes the SAOCallNegotiator API used by ANL:
    ``__call__(state) -> SAOResponse``.  Internally it also keeps explicit
    ``propose`` and ``respond`` methods so the accept/offer logic is easy to
    connect to other wrappers later.
    """

    _MAX_OUTCOMES = 50000
    _TOP_POOL = 256
    _WAVE_AMPLITUDE = 0.08
    _WAVE_DECAY_POWER = 1.0
    _WAVE_FREQUENCY = 6.0

    @staticmethod
    def _env_enabled(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._outcomes: list[Outcome] = []
        self._rational: list[Outcome] = []
        self._self_utility_cache: dict[tuple[Any, ...], float] = {}
        self._opponent_utility_cache: dict[tuple[Any, ...], float] = {}
        self._issue_values: list[list[Any]] = []
        self._opp_counts: list[Counter] = []
        self._self_counts: list[Counter] = []
        self._opp_weights: list[float] = []
        self._opp_values: list[dict[Any, float]] = []
        self._score_osi_weights_mean: list[float] | None = None
        self._score_osi_weights_std: list[float] | None = None
        self._score_osi_values_mean: list[list[float]] | None = None
        self._score_osi_values_std: list[list[float]] | None = None
        self._score_private_opp_utils: list[float] | None = None
        self._last_self_offer: Outcome | None = None
        self._best_opp_offer_utility: float | None = None
        self._wave_phase = random.uniform(0.0, 2.0 * math.pi)

    def on_preferences_changed(self, changes):
        self._initialize()

    def on_negotiation_start(self, state: SAOState) -> None:
        self._initialize()

    @property
    def opponent_ufun(self):
        return self.private_info.get("opponent_ufun", LambdaMultiFun(f=lambda x: 0.5))

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if not self._outcomes:
            self._initialize()

        offer = state.current_offer
        if offer is not None:
            self.update_opponent_model(state)
            if self.respond(state):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        counter = self.propose(state)
        if counter is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        self._last_self_offer = counter
        self._record_self_offer(counter)
        return SAOResponse(ResponseType.REJECT_OFFER, counter)

    def respond(self, state: SAOState) -> bool:
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return False
        reserved = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        util = self._self_utility(offer)
        if util < reserved:
            return False

        target = self._target_utility(state.relative_time)
        planned = self.propose(state)
        planned_util = self._self_utility(planned) if planned is not None else reserved
        return util >= min(target, planned_util)

    def propose(self, state: SAOState) -> Outcome | None:
        if self.ufun is None:
            return None
        candidates = self._rational or self._outcomes
        if not candidates:
            return None

        relative_time = min(max(float(state.relative_time), 0.0), 1.0)
        target = self._target_utility(relative_time)
        pool = self._near_target_pool(candidates, target, relative_time)
        best_score = -math.inf
        best: Outcome | None = None
        for outcome in pool:
            my_u = self._self_utility(outcome)
            if my_u + 1e-9 < float(getattr(self.ufun, "reserved_value", 0.0) or 0.0):
                continue
            opp_u = self._opponent_utility(outcome)
            closeness = -abs(my_u - target)
            target_weight = self._target_weight(state.relative_time)
            score = target_weight * closeness + (1.0 - target_weight) * opp_u
            if score > best_score:
                best_score = score
                best = outcome
        return best if best is not None else random.choice(candidates)

    def update_opponent_model(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None:
            return
        if self.ufun is not None:
            try:
                utility = self._self_utility(offer)
                if self._best_opp_offer_utility is None:
                    self._best_opp_offer_utility = utility
                else:
                    self._best_opp_offer_utility = max(
                        self._best_opp_offer_utility, utility)
            except Exception:
                pass
        values = self._as_tuple(offer)
        if len(values) != len(self._opp_counts):
            return
        for i, value in enumerate(values):
            self._opp_counts[i][value] += 1
        self._rebuild_opponent_model()

    def _record_self_offer(self, offer: Outcome | None) -> None:
        if offer is None:
            return
        values = self._as_tuple(offer)
        if len(values) != len(self._self_counts):
            return
        for i, value in enumerate(values):
            self._self_counts[i][value] += 1

    def _initialize(self) -> None:
        if self.ufun is None:
            return
        outcomes = list(self._enumerate_outcomes(self._MAX_OUTCOMES))
        if not outcomes:
            return
        self._outcomes = outcomes
        self._self_utility_cache = {}
        self._opponent_utility_cache = {}
        for outcome in outcomes:
            self._self_utility_cache[self._as_tuple(outcome)] = float(self.ufun(outcome))
        reserved = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        self._rational = sorted(
            [outcome for outcome in outcomes if self._self_utility(outcome) >= reserved],
            key=self._self_utility,
            reverse=True,
        )
        if not self._rational:
            self._rational = sorted(
                outcomes, key=self._self_utility, reverse=True
            )
        self._issue_values = self._infer_issue_values(outcomes)
        self._opp_counts = [Counter() for _ in self._issue_values]
        self._self_counts = [Counter() for _ in self._issue_values]
        self._score_osi_weights_mean = None
        self._score_osi_weights_std = None
        self._score_osi_values_mean = None
        self._score_osi_values_std = None
        self._score_private_opp_utils = None
        self._last_self_offer = None
        self._best_opp_offer_utility = None
        self._wave_phase = random.uniform(0.0, 2.0 * math.pi)
        self._rebuild_opponent_model()

    def _enumerate_outcomes(self, limit: int) -> Iterable[Outcome]:
        if self.nmi is None:
            return []
        source = self.nmi.outcome_space.enumerate_or_sample()
        outcomes = []
        for outcome in source:
            outcomes.append(outcome)
            if len(outcomes) >= limit:
                break
        return outcomes

    def _infer_issue_values(self, outcomes: list[Outcome]) -> list[list[Any]]:
        n_issues = len(self._as_tuple(outcomes[0]))
        values = [set() for _ in range(n_issues)]
        for outcome in outcomes:
            for i, value in enumerate(self._as_tuple(outcome)):
                values[i].add(value)
        return [list(vs) for vs in values]

    def _target_utility(self, relative_time: float) -> float:
        assert self.ufun is not None
        reserved = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        max_u = self._max_self_utility()
        t = min(max(float(relative_time), 0.0), 1.0)
        if self._last_self_offer is None:
            return min(max_u, max(reserved, 0.98))

        td_target = self._timedependence_target(t, reserved, max_u)
        if t >= 0.4:
            return td_target

        start = max(0.8, reserved, self._best_opp_offer_utility or reserved)
        start = min(max_u, max(reserved, start))
        ratio = t / 0.4
        target = (1.0 - ratio) * start + ratio * td_target
        return min(max_u, max(reserved, target))

    def _max_self_utility(self) -> float:
        assert self.ufun is not None
        try:
            return float(self.ufun.max())
        except Exception:
            return max(self._self_utility(outcome) for outcome in self._outcomes)

    def _timedependence_target(self, t: float, reserved: float, max_u: float) -> float:
        power = min(max(2.0 + 3.0 * (1.0 - reserved), 2.0), 5.0)
        concession = t**power
        base = reserved + (max_u - reserved) * (1.0 - concession)
        wave = (
            self._WAVE_AMPLITUDE *
            ((1.0 - t) ** self._WAVE_DECAY_POWER) *
            math.sin(2.0 * math.pi * self._WAVE_FREQUENCY * t + self._wave_phase)
        )
        return min(max_u, max(reserved, base + wave))

    def _target_weight(self, relative_time: float) -> float:
        t = min(max(float(relative_time), 0.0), 1.0)
        return 1.0 - 0.2 * (t**3)

    def _near_target_pool(
        self, outcomes: list[Outcome], target: float, relative_time: float
    ) -> list[Outcome]:
        lower = max(0.0, target - 0.1 * (relative_time**2))
        bounded = [
            outcome for outcome in outcomes
            if lower <= self._self_utility(outcome) <= 1.0
        ]
        if not bounded:
            bounded = outcomes
        ranked = sorted(
            bounded,
            key=lambda outcome: (
                abs(self._self_utility(outcome) - target),
                -self._opponent_utility(outcome),
            ),
        )
        return ranked[: min(len(ranked), self._TOP_POOL)]

    def _self_utility(self, outcome: Outcome | None) -> float:
        if outcome is None or self.ufun is None:
            return 0.0
        key = self._as_tuple(outcome)
        cached = self._self_utility_cache.get(key)
        if cached is not None:
            return cached
        value = float(self.ufun(outcome))
        self._self_utility_cache[key] = value
        return value

    def _rebuild_opponent_model(self) -> None:
        if not self._issue_values:
            self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda x: 0.5)
            return

        value_tables: list[dict[Any, float]] = []
        spreads = []
        for values, counts in zip(self._issue_values, self._opp_counts):
            if not values:
                value_tables.append({})
                spreads.append(0.0)
                continue
            smoothed = {value: float(counts[value]) + 1.0 for value in values}
            max_count = max(smoothed.values())
            table = {
                value: smoothed[value] / max(max_count, 1e-8)
                for value in values
            }
            value_tables.append(table)
            if table:
                table_values = list(table.values())
                mean = sum(table_values) / len(table_values)
                spreads.append(
                    sum((x - mean) ** 2 for x in table_values) / len(table_values)
                )
            else:
                spreads.append(0.0)

        if sum(spreads) <= 1e-9:
            weights = [1.0 / len(value_tables)] * len(value_tables)
        else:
            denom = sum(spreads)
            weights = [spread / denom for spread in spreads]

        self._opp_values = value_tables
        self._opp_weights = weights
        self._opponent_utility_cache = {}
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=lambda outcome: self._score_opponent_utility(outcome)
        )

    def _opponent_utility(self, outcome: Outcome | None) -> float:
        if outcome is None or not self._opp_values:
            return 0.5
        key = self._as_tuple(outcome)
        cached = self._opponent_utility_cache.get(key)
        if cached is not None:
            return cached
        values = self._as_tuple(outcome)
        if len(values) != len(self._opp_values):
            return 0.5
        utility = 0.0
        for weight, table, value in zip(self._opp_weights, self._opp_values, values):
            utility += weight * table.get(value, 0.5)
        utility = float(utility)
        self._opponent_utility_cache[key] = utility
        return utility

    def _score_opponent_utility(self, outcome: Outcome | None) -> float:
        if outcome is None or not self._issue_values:
            return 0.5
        values = self._as_tuple(outcome)
        if len(values) != len(self._issue_values):
            return 0.5
        weights = self._score_issue_weights()
        if not weights:
            return 0.5

        utility = 0.0
        for i, value in enumerate(values):
            value_utility = self._score_value_utility(i, value)
            utility += float(weights[i]) * value_utility
        return float(utility)

    def _private_rank_utility(self, outcome: Outcome) -> float | None:
        utilities = self._score_private_opp_utils
        if utilities is None:
            return None
        values = self._as_tuple(outcome)
        if len(values) != len(self._issue_values) or len(values) > 6:
            return None
        index = 0
        for i, value in enumerate(values):
            if len(self._issue_values[i]) > 5:
                return None
            try:
                value_index = self._issue_values[i].index(value)
            except ValueError:
                return None
            index += int(value_index) * (5 ** (5 - i))
        if 0 <= index < len(utilities):
            return float(min(max(utilities[index], 0.0), 1.0))
        return None

    def _score_value_utility(self, issue: int, value: Any) -> float:
        base = self._frequency_value_utility(issue, value)
        if self._env_enabled("MIYA_FORCE_FREQUENCY_SCORE"):
            return base
        means = self._score_osi_values_mean
        stds = self._score_osi_values_std
        if (
            means is None or stds is None or issue >= len(means) or
            issue >= len(stds) or issue >= len(self._issue_values)
        ):
            return base
        try:
            value_index = self._issue_values[issue].index(value)
        except ValueError:
            return base
        if value_index >= len(means[issue]) or value_index >= len(stds[issue]):
            return base
        return min(max(float(means[issue][value_index]), 0.0), 1.0)

    def _frequency_value_utility(self, issue: int, value: Any) -> float:
        counts = self._opp_counts[issue] if issue < len(self._opp_counts) else Counter()
        values = self._issue_values[issue] if issue < len(self._issue_values) else []
        if value not in values:
            return 0.5
        count = float(counts.get(value, 0.0)) + 1.0
        max_count = max([float(counts.get(v, 0.0)) + 1.0 for v in values] or [1.0])
        return float(count / max(max_count, 1e-8))

    def _score_issue_weights(self) -> list[float]:
        n_issues = len(self._issue_values)
        if n_issues <= 0:
            return []
        uniform = [1.0 / n_issues] * n_issues
        if len(self._opp_weights) == n_issues and sum(self._opp_weights) > 1e-8:
            base_weights = [max(0.0, float(x)) for x in self._opp_weights]
            base_total = sum(base_weights)
            base_weights = [x / base_total for x in base_weights]
        else:
            base_weights = uniform
        means = self._score_osi_weights_mean
        stds = self._score_osi_weights_std
        if self._env_enabled("MIYA_FORCE_FREQUENCY_SCORE"):
            return base_weights
        if means is None or stds is None:
            return base_weights

        means = [max(0.0, float(x)) for x in means[:n_issues]]
        if len(means) != n_issues:
            return base_weights
        total = sum(means)
        if total <= 1e-8:
            osi_weights = base_weights
        else:
            osi_weights = [x / total for x in means]

        mixed_total = sum(osi_weights)
        if mixed_total <= 1e-8:
            return uniform
        return [x / mixed_total for x in osi_weights]

    def _as_tuple(self, outcome: Outcome) -> tuple[Any, ...]:
        if isinstance(outcome, dict):
            return tuple(outcome.values())
        return tuple(outcome)
