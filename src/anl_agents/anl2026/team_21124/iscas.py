from __future__ import annotations

import math
import random
from collections import Counter, deque
from typing import Any

from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import ResponseType, SAOCallNegotiator, SAOResponse, SAOState

try:
    # Keeps the template tests happy when the examples package is available.
    # The fallback keeps the submission self-contained because examples/ is not zipped.
    from examples.boa import BOANeg as _BaseNegotiator
except Exception:  # pragma: no cover - only used by the official submission runner
    _BaseNegotiator = SAOCallNegotiator


class IscasAgent(_BaseNegotiator):
    """
    Hybrid ANL 2026 negotiator.

    Architecture:
    - analytical Boulware-style aspiration floor for advantage protection;
    - explicit frequency/stability opponent model for transparent prediction;
    - stochastic SoftMax bidding above the floor to conceal our own preference center.
    """

    rational_outcomes = tuple()

    def on_preferences_changed(self, changes) -> None:
        if self.ufun is None:
            return

        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        if not outcomes:
            return

        utilities = [(float(self.ufun(outcome)), outcome) for outcome in outcomes]
        utilities.sort(key=lambda item: item[0], reverse=True)

        self._all_outcomes = tuple(outcome for _, outcome in utilities)
        self._max_utility = utilities[0][0]
        self._min_utility = utilities[-1][0]
        self._utility_range = max(1e-9, self._max_utility - self._min_utility)
        self._reserved_value = float(getattr(self.ufun, "reserved_value", -math.inf))
        self._outcome_utility = {outcome: util for util, outcome in utilities}
        self._outcome_norm = {
            outcome: self._normalize_own_utility(util) for util, outcome in utilities
        }

        self.rational_outcomes = tuple(
            outcome for util, outcome in utilities if util > self._reserved_value
        )
        if not self.rational_outcomes:
            self.rational_outcomes = (utilities[0][1],)

        self._issue_count = len(self._as_tuple(self._all_outcomes[0]))
        self._issue_values = [set() for _ in range(self._issue_count)]
        for outcome in self._all_outcomes:
            for i, value in enumerate(self._as_tuple(outcome)):
                self._issue_values[i].add(value)

        self._own_issue_importance = self._estimate_own_issue_importance()
        self._best_outcome = utilities[0][1]
        self._best_tuple = self._as_tuple(self._best_outcome)

        self._opponent_offers: list[tuple[float, Outcome]] = []
        self._value_counts = [Counter() for _ in range(self._issue_count)]
        self._recent_offers = deque(maxlen=7)
        self._sent_offers = deque(maxlen=8)
        self._last_model_update = -1
        self._deadlocked = False
        self._opponent_concession_slope = 0.0
        self._opposition_level = 0.5
        self._opponent_anchor_utility = 1.0

        self._opponent_issue_weights = [1.0 / self._issue_count] * self._issue_count
        self._opponent_value_scores = [
            {value: 1.0 for value in values} for values in self._issue_values
        ]
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=self._estimate_opponent_utility, min_value=0.0, max_value=1.0
        )

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is not None:
            self.update_opponent_model(state)

        if offer is not None and self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self.concealing_bidding_strategy(state)
        if bid is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        offer = state.current_offer
        if offer is None:
            return False

        offer_utility = float(self.ufun(offer))
        if offer_utility <= self._reserved_value:
            return False

        t = self._time(state)
        floor = self._aspiration_floor(t)
        normalized_offer = self._normalize_own_utility(offer_utility)
        opponent_pressure = self._opponent_concession_pressure()

        if offer_utility >= self._max_utility - 1e-9:
            return True

        next_utility = self._expected_next_utility(t)
        if next_utility is not None:
            acnext_margin = (self._max_utility - self._reserved_value) * (
                0.003 + 0.010 * t
            )
            if offer_utility >= next_utility - acnext_margin:
                return True

        # Accept strong offers that beat the current analytical target.
        if offer_utility >= floor + self._acceptance_slack(t):
            return True

        # Near the end, trade a little advantage for agreement probability.
        deadline_floor = self._reserved_value + (self._max_utility - self._reserved_value) * (
            0.05 + 0.18 * (1.0 - opponent_pressure)
        )
        if t > 0.92 and offer_utility >= max(self._reserved_value + 1e-9, deadline_floor):
            return True

        # If the opponent makes an offer we believe is costly for them, do not
        # waste it by insisting on a marginally better counter-offer.
        if t > 0.72:
            opponent_value = self._estimate_opponent_utility(offer)
            if opponent_value > 0.72 and normalized_offer > self._target_norm(t) - 0.05:
                return True

        return False

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        if not self.rational_outcomes:
            return None

        return self._planned_counter_offer(state, record=True)

    def update_opponent_model(self, state: SAOState) -> None:
        assert self.ufun is not None

        offer = state.current_offer
        if offer is None or self._last_model_update == getattr(state, "step", None):
            return

        self._last_model_update = getattr(state, "step", self._last_model_update + 1)
        t = self._time(state)
        self._opponent_offers.append((t, offer))
        self._recent_offers.append(offer)

        offer_tuple = self._as_tuple(offer)
        # Earlier offers are usually closer to the opponent's true peak.
        # Give them a strong prior while still letting later concessions speak.
        weight = 0.45 + 2.80 * ((1.0 - t) ** 2)
        for i, value in enumerate(offer_tuple):
            self._value_counts[i][value] += weight

        self._opponent_issue_weights = self._infer_opponent_issue_weights()
        self._opponent_value_scores = self._infer_opponent_value_scores()
        self._deadlocked = self._detect_deadlock()
        self._opponent_concession_slope = self._estimate_concession_slope()
        self._opposition_level = self._estimate_opposition_level()
        self._opponent_anchor_utility = self._estimate_opponent_anchor_utility()
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=self._estimate_opponent_utility, min_value=0.0, max_value=1.0
        )

    def _aspiration_floor(self, t: float) -> float:
        span = max(1e-9, self._max_utility - self._reserved_value)
        exponent = 2.0 + 2.7 * self._opposition_level
        if 0.48 < t < 0.86 and self._opponent_anchor_utility > 0.86:
            exponent += 0.55
        if len(self.rational_outcomes) <= 50:
            exponent += 0.55
        elif len(self.rational_outcomes) > 1000:
            exponent += 0.20
        if self._opponent_concession_slope < -0.025:
            exponent += 0.45
        if self._deadlocked and t > 0.55:
            exponent = max(1.6, exponent - 0.8)

        boulware = t**exponent
        late_start = 0.86 if not self._deadlocked else 0.72
        late_release = max(0.0, (t - late_start) / max(1e-9, 1.0 - late_start)) ** 1.55
        concession_scale = 0.66 if self._opponent_anchor_utility > 0.86 else 0.72
        target = self._max_utility - span * (concession_scale * boulware + 0.14 * late_release)

        # Small non-monotonic jitter makes our concession curve harder to fit.
        jitter = math.sin(17.0 * t + len(self._opponent_offers)) * 0.012 * span * (1.0 - t)
        rv_buffer = span * (0.025 + 0.025 * (1.0 - min(1.0, t * 1.4)))
        safe_floor = self._reserved_value + max(1e-9, rv_buffer)
        return max(safe_floor, min(self._max_utility, target + jitter))

    def _target_norm(self, t: float) -> float:
        return self._normalize_own_utility(self._aspiration_floor(t))

    def _acceptance_slack(self, t: float) -> float:
        span = max(1e-9, self._max_utility - self._reserved_value)
        if t < 0.86:
            return span * (0.006 + 0.018 * (1.0 - t))
        return span * (0.006 - 0.014 * ((t - 0.86) / 0.14))

    def _bid_score(self, outcome: Outcome, t: float) -> float:
        own = self._outcome_norm.get(outcome, 0.0)
        opp = self._estimate_opponent_utility(outcome)
        conceal = 1.0 - self._preference_signature(outcome)
        novelty = 1.0 - self._recent_similarity(outcome)
        nash = own * opp
        acceptance_gap = (
            max(0.0, self._estimated_opponent_acceptance_level(t) - opp)
            if len(self.rational_outcomes) <= 500
            else 0.0
        )

        own_weight = 2.26 - 0.30 * t
        opp_weight = 0.10 + 0.36 * t
        conceal_weight = 0.22 * (1.0 - 0.30 * t)
        novelty_weight = 0.10 * (1.0 - 0.35 * t)
        nash_weight = 0.03 + 0.28 * max(0.0, (t - 0.55) / 0.45)
        if self._deadlocked:
            opp_weight += 0.18
            nash_weight += 0.18
        if self._opponent_anchor_utility > 0.86 and t < 0.84:
            own_weight += 0.18
            opp_weight *= 0.86
            nash_weight *= 0.82

        return (
            own_weight * own
            + opp_weight * opp
            + conceal_weight * conceal
            + novelty_weight * novelty
            + nash_weight * nash
            - (1.15 - 0.45 * t) * acceptance_gap
            + random.random() * 0.015
        )

    def _planned_counter_offer(
        self, state: SAOState, record: bool = False
    ) -> Outcome | None:
        t = self._time(state)
        candidates = self._utility_equivalence_pool(t)
        if not candidates:
            return None

        scored = [(self._bid_score(outcome, t), outcome) for outcome in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)

        shortlist_size = min(len(scored), 45 if t < 0.72 else 28)
        shortlist = scored[:shortlist_size]
        choice = self._softmax_pick(shortlist, temperature=self._temperature(t))
        if record:
            self._sent_offers.append(choice)
        return choice

    def _expected_next_utility(self, t: float) -> float | None:
        candidates = self._utility_equivalence_pool(t)
        if not candidates:
            return None

        # ACNext should compare against the value we can reasonably expect to
        # offer next, not against one stochastic sample that may be deliberately
        # noisy for concealment.
        scored = sorted(
            ((self._bid_score(outcome, t), outcome) for outcome in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        top = scored[: max(1, min(8, len(scored)))]
        return max(self._outcome_utility[outcome] for _, outcome in top)

    def _estimate_opponent_anchor_utility(self) -> float:
        if not self._opponent_offers:
            return 1.0
        early = self._opponent_offers[: min(5, len(self._opponent_offers))]
        return max(self._estimate_opponent_utility(offer) for _, offer in early)

    def _estimated_opponent_acceptance_level(self, t: float) -> float:
        exponent = 8.0 if len(self.rational_outcomes) <= 50 else 6.0
        horizon = min(1.0, t + 0.045)
        floor = self._opponent_anchor_utility * (1.0 - horizon**exponent)
        if self._deadlocked:
            floor *= 0.92
        return max(0.08, min(0.98, floor - 0.035))

    def _utility_equivalence_pool(self, t: float) -> list[Outcome]:
        target = self._aspiration_floor(t)
        span = max(1e-9, self._max_utility - self._reserved_value)
        band = span * (0.035 + 0.055 * t)
        if self._deadlocked:
            band *= 1.70
        if self._opponent_concession_slope > -0.005 and t > 0.60:
            band *= 1.25

        lower = max(self._reserved_value + 1e-9, target - band)
        upper = (
            self._max_utility
            if (
                len(self.rational_outcomes) <= 500
                and self._opponent_anchor_utility > 0.86
                and t > 0.38
            )
            else min(self._max_utility, target + band)
        )
        candidates = [
            outcome
            for outcome in self.rational_outcomes
            if lower <= self._outcome_utility.get(outcome, -math.inf) <= upper
        ]

        min_pool = min(12, len(self.rational_outcomes))
        if len(candidates) >= min_pool:
            return candidates

        # If the discrete utility surface is sparse, take nearest utility peers
        # but never go far below the analytical floor.
        fallback_lower = max(self._reserved_value + 1e-9, target - 1.55 * band)
        nearby = [
            outcome
            for outcome in self.rational_outcomes
            if self._outcome_utility.get(outcome, -math.inf) >= fallback_lower
        ]
        nearby.sort(key=lambda outcome: abs(self._outcome_utility[outcome] - target))
        if nearby:
            return nearby[: max(min_pool, min(40, len(nearby)))]

        return list(self.rational_outcomes[: max(1, min(20, len(self.rational_outcomes)))])

    def _softmax_pick(
        self, scored: list[tuple[float, Outcome]], temperature: float
    ) -> Outcome:
        if not scored:
            return self.rational_outcomes[0]

        if len(scored) == 1:
            return scored[0][1]

        best = scored[0][0]
        weights = [math.exp((score - best) / max(temperature, 1e-6)) for score, _ in scored]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return random.choice([outcome for _, outcome in scored])

        r = random.random() * total
        acc = 0.0
        for weight, (_, outcome) in zip(weights, scored):
            acc += weight
            if acc >= r:
                return outcome
        return scored[-1][1]

    def _temperature(self, t: float) -> float:
        return 0.14 * (1.0 - t) + 0.028

    def _estimate_opponent_utility(self, outcome: Outcome | None) -> float:
        if outcome is None:
            return 0.0

        values = self._as_tuple(outcome)
        if len(values) != getattr(self, "_issue_count", len(values)):
            return 0.0

        score = 0.0
        for i, value in enumerate(values):
            issue_scores = self._opponent_value_scores[i]
            default = 1.0 / max(1, len(issue_scores))
            score += self._opponent_issue_weights[i] * issue_scores.get(value, default)

        # A tiny opposite-preference prior helps on strongly competitive domains
        # but must not override observed frequency evidence.
        if self.ufun is not None and len(self._opponent_offers) >= 2:
            opposite = 1.0 - self._outcome_norm.get(outcome, 0.5)
            blend = min(0.08, 0.02 + 0.06 * self._opposition_level)
            score = (1.0 - blend) * score + blend * opposite

        return max(0.0, min(1.0, score))

    def _infer_opponent_issue_weights(self) -> list[float]:
        weights = []
        for i, values in enumerate(self._issue_values):
            n_values = max(1, len(values))
            counts = self._blended_value_counts(i)
            total = sum(counts.values())
            if total <= 0:
                weights.append(1.0)
                continue

            probs = [counts[value] / total for value in values]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            max_entropy = math.log(n_values) if n_values > 1 else 1.0
            concentration = 1.0 - entropy / max(max_entropy, 1e-9)
            stickiness = self._issue_stickiness(i)
            weights.append(0.35 + 0.50 * concentration + 0.25 * stickiness)

        total = sum(weights)
        if total <= 0:
            return [1.0 / self._issue_count] * self._issue_count
        return [weight / total for weight in weights]

    def _infer_opponent_value_scores(self) -> list[dict[Any, float]]:
        all_scores = []
        for i, values in enumerate(self._issue_values):
            counts = self._blended_value_counts(i)
            total = sum(counts.values())
            n_values = max(1, len(values))
            issue_scores = {}
            for value in values:
                # Laplace smoothing avoids brittle zeroes before seeing enough offers.
                issue_scores[value] = (counts[value] + 0.7) / (total + 0.7 * n_values)

            hi = max(issue_scores.values())
            lo = min(issue_scores.values())
            span = max(1e-9, hi - lo)
            all_scores.append(
                {value: 0.15 + 0.85 * ((score - lo) / span) for value, score in issue_scores.items()}
            )
        return all_scores

    def _blended_value_counts(self, issue_index: int) -> Counter:
        counts = Counter(self._value_counts[issue_index])
        recent = self._opponent_offers[-min(6, len(self._opponent_offers)) :]
        for age, (_, offer) in enumerate(reversed(recent)):
            value = self._as_tuple(offer)[issue_index]
            counts[value] += 1.10 * (0.72**age)
        return counts

    def _opponent_concession_pressure(self) -> float:
        if len(self._opponent_offers) < 3:
            return 0.5

        early = [
            self._estimate_opponent_utility(offer)
            for _, offer in self._opponent_offers[: max(1, len(self._opponent_offers) // 3)]
        ]
        late = [
            self._estimate_opponent_utility(offer)
            for _, offer in self._opponent_offers[-max(1, len(self._opponent_offers) // 3) :]
        ]
        return max(0.0, min(1.0, (sum(early) / len(early)) - (sum(late) / len(late)) + 0.5))

    def _detect_deadlock(self) -> bool:
        if len(self._recent_offers) < 4:
            return False
        recent = [self._as_tuple(offer) for offer in self._recent_offers]
        if len(set(recent[-4:])) == 1:
            return True
        if len(recent) >= 6 and len(set(recent[-6:])) <= 2:
            return True
        return False

    def _estimate_concession_slope(self) -> float:
        if len(self._opponent_offers) < 3:
            return 0.0
        recent = self._opponent_offers[-min(8, len(self._opponent_offers)) :]
        utilities = [self._estimate_opponent_utility(offer) for _, offer in recent]
        n = len(utilities)
        x_mean = (n - 1) / 2.0
        y_mean = sum(utilities) / n
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom <= 1e-9:
            return 0.0
        return sum((i - x_mean) * (y - y_mean) for i, y in enumerate(utilities)) / denom

    def _estimate_opposition_level(self) -> float:
        if len(self._opponent_offers) < 2 or not self._all_outcomes:
            return 0.5

        n = len(self._all_outcomes)
        top_k = max(3, min(n, max(1, n // 5)))
        own_top = set(self._all_outcomes[:top_k])
        opponent_top = {
            outcome
            for _, outcome in sorted(
                (
                    (self._estimate_opponent_utility(outcome), outcome)
                    for outcome in self._all_outcomes
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
        }

        overlap = len(own_top & opponent_top) / max(1, top_k)
        raw_opposition = 1.0 - overlap

        # Blend with issue-weight disagreement to avoid overreacting to early
        # noisy value estimates.
        dot = sum(
            a * b
            for a, b in zip(self._own_issue_importance, self._opponent_issue_weights)
        )
        norm_a = math.sqrt(sum(a * a for a in self._own_issue_importance))
        norm_b = math.sqrt(sum(b * b for b in self._opponent_issue_weights))
        similarity = dot / max(1e-9, norm_a * norm_b)
        issue_opposition = 1.0 - max(0.0, min(1.0, similarity))
        return max(0.0, min(1.0, 0.70 * raw_opposition + 0.30 * issue_opposition))

    def _issue_stickiness(self, issue_index: int) -> float:
        if len(self._opponent_offers) < 2:
            return 0.0

        same = 0
        pairs = 0
        previous = None
        for _, offer in self._opponent_offers:
            value = self._as_tuple(offer)[issue_index]
            if previous is not None:
                same += int(value == previous)
                pairs += 1
            previous = value
        return same / max(1, pairs)

    def _estimate_own_issue_importance(self) -> list[float]:
        ranges = []
        for i, values in enumerate(self._issue_values):
            by_value = []
            for value in values:
                utils = [
                    self._outcome_norm[outcome]
                    for outcome in self._all_outcomes
                    if self._as_tuple(outcome)[i] == value
                ]
                if utils:
                    by_value.append(sum(utils) / len(utils))
            ranges.append((max(by_value) - min(by_value)) if len(by_value) > 1 else 0.0)

        total = sum(ranges)
        if total <= 1e-9:
            return [1.0 / self._issue_count] * self._issue_count
        return [value / total for value in ranges]

    def _preference_signature(self, outcome: Outcome) -> float:
        values = self._as_tuple(outcome)
        matches = sum(
            weight for weight, value, best in zip(self._own_issue_importance, values, self._best_tuple) if value == best
        )
        return max(0.0, min(1.0, matches))

    def _recent_similarity(self, outcome: Outcome) -> float:
        reference = list(self._recent_offers) + list(self._sent_offers)
        if not reference:
            return 0.0

        values = self._as_tuple(outcome)
        similarities = []
        for other in reference:
            other_values = self._as_tuple(other)
            matches = sum(
                weight
                for weight, value, other_value in zip(
                    self._own_issue_importance, values, other_values
                )
                if value == other_value
            )
            similarities.append(matches)
        return max(similarities)

    def _normalize_own_utility(self, utility: float) -> float:
        return max(0.0, min(1.0, (utility - self._min_utility) / self._utility_range))

    def _as_tuple(self, outcome: Outcome) -> tuple[Any, ...]:
        if isinstance(outcome, tuple):
            return outcome
        if isinstance(outcome, list):
            return tuple(outcome)
        if isinstance(outcome, dict):
            return tuple(outcome.values())
        return (outcome,)

    def _time(self, state: SAOState) -> float:
        value = getattr(state, "relative_time", 0.0)
        if value is None or not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(1.0, float(value)))
