from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import SAOCallNegotiator, ResponseType, SAOResponse, SAOState


class PerikosV3(SAOCallNegotiator):
    """
    PeriKos ANL 2026 negotiator.

    The agent keeps a high utility target early, learns a lightweight additive
    model from opponent offers, and gradually shifts from value protection to
    agreement seeking as the deadline approaches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False
        self.rational_outcomes = tuple()
        self.utility_by_outcome = {}
        self.own_reserved_value = 0.0
        self.own_max_utility = 1.0
        self.own_utility_range = 1.0
        self.opponent_value_counts = tuple()
        self.recent_opponent_value_counts = tuple()
        self.own_value_counts = tuple()
        self.own_value_signal = tuple()
        self.issue_salience = tuple()
        self.counter_pressure = tuple()
        self.opponent_offer_count = 0
        self.own_offer_count = 0
        self.best_received_offer = None
        self.best_received_utility = float("-inf")
        self.last_planned_offer = None
        self.compact_conflict_domain = False

    def on_preferences_changed(self, changes):
        self.initialize_strategy()

    def initialize_strategy(self):
        if self.ufun is None:
            return

        self.own_reserved_value = float(self.ufun.reserved_value)
        self.own_max_utility = float(self.ufun.max())
        self.own_utility_range = max(
            self.own_max_utility - self.own_reserved_value, 1e-9
        )

        ranked = [
            (float(self.ufun(outcome)), outcome)
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if float(self.ufun(outcome)) > self.own_reserved_value
        ]
        ranked.sort(reverse=True, key=lambda item: item[0])
        self.rational_outcomes = tuple(outcome for _, outcome in ranked)
        self.utility_by_outcome = {outcome: utility for utility, outcome in ranked}
        self.compact_conflict_domain = (
            len(self.rational_outcomes) <= 3
            and len(self.nmi.outcome_space.issues) <= 1
        )

        self.opponent_value_counts = tuple(
            {value: 1.0 for value in issue.values}
            for issue in self.nmi.outcome_space.issues
        )
        self.recent_opponent_value_counts = tuple(
            {value: 1.0 for value in issue.values}
            for issue in self.nmi.outcome_space.issues
        )
        self.own_value_counts = tuple(
            {value: 0.0 for value in issue.values}
            for issue in self.nmi.outcome_space.issues
        )
        self.own_value_signal = self.build_own_value_signal()
        self.issue_salience = tuple(1.0 for _ in self.nmi.outcome_space.issues)
        self.counter_pressure = tuple(1.0 for _ in self.nmi.outcome_space.issues)
        self.opponent_offer_count = 0
        self.own_offer_count = 0
        self.best_received_offer = None
        self.best_received_utility = float("-inf")
        self.last_planned_offer = None
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=self.estimate_reported_opponent_utility,
            min_value=0.0,
            max_value=1.0,
        )
        self.initialized = True

    def build_own_value_signal(self) -> tuple[dict, ...]:
        if self.ufun is None:
            return tuple()

        totals = tuple(
            {value: 0.0 for value in issue.values}
            for issue in self.nmi.outcome_space.issues
        )
        counts = tuple(
            {value: 0 for value in issue.values}
            for issue in self.nmi.outcome_space.issues
        )
        for outcome in self.rational_outcomes:
            utility = self.normalized_own_utility(outcome)
            for issue_index, value in enumerate(outcome):
                totals[issue_index][value] += utility
                counts[issue_index][value] += 1

        signal = []
        for issue_index, values in enumerate(totals):
            averages = {
                value: totals[issue_index][value] / max(counts[issue_index][value], 1)
                for value in values
            }
            low, high = min(averages.values()), max(averages.values())
            span = max(high - low, 1e-9)
            signal.append(
                {value: (score - low) / span for value, score in averages.items()}
            )
        return tuple(signal)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if not self.initialized:
            self.initialize_strategy()

        offer = state.current_offer
        if offer is not None:
            self.learn_counter_pressure(offer, state.relative_time)
            self.update_opponent_model(state)
            if self.acceptance_strategy(state):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        counter_offer = self.concealing_bidding_strategy(state)
        if counter_offer is None:
            if offer is not None and float(self.ufun(offer)) > self.own_reserved_value:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self.last_planned_offer = counter_offer
        self.remember_own_offer(counter_offer)
        return SAOResponse(ResponseType.REJECT_OFFER, counter_offer)

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        offer = state.current_offer
        if offer is None:
            return False

        offer_utility = float(self.ufun(offer))
        if offer_utility <= self.own_reserved_value:
            return False

        target = self.acceptance_level(state.relative_time)
        planned_offer = self.concealing_bidding_strategy(state)
        planned_utility = (
            float(self.ufun(planned_offer)) if planned_offer is not None else target
        )
        opponent_fit = self.estimate_opponent_utility(offer)

        if offer_utility >= max(target, planned_utility - 0.015 * self.own_utility_range):
            return True
        if state.relative_time > 0.62 and offer_utility >= planned_utility:
            return True
        if state.relative_time > 0.90 and offer_utility >= planned_utility:
            return True
        if state.relative_time > 0.985 and offer_utility >= self.own_reserved_value + self.late_floor(state.relative_time) * self.own_utility_range:
            return True
        return False

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        assert self.ufun

        if not self.rational_outcomes:
            return None

        time = state.relative_time
        aspiration = self.aspiration_level(time)
        if self.needs_agreement_push(time):
            aspiration = min(
                aspiration,
                self.own_reserved_value + self.late_floor(time) * self.own_utility_range,
            )
        candidates = [
            outcome
            for outcome in self.rational_outcomes
            if self.utility_by_outcome[outcome] >= aspiration
        ]

        if not candidates:
            fallback_size = min(len(self.rational_outcomes), 120)
            candidates = list(self.rational_outcomes[:fallback_size])
        elif len(candidates) > 2500:
            candidates = candidates[:2500]

        if time < 0.18:
            candidates = candidates[: min(len(candidates), 35)]
        elif time < 0.50:
            candidates = candidates[: min(len(candidates), max(30, len(candidates) // 2))]

        scored = [(self.offer_score(outcome, time), outcome) for outcome in candidates]
        scored.sort(reverse=True, key=lambda item: item[0])

        return scored[0][1]

    def learn_counter_pressure(self, offer: Outcome, relative_time: float) -> None:
        if self.last_planned_offer is None:
            return
        pressure = list(self.counter_pressure)
        if len(pressure) < len(offer):
            pressure.extend([1.0] * (len(offer) - len(pressure)))
        for issue_index, (planned_value, offered_value) in enumerate(
            zip(self.last_planned_offer, offer)
        ):
            if issue_index >= len(pressure):
                continue
            if planned_value == offered_value:
                pressure[issue_index] += 0.008
            else:
                pressure[issue_index] += 0.030 + 0.020 * (1.0 - relative_time)
        if pressure:
            average = sum(pressure) / len(pressure)
            self.counter_pressure = tuple(
                max(0.50, min(2.20, value / max(average, 1e-9)))
                for value in pressure
            )

    def update_opponent_model(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        if offer_utility > self.best_received_utility:
            self.best_received_offer = offer
            self.best_received_utility = offer_utility

        self.opponent_offer_count += 1
        early_weight = 0.45 + (1.0 - state.relative_time)
        recency_weight = 0.7 + 0.8 * state.relative_time
        for issue_index, value in enumerate(offer):
            if issue_index >= len(self.opponent_value_counts):
                continue
            self.opponent_value_counts[issue_index][value] = (
                self.opponent_value_counts[issue_index].get(value, 1.0) + early_weight
            )
            recent_counts = self.recent_opponent_value_counts[issue_index]
            for known_value in recent_counts:
                recent_counts[known_value] *= 0.86
            recent_counts[value] = recent_counts.get(value, 1.0) + recency_weight
        self.update_issue_salience()

    def remember_own_offer(self, offer: Outcome) -> None:
        self.own_offer_count += 1
        for issue_index, value in enumerate(offer):
            if issue_index < len(self.own_value_counts):
                self.own_value_counts[issue_index][value] = (
                    self.own_value_counts[issue_index].get(value, 0.0) + 1.0
                )

    def estimate_opponent_utility(self, outcome: Outcome | None) -> float:
        if outcome is None:
            return 0.0
        if self.opponent_offer_count <= 0:
            return 0.5

        issue_scores = []
        issue_weights = []
        for issue_index, value in enumerate(outcome):
            if issue_index >= len(self.opponent_value_counts):
                continue

            long_counts = self.opponent_value_counts[issue_index]
            recent_counts = self.recent_opponent_value_counts[issue_index]
            long_score = self.normalized_count_score(long_counts, value)
            recent_score = self.normalized_count_score(recent_counts, value)
            issue_scores.append(0.56 * long_score + 0.44 * recent_score)

            values = list(long_counts.values())
            spread = (max(values) - min(values)) / max(max(values), 1e-9)
            salience = (
                self.issue_salience[issue_index]
                if issue_index < len(self.issue_salience)
                else 1.0
            )
            pressure = (
                self.counter_pressure[issue_index]
                if issue_index < len(self.counter_pressure)
                else 1.0
            )
            issue_weights.append(0.72 * (0.18 + spread) + 0.18 * salience + 0.10 * pressure)

        if not issue_scores:
            return 0.5

        weight_sum = sum(issue_weights)
        if weight_sum <= 0:
            return sum(issue_scores) / len(issue_scores)
        return sum(s * w for s, w in zip(issue_scores, issue_weights)) / weight_sum

    def estimate_reported_opponent_utility(self, outcome: Outcome | None) -> float:
        if outcome is None:
            return 0.0
        if self.opponent_offer_count <= 0:
            return 0.5

        issue_scores = []
        issue_weights = []
        for issue_index, value in enumerate(outcome):
            if issue_index >= len(self.opponent_value_counts):
                continue

            long_counts = self.opponent_value_counts[issue_index]
            recent_counts = self.recent_opponent_value_counts[issue_index]
            long_score = self.normalized_count_score(long_counts, value)
            recent_score = self.normalized_count_score(recent_counts, value)
            issue_scores.append(0.70 * long_score + 0.30 * recent_score)

            values = list(long_counts.values())
            spread = (max(values) - min(values)) / max(max(values), 1e-9)
            recent_values = list(recent_counts.values())
            recent_spread = (max(recent_values) - min(recent_values)) / max(
                max(recent_values), 1e-9
            )
            salience = (
                self.issue_salience[issue_index]
                if issue_index < len(self.issue_salience)
                else 1.0
            )
            issue_weights.append(
                0.58 * (0.18 + spread)
                + 0.26 * (0.18 + recent_spread)
                + 0.16 * salience
            )

        if not issue_scores:
            return 0.5

        weight_sum = sum(issue_weights)
        if weight_sum <= 0:
            return sum(issue_scores) / len(issue_scores)
        return sum(s * w for s, w in zip(issue_scores, issue_weights)) / weight_sum

    def update_issue_salience(self) -> None:
        weights = []
        for long_counts, recent_counts in zip(
            self.opponent_value_counts, self.recent_opponent_value_counts
        ):
            long_values = list(long_counts.values())
            recent_values = list(recent_counts.values())
            long_spread = (max(long_values) - min(long_values)) / max(max(long_values), 1e-9)
            recent_spread = (max(recent_values) - min(recent_values)) / max(max(recent_values), 1e-9)
            weights.append(0.25 + 0.45 * long_spread + 0.55 * recent_spread)
        if not weights:
            self.issue_salience = tuple()
            return
        average = sum(weights) / len(weights)
        self.issue_salience = tuple(
            max(0.35, min(2.25, weight / max(average, 1e-9)))
            for weight in weights
        )

    def normalized_count_score(self, counts: dict, value) -> float:
        if not counts:
            return 0.5
        count_values = list(counts.values())
        low, high = min(count_values), max(count_values)
        if high <= low:
            return 0.5
        return (counts.get(value, low) - low) / (high - low)

    def aspiration_level(self, relative_time: float) -> float:
        if self.compact_conflict_domain and relative_time > 0.82:
            return self.own_reserved_value + 0.12 * self.own_utility_range
        concession = 0.015 + 0.30 * (relative_time**3.2)
        if relative_time > 0.93:
            concession += 0.18 * ((relative_time - 0.93) / 0.07) ** 2
        target = self.own_max_utility - self.own_utility_range * concession
        return max(
            self.own_reserved_value + self.minimum_advantage_fraction(relative_time) * self.own_utility_range,
            target,
        )

    def acceptance_level(self, relative_time: float) -> float:
        if self.compact_conflict_domain and relative_time > 0.82:
            return self.own_reserved_value + 0.12 * self.own_utility_range
        concession = 0.035 + 0.34 * (relative_time**3.0)
        if relative_time > 0.94:
            concession += 0.20 * ((relative_time - 0.94) / 0.06) ** 2
        target = self.own_max_utility - self.own_utility_range * concession
        return max(
            self.own_reserved_value + self.acceptance_floor(relative_time) * self.own_utility_range,
            target,
        )

    def minimum_advantage_fraction(self, relative_time: float) -> float:
        if self.compact_conflict_domain and relative_time > 0.82:
            return 0.12
        return 0.625

    def acceptance_floor(self, relative_time: float) -> float:
        if self.compact_conflict_domain and relative_time > 0.82:
            return 0.12
        return 0.65

    def late_floor(self, relative_time: float) -> float:
        if self.compact_conflict_domain:
            return 0.12
        return 0.58

    def needs_agreement_push(self, relative_time: float) -> bool:
        if self.compact_conflict_domain:
            return relative_time > 0.82
        return False

    def offer_score(self, outcome: Outcome, relative_time: float) -> float:
        our_score = (
            self.utility_by_outcome[outcome] - self.own_reserved_value
        ) / self.own_utility_range
        opponent_score = self.estimate_opponent_utility(outcome)
        diversity_score = self.offer_diversity(outcome)
        inertia_score = self.inertia(outcome)
        shielding_score = self.preference_shield(outcome)

        if self.needs_agreement_push(relative_time):
            return (
                0.48 * our_score
                + 0.38 * opponent_score
                + 0.07 * inertia_score
                + 0.04 * diversity_score
                + 0.03 * shielding_score
            )

        if relative_time < 0.28:
            return 0.70 * our_score + 0.20 * shielding_score + 0.08 * diversity_score + 0.02 * opponent_score
        if relative_time < 0.70:
            return 0.64 * our_score + 0.18 * opponent_score + 0.11 * shielding_score + 0.05 * diversity_score + 0.02 * inertia_score
        return 0.60 * our_score + 0.30 * opponent_score + 0.04 * shielding_score + 0.02 * diversity_score + 0.04 * inertia_score

    def offer_diversity(self, outcome: Outcome) -> float:
        if self.own_offer_count <= 0:
            return 1.0

        scores = []
        for issue_index, value in enumerate(outcome):
            if issue_index >= len(self.own_value_counts):
                continue
            counts = self.own_value_counts[issue_index]
            max_count = max(counts.values()) if counts else 0.0
            if max_count <= 0:
                scores.append(1.0)
            else:
                scores.append(1.0 - counts.get(value, 0.0) / max_count)

        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    def inertia(self, outcome: Outcome) -> float:
        if self.best_received_offer is None:
            return 0.5
        matches = sum(1 for a, b in zip(outcome, self.best_received_offer) if a == b)
        return matches / max(len(outcome), 1)

    def normalized_own_utility(self, outcome: Outcome) -> float:
        return (
            self.utility_by_outcome.get(outcome, self.own_reserved_value)
            - self.own_reserved_value
        ) / self.own_utility_range

    def preference_shield(self, outcome: Outcome) -> float:
        if not self.own_value_signal:
            return 0.5
        scores = []
        for issue_index, value in enumerate(outcome):
            if issue_index >= len(self.own_value_signal):
                continue
            scores.append(1.0 - self.own_value_signal[issue_index].get(value, 0.5))
        if not scores:
            return 0.5
        return sum(scores) / len(scores)
