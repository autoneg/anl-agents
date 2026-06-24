"""Agent360 — ANL 2026 submission (V4.2 as ``Agent360``).

Packaged as ``agent360.py`` in ``submitted_v42.zip``.
Same negotiation logic as tournament #19055 (rank 6).
"""

from __future__ import annotations

__version__ = "4.2.0"

import random
from collections import Counter
from typing import Any, Literal

from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import ResponseType, SAOCallNegotiator, SAOResponse, SAOState


class FrequencyOpponentModel:
    """
    Smith-style frequency model of the opponent's preferences.

    Each time the opponent makes an offer, we count how often they use each
    issue-value. Values they repeat often are treated as more valuable to them.
    """

    def __init__(self, num_issues: int) -> None:
        self.num_issues = num_issues
        # opponent_preference_counts[issue_index][value] = how many times seen
        self.opponent_preference_counts: list[dict[Any, int]] = [
            {} for _ in range(num_issues)
        ]

    def record_opponent_offer(self, offer: Outcome) -> None:
        """Update counts from one opponent bid."""
        for issue_index, value in enumerate(offer):
            counts_for_issue = self.opponent_preference_counts[issue_index]
            counts_for_issue[value] = counts_for_issue.get(value, 0) + 1

    def has_observations(self) -> bool:
        """True once we have seen at least one value on some issue."""
        return any(self.opponent_preference_counts[i] for i in range(self.num_issues))

    def estimated_opponent_utility(self, offer: Outcome) -> float:
        """
        Score in [0, 1]: for each issue, value frequency / max frequency on that issue,
        then average across issues. Uniform issue weights (Smith default).
        """
        if self.num_issues == 0:
            return 0.5

        utility_sum = 0.0
        for issue_index in range(self.num_issues):
            counts_for_issue = self.opponent_preference_counts[issue_index]
            if not counts_for_issue:
                utility_sum += 0.5
                continue

            value = offer[issue_index]
            count_for_value = counts_for_issue.get(value, 0)
            max_count_on_issue = max(counts_for_issue.values())
            normalized = count_for_value / max_count_on_issue if max_count_on_issue else 0.5
            utility_sum += normalized

        return utility_sum / self.num_issues


class Agent360Base(SAOCallNegotiator):
    """
    V1 phased negotiation agent (gradient decoy baseline).

    Phases (by relative_time t in [0, 1]):
      1. Decoy   — bid outcomes that misrepresent which issues we care about.
      2. Transition — blend decoy persona with gradually lower true aspirations.
      3. Closing — use opponent model to pick bids they may accept while we still win.
    """

    # Phase boundaries (relative time)
    DECOY_PHASE_END = 0.35
    TRANSITION_PHASE_END = 0.75

    # Closing-phase tuning (override in subclasses for ablations)
    CLOSING_MIN_UTILITY_START = 0.72
    CLOSING_MIN_UTILITY_END = 0.52
    CLOSING_OPPONENT_WEIGHT_BASE = 0.15
    CLOSING_OPPONENT_WEIGHT_SLOPE = 0.35
    CLOSING_OPPONENT_WEIGHT_CAP = 0.45
    TRANSITION_DECOY_MIX_UNTIL = 0.6

    # Populated in on_preferences_changed
    rational_outcomes: tuple[Outcome, ...] = ()
    decoy_outcomes: tuple[Outcome, ...] = ()

    opponent_frequency_model: FrequencyOpponentModel | None = None
    last_counter_offer: Outcome | None = None
    negotiation_seat: int = 0
    n_negotiators: int = 2
    _opponent_offer_count: int = 0
    _recent_own_bids: list[Outcome] | None = None

    # Cap for decoy-rotation history (subclasses may use when first seat)
    OWN_BID_HISTORY_CAP = 12

    def is_first_negotiator(self) -> bool:
        """True when this agent was added first (opens the negotiation)."""
        return self.negotiation_seat == 0

    def _init_negotiation_seat(self) -> None:
        """Record add-order seat from the mechanism (0 = first proposer)."""
        nmi = self.nmi
        if nmi is None:
            return
        self.n_negotiators = nmi.n_negotiators
        mechanism = getattr(nmi, "_mechanism", None)
        if mechanism is not None and self in mechanism.negotiators:
            self.negotiation_seat = mechanism.negotiators.index(self)

    def effective_closing_opponent_weight_cap(self) -> float:
        """Closing opponent-model weight; override for seat-based profiles."""
        return self.CLOSING_OPPONENT_WEIGHT_CAP

    def decoy_phase_end(self) -> float:
        """Decoy→transition boundary; subclasses may set `_decoy_phase_end` per negotiation."""
        return getattr(self, "_decoy_phase_end", self.DECOY_PHASE_END)

    def transition_phase_end(self) -> float:
        """Transition→closing boundary; subclasses may set `_transition_phase_end` per negotiation."""
        return getattr(self, "_transition_phase_end", self.TRANSITION_PHASE_END)

    def transition_allowed(self) -> bool:
        """True when the agent may leave decoy for transition (time alone is not enough if False)."""
        return True

    def effective_transition_decoy_mix_until(self) -> float:
        """Keep mixing decoy outcomes through this fraction of the transition phase."""
        return self.TRANSITION_DECOY_MIX_UNTIL

    def transition_progress_scale(self) -> float:
        """Scale transition progress (lower = slower shift toward true preferences)."""
        return 1.0

    def estimated_opponent_utility(self, offer: Outcome) -> float:
        """Score in [0, 1] for how much the opponent wants ``offer``; override in V3."""
        if self.opponent_frequency_model is None:
            return 0.5
        return self.opponent_frequency_model.estimated_opponent_utility(offer)

    def _scaled_transition_progress(
        self, relative_time: float, decoy_end: float, transition_end: float
    ) -> float:
        raw = (relative_time - decoy_end) / max(1e-9, transition_end - decoy_end)
        return min(1.0, raw * self.transition_progress_scale())

    def on_preferences_changed(self, changes):
        """Build outcome pools and initialize opponent utility estimate."""
        if self.ufun is None:
            return

        self._init_negotiation_seat()

        # All outcomes above reservation, sorted best-for-us first
        utility_and_outcome = [
            (self.ufun(outcome), outcome)
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(outcome) > self.ufun.reserved_value
        ]
        self.rational_outcomes = tuple(
            outcome for _, outcome in sorted(utility_and_outcome, reverse=True)
        )
        self.decoy_outcomes = self._build_decoy_pool()
        self.last_counter_offer = None
        self._opponent_offer_count = 0
        self._recent_own_bids = []

        num_issues = len(self.rational_outcomes[0]) if self.rational_outcomes else 0
        self.opponent_frequency_model = FrequencyOpponentModel(num_issues)
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=self.opponent_frequency_model.estimated_opponent_utility
        )

    def _build_decoy_pool(self) -> tuple[Outcome, ...]:
        """
        Outcomes that look like we care about different issues than we really do.

        1. Infer our true favorite value per issue from our top outcomes.
        2. Keep rational outcomes that disagree on many issues but stay above a utility floor.
        """
        if not self.rational_outcomes:
            return ()

        num_rational = len(self.rational_outcomes)
        top_k = max(3, min(30, num_rational // 10 or 3))
        our_best_outcomes = self.rational_outcomes[:top_k]
        num_issues = len(our_best_outcomes[0])

        # Most common value per issue among our best outcomes = our true preference signal
        true_preferred_value_per_issue: list[Any] = []
        for issue_index in range(num_issues):
            values_in_top = [o[issue_index] for o in our_best_outcomes]
            true_preferred_value_per_issue.append(
                Counter(values_in_top).most_common(1)[0][0]
            )

        reserved_utility = float(self.ufun.reserved_value)
        min_decoy_utility = max(
            reserved_utility, float(self.ufun(our_best_outcomes[-1])) * 0.55
        )

        decoy_candidates: list[Outcome] = []
        for outcome in self.rational_outcomes:
            outcome_utility = float(self.ufun(outcome))
            if outcome_utility < min_decoy_utility:
                break

            num_mismatched_issues = sum(
                1
                for issue_index in range(num_issues)
                if outcome[issue_index] != true_preferred_value_per_issue[issue_index]
            )
            min_mismatches = max(1, num_issues // 3)
            if num_mismatched_issues >= min_mismatches:
                decoy_candidates.append(outcome)

        # Fallback: mid-ranked rational outcomes if we found too few decoys
        if len(decoy_candidates) < 3:
            mid_start = min(top_k, num_rational - 1)
            mid_end = min(num_rational, mid_start + max(10, num_rational // 5))
            decoy_candidates = list(self.rational_outcomes[mid_start:mid_end])

        return tuple(decoy_candidates)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Main SAO loop: accept partner offer or send a counter-offer."""
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        partner_offer = state.current_offer

        if partner_offer is None:
            counter = self.concealing_bidding_strategy(state)
            self.last_counter_offer = counter
            return SAOResponse(ResponseType.REJECT_OFFER, counter)

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, partner_offer)

        counter = self.concealing_bidding_strategy(state)
        self.last_counter_offer = counter
        return SAOResponse(ResponseType.REJECT_OFFER, counter)

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Accept if the offer meets a time-decaying aspiration, beats our next bid,
        or we are near the deadline with a safe deal.
        """
        assert self.ufun
        partner_offer = state.current_offer
        if partner_offer is None:
            return False

        offer_utility_for_me = float(self.ufun(partner_offer))
        relative_time = state.relative_time
        max_utility_for_me = float(self.ufun.max())

        # Aspiration level drops as deadline approaches
        aspiration_utility = max_utility_for_me * (1.0 - 0.55 * relative_time)
        if (
            offer_utility_for_me >= aspiration_utility
            and offer_utility_for_me > float(self.ufun.reserved_value)
        ):
            return True

        # Accept if partner offer is at least as good as what we would bid next
        our_next_bid = self.concealing_bidding_strategy(state)
        if our_next_bid is not None:
            utility_of_our_next_bid = float(self.ufun(our_next_bid))
            if offer_utility_for_me >= utility_of_our_next_bid - 1e-9:
                return True

        # Last resort before timeout
        if relative_time > 0.92 and offer_utility_for_me > float(
            self.ufun.reserved_value
        ) * 1.02:
            return True

        return False

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """Pick the next counter-offer based on negotiation phase."""
        if not self.rational_outcomes:
            return None

        relative_time = state.relative_time
        # Deterministic randomness per step (reproducible for a given negotiator id)
        rng = random.Random(hash((self.id, state.step)) & 0xFFFFFFFF)

        if self._in_decoy_phase(relative_time):
            bid = self._pick_decoy_bid(rng)
            self._record_own_bid(bid)
            return bid

        if relative_time < self.transition_phase_end():
            candidate_pool = self._build_transition_candidate_pool(relative_time, rng)
            bid = rng.choice(candidate_pool)
            self._record_own_bid(bid)
            return bid

        bid = self._pick_closing_bid(relative_time, rng)
        self._record_own_bid(bid)
        return bid

    def _in_decoy_phase(self, relative_time: float) -> bool:
        """Decoy phase by time, optionally extended until transition is allowed."""
        if relative_time < self.decoy_phase_end():
            return bool(self.decoy_outcomes)
        if not self.transition_allowed():
            return bool(self.decoy_outcomes)
        return False

    def _pick_decoy_bid(self, rng: random.Random) -> Outcome:
        if self.decoy_outcomes:
            return rng.choice(self.decoy_outcomes)
        return rng.choice(self.rational_outcomes)

    def _record_own_bid(self, bid: Outcome | None) -> None:
        if bid is None:
            return
        if self._recent_own_bids is None:
            self._recent_own_bids = []
        self._recent_own_bids.append(bid)
        if len(self._recent_own_bids) > self.OWN_BID_HISTORY_CAP:
            self._recent_own_bids.pop(0)

    def _build_transition_candidate_pool(
        self, relative_time: float, rng: random.Random
    ) -> list[Outcome]:
        """Gradually shift from decoy bids toward our true aspiration band."""
        max_utility_for_me = float(self.ufun.max())
        decoy_end = self.decoy_phase_end()
        transition_end = self.transition_phase_end()
        transition_progress = self._scaled_transition_progress(
            relative_time, decoy_end, transition_end
        )
        min_utility_in_band = max_utility_for_me * (0.92 - 0.35 * transition_progress)

        true_preference_band = [
            o
            for o in self.rational_outcomes
            if float(self.ufun(o)) >= min_utility_in_band
        ]
        if not true_preference_band:
            true_preference_band = list(self.rational_outcomes[:10])

        decoy_mix_until = self.effective_transition_decoy_mix_until()
        if self.decoy_outcomes and transition_progress < decoy_mix_until:
            num_decoy_slots = max(1, int(len(true_preference_band) * (1.0 - transition_progress)))
            decoy_slice = list(self.decoy_outcomes[: max(5, num_decoy_slots)])
            return decoy_slice + true_preference_band

        return true_preference_band

    def _pick_closing_bid(self, relative_time: float, rng: random.Random) -> Outcome:
        """
        Late phase: maximize a blend of our utility and estimated opponent utility.

        Goal: offer something they think is great (bait-and-switch after decoy phase)
        while still keeping enough utility for us.
        """
        max_utility_for_me = float(self.ufun.max())
        transition_end = self.transition_phase_end()
        min_closing_utility = max(
            float(self.ufun.reserved_value),
            max_utility_for_me
            * (
                self.CLOSING_MIN_UTILITY_START
                - (self.CLOSING_MIN_UTILITY_START - self.CLOSING_MIN_UTILITY_END)
                * (relative_time - transition_end)
                / max(1e-9, 1.0 - transition_end)
            ),
        )

        closing_candidates = [
            o for o in self.rational_outcomes if float(self.ufun(o)) >= min_closing_utility
        ]
        if not closing_candidates:
            closing_candidates = list(self.rational_outcomes[:20])

        opponent_model = self.opponent_frequency_model
        if opponent_model is None or not opponent_model.has_observations():
            return rng.choice(closing_candidates[: min(15, len(closing_candidates))])

        # Weight on opponent utility grows through the closing phase
        opponent_utility_weight = min(
            self.effective_closing_opponent_weight_cap(),
            self.CLOSING_OPPONENT_WEIGHT_BASE
            + self.CLOSING_OPPONENT_WEIGHT_SLOPE * (relative_time - transition_end),
        )
        best_combined_score = -1.0
        best_outcomes: list[Outcome] = []

        sample = (
            closing_candidates
            if len(closing_candidates) <= 40
            else rng.sample(closing_candidates, 40)
        )
        for outcome in sample:
            my_utility = float(self.ufun(outcome))
            their_estimated_utility = self.estimated_opponent_utility(outcome)
            combined_score = (1.0 - opponent_utility_weight) * (
                my_utility / max_utility_for_me
            ) + opponent_utility_weight * their_estimated_utility

            if combined_score > best_combined_score + 1e-9:
                best_combined_score = combined_score
                best_outcomes = [outcome]
            elif abs(combined_score - best_combined_score) < 1e-9:
                best_outcomes.append(outcome)

        return rng.choice(best_outcomes)

    def update_opponent_model(self, state: SAOState) -> None:
        """Learn from the opponent's latest offer."""
        partner_offer = state.current_offer
        if partner_offer is None or self.opponent_frequency_model is None:
            return
        self.opponent_frequency_model.record_opponent_offer(partner_offer)
        self._opponent_offer_count += 1


class Agent360V2(Agent360Base):
    """
    V2.4 decoy persona: maximal-mismatch decoy pool + first-seat min-offer gate.

    When opening (seat 0), stay in decoy until the opponent has made at least
    ``FIRST_MIN_OPPONENT_OFFERS`` bids — reduces curve-fit leaks without
    opponent-type routing.
    """

    FIRST_MIN_OPPONENT_OFFERS = 3

    def transition_allowed(self) -> bool:
        if not self.is_first_negotiator() or self.FIRST_MIN_OPPONENT_OFFERS <= 0:
            return True
        return self._opponent_offer_count >= self.FIRST_MIN_OPPONENT_OFFERS

    def _true_preferred_value_per_issue(self) -> list[Any]:
        if not self.rational_outcomes:
            return []

        num_rational = len(self.rational_outcomes)
        top_k = max(3, min(30, num_rational // 10 or 3))
        our_best_outcomes = self.rational_outcomes[:top_k]
        num_issues = len(our_best_outcomes[0])

        true_preferred: list[Any] = []
        for issue_index in range(num_issues):
            values_in_top = [o[issue_index] for o in our_best_outcomes]
            true_preferred.append(Counter(values_in_top).most_common(1)[0][0])
        return true_preferred

    def _build_decoy_pool(self) -> tuple[Outcome, ...]:
        """Maximal mismatch decoys for decoy and transition phases."""
        if not self.rational_outcomes:
            return ()

        true_preferred_value_per_issue = self._true_preferred_value_per_issue()
        if not true_preferred_value_per_issue:
            return super()._build_decoy_pool()

        num_issues = len(true_preferred_value_per_issue)
        num_rational = len(self.rational_outcomes)
        top_k = max(3, min(30, num_rational // 10 or 3))
        our_best_outcomes = self.rational_outcomes[:top_k]

        reserved_utility = float(self.ufun.reserved_value)
        min_decoy_utility = max(
            reserved_utility, float(self.ufun(our_best_outcomes[-1])) * 0.55
        )

        scored_candidates: list[tuple[int, Outcome]] = []
        for outcome in self.rational_outcomes:
            outcome_utility = float(self.ufun(outcome))
            if outcome_utility < min_decoy_utility:
                break

            num_mismatched_issues = sum(
                1
                for issue_index in range(num_issues)
                if outcome[issue_index] != true_preferred_value_per_issue[issue_index]
            )
            if num_mismatched_issues >= max(1, num_issues // 2):
                scored_candidates.append((num_mismatched_issues, outcome))

        if not scored_candidates:
            return super()._build_decoy_pool()

        max_mismatches = max(score for score, _ in scored_candidates)
        decoy_candidates = [
            outcome
            for score, outcome in scored_candidates
            if score == max_mismatches
        ]
        if len(decoy_candidates) < 3:
            near_max = [
                outcome
                for score, outcome in scored_candidates
                if score >= max_mismatches - 1
            ]
            decoy_candidates = near_max or decoy_candidates

        return tuple(decoy_candidates)


OpponentMode = Literal["unknown", "mirror", "conceding", "learner", "deceptive"]


def issue_weighted_smith_estimate(
    offers: list[Outcome],
    offer: Outcome,
    *,
    num_issues: int,
) -> float:
    """
    Smith estimate with higher weight on issues the opponent actually negotiates.

    Issues they repeat one value on (noise / decoy) get low weight; issues with
    spread signal real preferences — matches how rational-filter agents behave.
    """
    if not offers or num_issues == 0:
        return 0.5

    weighted_sum = 0.0
    weight_total = 0.0
    for issue_index in range(num_issues):
        values = [o[issue_index] for o in offers]
        counts = Counter(values)
        n = len(values)
        distinct = len(counts)
        max_count = max(counts.values())
        concentration = max_count / n
        issue_weight = max(0.12, (1.0 - concentration) + 0.18 * (distinct - 1))
        value = offer[issue_index]
        score = counts.get(value, 0) / max_count if max_count else 0.5
        weighted_sum += issue_weight * score
        weight_total += issue_weight

    if weight_total < 1e-12:
        return 0.5
    return weighted_sum / weight_total


class RecencyBlendedSmith:
    """Smith frequency model with extra weight on recent opponent bids."""

    def __init__(self, full_model: FrequencyOpponentModel, window: int = 5) -> None:
        self.full_model = full_model
        self.window = window
        self._recent: list[Outcome] = []

    def record(self, offer: Outcome) -> None:
        self._recent.append(offer)
        if len(self._recent) > self.window:
            self._recent.pop(0)

    def recent_count(self) -> int:
        return len(self._recent)

    def estimated(self, offer: Outcome) -> float:
        full_u = self.full_model.estimated_opponent_utility(offer)
        n = len(self._recent)
        if n < 2:
            return full_u

        recent_model = FrequencyOpponentModel(self.full_model.num_issues)
        for recent_offer in self._recent:
            recent_model.record_opponent_offer(recent_offer)
        recent_u = recent_model.estimated_opponent_utility(offer)

        weight = min(0.68, 0.22 + 0.09 * n)
        return (1.0 - weight) * full_u + weight * recent_u


class TimedOpponentModel:
    """Opponent Smith model that up-weights bids after their likely decoy phase."""

    def __init__(
        self,
        num_issues: int,
        *,
        late_time_threshold: float = 0.40,
        late_bid_weight: int = 3,
    ) -> None:
        self.num_issues = num_issues
        self.late_time_threshold = late_time_threshold
        self.late_bid_weight = late_bid_weight
        self._timed_offers: list[tuple[float, Outcome]] = []

    def record(self, relative_time: float, offer: Outcome) -> None:
        self._timed_offers.append((relative_time, offer))

    def offer_count(self) -> int:
        return len(self._timed_offers)

    def late_count(self) -> int:
        return sum(
            1 for t, _ in self._timed_offers if t >= self.late_time_threshold
        )

    def _build_weighted_model(self) -> FrequencyOpponentModel:
        model = FrequencyOpponentModel(self.num_issues)
        for relative_time, offer in self._timed_offers:
            repeats = (
                self.late_bid_weight
                if relative_time >= self.late_time_threshold
                else 1
            )
            for _ in range(repeats):
                model.record_opponent_offer(offer)
        return model

    def late_phase_estimated(self, offer: Outcome) -> float:
        late_offers = [
            o for t, o in self._timed_offers if t >= self.late_time_threshold
        ]
        if len(late_offers) >= 2:
            model = FrequencyOpponentModel(self.num_issues)
            for late_offer in late_offers:
                model.record_opponent_offer(late_offer)
            return model.estimated_opponent_utility(offer)
        if self._timed_offers:
            return self._build_weighted_model().estimated_opponent_utility(offer)
        return 0.5

    def late_offers(self) -> list[Outcome]:
        return [o for t, o in self._timed_offers if t >= self.late_time_threshold]

    def late_issue_weighted_estimated(self, offer: Outcome) -> float:
        late = self.late_offers()
        if len(late) >= 2:
            return issue_weighted_smith_estimate(
                late, offer, num_issues=self.num_issues
            )
        return self.late_phase_estimated(offer)

    def weighted_estimated(self, offer: Outcome) -> float:
        if not self._timed_offers:
            return 0.5
        return self._build_weighted_model().estimated_opponent_utility(offer)


class OfferTrajectoryModel:
    """Track opponent (time, Smith-estimated utility) pairs from their offers."""

    def __init__(self) -> None:
        self._samples: list[tuple[float, float]] = []

    def record(self, relative_time: float, estimated_utility: float) -> None:
        self._samples.append(
            (relative_time, max(0.0, min(1.0, estimated_utility)))
        )

    def sample_count(self) -> int:
        return len(self._samples)

    def has_observations(self) -> bool:
        return bool(self._samples)

    def last_sample(self) -> tuple[float, float] | None:
        if not self._samples:
            return None
        return self._samples[-1]

    def concession_slope(self) -> float:
        n = len(self._samples)
        if n < 2:
            return 0.0

        times = [t for t, _ in self._samples]
        utils = [u for _, u in self._samples]
        mean_t = sum(times) / n
        mean_u = sum(utils) / n
        var_t = sum((t - mean_t) ** 2 for t in times)
        if var_t < 1e-12:
            return 0.0
        cov = sum((t - mean_t) * (u - mean_u) for t, u in self._samples)
        return cov / var_t

    def predicted_utility_at(self, relative_time: float) -> float:
        if not self._samples:
            return 0.5
        if len(self._samples) == 1:
            return self._samples[0][1]

        last_t, last_u = self._samples[-1]
        predicted = last_u + self.concession_slope() * (relative_time - last_t)
        return max(0.0, min(1.0, predicted))

    def is_non_monotone(self, tolerance: float = 0.05) -> bool:
        if len(self._samples) < 3:
            return False
        utils = [u for _, u in self._samples]
        rises = sum(
            1 for i in range(1, len(utils)) if utils[i] > utils[i - 1] + tolerance
        )
        return rises >= 2

    def inconsistency_vs_trajectory(
        self, estimated_utility: float, relative_time: float
    ) -> float:
        return abs(estimated_utility - self.predicted_utility_at(relative_time))


class Agent360(Agent360V2):
    """
    Submission agent (V4.2): persona + opponent model + tournament-grade deal extraction.

    Proven ~9th on official panel. V4.3 changes reverted after rank-28 regression.
    """

    FIRST_MIN_OPPONENT_OFFERS = 4
    FIRST_DECOY_PHASE_END = 0.40
    FIRST_TRANSITION_DECOY_MIX_UNTIL = 0.85
    FIRST_TRANSITION_PROGRESS_SCALE = 0.72

    MIN_TRAJECTORY_SAMPLES = 3
    HONEST_CONCESSION_SLOPE = -0.04

    BAIT_THRESHOLD = 0.14
    BAIT_DISCOUNT = 0.30
    ACCEPT_BAIT_THRESHOLD = 0.14
    BAIT_MIN_TRAJECTORY_SAMPLES = 5
    ACCEPT_DEADLINE_SAFE = 0.90
    ACCEPT_CATASTROPHE_TIME = 0.95
    ACCEPT_LATE_RV_FACTOR = 1.0
    INCONSISTENCY_BLEND_THRESHOLD = 0.18
    INCONSISTENCY_BLEND = 0.38

    EARLY_DECOY_FLIP_RATE = 0.25
    EARLY_DECOY_MIN_OFFERS = 3
    EARLY_SMITH_SPREAD = 0.12
    MIRROR_MATCH_WINDOW = 4
    MIRROR_MATCH_MIN = 3

    RECENCY_WINDOW = 5
    STABLE_ISSUE_WINDOW = 4
    STABLE_ISSUE_BLEND = 0.14
    CONCEDING_SLOPE_THRESHOLD = -0.025

    LATE_TIME_THRESHOLD = 0.40
    LATE_BID_WEIGHT = 3
    LATE_BLEND_MAX = 0.55
    FIRST_LATE_BLEND_MAX = 0.68
    ISSUE_WEIGHT_BLEND_MAX = 0.32

    LEARNER_CONCENTRATION = 0.78
    LEARNER_MIN_OFFERS = 4

    FIRST_DECOY_NO_REPEAT_WINDOW = 5

    CONCEDING_EARLY_EXIT_MIN_OPP_OFFERS = 1
    CLOSING_CAP_CONCEDING = 0.52
    CLOSING_CAP_LEARNER = 0.48
    CLOSING_CAP_DECEPTIVE = 0.32
    CLOSING_CAP_UNKNOWN = 0.40
    CLOSING_CAP_MIRROR = 0.38
    CLOSING_MIN_UTILITY_START_LEARNER = 0.70
    CLOSING_MIN_UTILITY_START_DECEPTIVE = 0.75
    CLOSING_LEARNER_UTILITY_BOOST = 0.04
    CLOSING_LEARNER_SAMPLE_CAP = 40
    ASPIRATION_SLOPE_DEFAULT = 0.55
    ASPIRATION_SLOPE_CONCEDING = 0.42
    ASPIRATION_SLOPE_LEARNER = 0.52
    ASPIRATION_SLOPE_DECEPTIVE = 0.58
    ASPIRATION_DECEPTIVE_UNTIL = 0.85

    offer_trajectory_model: OfferTrajectoryModel | None = None
    _recency_blended: RecencyBlendedSmith | None = None
    _timed_opponent: TimedOpponentModel | None = None
    _eval_relative_time: float = 0.0
    _opponent_recent_offers: list[Outcome] | None = None
    _opponent_offer_history: list[tuple[float, Outcome]] | None = None

    def on_preferences_changed(self, changes) -> None:
        super().on_preferences_changed(changes)
        if getattr(self, "_recent_own_bids", None) is None:
            self._recent_own_bids = []
        self.offer_trajectory_model = OfferTrajectoryModel()
        self._opponent_recent_offers = []
        self._opponent_offer_history = []
        if self.opponent_frequency_model is not None:
            num_issues = self.opponent_frequency_model.num_issues
            self._recency_blended = RecencyBlendedSmith(
                self.opponent_frequency_model, window=self.RECENCY_WINDOW
            )
            self._timed_opponent = TimedOpponentModel(
                num_issues,
                late_time_threshold=self.LATE_TIME_THRESHOLD,
                late_bid_weight=self.LATE_BID_WEIGHT,
            )
            self.private_info["opponent_ufun"] = LambdaMultiFun(
                f=self._published_opponent_utility
            )

    def decoy_phase_end(self) -> float:
        if self.is_first_negotiator():
            return self.FIRST_DECOY_PHASE_END
        return Agent360Base.decoy_phase_end(self)

    def effective_transition_decoy_mix_until(self) -> float:
        if self.is_first_negotiator():
            return self.FIRST_TRANSITION_DECOY_MIX_UNTIL
        return super().effective_transition_decoy_mix_until()

    def transition_progress_scale(self) -> float:
        if self.is_first_negotiator():
            return self.FIRST_TRANSITION_PROGRESS_SCALE
        return 1.0

    def _anti_mirror_pool(self, pool: list[Outcome]) -> list[Outcome]:
        if not self._opponent_mirrors_us():
            return pool
        opp_offers = self._opponent_recent_offers or []
        if not opp_offers:
            return pool
        last_opp = opp_offers[-1]
        alt = [outcome for outcome in pool if outcome != last_opp]
        return alt or pool

    def _pick_decoy_bid(self, rng: random.Random) -> Outcome:
        """First seat: rotate decoys + utility jumps; break mirror tit-for-tat."""
        if not self.is_first_negotiator() or not self.decoy_outcomes:
            pool = list(self.decoy_outcomes) if self.decoy_outcomes else []
            if pool:
                return rng.choice(self._anti_mirror_pool(pool))
            return super()._pick_decoy_bid(rng)

        own_bids = getattr(self, "_recent_own_bids", None) or []
        pool = list(self.decoy_outcomes)

        if self.FIRST_DECOY_NO_REPEAT_WINDOW > 0 and own_bids:
            recent = set(own_bids[-self.FIRST_DECOY_NO_REPEAT_WINDOW :])
            filtered = [outcome for outcome in pool if outcome not in recent]
            if filtered:
                pool = filtered

        pool = self._anti_mirror_pool(pool)

        if own_bids:
            last_utility = float(self.ufun(own_bids[-1]))
            best_gap = -1.0
            tied: list[Outcome] = []
            for outcome in pool:
                gap = abs(float(self.ufun(outcome)) - last_utility)
                if gap > best_gap + 1e-9:
                    best_gap = gap
                    tied = [outcome]
                elif abs(gap - best_gap) < 1e-9:
                    tied.append(outcome)
            if tied:
                return rng.choice(tied)

        return rng.choice(pool)

    def _ensure_valid_bid(self, bid: Outcome | None) -> Outcome:
        if bid is not None and self.rational_outcomes:
            return bid
        if self.rational_outcomes:
            return self.rational_outcomes[0]
        if self.ufun is not None and self.nmi is not None:
            for outcome in self.nmi.outcome_space.enumerate_or_sample():
                if float(self.ufun(outcome)) > float(self.ufun.reserved_value):
                    return outcome
        raise RuntimeError("Agent360: no valid bid available")

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        partner_offer = state.current_offer

        if partner_offer is None:
            counter = self._ensure_valid_bid(self.concealing_bidding_strategy(state))
            self.last_counter_offer = counter
            return SAOResponse(ResponseType.REJECT_OFFER, counter)

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, partner_offer)

        counter = self._ensure_valid_bid(self.concealing_bidding_strategy(state))
        self.last_counter_offer = counter
        return SAOResponse(ResponseType.REJECT_OFFER, counter)

    def transition_allowed(self) -> bool:
        if self._opponent_conceding_early_exit():
            if self._opponent_offer_count >= self.CONCEDING_EARLY_EXIT_MIN_OPP_OFFERS:
                return True
        return super().transition_allowed()

    def _opponent_conceding_early_exit(self) -> bool:
        traj = self.offer_trajectory_model
        if traj is None or traj.sample_count() < self.MIN_TRAJECTORY_SAMPLES:
            return False
        if self._opponent_mode() != "conceding":
            return False
        return traj.concession_slope() <= self.CONCEDING_SLOPE_THRESHOLD

    def _closing_bluff_active(self, relative_time: float | None = None) -> bool:
        t = self._eval_relative_time if relative_time is None else relative_time
        if t < self.transition_phase_end():
            return False
        traj = self.offer_trajectory_model
        return traj is not None and traj.sample_count() >= self.MIN_TRAJECTORY_SAMPLES

    def update_opponent_model(self, state: SAOState) -> None:
        super().update_opponent_model(state)
        partner_offer = state.current_offer
        if (
            partner_offer is None
            or self.offer_trajectory_model is None
            or self.opponent_frequency_model is None
        ):
            return
        if self._recency_blended is not None:
            self._recency_blended.record(partner_offer)
        if self._timed_opponent is not None:
            self._timed_opponent.record(state.relative_time, partner_offer)
        smith_u = self.opponent_frequency_model.estimated_opponent_utility(
            partner_offer
        )
        self.offer_trajectory_model.record(state.relative_time, smith_u)
        if self._opponent_recent_offers is not None:
            self._opponent_recent_offers.append(partner_offer)
            if len(self._opponent_recent_offers) > self.MIRROR_MATCH_WINDOW:
                self._opponent_recent_offers.pop(0)
        if self._opponent_offer_history is not None:
            self._opponent_offer_history.append((state.relative_time, partner_offer))

    def _opponent_mirrors_us(self) -> bool:
        own_bids = getattr(self, "_recent_own_bids", None) or []
        opp_offers = self._opponent_recent_offers or []
        if not own_bids or not opp_offers:
            return False
        window = min(
            self.MIRROR_MATCH_WINDOW,
            len(own_bids),
            len(opp_offers),
        )
        if window < self.MIRROR_MATCH_MIN:
            return False
        own = own_bids[-window:]
        opp = opp_offers[-window:]
        matches = sum(1 for a, b in zip(own, opp) if a == b)
        return matches >= self.MIRROR_MATCH_MIN

    def _early_offers(self) -> list[Outcome]:
        history = self._opponent_offer_history
        if not history:
            return []
        return [o for t, o in history if t < self.LATE_TIME_THRESHOLD]

    def _early_issue_flip_rate(self) -> float:
        early = self._early_offers()
        if len(early) < 2:
            return 0.0
        flips = sum(1 for i in range(1, len(early)) if early[i] != early[i - 1])
        return flips / (len(early) - 1)

    def _early_smith_util_spread(self) -> float:
        early = self._early_offers()
        if len(early) < 3 or self.opponent_frequency_model is None:
            return 0.0
        utils = [
            self.opponent_frequency_model.estimated_opponent_utility(o) for o in early
        ]
        return max(utils) - min(utils)

    def _opponent_early_decoy_persona(self) -> bool:
        early = self._early_offers()
        if len(early) < self.EARLY_DECOY_MIN_OFFERS:
            return False
        if len(set(early)) == 1:
            return True
        if self._early_issue_flip_rate() >= self.EARLY_DECOY_FLIP_RATE:
            return True
        return self._early_smith_util_spread() >= self.EARLY_SMITH_SPREAD

    def _opponent_offer_concentration(self, offers: list[Outcome]) -> float:
        if not offers:
            return 0.0
        num_issues = len(offers[0])
        per_issue: list[float] = []
        for issue_index in range(num_issues):
            counts = Counter(o[issue_index] for o in offers)
            per_issue.append(max(counts.values()) / len(offers))
        return sum(per_issue) / num_issues

    def _opponent_late_bait_switch(self) -> bool:
        history = self._opponent_offer_history or []
        late = [o for t, o in history if t >= self.LATE_TIME_THRESHOLD]
        if len(late) < 4:
            return False
        mid = len(late) // 2
        first, second = late[:mid], late[mid:]
        num_issues = len(late[0])
        flips = 0
        for issue_index in range(num_issues):
            pref_first = Counter(o[issue_index] for o in first).most_common(1)[0][0]
            pref_second = Counter(o[issue_index] for o in second).most_common(1)[0][0]
            if pref_first != pref_second:
                flips += 1
        return flips >= max(1, num_issues // 2)

    def _opponent_shows_concealment_tactics(self) -> bool:
        traj = self.offer_trajectory_model
        if self._opponent_early_decoy_persona():
            return True
        if traj is not None and traj.is_non_monotone():
            return True
        return self._opponent_late_bait_switch()

    def _opponent_smith_learner_profile(self) -> bool:
        recent = self._opponent_recent_offers or []
        if len(recent) < self.LEARNER_MIN_OFFERS:
            return False
        if self._opponent_shows_concealment_tactics():
            return False
        concentration = self._opponent_offer_concentration(recent)
        traj = self.offer_trajectory_model
        if traj is not None and traj.is_non_monotone():
            return False
        return concentration >= self.LEARNER_CONCENTRATION

    def _opponent_mode(self) -> OpponentMode:
        if self._opponent_mirrors_us():
            return "mirror"
        traj = self.offer_trajectory_model
        if traj is None or traj.sample_count() < 2:
            return "unknown"
        if self._opponent_smith_learner_profile():
            return "learner"
        if self._opponent_shows_concealment_tactics():
            return "deceptive"
        if (
            traj.sample_count() >= self.MIN_TRAJECTORY_SAMPLES
            and traj.concession_slope() <= self.CONCEDING_SLOPE_THRESHOLD
        ):
            return "conceding"
        return "learner"

    def _stable_issue_match_score(self, offer: Outcome) -> float:
        recent = self._opponent_recent_offers or []
        if len(recent) < 3 or not offer:
            return 0.5
        window = recent[-self.STABLE_ISSUE_WINDOW :]
        num_issues = len(offer)
        matches = 0
        for issue_index in range(num_issues):
            values = [o[issue_index] for o in window]
            preferred = Counter(values).most_common(1)[0][0]
            if offer[issue_index] == preferred:
                matches += 1
        return matches / num_issues

    def _blended_opponent_utility(self, offer: Outcome) -> float:
        if self.opponent_frequency_model is None:
            return 0.5
        if self._opponent_mirrors_us():
            return self.opponent_frequency_model.estimated_opponent_utility(offer)

        full_u = self.opponent_frequency_model.estimated_opponent_utility(offer)
        blended = full_u

        timed = self._timed_opponent
        if timed is not None and timed.late_count() >= 2:
            late_u = timed.late_phase_estimated(offer)
            late_weight = min(
                self.FIRST_LATE_BLEND_MAX if self.is_first_negotiator() else self.LATE_BLEND_MAX,
                0.22 + 0.08 * timed.late_count(),
            )
            blended = (1.0 - late_weight) * blended + late_weight * late_u

        if timed is not None and len(timed.late_offers()) >= 3:
            issue_u = timed.late_issue_weighted_estimated(offer)
            issue_weight = min(
                self.ISSUE_WEIGHT_BLEND_MAX,
                0.10 + 0.04 * len(timed.late_offers()),
            )
            blended = (1.0 - issue_weight) * blended + issue_weight * issue_u

        if not self.is_first_negotiator() and self._recency_blended is not None:
            if self._recency_blended.recent_count() >= 2:
                recency_u = self._recency_blended.estimated(offer)
                recency_weight = min(
                    0.55, 0.18 + 0.08 * self._recency_blended.recent_count()
                )
                blended = (1.0 - recency_weight) * blended + recency_weight * recency_u

            mode = self._opponent_mode()
            if mode in ("learner", "conceding", "unknown") and (
                self._recency_blended.recent_count() >= 3
            ):
                stable = self._stable_issue_match_score(offer)
                blended = (1.0 - self.STABLE_ISSUE_BLEND) * blended + self.STABLE_ISSUE_BLEND * stable

        return max(0.0, min(1.0, blended))

    def _published_opponent_utility(self, offer: Outcome) -> float:
        return self._blended_opponent_utility(offer)

    def effective_closing_opponent_weight_cap(self) -> float:
        mode = self._opponent_mode()
        caps = {
            "conceding": self.CLOSING_CAP_CONCEDING,
            "learner": self.CLOSING_CAP_LEARNER,
            "deceptive": self.CLOSING_CAP_DECEPTIVE,
            "mirror": self.CLOSING_CAP_MIRROR,
            "unknown": self.CLOSING_CAP_UNKNOWN,
        }
        return caps.get(mode, self.CLOSING_CAP_UNKNOWN)

    def _closing_min_utility_start(self) -> float:
        mode = self._opponent_mode()
        if mode == "deceptive":
            return self.CLOSING_MIN_UTILITY_START_DECEPTIVE
        if mode == "learner":
            return self.CLOSING_MIN_UTILITY_START_LEARNER
        return self.CLOSING_MIN_UTILITY_START

    def _pick_closing_bid(self, relative_time: float, rng: random.Random) -> Outcome:
        max_utility_for_me = float(self.ufun.max())
        transition_end = self.transition_phase_end()
        min_start = self._closing_min_utility_start()
        min_closing_utility = max(
            float(self.ufun.reserved_value),
            max_utility_for_me
            * (
                min_start
                - (min_start - self.CLOSING_MIN_UTILITY_END)
                * (relative_time - transition_end)
                / max(1e-9, 1.0 - transition_end)
            ),
        )

        closing_candidates = [
            o for o in self.rational_outcomes if float(self.ufun(o)) >= min_closing_utility
        ]
        if not closing_candidates:
            closing_candidates = list(self.rational_outcomes[:20])

        opponent_model = self.opponent_frequency_model
        if opponent_model is None or not opponent_model.has_observations():
            return rng.choice(closing_candidates[: min(15, len(closing_candidates))])

        opponent_utility_weight = min(
            self.effective_closing_opponent_weight_cap(),
            self.CLOSING_OPPONENT_WEIGHT_BASE
            + self.CLOSING_OPPONENT_WEIGHT_SLOPE * (relative_time - transition_end),
        )
        best_combined_score = -1.0
        best_outcomes: list[Outcome] = []

        sample_cap = 40
        if self._opponent_mode() == "learner":
            sample_cap = self.CLOSING_LEARNER_SAMPLE_CAP
        sample = (
            closing_candidates
            if len(closing_candidates) <= sample_cap
            else rng.sample(closing_candidates, sample_cap)
        )
        learner_boost = (
            self.CLOSING_LEARNER_UTILITY_BOOST
            if self._opponent_mode() == "learner" and relative_time > 0.55
            else 0.0
        )
        my_weight = min(0.92, (1.0 - opponent_utility_weight) + learner_boost)

        for outcome in sample:
            my_utility = float(self.ufun(outcome))
            their_estimated_utility = self.estimated_opponent_utility(outcome)
            combined_score = my_weight * (
                my_utility / max_utility_for_me
            ) + (1.0 - my_weight) * their_estimated_utility

            if combined_score > best_combined_score + 1e-9:
                best_combined_score = combined_score
                best_outcomes = [outcome]
            elif abs(combined_score - best_combined_score) < 1e-9:
                best_outcomes.append(outcome)

        return rng.choice(best_outcomes)

    def _aspiration_time_slope(self, relative_time: float) -> float:
        mode = self._opponent_mode()
        if mode == "conceding":
            return self.ASPIRATION_SLOPE_CONCEDING
        if mode == "learner":
            return self.ASPIRATION_SLOPE_LEARNER
        if mode == "deceptive" and relative_time < self.ASPIRATION_DECEPTIVE_UNTIL:
            return self.ASPIRATION_SLOPE_DECEPTIVE
        return self.ASPIRATION_SLOPE_DEFAULT

    def acceptance_strategy(self, state: SAOState) -> bool:
        self._eval_relative_time = state.relative_time
        assert self.ufun
        partner_offer = state.current_offer
        if partner_offer is None:
            return False

        offer_utility_for_me = float(self.ufun(partner_offer))
        relative_time = state.relative_time
        max_utility_for_me = float(self.ufun.max())
        reserved = float(self.ufun.reserved_value)

        if relative_time >= self.ACCEPT_CATASTROPHE_TIME and offer_utility_for_me >= reserved - 1e-9:
            return True

        slope = self._aspiration_time_slope(relative_time)
        aspiration_utility = max_utility_for_me * (1.0 - slope * relative_time)
        accepted = (
            offer_utility_for_me >= aspiration_utility
            and offer_utility_for_me > reserved
        )

        if not accepted:
            our_next_bid = self.concealing_bidding_strategy(state)
            if our_next_bid is not None:
                if offer_utility_for_me >= float(self.ufun(our_next_bid)) - 1e-9:
                    accepted = True

        if (
            not accepted
            and relative_time > self.ACCEPT_DEADLINE_SAFE
            and offer_utility_for_me > reserved * self.ACCEPT_LATE_RV_FACTOR
        ):
            accepted = True

        if not accepted:
            return False
        if relative_time > self.ACCEPT_DEADLINE_SAFE:
            return True
        if self._partner_offer_looks_like_bait(state):
            return False
        return True

    def _partner_offer_looks_like_bait(self, state: SAOState) -> bool:
        partner_offer = state.current_offer
        if partner_offer is None or self.opponent_frequency_model is None:
            return False
        if self._opponent_mirrors_us():
            return False
        if self._opponent_mode() != "deceptive":
            return False
        if not self._opponent_shows_concealment_tactics():
            return False
        if self._opponent_smith_learner_profile():
            return False
        traj = self.offer_trajectory_model
        if traj is None or traj.sample_count() < self.MIN_TRAJECTORY_SAMPLES:
            return False
        if traj.concession_slope() < self.HONEST_CONCESSION_SLOPE:
            return False

        smith_u = self.opponent_frequency_model.estimated_opponent_utility(
            partner_offer
        )
        predicted = traj.predicted_utility_at(state.relative_time)
        return smith_u > predicted + self.ACCEPT_BAIT_THRESHOLD

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        self._eval_relative_time = state.relative_time
        bid = super().concealing_bidding_strategy(state)
        if bid is None:
            if not self.rational_outcomes:
                return None
            bid = self.rational_outcomes[0]
        if self._opponent_mirrors_us():
            opp_offers = self._opponent_recent_offers or []
            if opp_offers and bid == opp_offers[-1]:
                rng = random.Random(hash((self.id, state.step, "mirror")) & 0xFFFFFFFF)
                pool = self._anti_mirror_pool(list(self.rational_outcomes[:40]))
                if pool:
                    bid = rng.choice(pool)
        return bid

    def current_bluff_score(self) -> float:
        if self._opponent_mirrors_us():
            return 0.0
        traj = self.offer_trajectory_model
        if traj is None or traj.sample_count() < self.MIN_TRAJECTORY_SAMPLES:
            return 0.0
        slope = traj.concession_slope()
        if slope < self.HONEST_CONCESSION_SLOPE:
            return 0.0
        last = traj.last_sample()
        if last is None:
            return 0.0
        last_t, last_u = last
        return min(1.0, traj.inconsistency_vs_trajectory(last_u, last_t) * 1.75)

    def _should_apply_bait_discount(self) -> bool:
        if self._opponent_mirrors_us():
            return False
        if not self._closing_bluff_active():
            return False
        if self._opponent_mode() != "deceptive":
            return False
        if not self._opponent_late_bait_switch():
            return False
        if self._opponent_smith_learner_profile():
            return False
        traj = self.offer_trajectory_model
        if traj is None or traj.sample_count() < self.BAIT_MIN_TRAJECTORY_SAMPLES:
            return False
        return True

    def _offer_looks_like_bait(self, smith_u: float, relative_time: float) -> bool:
        traj = self.offer_trajectory_model
        if traj is None or not self._closing_bluff_active(relative_time):
            return False
        if traj.concession_slope() < self.HONEST_CONCESSION_SLOPE:
            return False
        predicted = traj.predicted_utility_at(relative_time)
        return smith_u > predicted + self.BAIT_THRESHOLD

    def estimated_opponent_utility(self, offer: Outcome) -> float:
        base_u = self._blended_opponent_utility(offer)
        if not self._should_apply_bait_discount():
            return base_u

        traj = self.offer_trajectory_model
        assert traj is not None

        relative_time = self._eval_relative_time
        predicted = traj.predicted_utility_at(relative_time)
        if not self._offer_looks_like_bait(base_u, relative_time):
            return base_u

        adjusted = base_u
        excess = base_u - predicted - self.BAIT_THRESHOLD
        adjusted = predicted + self.BAIT_THRESHOLD + excess * (1.0 - self.BAIT_DISCOUNT)

        inconsistency = abs(base_u - predicted)
        if inconsistency > self.INCONSISTENCY_BLEND_THRESHOLD:
            adjusted = (
                (1.0 - self.INCONSISTENCY_BLEND) * adjusted
                + self.INCONSISTENCY_BLEND * predicted
            )

        return max(0.0, min(1.0, adjusted))
