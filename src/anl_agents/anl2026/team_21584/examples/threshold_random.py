from __future__ import annotations

import random
from collections import defaultdict

from negmas.gb.components.models.ufun import UFunModel
from negmas.outcomes import Outcome
from negmas.sao.common import ResponseType, SAOState
from negmas.sao.negotiators.base import SAOPRNegotiator

from ..util import enumerate_outcomes

__all__ = ["ThresholdRandomInferenceAgent"]


class AdaptiveIssueValueModel(UFunModel):
    """Opponent model initialized at 0.5 for every issue value."""

    def __init__(self, *, reward_rate: float = 0.025, penalty_rate: float = 0.015):
        super().__init__()
        self._issue_value_scores: dict[int, dict[object, float]] = defaultdict(dict)
        self._reward_rate = reward_rate
        self._penalty_rate = penalty_rate

    def update_from_opponent_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        if offer is None or max_steps <= 0:
            return
        relative_time = max(0.0, min(1.0, step / max_steps))
        increment = self._reward_rate * (1.0 - relative_time)
        for issue_index, value in enumerate(offer):
            self._adjust(issue_index, value, increment)

    def penalize_rejected_self_offer(
        self, offer: Outcome | None, step: int, max_steps: int
    ) -> None:
        if offer is None or max_steps <= 0:
            return
        relative_time = max(0.0, min(1.0, step / max_steps))
        decrement = self._penalty_rate * (0.5 + 0.5 * relative_time)
        for issue_index, value in enumerate(offer):
            self._adjust(issue_index, value, -decrement)

    def eval(self, offer: Outcome) -> float:
        if offer is None:
            return 0.5
        if not offer:
            return 0.5
        total = 0.0
        for issue_index, value in enumerate(offer):
            total += self._issue_value_scores[issue_index].get(value, 0.5)
        return total / len(offer)

    def eval_normalized(self, offer: Outcome) -> float:
        return self.eval(offer)

    def __call__(self, offer: Outcome) -> float:
        return self.eval(offer)

    def _adjust(self, issue_index: int, value: object, delta: float) -> None:
        current = self._issue_value_scores[issue_index].get(value, 0.5)
        self._issue_value_scores[issue_index][value] = max(
            0.0, min(1.0, current + delta)
        )


class ThresholdRandomInferenceAgent(SAOPRNegotiator):
    """Threshold accepter with random unused offers above the current threshold."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = AdaptiveIssueValueModel()
        self.private_info["opponent_ufun"] = self._model
        self._outcomes: list[Outcome] = []
        self._used_in_cycle: set[Outcome] = set()
        self._made_first_offer = False
        self._last_self_offer: Outcome | None = None
        self._last_self_step = -1
        self._last_self_offer_pending = False
        self._last_opponent_offer: Outcome | None = None
        self._opponent_offer_count = 0
        self._same_issue_total = 0
        self._changed_issue_total = 0
        self._sequential_signal = 0.0
        self._distributed_signal = 0.0

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._model.set_negotiator(self)
        self._model.on_preferences_changed(changes)
        self.private_info["opponent_ufun"] = self._model
        self._outcomes = []
        if self.nmi is not None and self.ufun is not None:
            self._outcomes = enumerate_outcomes(
                self.nmi.outcome_space, max_outcomes=100000
            )
        self._used_in_cycle = set()
        self._made_first_offer = False
        self._last_self_offer = None
        self._last_self_step = -1
        self._last_self_offer_pending = False
        self._last_opponent_offer = None
        self._opponent_offer_count = 0
        self._same_issue_total = 0
        self._changed_issue_total = 0
        self._sequential_signal = 0.0
        self._distributed_signal = 0.0
        self._publish_private_info("unknown")

    def propose(self, state: SAOState, dest: str | None = None):
        _ = dest
        if self.ufun is None or not self._outcomes:
            return None
        threshold = self._threshold(state)
        candidates = self._candidates_above_threshold(threshold)
        if not candidates:
            offer = max(self._outcomes, key=lambda outcome: float(self.ufun(outcome)))
        elif not self._made_first_offer:
            offer = min(candidates, key=lambda outcome: float(self.ufun(outcome)))
        else:
            unused = [
                outcome for outcome in candidates if outcome not in self._used_in_cycle
            ]
            if not unused:
                self._used_in_cycle = set()
                unused = list(candidates)
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

        max_steps = self._max_steps()
        if self._last_self_offer_pending:
            self._model.penalize_rejected_self_offer(
                self._last_self_offer, self._last_self_step, max_steps
            )
            self._last_self_offer_pending = False

        self._model.update_from_opponent_offer(offer, int(state.step), max_steps)
        pattern = self._update_offer_pattern(offer)
        self._publish_private_info(pattern)

        offered_utility = float(self.ufun(offer))
        if offered_utility >= self._threshold(state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _threshold(self, state: SAOState) -> float:
        relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        if relative_time <= 0.0 and self.nmi is not None and self.nmi.n_steps:
            relative_time = int(state.step) / max(1, int(self.nmi.n_steps) - 1)
        relative_time = max(0.0, min(1.0, relative_time))
        return 0.7 - 0.1 * relative_time

    def _candidates_above_threshold(self, threshold: float) -> list[Outcome]:
        if self.ufun is None:
            return []
        return [
            outcome
            for outcome in self._outcomes
            if float(self.ufun(outcome)) >= threshold
        ]

    def _max_steps(self) -> int:
        if self.nmi is not None and self.nmi.n_steps:
            return max(1, int(self.nmi.n_steps))
        return 1

    def _update_offer_pattern(self, offer: Outcome) -> str:
        self._opponent_offer_count += 1
        if self._last_opponent_offer is None:
            self._last_opponent_offer = offer
            return "unknown"

        pairs = zip(self._last_opponent_offer, offer)
        same_count = sum(1 for previous, current in pairs if previous == current)
        issue_count = max(1, len(offer))
        changed_count = issue_count - same_count
        self._same_issue_total += same_count
        self._changed_issue_total += changed_count

        same_ratio = same_count / issue_count
        changed_ratio = changed_count / issue_count
        if same_ratio >= 2.0 / 3.0:
            self._sequential_signal += 1.0
        if changed_ratio >= 2.0 / 3.0:
            self._distributed_signal += 1.0

        self._last_opponent_offer = offer
        if self._sequential_signal > self._distributed_signal:
            return "sequential"
        if self._distributed_signal > self._sequential_signal:
            return "distributed"
        return "mixed"

    def _publish_private_info(self, pattern: str) -> None:
        comparisons = max(0, self._opponent_offer_count - 1)
        issue_observations = max(1, self._same_issue_total + self._changed_issue_total)
        changed_ratio = self._changed_issue_total / issue_observations
        same_ratio = self._same_issue_total / issue_observations
        self.private_info["opponent_offer_pattern"] = pattern
        self.private_info["opponent_strategy_analysis"] = {
            "opponent_offer_count": self._opponent_offer_count,
            "offer_comparisons": comparisons,
            "same_issue_ratio": same_ratio,
            "changed_issue_ratio": changed_ratio,
            "sequential_signal": self._sequential_signal,
            "distributed_signal": self._distributed_signal,
            "inference_model": "adaptive_all_0_5_issue_value_model",
        }
        self.private_info["active_opponent_model"] = (
            "AdaptiveIssueValueModel:" + pattern
        )
        self.private_info["evaluation_opponent_model"] = "AdaptiveIssueValueModel"
