from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional

import numpy as np
from negmas import Outcome, ResponseType, SAOState
from negmas.sao import SAONegotiator


class FreqModel:
    """
    Enhanced frequency-based opponent model with entropy-weighted issue importance,
    behavior classification, and dynamic concession exponent estimation.
    """

    def __init__(self, issues):
        self.issues = issues
        self.n = len(issues)
        self.value_counts: list[defaultdict] = [defaultdict(int) for _ in issues]
        self.total_bids = 0
        self.issue_weights: list[float] = [1.0 / max(self.n, 1)] * self.n
        self.value_weights: list[dict] = [
            {v: 1.0 / max(issue.cardinality, 1) for v in iter(issue)}
            for issue in issues
        ]
        self.utility_history: list[float] = []
        self.time_history: list[float] = []

    def update(self, bid: Outcome, t: float = 0.0) -> None:
        if bid is None:
            return
        self.total_bids += 1
        for i, value in enumerate(bid):
            self.value_counts[i][value] += 1
        self._refresh_value_weights()
        if self.total_bids >= 3:
            self._refresh_issue_weights()
        self.utility_history.append(self.estimate_utility(bid))
        self.time_history.append(t)

    def estimate_utility(self, bid: Outcome) -> float:
        if bid is None:
            return 0.0
        return sum(
            self.issue_weights[i] * self.value_weights[i].get(v, 0.0)
            for i, v in enumerate(bid)
        )

    def classify_behavior(self) -> str:
        if len(self.utility_history) < 5:
            return "unknown"
        diffs = np.diff(self.utility_history)
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs))
        if mean_d < -0.015:
            return "conceder"
        if abs(mean_d) < 0.008 and std_d < 0.02:
            return "boulware"
        if std_d > 0.05:
            return "random"
        return "selfish"

    def estimate_concession_exponent(self) -> float:
        """
        Estimate the opponent's Boulware concession exponent from their bid history.

        Uses the Faratin model: u(t) = u_min + span*(1 - t^(1/e)).
        Solves for e at each observed (t, u) pair and returns the median.
        Returns 1.0 (linear) when fewer than 6 observations are available.
        Returns 0.1 (very Boulware) when utility range is essentially flat.
        """
        if len(self.utility_history) < 6:
            return 1.0
        utils = self.utility_history
        times = self.time_history
        u_max = max(utils)
        u_min = min(utils)
        span = u_max - u_min
        if span < 0.02:
            return 0.1  # flat trajectory → very Boulware
        estimates: list[float] = []
        for t_val, u_val in zip(times, utils):
            if t_val < 0.05 or t_val > 0.95:
                continue
            # Normalised utility: 1 = early/high (best for opponent), 0 = late/low
            r = (u_val - u_min) / span
            if r < 0.05 or r > 0.95:
                continue
            try:
                # r = 1 - t^(1/e)  →  t^(1/e) = 1 - r  →  e = log(t)/log(1-r)
                log_t = math.log(t_val)
                log_1r = math.log(1.0 - r)
                if log_t < -1e-4 and log_1r < -1e-4:
                    e_est = log_t / log_1r
                    if 0.05 < e_est < 10.0:
                        estimates.append(e_est)
            except (ValueError, ZeroDivisionError):
                continue
        if not estimates:
            return 1.0
        return float(np.median(estimates))

    def _refresh_value_weights(self) -> None:
        for i in range(self.n):
            total = sum(self.value_counts[i].values()) or 1
            for v in self.value_weights[i]:
                self.value_weights[i][v] = self.value_counts[i].get(v, 0) / total

    def _refresh_issue_weights(self) -> None:
        entropies = [
            -sum(w * math.log(w + 1e-12) for w in self.value_weights[i].values())
            for i in range(self.n)
        ]
        max_e = max(entropies) + 1e-10
        raw = [max_e - e for e in entropies]
        total = sum(raw) + 1e-10
        self.issue_weights = [r / total for r in raw]

    def __call__(self, bid: Outcome) -> float:
        return self.estimate_utility(bid)


class EmEfAgent(SAONegotiator):
    """
    EmEfAgent for ANAC 2026 ANL.

    BOA architecture with four key capabilities:
    1. Enhanced entropy-weighted frequency opponent model.
    2. Adaptive Boulware bidding with dynamic concession-rate estimation.
    3. ACNext + Nash-point + time-panic acceptance strategy.
    4. Focused issue-targeted deception for preference concealment.
    """

    def __init__(
        self,
        *args,
        e: float = 0.20,
        deception_freq: float = 0.25,
        min_utility: float = 0.52,
        nash_radius: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._e_base = e
        self._e = e
        self._deception_freq = deception_freq
        self._min_utility = min_utility
        self._nash_radius = nash_radius

        self._opp_model: Optional[FreqModel] = None
        self._sorted_outcomes: list[Outcome] = []
        self._effective_min: float = min_utility

        self._deception_issue_idx: int = 0
        self._deception_rounds: int = 0

        # Nash bargaining cache
        self._nash_product: float = 0.0
        self._last_nash_update_bids: int = -10

        self._step_count: int = 0

    @property
    def opponent_ufun(self):
        """Required by ANL 2026 scoring infrastructure to compute Kendall correlation."""
        return self._opp_model

    # ------------------------------------------------------------------ setup

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        issues = self.nmi.issues
        self._opp_model = FreqModel(issues)
        reserved = float(self.ufun.reserved_value or 0.0) if self.ufun else 0.0
        self._effective_min = max(self._min_utility, reserved + 0.15)

        all_outcomes = list(self.nmi.discrete_outcomes())
        if self.ufun is not None:
            self._sorted_outcomes = sorted(
                all_outcomes,
                key=lambda o: float(self.ufun(o)),
                reverse=True,
            )
        else:
            self._sorted_outcomes = all_outcomes

        self._e = self._e_base
        self._deception_rounds = 0
        self._nash_product = 0.0
        self._last_nash_update_bids = -10
        self._step_count = 0

        my_weights = self._my_weights()
        n = len(issues)
        self._deception_issue_idx = (
            max(range(n), key=lambda i: my_weights[i]) if n > 0 else 0
        )

    # --------------------------------------------------------- NegMAS interface

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        offer = state.current_offer
        t = self._time(state)
        if offer is not None and self._opp_model is not None:
            self._opp_model.update(offer, t)
        if offer is not None and self._accept(offer, state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        _ = dest
        self._step_count += 1
        self._update_concession_rate()
        return self._bid(state)

    # -------------------------------------------------------- acceptance logic

    def _accept(self, offer: Outcome, state: SAOState) -> bool:
        if self.ufun is None:
            return False
        offer_u = float(self.ufun(offer))
        reserved = float(self.ufun.reserved_value or 0.0)

        # Hard floor
        if offer_u < reserved:
            return False

        target = self._target_adapted(state)

        # Rule 1 – ACNext: accept if at least as good as what we would propose
        if offer_u >= target:
            return True

        # Rule 2 – Nash-point acceptance: only late in game, within 10 % of target
        # The opponent model is unreliable early; never accept below 90 % of target.
        if self._nash_accept(offer, offer_u, reserved, state, target):
            return True

        # Rule 3 – Time panic: avoid no-deal near deadline
        panic_floor = max(reserved + 0.05, self._effective_min * 0.85)
        if self._time(state) > 0.90 and offer_u >= panic_floor:
            return True

        return False

    def _nash_accept(
        self,
        offer: Outcome,
        offer_u: float,
        reserved: float,
        state: SAOState,
        acnext_target: float,
    ) -> bool:
        """
        Accept if the offer is near the estimated Nash bargaining solution, but only
        after t > 0.70 and never below 90 % of the current ACNext target.

        This prevents accepting bad agreements when the opponent model is still
        inaccurate early in the negotiation.
        """
        t = self._time(state)
        if t < 0.70:
            return False
        if self._opp_model is None or self._opp_model.total_bids < 10:
            return False
        # Never widen acceptance more than 10 % below current ACNext target
        min_floor = max(reserved + 0.05, acnext_target * 0.90)
        if offer_u < min_floor:
            return False
        self._refresh_nash_product()
        if self._nash_product <= 0:
            return False
        offer_nash = offer_u * float(self._opp_model(offer))
        return offer_nash >= (1.0 - self._nash_radius) * self._nash_product

    def _refresh_nash_product(self) -> None:
        """Update the cached Nash bargaining product every 5 opponent bids."""
        if self._opp_model is None or self._opp_model.total_bids < 5:
            return
        bids = self._opp_model.total_bids
        if bids - self._last_nash_update_bids < 5:
            return
        self._last_nash_update_bids = bids
        if not self._sorted_outcomes or self.ufun is None:
            return
        reserved = float(self.ufun.reserved_value or 0.0)
        best = 0.0
        limit = min(300, len(self._sorted_outcomes))
        for o in self._sorted_outcomes[:limit]:
            own_u = float(self.ufun(o))
            if own_u < reserved:
                break  # sorted descending; anything further is worse
            opp_u = float(self._opp_model(o))
            product = own_u * opp_u
            if product > best:
                best = product
        self._nash_product = best

    # ------------------------------------------------ dynamic concession rate

    def _update_concession_rate(self) -> None:
        """
        Adjust the Boulware exponent e based on the opponent's estimated style.

        Changes are kept small (≤ 10 % of base) and only applied after 15 bids
        so the model has time to stabilise.  The adjustment ensures our concession
        speed loosely tracks the opponent:

        - Very Boulware opponent (e_opp < 0.15): tiny increase so we make some
          progress near the deadline and avoid a mutual no-deal.
        - Fast conceder (e_opp > 3.0): tiny decrease — they are coming to us,
          so we can stay slightly more patient.
        - Otherwise: revert to base e.
        """
        if self._opp_model is None or self._opp_model.total_bids < 15:
            return
        e_opp = self._opp_model.estimate_concession_exponent()
        if e_opp < 0.15:
            self._e = min(self._e_base * 1.10, self._e_base + 0.02)
        elif e_opp > 3.0:
            self._e = max(self._e_base * 0.95, self._e_base - 0.01)
        else:
            self._e = self._e_base

    # ----------------------------------------------------------- bidding logic

    def _bid(self, state: SAOState) -> Optional[Outcome]:
        target = self._target_adapted(state)
        reserved = float(self.ufun.reserved_value or 0.0) if self.ufun else 0.0
        target = max(target, reserved + 0.01, self._effective_min)
        if self._should_deceive(state):
            bid = self._deception_bid(target)
            if bid is not None:
                return bid
        return self._honest_bid(target)

    def _target_adapted(self, state: SAOState) -> float:
        return self._adapt(self._target(state), state)

    def _target(self, state: SAOState) -> float:
        t = self._time(state)
        u_max = 1.0
        u_min = max(
            (float(self.ufun.reserved_value or 0.0) + 0.01) if self.ufun else 0.01,
            self._effective_min,
        )
        target = u_min + (u_max - u_min) * (1.0 - t ** (1.0 / self._e))
        return float(np.clip(target, u_min, u_max))

    def _adapt(self, target: float, state: SAOState) -> float:
        if self._opp_model is None:
            return target
        behavior = self._opp_model.classify_behavior()
        if behavior == "conceder":
            target = min(target + 0.05, 0.95)
        elif behavior == "boulware" and self._time(state) > 0.70:
            target = max(target - 0.04, self._effective_min)
        return target

    def _honest_bid(self, target: float) -> Optional[Outcome]:
        if not self._sorted_outcomes:
            return None
        for outcome in self._sorted_outcomes:
            if float(self.ufun(outcome)) >= target:
                return outcome
        # Fallback: best outcome at or above reserved value
        reserved = float(self.ufun.reserved_value or 0.0) if self.ufun else 0.0
        for outcome in self._sorted_outcomes:
            if float(self.ufun(outcome)) >= reserved:
                return outcome
        return self._sorted_outcomes[0]

    def _should_deceive(self, state: SAOState) -> bool:
        t = self._time(state)
        return (0.10 < t < 0.88) and (random.random() < self._deception_freq)

    # ------------------------------------------------------- deception strategy

    def _deception_bid(self, min_utility: float) -> Optional[Outcome]:
        """
        Select a bid that maximally confuses the opponent's frequency model.

        Strategy: always show the WORST value on our most important issue while
        keeping own utility at or above min_utility.  This creates a concentrated
        frequency signal on the worst value of the highest-weight issue, causing
        the opponent's model to rank our most-preferred outcomes as least-preferred
        — maximum Kendall anti-correlation with our true utility function.
        """
        if not self._sorted_outcomes or self.ufun is None:
            return None
        self._deception_rounds += 1
        acceptable = [
            o for o in self._sorted_outcomes
            if float(self.ufun(o)) >= min_utility
        ]
        if not acceptable:
            return None

        my_weights = self._my_weights()
        target_issue = max(range(len(my_weights)), key=lambda i: my_weights[i])

        # Deterministically find the worst value for us on the target issue
        try:
            all_vals = list(iter(self.nmi.issues[target_issue]))
            ranked_asc = sorted(
                all_vals,
                key=lambda v: float(self.ufun.values[target_issue](v)),
            )
            worst_val = ranked_asc[0]
            # Among acceptable bids, prefer those using the worst value on target issue.
            # self._sorted_outcomes is descending by utility, so first match = best utility.
            for o in acceptable[:300]:
                if o[target_issue] == worst_val:
                    return o
        except (AttributeError, IndexError, TypeError):
            pass

        # Fallback: sample-based scoring
        sample = random.sample(acceptable, min(80, len(acceptable)))
        best_bid = max(
            sample,
            key=lambda b: self._deception_score_targeted(b, my_weights, target_issue),
        )
        return best_bid

    def _deception_score_targeted(
        self,
        bid: Outcome,
        my_weights: list[float],
        target_issue: int,
    ) -> float:
        """
        Score bid by how much it sacrifices the target issue.

        The target issue receives strong amplification; all other issues receive
        only a small weight. The final score is high when the bid gives a poor
        value on the target issue.
        """
        issues = self.nmi.issues
        score = 0.0
        for i, (issue, w) in enumerate(zip(issues, my_weights)):
            v = bid[i]
            try:
                all_v = list(iter(issue))
                ranked = sorted(all_v, key=lambda x: float(self.ufun.values[i](x)))
                idx = ranked.index(v)
                norm_rank = idx / max(len(ranked) - 1, 1)
            except (ValueError, AttributeError, TypeError):
                norm_rank = 0.5
            sacrifice = 1.0 - norm_rank   # 1 = worst value for us on this issue
            multiplier = 5.0 if i == target_issue else 0.1
            score += multiplier * w * sacrifice
        return score

    # ---------------------------------------------------------------- helpers

    def _my_weights(self) -> list[float]:
        try:
            raw = list(self.ufun.weights)
            total = sum(raw) + 1e-10
            return [w / total for w in raw]
        except Exception:
            n = len(self.nmi.issues)
            return [1.0 / n] * n

    def _time(self, state: SAOState) -> float:
        n = self.nmi.n_steps
        if n and n > 0:
            return min(float(state.step) / float(n), 1.0)
        return 0.5
