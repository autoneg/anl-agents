"""
CunningMerchant — A competitive agent for ANL 2026 (Automated Negotiation League).

ANL 2026 is a bilateral negotiation with a DECEPTION challenge. The agent's final
score has two components:
    Score = Advantage + Concealing
where:
    Advantage = (utility_of_agreement - reservation_value) / (max_utility - reservation_value)
    Concealing = how well we MISLEAD the opponent's model of our utility function
                 (measured by Kendall rank correlation — lower correlation = higher concealing score)

Strategy Overview:
    1. Aggressive Anchoring (Opening Move): Holds ground firmly with an elevated threshold 
       during the initial 15% of the negotiation to probe opponent concession rates.
    2. Aspiration-based Concession: Follows a strict time-dependent Boulware curve (e=4.0) 
       to project a tough, unyielding posture throughout the mid-game.
    3. Haggling Fluctuation (Zig-zag Strategy): Injects controlled, high-frequency utility 
       vibrations into bids to disrupt and dismantle the opponent's curve-fitting or 
       linear regression models.
    4. Active Preference Inversion (Throw a curve): Misleads the opponent by offering suboptimal 
       values on our highly critical issues while maximizing utility on irrelevant issues, 
       effectively flipping our apparent preference profile.
    5. Opponent Frequency Modeling & Trend Tracking: Utilizes frequency analysis for 
       issue estimation and real-time linear regression to monitor opponent pacing.
    6. Adaptive Last-Minute Compromise (Merchant Panic): Dynamically collapses acceptance 
       thresholds in the final 2.5% of rounds to guarantee a deal and completely eliminate 
       timeout risks.

Authors: Umut Murat (umutmurat275@gmail.com)
         Kaan Gönenli (kaan.gonenli@ozu.edu.tr)
Institution: Özyeğin University
Course: CS451 Project — ANAC 2026
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from negmas.outcomes import Outcome
from negmas.preferences import LinearMultiFun
from negmas.sao import SAOCallNegotiator, SAOResponse, SAOState, ResponseType


class CunningMerchant(SAOCallNegotiator):
    """
    CunningMerchant for ANL 2026 — bilateral negotiation with deception.

    The agent combines a strong negotiation strategy (aspiration concession,
    opponent modeling) with a deception layer that strategically chooses bids
    to confuse the opponent's model of our preferences.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Configurable parameters
    # ──────────────────────────────────────────────────────────────────────

    # Aspiration exponent: >1 = boulware (holds out), <1 = conceder
    ASPIRATION_EXPONENT: float = 4.0 

    # What fraction of the time we use concealing (deceptive) bids vs
    # honest (utility-maximizing) bids. Higher = more deceptive but riskier.
    CONCEAL_RATIO: float = 0.45 
    
    # Haggling volatility: maximum percentage of utility fluctuation to confuse opponent models
    HAGGLING_VOLATILITY: float = 0.03 

    # Opponent model learning rate — how fast we update issue weights
    OPPONENT_LEARNING_RATE: float = 0.1

    # Number of recent opponent offers for behavior trend estimation
    OPPONENT_WINDOW_SIZE: int = 5

    # Weight of opponent trend in threshold adjustment
    OPPONENT_TREND_WEIGHT: float = 0.15

    def on_preferences_changed(self, changes):
        """
        Called when preferences are set. This is our initialization point.
        """
        if self.ufun is None:
            return

        # ── Build sorted list of rational outcomes ──
        ufun_outcome = [
            (float(self.ufun(outcome)), outcome)
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if float(self.ufun(outcome)) > float(self.ufun.reserved_value)
        ]
        ufun_outcome.sort(reverse=True)

        self._rational_outcomes = tuple(item[1] for item in ufun_outcome)
        self._rational_utilities = tuple(item[0] for item in ufun_outcome)

        if self._rational_utilities:
            self._max_utility = self._rational_utilities[0]
            self._min_utility = float(self.ufun.reserved_value)
        else:
            self._max_utility = 0.0
            self._min_utility = 0.0

        # ── Initialize opponent preference model ──
        n_issues = len(self.nmi.outcome_space.issues)
        self._opponent_issue_weights = [1.0 / n_issues] * n_issues
        self._opponent_value_counts: dict[int, dict] = {}
        for i, issue in enumerate(self.nmi.outcome_space.issues):
            self._opponent_value_counts[i] = defaultdict(int)

        self._build_opponent_ufun()

        # ── Opponent behavior tracking ──
        self._opponent_offer_history: list[tuple[int, float]] = []

        # ── Deception: pre-compute issue rankings for concealing bids ──
        self._compute_issue_importance()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Main entry point — called each turn to produce an offer or accept.
        """
        offer = state.current_offer

        if self.ufun is None or not self._rational_outcomes:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is None:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self._generate_bid(state)
            )

        self._update_opponent_model(state)

        if self._should_accept(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(
            ResponseType.REJECT_OFFER, self._generate_bid(state)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Acceptance strategy
    # ──────────────────────────────────────────────────────────────────────

    def _should_accept(self, state: SAOState) -> bool:
        """
        Decide whether to accept the opponent's current offer.
        """
        offer = state.current_offer
        if offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        if offer_utility <= float(self.ufun.reserved_value):
            return False
            
        # --- HARD FLOOR ---
        # Never accept an offer lower than 65% of max utility during the first 85% of time.
        t = state.relative_time
        hard_floor = self._min_utility + 0.65 * (self._max_utility - self._min_utility)
        if t < 0.85 and offer_utility < hard_floor:
            return False
            
        threshold = self._calc_threshold(state)

        trend = self._estimate_opponent_trend()
        adjustment = trend * self.OPPONENT_TREND_WEIGHT * (self._max_utility - self._min_utility)
        threshold = max(self._min_utility, min(self._max_utility, threshold + adjustment))

        # --- MERCHANT PANIC MODE (Endgame Capitulation) ---
        if state.relative_time > 0.975:
            panic_floor = self._min_utility + 0.15 * (self._max_utility - self._min_utility)
            threshold = min(threshold, panic_floor)
            
        return offer_utility >= threshold

    def _calc_threshold(self, state: SAOState) -> float:
        """
        Calculate the aspiration-based acceptance threshold.
        """
        t = state.relative_time

        if t <= 0.15:
            return self._max_utility - 0.02 * (self._max_utility - self._min_utility)
        elif t >= 1.0:
            level = 0.0
        else:
            level = 1.0 - math.pow(t, self.ASPIRATION_EXPONENT)

        threshold = self._min_utility + level * (self._max_utility - self._min_utility)
        
        # --- HAGGLING FLUCTUATION (Zig-zag) ---
        if 0.15 <= t <= 0.90:
            vibration = random.uniform(-self.HAGGLING_VOLATILITY, self.HAGGLING_VOLATILITY)
            vibration_utility = vibration * (self._max_utility - self._min_utility)
            threshold = max(self._min_utility, min(self._max_utility, threshold + vibration_utility))
            
        return threshold

    # ──────────────────────────────────────────────────────────────────────
    # Bidding strategy (with preference inversion)
    # ──────────────────────────────────────────────────────────────────────

    def _generate_bid(self, state: SAOState) -> Outcome | None:
        """
        Generate a bid — either an honest aspiration-based bid or a concealing bid.
        """
        if not self._rational_outcomes:
            return None

        target = self._calc_threshold(state)

        margin = 0.04 * (self._max_utility - self._min_utility)
        candidates = [
            (i, outcome)
            for i, (outcome, util) in enumerate(
                zip(self._rational_outcomes, self._rational_utilities)
            )
            if target - margin <= util <= target + margin
        ]

        if not candidates:
            candidates = [
                (i, outcome)
                for i, (outcome, util) in enumerate(
                    zip(self._rational_outcomes, self._rational_utilities)
                )
                if util >= target - 2 * margin
            ]

        if not candidates:
            n = min(10, len(self._rational_outcomes))
            candidates = [(i, self._rational_outcomes[i]) for i in range(n)]

        if state.relative_time > 0.92:
            reveal_prob = 0.0
        else:
            reveal_prob = self.CONCEAL_RATIO * (1.0 - state.relative_time)

        if random.random() < reveal_prob and len(candidates) > 1:
            return self._pick_concealing_bid(candidates)
        else:
            candidates.sort(key=lambda x: self._rational_utilities[x[0]], reverse=True)
            return candidates[0][1]

    def _pick_concealing_bid(self, candidates: list[tuple[int, Outcome]]) -> Outcome:
        """
        Pick the candidate that best misleads the opponent about our true preferences.
        """
        if not hasattr(self, '_issue_importance_rank'):
            return candidates[0][1]

        best_score = -float('inf')
        best_outcome = candidates[0][1]

        for _, outcome in candidates:
            inversion_score = 0.0
            for issue_idx, importance in enumerate(self._issue_importance_rank):
                if issue_idx < len(outcome):
                    value = outcome[issue_idx]
                    is_best = (value == self._best_values.get(issue_idx))
                    if not is_best:
                        inversion_score += importance * 3.0  
                    else:
                        inversion_score -= importance * 2.0
            if inversion_score > best_score:
                best_score = inversion_score
                best_outcome = outcome

        return best_outcome

    def _compute_issue_importance(self):
        """
        Pre-compute which issues matter most to us and our best value per issue.
        """
        if self.ufun is None:
            return

        issues = self.nmi.outcome_space.issues
        n_issues = len(issues)

        importance = []
        self._best_values = {}

        for i, issue in enumerate(issues):
            values = list(issue.all)
            if not values:
                importance.append(0.0)
                continue

            base_outcome = list(self._rational_outcomes[0]) if self._rational_outcomes else [v for iss in issues for v in [list(iss.all)[0]]]

            best_val = values[0]
            best_util = -float('inf')
            utils = []

            for val in values:
                test = list(base_outcome)
                test[i] = val
                u = float(self.ufun(tuple(test)))
                utils.append(u)
                if u > best_util:
                    best_util = u
                    best_val = val

            self._best_values[i] = best_val

            if utils:
                importance.append(max(utils) - min(utils))
            else:
                importance.append(0.0)

        total = sum(importance) if importance else 1.0
        if total > 0:
            self._issue_importance_rank = [imp / total for imp in importance]
        else:
            self._issue_importance_rank = [1.0 / n_issues] * n_issues

    # ──────────────────────────────────────────────────────────────────────
    # Opponent preference modeling (frequency-based)
    # ──────────────────────────────────────────────────────────────────────

    def _update_opponent_model(self, state: SAOState) -> None:
        """
        Update our model of the opponent's preferences based on their latest offer.
        """
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return

        self._opponent_offer_history.append(
            (state.step, float(self.ufun(offer)))
        )

        for i in range(len(offer)):
            self._opponent_value_counts[i][offer[i]] += 1

        total_offers = state.step + 1
        issue_consistencies = []

        for i in range(len(self.nmi.outcome_space.issues)):
            counts = self._opponent_value_counts[i]
            if not counts:
                issue_consistencies.append(0.0)
                continue
            max_count = max(counts.values())
            consistency = max_count / total_offers
            issue_consistencies.append(consistency)

        total_consistency = sum(issue_consistencies)
        if total_consistency > 0:
            new_weights = [c / total_consistency for c in issue_consistencies]
        else:
            n = len(self.nmi.outcome_space.issues)
            new_weights = [1.0 / n] * n

        alpha = self.OPPONENT_LEARNING_RATE
        self._opponent_issue_weights = [
            alpha * new_w + (1 - alpha) * old_w
            for new_w, old_w in zip(new_weights, self._opponent_issue_weights)
        ]

        self._build_opponent_ufun()

    def _build_opponent_ufun(self):
        """
        Build/rebuild the opponent's estimated utility function.
        """
        issues = self.nmi.outcome_space.issues
        n_issues = len(issues)

        value_utils = []
        for i, issue in enumerate(issues):
            values = list(issue.all)
            counts = self._opponent_value_counts[i]
            total = sum(counts.values()) if counts else 1

            if total > 0 and counts:
                max_count = max(counts.values()) if counts else 1
                utils = {}
                for val in values:
                    count = counts.get(val, 0)
                    utils[val] = count / max_count if max_count > 0 else 0.5
            else:
                utils = {val: 0.5 for val in values}

            value_utils.append(utils)

        weights = list(self._opponent_issue_weights)
        vutils = list(value_utils)

        def opponent_eval(outcome):
            if outcome is None:
                return 0.0
            total = 0.0
            for i in range(min(len(outcome), n_issues)):
                w = weights[i] if i < len(weights) else 0.0
                vu = vutils[i] if i < len(vutils) else {}
                total += w * vu.get(outcome[i], 0.5)
            return total

        from negmas.preferences import LambdaMultiFun
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=opponent_eval)

    # ──────────────────────────────────────────────────────────────────────
    # Opponent behavior modeling (trend estimation)
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_opponent_trend(self) -> float:
        """
        Estimate how fast the opponent is conceding by looking at recent offers.
        """
        history = self._opponent_offer_history

        if len(history) < 2:
            return 0.0

        recent = history[-self.OPPONENT_WINDOW_SIZE:]
        n = len(recent)

        steps = [h[0] for h in recent]
        utils = [h[1] for h in recent]

        mean_step = sum(steps) / n
        mean_util = sum(utils) / n

        numerator = sum(
            (s - mean_step) * (u - mean_util)
            for s, u in zip(steps, utils)
        )
        denominator = sum((s - mean_step) ** 2 for s in steps)

        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator
        return max(-1.0, min(1.0, slope * 50.0))