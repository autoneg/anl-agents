from __future__ import annotations

from random import choice, random, shuffle
from typing import List, Optional, Tuple

import numpy as np

from negmas import (
    Outcome,
    PolyAspiration,
    PresortingInverseUtilityFunction,
    ResponseType,
)
from negmas.common import PreferencesChangeType
from negmas.gb.components.genius.models import GHardHeadedFrequencyModel
from negmas.sao import SAONegotiator, SAOState


class Anl(SAONegotiator):

    def __init__(
        self,
        *args,
        aspiration_type: str = "boulware",
        e: float = 0.2,
        conceal_prob: float = 0.9,
        conceal_strength: float = 0.6,
        accept_threshold_early: float = 0.85,
        accept_threshold_late: float = 0.75,
        min_accept_threshold: float = 0.55,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # --- Aspiration curve ---
        self._asp = PolyAspiration(1.0, aspiration_type)
        self._e = e

        # --- Concealment parameters ---
        self._conceal_prob = conceal_prob
        self._conceal_strength = conceal_strength

        # --- Acceptance thresholds ---
        self._accept_threshold_early = accept_threshold_early
        self._accept_threshold_late = accept_threshold_late
        self._min_accept_threshold = min_accept_threshold

        # --- Internal state (reset on on_preferences_changed) ---
        self._inv: Optional[PresortingInverseUtilityFunction] = None
        self._best_outcome: Optional[Outcome] = None
        self._worst_rational_outcome: Optional[Outcome] = None
        self._u_max: float = 1.0
        self._u_min: float = 0.0
        self._u_reserved: float = 0.0
        self._issues_count: int = 0

        # --- Opponent model (GHardHeadedFrequencyModel) ---
        # Attached after preferences are set via on_preferences_changed
        self._opponent_model: Optional[GHardHeadedFrequencyModel] = None

        # --- Concealment tracking ---
        # Records the issue-level utility rankings so we can invert them
        self._issue_utilities: List[Tuple[int, float]] = []
        # Previous bid (for concealment logic)
        self._prev_bid: Optional[Outcome] = None
        # Whether we have analyzed issue importance
        self._issues_analyzed: bool = False

        # --- Negotiation history for adaptation ---
        self._opponent_offers: List[Outcome] = []
        self._my_offers: List[Outcome] = []

    # ================================================================
    # NegMAS Lifecycle Callbacks
    # ================================================================

    def on_preferences_changed(self, changes):
        """Called when utility function is assigned or changed."""
        # Filter out trivial Scale changes
        meaningful = [
            c for c in changes
            if c.type not in (PreferencesChangeType.Scale,)
        ]
        if not meaningful:
            return

        if self.ufun is None:
            return

        # Build inverse utility function for efficient outcome lookup
        self._inv = PresortingInverseUtilityFunction(self.ufun, rational_only=True)
        self._inv.init()

        self._u_max = float(self._inv.max())
        self._u_min = float(self._inv.min())
        self._u_reserved = float(self.ufun(None)) if self.ufun(None) is not None else 0.0

        # Cache best outcome
        best = self._inv.best()
        self._best_outcome = best

        # Analyze issue-level importance for concealment
        self._analyze_issue_importance()

        # Initialize opponent model (requires self.negotiator reference)
        self._init_opponent_model()

    def _analyze_issue_importance(self):
        """
        Computes per-issue utility contribution to rank their importance.
        High-importance issues are those whose variation most affects our utility.
        We will VARY these in bids to deceive frequency-based opponent models.
        """
        if self.ufun is None or self.nmi is None:
            return

        outcome_space = self.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._issues_count = len(issues)
        if self._issues_count == 0:
            return

        # For each issue, estimate its marginal utility contribution
        # by sampling the max vs min value while holding others constant
        issue_importances = []
        try:
            # Use a representative mid-point outcome as baseline
            baseline_values = []
            for issue in issues:
                vals = list(issue.all) if hasattr(issue, 'all') else []
                if vals:
                    baseline_values.append(vals[len(vals) // 2])
                else:
                    baseline_values.append(0)
            baseline = tuple(baseline_values)

            for i, issue in enumerate(issues):
                vals = list(issue.all) if hasattr(issue, 'all') else []
                if len(vals) < 2:
                    issue_importances.append((i, 0.0))
                    continue
                # Utility at max vs min value for this issue
                hi_outcome = tuple(
                    vals[-1] if j == i else baseline[j]
                    for j in range(self._issues_count)
                )
                lo_outcome = tuple(
                    vals[0] if j == i else baseline[j]
                    for j in range(self._issues_count)
                )
                u_hi = float(self.ufun(hi_outcome) or 0.0)
                u_lo = float(self.ufun(lo_outcome) or 0.0)
                importance = abs(u_hi - u_lo)
                issue_importances.append((i, importance))
        except Exception:
            # Fallback: uniform importance
            issue_importances = [(i, 1.0) for i in range(self._issues_count)]

        # Sort: most important first
        issue_importances.sort(key=lambda x: x[1], reverse=True)
        self._issue_utilities = issue_importances
        self._issues_analyzed = True

    def _init_opponent_model(self):
        """Initialize the opponent model component."""
        try:
            self._opponent_model = GHardHeadedFrequencyModel(negotiator=self)
        except Exception:
            self._opponent_model = None

    # ================================================================
    # CORE METHODS: propose() and respond()
    # ================================================================

    def propose(self, state: SAOState, dest: Optional[str] = None) -> Optional[Outcome]:
        """
        Generate a bid to offer.

        Strategy:
        1. Compute aspiration level (Boulware curve)
        2. Find outcomes in the aspiration utility range
        3. Apply concealment perturbation to selected outcome
        4. Track bid history
        """
        if self._inv is None or self.ufun is None:
            # Fallback: random outcome
            if self.nmi is not None:
                return self.nmi.random_outcome()
            return None

        # --- Compute aspiration level ---
        level = self._compute_aspiration_level(state)

        # --- Find candidate outcomes near aspiration level ---
        candidate = self._find_candidate_outcome(level)
        if candidate is None:
            candidate = self._best_outcome

        # --- Apply concealment strategy ---
        deceived = self._apply_concealment(candidate, level, state)
        if deceived is not None:
            candidate = deceived

        # Track
        if candidate is not None:
            self._my_offers.append(candidate)
            self._prev_bid = candidate

        return candidate

    def respond(self, state: SAOState, source: Optional[str] = None) -> ResponseType:
        """
        Decide whether to accept, reject, or end the negotiation.

        Strategy:
        1. Reject None offers
        2. End if no rational agreement possible
        3. Accept if utility exceeds time-adjusted threshold
        4. Update opponent model
        """
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model with this offer
        if self._opponent_model is not None:
            try:
                self._opponent_model.on_partner_proposal(
                    state, source or "", offer
                )
            except Exception:
                pass

        self._opponent_offers.append(offer)

        # End negotiation if our max utility is below reservation
        my_u = float(self.ufun(offer) or 0.0)
        reserved = float(self.ufun(None) or 0.0)

        if self._u_max < reserved:
            return ResponseType.END_NEGOTIATION

        # Compute time-adjusted acceptance threshold
        threshold = self._compute_accept_threshold(state)

        # Accept if utility meets threshold
        if my_u >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline: accept anything rational
        if state.relative_time >= 0.98 and my_u >= reserved:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    # ================================================================
    # ASPIRATION LEVEL COMPUTATION
    # ================================================================

    def _compute_aspiration_level(self, state: SAOState) -> float:
        """
        Compute the target utility level for this round using the
        Boulware aspiration curve.

        Returns normalized utility level in [0, 1].
        """
        t = state.relative_time
        if state.step == 0:
            return 1.0

        # Boulware: utility_at(t) = 1 - t^(1/e) for e > 1
        # PolyAspiration handles this
        level = float(self._asp.utility_at(t))

        # Never go below reservation value
        level = max(level, self._u_reserved)
        return min(1.0, max(0.0, level))

    def _compute_accept_threshold(self, state: SAOState) -> float:
        """
        Compute the minimum utility required to accept an offer.
        Decreases over time from accept_threshold_early to min_accept_threshold.
        """
        t = state.relative_time
        if t < 0.5:
            return self._accept_threshold_early
        elif t < 0.8:
            # Linear interpolation from early to late
            frac = (t - 0.5) / 0.3
            return (
                self._accept_threshold_early * (1 - frac)
                + self._accept_threshold_late * frac
            )
        else:
            # Linear interpolation from late to min
            frac = (t - 0.8) / 0.2
            return (
                self._accept_threshold_late * (1 - frac)
                + self._min_accept_threshold * frac
            )

    # ================================================================
    # CANDIDATE OUTCOME SELECTION
    # ================================================================

    def _find_candidate_outcome(self, level: float) -> Optional[Outcome]:
        """
        Find an outcome with utility near the aspiration level.
        Tries progressively wider utility bands.
        """
        if self._inv is None:
            return None

        deltas = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
        for d in deltas:
            lo = max(0.0, level - d * 0.1)
            hi = min(1.0, level + d)
            outcome = self._inv.one_in((lo, hi), normalized=True)
            if outcome is not None:
                return outcome

        # Fallback: best available
        return self._inv.best()

    # ================================================================
    # PREFERENCE CONCEALMENT STRATEGY (ANL2026 KEY INNOVATION)
    # ================================================================

    def _apply_concealment(
        self,
        candidate: Optional[Outcome],
        level: float,
        state: SAOState,
    ) -> Optional[Outcome]:
        
        if candidate is None:
            return None

        # Decide whether to apply concealment this round
        if random() > self._conceal_prob:
            return None  # No concealment; use candidate as-is

        if not self._issues_analyzed or self._issues_count == 0:
            return None

        if self._inv is None or self.ufun is None:
            return None

        if self.nmi is None or self.nmi.outcome_space is None:
            return None

        outcome_space = self.nmi.outcome_space
        issues = outcome_space.issues
        if not issues:
            return None

        # Safety: minimum utility to maintain
        safety_level = max(
            self._u_reserved,
            level * (1.0 - 0.15 * self._conceal_strength),
        )

        # Get importance ranking: most important issues first
        sorted_by_importance = sorted(
            self._issue_utilities, key=lambda x: x[1], reverse=True
        )
        n_important = max(1, int(len(sorted_by_importance) * 0.5))
        important_indices = {idx for idx, _ in sorted_by_importance[:n_important]}

        # Try to find concealed outcome that still meets safety level
        best_concealed = None
        best_concealed_u = 0.0

        # Sample multiple candidate outcomes and select most deceptive
        # while maintaining sufficient utility
        try:
            lo = max(0.0, safety_level - 0.05)
            hi = min(1.0, level + 0.1)
            candidates = self._inv.some((lo, hi), normalized=True, n=12)
            if not candidates:
                return None
        except Exception:
            return None

        for c in candidates:
            u = float(self.ufun(c) or 0.0)
            if u < safety_level:
                continue

            # Compute deception score: how much this bid differs from
            # the "honest" signal on important issues
            # Higher = more deceptive (good for concealment)
            deception_score = self._compute_deception_score(
                c, important_indices, issues
            )

            # We want: high deception_score AND good utility
            # Combined objective: alpha * deception + (1-alpha) * utility
            alpha = self._conceal_strength
            combined = alpha * deception_score + (1 - alpha) * u

            if best_concealed is None or combined > best_concealed_u:
                best_concealed = c
                best_concealed_u = combined

        return best_concealed

    def _compute_deception_score(
        self,
        outcome: Outcome,
        important_indices: set,
        issues,
    ) -> float:
        
        if self._prev_bid is None or not important_indices:
            # First bid: random deception score (no history to compare)
            return random() * 0.5 + 0.5

        if len(outcome) != len(self._prev_bid):
            return 0.5

        n = len(outcome)
        if n == 0:
            return 0.5

        deception_score = 0.0
        for i in range(n):
            is_important = i in important_indices
            changed = (outcome[i] != self._prev_bid[i])

            if is_important and changed:
                # GOOD: varied important issue → misleads opponent
                deception_score += 1.0
            elif not is_important and not changed:
                # GOOD: stable unimportant issue → misleads opponent
                deception_score += 1.0
            # Other cases: no deception benefit

        return deception_score / n

    # ================================================================
    # UTILITY HELPERS
    # ================================================================

    def _normalized_utility(self, outcome: Outcome) -> float:
        """Compute normalized utility: (u - reserved) / (max - reserved)."""
        if self.ufun is None:
            return 0.0
        u = float(self.ufun(outcome) or 0.0)
        reserved = self._u_reserved
        u_range = self._u_max - reserved
        if u_range <= 1e-10:
            return 0.0
        return (u - reserved) / u_range

    def _estimate_opponent_utility(self, outcome: Outcome) -> float:
        """Estimate opponent's utility for an outcome using the opponent model."""
        if self._opponent_model is None:
            return 0.5
        try:
            val = self._opponent_model.eval(outcome)
            return float(val or 0.5)
        except Exception:
            return 0.5


# ================================================================
# STANDALONE TESTING
# ================================================================

if __name__ == "__main__":
   
    from negmas.sao import SAOMechanism
    from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
    from negmas.outcomes import make_issue, make_os
    from negmas import BoulwareTBNegotiator

    print("=== ANL 2026 Anl Local Test ===\n")

    issues = [
        make_issue(10, "price"),
        make_issue(5, "quantity"),
        make_issue(4, "delivery"),
    ]
    os = make_os(issues)

    n_trials = 5
    scores_anl = []
    scores_opp = []

    for trial in range(n_trials):
        session = SAOMechanism(issues=issues, n_steps=100)
        ufun_anl = LUFun.random(os, reserved_value=0.1)
        ufun_opp = LUFun.random(os, reserved_value=0.1)

        anl = Anl(name="Anl")
        opp = BoulwareTBNegotiator(name="Boulware")

        session.add(anl, ufun=ufun_anl)
        session.add(opp, ufun=ufun_opp)

        result = session.run()
        agreement = result.agreement

        if agreement is not None:
            u_anl = float(ufun_anl(agreement) or 0.0)
            u_opp = float(ufun_opp(agreement) or 0.0)
            r_anl = float(ufun_anl(None) or 0.0)
            r_opp = float(ufun_opp(None) or 0.0)
            mx_anl = float(max(ufun_anl(o) or 0.0 for o in os.enumerate_or_sample()) or 1.0)
            mx_opp = float(max(ufun_opp(o) or 0.0 for o in os.enumerate_or_sample()) or 1.0)
            adv_anl = (u_anl - r_anl) / max(1e-6, mx_anl - r_anl)
            adv_opp = (u_opp - r_opp) / max(1e-6, mx_opp - r_opp)
            scores_anl.append(adv_anl)
            scores_opp.append(adv_opp)
            print(f"Trial {trial+1}: Agreement={agreement}, "
                  f"Adv_ANL={adv_anl:.3f}, Adv_OPP={adv_opp:.3f}")
        else:
            scores_anl.append(0.0)
            scores_opp.append(0.0)
            print(f"Trial {trial+1}: No agreement")

    print(f"\nMean Advantage Anl: {np.mean(scores_anl):.3f}")
    print(f"Mean Advantage Opponent:      {np.mean(scores_opp):.3f}")
    print("\nTest complete. Anl is ready for ANL 2026 submission.")