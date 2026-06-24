
"""
=============================
OzuNegotiator — ANL 2026
=============================

Two ideas implemented:

1. Two-segment aspiration curve (ultra-stubborn):
   Flat segment (t < 0.90):  e=0.05 Boulware — nearly zero concession
                              t=0.50 → 0.9999, t=0.80 → 0.988, t=0.90 → 0.878
   Steep segment (t ≥ 0.90): linear drop from 0.878 to reserved over final 10%
                              All real concession happens in the last 10 rounds.

   Why: cooperative opponents concede toward you for 90 rounds. By the time
   you start dropping, they've already moved far. Against stubborn opponents,
   both approach the deadline zone — Phase 3b bidding protects us there.

2. Phase 3 bidding overhaul (protect against deadline exploitation):
   Phase 3a (0.75-0.90): rotate through our top 5 outcomes — some bid
                          diversity for Concealing, but no low-utility bids.
   Phase 3b (t ≥ 0.90):  always bid our single best available outcome.
                          By this point deception is irrelevant (model is
                          built from phases 1 and 2). Every offer must
                          maximize Advantage.
"""

import random
from collections import Counter
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class OzuNegotiator(SAOCallNegotiator):
    """
    ANL 2026 deceptive negotiation agent.

    Score = Advantage + Concealing
      Advantage  = utility(agreement) - reserved_value
      Concealing = 1 - opponent's Kendall accuracy on our ufun

    Architecture:
      1. Two-segment aspiration: flat (e=0.05) until t=0.90, linear drop after
      2. Four-phase bidding: deceptive random early, rotation mid, best-only late
      3. Single decoy issue: lock least important issue to fake value
      4. Basic frequency opponent model (Baarslag/Smith)
      5. AC_Const + AC_Aspiration + AC_Time(t > 0.97)
    """

    rational_outcomes = tuple()

    # ------------------------------------------------------------------ #
    #  INITIALIZATION
    # ------------------------------------------------------------------ #

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        self.rational_outcomes = tuple(
            o
            for _, o in sorted(
                (
                    (self.ufun(o), o)
                    for o in self.nmi.outcome_space.enumerate_or_sample()
                    if self.ufun(o) > self.ufun.reserved_value
                ),
                reverse=True,
            )
        )

        self._opp_freq: dict[int, dict] = {}
        self._opp_n = 0

        self._issue_imp = self._compute_issue_importance()

        self._decoy_issue: int | None = None
        self._decoy_val = None
        if len(self._issue_imp) > 1 and self.rational_outcomes:
            self._decoy_issue = min(
                range(len(self._issue_imp)), key=lambda i: self._issue_imp[i]
            )
            vals = [o[self._decoy_issue] for o in self.rational_outcomes]
            self._decoy_val = Counter(vals).most_common(1)[0][0]

        self._last_bid: Outcome | None = None
        self._made_middle_offer = False

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda _: 0.5)

    @property
    def opponent_ufun(self):
        """Read the estimated opponent utility function from private_info."""
        return self.private_info.get("opponent_ufun", None)

    def _compute_issue_importance(self) -> list[float]:
        if not self.rational_outcomes:
            return []
        n_issues = len(self.rational_outcomes[0])
        baseline = list(self.rational_outcomes[0])
        sample = list(self.nmi.outcome_space.enumerate_or_sample())[:300]
        importances = []
        for i in range(n_issues):
            vals = list({o[i] for o in sample})
            utils = []
            for v in vals:
                probe = baseline[:]
                probe[i] = v
                utils.append(float(self.ufun(tuple(probe))))
            importances.append(max(utils) - min(utils) if utils else 0.0)
        return importances

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        if offer is None:
            bid = self._bid(state)
            self._last_bid = bid
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        self.update_opponent_model(state)

        # Deadline middle-offer: at t > 0.97, before accepting, check if the
        # gap between our last bid and their offer is large (> 0.2 utility).
        # If so, counter once with the midpoint utility instead of capitulating.
        # This prevents a stubborn opponent from extracting a free cheap deal
        # via AC_Time. After one middle offer we commit to accepting next round.
        t = state.relative_time
        if t > 0.97 and not self._made_middle_offer and self._last_bid is not None:
            our_util = float(self.ufun(self._last_bid))
            their_util = float(self.ufun(offer))
            if our_util - their_util > 0.2:
                mid = self._middle_offer(offer)
                if mid is not None:
                    self._made_middle_offer = True
                    self._last_bid = mid
                    return SAOResponse(ResponseType.REJECT_OFFER, mid)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self._bid(state)
        self._last_bid = bid
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    # ------------------------------------------------------------------ #
    #  TWO-SEGMENT ASPIRATION CURVE
    # ------------------------------------------------------------------ #

    def _aspiration(self, t: float) -> float:
        """
        Segment 1 (t < 0.90) — nearly flat Boulware with e=0.05 (1/e=20):
          t=0.50 → 0.9999   (essentially max utility demanded)
          t=0.80 → 0.988
          t=0.90 → 0.878    (pivot point)

        Segment 2 (t >= 0.90) — linear drop to reserved:
          t=0.90 → 0.878 (same as pivot, continuous)
          t=0.95 → 0.439
          t=1.00 → reserved

        All meaningful concession is compressed into the final 10 rounds.
        """
        reserved = float(self.ufun.reserved_value)
        PIVOT_T = 0.90
        pivot_val = 1.0 - (PIVOT_T ** 20)  # ≈ 0.878

        if t < PIVOT_T:
            val = 1.0 - t ** 20
            return max(reserved, reserved + (1.0 - reserved) * val)
        else:
            progress = (t - PIVOT_T) / (1.0 - PIVOT_T)
            val = pivot_val * (1.0 - progress)
            return max(reserved, reserved + (1.0 - reserved) * val)

    # ------------------------------------------------------------------ #
    #  ACCEPTANCE STRATEGY
    # ------------------------------------------------------------------ #

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        1. AC_Const      — utility >= 90% of max: never refuse near-perfect deal
        2. AC_Aspiration — utility >= aspiration(t): Boulware, loosens near deadline
        3. AC_Time       — t > 0.97: last-resort safety net
                           Aspiration already drops steeply after t=0.90 so we
                           rarely need this — it just catches edge cases.
        """
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return False

        my_util = float(self.ufun(offer))
        t = state.relative_time
        reserved = float(self.ufun.reserved_value)

        if my_util <= reserved:
            return False

        if my_util >= float(self.ufun.max()) * 0.9:
            return True

        if my_util >= self._aspiration(t):
            return True

        if t > 0.97:
            return True

        return False

    # ------------------------------------------------------------------ #
    #  BIDDING STRATEGY
    # ------------------------------------------------------------------ #

    def _bid(self, state: SAOState) -> Outcome | None:
        """
        Phase 1 (t < 0.40):   all acceptable + decoy, uniform random
                               Maximum Concealing — opponent sees full chaos.

        Phase 2 (0.40-0.75):  top 40% + decoy, uniform random
                               Moderate Concealing, tighter utility floor.

        Phase 3a (0.75-0.90): top 5 outcomes, NO decoy.
                               Re-ordered by estimated opponent utility so we
                               offer the bid the opponent is most likely to
                               accept — while staying within our high floor.
                               Uses opponent model for the first time.

        Phase 3b (t >= 0.90): top 3 outcomes, pick by opponent utility.
                               Pure Advantage + agreement rate maximization.
                               All bids are near our max utility. We pick the
                               one the opponent wants most → fastest acceptance
                               at our best possible terms.

        The opponent model is used in phases 3a and 3b.
        """
        if not self.rational_outcomes:
            return None

        t = state.relative_time
        asp = self._aspiration(t)

        acceptable = [o for o in self.rational_outcomes if float(self.ufun(o)) >= asp]
        if not acceptable:
            n = max(1, len(self.rational_outcomes) // 10)
            acceptable = list(self.rational_outcomes[:n])

        if t < 0.40:
            pool = self._apply_decoy(acceptable)
            return random.choice(pool)

        if t < 0.75:
            n = max(1, int(len(acceptable) * 0.4))
            pool = self._apply_decoy(acceptable[:n])
            return random.choice(pool)

        if t < 0.90:
            # Phase 3a: pick from top 5 by our utility, prefer what opponent values
            top = acceptable[:min(5, len(acceptable))]
            return self._best_for_opponent(top)

        # Phase 3b: pick from top 3 by our utility, prefer what opponent values
        top = acceptable[:min(3, len(acceptable))]
        return self._best_for_opponent(top)

    def _best_for_opponent(self, candidates: list) -> Outcome | None:
        """
        Among candidates (all already above our utility floor), return the one
        the opponent values most according to our frequency model.

        This is the correct way to use the opponent model in bidding:
        - We never lower our utility floor to please the opponent
        - We only choose WHICH bid to make at our current floor
        - The opponent is more likely to accept a bid that is good for them
        - Result: faster agreements at OUR utility level → higher Advantage

        Falls back to the first candidate (our best) if model is not warm yet.
        """
        if not candidates:
            return None
        opp = self.opponent_ufun
        if opp is None or self._opp_n < 3:
            return candidates[0]  # not enough data — just bid our best
        try:
            return max(candidates, key=lambda o: float(opp(o)))
        except Exception:
            return candidates[0]

    def _middle_offer(self, opponent_offer: Outcome) -> Outcome | None:
        """
        Find the rational outcome whose utility for us is closest to the
        midpoint between our last bid utility and the opponent's offer utility,
        but never below that midpoint (we don't concede more than halfway).

        Example: our last bid = 0.90 utility, opponent offer = 0.40 utility
                 target midpoint = 0.65
                 → find the lowest-utility outcome that is still >= 0.65

        Why from above: we want to split the difference, not surrender.
        The opponent sees a fair compromise offer. If they reject, AC_Time
        will accept their next offer — but at least we extracted one step up.
        """
        if not self.rational_outcomes or self._last_bid is None:
            return None

        our_util = float(self.ufun(self._last_bid))
        their_util = float(self.ufun(opponent_offer))
        target = (our_util + their_util) / 2.0

        # rational_outcomes is sorted best-first, so outcomes above target
        # are at the front. The last element of that slice is closest to target.
        above = [o for o in self.rational_outcomes if float(self.ufun(o)) >= target]
        if above:
            return above[-1]

        # Nothing above target — return our best available (shouldn't happen often)
        return self.rational_outcomes[0]

    def _apply_decoy(self, candidates: list) -> list:
        if self._decoy_issue is None or self._decoy_val is None:
            return candidates
        filtered = [o for o in candidates if o[self._decoy_issue] == self._decoy_val]
        return filtered if filtered else candidates

    # ------------------------------------------------------------------ #
    #  OPPONENT MODEL
    # ------------------------------------------------------------------ #

    def update_opponent_model(self, state: SAOState) -> None:
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return

        self._opp_n += 1
        for i, val in enumerate(offer):
            if i not in self._opp_freq:
                self._opp_freq[i] = {}
            self._opp_freq[i][val] = self._opp_freq[i].get(val, 0) + 1

        if self._opp_n < 3:
            return

        freq = {i: dict(vc) for i, vc in self._opp_freq.items()}
        n_issues = len(self.rational_outcomes[0]) if self.rational_outcomes else len(offer)

        def opp_util(outcome) -> float:
            if outcome is None:
                return 0.0
            weights = []
            for i in range(n_issues):
                if i in freq and freq[i]:
                    total = sum(freq[i].values())
                    top = max(freq[i].values())
                    weights.append(top / total)
                else:
                    weights.append(0.5)
            w_total = sum(weights) or 1.0
            util = 0.0
            for i, val in enumerate(outcome):
                w = weights[i] / w_total
                if i in freq and freq[i]:
                    total = sum(freq[i].values())
                    util += w * (freq[i].get(val, 0) / total)
                else:
                    util += w * 0.5
            return util

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=opp_util)
