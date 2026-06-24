import random
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
# from opponent_tracker import OpponentTracker
# from .opponent_tracker_new import OpponentTrackerNew
try:
    from .ordinal_opponent_tracker import OrdinalOpponentTracker
except ImportError:
    from ordinal_opponent_tracker import OrdinalOpponentTracker
try:
    from .fake_profile_builder import FakeProfileBuilder
except ImportError:
    from fake_profile_builder import FakeProfileBuilder
import copy
import math
import numpy as np


class BalanceOKNegotiator(SAOCallNegotiator):
    """
    Your negotiator code. This is the ONLY class you need to implement.
    """

    rational_outcomes = tuple()

    # ── Hyperparameters ────────────────────────────────────────────
    params = {
        # -- Acceptance --
        "accept_time_threshold": 0.910000003,   # start conceding after this fraction of time
        "accept_ending_threshold_2": 0.9700000003,
        "accept_min_util_ratio":  0.90,  # min_u = max_u * this  (at t=1, threshold hits this)
        "accept_concede_power":   2,  # concession curve exponent (>1 = hold longer, rush late)
        "accept_oppo_factor":     0.15,  # how strongly u_pred_oppo biases acceptance threshold

        # -- Blitz (early snipe) --
        "blitz_time_max":  0.02,         # only snipe before this fraction of time
        "accept_ending_threshold_3": 0.98,
        "accept_ending_threshold_4": 0.99,
        "final_safe_surplus_ratio_098": 0.30,
        "final_safe_surplus_ratio_099": 0.20,
    }

    @property
    def opponent_ufun(self):
        tracker = getattr(self, "opponent_tracker", None)
        predicted = getattr(tracker, "predicted_oppo_ufun", None)
        if predicted is not None:
            return predicted
        return getattr(self, "private_info", {}).get("opponent_ufun", None)

    # ── Initialisation ──────────────────────────────────────────────

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2026, this is equivalent with initializing the agent.
        """

        if self.ufun is None:
            return

        # Print all issues with weights and option utilities
        issues = self.nmi.outcome_space.issues
        # print("=" * 60) 
        # print("Preference Profile:")
        for i, issue in enumerate(issues):
            w = self.ufun.weights[i]
            # print(f"  Issue: {issue.name}  (weight={w:.4f})")
            mapping = self.ufun.values[i].mapping
        #     for option, util in sorted(mapping.items(), key=lambda x: -x[1]):
                # print(f"    {option}: {util:.4f}")
        # print(f"  Reserved value: {self.ufun.reserved_value:.4f}")
        # print("=" * 60)

        # Create a list of all rational outcomes sorted by utility
        ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.rational_outcomes = tuple(_[1] for _ in sorted(ufun_outcome, reverse=True))

        # Compute best bids as a SET (O(1) lookup)
        if self.rational_outcomes:
            max_util = self.ufun(self.rational_outcomes[0])
            tolerance = 0.05
            self.best_bids = {
                o for o in self.rational_outcomes
                if abs(self.ufun(o) - max_util) <= tolerance
            }
        else:
            self.best_bids = set()

        # ── Classify true outcomes into three tiers ──
        n_rat = len(self.rational_outcomes)
        if n_rat > 0:
            max_u = self.ufun(self.rational_outcomes[0])

            tier1_abs = {o for o in self.rational_outcomes if self.ufun(o) >= max_u - 0.07}
            tier2_abs = {o for o in self.rational_outcomes if self.ufun(o) >= max_u - 0.19}
            tier3_abs = {o for o in self.rational_outcomes if self.ufun(o) >= max_u - 0.33}

            top13 = max(1, int(n_rat * 0.13))
            top28 = max(1, int(n_rat * 0.28))
            top43 = max(1, int(n_rat * 0.43))

            tier1_pct = set(self.rational_outcomes[:top13])
            tier2_pct = set(self.rational_outcomes[:top28])
            tier3_pct = set(self.rational_outcomes[:top43])

            # self._true_tier1 = tier1_abs | tier1_pct
            # self._true_tier2 = (tier2_abs | tier2_pct) - self._true_tier1
            # self._true_tier3 = (tier3_abs | tier3_pct) - self._true_tier1 - self._true_tier2

            self._true_tier1 = tier1_abs & tier1_pct
            self._true_tier2 = (tier2_abs & tier2_pct) - self._true_tier1
            self._true_tier3 = (tier3_abs & tier3_pct) - self._true_tier1 - self._true_tier2

            self._ending_tier = tier3_abs | tier3_pct
        else:
            self._true_tier1 = set()
            self._true_tier2 = set()
            self._true_tier3 = set()
            self._ending_tier = set()

        # ── Build helpers BEFORE _build_fake_profile ──

        issues = self.nmi.outcome_space.issues

        # True weights median (for within-issue obfuscation)
        self._weight_median = float(np.median(self.ufun.weights))

        # Cache option sort order per issue
        self._issue_options_sorted = [
            sorted(issues[i].values,
                   key=lambda o: self.ufun.values[i].mapping[o], reverse=True)
            for i in range(len(issues))
        ]

        # Build deceptive fake profile via FakeProfileBuilder
        builder = FakeProfileBuilder(
            true_weights=list(self.ufun.weights),
            issue_options_sorted=self._issue_options_sorted,
            ufun_values=[self.ufun.values[i].mapping for i in range(len(issues))],
            rational_outcomes=self.rational_outcomes,
            true_tier1=self._true_tier1,
            true_tier2=self._true_tier2,
            true_tier3=self._true_tier3,
        )
        (self._fake_weights_initial,
         self._fake_option_utils,
         self._blacklist) = builder.build()

        # ── Compute bid pools: true tier ∩ fake-acceptable tiers ──
        fake_tier1, fake_tier2, fake_tier3 = builder._classify_tiers_for_profile(
            self._fake_weights_initial, self._fake_option_utils
        )
        self._bid_tier1 = self._true_tier1 & (fake_tier1 | fake_tier2)
        self._bid_tier2 = self._true_tier2 & (fake_tier1 | fake_tier2 | fake_tier3)
        self._bid_tier3 = self._true_tier3 & (fake_tier1 | fake_tier2 | fake_tier3)

        # Cache issue index → option → true utility mapping
        self._option_utils = [
            {opt: self.ufun.values[i].mapping[opt]
             for opt in issues[i].values}
            for i in range(len(issues))
        ]

        # Initialize opponent model
        op_issues = copy.deepcopy(self.nmi.outcome_space.issues)
        issue_value_info = {issue.name: issue.values for issue in op_issues}

        self.opponent_tracker = OrdinalOpponentTracker(
            my_ufun=self.ufun,
            our_name="ooooooooo",
            opp_name="Opponent",
            safe_d_path=None,
            repeat=None,
            opponent_ufun=LambdaMultiFun(f=lambda x: 0.5),
            negotiator_index=0,
            curr_neg_index=0,
            issue_value_info=issue_value_info,
            op_issues=op_issues,
        )
        self.private_info["opponent_ufun"] = self.opponent_tracker.predicted_oppo_ufun

        # Track the "fake persona" state for consistency across rounds
        self._last_fake_weights = self._fake_weights_initial[:]

    # ── Main call ───────────────────────────────────────────────────

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called to (counter-)offer.
        """

        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is None:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self.deceptive_bidding_strategy(state)
            )

        if self._try_blitz_accept(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(
            ResponseType.REJECT_OFFER, self.deceptive_bidding_strategy(state)
        )




    def _oppo_pred_util(self, outcome) -> float:
        """Shortcut: predicted opponent utility of an outcome."""
        return float(self.opponent_tracker.predicted_oppo_ufun(outcome))

    def _max_oppo_pred_util(self) -> float:
        """Maximum utility the opponent can achieve under our current model."""
        tracker = self.opponent_tracker
        total = 0.0
        for i, w in enumerate(tracker.oppo_weights):
            max_v = max(tracker.oppo_values[i].mapping.values())
            total += w * max_v
        return total

    # ── Acceptance ──────────────────────────────────────────────────

    def _try_blitz_accept(self, state: SAOState) -> bool:
        """
        Early snipe: if the opponent gives us a tier-1 offer within the
        blitz window, take it immediately.
        """
        offer = state.current_offer
        if offer is None:
            return False

        if state.relative_time > self.params["blitz_time_max"]:
            return False

        if offer in self._true_tier1:
            self.opponent_tracker.record_oppo_offer(offer, state.relative_time)
            self.private_info["opponent_ufun"] = self.opponent_tracker.predicted_oppo_ufun
            return True

        return False

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Acceptance strategy:
          - Never accept ≤ reserved_value.
          - Last-chance: t > accept_last_chance_t → accept anything > reserved.
          - Early → only accept best_bids.
          - Mid/late → time-decaying threshold, softened by opponent model:
            if the opponent is making a big concession (their offer has low
            u_pred_oppo relative to their max), we accept more readily.
        """

        assert self.ufun

        offer = state.current_offer
        if offer is None:
            return False

        offer_util = self.ufun(offer)

        # ── Hard red line ──
        if offer_util <= self.ufun.reserved_value:
            return False

        p = self.params
        t = state.relative_time
        max_u = self.ufun(self.rational_outcomes[0]) if self.rational_outcomes else 1.0

        # ── Last-chance: accept anything above reserved ──
        # if t >= p["accept_last_chance_t"]:
        #     return True

        if offer in self._true_tier1:
            return True

        # ── Base threshold ──
        if t < p["accept_time_threshold"]:
            # Early: only accept _true_tier1
            if offer in self._true_tier1:
                return True
            base_threshold = max_u  # effectively never accept non-best early
        else:
            if offer in self._true_tier1 or offer in self._true_tier2:
                return True
            
            t_norm = (t - p["accept_time_threshold"]) / (1 - p["accept_time_threshold"])
            min_u = max_u * p["accept_min_util_ratio"]
            base_threshold = min_u + (max_u - min_u) * (1 - pow(t_norm, p["accept_concede_power"]))

        ## mtz adjust
        # if t >= p["accept_ending_threshold_2"]:
        #     if offer in self._ending_tier:
        #         return True
        ##----
        # ── Final-stage acceptance ──
        if t > p["accept_ending_threshold_4"]:
            safe_floor = self.ufun.reserved_value + p["final_safe_surplus_ratio_099"] * (
                max_u - self.ufun.reserved_value
            )
            oppo_util = self._oppo_pred_util(offer)

            return (
                offer_util > oppo_util
                and offer_util >= safe_floor
            )

        elif t > p["accept_ending_threshold_3"]:
            safe_floor = self.ufun.reserved_value + p["final_safe_surplus_ratio_098"] * (
                max_u - self.ufun.reserved_value
            )
            oppo_util = self._oppo_pred_util(offer)

            return (
                offer_util > oppo_util
                and offer_util >= safe_floor
            )

        elif t >= p["accept_ending_threshold_2"]:
            return offer in self._ending_tier
        ## ----##


        # ── Opponent-model adjustment ──
        oppo_util = self._oppo_pred_util(offer)
        oppo_max = self._max_oppo_pred_util()
        if oppo_max > 0:
            oppo_norm = oppo_util / oppo_max  # ∈ [0,1]
            # oppo_norm high → opponent is asking for a lot → raise threshold
            # oppo_norm low  → opponent is conceding     → lower threshold
            adjustment = p["accept_oppo_factor"] * (oppo_norm - 0.5)
            effective_threshold = base_threshold * (1 + adjustment)
        else:
            effective_threshold = base_threshold

        return offer_util >= effective_threshold

    # ── Bidding ─────────────────────────────────────────────────────

    def deceptive_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Tier-based bidding with fake-persona consistency:
          t < 0.75  → random from bid_tier1 (true_t1 ∩ fake_t1|t2)
          0.75 ≤ t < 0.91 → random from bid_tier1 ∪ bid_tier2
          t ≥ 0.91  → best opponent-model match within bid_tier1 ∪ bid_tier2 ∪ bid_tier3
        """
        if not self.rational_outcomes:
            return None

        t = state.relative_time

        if t < 0.75:
            pool = list(self._bid_tier1)
        elif t < 0.91:
            pool = list(self._bid_tier1 | self._bid_tier2)
        else:
            pool = list(self._bid_tier1 | self._bid_tier2 | self._bid_tier3)

        if not pool:
            pool = [self.rational_outcomes[0]]

        if t < 0.91:
            offer = random.choice(pool)
        else:
            best = max(pool, key=lambda o: self._oppo_pred_util(o))
            offer = best

        self.opponent_tracker.record_self_offer(offer)

        # Preemptive acceptance bookkeeping exists on the older tracker only.
        # The KD ordinal tracker learns from rejected-self / opponent-counter pairs.
        if (
            hasattr(self.opponent_tracker, "opponent_accepted_offers")
            and hasattr(self.opponent_tracker, "record_opponent_accepted")
        ):
            self._pre_accept_state = list(self.opponent_tracker.opponent_accepted_offers)
            self.opponent_tracker.record_opponent_accepted()
        self.private_info["opponent_ufun"] = self.opponent_tracker.predicted_oppo_ufun

        return offer

    # ── Opponent modelling ──────────────────────────────────────────

    def update_opponent_model(self, state: SAOState) -> None:
        """
        Update the opponent model based on the current offer.
        """

        assert self.ufun

        # Rollback speculative acceptance if the active tracker supports it.
        if (
            hasattr(self, '_pre_accept_state')
            and hasattr(self.opponent_tracker, "opponent_accepted_offers")
        ):
            self.opponent_tracker.opponent_accepted_offers = self._pre_accept_state
            del self._pre_accept_state

        offer = state.current_offer
        if offer is not None:
            self.opponent_tracker.record_oppo_offer(offer, state.relative_time)

        # Sync the predicted opponent ufun to the field that the scoring system evaluates
        self.private_info["opponent_ufun"] = self.opponent_tracker.predicted_oppo_ufun
        # self.private_info["opponent_ufun"] = None