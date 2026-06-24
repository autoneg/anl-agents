import random
from collections import defaultdict
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class DecepTon(SAOCallNegotiator):
    """
    DecepTor: Deceptive negotiating agent for ANL 2026.

    Architecture (BOA + Deception Layer):
    - Bidding: time-based Boulware with DANS-adaptive exponent e
    - Opponent model: GSmithFrequencyModel (inverse-change-frequency weights)
    - Acceptance: composite four-condition strategy (Algorithm 1 from report)
    - Deception: utility-equivalent bid swaps targeting Kendall-tau score
    """

    # ------------------------------------------------------------------ init --

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        # --- per-negotiation state ---
        # e=0.2 → Boulware (α(t) stays near u_max for most of negotiation)
        # e=3.0 → Conceder (α(t) drops rapidly)
        # Note: the report's verbal label for e values is inverted; the math here is correct.
        self._e: float = 0.2          # Boulware start
        self._s_t: float = 0.0        # EMA of estimated opponent utility change
        self._prev_opp_util: float = 0.5
        self._delta_hat: float = 0.0  # sliding-window concession rate
        self._prev_offer = None
        self._opp_util_history: list[float] = []

        # --- enumerate + rank outcomes ---
        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        scored = sorted(((float(self.ufun(o)), o) for o in outcomes), reverse=True)
        self._all_utils: list[float] = [u for u, _ in scored]
        self._all_outcomes: list = [o for _, o in scored]
        r = float(self.ufun.reserved_value)
        self._rational_outcomes: list = [
            o for u, o in zip(self._all_utils, self._all_outcomes)
            if u > r
        ]

        # --- opponent frequency model ---
        self._issues = self.nmi.outcome_space.issues
        self._n_issues: int = len(self._issues)
        self._issue_counts: list[dict] = [defaultdict(float) for _ in range(self._n_issues)]
        self._issue_changes: list[int] = [0] * self._n_issues

        # --- own issue weights (needed for deception) ---
        self._estimate_issue_weights()

        # --- pre-compute H (honest) / D (deceptive) bid sets ---
        self._precompute_bid_sets()

        # expose a live opponent ufun estimate (updated each turn via closure)
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=lambda x: self._estimate_opp_util(x)
        )

    # --------------------------------------------------------- outcome helpers --

    def _val_at(self, outcome, idx: int):
        """Value of issue idx in an outcome (tuple or dict)."""
        if isinstance(outcome, dict):
            return outcome[self._issues[idx].name]
        return outcome[idx]

    # -------------------------------------------------- issue weight estimation --

    def _estimate_issue_weights(self):
        """
        Estimate own issue importance by utility range when varying each issue
        independently while holding all other issues at their globally best value.
        """
        if self.ufun is None or not self._all_outcomes or self._n_issues == 0:
            self._issue_weights = [1.0 / max(self._n_issues, 1)] * self._n_issues
            return

        best = self._all_outcomes[0]
        weights: list[float] = []
        for i, issue in enumerate(self._issues):
            try:
                vals = list(issue.all)
                utils: list[float] = []
                for v in vals:
                    if isinstance(best, dict):
                        o: dict | tuple = dict(best)
                        o[issue.name] = v  # type: ignore[index]
                    else:
                        tmp = list(best)
                        tmp[i] = v
                        o = tuple(tmp)
                    utils.append(float(self.ufun(o)))
                weights.append(max(utils) - min(utils) if len(utils) >= 2 else 0.0)
            except Exception:
                weights.append(0.0)

        total = sum(weights)
        if total > 0:
            self._issue_weights = [w / total for w in weights]
        else:
            self._issue_weights = [1.0 / self._n_issues] * self._n_issues

    # -------------------------------------------------------- deception setup --

    def _precompute_bid_sets(self):
        """
        Partition rational_outcomes into:
          H (honest): top-weight issue is at its globally optimal value.
          D (deceptive): top-weight issue is at a sub-optimal value,
                         so the opponent's frequency model is corrupted.
        """
        if not self._rational_outcomes or self._n_issues == 0:
            self._honest_bids: list = list(self._rational_outcomes)
            self._deceptive_bids: list = []
            return

        top_idx = int(self._issue_weights.index(max(self._issue_weights)))
        top_best_val = self._val_at(self._all_outcomes[0], top_idx) if self._all_outcomes else None

        honest: list = []
        deceptive: list = []
        for o in self._rational_outcomes:
            if top_best_val is not None and self._val_at(o, top_idx) != top_best_val:
                deceptive.append(o)
            else:
                honest.append(o)

        self._honest_bids = honest
        self._deceptive_bids = deceptive

    # --------------------------------------------------- bidding / aspiration --

    def _compute_aspiration(self, t: float) -> float:
        """
        α(t) = r + 0.05 + (1 − t^(1/e)) · (u_max − r − 0.05)
        Monotonically non-increasing; starts near u_max, decays toward r+0.05.
        """
        r = float(self.ufun.reserved_value)
        u_max = float(self.ufun.max())
        t_power = max(t, 0.0) ** (1.0 / max(self._e, 0.01))
        alpha = r + 0.05 + (1.0 - t_power) * (u_max - r - 0.05)
        return max(alpha, r + 0.01)

    # ---------------------------------------------- opponent model (frequency) --

    def _estimate_opp_util(self, outcome) -> float:
        """
        GSmithFrequencyModel with inverse-change-frequency issue weights:
          val(i,v) = count(i,v) / Σ_v' count(i,v')
          w(i)     = 1/(1+changes(i)) / Σ_j 1/(1+changes(j))
          û_B(b)   = Σ_i w(i) · val(i, b[i])
        """
        if outcome is None or self._n_issues == 0:
            return 0.5
        total_w = wv = 0.0
        for i in range(self._n_issues):
            counts = self._issue_counts[i]
            total_count = sum(counts.values())
            v = self._val_at(outcome, i)
            vu = (counts[v] / total_count) if total_count > 0 else 0.5
            w = 1.0 / (1.0 + self._issue_changes[i])
            wv += w * vu
            total_w += w
        return wv / total_w if total_w > 0 else 0.5

    def _update_opp_model(self, offer) -> None:
        """Update frequency counts and per-issue change counters."""
        if offer is None:
            return
        for i in range(self._n_issues):
            v = self._val_at(offer, i)
            self._issue_counts[i][v] += 1
            if self._prev_offer is not None and self._val_at(self._prev_offer, i) != v:
                self._issue_changes[i] += 1
        self._prev_offer = offer

    # ------------------------------------------------- adaptive e via DANS/EMA --

    def _adapt_e(self) -> None:
        """
        EMA tracks opponent utility changes:
          s_t = 0.8·s_{t-1} + 0.2·(û_t − û_{t-1})
        s_t > 0.05  → opponent conceding → stay Boulware (increase e)
        s_t < -0.05 → opponent stubborn  → concede faster (decrease e)
        """
        if self._prev_offer is None:
            return
        curr_u = self._estimate_opp_util(self._prev_offer)
        self._s_t = 0.8 * self._s_t + 0.2 * (curr_u - self._prev_opp_util)
        self._prev_opp_util = curr_u
        if self._s_t > 0.05:
            # Opponent conceding → stay Boulware (decrease e toward 0.2)
            self._e = max(self._e - 0.15, 0.2)
        elif self._s_t < -0.05:
            # Opponent stubborn → concede faster (increase e toward 3.0)
            self._e = min(self._e + 0.15, 3.0)

    # ----------------------------------------------- concession rate tracking --

    def _update_delta_hat(self, offer) -> None:
        """
        Sliding-window estimate of opponent concession rate (from our utility view):
          δ̂(t) ≈ (u_A(b^opp_last) − u_A(b^opp_first)) / window_size
        Negative δ̂ late in the negotiation triggers Condition C acceptance.
        """
        if offer is None:
            return
        self._opp_util_history.append(float(self.ufun(offer)))
        window = min(5, len(self._opp_util_history))
        if window >= 2:
            recent = self._opp_util_history[-window:]
            self._delta_hat = (recent[-1] - recent[0]) / (window - 1)

    # ------------------------------------------------- Nash point approximation --

    def _nash_util_a(self) -> float:
        """
        Approximate Nash utility for us by maximising (u_A − r)·û_B
        over the top-200 rational outcomes ranked by u_A.
        """
        r = float(self.ufun.reserved_value)
        best_u_a, best_prod = r + 0.01, 0.0
        for o in self._rational_outcomes[:200]:
            u_a = float(self.ufun(o))
            prod = (u_a - r) * self._estimate_opp_util(o)
            if prod > best_prod:
                best_prod = prod
                best_u_a = u_a
        return best_u_a

    # ----------------------------------------------- acceptance strategy (Alg 1) --

    def _should_accept(self, offer, state: SAOState) -> bool:
        """
        Algorithm 1 – Composite Acceptance Decision:
          Cond A: u_A(offer) >= α(t)             [ACNext guarantee]
          Cond B: t >= 0.90 and u >= 0.95·u_Nash [Nash proximity at deadline]
          Cond C: δ̂ < 0 and t > 0.70            [opponent reversing late]
          Guard:  never accept below reservation
        """
        if offer is None:
            return False
        r = float(self.ufun.reserved_value)
        u = float(self.ufun(offer))
        t = state.relative_time

        if u < r:
            return False
        if u >= self._compute_aspiration(t):
            return True
        if t >= 0.90 and u >= 0.95 * self._nash_util_a():
            return True
        # Condition C: opponent reversing late — only accept if still above 80% of aspiration at t=0.7
        if self._delta_hat < 0 and t > 0.70:
            floor = self._compute_aspiration(0.70) * 0.80
            if u >= floor:
                return True
        return False

    # ---------------------------------------------------- deception layer / bid --

    def _select_bid(self, alpha: float, t: float) -> Outcome | None:
        """
        Deception schedule:  p_dec(t) = min(0.25, 0.15 + 0.10·t)
        If rand < p_dec and D_{α(t)} ≠ ∅ → pick from deceptive set
        Otherwise                          → Nash-optimal honest bid
        Deceptive bids are constrained to ≥ 0.92·α(t) (≤ 8% utility loss).
        """
        r = float(self.ufun.reserved_value)
        min_u = max(alpha * 0.92, r + 0.01)
        p_dec = min(0.25, 0.15 + 0.10 * t)

        dec_cands = [o for o in self._deceptive_bids if float(self.ufun(o)) >= min_u]
        hon_cands = [o for o in self._honest_bids if float(self.ufun(o)) >= min_u]

        # Deceptive bid path
        if random.random() < p_dec and dec_cands:
            return random.choice(dec_cands)

        # Honest bid: maximise Nash proximity (u_A − r)·û_B
        if hon_cands:
            top = hon_cands[:50]  # cap to avoid per-turn O(N) opponent model calls
            return max(top, key=lambda o: (float(self.ufun(o)) - r) * self._estimate_opp_util(o))

        # Fallback: any rational outcome above threshold
        for o in self._rational_outcomes:
            if float(self.ufun(o)) >= min_u:
                return o

        return self._rational_outcomes[0] if self._rational_outcomes else None

    # ---------------------------------------------------------------- main loop --

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        t = state.relative_time

        # First call — no offer yet; open with near-maximum utility bid
        if offer is None:
            alpha = self._compute_aspiration(0.0)
            bid = self._select_bid(alpha, 0.0)
            if bid is None:
                return SAOResponse(ResponseType.END_NEGOTIATION, None)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        # Update models
        self._update_opp_model(offer)
        self._adapt_e()
        self._update_delta_hat(offer)

        # Accept?
        if self._should_accept(offer, state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # Counter-offer
        alpha = self._compute_aspiration(t)
        bid = self._select_bid(alpha, t)
        if bid is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)
