import math

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction


class Badiron(SAOCallNegotiator):
    """Badiron — ANL 2026 bilateral negotiation agent (v4).

    The per-negotiation score has two additive halves (``2026cfp.pdf`` §6 and the
    NegMAS ``anl2026`` metric)::

        score = advantage + concealing_share
          advantage        = (u(deal) - reserve) / (max_u - reserve)   in [0, 1]
          concealing_share = tau_self / (tau_self + tau_opp)           a contest
          tau = (1 + kendall(true_ufun, model)) / 2

    Three levers move the score; this agent pushes all three, but is *advantage
    first* (the half with the most spread against a strong field):

    1.  **Deal quality (advantage).**  A firm time-based Boulware aspiration that
        holds near our maximum and only concedes in the final stretch, with an
        *adaptive floor* that rises as the opponent reveals what they are willing
        to give us — so we exploit opponents that concede, yet still drop in far
        enough at the very end to close a deal against tough ones rather than
        timing out into a zero-advantage disagreement.

    2.  **Opponent modelling (raise tau_self).**  A frequency model (value
        preference proportional to offer frequency; issue weight proportional to
        offer-distribution concentration) republished every turn under
        ``private_info['opponent_ufun']`` — the exact key the scorer reads.
        Forgetting this single line forfeits the entire concealing half silently.

    3.  **Pareto-frontier bidding.**  Among the outcomes still acceptable to us we
        offer the one the opponent is estimated to value most.  This keeps every
        concession on the efficient frontier: the opponent accepts sooner, on
        terms that are still good for us — directly buying advantage, and as a
        free side effect spreading our offers over many outcomes (mild
        concealment) instead of repeating one bid.
    """

    rational_outcomes = tuple()

    # --- strategy parameters (swept empirically; see tests/tough_bench.py) ----
    concession_exponent = 12.0     # Boulware curvature (>1 = firm). target = max - span*t^e.
    #                                Empirically firm wins: hold near max, let the opponent
    #                                concede, only release at the very end.
    accept_slack = 0.0             # extra utility required before accepting (AC_next)
    good_enough = 0.98             # accept any offer at/above this normalized advantage
    #                                immediately, regardless of time (capture near-max deals)
    min_floor_frac = 0.0           # optional minimum advantage we insist on in offers; 0 = off
    #                                (sweeping all the way down late captures low-but-positive
    #                                deals, which beat a zero-advantage disagreement).
    endgame_time = 0.97            # only in the final stretch do we sweep the offer down to
    #                                the (adaptive) floor to close — conceding earlier just
    #                                hands firm opponents the surplus and lowers our advantage.
    final_time = 0.98              # after this, accept anything strictly above the reserve
    adaptive_floor = True          # let the floor rise toward the best offer the opp made us
    floor_fraction = 0.85          # how much of best_received feeds the adaptive floor
    first_offer_boost = 0.5        # rank boost for the opponent's first-offer values in our
    #                                model of them (their opening ≈ their ideal point)
    max_candidate_outcomes = 6000  # cap the working outcome set for speed on huge spaces

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # opponent frequency model
        self._issues = []
        self._opp_counts: list[dict] = []
        self._opp_first_step: list[dict] = []  # step each value first appeared
        self._opp_total = 0
        self._first_offer: Outcome | None = None  # opponent's revealed ideal point
        # fast internal opponent scorer: per-issue {value: utility}, and weights
        self._opp_val_score: list[dict] = []
        self._opp_weights: list[float] = []
        # best (for us) outcome the opponent has actually offered
        self._best_received: Outcome | None = None
        self._best_received_util = float("-inf")

    # ----------------------------------------------------------------- setup
    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        self._os = self.nmi.outcome_space
        self._issues = list(self._os.issues)

        outcomes = list(self._os.enumerate_or_sample())
        ufun_outcome = [(float(self.ufun(o)), o) for o in outcomes]

        self._max_util = max((u for u, _ in ufun_outcome), default=1.0)
        self._min_util = min((u for u, _ in ufun_outcome), default=0.0)

        reserve = float(self.ufun.reserved_value)
        self._reserve = reserve if math.isfinite(reserve) else self._min_util
        if self._max_util <= self._reserve:
            self._max_util = self._reserve + 1e-9

        # Rational outcomes, sorted best-first (by our utility).
        rational = sorted(
            (t for t in ufun_outcome if t[0] > self._reserve),
            key=lambda x: x[0],
            reverse=True,
        )
        if not rational:
            rational = sorted(ufun_outcome, key=lambda x: x[0], reverse=True)[:1]

        # Cap the working set on huge spaces: always keep the best outcomes, and
        # sample the rest so the opponent-favourable (lower) region stays present.
        if len(rational) > self.max_candidate_outcomes:
            keep_top = self.max_candidate_outcomes // 2
            head = rational[:keep_top]
            tail = rational[keep_top:]
            step = max(1, len(tail) // (self.max_candidate_outcomes - keep_top))
            rational = head + tail[::step]

        self.rational_outcomes_with_utility = rational
        self.rational_outcomes = tuple(o for _, o in rational)
        self._cand_utils = [u for u, _ in rational]  # descending

        # opponent model state
        self._opp_counts = [dict() for _ in self._issues]
        self._opp_first_step = [dict() for _ in self._issues]
        self._opp_total = 0
        self._first_offer = None
        self._best_received = None
        self._best_received_util = float("-inf")
        self._recompute_opponent_model()
        self._publish_opponent_ufun()

    # ------------------------------------------------------------- concession
    def _offer_floor(self) -> float:
        """Lowest utility we will *proactively offer*.

        At least ``reserve + min_floor_frac*(max-reserve)`` — conceding below
        that buys ~0 advantage (a deal there is barely better than walking away)
        while handing firm opponents the surplus, so there is no point.  Raised
        further toward a fraction of the best offer the opponent has already made
        us, since that utility is effectively on the table (we can re-offer it).
        """
        span = self._max_util - self._reserve
        floor = self._reserve + self.min_floor_frac * span
        if self.adaptive_floor and math.isfinite(self._best_received_util):
            floor = max(floor, self._reserve
                        + self.floor_fraction * (self._best_received_util - self._reserve))
        return max(self._reserve, min(floor, self._max_util))

    def _target_utility(self, t: float) -> float:
        """Our aspiration utility at relative time ``t``: a Boulware curve from
        ``max`` down toward the offer floor, with a faster linear sweep to the
        floor across the endgame so the opponent gets several steps to accept an
        at-floor offer before the deadline."""
        t = max(0.0, min(1.0, t))
        floor = self._offer_floor()
        span = self._max_util - floor
        boulware = self._max_util - span * math.pow(t, self.concession_exponent)

        eg = self.endgame_time
        if t >= eg and eg < 1.0:
            cap = self._max_util - span * math.pow(eg, self.concession_exponent)
            frac = (t - eg) / (1.0 - eg)
            endgame = cap - frac * (cap - floor)
            return min(boulware, endgame)
        return boulware

    # --------------------------------------------------------------- main loop
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        if offer is not None:
            self._update_opponent_model(offer)

        if offer is not None and self._accepts(state, offer):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self._bid(state))

    # ----------------------------------------------------------- acceptance
    def _accepts(self, state: SAOState, offer: Outcome) -> bool:
        u = float(self.ufun(offer))
        if u <= self._reserve:
            return False

        t = state.relative_time
        n_steps = self.nmi.n_steps or 100
        step = 1.0 / n_steps

        # Excellent offer: grab a near-maximal deal immediately, whatever the time
        # — holding out for the last sliver of utility risks the offer not
        # recurring (it costs us at most 1-good_enough of advantage).
        span = self._max_util - self._reserve
        if span > 0 and (u - self._reserve) / span >= self.good_enough:
            return True

        # AC_next: accept anything at least as good as our own upcoming demand.
        target_next = self._target_utility(t + step)
        if u + self.accept_slack >= target_next:
            return True

        # End game: capture a strong, possibly non-recurring offer (the best the
        # opponent has shown) once we are near the deadline — but only if it
        # clears our floor, so we do not prematurely lock in a near-worthless
        # deal that a firmer hold could beat.
        if (t >= self.endgame_time and u >= self._best_received_util - 1e-9
                and u >= self._offer_floor()):
            return True
        # Last resort: any positive-advantage deal beats timing out at zero.
        if t >= self.final_time:
            return True
        return False

    # ------------------------------------------------------------- bidding
    def _bid(self, state: SAOState) -> Outcome | None:
        if not self.rational_outcomes_with_utility:
            return None
        t = state.relative_time
        target = self._target_utility(t)

        # Acceptable set = outcomes at or above target (our aspiration). Among
        # them, offer the one the opponent likes most (efficient concession).
        k = self._count_at_least(target)
        if k <= 0:
            k = 1
        choice = self._best_for_opponent(k)

        # End game: also keep the opponent's best gift to us on the table as a
        # focal point they have already shown they will offer.
        if t >= self.endgame_time and self._best_received is not None:
            if float(self.ufun(self._best_received)) > self._reserve:
                if self._opp_score(self._best_received) >= self._opp_score(choice):
                    return self._best_received
        return choice

    def _count_at_least(self, target: float) -> int:
        """Number of (descending) candidate utilities >= target, via bisection."""
        lo, hi = 0, len(self._cand_utils)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cand_utils[mid] >= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _best_for_opponent(self, k: int) -> Outcome:
        """Pick the opponent-favourite among the top-k acceptable outcomes,
        breaking ties toward the highest utility *for us*.

        ``cands`` is sorted by our utility (descending), so the first candidate
        attaining the best opponent score is also the best-for-us among the
        opponent-equivalent options — a strictly advantage-optimal, deterministic
        choice (no random jitter, which only added variance)."""
        cands = self.rational_outcomes[:k]
        if len(cands) == 1:
            return cands[0]
        best_o, best_s = cands[0], self._opp_score(cands[0])
        for o in cands[1:]:
            s = self._opp_score(o)
            if s > best_s + 1e-12:
                best_o, best_s = o, s
        return best_o

    # ------------------------------------------------------- opponent model
    def _update_opponent_model(self, offer: Outcome) -> None:
        if not self._issues:
            return
        if self._first_offer is None:
            self._first_offer = offer  # the opponent opens near their ideal point
        step = self._opp_total  # 0-based index of this offer in the sequence
        self._opp_total += 1
        for i, value in enumerate(offer):
            if i >= len(self._opp_counts):
                break
            self._opp_counts[i][value] = self._opp_counts[i].get(value, 0) + 1
            if value not in self._opp_first_step[i]:
                self._opp_first_step[i][value] = step

        if self.ufun is not None:
            u = float(self.ufun(offer))
            if u > self._best_received_util:
                self._best_received_util = u
                self._best_received = offer

        self._recompute_opponent_model()
        self._publish_opponent_ufun()

    def _opp_score(self, outcome: Outcome) -> float:
        """Fast estimated opponent utility of an outcome (mirrors the published
        model's ranking, without rebuilding the ufun object each lookup)."""
        s = 0.0
        for i, value in enumerate(outcome):
            if i >= len(self._opp_val_score):
                break
            s += self._opp_weights[i] * self._opp_val_score[i].get(value, 0.0)
        return s

    def _recompute_opponent_model(self) -> None:
        """Recompute per-issue value utilities and issue weights from the offer
        history.

        value utility  = **first-appearance order**: an opponent concedes from
                         their best outcomes to their worst, so the EARLIER a
                         value first appears in their offers the more they prefer
                         it (never-offered => 0). This tracks their true ranking
                         far better than raw offer frequency (kendall_opt
                         ~0.78 vs ~0.73 empirically). Plus a boost for the value
                         in their very first offer (their ideal point).
        issue weight   = 1 - normalized entropy (concentrated => important)
        """
        self._opp_val_score = []
        self._opp_weights = []
        first = self._first_offer
        n_seen = max(1, self._opp_total)
        for i, issue in enumerate(self._issues):
            counts = self._opp_counts[i]
            first_step = self._opp_first_step[i]
            all_vals = list(issue.all)
            n = len(all_vals)
            eps = 1e-6
            ideal = first[i] if (first is not None and i < len(first)) else None

            if not counts:
                self._opp_val_score.append({
                    v: (1.0 + eps * k) + (self.first_offer_boost if v == ideal else 0.0)
                    for k, v in enumerate(all_vals)
                })
                self._opp_weights.append(1.0)
                continue

            self._opp_val_score.append({
                v: ((1.0 - first_step[v] / n_seen) if v in first_step else 0.0)
                + eps * k + (self.first_offer_boost if v == ideal else 0.0)
                for k, v in enumerate(all_vals)
            })

            total = sum(counts.values())
            if n > 1 and total > 0:
                entropy = -sum((c / total) * math.log(c / total)
                               for c in counts.values() if c > 0)
                weight = 1.0 - entropy / math.log(n)
            else:
                weight = 1.0
            self._opp_weights.append(max(weight, 1e-3))

        tot = sum(self._opp_weights)
        if tot > 0:
            self._opp_weights = [w / tot for w in self._opp_weights]
        elif self._opp_weights:
            self._opp_weights = [1.0 / len(self._opp_weights)] * len(self._opp_weights)

    def _publish_opponent_ufun(self) -> None:
        """Publish a real LinearAdditiveUtilityFunction under the key the scorer
        reads (``private_info['opponent_ufun']``). Built from the same value
        scores / weights used internally so the published ranking is consistent."""
        if not self._issues or not self._opp_val_score:
            return
        try:
            model = LinearAdditiveUtilityFunction(
                values=[dict(d) for d in self._opp_val_score],
                weights=list(self._opp_weights),
                outcome_space=self._os,
            )
            self.private_info["opponent_ufun"] = model
        except Exception:
            pass
