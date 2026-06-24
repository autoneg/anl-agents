"""SBDANL — ANL 2026 (SBD team) (preference-concealing bilateral negotiation).

Design principle (the right framing): do NOT impose a wiggle. Instead make our bids
hard to OPPONENT-MODEL; any "snake" shape in the utility trace is then an emergent
by-product, not a goal.

How a frequency opponent model reads us:
    It counts which issue-VALUES we offer; values we offer often are assumed to be
    our preferred ones. So to keep tau(opp) (their estimate of us) low we must make
    our offered-value histogram ~uniform, i.e. offer our non-preferred values about
    as often as our preferred ones -- WITHOUT giving away utility.

Key trick — conceal along the iso-utility set:
    Among outcomes whose utility (for us) lies in a thin band above a Boulware target,
    there are usually several different value combinations. We pick the one that most
    FLATTENS our own offered-value histogram (least-used values first), lightly
    blended with the opponent's estimated utility so we still reach agreement. This
    scrambles the opponent's frequency model (lowers tau(opp) -> bigger concealing
    share) while keeping our utility high (protects advantage).

ANL 2026 score = advantage + tau(self) / (tau(self) + tau(opp)), where
    tau = (kendall_tau_b(estimate, truth) + 1) / 2.
We also run our own frequency model of the opponent to raise tau(self); without an
opponent model the concealing point is always zero.

Three required strategy pieces (also the report sections):
  * Concealing bidding : `concealing_bidding_strategy`  (iso-utility value scrambling)
  * Opponent modeling  : `update_opponent_model`        (Smith-style frequency model)
  * Acceptance         : `acceptance_strategy`          (ACnext-like, vs Boulware base)

The legacy utility-wiggle (`wiggle_amp`) is kept for experiments but defaults to 0:
benchmarks showed it costs a lot of advantage for little concealment.
"""

from __future__ import annotations

import math
import random

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun


class SBDANL(SAOCallNegotiator):
    """A preference-concealing negotiator with a wiggling ("snake") concession curve."""

    # ---- tunable hyper-parameters ----
    # Defaults chosen from benchmark_snake.py. Surprising empirical finding: pushing
    # conceal_weight UP (histogram-flattening) BACKFIRES against a frequency opponent
    # model -- it slightly raises tau(opp). The reason is fundamental: utility is ~a
    # function of which values you offer, so any high-utility (good-advantage) bid
    # necessarily exposes your high-utility values. "Greedy toward the opponent within
    # a high-utility band" (conceal_weight=0) already extracts the only *free*
    # concealment. So the real levers are advantage (e_concession) and tau(self).
    # Robustness sweep (benchmark_e_robustness.py) vs a soft->tough panel: higher e is
    # robustly better (no mutual-no-agreement collapse, thanks to the end-game grab)
    # up to ~20-30; panel-mean peaks at e=30 but e>=50 starts losing to reciprocating
    # opponents (TitForTat) and degrades self-play agreement. e=20 keeps ~99% of the
    # peak with far better behaviour vs stubborn/reciprocal agents -> robust default.
    e_concession: float = 20.0   # base concession exponent (>1 => Boulware: concede late)
    band: float = 0.12           # width of the iso-utility band (above target) to bid in
    conceal_weight: float = 0.0   # 0 = greedy-toward-opponent, 1 = pure histogram-flattening
    # ---- opponent model (raises tau_self = how well we estimate THEM) ----
    # Best in A/B (benchmark_snake.py): stability issue-weights + early-offer recency
    # beat the variance/no-recency model, though only by ~0.3% -- tau_self is near an
    # information ceiling (~0.544) because opponents reveal few distinct offers.
    om_weight_mode: str = "stability"  # issue weights: "variance" | "stability" | "blend"
    om_recency: float = 2.0            # >0 emphasizes EARLIER offers (reveal their ideal)
    accept_margin: float = 0.0   # accept if offer >= Boulware base - margin (norm utility)
    # Legacy utility-wiggle (off by default; see module docstring).
    wiggle_amp: float = 0.0      # half-depth of the swing below base (norm-utility units)
    wiggle_cycles: float = 5.0   # number of full oscillations over the negotiation
    wiggle_env: float = 0.5      # envelope exponent: swing depth ~ (1-t)**wiggle_env

    def on_preferences_changed(self, changes):  # noqa: ARG002
        if self.ufun is None:
            return

        # Normalisation bounds for utility -> [0, 1].
        self._umax = float(self.ufun.max())
        self._umin = float(self.ufun.min())
        self._rng = max(self._umax - self._umin, 1e-9)
        self._rv = float(self.ufun.reserved_value)
        self._rv_norm = self._norm(self._rv)

        # Rational outcomes (utility > reserved value), with cached normalized utility,
        # sorted by utility descending.
        scored = [
            (self._norm(float(self.ufun(o))), o)
            for o in self.nmi.outcome_space.enumerate_or_sample()
            if float(self.ufun(o)) > self._rv
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        self._rational = scored                       # list[(u_norm, outcome)]
        self._rational_u = [u for u, _ in scored]     # parallel utility list

        # Opponent model state: per-issue value frequency counts.
        self._issues = list(self.nmi.outcome_space.issues)
        self._issue_values = [
            list(i.all) if i.is_discrete() else None for i in self._issues
        ]
        self._counts: list[dict] = [
            {v: 0.0 for v in vals} if vals is not None else {}
            for vals in self._issue_values
        ]
        # Histogram of the values WE have offered (used to flatten our own footprint).
        self._self_counts: list[dict] = [
            {v: 0.0 for v in vals} if vals is not None else {}
            for vals in self._issue_values
        ]
        # Issue-weight estimation via value stability across consecutive opponent offers.
        self._stable = [0.0] * len(self._issues)
        self._change = [0.0] * len(self._issues)
        self._prev_offer = None
        self._n_seen = 0
        # Seed a non-degenerate estimate so we never accidentally report a constant
        # ufun (which would score tau = 0).
        self._rebuild_opponent_ufun()

    # ----------------------------- main loop --------------------------------
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:  # noqa: ARG002
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if not self._rational:
            # Nothing rational to offer; bail to reservation.
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        if offer is not None:
            self.update_opponent_model(state)
            if self.acceptance_strategy(state):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(
            ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state)
        )

    # --------------------------- acceptance ---------------------------------
    def acceptance_strategy(self, state: SAOState) -> bool:
        offer = state.current_offer
        if offer is None:
            return False
        u = self._norm(float(self.ufun(offer)))
        if u <= self._rv_norm:
            return False
        t = state.relative_time
        # Accept against the (smooth) Boulware base, NOT the wiggling offer target, so
        # that deep offering-troughs never trick us into accepting a poor offer.
        if u >= self._base(t) - self.accept_margin:
            return True
        # ... and grab anything above reservation in the final moments.
        if t > 0.97 and u > self._rv_norm:
            return True
        return False

    # ------------------------ concealing bidding ----------------------------
    def _base(self, t: float) -> float:
        """Smooth Boulware base: stays near 1.0 early, drops to rv as t -> 1."""
        return 1.0 - (1.0 - self._rv_norm) * (t ** self.e_concession)

    def _target(self, t: float) -> float:
        """Wiggling ("ぐわんぐわん") offer target, clipped to [rv, 1].

        The swing is taken DOWNWARD from the base: target = base - amp*(1 - sin),
        so peaks just touch the base (no top-clipping that would flatten the wave)
        and troughs plunge to base - 2*amp, where we genuinely offer low-utility
        (non-preferred) outcomes. The (1-t)**env envelope lets the swings shrink
        near the deadline so we still converge.
        """
        base = self._base(t)
        amp = self.wiggle_amp * ((1.0 - t) ** self.wiggle_env)
        swing = amp * (1.0 - math.sin(2.0 * math.pi * self.wiggle_cycles * t))
        return min(1.0, max(self._rv_norm, base - swing))

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        target = self._target(state.relative_time)
        # Iso-utility band just above the Boulware target (protects advantage).
        candidates = [o for u, o in self._rational if target - 1e-9 <= u <= target + self.band]
        if not candidates:
            candidates = [o for u, o in self._rational if u >= target - 1e-9]
        if not candidates:
            idx = min(
                range(len(self._rational_u)),
                key=lambda i: abs(self._rational_u[i] - target),
            )
            chosen = self._rational[idx][1]
        else:
            chosen = self._select_concealing(candidates)
        self._record_offer(chosen)
        return chosen

    def _select_concealing(self, outs: list[Outcome]) -> Outcome:
        """Pick a band outcome that flattens our offered-value histogram (low
        exposure) while still being attractive to the opponent (for agreement)."""
        # Exposure: how much each candidate reuses values we've already offered a lot.
        expo = [
            sum(self._self_counts[i].get(o[i], 0.0) for i in range(len(o)))
            for o in outs
        ]
        e_lo, e_rng = min(expo), (max(expo) - min(expo)) or 1.0
        ou = self.opponent_ufun
        if ou is not None:
            ov = [float(ou(o)) for o in outs]
            o_lo, o_rng = min(ov), (max(ov) - min(ov)) or 1.0
        else:
            ov, o_lo, o_rng = [0.0] * len(outs), 0.0, 1.0
        cw = self.conceal_weight
        best_i, best_s = 0, -1e18
        for i in range(len(outs)):
            conceal = 1.0 - (expo[i] - e_lo) / e_rng       # high => rarely-used values
            agree = (ov[i] - o_lo) / o_rng                 # high => opponent likes it
            s = cw * conceal + (1.0 - cw) * agree + 1e-6 * random.random()
            if s > best_s:
                best_s, best_i = s, i
        return outs[best_i]

    def _record_offer(self, o: Outcome) -> None:
        for i, v in enumerate(o):
            counts = self._self_counts[i]
            if v in counts:
                counts[v] += 1.0
            elif self._issue_values[i] is not None:
                counts[v] = 1.0

    # ------------------------- opponent modeling ----------------------------
    def update_opponent_model(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None:
            return
        self._n_seen += 1
        # Recency: earlier offers (near their ideal) weigh more when om_recency > 0.
        w = 1.0 + self.om_recency * (1.0 - state.relative_time)
        for idx, val in enumerate(offer):
            if self._issue_values[idx] is None:
                continue
            counts = self._counts[idx]
            counts[val] = counts.get(val, 0.0) + w
        # Issue-weight signal: how often the value at each issue stays put vs changes.
        if self._prev_offer is not None:
            for idx in range(len(offer)):
                if self._issue_values[idx] is None:
                    continue
                if offer[idx] == self._prev_offer[idx]:
                    self._stable[idx] += 1.0
                else:
                    self._change[idx] += 1.0
        self._prev_offer = offer
        self._rebuild_opponent_ufun()

    def _rebuild_opponent_ufun(self) -> None:
        """Frequency opponent model -> LinearAdditiveUtilityFunction estimate.

        value score  = (recency-weighted) offer frequency, normalized per issue;
        issue weight = "variance" of value scores, and/or value "stability" across
                       consecutive offers (issues kept constant => important), per
                       `om_weight_mode`.
        """
        values: dict[str, TableFun] = {}
        names: list[str] = []
        var_w: list[float] = []
        stab_w: list[float] = []
        for idx, issue in enumerate(self._issues):
            vals = self._issue_values[idx]
            if not vals:
                continue
            counts = self._counts[idx]
            mx = max((counts.get(v, 0.0) for v in vals), default=0.0)
            if mx <= 0.0:
                table = {v: 1.0 for v in vals}  # uniform prior before any observation
            else:
                table = {v: counts.get(v, 0.0) / mx for v in vals}
            values[issue.name] = TableFun(table)
            names.append(issue.name)
            scores = list(table.values())
            mean = sum(scores) / len(scores)
            var_w.append(sum((s - mean) ** 2 for s in scores) / len(scores))
            tot = self._stable[idx] + self._change[idx]
            stab_w.append(self._stable[idx] / tot if tot > 0 else 0.0)

        if not values:
            return

        def _normalize(xs: list[float]) -> list[float]:
            s = sum(xs)
            return [x / s for x in xs] if s > 0 else [1.0 / len(xs)] * len(xs)

        if self.om_weight_mode == "variance":
            raw = var_w
        elif self.om_weight_mode == "stability":
            raw = stab_w
        else:  # blend the two normalized signals
            vn, sn = _normalize(var_w), _normalize(stab_w)
            raw = [(a + b) / 2.0 for a, b in zip(vn, sn)]

        total = sum(raw)
        if total <= 0:
            weights = {n: 1.0 / len(names) for n in names}
        else:
            weights = {n: (r / total) for n, r in zip(names, raw)}
        est = LinearAdditiveUtilityFunction(
            values=values,
            weights=weights,
            outcome_space=self.nmi.outcome_space,
        )
        self.private_info["opponent_ufun"] = est

    # ------------------------------ helpers ---------------------------------
    def _norm(self, u: float) -> float:
        return (u - self._umin) / self._rng
