"""AgentTokyoV11 — v10 custom-model deal-closer, refined (no recency, early_w=0.12).

SELF-CONTAINED single file (no cross-module imports) so it loads however the grader
packages it (it is imported as a package submodule, e.g. team_xxx.agent_tokyo_v3).

Tokyo v1/v2 sit at the "hardball-lite" end of the aggression axis (high Nash floor,
moderate Boulware) and time out ~30-45% of negotiations. v3 occupies the OPPOSITE,
agreeable end on purpose, to give the roster a genuinely different strategic bet:

  * close almost everything (target BOA/MAP-like ~95-99% agreement) — every timeout
    is a reservation-value zero, so on TrueSkill an agent that consistently lands
    positive, low-variance outcomes can out-rank a higher-but-spikier one;
  * lower Nash-anchored floor (NASH_FACTOR=0.72), gentle fixed concession (beta=2),
    early panic (0.88), and an ABSOLUTE acceptance threshold so any clearly-good offer
    is taken rather than haggled away;
  * but it KEEPS the deal-closer edge: among bids clearing its (lower) floor it still
    offers the opponent's favourite via the online Bayesian model, and it reports that
    model (hardened against degeneracy) for the Kendall/tau half of the score.

Never accepts/offers below the reservation value. Stateless across negotiations; no
disk/global/class-level writes; no print statements; outcome materialisation capped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from negmas.sao import SAOCallNegotiator, ResponseType, SAOResponse
from negmas.gb.components.genius.models import GScalableBayesianModel
from negmas.preferences.ops import pareto_frontier, nash_points
from negmas.preferences.base_ufun import BaseUtilityFunction

if TYPE_CHECKING:  # pragma: no cover
    from negmas.sao import SAOState


class _ReportedOppUfun(BaseUtilityFunction):
    """Thin callable ufun reported to the grader for the Kendall-tau (tau) bonus."""

    def __init__(self, eval_fn):
        super().__init__()
        self._eval_fn = eval_fn
        self.reserved_value = 0.0

    def eval(self, outcome):
        if outcome is None:
            return 0.0
        try:
            return float(self._eval_fn(outcome))
        except Exception:
            return 0.0

    def __call__(self, outcome):
        return self.eval(outcome)


class _OppForNash:
    def __init__(self, eval_fn, reserved=0.0):
        self._eval_fn = eval_fn
        self.reserved_value = reserved

    def __call__(self, outcome):
        try:
            return float(self._eval_fn(outcome))
        except Exception:
            return 0.0



class _CustomKendallModel:
    """Custom opponent model tuned for Kendall-tau ranking accuracy: per-issue value
    frequencies (concentration-weighted issues) with a mild recency weight and an
    early-appearance bonus (values offered early = their favourites)."""

    def __init__(self):
        self._neg = None; self._ni = 0; self._counts = None; self._first = None; self._t = 0

    def set_negotiator(self, neg):
        self._neg = neg

    def _initialize(self):
        try:
            self._ni = len(self._neg.nmi.outcome_space.issues)
        except Exception:
            self._ni = 0
        self._counts = [dict() for _ in range(self._ni)]
        self._first = [dict() for _ in range(self._ni)]

    def on_partner_proposal(self, state, partner_id, offer):
        if offer is None or self._counts is None:
            return
        self._t += 1
        w = 1.0
        for i in range(min(self._ni, len(offer))):
            v = offer[i]
            self._counts[i][v] = self._counts[i].get(v, 0.0) + w
            if v not in self._first[i]:
                self._first[i][v] = self._t

    def _issue_weight(self, i):
        c = self._counts[i]
        if not c:
            return 1.0 / max(1, self._ni)
        tot = sum(c.values())
        return (max(c.values()) / tot) if tot > 0 else 0.0

    def eval(self, outcome):
        if outcome is None or not self._counts:
            return 0.0
        ws = [self._issue_weight(i) for i in range(self._ni)]
        sw = sum(ws) or 1.0
        tot = 0.0
        for i in range(min(self._ni, len(outcome))):
            c = self._counts[i]
            mx = max(c.values()) if c else 0.0
            base = (c.get(outcome[i], 0.0) / mx) if mx > 0 else 0.5
            # early-appearance bonus: their early-offered values rank higher
            fa = self._first[i].get(outcome[i])
            early = (1.0 / (1.0 + 0.1 * (fa - 1))) if fa else 0.5
            tot += (ws[i] / sw) * (0.88 * base + 0.12 * early)
        return tot

    def __call__(self, o):
        return self.eval(o)


class AgentTokyoV11(SAOCallNegotiator):
    # --- concession / acceptance (AGREEABLE: closes a lot) ---
    CONCESSION_BETA = 2.0       # gentle, fixed
    PANIC_TIME = 0.88           # start securing a deal early
    ACCEPT_RATIO = 0.92         # accept at >= 92% of current aspiration
    ABS_ACCEPT_FRAC = 0.72      # also accept any offer >= this fraction of max util ...
    ABS_ACCEPT_AFTER = 0.5      # ... once past this point in time
    # --- advantage guardrail (lower floor than v1/v2 -> more concession room) ---
    NASH_FACTOR = 0.72
    NASH_RECOMPUTE_EVERY = 10
    NASH_OUTCOME_CAP = 1500
    # --- opponent model / bidding ---
    MODEL_WARMUP = 3
    MODEL_BLEND_FULL = 8
    BID_POOL_CAP = 400
    LEARNING_RATE = 0.1
    OUTCOME_CAP = 100000

    def _materialize_outcomes(self):
        sp = self.nmi.outcome_space
        try:
            card = sp.cardinality
        except Exception:
            card = float("inf")
        try:
            if card != float("inf") and card <= self.OUTCOME_CAP:
                return list(sp.enumerate_or_sample())
            return list(sp.sample(int(self.OUTCOME_CAP), with_replacement=False,
                                  fail_if_not_enough=False))
        except Exception:
            try:
                return list(sp.enumerate_or_sample())
            except Exception:
                return []

    # ---------------------------------------------------------------- setup
    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        rv = float(self.ufun.reserved_value)
        self._rv = rv

        outcomes = self._materialize_outcomes()
        self._all_outcomes = outcomes
        scored = [(float(self.ufun(o)), o) for o in outcomes]
        rational = [(u, o) for u, o in scored if u > rv]
        rational.sort(key=lambda x: -x[0])

        self._rational = rational
        self._max_util = rational[0][0] if rational else 1.0

        self._pool = rational[: self.BID_POOL_CAP]
        self._pool_my_u = (
            np.array([u for u, _ in self._pool]) if self._pool else np.zeros(0)
        )

        try:
            self._n_issues = len(self.nmi.outcome_space.issues)
        except Exception:
            self._n_issues = len(rational[0][1]) if rational else 0

        self._opp_model = _CustomKendallModel()
        try:
            self._opp_model.set_negotiator(self)
        except Exception:
            pass
        try:
            self._opp_model._initialize()
        except Exception:
            pass
        self._n_opp_offers_seen = 0
        self._best_opp_offer = None
        self._best_opp_offer_u = -float("inf")

        self._reported = _ReportedOppUfun(self._report_eval)
        self.private_info["opponent_ufun"] = self._reported

        self._nash_outcomes = self._build_nash_outcomes()
        self._nash_u_me = self._compute_nash_u_me()
        self._floor_u_me = max(self._rv, self.NASH_FACTOR * self._nash_u_me)
        self._last_nash_step = 0
        self._planned_offer_u = None

    def _build_nash_outcomes(self):
        outs = self._all_outcomes
        if len(outs) <= self.NASH_OUTCOME_CAP:
            return outs
        half = self.NASH_OUTCOME_CAP // 2
        by_me = [o for _, o in self._rational[:half]]
        worst = sorted(self._rational, key=lambda x: x[0])[:half]
        by_opp = [o for _, o in worst]
        seen = set()
        merged = []
        for o in by_me + by_opp:
            if o not in seen:
                seen.add(o)
                merged.append(o)
        return merged if merged else outs

    # ---------------------------------------------------------------- opp model
    def _opp_eval(self, outcome) -> float:
        if outcome is None:
            return 0.0
        n = self._n_opp_offers_seen
        if n < self.MODEL_WARMUP:
            return 1.0 - float(self.ufun(outcome))
        try:
            model_val = float(self._opp_model.eval(outcome))
        except Exception:
            return 1.0 - float(self.ufun(outcome))
        if not np.isfinite(model_val):
            return 1.0 - float(self.ufun(outcome))
        if n >= self.MODEL_BLEND_FULL:
            return model_val
        w = n / float(self.MODEL_BLEND_FULL)
        return w * model_val + (1.0 - w) * (1.0 - float(self.ufun(outcome)))

    def _report_eval(self, outcome) -> float:
        base = self._opp_eval(outcome)
        try:
            tie = 1.0 - float(self.ufun(outcome))
        except Exception:
            tie = 0.0
        return base + 1e-6 * tie

    def _compute_nash_u_me(self) -> float:
        try:
            opp = _OppForNash(self._opp_eval, reserved=0.0)
            frontier_utils, _ = pareto_frontier(
                ufuns=(self.ufun, opp),
                outcomes=self._nash_outcomes,
                sort_by_welfare=True,
            )
            if not frontier_utils:
                return self._max_util * 0.5
            pts = nash_points(
                ufuns=(self.ufun, opp),
                frontier=frontier_utils,
                outcomes=self._nash_outcomes,
            )
            if not pts:
                return max(u[0] for u in frontier_utils)
            return float(pts[0][0][0])
        except Exception:
            return self._max_util * 0.5

    # ---------------------------------------------------------------- main loop
    def __call__(self, state, dest=None):
        offer = state.current_offer
        if self.ufun is None or not self._rational:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is not None:
            self._observe(state, offer)

        if (state.step - self._last_nash_step) >= self.NASH_RECOMPUTE_EVERY:
            self._nash_u_me = self._compute_nash_u_me()
            self._floor_u_me = max(self._rv, self.NASH_FACTOR * self._nash_u_me)
            self._last_nash_step = state.step

        my_offer = self._select_offer(state)

        if offer is not None and self._should_accept(state, offer):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer)

    def _observe(self, state, offer):
        u = float(self.ufun(offer))
        if u > self._best_opp_offer_u and u > self._rv:
            self._best_opp_offer_u = u
            self._best_opp_offer = offer
        try:
            partner_id = getattr(state, "current_proposer", None) or "opponent"
            self._opp_model.on_partner_proposal(
                state=state, partner_id=partner_id, offer=offer
            )
            self._n_opp_offers_seen += 1
        except Exception:
            pass

    # ---------------------------------------------------------------- concession
    def _aspiration_at(self, t) -> float:
        floor = self._floor_u_me
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        return self._max_util - (self._max_util - floor) * (t ** self.CONCESSION_BETA)

    def _should_accept(self, state, offer) -> bool:
        u = float(self.ufun(offer))
        if u <= self._rv:
            return False
        t = state.relative_time
        asp = self._aspiration_at(t)
        if u >= asp * self.ACCEPT_RATIO:
            return True
        # absolute acceptance: take any clearly-good offer in the second half.
        if t >= self.ABS_ACCEPT_AFTER and u >= self.ABS_ACCEPT_FRAC * self._max_util:
            return True
        if self._planned_offer_u is not None and u >= self._planned_offer_u - 1e-9:
            return True
        if t >= self.PANIC_TIME:
            return True
        return False

    # ---------------------------------------------------------------- bidding
    def _select_offer(self, state):
        t = state.relative_time
        pool = self._pool
        if not pool:
            return self._rational[0][1] if self._rational else None

        if t >= self.PANIC_TIME and self._best_opp_offer is not None:
            self._planned_offer_u = self._best_opp_offer_u
            return self._best_opp_offer

        asp = self._aspiration_at(t)
        idx = int(np.searchsorted(-self._pool_my_u, -(asp - 1e-9), side="right"))
        if idx <= 0:
            idx = 1
        acceptable = pool[:idx]

        best_u, best_o, best_score = None, None, -float("inf")
        for u_me, o in acceptable:
            s = self._opp_eval(o)
            if s > best_score or (s == best_score and (best_u is None or u_me > best_u)):
                best_score, best_u, best_o = s, u_me, o
        if best_o is None:
            best_u, best_o = pool[0]
        self._planned_offer_u = best_u
        return best_o
