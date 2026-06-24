"""
MirageV145 — v17 hardball (B=64, Nash-anchored) + value-balance concealment (v137)
              + SCENARIO-ADAPTIVE reported opponent model (new in v145).

Thesis (v145): the reported opponent_ufun drives the tau (model-accuracy) half of
the ANL 2026 score, and tau is the only locally-trustworthy, scenario-general lever.
v137's report was a frequency model + a fixed *competitive* tie-breaker (1 - u_me).
That prior is correct in zero-sum scenarios but ACTIVELY WRONG in cooperative ones
(where the opponent prefers what we prefer), silently forfeiting tau there.

v145 replaces it with a report that:
  (1) infers scenario cooperativeness on-line from the opponent's revealed offers
      (do their offers sit ABOVE or BELOW our average utility?), and
  (2) blends the frequency model with a *scenario-adaptive* prior via offer-count
      shrinkage: prior-heavy early / data-starved, frequency-heavy once informed,
      with a small persistent prior floor to correct systematic frequency errors.

Bidding is BYTE-IDENTICAL to v17 (concession, Nash floor, acceptance, _opp_eval,
_select_offer all unchanged) -> zero advantage risk. Only the *reported* model
(private_info["opponent_ufun"]) changes, which affects tau only.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau

from negmas.sao import SAOCallNegotiator, ResponseType, SAOResponse
from negmas.gb.components.genius.models import GHardHeadedFrequencyModel
from negmas.preferences.ops import pareto_frontier, nash_points

if TYPE_CHECKING:
    from negmas.sao import SAOState


class _OppWrap:
    def __init__(self, eval_fn, reserved=0.0):
        self._e = eval_fn
        self.reserved_value = reserved

    def __call__(self, o):
        return self._e(o)


class _InverseUFun:
    def __init__(self, true_ufun):
        self._u = true_ufun
        self.reserved_value = 0.0

    def __call__(self, outcome):
        return 1.0 - float(self._u(outcome))

    def eval(self, outcome):
        return self.__call__(outcome)


class _AdaptiveReport:
    """Reported opponent_ufun (tau lever only; never used for bidding).

    report(o) = alpha * freq_model(o) + (1 - alpha) * prior(o) + 1e-7 * (1 - u_me)
      - alpha   = shrinkage on # opponent offers seen, capped so the prior keeps a
                  small persistent weight (corrects systematic frequency-model error).
      - prior   = coop_w * u_me + (1 - coop_w) * (1 - u_me), where coop_w in [0,1]
                  comes from the negotiator's on-line cooperativeness estimate:
                  competitive (coop_w->0) => 1 - u_me ; cooperative (coop_w->1) => u_me.
      - the 1e-7 term guarantees the reported ranking is never flat (no tau forfeit).
    """

    ALPHA_CAP = 0.85       # max frequency-model weight (>=15% stays on the prior)
    ALPHA_SCALE = 8.0      # offers needed for alpha to approach the cap

    def __init__(self, neg, model, true_ufun):
        self._neg = neg
        self._m = model
        self._u = true_ufun
        self.reserved_value = 0.0

    def eval(self, outcome):
        try:
            base = float(self._m.eval(outcome))
        except Exception:
            base = 0.0
        if base != base:  # NaN
            base = 0.0
        try:
            u_me = float(self._u(outcome))
        except Exception:
            u_me = 0.0
        n = getattr(self._neg, "_n_opp_offers_seen", 0)
        alpha = self.ALPHA_CAP * min(1.0, n / self.ALPHA_SCALE)
        rho = getattr(self._neg, "_coop_rho", 0.0)
        coop_w = 0.5 * (rho + 1.0)
        prior = coop_w * u_me + (1.0 - coop_w) * (1.0 - u_me)
        return alpha * base + (1.0 - alpha) * prior + 1e-7 * (1.0 - u_me)

    def __call__(self, outcome):
        return self.eval(outcome)


class MirageV145(SAOCallNegotiator):
    BOULWARE_EXPONENT = 64.0      # v17 sweep-tuned
    PANIC_TIME = 0.99             # v17 sweep-tuned
    SOFT_ACCEPT_RATIO = 0.95
    NASH_FACTOR = 0.95            # v17 sweep-tuned
    CANDIDATE_CAP = 30
    OUTCOME_SAMPLE_CAP = 200
    SW_WEIGHT_EARLY = 0.0
    SW_WEIGHT_LATE = 3.0
    CONCEAL_WEIGHT = 1.0
    BALANCE_WEIGHT = 2.5      # balance revealed value-frequencies (concealment)
    NASH_RECOMPUTE_EVERY = 10
    DEGENERATE_THRESHOLD = 8

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        rv = float(self.ufun.reserved_value)
        self._rv = rv

        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        self._all_outcomes = outcomes
        scored = [(float(self.ufun(o)), o) for o in outcomes]
        rational = [(u, o) for u, o in scored if u > rv]
        rational.sort(key=lambda x: -x[0])

        self._rational = rational
        self._rational_outcomes = tuple(o for _, o in rational)
        self._rational_utils = tuple(u for u, _ in rational)
        self._max_util = rational[0][0] if rational else 1.0
        self._degenerate = len(rational) <= self.DEGENERATE_THRESHOLD

        # cooperativeness estimate inputs (tau report only)
        all_u = [u for u, _ in scored]
        self._mean_u_me_all = float(np.mean(all_u)) if all_u else 0.5
        self._u_me_spread = (max(all_u) - min(all_u)) if all_u else 1.0
        self._opp_offer_u_me_sum = 0.0
        self._opp_offer_count = 0
        self._coop_rho = 0.0

        try:
            self._n_issues = len(self.nmi.outcome_space.issues)
        except Exception:
            self._n_issues = len(rational[0][1]) if rational else 0

        if len(outcomes) > self.OUTCOME_SAMPLE_CAP:
            keep = set(o for _, o in rational[: self.OUTCOME_SAMPLE_CAP // 2])
            rest = [o for o in outcomes if o not in keep]
            random.shuffle(rest)
            keep.update(rest[: self.OUTCOME_SAMPLE_CAP - len(keep)])
            self._kt_outcomes = list(keep)
        else:
            self._kt_outcomes = outcomes
        self._kt_my_utils = np.array([float(self.ufun(o)) for o in self._kt_outcomes])

        self._sim_value_freq = [dict() for _ in range(self._n_issues)]
        self._sim_n_bids = 0
        self._sim_mf0 = [0] * self._n_issues  # per-issue max freq, recomputed once per _select_offer
        self._best_opp_offer = None
        self._best_opp_offer_u = -float("inf")

        self._opp_model = GHardHeadedFrequencyModel()
        self._opp_model.set_negotiator(self)
        try:
            self._opp_model._initialize()
        except Exception:
            pass
        self._prior_opp = _InverseUFun(self.ufun)
        self._n_opp_offers_seen = 0
        self._opp_rv_estimate = 1.0  # start optimistic; refine downward
        self._report_wrap = _AdaptiveReport(self, self._opp_model, self.ufun)
        self.private_info["opponent_ufun"] = self._report_wrap

        self._nash_u_me = self._compute_nash_u_me()
        self._floor_u_me = max(self._rv, self.NASH_FACTOR * self._nash_u_me)
        self._last_nash_step = 0

    def _opp_eval(self, outcome) -> float:
        if self._n_opp_offers_seen >= 5:
            try:
                return float(self._opp_model.eval(outcome))
            except Exception:
                pass
        return self._prior_opp(outcome)

    def _compute_nash_u_me(self) -> float:
        try:
            opp_wrap = _OppWrap(self._opp_eval, reserved=self._opp_rv_estimate)
            frontier_utils, _ = pareto_frontier(
                ufuns=(self.ufun, opp_wrap),
                outcomes=self._kt_outcomes,
                sort_by_welfare=True,
            )
            if not frontier_utils:
                return self._max_util * 0.5
            nash = nash_points(
                ufuns=(self.ufun, opp_wrap),
                frontier=frontier_utils,
                outcomes=self._kt_outcomes,
            )
            if not nash:
                return max(u[0] for u in frontier_utils)
            (utils, _) = nash[0]
            return float(utils[0])
        except Exception:
            return self._max_util * 0.5

    def __call__(self, state, dest=None):
        offer = state.current_offer
        if self.ufun is None or not self._rational_outcomes:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self._select_offer(state))

        u_off = float(self.ufun(offer))
        if u_off > self._best_opp_offer_u and u_off > self._rv:
            self._best_opp_offer_u = u_off
            self._best_opp_offer = offer

        # cooperativeness estimate (tau report only; does NOT affect bidding)
        self._opp_offer_u_me_sum += u_off
        self._opp_offer_count += 1
        if self._u_me_spread > 1e-9:
            mean_opp = self._opp_offer_u_me_sum / self._opp_offer_count
            self._coop_rho = max(-1.0, min(1.0,
                (mean_opp - self._mean_u_me_all) / (0.5 * self._u_me_spread)))

        self._update_opponent(state, offer)

        # Update opp rv estimate: min model-utility they've offered, with floor 0
        try:
            opp_u_of_their_offer = self._opp_eval(offer)
            if 0.001 <= opp_u_of_their_offer < self._opp_rv_estimate:
                self._opp_rv_estimate = opp_u_of_their_offer
        except Exception:
            pass

        if (state.step - self._last_nash_step) >= self.NASH_RECOMPUTE_EVERY:
            self._nash_u_me = self._compute_nash_u_me()
            self._floor_u_me = max(self._rv, self.NASH_FACTOR * self._nash_u_me)
            self._last_nash_step = state.step

        if self._should_accept(state, offer):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return SAOResponse(ResponseType.REJECT_OFFER, self._select_offer(state))

    def _aspiration_at(self, t):
        floor = self._floor_u_me
        return self._max_util - (self._max_util - floor) * (t ** self.BOULWARE_EXPONENT)

    def _should_accept(self, state, offer):
        u = float(self.ufun(offer))
        if u <= self._rv:
            return False
        t = state.relative_time
        asp = self._aspiration_at(t)
        if u >= asp * self.SOFT_ACCEPT_RATIO:
            return True
        if t >= self.PANIC_TIME:
            return True
        return False

    def _sw_weight(self, t):
        return self.SW_WEIGHT_EARLY + (self.SW_WEIGHT_LATE - self.SW_WEIGHT_EARLY) * t

    def _simulated_eval(self, outcome, extra=None):
        if self._n_issues == 0:
            return 0.5
        total = 0.0
        per_issue = 1.0 / self._n_issues
        for i in range(self._n_issues):
            freq = self._sim_value_freq[i].get(outcome[i], 0)
            if extra is not None and extra[i] == outcome[i]:
                freq += 1
            max_freq = self._sim_mf0[i]
            if extra is not None:
                extra_count = self._sim_value_freq[i].get(extra[i], 0) + 1
                max_freq = max(max_freq, extra_count)
            norm = (freq / max_freq) if max_freq > 0 else 1.0
            total += per_issue * norm
        return total

    def _conceal_score(self, candidate):
        if self._sim_n_bids < 2:
            return sum(self._sim_value_freq[i].get(candidate[i], 0) for i in range(self._n_issues))
        sim_utils = np.array([self._simulated_eval(o, extra=candidate) for o in self._kt_outcomes])
        if sim_utils.std() < 1e-9 or self._kt_my_utils.std() < 1e-9:
            return 0.0
        try:
            tau, _ = kendalltau(sim_utils, self._kt_my_utils)
            if np.isnan(tau):
                return 0.0
            return float(tau)
        except Exception:
            return 0.0

    def _select_offer(self, state):
        t = state.relative_time

        # Degenerate: bid the max-min outcome from round 1 (using inverse prior)
        if self._degenerate:
            best = max(self._rational, key=lambda x: min(x[0], self._opp_eval(x[1])))
            return best[1]

        if t >= self.PANIC_TIME and self._best_opp_offer is not None:
            best = self._best_opp_offer
            for i in range(self._n_issues):
                v = best[i]
                self._sim_value_freq[i][v] = self._sim_value_freq[i].get(v, 0) + 1
            self._sim_n_bids += 1
            return best

        asp = self._aspiration_at(t)
        candidates = [(u, o) for u, o in self._rational if u >= asp - 1e-9]
        if not candidates:
            top_k = max(1, len(self._rational_outcomes) // 10)
            candidates = [(self._rational_utils[i], self._rational_outcomes[i]) for i in range(top_k)]

        if len(candidates) > self.CANDIDATE_CAP:
            candidates = random.sample(candidates, self.CANDIDATE_CAP)

        sw_w = self._sw_weight(t)
        # cache per-issue max frequency once per step (reused by _conceal_score /
        # _simulated_eval and the balance term below; was recomputed ~thousands of times)
        self._sim_mf0 = [max(d.values()) if d else 0 for d in self._sim_value_freq]
        best = None
        best_score = float("inf")
        for u_me, c in candidates:
            conceal = self._conceal_score(c)
            opp_u = self._opp_eval(c)
            sw = u_me + opp_u
            # balance: penalise repeating values we've already revealed often -> flattens
            # our revealed value distribution -> opponent's frequency model learns less.
            bal = 0.0
            for i in range(self._n_issues):
                mx = self._sim_mf0[i]
                if mx > 0:
                    bal += self._sim_value_freq[i].get(c[i], 0) / mx
            bal = bal / max(1, self._n_issues)
            score = self.CONCEAL_WEIGHT * conceal - sw_w * sw + self.BALANCE_WEIGHT * bal
            if score < best_score:
                best_score = score
                best = c

        if best is None:
            best = candidates[0][1]

        for i in range(self._n_issues):
            v = best[i]
            self._sim_value_freq[i][v] = self._sim_value_freq[i].get(v, 0) + 1
        self._sim_n_bids += 1
        return best

    def _update_opponent(self, state, offer):
        try:
            partner_id = getattr(state, "current_proposer", None) or "opponent"
            self._opp_model.on_partner_proposal(
                state=state, partner_id=partner_id, offer=offer
            )
            self._n_opp_offers_seen += 1
        except Exception:
            pass
        self.private_info["opponent_ufun"] = self._report_wrap
