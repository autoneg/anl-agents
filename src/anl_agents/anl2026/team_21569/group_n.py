"""
DecepTor: A Deceptive Negotiating Agent for ANL 2026
CS 451 — Spring 2026
Authors: Cetin Hosafci, Mert Altintas
Ozyegin University, Department of Computer Science

Architecture: BOA (Bidding · Opponent-modeling · Acceptance)
Novel contributions:
  1. Utility-equivalent bid swaps (deception layer)
  2. Adaptive Boulware exponent via DANS classification
  3. Composite four-condition acceptance strategy
"""

from __future__ import annotations

import random
from typing import Optional

from negmas import SAONegotiator, SAOResponse, SAOState
from negmas.outcomes import Outcome
from negmas.preferences import UtilityFunction
from negmas.sao import ResponseType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_E: float = 3.0          # default Boulware exponent
E_MIN: float = 0.2              # most Boulware (slow concede)
E_MAX: float = 5.0              # most Conceder (fast concede)
RESERVATION_MARGIN: float = 0.05
EMA_ALPHA: float = 0.2          # EMA weight for concession tracker
DECEPTION_BASE: float = 0.15
DECEPTION_SLOPE: float = 0.10
DECEPTION_CAP: float = 0.25
DECEPTION_FLOOR: float = 0.92   # deceptive bid must be >= 0.92 * alpha(t)
EPSILON_U: float = 0.02         # utility-equivalence window
RANK_DELTA: int = 1             # minimum rank displacement for deceptive bid
NASH_TOP_K: int = 200           # candidates for Nash approximation
INV_FREQ_WINDOW: int = 5        # sliding window for concession rate


# ===========================================================================
# Opponent Model — GSmithFrequencyModel with inverse-change-frequency weights
# ===========================================================================
class GSmithFrequencyModel:
    """Preference estimator using issue-value counting with
    inverse-change-frequency issue weights (Tunali et al. 2017)."""

    def __init__(self, issues: list[str], values: dict[str, list]):
        self.issues = issues
        self.values = values
        # count[issue][value] = number of times opponent offered this value
        self.count: dict[str, dict] = {i: {v: 0 for v in values[i]} for i in issues}
        # track previous offer to detect changes
        self._prev_offer: Optional[Outcome] = None
        self.changes: dict[str, int] = {i: 0 for i in issues}
        self._n_updates: int = 0

    def update(self, offer: Outcome) -> None:
        """Register a new opponent offer."""
        if offer is None:
            return
        self._n_updates += 1
        for idx, issue in enumerate(self.issues):
            val = offer[idx] if isinstance(offer, tuple) else offer.get(issue)
            if val is not None:
                self.count[issue][val] = self.count[issue].get(val, 0) + 1
                if self._prev_offer is not None:
                    prev_val = (
                        self._prev_offer[idx]
                        if isinstance(self._prev_offer, tuple)
                        else self._prev_offer.get(issue)
                    )
                    if prev_val != val:
                        self.changes[issue] += 1
        self._prev_offer = offer

    def _issue_weight(self, issue: str) -> float:
        """Inverse-change-frequency weight (Tunali et al. 2017, Eq. 6)."""
        denom = sum(1.0 / (1.0 + self.changes[i]) for i in self.issues)
        return (1.0 / (1.0 + self.changes[issue])) / denom if denom > 0 else 1.0 / len(self.issues)

    def _value_score(self, issue: str, val) -> float:
        """Normalised frequency score for a value on an issue."""
        total = sum(self.count[issue].values())
        if total == 0:
            n = len(self.values[issue])
            return 1.0 / n if n > 0 else 0.0
        return self.count[issue].get(val, 0) / total

    def estimate(self, offer: Outcome) -> float:
        """Return estimated opponent utility for *offer* (Eq. 7)."""
        if offer is None:
            return 0.0
        score = 0.0
        for idx, issue in enumerate(self.issues):
            val = offer[idx] if isinstance(offer, tuple) else offer.get(issue)
            score += self._issue_weight(issue) * self._value_score(issue, val)
        return score


# ===========================================================================
# DANS Behaviour Tracker
# ===========================================================================
class DANSMove:
    FORTUNATE = "Fortunate"
    SELFISH = "Selfish"
    CONCESSION = "Concession"
    NICE = "Nice"
    SILENT = "Silent"
    UNFORTUNATE = "Unfortunate"


class BehaviorTracker:
    """Classify each opponent move per the DANS framework
    (Hindriks et al. 2011) and track smoothed concession rate."""

    def __init__(self):
        self.history: list[str] = []
        self._smooth_delta: float = 0.0
        self._prev_own_u: Optional[float] = None
        self._prev_opp_u: Optional[float] = None
        self._prev_t: float = 0.0
        self._window: list[float] = []

    def classify(
        self,
        delta_own: float,
        delta_opp_est: float,
    ) -> str:
        """Return DANS label for a single move."""
        EPS = 1e-4
        up_own = delta_own > EPS
        dn_own = delta_own < -EPS
        up_opp = delta_opp_est > EPS
        dn_opp = delta_opp_est < -EPS

        if up_own and up_opp:
            return DANSMove.FORTUNATE
        if up_own and dn_opp:
            return DANSMove.SELFISH
        if dn_own and up_opp:
            return DANSMove.CONCESSION
        if abs(delta_own) <= EPS and up_opp:
            return DANSMove.NICE
        if abs(delta_own) <= EPS and abs(delta_opp_est) <= EPS:
            return DANSMove.SILENT
        if dn_own and dn_opp:
            return DANSMove.UNFORTUNATE
        return DANSMove.SILENT

    def update(
        self,
        own_u: float,
        opp_est_u: float,
        t: float,
    ) -> str:
        """Update tracker and return the classified move label."""
        delta_own = own_u - self._prev_own_u if self._prev_own_u is not None else 0.0
        delta_opp = opp_est_u - self._prev_opp_u if self._prev_opp_u is not None else 0.0

        label = self.classify(delta_own, delta_opp)
        self.history.append(label)

        # smoothed concession rate (sliding window, Eq. 8)
        dt = max(t - self._prev_t, 1e-6)
        inst_delta = delta_own / dt
        self._window.append(inst_delta)
        if len(self._window) > INV_FREQ_WINDOW:
            self._window.pop(0)
        self._smooth_delta = sum(self._window) / len(self._window)

        self._prev_own_u = own_u
        self._prev_opp_u = opp_est_u
        self._prev_t = t
        return label

    @property
    def smooth_delta(self) -> float:
        return self._smooth_delta

    def dominant_type(self) -> str:
        if not self.history:
            return DANSMove.SILENT
        return max(set(self.history), key=self.history.count)


# ===========================================================================
# Deception Layer — utility-equivalent bid swaps
# ===========================================================================
class DeceptionLayer:
    """
    Pre-computes, for each utility level, an honest set H_u and a deceptive
    set D_u.  At selection time it samples a deceptive bid with probability
    p_dec(t) and an honest one otherwise.
    """

    def __init__(
        self,
        outcomes: list[Outcome],
        ufun: UtilityFunction,
        issues: list[str],
        values: dict[str, list],
        issue_weights: dict[str, float],
    ):
        self.outcomes = outcomes
        self.ufun = ufun
        self.issues = issues
        self.values = values
        self.issue_weights = issue_weights  # true weights (sorted descending)
        # sorted issue names by weight (highest first)
        self._sorted_issues: list[str] = sorted(
            issues, key=lambda i: issue_weights.get(i, 0), reverse=True
        )
        # cache: utility → (H_u list, D_u list)
        self._cache: dict[float, tuple[list, list]] = {}

    def _u(self, outcome: Outcome) -> float:
        v = self.ufun(outcome)
        return float(v) if v is not None else 0.0

    def _is_deceptive(self, outcome: Outcome) -> bool:
        """A bid is deceptive if high-weight issues take non-optimal values."""
        if not self._sorted_issues:
            return False
        top_issue = self._sorted_issues[0]
        idx = self.issues.index(top_issue)
        val = outcome[idx] if isinstance(outcome, tuple) else outcome.get(top_issue)
        # optimal value = the one with the highest ufun contribution
        best_val = self.values[top_issue][0]  # assume sorted best-first
        return val != best_val

    def _build_sets(self, alpha: float) -> tuple[list, list]:
        key = round(alpha, 3)
        if key in self._cache:
            return self._cache[key]

        H, D = [], []
        for o in self.outcomes:
            u = self._u(o)
            if abs(u - alpha) <= EPSILON_U or u >= alpha:
                if self._is_deceptive(o):
                    D.append(o)
                else:
                    H.append(o)
        self._cache[key] = (H, D)
        return H, D

    def pdec(self, t: float) -> float:
        """Deceptive bid probability schedule (Eq. 9)."""
        return min(DECEPTION_CAP, DECEPTION_BASE + DECEPTION_SLOPE * t)

    def select_bid(
        self,
        alpha: float,
        t: float,
        opp_model: GSmithFrequencyModel,
        reservation: float,
    ) -> Optional[Outcome]:
        H, D = self._build_sets(alpha)

        # filter: never go below reservation + margin
        floor_u = reservation + RESERVATION_MARGIN
        H = [o for o in H if self._u(o) >= floor_u]
        D = [o for o in D if self._u(o) >= max(floor_u, DECEPTION_FLOOR * alpha)]

        if not H and not D:
            return None  # fallback: caller will handle

        use_deception = D and random.random() < self.pdec(t)

        if use_deception:
            # pick the deceptive bid that maximises Kendall distance to true ranking
            # proxy: pick bid whose top-issue value is ranked worst
            best = max(D, key=lambda o: self._u(o))
            return best
        else:
            if not H:
                return random.choice(D)
            # honest: pick bid closest to Nash point (max opp estimated utility)
            return max(H, key=lambda o: opp_model.estimate(o))


# ===========================================================================
# Main Agent
# ===========================================================================
class GroupN(SAONegotiator):
    """DecepTor — bilateral negotiating agent for ANL 2026."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_negotiation_start(self, state: SAOState) -> None:  # type: ignore[override]
        nmi = self.nmi
        self._issues: list[str] = [str(i) for i in nmi.issues]
        self._values: dict[str, list] = {
            str(i): list(i.values) for i in nmi.issues  # type: ignore[attr-defined]
        }
        self._outcomes: list[Outcome] = list(nmi.outcome_space.enumerate_or_sample(max_cardinality=2000))  # type: ignore[attr-defined]

        # utility function
        self._ufun = self.ufun
        self._reservation: float = float(self._ufun.reserved_value or 0.0)

        # max achievable utility
        self._umax: float = max(float(self._ufun(o) or 0) for o in self._outcomes)

        # sort outcomes by own utility descending
        self._outcomes.sort(key=lambda o: float(self._ufun(o) or 0), reverse=True)

        # issue weights (heuristic: from ufun if linear-additive)
        self._issue_weights: dict[str, float] = self._extract_weights()

        # modules
        self._opp_model = GSmithFrequencyModel(self._issues, self._values)
        self._behavior_tracker = BehaviorTracker()
        self._deception_layer = DeceptionLayer(
            self._outcomes,
            self._ufun,
            self._issues,
            self._values,
            self._issue_weights,
        )

        # adaptive exponent
        self._e: float = INITIAL_E
        self._ema_s: float = 0.0

        # concession history for EMA
        self._prev_opp_est: Optional[float] = None

        # Nash approximation cache
        self._nash_u: Optional[float] = None

    # ------------------------------------------------------------------
    # BOA — Bidding
    # ------------------------------------------------------------------
    def _extract_weights(self) -> dict[str, float]:
        """Try to read weights from a LinearAdditiveUtilityFunction; fallback to uniform."""
        try:
            w = {}
            for i in self._issues:
                w[i] = float(getattr(self._ufun, "weights", {}).get(i, 1.0 / len(self._issues)))
            return w
        except Exception:
            return {i: 1.0 / len(self._issues) for i in self._issues}

    def _alpha(self, t: float) -> float:
        """Aspiration level at time t (Faratin et al. 1998, Eq. 3)."""
        r = self._reservation + RESERVATION_MARGIN
        span = self._umax - r
        return r + (1.0 - t ** (1.0 / max(self._e, 1e-6))) * span

    def _adapt_e(self) -> None:
        """Adjust Boulware exponent based on opponent estimated utility EMA."""
        opp_est = self._opp_model.estimate(self._last_opp_offer) if hasattr(self, "_last_opp_offer") and self._last_opp_offer is not None else 0.0
        if self._prev_opp_est is not None:
            delta = opp_est - self._prev_opp_est
            self._ema_s = (1 - EMA_ALPHA) * self._ema_s + EMA_ALPHA * delta
            if self._ema_s > 0.05:
                # opponent improving for us → stay Boulware
                self._e = min(self._e * 1.05, E_MAX)
            elif self._ema_s < -0.05:
                # opponent moving away → concede faster
                self._e = max(self._e * 0.95, E_MIN)
        self._prev_opp_est = opp_est

    def _propose(self, state: SAOState) -> Outcome:
        t = state.relative_time
        alpha = self._alpha(t)
        bid = self._deception_layer.select_bid(
            alpha, t, self._opp_model, self._reservation
        )
        if bid is not None:
            return bid
        # fallback: best outcome above floor
        floor = self._reservation + RESERVATION_MARGIN
        for o in self._outcomes:
            if float(self._ufun(o) or 0) >= floor:
                return o
        return self._outcomes[0]

    # ------------------------------------------------------------------
    # BOA — Acceptance (Algorithm 1)
    # ------------------------------------------------------------------
    def _nash_utility(self) -> float:
        """Approximate Nash utility from top-K outcomes."""
        if self._nash_u is not None:
            return self._nash_u
        r = self._reservation
        best = 0.0
        for o in self._outcomes[:NASH_TOP_K]:
            ua = float(self._ufun(o) or 0)
            ub = self._opp_model.estimate(o)
            nash_prod = (ua - r) * ub
            if nash_prod > best:
                best = nash_prod
                self._nash_u = ua
        return self._nash_u or 0.0

    def _should_accept(self, offer: Outcome, state: SAOState) -> bool:
        t = state.relative_time
        ua = float(self._ufun(offer) or 0)
        alpha = self._alpha(t)
        r = self._reservation

        # Guard: never accept below reservation
        if ua < r:
            return False

        # Condition A: ACNext guarantee
        if ua >= alpha:
            return True

        # Condition B: Nash-point check (final 10%)
        if t >= 0.90 and ua >= 0.95 * self._nash_utility():
            return True

        # Condition C: opponent reversing concession late
        if self._behavior_tracker.smooth_delta < 0 and t > 0.70:
            return True

        return False

    # ------------------------------------------------------------------
    # SAO interface
    # ------------------------------------------------------------------
    def __call__(self, state: SAOState, **kwargs) -> SAOResponse:  # type: ignore[override]
        offer = state.current_offer

        # initialise on first call if on_negotiation_start wasn't triggered
        if not hasattr(self, "_opp_model"):
            self.on_negotiation_start(state)

        if offer is not None:
            self._last_opp_offer = offer
            self._opp_model.update(offer)
            opp_est = self._opp_model.estimate(offer)
            own_u = float(self._ufun(offer) or 0)
            self._behavior_tracker.update(own_u, opp_est, state.relative_time)
            self._adapt_e()
            # reset Nash cache (model updated)
            self._nash_u = None

            if self._should_accept(offer, state):
                return SAOResponse(response=ResponseType.ACCEPT_OFFER)

        my_bid = self._propose(state)
        return SAOResponse(response=ResponseType.REJECT_OFFER, outcome=my_bid)
