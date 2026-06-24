from collections import Counter, defaultdict, deque
from math import pow
from scipy.stats import chi2_contingency

from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import ResponseType, SAOState, SAOResponse
from negmas.sao.negotiators.modular import BOANegotiator
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.components.acceptance import ACNext
from negmas.gb.components.genius.models import GSmithFrequencyModel


class DistributionFrequencyModel:
    """Tunali et al. 2017 — chi-squared window opponent model."""

    def __init__(self, k=5, alpha=10, beta=5, gamma=0.25):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._issues = []
        self._issue_values = {}
        self._issue_weights = {}
        self._value_counts = {}
        self._cur = []
        self._prev = []
        self._n = 0

    def initialize(self, nmi):
        try:
            issues = nmi.outcome_space.issues
            n = len(issues)
            self._issues = list(range(n))
            self._issue_values = {}
            self._issue_weights = {i: 1.0 / n for i in range(n)}
            self._value_counts = {}
            for i, iss in enumerate(issues):
                vals = list(iss.all)
                self._issue_values[i] = vals
                self._value_counts[i] = {v: 0 for v in vals}
            self._cur, self._prev, self._n = [], [], 0
        except Exception:
            pass

    def _freq(self, i, v, offers):
        n = len(self._issue_values.get(i, [None]))
        if not offers:
            return 1.0 / max(n, 1)
        return (1 + sum(1 for o in offers if o[i] == v)) / (n + len(offers))

    def _importance(self, i, v):
        counts = self._value_counts.get(i, {})
        if not counts or self._n == 0:
            return 1.0 / max(len(self._issue_values.get(i, [None])), 1)
        num = (1 + counts.get(v, 0)) ** self.gamma
        den = max((1 + c) ** self.gamma for c in counts.values())
        return num / (den + 1e-10)

    def update(self, offer, t):
        if not self._issues:
            return
        try:
            for i in self._issues:
                self._value_counts[i][offer[i]] = self._value_counts[i].get(offer[i], 0) + 1
            self._n += 1
            self._cur.append(offer)
            if len(self._cur) >= self.k:
                if self._prev:
                    self._reweight(t)
                self._prev = self._cur.copy()
                self._cur = []
        except Exception:
            pass

    def _reweight(self, t):
        if not self._prev or not self._cur:
            return
        unchanged, conceded = [], False
        for i in self._issues:
            vals = self._issue_values.get(i, [])
            if len(vals) < 2:
                continue
            s = max(len(self._prev), 1)
            op = [max(1, int(self._freq(i, v, self._prev) * s)) for v in vals]
            oc = [max(1, int(self._freq(i, v, self._cur) * s)) for v in vals]
            try:
                changed = chi2_contingency([op, oc])[1] <= 0.05
            except Exception:
                changed = False
            if not changed:
                unchanged.append(i)
            else:
                w = self._issue_weights.get(i, 0)
                ep = sum(w * self._importance(i, o[i]) for o in self._prev) / len(self._prev)
                ec = sum(w * self._importance(i, o[i]) for o in self._cur) / len(self._cur)
                if ec < ep:
                    conceded = True
        if conceded and unchanged and len(unchanged) < len(self._issues):
            delta = self.alpha * (1.0 - t ** self.beta)
            for i in unchanged:
                self._issue_weights[i] = self._issue_weights.get(i, 0) + delta
            total = sum(self._issue_weights.values())
            if total > 0:
                for i in self._issues:
                    self._issue_weights[i] /= total

    def estimate(self, outcome):
        if not self._issues:
            return 0.5
        try:
            n = max(len(self._issues), 1)
            return min(1.0, max(0.0, sum(
                self._issue_weights.get(i, 1.0 / n) * self._importance(i, outcome[i])
                for i in self._issues
            )))
        except Exception:
            return 0.5

    def ready(self):
        return self._n >= self.k


class OwnOfferPrivacyModel:
    """Cheap in-memory estimate of how readable our own offer stream is."""

    def __init__(self):
        self._issue_values = []
        self._value_counts = defaultdict(Counter)
        self._offers = deque(maxlen=20)
        self._utils = deque(maxlen=20)

    def initialize(self, issue_values):
        self._issue_values = [set(vs) for vs in issue_values]
        self._value_counts = defaultdict(Counter)
        self._offers.clear()
        self._utils.clear()

    def update(self, outcome, utility_norm):
        if outcome is None:
            return
        self._offers.append(outcome)
        self._utils.append(max(0.0, min(1.0, float(utility_norm))))
        for i, v in enumerate(outcome):
            self._value_counts[i][v] += 1

    def score(self, outcome, utility_norm):
        if outcome is None or not self._issue_values:
            return 1.0
        n_issues = max(len(outcome), 1)
        repeats = 0.0
        balance = 0.0
        novelty = 1.0
        if self._offers:
            last = self._offers[-1]
            novelty = sum(a != b for a, b in zip(outcome, last)) / n_issues
            repeats = 1.0 if outcome in self._offers else 0.0
        for i, v in enumerate(outcome):
            values = max(len(self._issue_values[i]) if i < len(self._issue_values) else 1, 1)
            total = max(len(self._offers), 1)
            balance += 1.0 - min(1.0, self._value_counts[i].get(v, 0) / max(total / values, 1.0))
        balance /= n_issues

        signal_noise = 1.0
        if len(self._utils) >= 3:
            vals = list(self._utils)[-3:] + [max(0.0, min(1.0, float(utility_norm)))]
            diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            clean_down = all(d <= 0.015 for d in diffs) and any(d < -0.005 for d in diffs)
            clean_up = all(d >= -0.015 for d in diffs) and any(d > 0.005 for d in diffs)
            signal_noise = 0.20 if clean_down or clean_up else 1.0

        return max(0.0, min(1.0,
            0.35 * novelty + 0.35 * balance + 0.15 * (1.0 - repeats) + 0.15 * signal_noise
        ))


class AgentNexus(BOANegotiator):
    """ANL 2026 """

    MAX_OUTCOMES  = 1000
    MIN_BID_FLOOR = 0.75

    def __init__(self, *args, **kwargs):
        offering = TimeBasedOfferingPolicy()
        kwargs |= dict(acceptance=ACNext(offering), offering=offering, model=GSmithFrequencyModel())
        super().__init__(*args, **kwargs)

        self._outcomes: list[tuple[float, Outcome]] = []
        self._issue_values: list[set] = []
        self._max_u    = 1.0
        self._last_bid: Outcome | None = None
        self._own_hist = deque(maxlen=20)
        self._own_n    = 0
        self._opp_cnts = defaultdict(Counter)
        self._opp_hist = deque(maxlen=15)
        self._opp_n    = 0

        self._model       = DistributionFrequencyModel()
        self._ufun_cache  = {}
        self._rational    = []
        self._recv_utils  = deque(maxlen=50)
        self._opposition  = 0.5
        self._round_cache = {}
        self._sel_cache: Outcome | None = None
        self._sel_step    = -1
        self._val_freq    = defaultdict(Counter)
        self._opp_totals  = {}
        self._own_privacy = OwnOfferPrivacyModel()
        self._profile = {}
        self._all_outcome_count = 0
        self._target_hold_end = 0.60
        self._target_mid_end = 0.88
        self._target_floor_ratio = 0.68
        self._golden_accept = 0.965
        self._privacy_weight = 0.16
        self._tiny_high_frac = 0.05
        self._tiny_min_high = 2
        self._rescue_stall_time = 0.90

    def on_preferences_changed(self, changes) -> None:
        super().on_preferences_changed(changes or [])
        if self.ufun is None or self.nmi is None:
            return
        try:
            self._model.initialize(self.nmi)
        except Exception:
            pass

        self._ufun_cache = {}
        outcomes, issue_values = [], []
        all_outcome_count = 0
        for o in self.nmi.outcome_space.enumerate_or_sample():
            all_outcome_count += 1
            u = float(self.ufun(o))
            if u > float(self.ufun.reserved_value):
                outcomes.append((u, o))
                self._ufun_cache[o] = u
            for i, v in enumerate(o):
                if i >= len(issue_values):
                    issue_values.append(set())
                issue_values[i].add(v)

        self._outcomes     = sorted(outcomes, key=lambda x: x[0], reverse=True)
        self._issue_values = issue_values
        self._all_outcome_count = all_outcome_count
        self._max_u        = self._outcomes[0][0] if self._outcomes else 1.0
        self._rational     = [o for _, o in self._outcomes[:self.MAX_OUTCOMES]]

        self._opp_cnts    = defaultdict(Counter)
        self._opp_hist.clear()
        self._recv_utils.clear()
        self._opp_n       = 0
        self._own_n       = 0
        self._sel_cache   = None
        self._sel_step    = -1
        self._round_cache = {}
        self._val_freq    = defaultdict(Counter)
        self._opp_totals  = {}
        self._own_hist.clear()
        self._last_bid    = None
        self._own_privacy.initialize(self._issue_values)

        high = sum(1 for u, _ in self._outcomes if u >= self._max_u * 0.80)
        self._opposition = 1.0 - high / max(len(self._outcomes), 1)
        self._configure_profile()

        # Set initial opponent model for Concealing scoring
        # (will be updated dynamically after each round — see _update_boa_model)
        try:
            self.private_info["opponent_ufun"] = LambdaMultiFun(
                f=lambda o: self._model.estimate(o)
            )
        except Exception:
            pass

    def _u(self, o) -> float:
        if o not in self._ufun_cache:
            try:
                self._ufun_cache[o] = float(self.ufun(o))
            except Exception:
                self._ufun_cache[o] = 0.0
        return self._ufun_cache[o]

    @property
    def acceptance_strategy(self):
        return self._acceptance

    @property
    def concealing_bidding_strategy(self):
        return self._offering

    def update_opponent_model(self, *args, **kwargs):
        return self._model

    def _hardliner(self) -> float:
        if 'h' not in self._round_cache:
            self._round_cache['h'] = self._calc_hardliner()
        return self._round_cache['h']

    def _concession_rate(self) -> float:
        if 'c' not in self._round_cache:
            self._round_cache['c'] = self._calc_concession()
        return self._round_cache['c']

    def _calc_concession(self) -> float:
        if len(self._opp_hist) < 4:
            return 0.0
        h = list(self._opp_hist)
        mid = len(h) // 2
        return sum(x[1] for x in h[mid:]) / (len(h) - mid) - sum(x[1] for x in h[:mid]) / mid

    def _calc_hardliner(self) -> float:
        if len(self._opp_hist) < 4:
            return 0.0
        c  = self._calc_concession()
        ru = sum(x[1] for x in self._opp_hist) / len(self._opp_hist)
        lw = 1.0 - min(1.0, ru / self._max_u)
        nc = 1.0 if c <= 0.02 * self._max_u else 0.0
        return max(0.0, min(1.0, 0.65 * lw + 0.35 * nc))

    def __call__(self, state: SAOState, dest=None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        t     = float(state.relative_time)
        step  = int(state.step)
        self._round_cache = {}

        if offer is not None:
            ou      = self._u(offer)
            is_bait = self._is_bait(ou, t)       # check BEFORE updating history
            self._track_opp(offer, t)
            if not is_bait and self._should_accept(offer, ou, t, step):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self._cached_select(t, step)
        bid = self._enforce_floor(bid, t)
        self._track_own(bid)
        self._update_boa_model(state, dest)

        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    def _update_boa_model(self, state: SAOState, dest) -> None:
        try:
            super().__call__(state, dest)
        except Exception:
            pass

        # Override with superior DistributionFrequencyModel when ready
        if self._model.ready():
            try:
                self.private_info["opponent_ufun"] = LambdaMultiFun(
                    f=lambda o: self._model.estimate(o)
                )
            except Exception:
                pass

    def _configure_profile(self) -> None:
        rational_total = max(len(self._outcomes), 1)
        total = max(self._all_outcome_count, rational_total)
        reserved = float(self.ufun.reserved_value) if self.ufun else 0.0
        densities = {
            "d95": sum(1 for u, _ in self._outcomes if u >= self._max_u * 0.95) / total,
            "d90": sum(1 for u, _ in self._outcomes if u >= self._max_u * 0.90) / total,
            "d80": sum(1 for u, _ in self._outcomes if u >= self._max_u * 0.80) / total,
            "d70": sum(1 for u, _ in self._outcomes if u >= self._max_u * 0.70) / total,
        }
        issue_count = len(self._issue_values)
        avg_values = (
            sum(len(vs) for vs in self._issue_values) / issue_count
            if issue_count else 0.0
        )
        rv_ratio = reserved / max(self._max_u, 0.01)
        self._profile = {
            "outcomes": total,
            "rational": len(self._outcomes),
            "rv_ratio": rv_ratio,
            "issues": issue_count,
            "avg_values": avg_values,
            **densities,
        }

        dense_top = densities["d90"] >= 0.18
        sparse_top = densities["d90"] <= 0.06
        self._target_hold_end = 0.56 if dense_top else 0.66 if sparse_top else 0.60
        self._target_mid_end = 0.86 if dense_top else 0.90 if sparse_top else 0.88
        self._target_floor_ratio = 0.65 if dense_top else 0.72 if sparse_top else 0.68
        self._golden_accept = 0.955 if dense_top else 0.975 if sparse_top else 0.965
        if rv_ratio >= 0.82:
            self._golden_accept = min(0.99, self._golden_accept + 0.01)
            self._target_floor_ratio = max(self._target_floor_ratio, rv_ratio)
        self._privacy_weight = 0.20 if dense_top else 0.10 if sparse_top else 0.16
        self._tiny_high_frac = 0.035 if sparse_top else 0.07 if dense_top else 0.05
        self._tiny_min_high = 1 if total <= 10 else 2
        self._rescue_stall_time = 0.93 if sparse_top else 0.88 if dense_top else 0.90

    # ──────────────────────────────────────────────────────────
    # ACCEPTANCE — 6-layer strict guard
    # ──────────────────────────────────────────────────────────

    def _should_accept(self, offer: Outcome, ou: float, t: float, step: int) -> bool:
        reserved = float(self.ufun.reserved_value)
        if ou <= reserved:
            return False

        if t < 0.65 and ou >= self._max_u * self._golden_accept:
            return True

        # Strict tiered guards — force opponent to make better offers
        if t < 0.60 and ou < self._max_u * 0.90:
            return False
        if t < 0.75 and ou < self._max_u * 0.85:
            return False
        if t < 0.85 and ou < self._max_u * 0.80:
            return False
        if t < 0.95 and ou < self._max_u * 0.72:
            return False

        # Extreme domain rescue
        if self._tiny_domain() and (t > 0.97 or (t > self._rescue_stall_time and self._stalled())):
            c = self._conflict_offer()
            if c is not None and ou >= self._u(c) * 0.98:
                return True

        # ACcombi MAXW — Baarslag et al. 2010, Table 4 top performer (0.675)
        if t >= 0.99 and len(self._opp_hist) >= 3:
            ws = 2.0 * t - 1.0
            wu = [u for wt, u, _, _ in self._opp_hist if wt >= ws]
            if not wu:
                wu = [u for _, u, _, _ in self._opp_hist]
            if ou >= max(wu):
                return True

        # Scenario-adapted target utility
        if ou >= self._target(t):
            return True

        # ACnext-style: accept if better than our planned bid × margin
        planned = self._cached_select(t, step)
        if planned is not None and ou >= self._u(planned) * (0.995 - 0.06 * t):
            return True

        # Absolute last resort — never leave with nothing
        if t > 0.97 and ou >= max(reserved, self._max_u * 0.60):
            return True

        return False

    def _is_bait(self, ou: float, t: float) -> bool:
        """Wongkamjan et al. 2025 — reject suspiciously generous early offers."""
        if len(self._recv_utils) < 5 or t > 0.75:
            return False
        avg = sum(list(self._recv_utils)[-5:]) / 5
        return ou > avg * 1.20 and t < 0.55

    def _target(self, t: float) -> float:
        """Ultra-Boulware: 95% until t=0.60-0.72, then very slow concession."""
        t        = max(0.0, min(1.0, t))
        reserved = float(self.ufun.reserved_value)
        floor    = max(reserved, self._max_u * 0.68)
        floor    = max(floor, self._max_u * self._target_floor_ratio)
        p1       = min(self._target_hold_end + self._opposition * 0.12, 0.72)
        if t < p1:
            return max(floor, self._max_u * 0.95)
        if t < self._target_mid_end:
            c = pow((t - p1) / max(self._target_mid_end - p1, 0.01), 3.0)
            return self._max_u * 0.95 - self._max_u * 0.18 * c
        c = pow((t - self._target_mid_end) / max(1.0 - self._target_mid_end, 0.01), 1.2)
        return max(floor, self._max_u * 0.77 - self._max_u * 0.09 * c)

    # ──────────────────────────────────────────────────────────
    # BID SELECTION — Aggressive + maximally concealing
    # ──────────────────────────────────────────────────────────

    def _cached_select(self, t: float, step: int) -> Outcome | None:
        if self._sel_step != step:
            self._sel_cache = self._select_bid(t, step)
            self._sel_step  = step
        return self._sel_cache

    def _select_bid(self, t: float, step: int) -> Outcome | None:
        if not self._outcomes:
            return None

        if self._tiny_domain() and (t > 0.97 or (t > self._rescue_stall_time and self._stalled())):
            return self._conflict_offer()

        # Nash bid when completely stalled vs hardliner
        if self._hardliner() > 0.65 and t > 0.72 and self._model.ready():
            nb = self._nash_bid()
            if nb is not None:
                return nb

        target     = self._target(t)
        candidates = [(u, o) for u, o in self._outcomes if u >= target]
        if not candidates:
            floor = max(
                float(self.ufun.reserved_value) if self.ufun else 0.0,
                self._max_u * 0.65,
            )
            candidates = [(u, o) for u, o in self._outcomes if u >= floor] or self._outcomes

        reserved = float(self.ufun.reserved_value)

        if t < 0.50:
            ow, aw, pw, mw = 0.78, 0.04, 0.02, self._privacy_weight
        elif t < 0.80:
            ow, aw, pw, mw = 0.65, 0.14, 0.08, max(0.08, self._privacy_weight - 0.03)
        else:
            ow, aw, pw, mw = 0.52, 0.26, 0.14, max(0.04, self._privacy_weight - 0.08)

        h  = self._hardliner()
        c  = self._concession_rate()
        mc = {}

        ranked = []
        for u, o in candidates:
            os = (u - reserved) / max(self._max_u - reserved, 0.01)
            xs = self._model.estimate(o) if self._model.ready() else self._opp_util(o)
            if o not in mc:
                mc[o] = self._mask(o)
            ls  = self._own_privacy.score(o, os)
            ms  = 0.55 * mc[o] + 0.45 * ls
            acs = self._acc_lik(xs, o, t, h, c)
            ps  = 0.45 * os * xs + 0.35 * (0.5 * os + 0.5 * xs) + 0.20 * (1.0 - abs(os - xs))
            ds  = max(0.0, (t - 0.85) / 0.15) * xs if t >= 0.85 else 0.0
            nb  = os * xs * 0.04
            score = ow * os + aw * acs + pw * ps + mw * ms + 0.03 * ds + nb
            ranked.append((score, u, o, ms))

        ranked.sort(reverse=True)
        top   = ranked[0][1]
        slack = 0.015 if t < 0.55 else 0.040
        near  = [x for x in ranked if top - x[1] <= self._max_u * slack]
        if len(near) >= 5:
            near.sort(key=lambda x: (x[3], x[0]), reverse=True)
            near = near[:min(10, len(near))]
        else:
            near = ranked[:max(1, min(7, len(ranked)))]
        return near[(step * 7) % len(near)][2]

    def _nash_bid(self) -> Outcome | None:
        reserved = float(self.ufun.reserved_value)
        best, bid = -1.0, None
        for u, o in self._outcomes[:200]:
            os = (u - reserved) / max(self._max_u - reserved, 0.01)
            try:
                xs = self._model.estimate(o) if self._model.ready() else self._opp_util(o)
            except Exception:
                xs = self._opp_util(o)
            n = os * xs
            if n > best:
                best, bid = n, o
        return bid

    def _enforce_floor(self, bid: Outcome | None, t: float) -> Outcome | None:
        if bid is None or self.ufun is None or t >= 0.88:
            return bid
        floor = self._max_u * self.MIN_BID_FLOOR
        if self._u(bid) >= floor:
            return bid
        return next(
            (o for o in self._rational if self._u(o) >= floor),
            self._rational[0] if self._rational else bid
        )

    # ──────────────────────────────────────────────────────────
    # OPPONENT MODEL + SCORING
    # ──────────────────────────────────────────────────────────

    def _acc_lik(self, xs, outcome, t, h, c) -> float:
        t  = max(0.0, min(1.0, t))
        sm = self._similarity(outcome)
        return max(0.0, min(1.0,
            0.58 * xs + 0.22 * sm + 0.18 * t
            + h * sm * (0.18 + 0.32 * t)
            - max(0.0, c) * (1.0 - t) * 0.10
        ))

    def _similarity(self, outcome: Outcome) -> float:
        if not self._opp_hist:
            return 0.5
        n = len(outcome)
        if n == 0:
            return 0.5
        recent = list(self._opp_hist)[-min(3, len(self._opp_hist)):]
        best   = 0.0
        for _, _, _, prev in recent:
            if not prev:
                continue
            sim = sum(a == b for a, b in zip(outcome, prev)) / n
            if sim > best:
                best = sim
        return best

    def _opp_util(self, outcome: Outcome) -> float:
        if self._opp_n == 0:
            return 0.5
        ws, wts = [], []
        for i, v in enumerate(outcome):
            counts = self._opp_cnts[i]
            total  = self._opp_totals.get(i, 0)
            if total == 0:
                continue
            nv = max(1, len(self._issue_values[i]))
            ws.append((counts[v] + 1) / (total + nv))
            wts.append(max(counts.values()) / total)
        if not ws:
            return 0.5
        wsum = sum(wts)
        return sum(s * w for s, w in zip(ws, wts)) / wsum if wsum > 0 else sum(ws) / len(ws)

    # ──────────────────────────────────────────────────────────
    # MASKING
    # ──────────────────────────────────────────────────────────

    def _mask(self, outcome: Outcome) -> float:
        if self._last_bid is None:
            return 1.0
        n = len(outcome)
        if n == 0:
            return 1.0
        novelty  = sum(a != b for a, b in zip(outcome, self._last_bid)) / n
        repeated = 1.0 if outcome in self._own_hist else 0.0
        reuse    = self._val_reuse(outcome)
        return 0.55 * novelty + 0.30 * (1.0 - reuse) + 0.15 * (1.0 - repeated)

    def _val_reuse(self, outcome: Outcome) -> float:
        nr = len(self._own_hist)
        if nr == 0:
            return 0.0
        reused = total = 0
        for i, v in enumerate(outcome):
            reused += self._val_freq.get(i, {}).get(v, 0)
            total  += nr
        return reused / total if total else 0.0

    # ──────────────────────────────────────────────────────────
    # TRACKING
    # ──────────────────────────────────────────────────────────

    def _track_opp(self, offer: Outcome, t: float) -> None:
        self._opp_n += 1
        for i, v in enumerate(offer):
            self._opp_cnts[i][v] += 1
            self._opp_totals[i] = self._opp_totals.get(i, 0) + 1
        ou = self._u(offer)
        try:
            self._model.update(offer, t)
        except Exception:
            pass
        xs = self._model.estimate(offer) if self._model.ready() else self._opp_util(offer)
        self._opp_hist.append((t, ou, xs, offer))
        self._recv_utils.append(ou)

    def _track_own(self, bid: Outcome | None) -> None:
        if bid is None:
            return
        self._own_n += 1
        self._last_bid = bid
        self._own_hist.append(bid)
        reserved = float(self.ufun.reserved_value) if self.ufun else 0.0
        un = (self._u(bid) - reserved) / max(self._max_u - reserved, 0.01)
        self._own_privacy.update(bid, un)
        self._val_freq = defaultdict(Counter)
        for prev in self._own_hist:
            for i, v in enumerate(prev):
                self._val_freq[i][v] += 1

    # ──────────────────────────────────────────────────────────
    # EXTREME DOMAIN HANDLING
    # ──────────────────────────────────────────────────────────

    def _tiny_domain(self) -> bool:
        if len(self._outcomes) <= 3:
            return True
        high = [u for u, _ in self._outcomes if u >= self._max_u * 0.70]
        return len(high) <= max(self._tiny_min_high, int(self._tiny_high_frac * len(self._outcomes)))

    def _stalled(self) -> bool:
        if not (self._own_n >= 5 or self._opp_n >= 5):
            return False
        rv = sum(x[1] for x in list(self._opp_hist)[-4:]) / min(4, len(self._opp_hist)) if self._opp_hist else 0.0
        return rv < self._max_u * 0.30 or self._hardliner() > 0.55 or self._concession_rate() <= self._max_u * 0.02

    def _conflict_offer(self) -> Outcome | None:
        if not self._outcomes:
            return None
        best, best_o = float("-inf"), self._outcomes[-1][1]
        h, c = self._hardliner(), self._concession_rate()
        for u, o in self._outcomes[:200]:
            os  = u / self._max_u if self._max_u else u
            xs  = self._model.estimate(o) if self._model.ready() else self._opp_util(o)
            acs = self._acc_lik(xs, o, 1.0, h, c)
            s   = 0.45 * min(os, xs) + 0.35 * (1.0 - abs(os - xs)) + 0.20 * acs
            if s > best:
                best, best_o = s, o
        return best_o


class AgentNexusNoBait(AgentNexus):
    def _is_bait(self, ou: float, t: float) -> bool:
        return False


class AgentNexusNoSuperCall(AgentNexus):
    def _update_boa_model(self, state: SAOState, dest) -> None:
        if self._model.ready():
            try:
                self.private_info["opponent_ufun"] = LambdaMultiFun(
                    f=lambda o: self._model.estimate(o)
                )
            except Exception:
                pass


class AgentNexusGoldenAccept(AgentNexus):
    def _configure_profile(self) -> None:
        super()._configure_profile()
        self._golden_accept = min(self._golden_accept, 0.945)


class AgentNexusSoftAccept(AgentNexus):
    def _should_accept(self, offer: Outcome, ou: float, t: float, step: int) -> bool:
        reserved = float(self.ufun.reserved_value)
        if ou <= reserved:
            return False

        if t < 0.55 and ou >= self._max_u * min(self._golden_accept, 0.955):
            return True
        if t < 0.60 and ou < self._max_u * 0.87:
            return False
        if t < 0.75 and ou < self._max_u * 0.82:
            return False
        if t < 0.85 and ou < self._max_u * 0.77:
            return False
        if t < 0.95 and ou < self._max_u * 0.70:
            return False

        if self._tiny_domain() and (t > 0.97 or (t > self._rescue_stall_time and self._stalled())):
            c = self._conflict_offer()
            if c is not None and ou >= self._u(c) * 0.98:
                return True

        if t >= 0.99 and len(self._opp_hist) >= 3:
            ws = 2.0 * t - 1.0
            wu = [u for wt, u, _, _ in self._opp_hist if wt >= ws]
            if not wu:
                wu = [u for _, u, _, _ in self._opp_hist]
            if ou >= max(wu):
                return True

        if ou >= self._target(t) * 0.985:
            return True

        planned = self._cached_select(t, step)
        if planned is not None and ou >= self._u(planned) * (0.985 - 0.07 * t):
            return True

        if t > 0.965 and ou >= max(reserved, self._max_u * 0.58):
            return True

        return False


class AgentNexusK7Model(AgentNexus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = DistributionFrequencyModel(k=7)


class AgentNexusK10Model(AgentNexus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = DistributionFrequencyModel(k=10)


Agentnexus = AgentNexus
agentnexus = AgentNexus

try:
    from negmas.sao import SAOCallNegotiator

    SAOCallNegotiator.register(AgentNexus)
except Exception:
    pass
