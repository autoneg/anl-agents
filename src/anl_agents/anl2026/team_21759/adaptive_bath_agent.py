"""
AdaptiveBathNegotiator for ANL2026 / NegMAS SAO negotiations.

Design philosophy (hybrid of MiCRO + MiCRO_OM + ShadowBath):
  1. MiCRO-inspired descending aspiration: start high, monotonically concede to a
     safety floor. NO late rise (too risky — opponents may walk or time out).
  2. Adaptive concession exponent e(t) — the speed of descent changes based on
     real-time opponent trajectory classification (Hardliner / Conceder / Regular).
  3. MiCRO_OM-style proposal selection: within an aspiration utility band, pick the
     offer that maximizes weighted self-utility + opponent-model estimate + bonuses.
  4. Robust acceptance: MiCRO base rule (accept iff offer >= our next target),
     softened only near the deadline and when the opponent is clearly conceding.
  5. Keep ShadowBath's strong SFM+DFM opponent model core, plus a new lightweight
     trajectory analyzer (simple slope + variance over a sliding window).

Key references:
  - de Jonge & Zhang (2024) "MiCRO is Near-Optimal" (micro.pdf)
  - de Jonge (2019) "An Analysis of the Linear Bilateral ANAC Domains"
    using the MiCRO strategy (micro2.pdf)
  - Tunalı, O., Aydoğan, R., Sanchez-Anguix, V. (2017)
    "Rethinking Frequency Opponent Modeling" (Distribution-based FM)
  - The user's MiCRO_OM implementation: MiCRO descending schedule +
    opponent-model-based offer selection within the utility band.
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Sequence

try:  # NegMAS 0.15+
    from negmas.sao import SAONegotiator
except Exception:  # older public examples often expose it at top level
    from negmas import SAONegotiator  # type: ignore

try:
    from negmas import ResponseType
except Exception:
    from negmas.sao import ResponseType  # type: ignore


Outcome = Any
EPS = 1e-9


# =========================================================================
# Small helpers
# =========================================================================


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _normalize_positive(values: list[float]) -> list[float]:
    if not values:
        return []
    values = [max(0.0, _safe_float(v)) for v in values]
    s = sum(values)
    if s <= EPS:
        f = 1.0 / len(values)
        return [f for _ in values]
    return [v / s for v in values]


def _clip(x: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def _percentile(sorted_vals: Sequence[float], q: float, default: float = 0.0) -> float:
    if not sorted_vals:
        return default
    n = len(sorted_vals)
    q = _clip(q, 0.0, 1.0)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def _outcome_to_values(outcome: Outcome, issue_names: list[str] | None = None) -> tuple[Any, ...]:
    if outcome is None:
        return tuple()
    if isinstance(outcome, dict):
        if issue_names:
            return tuple(outcome.get(k) for k in issue_names)
        return tuple(outcome[k] for k in sorted(outcome))
    if isinstance(outcome, (tuple, list)):
        return tuple(outcome)
    return (outcome,)


# =========================================================================
# Chi-square helper (for DFM)
# =========================================================================


def _critical_chi_square_95(df: int) -> float:
    table = {
        1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
        6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    z = 1.6448536269514722
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


# =========================================================================
# Opponent Trajectory Analyzer (NEW)
# =========================================================================


class OpponentType(IntEnum):
    UNKNOWN = 0
    CONCEDING = 1      # opponent offers are getting better for us
    HARD_LINEAR = 2    # opponent offers are staying flat / getting worse for us
    REGULAR = 3        # roughly linear
    ERRATIC = 4        # high variance — possibly random strategy


class OpponentTrajectoryAnalyzer:
    """Lightweight opponent trajectory classifier.

    Tracks opponent offer utilities (from *our* perspective) and classifies
    the opponent's behaviour into one of four types:
        CONCEDING   → their offers trend upward (good for us)
        HARD_LINEAR → flat or downward trend (tough opponent)
        REGULAR     → steady linear concession
        ERRATIC     → high variance, possibly random

    The analysis is used by the main agent to adapt its own concession speed.
    """

    def __init__(self, window_size: int = 8, min_samples: int = 4):
        self.window_size = max(4, int(window_size))
        self.min_samples = max(3, int(min_samples))
        self._utils: list[float] = []
        self._classifications: list[OpponentType] = []

    def update(self, my_utility: float) -> None:
        """Append the utility of the opponent's latest offer (from *our* perspective)."""
        self._utils.append(my_utility)

    def classify(self) -> OpponentType:
        """Return current classification based on slope and variance."""
        recent = self._utils[-self.window_size:]
        if len(recent) < self.min_samples:
            return OpponentType.UNKNOWN
        n = len(recent)
        # Simple linear trend: slope of x (time index) vs y (utility)
        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n
        num = den = 0.0
        for i, y in enumerate(recent):
            dx = i - mean_x
            dy = y - mean_y
            num += dx * dy
            den += dx * dx
        slope = num / den if abs(den) > EPS else 0.0
        # Variance around the trend
        var = 0.0
        if den > EPS:
            for i, y in enumerate(recent):
                fitted = mean_y + slope * (i - mean_x)
                var += (y - fitted) ** 2
            var = math.sqrt(var / n)
        else:
            var = sum((y - mean_y) ** 2 for y in recent) / n
            var = math.sqrt(max(EPS, var))

        # Safety: normalise slope to the utility range
        u_range = max(recent) - min(recent) if max(recent) - min(recent) > EPS else 1.0
        norm_slope = slope / u_range  # slope as fraction of observed range

        # Classify
        if var > 0.08:  # high variance → erratic/random behaviour
            cls = OpponentType.ERRATIC
        elif norm_slope > 0.08:  # clearly improving for us → conceding
            cls = OpponentType.CONCEDING
        elif norm_slope < -0.03:  # declining → hardliner
            cls = OpponentType.HARD_LINEAR
        else:
            cls = OpponentType.REGULAR

        self._classifications.append(cls)
        # Keep only recent classifications for smoothing
        if len(self._classifications) > self.window_size:
            self._classifications.pop(0)
        return cls

    @property
    def stable_type(self) -> OpponentType:
        """Majority vote over recent classifications."""
        if not self._classifications:
            return OpponentType.UNKNOWN
        counts: dict[OpponentType, int] = {}
        for c in self._classifications:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.__getitem__)

    def concession_speed(self) -> float:
        """How quickly the opponent's offers are improving (0 = none, 1 = fast)."""
        if len(self._utils) < self.min_samples:
            return 0.0
        recent = self._utils[-min(self.window_size, len(self._utils)):]
        if len(recent) < 2:
            return 0.0
        earliest = recent[0]
        latest = recent[-1]
        spread = latest - earliest
        # Normalise to a rough range: typical utility range ~ 0.4~1.0
        return _clip(spread / 0.5, 0.0, 1.0)

    @property
    def confidence(self) -> float:
        """How confident we are in the classification (0–1)."""
        n = len(self._utils)
        if n < self.min_samples:
            return 0.0
        return _clip((n - self.min_samples) / max(1, self.window_size), 0.0, 1.0)


# =========================================================================
# Opponent model: same SFM + DFM as ShadowBath (proven effective)
# =========================================================================


@dataclass
class OpponentIssueSpace:
    issue_names: list[str] = field(default_factory=list)
    values: list[list[Any]] = field(default_factory=list)

    def ensure_from_outcome(self, outcome_values: tuple[Any, ...]) -> None:
        if not self.issue_names:
            self.issue_names = [f"i{i}" for i in range(len(outcome_values))]
        while len(self.values) < len(outcome_values):
            self.values.append([])
        for i, v in enumerate(outcome_values):
            if v not in self.values[i]:
                self.values[i].append(v)

    @property
    def n_issues(self) -> int:
        return len(self.issue_names) or len(self.values)


class SmithFrequencyOpponentModel:
    """Smith/HardHeaded-style frequency opponent model. Same as ShadowBath."""

    def __init__(self, space: OpponentIssueSpace, no_change_increment: float = 0.05):
        self.space = space
        self.no_change_increment = no_change_increment
        self.value_counts: list[Counter[Any]] = []
        self.stability_scores: list[float] = []
        self.weights: list[float] = []
        self.offers: list[tuple[Any, ...]] = []
        self._resize()

    def _resize(self) -> None:
        n = self.space.n_issues
        while len(self.value_counts) < n:
            self.value_counts.append(Counter())
        while len(self.stability_scores) < n:
            self.stability_scores.append(0.0)
        if len(self.weights) != n:
            self.weights = [1.0 / n for _ in range(n)] if n else []

    def update(self, outcome_values: tuple[Any, ...], relative_time: float = 0.0) -> None:
        self.space.ensure_from_outcome(outcome_values)
        self._resize()
        previous = self.offers[-1] if self.offers else None
        self.offers.append(outcome_values)
        for i, v in enumerate(outcome_values):
            self.value_counts[i][v] += 1
            if previous is not None and i < len(previous) and previous[i] == v:
                self.stability_scores[i] += self.no_change_increment * (1.0 - 0.35 * relative_time)
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        raw: list[float] = []
        for i, counts in enumerate(self.value_counts):
            total = sum(counts.values())
            dominance = (max(counts.values()) / total) if total > 0 else 0.0
            raw.append(1.0 + self.stability_scores[i] + 0.35 * dominance)
        self.weights = _normalize_positive(raw)

    def value_score(self, issue_index: int, value: Any) -> float:
        if issue_index >= len(self.value_counts) or not self.value_counts[issue_index]:
            return 0.5
        counts = self.value_counts[issue_index]
        max_count = max(counts.values())
        return (counts.get(value, 0) + 1.0) / (max_count + 1.0)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        values = _outcome_to_values(outcome, issue_names or self.space.issue_names)
        if not self.weights or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self.weights[i] if i < len(self.weights) else 1.0 / max(1, len(values))
            total += w * self.value_score(i, v)
        return _clip(total, 0.0, 1.0)


class DistributionBasedFrequencyOpponentModel:
    """DFM inspired by Tunalı et al. (2017). Same as ShadowBath."""

    def __init__(
        self,
        space: OpponentIssueSpace,
        window_size: int = 4,
        alpha: float = 0.18,
        beta: float = 1.6,
        gamma: float = 0.70,
    ):
        self.space = space
        self.window_size = max(2, int(window_size))
        self.alpha = max(0.0, alpha)
        self.beta = max(0.1, beta)
        self.gamma = _clip(gamma, 0.05, 1.0)
        self.offers: list[tuple[Any, ...]] = []
        self.value_counts: list[Counter[Any]] = []
        self.weights: list[float] = []
        self._resize()

    def _resize(self) -> None:
        n = self.space.n_issues
        while len(self.value_counts) < n:
            self.value_counts.append(Counter())
        if len(self.weights) != n:
            self.weights = [1.0 / n for _ in range(n)] if n else []

    def update(self, outcome_values: tuple[Any, ...], relative_time: float = 0.0) -> None:
        self.space.ensure_from_outcome(outcome_values)
        self._resize()
        self.offers.append(outcome_values)
        for i, v in enumerate(outcome_values):
            self.value_counts[i][v] += 1
        if len(self.offers) >= 2 * self.window_size and len(self.offers) % self.window_size == 0:
            self._update_issue_weights(relative_time)

    def _value_domain(self, issue_index: int) -> list[Any]:
        if issue_index < len(self.space.values) and self.space.values[issue_index]:
            return list(self.space.values[issue_index])
        if issue_index < len(self.value_counts):
            return list(self.value_counts[issue_index].keys())
        return []

    def _window_counts(self, window: list[tuple[Any, ...]], issue_index: int) -> Counter[Any]:
        c: Counter[Any] = Counter()
        for o in window:
            if issue_index < len(o):
                c[o[issue_index]] += 1
        return c

    def _smoothed_distribution(self, window: list[tuple[Any, ...]], issue_index: int) -> dict[Any, float]:
        domain = self._value_domain(issue_index)
        if not domain:
            return {}
        counts = self._window_counts(window, issue_index)
        denom = len(window) + len(domain)
        return {v: (1.0 + counts.get(v, 0)) / max(EPS, denom) for v in domain}

    def value_score(self, issue_index: int, value: Any) -> float:
        if issue_index >= len(self.value_counts) or not self.value_counts[issue_index]:
            return 0.5
        counts = self.value_counts[issue_index]
        domain = self._value_domain(issue_index)
        if not domain:
            return 0.5
        max_count = max((counts.get(v, 0) for v in domain), default=0)
        numerator = (1.0 + counts.get(value, 0)) ** self.gamma
        denominator = (1.0 + max_count) ** self.gamma
        return _clip(numerator / max(EPS, denominator), 0.0, 1.0)

    def _chi_square_stat(self, prev: dict[Any, float], curr: dict[Any, float], k: int) -> float:
        stat = 0.0
        for v in set(prev) | set(curr):
            expected = max(EPS, prev.get(v, 0.0) * k)
            observed = curr.get(v, 0.0) * k
            stat += (observed - expected) ** 2 / expected
        return stat

    def _expected_issue_utility(self, dist: dict[Any, float], issue_index: int) -> float:
        return sum(self.value_score(issue_index, v) * p for v, p in dist.items())

    def _update_issue_weights(self, relative_time: float) -> None:
        k = self.window_size
        prev_window = self.offers[-2 * k: -k]
        curr_window = self.offers[-k:]
        unchanged_issues: list[int] = []
        concession_detected = False
        n = self.space.n_issues
        for i in range(n):
            prev_dist = self._smoothed_distribution(prev_window, i)
            curr_dist = self._smoothed_distribution(curr_window, i)
            if not prev_dist or not curr_dist:
                continue
            df = max(1, len(set(prev_dist) | set(curr_dist)) - 1)
            stat = self._chi_square_stat(prev_dist, curr_dist, k)
            if stat <= _critical_chi_square_95(df):
                unchanged_issues.append(i)
            else:
                prev_eu = self._expected_issue_utility(prev_dist, i)
                curr_eu = self._expected_issue_utility(curr_dist, i)
                if curr_eu + 1e-5 < prev_eu:
                    concession_detected = True
        if concession_detected and 0 < len(unchanged_issues) < n:
            delta = self.alpha * max(0.0, 1.0 - relative_time ** self.beta)
            raw = list(self.weights)
            for i in unchanged_issues:
                raw[i] += delta
            self.weights = _normalize_positive(raw)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        values = _outcome_to_values(outcome, issue_names or self.space.issue_names)
        if not self.weights or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self.weights[i] if i < len(self.weights) else 1.0 / max(1, len(values))
            total += w * self.value_score(i, v)
        return _clip(total, 0.0, 1.0)


class CombinedFrequencyOpponentModel:
    """Adaptive mixture of SFM and DFM. Same as ShadowBath."""

    def __init__(self, space: OpponentIssueSpace, window_size: int = 4):
        self.space = space
        self.smith = SmithFrequencyOpponentModel(space)
        self.dfm = DistributionBasedFrequencyOpponentModel(space, window_size=window_size)
        self.offers: list[tuple[Any, ...]] = []
        self.window_size = max(2, int(window_size))

    def update(self, outcome: Outcome, relative_time: float = 0.0) -> None:
        values = _outcome_to_values(outcome, self.space.issue_names)
        if not values:
            return
        self.space.ensure_from_outcome(values)
        self.offers.append(values)
        self.smith.update(values, relative_time)
        self.dfm.update(values, relative_time)

    @property
    def dfm_weight(self) -> float:
        n = len(self.offers)
        confidence = _clip((n - self.window_size) / max(1, 3 * self.window_size), 0.0, 1.0)
        return min(0.85, 0.35 + 0.50 * confidence)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        wd = self.dfm_weight
        s = self.smith(outcome, issue_names or self.space.issue_names)
        d = self.dfm(outcome, issue_names or self.space.issue_names)
        return _clip((1.0 - wd) * s + wd * d, 0.0, 1.0)


# =========================================================================
# Opponent utility adapter (for compatibility)
# =========================================================================


class OpponentUtilityAdapter:
    def __init__(self, negotiator: "AdaptiveBathNegotiator"):
        self.negotiator = negotiator
        self.reserved_value = 0.0

    def __call__(self, outcome: Outcome) -> float:
        return self.negotiator.estimate_opponent_utility(outcome)


# =========================================================================
# Main Agent
# =========================================================================


class AdaptiveBathNegotiator(SAONegotiator):
    """ANL2026 negotiator using MiCRO-style descending aspiration + adaptive
    concession + opponent-model-guided offer selection.

    Key changes vs ShadowBathNegotiator:
      - MiCRO-style descending target (no bathtub / late rise);
      - Adaptive concession exponent e(t) based on opponent trajectory;
      - Opponent classification (Hardliner / Conceder / Regular / Erratic);
      - Cleaner, MiCRO-inspired acceptance rule.
    """

    # --- Class-level threshold constants ---
    FLOOR_FRACTION: float = 0.55
    BASE_E: float = 3.0
    HARDLINER_E_DELTA: float = -1.0
    CONCEDER_E_DELTA: float = 1.2
    ERRATIC_E_DELTA: float = 0.3
    EARLY_REJECT_THRESHOLD: float = 0.30
    MID_CONCESSION_CHECK: float = 0.75
    LATE_PHASE_START: float = 0.65
    BEST_HISTORIC_THRESHOLD: float = 0.90
    FINAL_ACCEPT_THRESHOLD: float = 0.95
    STALEMATE_WINDOW: int = 6
    STALEMATE_VARIANCE: float = 0.02
    TFT_PROBABILITY: float = float(os.environ.get("ANL_TFT_PROB", "0.15"))
    BLUFF_PROBABILITY: float = float(os.environ.get("ANL_BLUFF_PROB", "0.04"))
    BLUFF_MAX_TIME: float = 0.60

    def __init__(
        self,
        *args: Any,
        # --- aspiration parameters ---
        # Starting fraction of max utility
        initial_target_fraction: float = 1.0,
        # The final floor as fraction of max (we never go below this)
        floor_fraction: float = 0.55,
        # --- adaptation parameters ---
        # Base "e" exponent for the target curve target = max - (max - floor) * t^e
        # Higher e = stay demanding longer (slower concession)
        base_e: float = 3.0,
        # How much we adjust e for various opponent types (delta added to base_e)
        hardliner_e_delta: float = -1.0,      # concede faster vs Hardliner to find middle ground
        conceder_e_delta: float = 1.2,         # stay firmer vs Conceder — let them come to us
        erratic_e_delta: float = 0.3,          # slightly firmer vs Erratic
        # --- proposal selection ---
        candidate_pool_limit: int = 12000,
        # --- opponent model ---
        window_size: int = 4,
        # --- exploration ---
        rng_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Aspiration
        self.initial_target_fraction = _clip(initial_target_fraction, 0.5, 1.0)
        self.floor_fraction = _clip(floor_fraction, 0.20, 0.70)
        self.base_e = max(0.2, base_e)
        self.hardliner_e_delta = hardliner_e_delta
        self.conceder_e_delta = conceder_e_delta
        self.erratic_e_delta = erratic_e_delta
        self.candidate_pool_limit = max(500, int(candidate_pool_limit))
        self.window_size = max(2, int(window_size))
        self.rng = random.Random(rng_seed)

        # Opponent model (same core as ShadowBath)
        self._space = OpponentIssueSpace()
        self._model = CombinedFrequencyOpponentModel(self._space, window_size=self.window_size)
        self.opponent_model = self._model
        self._opponent_ufun_adapter = OpponentUtilityAdapter(self)

        # Trajectory analyzer (NEW)
        self._trajectory = OpponentTrajectoryAnalyzer(window_size=window_size * 3, min_samples=5)

        # Cached domain info
        self._outcomes: list[Outcome] = []
        self._outcome_values: dict[Outcome, float] = {}
        self._outcome_by_values: dict[tuple, Outcome] = {}
        self._best: Outcome | None = None
        self._min_u = 0.0
        self._max_u = 1.0
        self._reserved = 0.0
        self._current_e = self.base_e  # effective e for current round

        # State tracking
        self._last_proposal: Outcome | None = None
        self._last_proposal_u = 0.0
        self._last_seen_key: tuple[int, tuple[Any, ...]] | None = None
        self._received_offers: list[Outcome] = []
        self._received_utils: list[float] = []
        self._received_keys: list[tuple[Any, ...]] = []
        self._self_offer_keys: list[tuple[Any, ...]] = []
        self._self_offer_counts: Counter[tuple[Any, ...]] = Counter()
        self._offer_counts: Counter[tuple[Any, ...]] = Counter()
        self._recent_self_offers: deque[Outcome] = deque(maxlen=6)
        self._initialized = False
        self._am_first_mover: bool | None = None  # set on first propose/respond

    # ---------- Property for NegMAS compatibility ----------

    @property
    def opponent_ufun(self) -> OpponentUtilityAdapter:
        return self._opponent_ufun_adapter

    # ---------- NegMAS callbacks ----------

    def on_preferences_changed(self, changes: Any) -> None:
        self._initialize_cached_domain(force=True)
        try:
            super().on_preferences_changed(changes)
        except Exception:
            pass

    def respond(self, state: Any, offer: Outcome | str | None = None, source: str = "") -> Any:
        # Adapt to NegMAS version differences
        if isinstance(offer, str) and source == "":
            source = offer
            offer = None
        if offer is None:
            offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._initialize_cached_domain()
        self._record_opponent_offer(offer, state)

        # Detect mover order on first interaction
        if self._am_first_mover is None:
            self._am_first_mover = False  # responding before proposing → second mover

        offer_u = self._u(offer)
        target = self._target_utility(state)
        t = self._relative_time(state)
        # In short negotiations t advances in larger steps, so we scale
        # time-dependent thresholds to match the compressed timeline.
        t_scale = min(1.0, self._n_steps / 50.0) if self._n_steps > 0 else 1.0
        t_effective = _clip(t / t_scale, 0.0, 1.0) if t_scale > EPS else t
        rng = max(EPS, self._max_u - self._reserved)
        floor = self._reserved + 0.01 * rng

        # --- Early phase: reject unless offer is exceptionally good ---
        # Second-mover compensation: when opponent proposes first, they set the
        # anchor.  We relax the early gate slightly so we don't walk away from
        # a workable opening offer just because we didn't get to move first.
        early_threshold = 0.95
        if self._am_first_mover is False:
            early_threshold = 0.88  # more flexible when we move second
        if t_effective < 0.10:
            if offer_u < early_threshold * self._max_u:
                return ResponseType.REJECT_OFFER

        # --- Opponent preference gate: if opponent likes this offer but it's
        #     below our target, reject and wait for them to concede further. ---
        if self.EARLY_REJECT_THRESHOLD < t_effective < 0.85 and offer_u < target - 0.02 * rng:
            if self.estimate_opponent_utility(offer) > 0.55:
                return ResponseType.REJECT_OFFER

        # --- MiCRO base rule: accept if offer ≥ our target ---
        if offer_u + 1e-12 >= target:
            return ResponseType.ACCEPT_OFFER

        # --- Require opponent concession before mid-phase acceptance ---
        # Use both slow (8-sample, reliable) and fast (4-sample, reactive)
        # concession detectors.  If EITHER signals concession, allow acceptance.
        opponent_conceding = (
            self._opponent_has_conceded(min_delta=0.05)
            or self._opponent_conceding_fast(min_delta=0.04)
        )
        if t_effective < self.MID_CONCESSION_CHECK and not opponent_conceding:
            if offer_u + 0.05 * rng < target:
                return ResponseType.REJECT_OFFER

        # --- Mid/late adjustments ---
        if t_effective > self.LATE_PHASE_START:
            urgency = _clip((t_effective - self.LATE_PHASE_START) / 0.35, 0.0, 1.0)
            gap = (0.02 + 0.10 * urgency) * rng

            opp_type = self._trajectory.stable_type
            if opp_type == OpponentType.CONCEDING and self._trajectory.confidence > 0.3:
                gap += 0.05 * rng

            # Time-based conceder: they'll keep conceding on their curve.
            # Accept slightly earlier to lock in a good deal before they
            # realise they're being exploited.
            if opp_type == OpponentType.CONCEDING and self._opponent_is_time_based():
                gap += 0.04 * rng

            if offer_u + gap >= target:
                return ResponseType.ACCEPT_OFFER

        # --- Very end: accept if offer is best we've seen or above floor ---
        # In zero-sum scenarios (NiceOrDie-like), be more stubborn at the end:
        # accepting a poor deal is worse than no deal since opponent gets almost
        # everything and we get almost nothing.
        if self._is_zero_sum_scenario:
            if t_effective > self.BEST_HISTORIC_THRESHOLD and offer_u >= self._floor_utility():
                return ResponseType.ACCEPT_OFFER
        else:
            if t_effective > self.BEST_HISTORIC_THRESHOLD:
                if self._received_utils and offer_u >= max(floor, max(self._received_utils) - 0.015 * rng):
                    return ResponseType.ACCEPT_OFFER
            if t_effective > self.FINAL_ACCEPT_THRESHOLD and offer_u >= self._floor_utility():
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state: Any) -> Outcome:
        if self._am_first_mover is None:
            self._am_first_mover = True  # we are proposing first → first mover
        self._initialize_cached_domain()
        offer = self._select_offer(state)
        if offer is None:
            try:
                offer = self.nmi.random_outcomes(1)[0]
            except Exception:
                offer = self._best
        self._last_proposal = offer
        self._last_proposal_u = self._u(offer) if offer is not None else 0.0
        if offer is not None:
            self._remember_self_offer(offer)
        return offer

    # ---------- Public helpers ----------

    def estimate_opponent_utility(self, outcome: Outcome) -> float:
        return self._model(outcome, self._space.issue_names)

    # ---------- Initialization ----------

    def _initialize_cached_domain(self, force: bool = False) -> None:
        if self._initialized and not force:
            return
        self._reserved = self._get_reserved_value()
        self._extract_issue_space()
        self._outcomes = self._collect_outcomes()
        self._outcome_values.clear()
        if self._outcomes:
            for o in self._outcomes:
                self._outcome_values[o] = self._u(o)
                self._space.ensure_from_outcome(_outcome_to_values(o, self._space.issue_names))
            # Pre-index outcomes by values tuple for O(1) lookup
            self._outcome_by_values: dict[tuple, Outcome] = {}
            for o in self._outcomes:
                key = _outcome_to_values(o, self._space.issue_names)
                self._outcome_by_values[key] = o
            self._best = max(self._outcomes, key=lambda x: self._outcome_values.get(x, -float("inf")))
            self._max_u = self._outcome_values.get(self._best, 1.0)
            self._min_u = min(self._outcome_values.values())
        else:
            try:
                worst, best = self.ufun.extreme_outcomes()
                self._min_u, self._max_u = self._u(worst), self._u(best)
                self._best = best
            except Exception:
                self._best = None
                self._min_u, self._max_u = 0.0, 1.0
        self._reserved = _clip(self._reserved, self._min_u, self._max_u)
        self._initialized = True

    def _extract_issue_space(self) -> None:
        os = getattr(getattr(self, "nmi", None), "outcome_space", None)
        issues = getattr(os, "issues", None)
        names: list[str] = []
        values: list[list[Any]] = []
        if issues:
            for i, issue in enumerate(issues):
                name = getattr(issue, "name", None) or f"i{i}"
                names.append(str(name))
                vals = getattr(issue, "values", None)
                if callable(vals):
                    try:
                        vals = list(vals())
                    except Exception:
                        vals = None
                if vals is None:
                    try:
                        vals = list(issue)
                    except Exception:
                        vals = []
                try:
                    values.append(list(vals))
                except Exception:
                    values.append([])
        if names:
            self._space.issue_names = names
            self._space.values = values

    def _collect_outcomes(self) -> list[Outcome]:
        nmi = getattr(self, "nmi", None)
        outcomes = None
        for attr in ("outcomes", "discrete_outcomes"):
            try:
                outcomes = getattr(nmi, attr)
                if callable(outcomes):
                    outcomes = outcomes()
                if outcomes:
                    break
            except Exception:
                outcomes = None
        if outcomes is not None:
            try:
                outcomes = list(outcomes)
            except Exception:
                outcomes = None
        if not outcomes:
            os = getattr(nmi, "outcome_space", None)
            for attr in ("enumerate_or_sample", "enumerate", "to_discrete"):
                try:
                    f = getattr(os, attr)
                    if attr == "enumerate_or_sample":
                        outcomes = list(f(self.candidate_pool_limit))
                    else:
                        outcomes = list(f())
                    if outcomes:
                        break
                except Exception:
                    outcomes = None
        if outcomes is None:
            try:
                outcomes = list(nmi.random_outcomes(self.candidate_pool_limit))
            except Exception:
                outcomes = []
        if len(outcomes) > self.candidate_pool_limit:
            outcomes = self.rng.sample(list(outcomes), self.candidate_pool_limit)
        return list(outcomes)

    def _get_reserved_value(self) -> float:
        for obj in (getattr(self, "ufun", None), getattr(self, "preferences", None)):
            for attr in ("reserved_value", "reservation_value", "reserved"):
                try:
                    v = getattr(obj, attr)
                    if v is not None:
                        return _safe_float(v, 0.0)
                except Exception:
                    pass
        return 0.0

    # ---------- Utility helpers ----------

    def _u(self, outcome: Outcome) -> float:
        if outcome is None:
            return self._reserved
        if outcome in self._outcome_values:
            return self._outcome_values[outcome]
        try:
            return _safe_float(self.ufun(outcome), self._reserved)
        except Exception:
            return self._reserved

    def _relative_time(self, state: Any) -> float:
        t = getattr(state, "relative_time", None)
        if t is not None:
            return _clip(_safe_float(t), 0.0, 1.0)
        step = _safe_float(getattr(state, "step", 0), 0.0)
        n_steps = _safe_float(getattr(getattr(self, "nmi", None), "n_steps", 0), 0.0)
        if n_steps > 0:
            return _clip((step + 1.0) / (n_steps + 1.0), 0.0, 1.0)
        return 0.0

    # ---------- History & opponent model updates ----------

    def _remember_self_offer(self, offer: Outcome) -> None:
        values = _outcome_to_values(offer, self._space.issue_names)
        if not values:
            return
        self._recent_self_offers.append(offer)
        self._self_offer_counts[values] += 1
        self._self_offer_keys.append(values)

    def _record_opponent_offer(self, offer: Outcome, state: Any) -> None:
        values = _outcome_to_values(offer, self._space.issue_names)
        if not values:
            return
        step = int(_safe_float(getattr(state, "step", len(self._received_offers)), len(self._received_offers)))
        key = (step, values)
        if key == self._last_seen_key:
            return
        self._last_seen_key = key
        self._received_offers.append(offer)
        u = self._u(offer)
        self._received_utils.append(u)
        self._received_keys.append(values)
        self._offer_counts[values] += 1
        self._model.update(offer, self._relative_time(state))
        # Update trajectory analyzer
        self._trajectory.update(u)

    # ---------- Aspiration target (MiCRO-style, no late rise) ----------

    def _detect_stalemate(self) -> bool:
        """Detect if opponent is stalling — repeating near-identical offers."""
        if len(self._received_utils) < self.STALEMATE_WINDOW:
            return False
        recent = self._received_utils[-self.STALEMATE_WINDOW:]
        return (max(recent) - min(recent)) < self.STALEMATE_VARIANCE

    def _detect_rejection_loop(self) -> bool:
        """Detect if OUR proposals are being repeatedly rejected.

        When our last 4 self-offer utilities are all within a narrow band
        and haven't led to agreement, we're stuck proposing the same level
        while the opponent keeps rejecting.  Time to concede faster.
        """
        if len(self._self_offer_keys) < 4:
            return False
        # Get utilities of our last 4 proposals via O(1) lookup
        # (previously iterated entire _outcome_values dict — O(n) per key)
        recent_self = []
        for key in self._self_offer_keys[-4:]:
            outcome = self._outcome_by_values.get(key)
            if outcome is not None:
                recent_self.append(self._outcome_values.get(outcome, 0.0))
        if len(recent_self) < 4:
            return False
        # If our last 4 proposals all had utility within 5% of each other,
        # we're not exploring enough → stuck in a rejection loop.
        return (max(recent_self) - min(recent_self)) < 0.05 * (self._max_u - self._reserved)

    def _opponent_is_time_based(self) -> bool:
        """Detect if opponent follows a time-based offering curve.

        Time-based opponents (BOA/MAPNeg/Boulware/etc.) produce offers that
        change monotonically with low step-to-step variance.  ACNext agents
        (MAPNeg, BOANeg) accept when our offer ≥ their next planned offer,
        so knowing they're time-based lets us target just above that threshold.
        """
        utils = self._received_utils
        if len(utils) < 6:
            return False
        recent = utils[-8:] if len(utils) >= 8 else utils
        n = len(recent)
        # Check monotonicity: all steps should go in the same direction
        # (either all non-increasing = conceder, or all non-decreasing = hardliner)
        steps_up = sum(1 for i in range(1, n) if recent[i] > recent[i-1] + 0.005)
        steps_down = sum(1 for i in range(1, n) if recent[i] < recent[i-1] - 0.005)
        total_steps = n - 1
        if total_steps < 3:
            return False
        # Strongly monotonic in one direction
        monotonic_ratio = max(steps_up, steps_down) / total_steps
        if monotonic_ratio < 0.75:
            return False
        # Low variance in step sizes → consistent concession curve
        step_sizes = [abs(recent[i] - recent[i-1]) for i in range(1, n)]
        if not step_sizes:
            return False
        mean_step = sum(step_sizes) / len(step_sizes)
        if mean_step < 0.001:
            return False  # flat → stalemate, not time-based
        variance = sum((s - mean_step) ** 2 for s in step_sizes) / len(step_sizes)
        cv = variance ** 0.5 / mean_step if mean_step > EPS else 99.0
        return cv < 1.5  # coefficient of variation < 1.5 → consistent curve

    def _opponent_has_conceded(self, min_delta: float = 0.05) -> bool:
        """Check if opponent has meaningfully improved their offers to us.

        min_delta is scaled by the observed utility range so the threshold
        adapts to both wide-range and narrow-range domains.
        """
        if len(self._received_utils) < 8:
            return False
        observed_range = max(self._received_utils) - min(self._received_utils)
        if observed_range < EPS:
            observed_range = 1.0
        effective_delta = min_delta * observed_range
        early_avg = sum(self._received_utils[:4]) / 4.0
        recent_avg = sum(self._received_utils[-4:]) / 4.0
        return (recent_avg - early_avg) >= effective_delta

    def _opponent_conceding_fast(self, min_delta: float = 0.04) -> bool:
        """Fast concession check — only needs 4 samples (2 early vs 2 recent).

        Complements the slower 8-sample _opponent_has_conceded for quicker
        reaction to opponent softening.
        """
        if len(self._received_utils) < 4:
            return False
        observed_range = max(self._received_utils) - min(self._received_utils)
        if observed_range < EPS:
            observed_range = 1.0
        effective_delta = min_delta * observed_range
        early_avg = sum(self._received_utils[:2]) / 2.0
        recent_avg = sum(self._received_utils[-2:]) / 2.0
        return (recent_avg - early_avg) >= effective_delta

    def _effective_e(self, t: float) -> float:
        """Adaptive concession exponent with mirror reciprocity correction.

        Combines opponent-type adaptation (hardliner/conceder/erratic deltas)
        with a gentle reciprocity nudge based on opponent concession speed.
        """
        opp_type = self._trajectory.stable_type
        confidence = self._trajectory.confidence
        e = self.base_e

        # Short negotiations need proportionally faster concession.
        # e.g. 30 steps → e~2.2, 20 steps → e~1.5, 10 steps → e~0.8.
        # With slow Boulware the target barely moves before the deadline.
        if self._is_short_negotiation:
            if self._n_steps <= 15:
                e = 0.8  # ultra-short: near-linear concession
            elif self._n_steps <= 25:
                e -= 1.5  # very short: aggressive concession
            else:
                e -= 0.8  # moderately short

        # In zero-sum scenarios we need a different approach:
        # - Acceptance stays tough (handled in respond)
        # - But proposals must gradually explore compromise outcomes,
        #   otherwise we never reach agreement and the framework defaults
        #   to the opponent's last offer.
        # Concede FASTER (lower e) so our proposal band widens to include
        # the Nash compromise before the deadline.
        if self._is_zero_sum_scenario:
            e -= 1.0  # faster concession to find the compromise zone

        if confidence >= 0.2:
            if opp_type == OpponentType.HARD_LINEAR:
                e += self.hardliner_e_delta
            elif opp_type == OpponentType.CONCEDING:
                e += self.conceder_e_delta
            elif opp_type == OpponentType.ERRATIC:
                e += self.erratic_e_delta

            if self._detect_stalemate():
                e -= 0.5

        # Rejection loop: our proposals keep getting rejected without
        # movement.  Concede faster to break the deadlock and find
        # outcomes the opponent might actually accept.
        if self._detect_rejection_loop():
            e -= 0.7

        # Mirror reciprocity: nudge toward opponent's concession speed.
        # Against conceders (speed high), stay firmer; against hardliners
        # (speed low), concede to find middle ground.
        speed = self._trajectory.concession_speed()
        e += (speed - 0.5) * 1.2

        # Time-based opponent (BOA/MAPNeg with ACNext): be slightly firmer.
        # They follow a predetermined concession curve, so conceding faster
        # just gives them a better deal without making agreement more likely.
        if self._opponent_is_time_based():
            e += 0.2

        return _clip(e, 0.4, 8.0)

    def _floor_utility(self) -> float:
        rng = max(EPS, self._max_u - self._reserved)
        floor = self._reserved + self.floor_fraction * rng
        # In zero-sum scenarios the Nash equilibrium may lie below our
        # normal floor.  Clamp to the 30th percentile of possible utilities
        # so we don't reject the only mutually-acceptable outcome.
        if self._is_zero_sum_scenario and self._outcomes:
            all_utils = sorted([self._u(o) for o in self._outcomes])
            p30 = _percentile(all_utils, 0.30, self._reserved)
            floor = min(floor, max(self._reserved, p30))
        return floor

    @property
    def _n_steps(self) -> int:
        """Estimated total number of negotiation steps."""
        try:
            n = getattr(getattr(self, "nmi", None), "n_steps", 0)
            if n > 0:
                return int(n)
        except Exception:
            pass
        return 100  # default assumption

    @property
    def _is_short_negotiation(self) -> bool:
        """Detect negotiations with very few steps (e.g. NiceOrDie)."""
        return self._n_steps <= 30

    @property
    def _n_issues(self) -> int:
        """Number of issues in the negotiation domain."""
        return self._space.n_issues

    @property
    def _is_nice_or_die_like(self) -> bool:
        """Single-issue, tiny-outcome, extreme-opposition scenario.

        These are fundamentally different from normal negotiations —
        there's no middle ground to gradually converge on.  We must
        directly target the compromise (median) outcome.
        """
        if not self._initialized:
            return False
        return (
            self._n_issues == 1
            and len(self._outcomes) <= 3
            and (self._max_u - self._min_u) >= 0.75
        )

    @property
    def _is_zero_sum_scenario(self) -> bool:
        """Detect extremely competitive (near-zero-sum) scenarios.

        Characteristics: wide utility range, very few discrete outcomes,
        and high opposition — typical of NiceOrDie / battle-of-the-sexes
        domains where one party's gain is the other's loss.
        """
        if not self._initialized:
            return False
        utility_range = self._max_u - self._min_u
        n_outcomes = len(self._outcomes)
        # Wide range + tiny outcome space → zero-sum-like
        if utility_range < 0.75:
            return False
        if n_outcomes > 20:
            return False
        # The wide range and tiny outcome space are sufficient signals.
        # We don't require opponent offers to be polarised — a smart opponent
        # may stick to its best outcome, giving us 0 utility every round.
        return True

    def _target_utility(self, state: Any) -> float:
        """MiCRO-style descending aspiration target.

        target(t) = max_u - (max_u - floor) * t^e(t)

        Where e(t) adapts based on opponent trajectory. The target is strictly
        non-increasing: we never demand MORE as time goes on (no late rise).
        """
        t = self._relative_time(state)
        floor = self._floor_utility()
        e = self._effective_e(t)
        # Store for diagnostic purposes
        self._current_e = e

        # Compute raw target
        high = self._max_u * self.initial_target_fraction
        raw_target = high - (high - floor) * (t ** e)

        # Ensure monotonic: target cannot go up from last round
        clipped = _clip(raw_target, self._reserved, self._max_u)
        if hasattr(self, "_prev_target") and clipped > self._prev_target:
            clipped = self._prev_target
        self._prev_target = clipped

        return clipped

    # ---------- Offer selection (MiCRO_OM style) ----------

    def _select_offer(self, state: Any) -> Outcome | None:
        if not self._outcomes:
            return self._best

        # NiceOrDie-like (1 issue, ≤3 outcomes, wide utility range):
        # there is no "gradual convergence" — just three outcomes with
        # extreme polarisation.  Directly anchor on the compromise.
        if self._is_nice_or_die_like:
            sorted_outs = sorted(self._outcomes, key=lambda o: self._u(o))
            compromise = sorted_outs[len(sorted_outs) // 2]
            # Occasionally test the opponent by proposing our best outcome,
            # but 2/3 of the time stick to compromise.
            if self.rng.random() < 0.65:
                return compromise
            # 35% of the time: propose our best to probe opponent flexibility
            return max(self._outcomes, key=lambda o: self._u(o))

        target = self._target_utility(state)
        # Preference concealment: add ±5% random jitter so target doesn't
        # precisely reveal our true utility curve.
        jitter = 1.0 + self.rng.uniform(-0.05, 0.05)
        target = _clip(target * jitter, self._reserved, self._max_u)
        t = self._relative_time(state)
        # Time scale for short negotiations (consistent with respond).
        t_scale = min(1.0, self._n_steps / 50.0) if self._n_steps > 0 else 1.0
        rng = max(EPS, self._max_u - self._reserved)

        # --- Build candidate pool ---
        # Band width: wider early (exploration), narrower late (exploitation)
        band_fraction = 0.04 + 0.06 * max(0.0, 1.0 - t * 1.2)
        # Adaptive band: adjust to outcome space size
        outcome_count = len(self._outcomes)
        if outcome_count > 5000:
            band_fraction *= 1.5
        elif outcome_count < 100:
            band_fraction *= 2.0
        # Narrow band further when opponent model is confident
        if self._trajectory.confidence > 0.5:
            band_fraction *= 0.8
        band = band_fraction * rng
        lower = max(self._reserved, target - band)
        upper = min(self._max_u, target + band * 0.3)

        candidates = [o for o in self._outcomes if lower <= self._u(o) <= upper]

        # Also consider recent opponent offers if they are close to our target,
        # as they represent a revealed willingness by the opponent.
        opp_slack = (0.01 + 0.06 * t) * rng
        guarded_lower = max(self._reserved, lower - opp_slack)
        for o in reversed(self._received_offers[-15:]):
            if self._u(o) >= guarded_lower and o not in candidates:
                candidates.append(o)

        # In zero-sum scenarios, force-inject the compromise (Nash-like)
        # outcome into candidates.  Scale the time threshold for short
        # negotiations where each step covers more relative time.
        _t_thresh = 0.25 * min(1.0, self._n_steps / 50.0) if self._n_steps > 0 else 0.25
        if self._is_zero_sum_scenario and t > _t_thresh:
            # The "compromise" is the outcome closest to the median utility.
            sorted_outs = sorted(self._outcomes, key=lambda o: self._u(o))
            mid_idx = len(sorted_outs) // 2
            for o in sorted_outs[max(0, mid_idx - 1):mid_idx + 2]:
                if o not in candidates:
                    candidates.append(o)

        # If band is too narrow, expand to nearest outcomes (excluding
        # outcomes at or below reserved value — proposing those is suicidal).
        if len(candidates) < 10:
            min_safe = self._reserved + 0.02 * rng
            safe_outs = [o for o in self._outcomes if self._u(o) > min_safe]
            if not safe_outs:
                safe_outs = [o for o in self._outcomes if self._u(o) > self._reserved]
            ordered = sorted(safe_outs, key=lambda o: abs(self._u(o) - target))
            candidates = ordered[:min(80, len(ordered))]
            for o in reversed(self._received_offers[-15:]):
                if self._u(o) >= guarded_lower and o not in candidates:
                    candidates.append(o)
        elif len(candidates) > 500:
            # Keep recent opponent offers pinned, randomly sample the rest.
            # Force-include the last 20 opponent offers even if outside band.
            forced = list(self._received_offers[-20:])
            forced_keys = {_outcome_to_values(o, self._space.issue_names) for o in forced}
            rest = [o for o in candidates if _outcome_to_values(o, self._space.issue_names) not in forced_keys]
            need = max(0, 500 - len(forced))
            candidates = forced + (self.rng.sample(rest, need) if len(rest) > need else rest)

        # --- Tit-for-Tat innovation ---
        # Copy opponent's last offer but improve one issue in our favour.
        # TFT is one of the most robust negotiation strategies — it signals
        # willingness to cooperate while still optimising.
        tft_candidate: Outcome | None = None
        if (self._received_offers and len(self._received_keys) >= 2
                and self.rng.random() < self.TFT_PROBABILITY):  # 15% chance of TFT move
            last = self._received_offers[-1]
            last_u = self._u(last)
            last_vals = list(_outcome_to_values(last, self._space.issue_names))
            if last_vals and last_u > self._reserved:
                # Find the issue where we can improve our utility most
                best_improvement = 0.0
                best_alt_vals = None
                for i in range(len(last_vals)):
                    if i >= len(self._space.values):
                        continue
                    for alt_val in self._space.values[i]:
                        alt = list(last_vals)
                        alt[i] = alt_val
                        # Find the actual outcome via O(1) lookup
                        # (previously iterated all _outcomes — O(n) per alt)
                        alt_tuple = tuple(alt)
                        matching_outcome = self._outcome_by_values.get(alt_tuple)
                        if matching_outcome is not None:
                            alt_u = self._u(matching_outcome)
                            improvement = alt_u - last_u
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_alt_vals = matching_outcome
                if best_alt_vals is not None and best_improvement > 0.01:
                    tft_candidate = best_alt_vals

        # --- Score candidates ---
        best_o: Outcome | None = None
        best_score = -float("inf")

        # Weights shift over time: early = explore/maximise self; late = lock in deal
        opp_weight = 0.20 + 0.40 * t          # opponent model influence grows
        self_weight = 0.55 - 0.18 * t         # self utility influence shrinks
        closeness_weight = 0.15                # stay near target
        novelty_weight = 0.06 + 0.04 * (1.0 - t)
        # Preference concealment: fluctuate novelty weight by round parity
        # so our offer pattern is less predictable.
        round_num = len(self._self_offer_keys)
        novelty_weight *= 1.5 if round_num % 2 == 0 else 0.7
        opp_offer_weight = 0.08 + 0.12 * t
        diversity_weight = 0.05

        for o in candidates:
            su = self._u(o)
            ou = self.estimate_opponent_utility(o)
            closeness = 1.0 - min(1.0, abs(su - target) / max(EPS, band * 2.5))
            diversity = self._diversity_bonus(o)
            novelty = self._novelty_bonus(o)
            opponent_exact = self._opponent_offer_bonus(o)
            repeat_penalty = self._repeat_penalty(o)

            # Dampen opponent-offer bonus when the outcome is poor for us.
            # Outcomes the opponent loves but give us nothing should not be
            # proposed just because the opponent keeps offering them.
            su_norm = (su - self._reserved) / rng
            opponent_exact *= max(0.15, su_norm)

            # In zero-sum, the Nash compromise (mid-utility outcome) needs
            # a scoring boost to overcome the self-utility dominance of
            # extreme outcomes.  Without this we'd never propose compromises.
            nash_bonus = 0.0
            if self._is_zero_sum_scenario and t > 0.35 * t_scale:
                # Bonus proportional to how "central" the outcome is in
                # utility space — favours compromise over extremes.
                nash_bonus = 0.80 * (1.0 - abs(su_norm - 0.5) * 2.0)  # max at 0.5
                nash_bonus *= min(1.0, (t - 0.35 * t_scale) / max(EPS, 0.40 * t_scale))

            score = (
                self_weight * ((su - self._reserved) / rng)
                + opp_weight * ou
                + closeness_weight * closeness
                + novelty_weight * novelty
                + opp_offer_weight * opponent_exact
                + diversity_weight * diversity
                + nash_bonus
                - 0.15 * repeat_penalty
                + self.rng.random() * 0.002
            )
            if score > best_score:
                best_score = score
                best_o = o

        # Preference concealment: 4% chance to bluff in early-mid phase.
        # Never bluff against hardliners — too risky.
        if (
            best_o is not None
            and t < self.BLUFF_MAX_TIME
            and self._trajectory.stable_type != OpponentType.HARD_LINEAR
            and self.rng.random() < self.BLUFF_PROBABILITY
        ):
            floor_u = self._floor_utility()
            bluff_lower = max(self._reserved, floor_u)
            bluff_upper = target - 0.05 * rng
            if bluff_lower < bluff_upper:
                bluff_candidates = [
                    o for o in self._outcomes
                    if bluff_lower <= self._u(o) <= bluff_upper
                ]
                if bluff_candidates:
                    best_o = max(bluff_candidates, key=lambda o: self.estimate_opponent_utility(o))

        # In zero-sum scenarios, periodically force-propose the compromise
        # (median-utility) outcome to break the deadlock.  Without this we'd
        # keep proposing extreme outcomes that the opponent always rejects.
        # In short negotiations, force compromise every other proposal.
        _mod = 2 if self._is_short_negotiation else 3
        if (
            self._is_zero_sum_scenario
            and t > 0.25 * t_scale
            and best_o is not None
            and len(self._self_offer_keys) % _mod == (_mod - 1)
        ):
            sorted_outs = sorted(self._outcomes, key=lambda o: self._u(o))
            compromise = sorted_outs[len(sorted_outs) // 2]
            if self._u(compromise) > self._reserved:
                best_o = compromise

        # Tit-for-Tat innovation: use TFT candidate when available and the
        # normal best isn't clearly superior.
        if tft_candidate is not None:
            tft_u = self._u(tft_candidate)
            best_u = self._u(best_o) if best_o is not None else 0.0
            # Only use TFT if it gives us ≥ 90% of the normal best utility
            # AND is above reserved value (safety check).
            if (best_o is None or tft_u >= 0.90 * best_u) and tft_u > self._reserved:
                return tft_candidate

        # Safety: never propose an outcome at or below reserved value.
        if best_o is not None and self._u(best_o) <= self._reserved:
            safe = [o for o in self._outcomes if self._u(o) > self._reserved]
            if safe:
                best_o = max(safe, key=lambda o: self._u(o))
            else:
                best_o = self._best

        return best_o or self._best

    # ---------- Bonus functions (same as ShadowBath) ----------

    def _novelty_bonus(self, outcome: Outcome) -> float:
        values = _outcome_to_values(outcome, self._space.issue_names)
        if not values:
            return 0.5
        count = self._self_offer_counts.get(values, 0)
        if count <= 0:
            return 1.0
        return _clip(1.0 / (1.0 + count), 0.0, 1.0)

    def _opponent_offer_bonus(self, outcome: Outcome) -> float:
        values = _outcome_to_values(outcome, self._space.issue_names)
        if not values or not self._received_keys:
            return 0.0
        count = self._offer_counts.get(values, 0)
        if count <= 0:
            return 0.0
        max_count = max(self._offer_counts.values()) if self._offer_counts else 1
        latest = 0
        for idx, key in enumerate(self._received_keys):
            if key == values:
                latest = idx + 1
        recency = latest / max(1, len(self._received_keys))
        frequency = count / max(1, max_count)
        return _clip(0.45 + 0.35 * frequency + 0.20 * recency, 0.0, 1.0)

    def _diversity_bonus(self, outcome: Outcome) -> float:
        if not self._recent_self_offers:
            return 0.5
        vals = _outcome_to_values(outcome, self._space.issue_names)
        if not vals:
            return 0.5
        distances: list[float] = []
        for prev in self._recent_self_offers:
            pvals = _outcome_to_values(prev, self._space.issue_names)
            n = max(1, min(len(vals), len(pvals)))
            mismatch = sum(1 for a, b in zip(vals[:n], pvals[:n]) if a != b) / n
            distances.append(mismatch)
        return _clip(sum(distances) / max(1, len(distances)), 0.0, 1.0)

    def _repeat_penalty(self, outcome: Outcome) -> float:
        vals = _outcome_to_values(outcome, self._space.issue_names)
        if not vals:
            return 0.0
        recent_count = sum(1 for o in self._recent_self_offers if _outcome_to_values(o, self._space.issue_names) == vals)
        global_count = self._self_offer_counts.get(vals, 0)
        recent_penalty = recent_count / max(1, len(self._recent_self_offers))
        global_penalty = min(1.0, global_count / 4.0)
        return _clip(0.65 * recent_penalty + 0.35 * global_penalty, 0.0, 1.0)


# Backward-compatible alias
MyNegotiator = AdaptiveBathNegotiator