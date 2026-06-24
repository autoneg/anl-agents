"""
LionelWei v4.9 negotiation agent for ANL2026 / NegMAS SAO negotiations.

v4.9 = v4.8 + two LLM-evolved (openevolve) structural improvements, each strictly
dominant over v4.8 on the local 252-match matrix (zero per-cell regression):
  1. Bidding score: the opponent term is a time-blended Renting f2/f3 mix
     (f2 toward the opponent's last bid, f3 locking onto the best-for-us offer),
     replacing the raw opponent-utility term.
  2. Acceptance: opponent-adaptive late-game relaxation of the threshold
     (concede-detecting late_factor) on top of the mid-game multiplier.
Net: advantage_mean +0.0019 vs v4.8, zero regression. Standalone (no genome dep).

Based on the LionelNeg design:
1. Segmented acceptance:
   - early phase: accept only offers no worse than the agent's own next proposal;
   - late phase: accept offers that can be worse than the agent's own next proposal,
     with a deadline-aware safety floor above the reservation value.
2. Bathtub-shaped bidding target:
   - starts high, concedes to a dynamically inferred valley, then rises late;
   - the valley is weighted from the opponent's historical offers;
   - the late rise is not fixed. It is inferred from the opponent's history and the
     combined opponent model.
3. Bidding preferences:
   - prefer outcomes we have not proposed before;
   - prefer exact outcomes previously proposed by the opponent when they still satisfy
     our current utility target band.
4. Opponent model:
   - combines a Smith/HardHeaded-style frequency model with a distribution-based
     frequency model inspired by:
       Tunalı, O., Aydoğan, R., Sanchez-Anguix, V. (2017),
       "Rethinking Frequency Opponent Modeling in Automated Negotiation".

No third-party dependency beyond NegMAS/ANL2026 is required.
"""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

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


def _normalize_positive(values: list[float], fallback: float | None = None) -> list[float]:
    if not values:
        return []
    values = [max(0.0, _safe_float(v)) for v in values]
    s = sum(values)
    if s <= EPS:
        f = 1.0 / len(values) if fallback is None else fallback
        return [f for _ in values]
    return [v / s for v in values]


def _percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    vals = sorted(_safe_float(v) for v in values)
    q = min(1.0, max(0.0, q))
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _clip(x: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def _outcome_to_values(outcome: Outcome, issue_names: list[str] | None = None) -> tuple[Any, ...]:
    """Converts NegMAS outcomes to a stable tuple representation.

    NegMAS outcomes are usually tuples. Some utilities can expose dict-like outcomes.
    This helper handles both without depending on internals.
    """
    if outcome is None:
        return tuple()
    if isinstance(outcome, dict):
        if issue_names:
            return tuple(outcome.get(k) for k in issue_names)
        return tuple(outcome[k] for k in sorted(outcome))
    if isinstance(outcome, (tuple, list)):
        return tuple(outcome)
    return (outcome,)


def _critical_chi_square_95(df: int) -> float:
    """Small table + approximation for a 95% chi-square critical value.

    The DFM paper uses a chi-square test with p > 0.05 to decide if two issue-value
    distributions are statistically equivalent. We avoid scipy by using common table
    values for small degrees of freedom and a Wilson-Hilferty approximation for the
    tail. This is enough for an online opponent-model heuristic.
    """
    table = {
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    # Wilson-Hilferty transform with z_0.95 ~= 1.64485
    z = 1.6448536269514722
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


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
    """Smith/HardHeaded-style frequency opponent model.

    It uses two common assumptions from classical frequency models:
    - values offered more often are likely more valuable to the opponent;
    - issues that stay unchanged across consecutive opponent offers are likely more
      important to the opponent.

    This is a local implementation rather than a wrapper around NegMAS's
    GSmithFrequencyModel, because competition templates may not expose component
    internals uniformly across versions.
    """

    def __init__(self, space: OpponentIssueSpace, no_change_increment: float = 0.08):
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
                # Slight decay over time: early no-change evidence is more informative.
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
        if issue_index >= len(self.value_counts):
            return 0.5
        counts = self.value_counts[issue_index]
        if not counts:
            return 0.5
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
    """Distribution-based frequency model inspired by Tunalı et al. (2017).

    Key ideas implemented:
    - value evaluation uses Laplace smoothing and an exponential filter gamma;
    - issue-weight updates compare disjoint windows of opponent offers;
    - weights of issues whose distribution remains stable are increased only when
      some other issue appears to show concession;
    - update impact decays as alpha * (1 - t**beta).
    """

    def __init__(
        self,
        space: OpponentIssueSpace,
        window_size: int = 4,
        alpha: float = 0.075,
        beta: float = 1.6,
        gamma: float = 0.55,
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
        if issue_index >= len(self.value_counts):
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
        # Convert smoothed probabilities back to expected pseudo-counts.
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
        previous_window = self.offers[-2 * k : -k]
        current_window = self.offers[-k:]
        unchanged_issues: list[int] = []
        concession_detected = False
        n = self.space.n_issues

        for i in range(n):
            prev_dist = self._smoothed_distribution(previous_window, i)
            curr_dist = self._smoothed_distribution(current_window, i)
            if not prev_dist or not curr_dist:
                continue
            df = max(1, len(set(prev_dist) | set(curr_dist)) - 1)
            stat = self._chi_square_stat(prev_dist, curr_dist, k)
            same_distribution = stat <= _critical_chi_square_95(df)
            if same_distribution:
                unchanged_issues.append(i)
            else:
                prev_eu = self._expected_issue_utility(prev_dist, i)
                curr_eu = self._expected_issue_utility(curr_dist, i)
                if curr_eu + 1e-5 < prev_eu:
                    concession_detected = True

        if concession_detected and 0 < len(unchanged_issues) < n:
            delta = self.alpha * max(0.0, 1.0 - relative_time**self.beta)
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
    """Adaptive mixture of Smith Frequency Model and DFM."""

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
        # SFM is more useful early; DFM becomes more reliable after enough windows.
        n = len(self.offers)
        confidence = _clip((n - self.window_size) / max(1, 3 * self.window_size), 0.0, 1.0)
        return 0.35 + 0.30 * confidence

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        wd = self.dfm_weight
        s = self.smith(outcome, issue_names or self.space.issue_names)
        d = self.dfm(outcome, issue_names or self.space.issue_names)
        return _clip((1.0 - wd) * s + wd * d, 0.0, 1.0)

    def issue_weights(self) -> dict[str, float]:
        wd = self.dfm_weight
        names = self.space.issue_names or [f"i{i}" for i in range(self.space.n_issues)]
        out: dict[str, float] = {}
        for i, name in enumerate(names):
            sw = self.smith.weights[i] if i < len(self.smith.weights) else 0.0
            dw = self.dfm.weights[i] if i < len(self.dfm.weights) else 0.0
            out[name] = (1.0 - wd) * sw + wd * dw
        return out


class OpponentUtilityAdapter:
    """Callable utility-like wrapper exposed for external evaluators if needed."""

    def __init__(self, negotiator: "LionelWei"):
        self.negotiator = negotiator
        self.reserved_value = 0.0

    def __call__(self, outcome: Outcome) -> float:
        return self.negotiator.estimate_opponent_utility(outcome)


class LionelWei(SAONegotiator):
    """ANL2026 negotiator using bathtub bidding + segmented acceptance."""

    def __init__(
        self,
        *args: Any,
        window_size: int = 4,
        valley_time: float = 0.68,
        early_end: float = 0.45,
        late_start: float = 0.82,
        valley_fraction: float = 0.61,
        candidate_pool_limit: int = 12000,
        rng_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = max(2, int(window_size))
        self.valley_time = _clip(valley_time, 0.45, 0.8)
        self.early_end = _clip(early_end, 0.2, 0.65)
        self.late_start = _clip(late_start, self.valley_time + 0.05, 0.95)
        self.valley_fraction = _clip(valley_fraction, 0.35, 0.85)
        self.candidate_pool_limit = max(500, int(candidate_pool_limit))
        self.rng = random.Random(rng_seed)

        self._space = OpponentIssueSpace()
        self._model = CombinedFrequencyOpponentModel(self._space, window_size=self.window_size)
        self.opponent_model = self._model
        self._opponent_ufun_adapter = OpponentUtilityAdapter(self)
        # Mirror model: simulates what a frequency-based opponent (BOANeg/MAPNeg)
        # has inferred about US from our own offers. Used only for the adversarial
        # tie-break in _select_offer; never for bidding/acceptance decisions.
        self._mirror = SmithFrequencyOpponentModel(self._space)
        # Track B switch — keep False until Track A results are attributed.
        self._adversarial_tiebreak = False
        self._mirror_pairs: list[tuple[Outcome, Outcome]] = []
        self._last_t = 0.0

        self._outcomes: list[Outcome] = []
        self._outcome_values: dict[Outcome, float] = {}
        self._best: Outcome | None = None
        self._worst: Outcome | None = None
        self._min_u = 0.0
        self._max_u = 1.0
        self._reserved = 0.0
        self._last_proposal: Outcome | None = None
        self._last_proposal_u = 0.0
        self._last_seen_key: tuple[int, tuple[Any, ...]] | None = None
        self._received_offers: list[Outcome] = []
        self._received_utils: list[float] = []
        self._offer_counts: Counter[tuple[Any, ...]] = Counter()
        self._received_keys: list[tuple[Any, ...]] = []
        self._self_offer_counts: Counter[tuple[Any, ...]] = Counter()
        self._self_offer_keys: list[tuple[Any, ...]] = []
        self._recent_self_offers: deque[Outcome] = deque(maxlen=6)
        self._initialized = False
        # A1: monotone acceptance threshold tracking
        self._min_threshold_seen: float = float("inf")
        # A2: valley drop rate limiter
        self._prev_valley: float | None = None
        # C2: extreme scenario flag (NiceOrDie-type, >85% outcomes near reservation)
        self._extreme_scenario: bool = False

    @property
    def opponent_ufun(self) -> OpponentUtilityAdapter:
        return self._opponent_ufun_adapter

    # ---------- NegMAS callbacks ----------

    def on_preferences_changed(self, changes: Any) -> None:  # called once ufun is attached/changed
        self._initialize_cached_domain(force=True)
        try:
            super().on_preferences_changed(changes)
        except Exception:
            pass

    def respond(self, state: Any, offer: Outcome | str | None = None, source: str = "") -> Any:
        try:
            # NegMAS versions differ: respond(state, source) vs respond(state, offer, nid).
            if isinstance(offer, str) and source == "":
                source = offer
                offer = None
            if offer is None:
                offer = getattr(state, "current_offer", None)
            if offer is None:
                return ResponseType.REJECT_OFFER

            self._initialize_cached_domain()
            self._record_opponent_offer(offer, state)

            offer_u = self._u(offer)
            next_offer = self._select_offer(state, update_memory=False)
            next_u = self._u(next_offer) if next_offer is not None else self._target_utility(state)
            rng = max(EPS, self._max_u - self._reserved)
            floor = self._reserved + 0.01 * rng
            # C2: in extreme scenarios refuse near-zero-advantage deals throughout
            if self._extreme_scenario:
                floor = max(floor, self._reserved + 0.15 * rng)

            # ACNext + target floor: accept if opponent's offer ≥ our planned offer.
            # Also enforce 85% of Boulware target as a floor in normal scenarios —
            # prevents opponent_exact_weight from pulling next_u below our aspiration
            # and triggering premature acceptance via the downward ratchet.
            target_u = self._target_utility(state)
            if self._extreme_scenario:
                threshold = max(floor, next_u)
            else:
                # Hard-deadlock check (stricter than time_based propose threshold).
                # Only lower acceptance bar when opponent truly gives us near-zero
                # utility (<5% of range). This avoids the A1-ratchet trap where a
                # false positive permanently locks the threshold below what a normally-
                # conceding opponent (Car × Boulware → eventually 0.952) would need.
                _n = len(self._received_offers)
                _t = self._relative_time(state)
                # Hard-deadlock: opponent gives us near-zero utility even in late game.
                # t > 0.80 guard prevents A1-ratchet lock-in from early rounds where
                # any opponent (including BOANeg adapting in Car) starts low. By t=0.80,
                # a genuinely cooperative opponent will have improved; only true time-based
                # opponents in misaligned domains (Camera × Boulware) stay near 0.
                _hard_deadlock = (
                    _n >= 12
                    and _t > 0.80
                    and sum(self._u(o) for o in self._received_offers[-8:]) / 8
                    < self._reserved + 0.05 * rng
                )
                if _hard_deadlock:
                    threshold = max(floor, target_u * 0.70)
                else:
                    # Mid-game multiplier 0.92 → 0.87 by t=0.90: BOANeg harvests
                    # 0.876+ in Grocery by simply not accepting 0.80-0.82 at
                    # t≈0.62; match its patience, relax near the deadline.
                    _mult = 0.92 - 0.05 * _clip((_t - 0.75) / 0.15, 0.0, 1.0)
                    # v4.9 (LLM-evolved, RenEtAl22): opponent-adaptive late relaxation.
                    # If the opponent has conceded to us recently, relax the bar earlier
                    # and smoother (close the deal); against a slow/Boulware opponent
                    # hold firm and relax only very late (after t≈0.92).
                    opp_conceding = False
                    if len(self._received_offers) >= 4:
                        u_hist = [self._u(o) for o in self._received_offers[-4:]]
                        if max(u_hist) > u_hist[0] + 0.01 * rng:
                            opp_conceding = True
                    if opp_conceding:
                        late_factor = 1.0 - 0.4 * _clip((_t - 0.85) / 0.15, 0.0, 1.0)
                    else:
                        late_factor = 1.0 - 0.5 * _clip((_t - 0.92) / 0.08, 0.0, 1.0)
                    threshold = max(floor, next_u, target_u * _mult * late_factor)

            # A1: threshold must never rise — clamp to running minimum, then update.
            threshold = max(floor, min(threshold, self._min_threshold_seen))
            self._min_threshold_seen = threshold

            if offer_u + 1e-12 >= threshold:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER
        except Exception:
            return ResponseType.REJECT_OFFER

    def propose(self, state: Any) -> Outcome:
        try:
            self._initialize_cached_domain()
            offer = self._select_offer(state, update_memory=True)
            if offer is None:
                try:
                    offer = self.nmi.random_outcomes(1)[0]
                except Exception:
                    offer = self._best
            if offer is None and self._outcomes:
                offer = max(self._outcomes, key=lambda o: self._u(o))
            self._last_proposal = offer
            self._last_proposal_u = self._u(offer) if offer is not None else 0.0
            if offer is not None:
                self._remember_self_offer(offer)
            return offer
        except Exception:
            try:
                return self._best or self.nmi.random_outcomes(1)[0]
            except Exception:
                return self._best

    # ---------- Public opponent-estimate helpers ----------

    def _internal_opponent_score(self, outcome: Outcome) -> float:
        """True internal model estimate — used for bidding decisions only."""
        return self._model(outcome, self._space.issue_names)

    def estimate_opponent_utility(self, outcome: Outcome) -> float:
        """Exposed to the ANL2026 Concealing evaluator."""
        return self._internal_opponent_score(outcome)

    def utility_estimate(self, outcome: Outcome) -> float:
        """Alias for tournament code that looks for a generic estimator."""
        return self.estimate_opponent_utility(outcome)

    def opponent_utility(self, outcome: Outcome) -> float:
        """Alias for analysis scripts."""
        return self.estimate_opponent_utility(outcome)

    # ---------- Initialization and utility helpers ----------

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
            self._best = max(self._outcomes, key=lambda x: self._outcome_values.get(x, -float("inf")))
            self._worst = min(self._outcomes, key=lambda x: self._outcome_values.get(x, float("inf")))
            self._max_u = self._outcome_values.get(self._best, 1.0)
            self._min_u = self._outcome_values.get(self._worst, 0.0)
        else:
            try:
                self._worst, self._best = self.ufun.extreme_outcomes()
                self._min_u, self._max_u = self._u(self._worst), self._u(self._best)
            except Exception:
                self._best, self._worst = None, None
                self._min_u, self._max_u = 0.0, 1.0
        self._reserved = _clip(self._reserved, self._min_u, self._max_u)
        self._min_threshold_seen = float("inf")
        self._prev_valley = None
        self._extreme_scenario = self._compute_extreme_scenario()
        # Pre-sample outcome pairs once for the adversarial tie-break's Kendall
        # approximation; pairs with distinct self-utility carry ranking signal.
        self._mirror_pairs = []
        if len(self._outcomes) >= 4:
            n_pairs = min(120, 3 * len(self._outcomes))
            for _ in range(n_pairs):
                a, b = self.rng.sample(self._outcomes, 2)
                if abs(self._u(a) - self._u(b)) > 1e-6:
                    self._mirror_pairs.append((a, b))
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

    # ---------- History and opponent model ----------

    def _remember_self_offer(self, offer: Outcome) -> None:
        values = _outcome_to_values(offer, self._space.issue_names)
        if not values:
            return
        self._recent_self_offers.append(offer)
        self._self_offer_counts[values] += 1
        self._self_offer_keys.append(values)
        self._mirror.update(values, self._last_t)

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
        self._received_utils.append(self._u(offer))
        self._received_keys.append(values)
        self._offer_counts[values] += 1
        self._model.update(offer, self._relative_time(state))

    def _estimated_rejection_risk(self, offer: Outcome | None) -> float:
        if offer is None or not self._received_offers:
            return 0.5
        proposed_score = self._internal_opponent_score(offer)
        hist_scores = [self._internal_opponent_score(o) for o in self._received_offers[-max(3, self.window_size) :]]
        anchor = sum(hist_scores) / max(1, len(hist_scores))
        return _clip((anchor - proposed_score + 0.15) / 0.3, 0.0, 1.0)

    # ---------- Bathtub target and offering ----------

    def _valley_utility(self) -> float:
        rng = max(EPS, self._max_u - self._reserved)
        # C1+C2: raise valley floor against hard opponents and in extreme scenarios
        effective_fraction = self.valley_fraction
        if self._extreme_scenario:
            effective_fraction = _clip(effective_fraction + 0.15, 0.35, 0.90)
        hardness = self._detect_opponent_hardness()
        effective_fraction = _clip(effective_fraction + 0.15 * hardness, 0.35, 0.90)
        prior = self._reserved + effective_fraction * rng
        if not self._received_offers:
            return prior

        weighted_utils: list[float] = []
        weights: list[float] = []
        n = len(self._received_offers)
        max_count = max(self._offer_counts.values()) if self._offer_counts else 1
        for idx, offer in enumerate(self._received_offers):
            key = _outcome_to_values(offer, self._space.issue_names)
            su = self._u(offer)
            recency = (idx + 1) / max(1, n)
            frequency = self._offer_counts.get(key, 1) / max(1, max_count)
            opp_support = self._internal_opponent_score(offer)
            w = 0.35 + 0.45 * recency + 0.30 * frequency + 0.25 * opp_support
            weighted_utils.append(su)
            weights.append(w)

        wsum = sum(weights)
        if wsum <= EPS:
            return prior
        wmean = sum(u * w for u, w in zip(weighted_utils, weights)) / wsum
        wq65 = self._weighted_quantile(weighted_utils, weights, 0.65, default=wmean)
        best_received = max(self._received_utils) if self._received_utils else wmean

        history_anchor = 0.50 * wmean + 0.35 * wq65 + 0.15 * best_received
        confidence = _clip(n / max(1, 2.5 * self.window_size), 0.0, 1.0)
        valley = (1.0 - confidence) * prior + confidence * history_anchor

        safety_floor = self._reserved + 0.30 * rng
        safety_cap = self._max_u - 0.12 * rng
        # A2: limit how fast the valley can drop each call — at most 5% of rng per round.
        if self._prev_valley is not None:
            valley = max(valley, self._prev_valley - 0.05 * rng)
        result = _clip(valley, safety_floor, safety_cap)
        self._prev_valley = result
        return result

    def _compute_extreme_scenario(self) -> bool:
        """True when >85% of outcomes are near reservation — e.g. NiceOrDie."""
        if not self._outcomes:
            return False
        rng = max(EPS, self._max_u - self._reserved)
        low_threshold = self._reserved + 0.10 * rng
        low_count = sum(
            1 for o in self._outcomes
            if self._outcome_values.get(o, self._u(o)) <= low_threshold
        )
        return (low_count / max(1, len(self._outcomes))) > 0.85

    def _detect_opponent_hardness(self) -> float:
        """Returns [0, 1]: 0 = opponent conceding fast, 1 = barely moving."""
        n = len(self._received_utils)
        if n < 4:
            return 0.5
        rng = max(EPS, self._max_u - self._reserved)
        recent = self._received_utils[-min(8, n):]
        improvements = [max(0.0, recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
        avg_improvement = sum(improvements) / max(1, len(improvements))
        # opponent improving ≥5% rng/round → soft; near 0 → hard
        return _clip(1.0 - avg_improvement / max(EPS, 0.05 * rng), 0.0, 1.0)

    def _weighted_quantile(
        self, values: Sequence[float], weights: Sequence[float], q: float, default: float = 0.0
    ) -> float:
        if not values or not weights or len(values) != len(weights):
            return default
        pairs = sorted((float(v), max(0.0, float(w))) for v, w in zip(values, weights))
        total = sum(w for _, w in pairs)
        if total <= EPS:
            return default
        cutoff = _clip(q, 0.0, 1.0) * total
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= cutoff:
                return v
        return pairs[-1][0]

    def _late_peak_utility(self) -> float:
        valley = self._valley_utility()
        rng = max(EPS, self._max_u - self._reserved)
        if not self._received_offers:
            return min(self._max_u, valley + 0.08 * rng)

        best_received = max(self._received_utils)
        recent = self._received_utils[-max(3, self.window_size) :]
        prev = self._received_utils[-2 * max(3, self.window_size) : -max(3, self.window_size)]
        recent_avg = sum(recent) / max(1, len(recent))
        prev_avg = sum(prev) / max(1, len(prev)) if prev else recent_avg
        concession_trend = max(0.0, recent_avg - prev_avg)

        attainable = valley
        if self._outcomes:
            hist_opp_scores = [self._internal_opponent_score(o) for o in self._received_offers]
            opp_cut = max(0.40, _percentile(hist_opp_scores, 0.45, 0.5) - 0.04)
            sampled = self._top_self_candidates(limit=min(len(self._outcomes), 2500))
            for o in sampled:
                if self._internal_opponent_score(o) >= opp_cut:
                    attainable = max(attainable, self._u(o))

        behavior_peak = best_received + 0.05 * rng + 1.25 * concession_trend
        model_peak = attainable
        confidence = _clip(len(self._received_offers) / max(1, 3 * self.window_size), 0.0, 1.0)
        peak = valley + confidence * (0.55 * model_peak + 0.45 * behavior_peak - valley)
        cap = self._max_u - 0.015 * rng
        return _clip(peak, valley, cap)

    def _target_utility(self, state: Any) -> float:
        """Boulware polynomial: 1 - t^5. Stays near max_u for ~70% of time, then drops fast.
        In extreme scenarios (NiceOrDie-type) the floor drops to reservation so we can
        reach the win-win zone by t≈0.91; C2 in respond() guards the 0.15*rng minimum."""
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        floor_u = self._reserved if self._extreme_scenario else self._reserved + 0.10 * rng
        asp = 1.0 - math.pow(t, 5.0)
        target = floor_u + asp * (self._max_u - floor_u)
        return _clip(target, self._reserved, self._max_u)

    def _top_self_candidates(self, limit: int = 2500) -> list[Outcome]:
        if not self._outcomes:
            return []
        if len(self._outcome_values) < len(self._outcomes):
            for o in self._outcomes:
                self._outcome_values.setdefault(o, self._u(o))
        ordered = sorted(self._outcomes, key=lambda o: self._outcome_values.get(o, self._reserved), reverse=True)
        return ordered[: min(limit, len(ordered))]

    def _select_offer(self, state: Any, update_memory: bool = True) -> Outcome | None:
        if not self._outcomes:
            return self._best
        target = self._target_utility(state)
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        band = (0.025 + 0.055 * (1.0 - min(t, 0.8))) * rng
        lower = max(self._reserved, target - band)
        upper = min(self._max_u, target + 0.75 * band)

        candidates = [o for o in self._outcomes if lower <= self._u(o) <= upper]

        opponent_slack = (0.015 + 0.075 * t) * rng
        guarded_lower = max(self._reserved, lower - opponent_slack)
        for o in reversed(self._received_offers[-20:]):
            if self._u(o) >= guarded_lower and o not in candidates:
                candidates.append(o)

        if len(candidates) < 8:
            ordered = sorted(self._outcomes, key=lambda o: abs(self._u(o) - target))
            candidates = ordered[: min(80, len(ordered))]
            for o in reversed(self._received_offers[-20:]):
                if self._u(o) >= guarded_lower and o not in candidates:
                    candidates.append(o)
        elif len(candidates) > 600:
            pinned = [o for o in self._received_offers[-20:] if o in candidates]
            pinned_keys = {_outcome_to_values(o, self._space.issue_names) for o in pinned}
            rest = [o for o in candidates if _outcome_to_values(o, self._space.issue_names) not in pinned_keys]
            need = max(0, 600 - len(pinned))
            candidates = pinned + (self.rng.sample(rest, need) if len(rest) > need else rest)

        # Hard floor: never propose below target range.
        # Mid-game: hold our Pareto step — frequency-based opponents (MAPNeg in
        # Laptop) concede at t≈0.69 if we don't surrender to their standing offer
        # first; a fixed 0.20 slack let their offer through at t>0.65 and we caved
        # one round before they would have. End-game: ramp back to the generous
        # 0.20 slack so deadline deals (Car×MiCRO) still close.
        if not self._extreme_scenario:
            gate = _clip((t - 0.80) / 0.15, 0.0, 1.0)
            slack = (0.08 + 0.12 * gate) * rng
            min_self_u = max(self._reserved + EPS, target - slack)
        else:
            min_self_u = self._reserved + EPS
        viable = [o for o in candidates if self._u(o) >= min_self_u]
        if not viable:
            viable = [o for o in candidates if self._u(o) > self._reserved + EPS]
            if not viable:
                return self._best
        candidates = viable

        self._last_t = t
        best_o: Outcome | None = None
        best_score = -float("inf")
        scored: list[tuple[float, Outcome]] = []
        # v7.0 (evolved, Run6 best, combined_score=+0.007884 vs v4.9 baseline):
        # Variance-based frequency-holder detection, one-sided f3, patience-dependent
        # late-game relaxation. Targets hard_cell_gain against BOA/MAP and Boulware/Tough.
        last_opp_offer = self._received_offers[-1] if self._received_offers else None
        best_for_us_opp_ou = 0.5
        n_opp = len(self._received_offers)
        if self._received_offers:
            best_su = -float("inf")
            for oo in self._received_offers[-30:]:
                su_oo = self._u(oo)
                if su_oo > best_su:
                    best_su = su_oo
                    best_for_us_opp_ou = self._internal_opponent_score(oo)
        # Signal 1: momentum — slope of self._u over opponent's recent offers
        opp_slope = 0.0
        if n_opp >= 4:
            rec = self._received_offers[-min(12, n_opp):]
            n = len(rec)
            sx = sy = sxy = sx2 = 0.0
            for i, oo in enumerate(rec):
                x, y = float(i), self._u(oo)
                sx += x; sy += y; sxy += x * y; sx2 += x * x
            d = n * sx2 - sx * sx
            if abs(d) > EPS:
                opp_slope = (n * sxy - sx * sy) / (d * max(EPS, rng))
        # Signal 2: magnitude — total opponent concession in their own utility
        total_conc = 0.0
        if n_opp >= 2:
            first_ou = self._internal_opponent_score(self._received_offers[0])
            last_ou = self._internal_opponent_score(self._received_offers[-1])
            total_conc = max(0.0, first_ou - last_ou)
        # patience: 1.0 when opponent conceded nothing (Boulware/Tough), 0.0 when >=0.20
        patience = max(0.0, min(1.0, (0.20 - total_conc) / 0.20))
        # Signal 1b: opponent utility concession slope (their own terms)
        ou_slope = 0.0
        if n_opp >= 4:
            rec_ou = self._received_offers[-min(12, n_opp):]
            n_ou = len(rec_ou)
            sx = sy = sxy = sx2 = 0.0
            for i, oo in enumerate(rec_ou):
                x, y = float(i), self._internal_opponent_score(oo)
                sx += x; sy += y; sxy += x * y; sx2 += x * x
            d = n_ou * sx2 - sx * sx
            if abs(d) > EPS:
                ou_slope = (n_ou * sxy - sx * sy) / (d * max(EPS, rng))
        # Variance of recent opponent self-utility offers (frequency-holder detection)
        _recent_su = [self._u(oo) for oo in self._received_offers[-min(8, n_opp):]]
        _su_std = 1.0
        if len(_recent_su) >= 4:
            _mean_su = sum(_recent_su) / len(_recent_su)
            _var_su = sum((x - _mean_su) ** 2 for x in _recent_su) / len(_recent_su)
            _su_std = _var_su ** 0.5
        # Pareto frontier approximation (O(n log n))
        _sorted_idx = sorted(range(len(candidates)),
                             key=lambda _j: self._u(candidates[_j]), reverse=True)
        _pareto_front = set()
        _run_max_ou = -1.0
        for _j in _sorted_idx:
            _ou_j = self._internal_opponent_score(candidates[_j])
            if _ou_j > _run_max_ou:
                _pareto_front.add(_j)
                _run_max_ou = _ou_j
        # Adaptive mid-game peak: shifts later vs hardliners (patience~1 -> peak at 0.73)
        _mg_peak = max(0.0, 1.0 - abs(t - (0.67 + 0.06 * patience)) / 0.18)
        # Discrete concession-step detector (tighter threshold: 0.05)
        _conc_step = 0.0
        _su_hist = []
        if n_opp >= 3:
            _su_hist = [self._u(oo) for oo in self._received_offers[-4:]]
            _max_jump = 0.0
            for _si in range(1, len(_su_hist)):
                _j = _su_hist[_si] - _su_hist[_si - 1]
                if _j > _max_jump:
                    _max_jump = _j
            _conc_step = min(1.0, _max_jump / 0.05)
        # Hard-hold classifier: opponent not conceding (Boulware/Tough/early freq)
        is_hardliner = (patience > 0.75) and (opp_slope < 0.005) and (t > 0.20)
        # Frequency-holder detector: opponents that hold steady then concede late (BOA, MAP)
        is_freq_holder = (not is_hardliner) and (patience > 0.3) and (opp_slope < 0.005) and (_su_std < 0.015) and (t > 0.45) and (t < 0.78)
        # concession-step factor for hardliners: only if the step is a real upward move
        _hl_conc = _conc_step if (is_hardliner and _conc_step > 0.3 and _su_hist and _su_hist[-1] > _su_hist[-2]) else 0.0
        # mid-game hold amplifier (narrow peak, damped by concession step)
        _mid_hold = max(0.0, 1.0 - abs(t - 0.65) / 0.08) * (1.0 - _hl_conc * 0.5) if is_hardliner else 0.0
        # mid-game hold for frequency holders (shifted later, narrower)
        _freq_mid = max(0.0, 1.0 - abs(t - 0.68) / 0.10) if is_freq_holder else 0.0
        # late-game relaxation: patience-dependent start for hardliners
        _late_start = 0.88 + 0.05 * patience
        _late = max(0.0, min(1.0, (t - _late_start) / 0.07)) if (is_hardliner and t > _late_start) else 0.0
        for _i, o in enumerate(candidates):
            su = self._u(o)
            ou = self._internal_opponent_score(o)
            closeness = 1.0 - min(1.0, abs(su - target) / max(EPS, band * 2.5))
            diversity = self._diversity_bonus(o)
            novelty = self._novelty_bonus(o)
            opponent_exact = self._opponent_offer_bonus(o)
            repeat_penalty = self._repeat_penalty(o)
            su_norm = (su - self._reserved) / rng
            e_ramp = 0.42 * max(0.0, (t - (0.72 + 0.06 * patience)) / 0.28) ** 1.8
            s_mod = -0.07 * min(1.0, max(0.0, opp_slope) / 0.02)
            opp_weight = max(0.10, 0.18 + 0.08 * t + e_ramp + s_mod - 0.08 * patience)
            self_weight = 0.48 - 0.10 * t + max(0.0, -s_mod) * 0.4 + 0.06 * patience + 0.04 * _conc_step + 0.06 * is_hardliner
            novelty_weight = 0.10 + 0.05 * (1.0 - t)
            opponent_exact_weight = (0.08 + 0.08 * t) * (1.0 - patience) * (1.0 - 0.5 * _conc_step)
            p_peak = 0.65 + 0.08 * patience
            p_width = 0.18 + 0.04 * patience
            pareto_weight = 0.12 + 0.09 * max(0.0, 1.0 - abs(t - p_peak) / p_width)
            _frontier_bonus = (0.02 + 0.04 * _mg_peak + 0.06 * is_hardliner) if _i in _pareto_front else 0.0
            if is_hardliner:
                self_weight += 0.20 * (1.0 - _hl_conc * 0.6) * (1.0 - _late * 0.3)
                self_weight += 0.08 * _mid_hold
                opponent_exact_weight = 0.06 * _hl_conc + 0.10 * _late
                f3_weight = 1.0 - 0.4 * _hl_conc - 0.3 * _late
                if _i in _pareto_front:
                    _frontier_bonus += 0.08 * (1.0 - 0.5 * _hl_conc) + 0.04 * _mid_hold
            elif is_freq_holder:
                self_weight += 0.12 * _freq_mid * (1.0 - _late * 0.3)
                opponent_exact_weight *= 0.3
                f3_weight = min(0.85, 0.4 + 0.3 * t + 0.15 * patience)
                if _i in _pareto_front:
                    _frontier_bonus += 0.06 * _freq_mid
            else:
                f3_weight = min(0.65, 0.1 + 0.2 * t + 0.15 * patience)
                if _conc_step > 0.2:
                    f3_weight = max(0.05, f3_weight - 0.15)
                if (t > 0.55 and t < 0.75 and 0.3 < patience < 0.8
                        and opp_slope < 0.003 and ou_slope > 0.001):
                    f3_weight = min(0.85, f3_weight + 0.2)
                    self_weight += 0.05
                    opponent_exact_weight *= 0.4
                    pareto_weight += 0.03
            if last_opp_offer is not None:
                ou_last = self._internal_opponent_score(last_opp_offer)
                f2_val = min(1.0, 1.0 + ou - ou_last)
            else:
                f2_val = ou
            if is_hardliner or is_freq_holder:
                f3_val = 1.0 - max(0.0, best_for_us_opp_ou - ou)
            else:
                f3_val = 1.0 - min(1.0, abs(ou - best_for_us_opp_ou))
            opponent_component = (1.0 - f3_weight) * f2_val + f3_weight * f3_val
            score = (
                self_weight * su_norm
                + opp_weight * opponent_component
                + pareto_weight * su_norm * max(0.0, ou)
                + 0.20 * closeness
                + 0.07 * diversity
                + novelty_weight * novelty
                + opponent_exact_weight * opponent_exact
                - 0.14 * repeat_penalty
                + _frontier_bonus
                + self.rng.random() * 0.002
            )
            if score > best_score:
                best_score = score
                best_o = o
            scored.append((score, o))
        # B2: adversarial tie-break — among near-best candidates, propose the one
        # that least improves the opponent's (simulated) frequency model of our
        # ranking. Advantage-neutral by construction: only reorders candidates
        # within 0.015 of the best combined score. Gated to update_memory=True
        # so respond()'s next_u stays deterministic and cheap.
        if (
            self._adversarial_tiebreak
            and update_memory
            and best_o is not None
            and self._mirror_pairs
            and len(self._mirror.offers) >= 5
        ):
            near = [(s, o) for s, o in scored if s >= best_score - 0.015]
            if len(near) >= 2:
                names = self._space.issue_names
                best_o = min(
                    near,
                    key=lambda so: self._mirror_concordance_after(
                        _outcome_to_values(so[1], names)
                    ),
                )[1]
        return best_o or self._best

    def _mirror_concordance_after(self, values: tuple[Any, ...]) -> float:
        """Approximate Kendall agreement between our true ranking and the mirror
        model (a frequency-based opponent's simulated view of us) if we proposed
        `values`. Uses a temporary in-place count increment, reverted before
        returning, to avoid copying the model per candidate."""
        counts = self._mirror.value_counts
        for i, v in enumerate(values):
            if i < len(counts):
                counts[i][v] += 1
        try:
            names = self._space.issue_names
            agree = 0
            total = 0
            for a, b in self._mirror_pairs:
                ua, ub = self._u(a), self._u(b)
                ma = self._mirror(a, names)
                mb = self._mirror(b, names)
                if abs(ma - mb) <= 1e-9:
                    continue
                total += 1
                if (ua > ub) == (ma > mb):
                    agree += 1
            return (agree / total) if total else 0.5
        finally:
            for i, v in enumerate(values):
                if i < len(counts):
                    counts[i][v] -= 1

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


# Submission entry class is LionelWei (matches Agent Name / Agent Class fields).
# Aliases below keep older import paths working; all point at the same class.
LionelNegNegotiator = LionelWei
LionelWeiAgent = LionelWei
MyNegotiator = LionelWei
ShadowBathNegotiator = LionelWei
