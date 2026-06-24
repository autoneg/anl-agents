"""
ChangAgent for ANL2026 / NegMAS SAO negotiations.

Design goals requested by the user:
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
       "Rethinking Frequency Opponent Modeling in Automated Negotiation";
   - adds a lightweight transition/recency model and confidence-aware ensemble.
5. v3 improvements:
   - online opponent-type classification;
   - forecast-aware acceptance in the middle/late phases;
   - Pareto-like candidate pool, UCB arm bias, and controlled repetition;
   - privacy-aware probe/decoy scoring inside the current utility band.

No third-party dependency beyond NegMAS/ANL2026 is required.
"""

from __future__ import annotations

import json
import math
import os
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
__version__ = "4.4.6-chang"




def _load_tunable_overrides() -> dict[str, Any]:
    """Optional local-tuning hook.

    During official submission this environment variable is normally unset and
    defaults are used. Local tuning scripts can set CHANG_AGENT_PARAMS to a JSON
    object, for example {"valley_fraction": 0.66, "late_start": 0.80}.
    The agent does not write or persist anything, so this does not create
    cross-negotiation memory.
    """
    raw = os.environ.get("CHANG_AGENT_PARAMS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
    return default


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




@dataclass
class OppEval:
    """Opponent utility estimate with uncertainty and confidence.

    mean: estimated opponent utility in [0, 1].
    var: disagreement among submodels.
    conf: amount/reliability of evidence in [0, 1].
    """

    mean: float = 0.5
    var: float = 0.04
    conf: float = 0.05


@dataclass
class ArmStats:
    n: int = 0
    mean: float = 0.35

    def update(self, reward: float) -> None:
        reward = _clip(float(reward), 0.0, 1.0)
        self.n += 1
        self.mean += (reward - self.mean) / max(1, self.n)


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

    def confidence(self) -> float:
        n = len(self.offers)
        domain_factor = max(1, self.space.n_issues)
        return _clip(0.08 + n / max(6.0, 3.0 * domain_factor + 3.0), 0.08, 0.85)

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

    def confidence(self) -> float:
        n = len(self.offers)
        if n < 2 * self.window_size:
            return _clip(0.05 + n / max(1.0, 4.0 * self.window_size), 0.05, 0.35)
        windows = n / max(1.0, self.window_size)
        return _clip(0.20 + windows / 8.0, 0.20, 0.90)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        values = _outcome_to_values(outcome, issue_names or self.space.issue_names)
        if not self.weights or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self.weights[i] if i < len(self.weights) else 1.0 / max(1, len(values))
            total += w * self.value_score(i, v)
        return _clip(total, 0.0, 1.0)


class TransitionRatioOpponentModel:
    """Lightweight transition/recency frequency model.

    This expert gives more weight to values proposed recently and to issues that
    the opponent keeps stable while changing other issues. It is not a full
    Bayesian model; it is a cheap correction to classical frequency estimates.
    """

    def __init__(self, space: OpponentIssueSpace, decay: float = 0.965):
        self.space = space
        self.decay = _clip(decay, 0.85, 0.995)
        self.value_mass: list[Counter[Any]] = []
        self.stability: list[float] = []
        self.change: list[float] = []
        self.weights: list[float] = []
        self.offers: list[tuple[Any, ...]] = []
        self._resize()

    def _resize(self) -> None:
        n = self.space.n_issues
        while len(self.value_mass) < n:
            self.value_mass.append(Counter())
        while len(self.stability) < n:
            self.stability.append(0.0)
        while len(self.change) < n:
            self.change.append(0.0)
        if len(self.weights) != n:
            self.weights = [1.0 / n for _ in range(n)] if n else []

    def update(self, outcome_values: tuple[Any, ...], relative_time: float = 0.0) -> None:
        self.space.ensure_from_outcome(outcome_values)
        self._resize()
        prev = self.offers[-1] if self.offers else None
        # Exponential recency: recent values matter, but old anchors do not vanish.
        for c in self.value_mass:
            for k in list(c.keys()):
                c[k] *= self.decay
                if c[k] < 1e-8:
                    del c[k]
        self.offers.append(outcome_values)
        for i, v in enumerate(outcome_values):
            self.value_mass[i][v] += 1.0
            if prev is not None and i < len(prev):
                if prev[i] == v:
                    self.stability[i] += 1.0 - 0.25 * relative_time
                else:
                    self.change[i] += 1.0
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        raw: list[float] = []
        for i, c in enumerate(self.value_mass):
            total = sum(c.values())
            dominance = max(c.values()) / total if total > EPS else 0.0
            rigidity = self.stability[i] / max(1.0, self.change[i] + self.stability[i])
            raw.append(1.0 + 0.65 * rigidity + 0.35 * dominance)
        self.weights = _normalize_positive(raw)

    def value_score(self, issue_index: int, value: Any) -> float:
        if issue_index >= len(self.value_mass):
            return 0.5
        c = self.value_mass[issue_index]
        if not c:
            return 0.5
        mx = max(c.values())
        return _clip((c.get(value, 0.0) + 0.35) / (mx + 0.35), 0.0, 1.0)

    def confidence(self) -> float:
        n = len(self.offers)
        return _clip(0.05 + n / 14.0, 0.05, 0.82)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        values = _outcome_to_values(outcome, issue_names or self.space.issue_names)
        if not self.weights or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self.weights[i] if i < len(self.weights) else 1.0 / max(1, len(values))
            total += w * self.value_score(i, v)
        return _clip(total, 0.0, 1.0)


class NeighborSimilarityOpponentModel:
    """Outcome-neighborhood opponent model.

    Frequency models can undervalue an outcome that combines values the opponent
    likes but has not proposed as an exact bundle. This expert scores complete
    outcomes by similarity to the opponent's recent/frequent offers, giving a
    mild boost to candidates that sit near the opponent's observed neighborhood.
    """

    def __init__(self, space: OpponentIssueSpace, recency_decay: float = 0.965):
        self.space = space
        self.recency_decay = _clip(recency_decay, 0.88, 0.995)
        self.offers: list[tuple[Any, ...]] = []
        self.counts: Counter[tuple[Any, ...]] = Counter()

    def update(self, outcome_values: tuple[Any, ...], relative_time: float = 0.0) -> None:
        self.space.ensure_from_outcome(outcome_values)
        self.offers.append(outcome_values)
        self.counts[outcome_values] += 1

    def confidence(self) -> float:
        n = len(self.offers)
        diversity = len(self.counts) / max(1, n)
        # High diversity means the neighborhood evidence covers more of the space.
        return _clip(0.04 + n / 16.0 + 0.18 * diversity, 0.04, 0.78)

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        values = _outcome_to_values(outcome, issue_names or self.space.issue_names)
        if not values or not self.offers:
            return 0.5
        n_total = len(self.offers)
        max_count = max(self.counts.values()) if self.counts else 1
        best_support = 0.0
        # Recent offers matter most, but keep a little memory of earlier repeated
        # anchors because BOA/Genius-style agents may return to them late.
        for idx, seen in enumerate(self.offers[-18:]):
            n = max(1, min(len(values), len(seen)))
            sim = sum(1 for a, b in zip(values[:n], seen[:n]) if a == b) / n
            if sim <= 0.0:
                continue
            recency = (idx + 1) / min(18, n_total)
            freq = self.counts.get(seen, 1) / max(1, max_count)
            support = sim * (0.48 + 0.34 * recency + 0.18 * freq)
            if support > best_support:
                best_support = support
        # Similarity is evidence of acceptability, not a full utility oracle. Keep
        # the range moderate so this expert nudges rather than dominates.
        return _clip(0.28 + 0.62 * best_support, 0.0, 0.94)


class CombinedFrequencyOpponentModel:
    """Confidence-aware ensemble of frequency, transition, and neighborhood experts."""

    def __init__(self, space: OpponentIssueSpace, window_size: int = 4, neighbor_weight_scale: float = 1.0):
        self.space = space
        self.smith = SmithFrequencyOpponentModel(space)
        self.dfm = DistributionBasedFrequencyOpponentModel(space, window_size=window_size)
        self.transition = TransitionRatioOpponentModel(space)
        self.neighbor = NeighborSimilarityOpponentModel(space)
        self.neighbor_weight_scale = _clip(neighbor_weight_scale, 0.0, 2.0)
        self.offers: list[tuple[Any, ...]] = []
        self.window_size = max(2, int(window_size))
        self.last_eval = OppEval()

    def update(self, outcome: Outcome, relative_time: float = 0.0) -> None:
        values = _outcome_to_values(outcome, self.space.issue_names)
        if not values:
            return
        self.space.ensure_from_outcome(values)
        self.offers.append(values)
        self.smith.update(values, relative_time)
        self.dfm.update(values, relative_time)
        self.transition.update(values, relative_time)
        self.neighbor.update(values, relative_time)

    @property
    def dfm_weight(self) -> float:
        n = len(self.offers)
        confidence = _clip((n - self.window_size) / max(1, 3 * self.window_size), 0.0, 1.0)
        return 0.35 + 0.30 * confidence

    def confidence(self) -> float:
        neighbor_prior = 0.10 * self.neighbor.confidence() * self.neighbor_weight_scale
        return _clip(
            0.27 * self.smith.confidence()
            + 0.34 * self.dfm.confidence()
            + 0.29 * self.transition.confidence()
            + neighbor_prior,
            0.05,
            0.92,
        )

    def evaluate(self, outcome: Outcome, issue_names: list[str] | None = None) -> OppEval:
        names = issue_names or self.space.issue_names
        wd = self.dfm_weight
        smith_v = self.smith(outcome, names)
        dfm_v = self.dfm(outcome, names)
        transition_v = self.transition(outcome, names)
        neighbor_v = self.neighbor(outcome, names)
        classical_support = max(smith_v, dfm_v, transition_v)
        # Treat neighborhood similarity as a calibration expert, not as a
        # standalone oracle. It should speak loudly only when at least one
        # frequency/recency expert also thinks the candidate is plausible.
        neighbor_agreement = _clip((classical_support - 0.42) / 0.34, 0.0, 1.0)
        vals = [smith_v, dfm_v, transition_v, neighbor_v]
        confs = [
            self.smith.confidence() * (1.10 - 0.35 * wd),
            self.dfm.confidence() * (0.65 + 0.75 * wd),
            self.transition.confidence() * 0.88,
            self.neighbor.confidence() * (0.10 + 0.38 * neighbor_agreement) * self.neighbor_weight_scale,
        ]
        z = sum(max(1e-6, c) for c in confs)
        ws = [max(1e-6, c) / z for c in confs]
        mean = sum(w * v for w, v in zip(ws, vals))
        var = sum(w * (v - mean) ** 2 for w, v in zip(ws, vals)) + 0.0008
        conf = self.confidence() * (1.0 - min(0.45, 2.0 * math.sqrt(var)))
        out = OppEval(mean=_clip(mean, 0.0, 1.0), var=_clip(var, 0.0, 0.25), conf=_clip(conf, 0.03, 0.92))
        self.last_eval = out
        return out

    def __call__(self, outcome: Outcome, issue_names: list[str] | None = None) -> float:
        return self.evaluate(outcome, issue_names).mean

    def issue_weights(self) -> dict[str, float]:
        wd = self.dfm_weight
        wt = 0.22 + 0.10 * self.transition.confidence()
        names = self.space.issue_names or [f"i{i}" for i in range(self.space.n_issues)]
        out: dict[str, float] = {}
        for i, name in enumerate(names):
            sw = self.smith.weights[i] if i < len(self.smith.weights) else 0.0
            dw = self.dfm.weights[i] if i < len(self.dfm.weights) else 0.0
            tw = self.transition.weights[i] if i < len(self.transition.weights) else 0.0
            out[name] = (1.0 - wd - wt / 2.0) * sw + wd * dw + wt * tw
        return out


class OpponentClassifier:
    """Small online rule-based classifier for meta-strategy switching."""

    def __init__(self) -> None:
        self.kind = "unknown"
        self.confidence = 0.0

    def classify(
        self,
        received_utils: Sequence[float],
        received_keys: Sequence[tuple[Any, ...]],
        self_offer_utils: Sequence[float],
        utility_range: float,
    ) -> tuple[str, float]:
        n = len(received_utils)
        if n < 4:
            self.kind, self.confidence = "unknown", _clip(n / 4.0, 0.0, 0.35)
            return self.kind, self.confidence
        rng = max(EPS, utility_range)
        deltas = [received_utils[i] - received_utils[i - 1] for i in range(1, n)]
        monotone = sum(d >= -0.015 * rng for d in deltas) / max(1, len(deltas))
        total_improve = received_utils[-1] - received_utils[0]
        avg_improve = total_improve / max(1, n - 1)
        repeat_rate = 0.0
        if received_keys:
            c = Counter(received_keys)
            repeat_rate = max(c.values()) / max(1, len(received_keys))
        half = max(2, n // 2)
        early = received_utils[:half]
        late = received_utils[-half:]
        early_slope = (early[-1] - early[0]) / max(1, len(early) - 1)
        late_slope = (late[-1] - late[0]) / max(1, len(late) - 1)
        reciprocity = 0.0
        if len(self_offer_utils) >= 3 and len(received_utils) >= 3:
            m = min(len(self_offer_utils), len(received_utils)) - 1
            # If we concede (our utility decreases), TFT-like opponents often improve their next offer to us.
            xs = [-(self_offer_utils[-m + i] - self_offer_utils[-m + i - 1]) for i in range(1, m)]
            ys = [(received_utils[-m + i] - received_utils[-m + i - 1]) for i in range(1, m)]
            if xs and ys:
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                vx = sum((x - mx) ** 2 for x in xs)
                vy = sum((y - my) ** 2 for y in ys)
                if vx > EPS and vy > EPS:
                    corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)
                    reciprocity = _clip((corr + 1.0) / 2.0, 0.0, 1.0)

        if repeat_rate > 0.72 and total_improve < 0.08 * rng:
            kind = "hardheaded"
            conf = 0.65 + 0.25 * repeat_rate
        elif reciprocity > 0.67 and n >= 6:
            kind = "tft"
            conf = 0.50 + 0.35 * reciprocity
        elif monotone > 0.74 and avg_improve > 0.004 * rng:
            if late_slope > max(0.001 * rng, 1.8 * max(early_slope, 0.0)):
                kind = "boulware"
            else:
                kind = "linear"
            conf = 0.45 + 0.35 * monotone + min(0.12, abs(avg_improve) / rng)
        elif repeat_rate > 0.45:
            kind = "hardheaded"
            conf = 0.45 + 0.30 * repeat_rate
        else:
            volatility = sum(abs(d) for d in deltas) / max(EPS, abs(total_improve) + 0.02 * rng)
            kind = "portfolio_like" if volatility > 2.8 else "unknown"
            conf = 0.35 + min(0.35, volatility / 10.0)
        self.kind, self.confidence = kind, _clip(conf, 0.0, 0.92)
        return self.kind, self.confidence


class OpponentUtilityAdapter:
    """Callable utility-like wrapper exposed for external evaluators if needed."""

    def __init__(self, negotiator: "ChangAgent"):
        self.negotiator = negotiator
        self.reserved_value = 0.0

    def __call__(self, outcome: Outcome) -> float:
        return self.negotiator.estimate_opponent_utility(outcome)


class _ChangAgentCustom(SAONegotiator):
    """ANL2026 negotiator using bathtub bidding + segmented acceptance."""

    def __init__(
        self,
        *args: Any,
        window_size: int = 4,
        valley_time: float = 0.66,
        early_end: float = 0.38,
        late_start: float = 0.79,
        valley_fraction: float = 0.64,
        candidate_pool_limit: int = 12000,
        rng_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Local tuning hook. Command-line tournament scripts can set
        # CHANG_AGENT_PARAMS without changing this file. Unknown keys are ignored.
        _ov = _load_tunable_overrides()
        window_size = int(_ov.get("window_size", window_size))
        valley_time = float(_ov.get("valley_time", valley_time))
        early_end = float(_ov.get("early_end", early_end))
        late_start = float(_ov.get("late_start", late_start))
        valley_fraction = float(_ov.get("valley_fraction", valley_fraction))
        candidate_pool_limit = int(_ov.get("candidate_pool_limit", candidate_pool_limit))
        if "rng_seed" in _ov:
            try:
                rng_seed = int(_ov["rng_seed"]) if _ov["rng_seed"] is not None else None
            except Exception:
                rng_seed = None

        self.window_size = max(2, int(window_size))
        self.valley_time = _clip(valley_time, 0.45, 0.8)
        self.early_end = _clip(early_end, 0.2, 0.65)
        self.late_start = _clip(late_start, self.valley_time + 0.05, 0.95)
        self.valley_fraction = _clip(valley_fraction, 0.35, 0.85)
        self.candidate_pool_limit = max(500, int(candidate_pool_limit))
        self.rng = random.Random(rng_seed)

        # Acceptance tuning. These are deliberately conservative compared with v3.
        self.accept_floor_fraction = _clip(float(_ov.get("accept_floor_fraction", 0.24)), 0.05, 0.50)
        self.middle_regret_fraction = _clip(float(_ov.get("middle_regret_fraction", 0.035)), 0.00, 0.12)
        self.late_regret_fraction = _clip(float(_ov.get("late_regret_fraction", 0.085)), 0.02, 0.22)
        self.final_regret_fraction = _clip(float(_ov.get("final_regret_fraction", 0.135)), 0.04, 0.30)
        self.wait_bias_fraction = _clip(float(_ov.get("wait_bias_fraction", 0.018)), 0.00, 0.06)
        self.proposal_floor_fraction = _clip(float(_ov.get("proposal_floor_fraction", 0.56)), 0.20, 0.75)
        self.middle_accept_floor_fraction = _clip(float(_ov.get("middle_accept_floor_fraction", 0.52)), 0.15, 0.65)
        self.anchor_guard_boost = _clip(float(_ov.get("anchor_guard_boost", 0.09)), 0.00, 0.18)
        self.anchor_accept_guard = _clip(float(_ov.get("anchor_accept_guard", 0.0)), 0.00, 1.00)
        self.anchor_floor_hold = _clip(float(_ov.get("anchor_floor_hold", 0.0)), 0.00, 1.00)
        self.offer_self_weight_scale = _clip(float(_ov.get("offer_self_weight_scale", 1.0)), 0.50, 1.80)
        self.offer_opp_weight_scale = _clip(float(_ov.get("offer_opp_weight_scale", 1.0)), 0.40, 1.60)
        self.offer_closeness_scale = _clip(float(_ov.get("offer_closeness_scale", 1.0)), 0.40, 1.80)
        self.offer_opponent_exact_scale = _clip(float(_ov.get("offer_opponent_exact_scale", 1.0)), 0.20, 1.80)
        self.offer_accept_prob_scale = _clip(float(_ov.get("offer_accept_prob_scale", 1.0)), 0.20, 1.80)
        self.offer_repeat_penalty_scale = _clip(float(_ov.get("offer_repeat_penalty_scale", 1.0)), 0.40, 2.20)
        self.counter_anchor_scale = _clip(float(_ov.get("counter_anchor_scale", 0.0)), 0.00, 2.00)
        self.counter_anchor_min_issues = max(1, int(_ov.get("counter_anchor_min_issues", 1)))
        self.opponent_neighbor_weight_scale = _clip(float(_ov.get("opponent_neighbor_weight_scale", 1.0)), 0.00, 2.00)
        self.offer_noise_scale = _clip(float(_ov.get("offer_noise_scale", 1.0)), 0.00, 2.00)
        self.candidate_prune_mode = str(_ov.get("candidate_prune_mode", "random")).strip().lower()
        if self.candidate_prune_mode not in {"random", "deterministic"}:
            self.candidate_prune_mode = "random"

        self._space = OpponentIssueSpace()
        self._model = CombinedFrequencyOpponentModel(
            self._space,
            window_size=self.window_size,
            neighbor_weight_scale=self.opponent_neighbor_weight_scale,
        )
        self.opponent_model = self._model
        # NegMAS/ANL2026 exposes opponent_ufun as a read-only property in some
        # versions. Keep the adapter in a private field and expose it via the
        # property below instead of assigning self.opponent_ufun here.
        self._opponent_ufun_adapter = OpponentUtilityAdapter(self)

        self._outcomes: list[Outcome] = []
        self._outcome_values: dict[Outcome, float] = {}
        self._outcome_key_cache: dict[Outcome, tuple[Any, ...]] = {}
        self._opp_eval_cache: dict[Outcome, OppEval] = {}
        self._self_sorted_cache: list[Outcome] | None = None
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
        self._self_offer_utils: list[float] = []
        self._opp_classifier = OpponentClassifier()
        self._opp_type = "unknown"
        self._opp_type_conf = 0.0
        self._last_selected_arm = "self_best"
        self._last_arm_offer: Outcome | None = None
        self._last_arm_offer_u = 0.0
        self._last_received_u_before_arm: float | None = None
        self._arm_stats: dict[str, ArmStats] = {
            "self_best": ArmStats(mean=0.42),
            "opp_seen": ArmStats(mean=0.46),
            "pareto": ArmStats(mean=0.48),
            "counter_anchor": ArmStats(mean=0.47),
            "probe": ArmStats(mean=0.34),
            "decoy": ArmStats(mean=0.33),
        }
        self._debug_last: dict[str, Any] = {}
        self._initialized = False

    @property
    def opponent_ufun(self) -> OpponentUtilityAdapter:
        """Estimated opponent utility for ANL2026 evaluators.

        Some NegMAS versions define this name as a read-only property on the
        base negotiator. Overriding it as a getter keeps the ANL2026-facing
        API while avoiding AttributeError during construction.
        """
        return self._opponent_ufun_adapter

    # ---------- NegMAS callbacks ----------

    def on_preferences_changed(self, changes: Any) -> None:  # called once ufun is attached/changed
        self._initialize_cached_domain(force=True)
        try:
            super().on_preferences_changed(changes)
        except Exception:
            pass

    def respond(self, state: Any, offer: Outcome | str | None = None, source: str = "") -> Any:
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
        target = self._target_utility(state)
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        curve = self._dynamic_curve_params()
        early_end = min(self.early_end, curve.get("early_end", self.early_end))
        late_start = curve.get("late_start", self.late_start)

        threshold, accept_reason = self._chang_acceptance_threshold(
            offer=offer,
            next_offer=next_offer,
            state=state,
            early_end=early_end,
            late_start=late_start,
        )
        forecast_accept = self._chang_accept_now_vs_wait(offer, next_offer, state, threshold)

        self._debug_last.update({
            "t": t, "opp_type": self._opp_type, "opp_type_conf": self._opp_type_conf,
            "accept_offer_u": offer_u, "accept_threshold": threshold,
            "forecast_accept": forecast_accept, "next_u": next_u,
            "target": target, "accept_reason": accept_reason,
        })
        # Strict early behavior is intentional: do not reveal the true boundary and
        # do not accept offers worse than the offer we are about to make.
        if t < early_end:
            return ResponseType.ACCEPT_OFFER if offer_u + 1e-12 >= threshold else ResponseType.REJECT_OFFER

        # Middle/late behavior: accept either when the deterministic threshold is
        # met, or when the explicit wait-vs-accept calculation says that waiting is
        # not worth the risk.  The forecast path is still guarded by the threshold
        # minus a small, phase-dependent regret allowance, so v4 does not accept
        # very weak offers merely because the deadline is close.
        if offer_u + 1e-12 >= threshold or forecast_accept:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


    def propose(self, state: Any) -> Outcome:
        self._initialize_cached_domain()
        offer = self._select_offer(state, update_memory=True)
        if offer is None:
            # Last-resort fallback. NegMAS NMI normally supports random_outcomes.
            try:
                offer = self.nmi.random_outcomes(1)[0]
            except Exception:
                offer = self._best
        self._last_proposal = offer
        self._last_proposal_u = self._u(offer)
        self._last_arm_offer = offer
        self._last_arm_offer_u = self._last_proposal_u
        self._last_received_u_before_arm = self._received_utils[-1] if self._received_utils else None
        if offer is not None:
            self._remember_self_offer(offer)
        return offer

    # ---------- Public opponent-estimate helpers ----------

    def _values_of(self, outcome: Outcome) -> tuple[Any, ...]:
        """Cached stable tuple representation for outcomes in the current domain."""
        try:
            cached = self._outcome_key_cache.get(outcome)
        except TypeError:
            return _outcome_to_values(outcome, self._space.issue_names)
        if cached is not None:
            return cached
        values = _outcome_to_values(outcome, self._space.issue_names)
        try:
            self._outcome_key_cache[outcome] = values
        except TypeError:
            pass
        return values

    def _opp_eval(self, outcome: Outcome) -> OppEval:
        try:
            cached = self._opp_eval_cache.get(outcome)
        except TypeError:
            return self._model.evaluate(outcome, self._space.issue_names)
        if cached is not None:
            return cached
        ev = self._model.evaluate(outcome, self._space.issue_names)
        try:
            self._opp_eval_cache[outcome] = ev
        except TypeError:
            pass
        return ev

    def estimate_opponent_utility(self, outcome: Outcome) -> float:
        return self._opp_eval(outcome).mean

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
        self._outcome_key_cache.clear()
        self._opp_eval_cache.clear()
        self._self_sorted_cache = None
        if self._outcomes:
            for o in self._outcomes:
                self._outcome_values[o] = self._u(o)
                self._space.ensure_from_outcome(self._values_of(o))
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
        # Keep reservation inside the observed/scaled utility interval.
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
            # Preserve extremes by sampling but leave enough diversity.
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
        values = self._values_of(offer)
        if not values:
            return
        self._recent_self_offers.append(offer)
        self._self_offer_counts[values] += 1
        self._self_offer_keys.append(values)
        self._self_offer_utils.append(self._u(offer))

    def _record_opponent_offer(self, offer: Outcome, state: Any) -> None:
        values = self._values_of(offer)
        if not values:
            return
        step = int(_safe_float(getattr(state, "step", len(self._received_offers)), len(self._received_offers)))
        key = (step, values)
        # Prevent repeated respond() calls for the same state from duplicating data.
        if key == self._last_seen_key:
            return
        self._last_seen_key = key
        self._received_offers.append(offer)
        self._received_utils.append(self._u(offer))
        self._received_keys.append(values)
        self._offer_counts[values] += 1
        self._model.update(offer, self._relative_time(state))
        self._opp_eval_cache.clear()
        self._update_opponent_type()
        self._update_arm_reward_from_response(offer)

    def _estimated_rejection_risk(self, offer: Outcome | None) -> float:
        if offer is None or not self._received_offers:
            return 0.5
        ev = self._opp_eval(offer)
        proposed_score = ev.mean
        hist_scores = [self._opp_eval(o).mean for o in self._received_offers[-max(3, self.window_size) :]]
        anchor = sum(hist_scores) / max(1, len(hist_scores))
        uncertainty = math.sqrt(max(0.0, ev.var))
        # Higher uncertainty and below-anchor opponent estimate imply more rejection risk.
        return _clip((anchor - proposed_score + 0.12 + 0.25 * uncertainty) / 0.32, 0.0, 1.0)

    def _update_opponent_type(self) -> None:
        rng = max(EPS, self._max_u - self._reserved)
        kind, conf = self._opp_classifier.classify(
            self._received_utils, self._received_keys, self._self_offer_utils, rng
        )
        # Smooth type switching: avoid a single noisy window changing behavior.
        if kind != self._opp_type and conf < max(0.42, self._opp_type_conf + 0.08):
            return
        self._opp_type, self._opp_type_conf = kind, conf

    def _dynamic_curve_params(self) -> dict[str, float]:
        n_issues = max(1, self._space.n_issues)
        conf = self._model.confidence()
        kind = self._opp_type
        valley_time = self.valley_time
        late_start = self.late_start
        early_end = self.early_end
        valley_shift = 0.0
        probe_budget = 0.055

        if kind == "hardheaded":
            valley_time += 0.035
            late_start += 0.035
            valley_shift -= 0.020
            probe_budget += 0.025
        elif kind == "boulware":
            valley_time += 0.020
            late_start += 0.025
            valley_shift -= 0.005
        elif kind == "linear":
            valley_time -= 0.030
            late_start -= 0.040
            valley_shift += 0.015
        elif kind == "tft":
            valley_time -= 0.055
            late_start -= 0.075
            early_end -= 0.050
            valley_shift += 0.030
            probe_budget -= 0.020
        elif kind == "portfolio_like":
            valley_time -= 0.020
            late_start -= 0.030
            probe_budget += 0.010

        if n_issues >= 5:
            valley_time += 0.015
            probe_budget += 0.010
        if conf > 0.62:
            late_start -= 0.018
        return {
            "valley_time": _clip(valley_time, 0.42, 0.78),
            "late_start": _clip(late_start, 0.58, 0.92),
            "early_end": _clip(early_end, 0.22, 0.55),
            "valley_shift": _clip(valley_shift, -0.06, 0.06),
            "probe_budget": _clip(probe_budget, 0.015, 0.10),
        }

    def _opponent_anchor_pressure(self) -> float:
        """Detect repeated low-utility anchors from stubborn/model-based agents."""
        if len(self._received_keys) < 4 or not self._received_utils:
            return 0.0
        total = len(self._received_keys)
        key, count = self._offer_counts.most_common(1)[0]
        repeat_ratio = count / max(1, total)
        rng = max(EPS, self._max_u - self._reserved)
        common_utils = [
            self._received_utils[i]
            for i, k in enumerate(self._received_keys)
            if k == key and i < len(self._received_utils)
        ]
        anchor_u = min(common_utils) if common_utils else min(self._received_utils)
        low_anchor = _clip((self._reserved + 0.58 * rng - anchor_u) / max(EPS, 0.34 * rng), 0.0, 1.0)
        repeat_pressure = _clip((repeat_ratio - 0.35) / 0.45, 0.0, 1.0)
        return _clip(0.65 * repeat_pressure + 0.35 * low_anchor, 0.0, 1.0)

    def _repeated_anchor_offer_pressure(self, offer: Outcome) -> float:
        """Pressure score for the current offer being a repeated low self-utility anchor."""
        values = self._values_of(offer)
        if not values or len(self._received_keys) < 5:
            return 0.0
        count = self._offer_counts.get(values, 0)
        if count < 3:
            return 0.0
        total = len(self._received_keys)
        rng = max(EPS, self._max_u - self._reserved)
        offer_u = self._u(offer)
        repeat_ratio = count / max(1, total)
        low_anchor = _clip((self._reserved + 0.58 * rng - offer_u) / max(EPS, 0.30 * rng), 0.0, 1.0)
        latest_idx = max((i for i, key in enumerate(self._received_keys) if key == values), default=-1)
        recency = (latest_idx + 1) / max(1, total)
        repeat_pressure = _clip((repeat_ratio - 0.28) / 0.42, 0.0, 1.0)
        recency_pressure = _clip((recency - 0.45) / 0.45, 0.0, 1.0)
        return _clip(0.50 * repeat_pressure + 0.35 * low_anchor + 0.15 * recency_pressure, 0.0, 1.0)

    def _utility_guard_floor(self, state: Any, late_start: float | None = None, for_proposal: bool = False) -> float:
        """A time-decaying self-utility floor for utility-focused robustness.

        The normal target curve is intentionally adaptive, but strong BOA/Genius
        style opponents can repeat their own optimum until the target/anchor logic
        learns a valley that is too low too early. This guard keeps early and
        middle offers/acceptances above a modest utility floor, then relaxes near
        the deadline to preserve agreement rate.
        """
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        if late_start is None:
            late_start = self._dynamic_curve_params().get("late_start", self.late_start)

        start_frac = self.proposal_floor_fraction if for_proposal else self.middle_accept_floor_fraction
        late_floor = max(self.accept_floor_fraction, 0.30 if for_proposal else 0.28)
        if t < 0.55:
            frac = start_frac
        elif t < late_start:
            progress = _clip((t - 0.55) / max(EPS, late_start - 0.55), 0.0, 1.0)
            pressure = self._opponent_anchor_pressure()
            if pressure > 0.0 and self.anchor_floor_hold > 0.0:
                progress *= 1.0 - self.anchor_floor_hold * pressure
            frac = start_frac * (1.0 - progress) + (late_floor + 0.04) * progress
        else:
            urgency = _clip((t - late_start) / max(EPS, 1.0 - late_start), 0.0, 1.0)
            frac = (late_floor + 0.04) * (1.0 - urgency) + self.accept_floor_fraction * urgency

        pressure = self._opponent_anchor_pressure()
        if pressure > 0.0 and t < 0.78:
            frac += self.anchor_guard_boost * pressure * (1.0 - t / 0.78)
        return self._reserved + _clip(frac, self.accept_floor_fraction, 0.75) * rng

    def _chang_acceptance_threshold(
        self,
        offer: Outcome,
        next_offer: Outcome | None,
        state: Any,
        early_end: float,
        late_start: float,
    ) -> tuple[float, str]:
        """Conservative segmented acceptance threshold for ChangAgent.

        The previous Chamel version could become too permissive late because it
        subtracted a large deadline gap from the next proposal.  ChangAgent uses
        a bounded-regret threshold: it can accept below the next own proposal,
        but only within a regret cap relative to either the planned proposal,
        the dynamic target, and the best offer already observed.
        """
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        target = self._target_utility(state)
        next_u = self._u(next_offer) if next_offer is not None else target
        offer_u = self._u(offer)
        best_seen = max(self._received_utils) if self._received_utils else self._reserved
        recent_q = _percentile(self._received_utils[-8:], 0.70, default=best_seen) if self._received_utils else self._reserved
        robust_floor = max(
            self._reserved + self.accept_floor_fraction * rng,
            self._utility_guard_floor(state, late_start=late_start, for_proposal=False),
        )
        anchor_offer_pressure = self._repeated_anchor_offer_pressure(offer)
        if self.anchor_accept_guard > 0.0 and anchor_offer_pressure > 0.0 and t < 0.94:
            if t < late_start:
                strict_frac = self.middle_accept_floor_fraction
            else:
                urgency = _clip((t - late_start) / max(EPS, 1.0 - late_start), 0.0, 1.0)
                strict_frac = self.middle_accept_floor_fraction * (1.0 - urgency) + 0.36 * urgency
            strict_floor = self._reserved + strict_frac * rng
            time_relax = 1.0 - _clip((t - 0.78) / 0.16, 0.0, 1.0)
            anti_anchor_floor = robust_floor + (
                strict_floor - robust_floor
            ) * self.anchor_accept_guard * anchor_offer_pressure * time_relax
            robust_floor = max(robust_floor, anti_anchor_floor)

        # If the opponent has actually offered us good deals, avoid accepting far
        # below that anchor.  Near the very end this anchor is relaxed.
        best_anchor = max(robust_floor, 0.55 * best_seen + 0.45 * recent_q)

        if t < early_end:
            return max(robust_floor, next_u - 1e-10, best_anchor - 0.006 * rng), "early_strict_acnext"

        if t < late_start:
            # Middle phase: use a min of next proposal and target, but guard it
            # with the best-seen anchor. This prevents premature acceptance of
            # mediocre offers while still accepting genuinely strong opponent
            # concessions.
            progress = _clip((t - early_end) / max(EPS, late_start - early_end), 0.0, 1.0)
            regret = (0.010 + self.middle_regret_fraction * progress) * rng
            anchor_relax = (0.006 + 0.020 * progress) * rng
            threshold = max(
                robust_floor,
                min(next_u, target + 0.018 * rng) - regret,
                best_anchor - anchor_relax,
            )
            # If the current offer is exactly something the opponent has repeated
            # and it is already close to our target, allow acceptance slightly
            # earlier to secure agreement.
            if (
                self._opponent_offer_bonus(offer) > 0.65
                and offer_u >= max(robust_floor, target - 0.030 * rng)
            ):
                threshold = max(robust_floor, min(threshold, offer_u))
            return _clip(threshold, robust_floor, self._max_u), "middle_bounded_regret"

        # Late phase: can accept below next_u, but not below a dynamic floor tied
        # to best history.  We relax smoothly and only become aggressive in the
        # last few percent of time.
        urgency = _clip((t - late_start) / max(EPS, 1.0 - late_start), 0.0, 1.0)
        regret = (self.late_regret_fraction * (urgency ** 1.25)) * rng
        final_regret = (self.final_regret_fraction * max(0.0, (t - 0.94) / 0.06) ** 1.7) * rng
        model_risk = self._estimated_rejection_risk(next_offer)
        risk_regret = (0.010 + 0.055 * model_risk) * (urgency ** 1.10) * rng
        anchor_relax = (0.025 + 0.085 * urgency + 0.075 * max(0.0, (t - 0.96) / 0.04)) * rng
        threshold = max(
            robust_floor,
            min(next_u, target + 0.010 * rng) - regret - risk_regret - final_regret,
            best_anchor - anchor_relax,
        )
        return _clip(threshold, robust_floor, self._max_u), "late_bounded_regret"

    def _chang_accept_now_vs_wait(
        self,
        current_offer: Outcome,
        next_offer: Outcome | None,
        state: Any,
        threshold: float,
    ) -> bool:
        """Accept-vs-wait gate with a regret guard.

        This replaces the more permissive v3 forecast shortcut.  Forecast
        acceptance is only allowed when the current offer is within a small,
        time-dependent band below the deterministic threshold and waiting has
        weak expected value.
        """
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        current_u = self._u(current_offer)
        if t < 0.50:
            return False

        next_u = self._u(next_offer) if next_offer is not None else self._target_utility(state)
        p_next = self._accept_probability(next_offer, state) if next_offer is not None else 0.0
        future_received = self._forecast_best_future_received_u(state, horizon=2 if t > 0.80 else 3)
        safe_floor = self._reserved + self.accept_floor_fraction * rng
        wait_u = max(future_received, p_next * next_u + (1.0 - p_next) * safe_floor)

        deadline_discount = (self.wait_bias_fraction + 0.075 * (t ** 4.0)) * rng
        if self._opp_type == "hardheaded":
            deadline_discount += 0.015 * rng
        if self._opp_type == "tft" and t < 0.82:
            deadline_discount -= 0.010 * rng

        allowed_below_threshold = (0.008 + 0.075 * max(0.0, (t - 0.75) / 0.25) ** 1.6) * rng
        guard = max(safe_floor, threshold - allowed_below_threshold)
        self._debug_last.update({
            "chang_wait_u": wait_u,
            "chang_p_next": p_next,
            "chang_future_received": future_received,
            "chang_forecast_guard": guard,
        })
        return current_u >= guard and current_u + deadline_discount >= wait_u

    def _accept_probability(self, offer: Outcome | None, state: Any) -> float:
        if offer is None:
            return 0.0
        ev = self._opp_eval(offer)
        t = self._relative_time(state)
        seen = self._opponent_offer_bonus(offer)
        sim = self._similarity_to_recent_opponent_offers(offer)
        uncertainty = math.sqrt(max(0.0, ev.var))
        # Hand-tuned logistic prior. It is deliberately conservative early.
        z = (
            self.accept_z_bias
            + self.accept_ev_weight * ev.mean
            + self.accept_seen_weight * seen
            + self.accept_sim_weight * sim
            + self.accept_time_weight * t
            + self.accept_conf_weight * ev.conf
            - self.accept_uncertainty_weight * uncertainty
        )
        if self._opp_type == "hardheaded":
            z -= self.accept_hardheaded_penalty * (1.0 - t)
        elif self._opp_type == "tft":
            z += self.accept_tft_sim_bonus * sim
        try:
            return _clip(1.0 / (1.0 + math.exp(-z)), 0.0, 1.0)
        except OverflowError:
            return 0.0 if z < 0 else 1.0

    def _forecast_best_future_received_u(self, state: Any, horizon: int = 3) -> float:
        if not self._received_utils:
            return self._reserved
        rng = max(EPS, self._max_u - self._reserved)
        recent = self._received_utils[-max(3, min(6, len(self._received_utils))) :]
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        else:
            slope = 0.0
        t = self._relative_time(state)
        optimism = 0.50 + 0.35 * (self._opp_type in {"linear", "tft"}) - 0.20 * (self._opp_type == "hardheaded")
        forecast = max(recent) + max(0.0, slope) * horizon * optimism * (1.0 - 0.35 * t)
        return _clip(forecast, self._reserved, self._max_u)

    def _accept_now_vs_wait(self, current_offer: Outcome, next_offer: Outcome | None, state: Any, phase: str) -> bool:
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        current_u = self._u(current_offer)
        next_u = self._u(next_offer) if next_offer is not None else self._target_utility(state)
        p_next = self._accept_probability(next_offer, state) if next_offer is not None else 0.0
        future_received = self._forecast_best_future_received_u(state, horizon=3 if phase == "middle" else 2)
        floor = self._reserved + 0.012 * rng
        wait_u = max(future_received, p_next * next_u + (1.0 - p_next) * floor)
        # Waiting near the deadline is risky. Early/middle waiting is less risky.
        deadline_risk = (0.008 + 0.070 * (t ** 3.0)) * rng
        if phase == "late":
            deadline_risk += 0.025 * rng
        model_risk = self._estimated_rejection_risk(next_offer) * (0.010 + 0.035 * t) * rng
        margin = (0.008 if phase == "middle" else 0.018) * rng
        self._debug_last.update({
            "wait_u": wait_u, "p_next": p_next, "future_received_u": future_received
        })
        return current_u + deadline_risk + model_risk + margin >= wait_u

    def _update_arm_reward_from_response(self, opponent_offer: Outcome) -> None:
        if not self._last_arm_offer or self._last_selected_arm not in self._arm_stats:
            return
        rng = max(EPS, self._max_u - self._reserved)
        current_received_u = self._u(opponent_offer)
        prev_received_u = self._last_received_u_before_arm
        concession = 0.0 if prev_received_u is None else max(0.0, current_received_u - prev_received_u) / rng
        last_norm = _clip((self._last_arm_offer_u - self._reserved) / rng, 0.0, 1.0)
        risk = self._estimated_rejection_risk(self._last_arm_offer)
        reward = 0.22 * last_norm + 0.46 * _clip(concession / 0.08, 0.0, 1.0) + 0.32 * (1.0 - risk)
        self._arm_stats[self._last_selected_arm].update(reward)

    def _arm_ucb(self, arm: str) -> float:
        st = self._arm_stats.get(arm)
        if st is None:
            return 0.0
        total = 1 + sum(a.n for a in self._arm_stats.values())
        explore = 0.18 * math.sqrt(math.log(total + 2.0) / (st.n + 1.0))
        # Avoid excessive probing/decoy in the last part.
        return st.mean + explore

    def _similarity_to_recent_opponent_offers(self, outcome: Outcome) -> float:
        vals = self._values_of(outcome)
        if not vals or not self._received_keys:
            return 0.0
        sims: list[float] = []
        for pvals in self._received_keys[-6:]:
            n = max(1, min(len(vals), len(pvals)))
            same = sum(1 for a, b in zip(vals[:n], pvals[:n]) if a == b) / n
            sims.append(same)
        return _clip(max(sims) if sims else 0.0, 0.0, 1.0)


    # ---------- Bathtub target and offering ----------

    def _valley_utility(self) -> float:
        """Dynamic bottom of the bathtub target.

        The fixed ``valley_fraction`` is only the prior. After receiving opponent
        offers, the bottom is pulled toward a weighted estimate derived from the
        utilities of exact offers the opponent actually proposed. Offers get more
        influence when they are recent, repeated, and compatible with the current
        opponent model. This keeps the middle concession level adaptive: a generous
        opponent raises the valley, while a tough opponent lowers it without going
        below a safety floor above our reservation value.
        """
        rng = max(EPS, self._max_u - self._reserved)
        prior = self._reserved + self.valley_fraction * rng
        if not self._received_offers:
            return prior

        weighted_utils: list[float] = []
        weights: list[float] = []
        n = len(self._received_offers)
        max_count = max(self._offer_counts.values()) if self._offer_counts else 1
        for idx, offer in enumerate(self._received_offers):
            key = self._values_of(offer)
            su = self._u(offer)
            recency = (idx + 1) / max(1, n)
            frequency = self._offer_counts.get(key, 1) / max(1, max_count)
            opp_support = self.estimate_opponent_utility(offer)
            w = 0.35 + 0.45 * recency + 0.30 * frequency + 0.25 * opp_support
            weighted_utils.append(su)
            weights.append(w)

        wsum = sum(weights)
        if wsum <= EPS:
            return prior
        wmean = sum(u * w for u, w in zip(weighted_utils, weights)) / wsum
        wq65 = self._weighted_quantile(weighted_utils, weights, 0.65, default=wmean)
        best_received = max(self._received_utils) if self._received_utils else wmean

        # The bottom should represent an attainable but not desperate middle target.
        # Weighted mean reflects the opponent's actual behavior; q65 rewards repeated
        # better offers; best_received has a limited effect so one lucky offer does not
        # dominate the whole curve.
        history_anchor = 0.50 * wmean + 0.35 * wq65 + 0.15 * best_received
        confidence = _clip(n / max(1, 2.5 * self.window_size), 0.0, 1.0)
        valley = (1.0 - confidence) * prior + confidence * history_anchor
        valley += self._dynamic_curve_params().get("valley_shift", 0.0) * rng

        safety_floor = self._reserved + 0.30 * rng
        safety_cap = self._max_u - 0.12 * rng
        return _clip(valley, safety_floor, safety_cap)

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
        """Learns the upward tail of the bathtub from the current negotiation.

        The late peak is conservative when the opponent has never offered anything
        good for us, and rises when the history/model indicates attainable outcomes
        with both decent self utility and decent opponent-estimated utility.
        """
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

        # Outcome-space support: what is the highest own utility among outcomes the
        # model thinks the opponent may still like?
        attainable = valley
        if self._outcomes:
            hist_opp_scores = [self._opp_eval(o).mean for o in self._received_offers]
            opp_cut = max(0.40, _percentile(hist_opp_scores, 0.45, 0.5) - 0.04)
            sampled = self._top_self_candidates(limit=min(len(self._outcomes), 2500))
            for o in sampled:
                ev = self._opp_eval(o)
                robust_opp = ev.mean - 0.20 * math.sqrt(max(0.0, ev.var))
                if robust_opp >= opp_cut:
                    attainable = max(attainable, self._u(o))

        behavior_peak = best_received + 0.05 * rng + 1.25 * concession_trend
        model_peak = attainable
        confidence = _clip(len(self._received_offers) / max(1, 3 * self.window_size), 0.0, 1.0)
        peak = valley + confidence * (0.55 * model_peak + 0.45 * behavior_peak - valley)
        # Avoid pathological optimism: do not jump all the way to max unless history
        # or model strongly supports it.
        cap = self._max_u - 0.015 * rng
        return _clip(peak, valley, cap)

    def _target_utility(self, state: Any) -> float:
        t = self._relative_time(state)
        high = self._max_u
        valley = self._valley_utility()
        late_peak = self._late_peak_utility()
        curve = self._dynamic_curve_params()
        valley_time = curve.get("valley_time", self.valley_time)
        if t <= valley_time:
            x = _clip(t / max(EPS, valley_time), 0.0, 1.0)
            # Smooth fall from high to valley.
            fall = math.sin(0.5 * math.pi * x) ** 0.72
            target = high - (high - valley) * fall
        else:
            x = _clip((t - valley_time) / max(EPS, 1.0 - valley_time), 0.0, 1.0)
            # Smooth late rise. Peak is dynamic, inferred above.
            rise = math.sin(0.5 * math.pi * x) ** 1.15
            target = valley + (late_peak - valley) * rise
        return _clip(target, self._reserved, self._max_u)

    def _top_self_candidates(self, limit: int = 2500) -> list[Outcome]:
        if not self._outcomes:
            return []
        if len(self._outcome_values) < len(self._outcomes):
            for o in self._outcomes:
                self._outcome_values.setdefault(o, self._u(o))
        if self._self_sorted_cache is None:
            self._self_sorted_cache = sorted(
                self._outcomes,
                key=lambda o: self._outcome_values.get(o, self._reserved),
                reverse=True,
            )
        return self._self_sorted_cache[: min(limit, len(self._self_sorted_cache))]

    def _select_offer(self, state: Any, update_memory: bool = True) -> Outcome | None:
        if not self._outcomes:
            return self._best
        target = self._target_utility(state)
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        curve = self._dynamic_curve_params()
        band = (0.024 + 0.060 * (1.0 - min(t, 0.82))) * rng
        lower = max(self._reserved, target - band)
        upper = min(self._max_u, target + 0.72 * band)
        guard_floor = self._utility_guard_floor(
            state,
            late_start=curve.get("late_start", self.late_start),
            for_proposal=True,
        )
        lower = max(lower, guard_floor)
        if upper < lower:
            upper = min(self._max_u, lower + max(0.35 * band, 0.012 * rng))

        candidates: dict[Outcome, set[str]] = {}

        def add(o: Outcome | None, arm: str) -> None:
            if o is None:
                return
            try:
                u = self._u(o)
            except Exception:
                return
            if u + 1e-12 < max(self._reserved, guard_floor - 0.012 * rng):
                return
            candidates.setdefault(o, set()).add(arm)

        # 1) Current utility-band candidates.
        for o in self._outcomes:
            u = self._u(o)
            if lower <= u <= upper:
                add(o, "self_best")

        # 2) Exact opponent offers and near-opponent anchors.
        opponent_slack = (0.018 + 0.085 * t) * rng
        guarded_lower = max(self._reserved, lower - opponent_slack)
        for o in reversed(self._received_offers[-24:]):
            if self._u(o) >= guarded_lower:
                add(o, "opp_seen")

        # 3) Pareto-like candidates from high self-utility subset, robust to model variance.
        top_subset = self._top_self_candidates(limit=min(len(self._outcomes), 2500))
        pareto_seed: list[tuple[float, Outcome]] = []
        opp_cut = 0.42
        if self._received_offers:
            opp_hist = [self._opp_eval(o).mean for o in self._received_offers]
            opp_cut = max(0.38, _percentile(opp_hist, 0.40, 0.5) - 0.05)
        for o in top_subset:
            su = self._u(o)
            if su < guarded_lower:
                continue
            ev = self._opp_eval(o)
            robust_opp = ev.mean - 0.25 * math.sqrt(max(0.0, ev.var))
            if robust_opp >= opp_cut or abs(su - target) <= 2.2 * band:
                social = su + 0.72 * robust_opp * rng
                pareto_seed.append((social, o))
        pareto_seed.sort(key=lambda x: x[0], reverse=True)
        for _, o in pareto_seed[:90]:
            add(o, "pareto")

        # 3b) Counter repeated low anchors with offers that look good to the
        # opponent while keeping our utility above the proposal guard.
        anchor_pressure = self._opponent_anchor_pressure()
        if (
            self.counter_anchor_scale > 0.0
            and anchor_pressure > 0.20
            and self._space.n_issues >= self.counter_anchor_min_issues
            and self._offer_counts
        ):
            anchor_values, _ = self._offer_counts.most_common(1)[0]
            counter_floor = max(guard_floor, self._reserved + (0.48 + 0.06 * (t < 0.62)) * rng)
            counter_seed: list[tuple[float, Outcome]] = []
            for o in top_subset:
                su = self._u(o)
                if su < counter_floor:
                    continue
                vals = self._values_of(o)
                n = max(1, min(len(vals), len(anchor_values)))
                similarity = sum(1 for a, b in zip(vals[:n], anchor_values[:n]) if a == b) / n
                ev = self._opp_eval(o)
                robust_opp = ev.mean - 0.16 * math.sqrt(max(0.0, ev.var))
                if similarity < 0.34 and robust_opp < 0.58:
                    continue
                self_norm = (su - self._reserved) / rng
                score = (
                    0.34 * robust_opp
                    + 0.27 * similarity
                    + 0.22 * self_norm
                    + 0.10 * self._novelty_bonus(o)
                    + 0.07 * self._diversity_bonus(o)
                    - 0.08 * self._repeat_penalty(o, state)
                )
                counter_seed.append((score, o))
            counter_seed.sort(key=lambda x: x[0], reverse=True)
            for _, o in counter_seed[: max(12, int(45 * self.counter_anchor_scale))]:
                add(o, "counter_anchor")

        # 4) Probe/decoy candidates are still inside a safe utility band.
        if t < 0.28 and self._outcomes:
            budget = curve.get("probe_budget", 0.055) * rng
            probe_floor = max(self._reserved, target - budget)
            probe_pool = [o for o in top_subset[:500] if self._u(o) >= probe_floor]
            probe_pool.sort(key=lambda o: (self._diversity_bonus(o), self._novelty_bonus(o)), reverse=True)
            for o in probe_pool[:18]:
                add(o, "probe")
        if 0.06 <= t <= 0.38:
            decoys = [o for o in list(candidates.keys()) if abs(self._u(o) - target) <= 1.4 * band]
            decoys.sort(key=lambda o: self._privacy_bonus(o, target, band), reverse=True)
            for o in decoys[:24]:
                add(o, "decoy")

        if len(candidates) < 8:
            ordered = sorted(self._outcomes, key=lambda o: abs(self._u(o) - target))
            for o in ordered[: min(100, len(ordered))]:
                add(o, "self_best")
            for o in reversed(self._received_offers[-24:]):
                if self._u(o) >= guarded_lower:
                    add(o, "opp_seen")

        if len(candidates) > 720:
            pinned = [o for o, arms in candidates.items() if "opp_seen" in arms]
            rest = [o for o in candidates if o not in pinned]
            need = max(0, 720 - len(pinned))
            if self.candidate_prune_mode == "deterministic" and len(rest) > need:
                def rough_prune_key(o: Outcome) -> tuple[float, float, float, str]:
                    su = self._u(o)
                    ev = self._opp_eval(o)
                    robust_opp = ev.mean - 0.20 * math.sqrt(max(0.0, ev.var))
                    self_norm = (su - self._reserved) / rng
                    closeness = 1.0 - min(1.0, abs(su - target) / max(EPS, band * 2.8))
                    score = (
                        0.54 * self_norm
                        + 0.30 * robust_opp
                        + 0.18 * closeness
                        + 0.08 * self._opponent_offer_bonus(o)
                        + 0.04 * self._novelty_bonus(o)
                        - 0.04 * self._repeat_penalty(o, state)
                    )
                    return (score, self._diversity_bonus(o), self._u(o), repr(self._values_of(o)))

                kept = pinned + sorted(rest, key=rough_prune_key, reverse=True)[:need]
            else:
                kept = pinned + (self.rng.sample(rest, need) if len(rest) > need else rest)
            candidates = {o: candidates[o] for o in kept}

        best_o: Outcome | None = None
        best_score = -float("inf")
        best_arm = "self_best"
        for o, arms in candidates.items():
            su = self._u(o)
            ev = self._opp_eval(o)
            ou_robust = ev.mean - 0.22 * math.sqrt(max(0.0, ev.var))
            closeness = 1.0 - min(1.0, abs(su - target) / max(EPS, band * 2.5))
            diversity = self._diversity_bonus(o)
            novelty = self._novelty_bonus(o)
            opponent_exact = self._opponent_offer_bonus(o)
            repeat_penalty = self._repeat_penalty(o, state)
            privacy = self._privacy_bonus(o, target, band) if "decoy" in arms or t < 0.36 else 0.0
            p_accept = self._accept_probability(o, state)

            # Meta-strategy weights.
            opp_weight = 0.22 + 0.34 * t
            self_weight = 0.49 - 0.13 * t
            if self._opp_type == "hardheaded":
                self_weight += 0.04
                opp_weight -= 0.03
            elif self._opp_type == "tft":
                opp_weight += 0.04
            elif self._opp_type == "linear":
                opp_weight += 0.02
            model_conf_factor = 0.55 + 0.45 * ev.conf
            opp_weight *= model_conf_factor
            self_weight *= self.offer_self_weight_scale
            opp_weight *= self.offer_opp_weight_scale

            arm = max(arms, key=lambda a: self._arm_ucb(a))
            arm_bonus = 0.050 * self._arm_ucb(arm)
            if arm == "opp_seen":
                arm_bonus += 0.025 + 0.035 * t
            elif arm == "pareto":
                arm_bonus += 0.015 + 0.020 * ev.conf
            elif arm == "counter_anchor":
                arm_bonus += self.counter_anchor_scale * (0.020 + 0.035 * anchor_pressure + 0.015 * ev.conf)
            elif arm == "probe":
                arm_bonus += max(0.0, 0.025 * (0.28 - t) / 0.28)
            elif arm == "decoy":
                arm_bonus += max(0.0, 0.025 * (0.38 - t) / 0.38)

            score = (
                self_weight * ((su - self._reserved) / rng)
                + opp_weight * ou_robust
                + (0.18 * self.offer_closeness_scale) * closeness
                + 0.055 * diversity
                + (0.095 + 0.045 * (1.0 - t)) * novelty
                + ((0.105 + 0.135 * t) * self.offer_opponent_exact_scale) * opponent_exact
                + (0.070 * self.offer_accept_prob_scale) * p_accept
                + 0.045 * privacy
                + arm_bonus
                - (0.135 * self.offer_repeat_penalty_scale) * repeat_penalty
                + self.rng.random() * 0.002 * self.offer_noise_scale
            )
            if score > best_score:
                best_score = score
                best_o = o
                best_arm = arm
        self._last_selected_arm = best_arm
        self._debug_last.update({"chosen_arm": best_arm, "candidate_count": len(candidates), "target": target})
        return best_o or self._best

    def _privacy_bonus(self, outcome: Outcome, target: float, band: float) -> float:
        su = self._u(outcome)
        if abs(su - target) > max(EPS, 1.7 * band):
            return 0.0
        vals = self._values_of(outcome)
        if not vals:
            return 0.0
        # Entropy gain: prefer offers that flatten our revealed value frequencies.
        gain = 0.0
        n_issues = max(1, len(vals))
        for i, v in enumerate(vals):
            counts = Counter(k[i] for k in self._self_offer_keys if i < len(k))
            before_total = sum(counts.values())
            before_max = max(counts.values()) if counts else 0
            before_dom = before_max / max(1, before_total)
            counts[v] += 1
            after_total = before_total + 1
            after_dom = max(counts.values()) / max(1, after_total)
            gain += max(0.0, before_dom - after_dom)
        entropy_gain = gain / n_issues
        # Leakage penalty: avoid repeatedly showing top self-utility pattern.
        novelty = self._novelty_bonus(outcome)
        exact_opp = self._opponent_offer_bonus(outcome)
        leakage = (1.0 - novelty) * 0.55 + max(0.0, (su - target) / max(EPS, band)) * 0.25
        return _clip(0.70 * entropy_gain + 0.25 * exact_opp - 0.45 * leakage + 0.35, 0.0, 1.0)

    def _novelty_bonus(self, outcome: Outcome) -> float:
        values = self._values_of(outcome)
        if not values:
            return 0.5
        count = self._self_offer_counts.get(values, 0)
        if count <= 0:
            return 1.0
        # Previously proposed outcomes are still allowed, but exact repeats lose
        # priority increasingly with the number of times we proposed them.
        return _clip(1.0 / (1.0 + count), 0.0, 1.0)

    def _opponent_offer_bonus(self, outcome: Outcome) -> float:
        values = self._values_of(outcome)
        if not values or not self._received_keys:
            return 0.0
        count = self._offer_counts.get(values, 0)
        if count <= 0:
            return 0.0
        max_count = max(self._offer_counts.values()) if self._offer_counts else 1
        # Recency matters: an offer proposed late by the opponent is usually a better
        # proxy for what they may accept now than an opening anchor.
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
        vals = self._values_of(outcome)
        if not vals:
            return 0.5
        distances: list[float] = []
        for prev in self._recent_self_offers:
            pvals = self._values_of(prev)
            n = max(1, min(len(vals), len(pvals)))
            mismatch = sum(1 for a, b in zip(vals[:n], pvals[:n]) if a != b) / n
            distances.append(mismatch)
        return _clip(sum(distances) / max(1, len(distances)), 0.0, 1.0)

    def _repeat_penalty(self, outcome: Outcome, state: Any | None = None) -> float:
        vals = self._values_of(outcome)
        if not vals:
            return 0.0
        recent_count = sum(1 for o in self._recent_self_offers if self._values_of(o) == vals)
        global_count = self._self_offer_counts.get(vals, 0)
        recent_penalty = recent_count / max(1, len(self._recent_self_offers))
        global_penalty = min(1.0, global_count / 4.0)
        base = 0.65 * recent_penalty + 0.35 * global_penalty
        # Controlled repetition: if an outcome is likely acceptable or exactly/near
        # an opponent anchor, do not punish repetition too hard.
        anchor = self._opponent_offer_bonus(outcome)
        if state is not None:
            p_acc = self._accept_probability(outcome, state)
        else:
            p_acc = 1.0 - self._estimated_rejection_risk(outcome)
        relief = 0.55 * max(anchor, p_acc)
        return _clip(base * (1.0 - relief), 0.0, 1.0)



class ChangAgent(_ChangAgentCustom):
    """Utility-focused ChangAgent with internalized Boulware/AC-Next control.

    The strong baseline behavior of CUHK/HardHeaded comes mostly from two
    simple mechanisms: an extremely Boulware time-dependent target and AC-Next
    acceptance. This class implements those mechanisms directly inside the
    original ChangAgent code path, while keeping ChangAgent's ensemble opponent
    model for tie-breaking among equally good self-utility bids.
    """

    def __init__(self, *args: Any, e: float = 0.08, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _ov = _load_tunable_overrides()
        self.boulware_e = _clip(float(_ov.get("boulware_e", e)), 0.025, 0.35)
        self.boulware_k = _clip(float(_ov.get("boulware_k", 0.0)), 0.0, 0.25)
        self.ac_time_threshold = _clip(float(_ov.get("ac_time_threshold", 1.0)), 0.90, 1.00)
        self.same_utility_band_fraction = _clip(
            float(_ov.get("same_utility_band_fraction", 0.012)), 0.001, 0.06
        )
        self.bayes_concession_weight = _clip(
            float(_ov.get("bayes_concession_weight", 0.10)), 0.0, 0.45
        )
        self.opponent_tiebreak_weight = _clip(
            float(_ov.get("opponent_tiebreak_weight", 0.42)), 0.0, 0.85
        )
        self.privacy_tiebreak_weight = _clip(
            float(_ov.get("privacy_tiebreak_weight", 0.10)), 0.0, 0.60
        )
        self.late_rebound_fraction = _clip(
            float(_ov.get("late_rebound_fraction", 0.35)), 0.0, 0.85
        )
        self.late_rebound_start = _clip(
            float(_ov.get("late_rebound_start", 0.955)), 0.82, 0.995
        )
        self.late_rebound_power = _clip(
            float(_ov.get("late_rebound_power", 1.45)), 0.35, 3.00
        )
        self.early_perturb_fraction = _clip(
            float(_ov.get("early_perturb_fraction", 0.0)), 0.0, 0.18
        )
        self.early_perturb_until = _clip(
            float(_ov.get("early_perturb_until", 0.16)), 0.02, 0.45
        )
        self.early_perturb_power = _clip(
            float(_ov.get("early_perturb_power", 1.25)), 0.35, 3.00
        )
        self.deadlock_rescue_start = _clip(
            float(_ov.get("deadlock_rescue_start", 0.979)), 0.88, 0.995
        )
        self.deadlock_rescue_enabled = _as_bool(_ov.get("deadlock_rescue_enabled"), False)
        self.deadlock_rescue_max_outcomes = max(
            3, int(_ov.get("deadlock_rescue_max_outcomes", 8))
        )
        self.deadlock_rescue_compromise_cap = _clip(
            float(_ov.get("deadlock_rescue_compromise_cap", 0.32)), 0.08, 0.45
        )
        self.deadlock_rescue_seen_cap = _clip(
            float(_ov.get("deadlock_rescue_seen_cap", 0.42)), 0.12, 0.70
        )
        self.domain_adaptive_mode = _as_bool(_ov.get("domain_adaptive_mode"), False)
        self.accept_calibration_mode = str(_ov.get("accept_calibration_mode", "") or "").lower()
        self.model_anneal_mode = _as_bool(_ov.get("model_anneal_mode"), False)
        self.layered_candidates_mode = _as_bool(_ov.get("layered_candidates_mode"), True)
        self.privacy_rotation_mode = _as_bool(_ov.get("privacy_rotation_mode"), False)
        self.opponent_ufun_calibration_mode = _as_bool(_ov.get("opponent_ufun_calibration_mode"), False)
        self.deadline_step_mode = _as_bool(_ov.get("deadline_step_mode"), False)

        self.domain_large_space_threshold = max(120, int(_ov.get("domain_large_space_threshold", 1200)))
        self.domain_small_space_threshold = max(3, int(_ov.get("domain_small_space_threshold", 48)))
        self.domain_adaptive_boulware_delta = _clip(
            float(_ov.get("domain_adaptive_boulware_delta", 0.012)), 0.0, 0.060
        )
        self.domain_adaptive_band_delta = _clip(
            float(_ov.get("domain_adaptive_band_delta", 0.004)), 0.0, 0.030
        )

        if self.accept_calibration_mode == "avg_v1":
            accept_defaults = {
                "accept_z_bias": -2.25,
                "accept_ev_weight": 3.35,
                "accept_seen_weight": 0.78,
                "accept_sim_weight": 0.72,
                "accept_time_weight": 0.74,
                "accept_conf_weight": 0.42,
                "accept_uncertainty_weight": 1.18,
                "accept_hardheaded_penalty": 0.30,
                "accept_tft_sim_bonus": 0.15,
            }
        else:
            accept_defaults = {
                "accept_z_bias": -2.15,
                "accept_ev_weight": 3.15,
                "accept_seen_weight": 0.90,
                "accept_sim_weight": 0.82,
                "accept_time_weight": 0.85,
                "accept_conf_weight": 0.45,
                "accept_uncertainty_weight": 1.25,
                "accept_hardheaded_penalty": 0.35,
                "accept_tft_sim_bonus": 0.18,
            }
        self.accept_z_bias = float(_ov.get("accept_z_bias", accept_defaults["accept_z_bias"]))
        self.accept_ev_weight = float(_ov.get("accept_ev_weight", accept_defaults["accept_ev_weight"]))
        self.accept_seen_weight = float(_ov.get("accept_seen_weight", accept_defaults["accept_seen_weight"]))
        self.accept_sim_weight = float(_ov.get("accept_sim_weight", accept_defaults["accept_sim_weight"]))
        self.accept_time_weight = float(_ov.get("accept_time_weight", accept_defaults["accept_time_weight"]))
        self.accept_conf_weight = float(_ov.get("accept_conf_weight", accept_defaults["accept_conf_weight"]))
        self.accept_uncertainty_weight = float(_ov.get("accept_uncertainty_weight", accept_defaults["accept_uncertainty_weight"]))
        self.accept_hardheaded_penalty = float(_ov.get("accept_hardheaded_penalty", accept_defaults["accept_hardheaded_penalty"]))
        self.accept_tft_sim_bonus = float(_ov.get("accept_tft_sim_bonus", accept_defaults["accept_tft_sim_bonus"]))

        self.model_anneal_min = _clip(float(_ov.get("model_anneal_min", 0.42)), 0.0, 1.0)
        self.model_anneal_power = _clip(float(_ov.get("model_anneal_power", 0.85)), 0.2, 3.0)
        self.model_anneal_uncertainty = _clip(float(_ov.get("model_anneal_uncertainty", 0.25)), 0.0, 1.0)

        self.layered_candidate_per_layer = max(1, int(_ov.get("layered_candidate_per_layer", 2)))
        self.layered_candidate_accept_threshold = _clip(
            float(_ov.get("layered_candidate_accept_threshold", 0.68)), 0.20, 0.95
        )
        self.layered_candidate_opp_threshold = _clip(
            float(_ov.get("layered_candidate_opp_threshold", 0.62)), 0.20, 0.95
        )
        self.layered_candidate_weight = _clip(float(_ov.get("layered_candidate_weight", 0.020)), 0.0, 0.20)
        self.layered_candidate_min_surplus = _clip(
            float(_ov.get("layered_candidate_min_surplus", 0.018)), 0.0, 0.20
        )

        self.privacy_rotation_weight = _clip(float(_ov.get("privacy_rotation_weight", 0.035)), 0.0, 0.25)
        self.privacy_rotation_band_fraction = _clip(
            float(_ov.get("privacy_rotation_band_fraction", 0.018)), 0.001, 0.08
        )

        self.opp_calibration_blend = _clip(float(_ov.get("opp_calibration_blend", 0.18)), 0.0, 0.60)
        self.opp_calibration_shrink = _clip(float(_ov.get("opp_calibration_shrink", 0.35)), 0.0, 0.80)

        self.deadline_step_start = _clip(float(_ov.get("deadline_step_start", 0.982)), 0.90, 0.998)
        self.deadline_step_floor_fraction = _clip(
            float(_ov.get("deadline_step_floor_fraction", 0.50)), 0.20, 0.90
        )
        self._domain_profile_cache: dict[str, float] | None = None

    def _domain_profile(self) -> dict[str, float]:
        if self._domain_profile_cache is not None:
            return self._domain_profile_cache
        self._initialize_cached_domain()
        rng = max(EPS, self._max_u - self._reserved)
        vals = [
            _clip((self._outcome_values.get(o, self._reserved) - self._reserved) / rng, 0.0, 1.0)
            for o in self._outcomes
        ]
        n = len(vals)
        if not vals:
            self._domain_profile_cache = {
                "n_outcomes": 0.0,
                "n_issues": float(max(0, self._space.n_issues)),
                "utility_mean": 0.0,
                "utility_std": 0.0,
                "top_density": 0.0,
                "near_top_density": 0.0,
                "small_discrete": 0.0,
                "large_space": 0.0,
                "peaky_self": 0.0,
            }
            return self._domain_profile_cache
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n)
        top_density = sum(v >= 0.96 for v in vals) / n
        near_top_density = sum(v >= 0.90 for v in vals) / n
        peaky_self = _clip((0.10 - near_top_density) / 0.10, 0.0, 1.0)
        self._domain_profile_cache = {
            "n_outcomes": float(n),
            "n_issues": float(max(0, self._space.n_issues)),
            "utility_mean": mean,
            "utility_std": math.sqrt(max(0.0, var)),
            "top_density": top_density,
            "near_top_density": near_top_density,
            "small_discrete": 1.0 if n <= self.domain_small_space_threshold else 0.0,
            "large_space": 1.0 if n >= self.domain_large_space_threshold else 0.0,
            "peaky_self": peaky_self,
        }
        return self._domain_profile_cache

    def _domain_adjusted_e_and_band(self) -> tuple[float, float]:
        e = self.boulware_e
        band_fraction = self.same_utility_band_fraction
        if not self.domain_adaptive_mode:
            return e, band_fraction
        profile = self._domain_profile()
        large = profile.get("large_space", 0.0)
        small = profile.get("small_discrete", 0.0)
        peaky = profile.get("peaky_self", 0.0)
        # Large/peaky domains can afford a little more candidate exploration while
        # staying stubborn; small domains need narrower bands to avoid dropping
        # below the best attainable self-utility tier.
        e += self.domain_adaptive_boulware_delta * (0.55 * large - 0.45 * small)
        band_fraction += self.domain_adaptive_band_delta * (0.70 * large + 0.30 * peaky - 0.50 * small)
        return _clip(e, 0.025, 0.35), _clip(band_fraction, 0.001, 0.06)

    def _opp_eval(self, outcome: Outcome) -> OppEval:
        ev = super()._opp_eval(outcome)
        if not self.opponent_ufun_calibration_mode:
            return ev
        sim = self._similarity_to_recent_opponent_offers(outcome)
        uncertainty = math.sqrt(max(0.0, ev.var))
        support = 0.65 * sim + 0.35 * ev.mean
        mean = (1.0 - self.opp_calibration_blend) * ev.mean + self.opp_calibration_blend * support
        mean = mean * (1.0 - self.opp_calibration_shrink * uncertainty) + 0.5 * self.opp_calibration_shrink * uncertainty
        conf = ev.conf * (1.0 - 0.35 * self.opp_calibration_shrink * uncertainty)
        return OppEval(mean=_clip(mean, 0.0, 1.0), var=ev.var, conf=_clip(conf, 0.03, 0.92))

    def _model_annealed_opponent_weight(self, state: Any, ev: OppEval) -> float:
        weight = self.opponent_tiebreak_weight
        if not self.model_anneal_mode:
            return weight
        t = self._relative_time(state)
        uncertainty = math.sqrt(max(0.0, ev.var))
        time_factor = self.model_anneal_min + (1.0 - self.model_anneal_min) * (t ** self.model_anneal_power)
        confidence_factor = 0.72 + 0.28 * ev.conf
        uncertainty_factor = 1.0 - self.model_anneal_uncertainty * uncertainty
        return weight * _clip(time_factor * confidence_factor * uncertainty_factor, 0.15, 1.05)

    def _privacy_rotation_bonus(self, outcome: Outcome, target: float, band: float) -> float:
        if not self.privacy_rotation_mode:
            return 0.0
        rng = max(EPS, self._max_u - self._reserved)
        if abs(self._u(outcome) - target) > max(band, self.privacy_rotation_band_fraction * rng):
            return 0.0
        vals = self._values_of(outcome)
        if not vals:
            return 0.0
        if not self._self_offer_keys:
            return 0.5
        pressure = 0.0
        for i, v in enumerate(vals):
            counts = Counter(k[i] for k in self._self_offer_keys if i < len(k))
            if not counts:
                continue
            pressure += counts.get(v, 0) / max(1, sum(counts.values()))
        exposure = pressure / max(1, len(vals))
        return _clip(1.0 - exposure, 0.0, 1.0)

    def _layered_candidates(
        self,
        admissible: Sequence[Outcome],
        effective_target: float,
        state: Any,
    ) -> list[Outcome]:
        if not self.layered_candidates_mode or not admissible:
            return []
        rng = max(EPS, self._max_u - self._reserved)
        target_norm = _clip((effective_target - self._reserved) / rng, 0.0, 1.0)
        layers = [
            (target_norm + self.layered_candidate_min_surplus, 1.01),
            (min(0.96, target_norm + 0.05), 1.01),
            (min(0.90, target_norm + 0.10), 1.01),
        ]
        selected: list[Outcome] = []
        seen: set[tuple[Any, ...]] = set()
        for lo, hi in layers:
            bucket: list[tuple[float, Outcome]] = []
            for o in admissible:
                su_norm = _clip((self._u(o) - self._reserved) / rng, 0.0, 1.0)
                if su_norm + 1e-12 < lo or su_norm > hi:
                    continue
                ev = self._opp_eval(o)
                robust_opp = ev.mean - 0.18 * math.sqrt(max(0.0, ev.var))
                p_accept = self._accept_probability(o, state)
                support = max(p_accept, robust_opp, self._opponent_offer_bonus(o))
                if support < self.layered_candidate_accept_threshold and robust_opp < self.layered_candidate_opp_threshold:
                    continue
                key = self._values_of(o)
                if key in seen:
                    continue
                bucket.append((0.62 * support + 0.25 * su_norm + 0.13 * self._novelty_bonus(o), o))
            for _, o in sorted(bucket, key=lambda x: x[0], reverse=True)[: self.layered_candidate_per_layer]:
                seen.add(self._values_of(o))
                selected.append(o)
        return selected

    def _bayesian_concession_signal(self) -> float:
        """Posterior mean of whether the opponent is conceding toward us.

        This is a lightweight Beta-Bernoulli behavior model: every positive
        improvement in received utility counts as concession evidence; flat or
        worse moves count as non-concession evidence. It avoids assuming a full
        parametric opponent utility function and is robust on short ANL runs.
        """
        if len(self._received_utils) < 4:
            return 0.5
        rng = max(EPS, self._max_u - self._reserved)
        alpha = 1.0
        beta = 1.0
        recent = self._received_utils[-18:]
        for prev, cur in zip(recent, recent[1:]):
            delta = (cur - prev) / rng
            if delta > 0.004:
                alpha += min(2.5, 0.7 + 12.0 * delta)
            else:
                beta += min(2.0, 0.7 + 6.0 * abs(delta))
        return _clip(alpha / max(EPS, alpha + beta), 0.0, 1.0)

    def _boulware_f(self, t: float) -> float:
        if self.boulware_e <= EPS:
            return self.boulware_k
        return self.boulware_k + (1.0 - self.boulware_k) * (t ** (1.0 / self.boulware_e))

    def _deadlock_compromise_utility(self) -> float | None:
        """Detect tiny, near-zero-sum domains with one low compromise outcome."""
        if not self._outcomes or len(self._outcomes) > self.deadlock_rescue_max_outcomes:
            return None
        rng = max(EPS, self._max_u - self._reserved)
        values = sorted({
            _clip((u - self._reserved) / rng, 0.0, 1.0)
            for u in (self._outcome_values.get(o, self._reserved) for o in self._outcomes)
        })
        if len(values) < 3 or values[0] > 0.03 or values[-1] < 0.97:
            return None
        low_compromises = [
            v for v in values
            if 0.04 <= v <= self.deadlock_rescue_compromise_cap
        ]
        if not low_compromises:
            return None
        compromise = max(low_compromises)
        if values[-1] - compromise < 0.45:
            return None
        return self._reserved + compromise * rng

    def _deadlock_rescue_target(self, t: float) -> float | None:
        if t < self.deadlock_rescue_start:
            return None
        compromise = self._deadlock_compromise_utility()
        if compromise is None:
            return None
        rng = max(EPS, self._max_u - self._reserved)
        if self._received_utils:
            best_seen = max(self._received_utils)
            if best_seen > self._reserved + self.deadlock_rescue_seen_cap * rng:
                return None
        return _clip(compromise, self._reserved, self._max_u)

    def _should_accept_deadlock_compromise(self, offer_u: float, t: float) -> bool:
        if not self.deadlock_rescue_enabled:
            return False
        if t < self.deadlock_rescue_start:
            return False
        compromise = self._deadlock_compromise_utility()
        if compromise is None:
            return False
        rng = max(EPS, self._max_u - self._reserved)
        if self._received_utils and max(self._received_utils) > self._reserved + self.deadlock_rescue_seen_cap * rng:
            return False
        if abs(offer_u - compromise) > max(0.006 * rng, 1e-6):
            return False
        repeated_compromise = sum(
            1 for u in self._received_utils
            if abs(u - compromise) <= max(0.006 * rng, 1e-6)
        )
        return repeated_compromise >= 2

    def _target_utility(self, state: Any) -> float:
        self._initialize_cached_domain()
        t = self._relative_time(state)
        rng = max(EPS, self._max_u - self._reserved)
        concession_signal = self._bayesian_concession_signal()

        # If the opponent is demonstrably conceding, hold a little harder. If not,
        # concede a hair earlier near the end. The effect is intentionally small:
        # Boulware shape remains the backbone.
        base_e, _ = self._domain_adjusted_e_and_band()
        e = base_e * (
            1.0
            - self.bayes_concession_weight * (concession_signal - 0.5)
            + 0.08 * (self._opp_type == "hardheaded")
        )
        e = _clip(e, 0.025, 0.35)
        f = self.boulware_k + (1.0 - self.boulware_k) * (t ** (1.0 / e))
        target = self._reserved + rng * (1.0 - f)
        if self.late_rebound_fraction > 0.0 and t > self.late_rebound_start:
            x = _clip((t - self.late_rebound_start) / max(EPS, 1.0 - self.late_rebound_start), 0.0, 1.0)
            rebound = self._reserved + rng * self.late_rebound_fraction * (x ** self.late_rebound_power)
            if self._received_utils:
                history_guard = 0.72 * max(self._received_utils) + 0.28 * rebound
                rebound = max(rebound, history_guard)
            target = max(target, rebound)
        if self.deadline_step_mode and t > self.deadline_step_start:
            step_floor = self._reserved + self.deadline_step_floor_fraction * rng
            best_seen = max(self._received_utils) if self._received_utils else self._reserved
            history_floor = 0.55 * best_seen + 0.45 * step_floor
            target = max(target, min(step_floor, history_floor))
        return _clip(target, self._reserved, self._max_u)

    def respond(self, state: Any, offer: Outcome | str | None = None, source: str = "") -> Any:
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
        t = self._relative_time(state)
        if t >= self.ac_time_threshold:
            return ResponseType.ACCEPT_OFFER
        if self._should_accept_deadlock_compromise(offer_u, t):
            self._debug_last.update({
                "t": t,
                "accept_offer_u": offer_u,
                "accept_reason": "deadlock_compromise_rescue",
                "target": self._target_utility(state),
            })
            return ResponseType.ACCEPT_OFFER

        next_offer = self._select_offer(state, update_memory=False)
        next_u = self._u(next_offer) if next_offer is not None else self._target_utility(state)
        self._debug_last.update({
            "t": t,
            "accept_offer_u": offer_u,
            "accept_threshold": next_u,
            "target": self._target_utility(state),
            "accept_reason": "internal_ac_next",
            "bayes_concession": self._bayesian_concession_signal(),
        })
        return ResponseType.ACCEPT_OFFER if offer_u + 1e-12 >= next_u else ResponseType.REJECT_OFFER

    def _select_offer(self, state: Any, update_memory: bool = True) -> Outcome | None:
        if not self._outcomes:
            self._initialize_cached_domain()
        if not self._outcomes:
            return self._best

        target = self._target_utility(state)
        rng = max(EPS, self._max_u - self._reserved)
        effective_target = target
        if update_memory and self.early_perturb_fraction > 0.0:
            t0 = self._relative_time(state)
            if t0 < self.early_perturb_until:
                fade = 1.0 - _clip(t0 / max(EPS, self.early_perturb_until), 0.0, 1.0)
                perturb = self.early_perturb_fraction * (fade ** self.early_perturb_power) * rng
                effective_target = max(self._reserved, target - perturb)
        top_subset = self._top_self_candidates(limit=len(self._outcomes))
        admissible = [o for o in top_subset if self._u(o) + 1e-12 >= effective_target]
        if not admissible:
            admissible = [self._best] if self._best is not None else top_subset[:1]
        if not admissible:
            return None

        closest_u = min(self._u(o) for o in admissible)
        _, band_fraction = self._domain_adjusted_e_and_band()
        band = max(0.0001 * rng, band_fraction * rng)
        near = [o for o in admissible if self._u(o) <= closest_u + band]
        if not near:
            near = admissible[:1]
        if self.layered_candidates_mode:
            existing = {self._values_of(o) for o in near}
            for o in self._layered_candidates(admissible, effective_target, state):
                key = self._values_of(o)
                if key not in existing:
                    existing.add(key)
                    near.append(o)

        t = self._relative_time(state)
        best_o: Outcome | None = None
        best_score = -float("inf")
        for o in near:
            su = self._u(o)
            ev = self._opp_eval(o)
            uncertainty = math.sqrt(max(0.0, ev.var))
            robust_opp = ev.mean - 0.18 * uncertainty
            closeness = 1.0 - min(1.0, abs(su - effective_target) / max(EPS, band + 0.001 * rng))
            p_accept = self._accept_probability(o, state)
            surplus = _clip((su - effective_target) / rng, 0.0, 1.0)
            score = (
                0.58 * closeness
                + self._model_annealed_opponent_weight(state, ev) * robust_opp
                + self.privacy_tiebreak_weight * self._privacy_bonus(o, target, band)
                + self.privacy_rotation_weight * self._privacy_rotation_bonus(o, target, band)
                + 0.10 * self._opponent_offer_bonus(o)
                + 0.08 * self._novelty_bonus(o)
                + 0.04 * self._diversity_bonus(o)
                - 0.05 * self._repeat_penalty(o, state)
                + 0.02 * t * p_accept
                + (
                    self.layered_candidate_weight * surplus * max(p_accept, robust_opp)
                    if self.layered_candidates_mode
                    else 0.0
                )
            )
            if score > best_score:
                best_score = score
                best_o = o

        self._last_selected_arm = "internal_boulware"
        self._debug_last.update({
            "chosen_arm": self._last_selected_arm,
            "candidate_count": len(near),
            "target": target,
            "effective_target": effective_target,
            "bayes_concession": self._bayesian_concession_signal(),
        })
        return best_o or admissible[0]


# Backward-compatible alias. Submission should use ChangAgent, but MyNegotiator
# lets older skeleton main.py files still instantiate the agent.
MyNegotiator = ChangAgent
