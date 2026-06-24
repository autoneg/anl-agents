"""
ErmanConcealNegotiator for ANAC/ANL 2026.

Submission fields:
    Agent Module: erman_conceal_negotiator
    Agent Class : ErmanConcealNegotiator

Design goals:
    * Safe: no file I/O, no global memory between negotiations, no extra dependencies.
    * Competitive: high self-utility with late but controlled concession.
    * Concealing: avoids repeatedly exposing only the same top-ranked outcomes.
    * Opponent-aware: builds a per-negotiation frequency model from opponent offers.

This file is intentionally self contained. It should be placed in the root of the
ANL 2026 skeleton project, or uploaded in a submission zip with requirements.txt.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Optional

try:
    from negmas.sao import SAONegotiator, ResponseType
except Exception:  # pragma: no cover - only for static analysis outside NegMAS
    SAONegotiator = object  # type: ignore

    class ResponseType:  # type: ignore
        ACCEPT_OFFER = "ACCEPT_OFFER"
        REJECT_OFFER = "REJECT_OFFER"
        END_NEGOTIATION = "END_NEGOTIATION"


MAX_CACHED_OUTCOMES = 16000
MAX_RANDOM_SAMPLES = 4000
EPS = 1e-9


class _OpponentModelProxy:
    """Callable proxy used by the negotiator and exposed to the platform if read.

    The model is intentionally simple and robust: for every issue position it
    counts values offered by the opponent. Outcomes matching frequent opponent
    values receive a higher score. This is per-negotiation state only.
    """

    def __init__(self, owner: "ErmanConcealNegotiator") -> None:
        self.owner = owner

    def __call__(self, outcome: Any) -> float:
        return self.owner.estimate_opponent_utility(outcome)

    def eval(self, outcome: Any) -> float:
        return self.owner.estimate_opponent_utility(outcome)

    def utility(self, outcome: Any) -> float:
        return self.owner.estimate_opponent_utility(outcome)


class ErmanConcealNegotiator(SAONegotiator):
    """A practical ANL 2026 negotiator.

    The strategy combines four ideas.

    1. Boulware-style self-utility: stay demanding early, concede mainly near
       the end, never below a safety margin above the reservation value.
    2. Frequency opponent model: estimate what the opponent likes from their
       offers and prefer high-opponent-value offers when they still preserve our
       target utility.
    3. Preference concealment: offer a rotating band of similarly good outcomes
       instead of always revealing the single best outcome or a monotone utility
       trace. This makes the observed bids less informative about exact weights.
    4. AC-next-like acceptance: accept offers that are at least as good as what
       we are about to propose, and accept rational offers close to the deadline.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random(20260617)
        self.opponent_model = _OpponentModelProxy(self)
        self._reset_session_state()

    # ------------------------------------------------------------------
    # NegMAS lifecycle hooks
    # ------------------------------------------------------------------
    def on_preferences_changed(self, changes: Any) -> None:  # type: ignore[override]
        self._reset_session_state(keep_model_proxy=True)
        try:
            return super().on_preferences_changed(changes)
        except AttributeError:
            return None

    def on_negotiation_start(self, state: Any) -> None:  # type: ignore[override]
        self._reset_session_state(keep_model_proxy=True)
        self._cache_outcomes()
        try:
            return super().on_negotiation_start(state)
        except AttributeError:
            return None

    def propose(self, state: Any, dest: Optional[str] = None) -> Any:  # type: ignore[override]
        """Return the next counter-offer."""
        self._maybe_record_current_offer(state)
        offer = self._select_offer(state, record_choice=False)
        if offer is not None:
            self._my_offers.append(offer)
        return offer

    def respond(self, state: Any, source: Optional[str] = None) -> Any:  # type: ignore[override]
        """Accept or reject the current offer."""
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._record_opponent_offer(offer)
        u = self._u(offer)
        t = self._relative_time(state)
        reserve = self._reserved_value()
        aspiration = self._aspiration(t)

        # What we are likely to offer now. If the opponent gives at least this
        # much, accepting saves a round and avoids losing the agreement.
        next_offer = self._select_offer(state, record_choice=False)
        next_u = self._u(next_offer) if next_offer is not None else aspiration

        # Main threshold: tough early, pragmatic late.
        threshold = max(reserve + 0.015, aspiration - 0.025)

        # Against very cooperative opponents, take strong offers immediately.
        if u >= max(0.985, reserve + 0.25):
            return ResponseType.ACCEPT_OFFER

        # AC-next style acceptance.
        if u + 0.012 >= next_u and u >= reserve + 0.015:
            return ResponseType.ACCEPT_OFFER

        # Ordinary acceptance threshold.
        if u >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Close to the deadline, avoid timeouts if the deal is rational.
        if t >= 0.985 and u >= reserve + 0.008:
            return ResponseType.ACCEPT_OFFER
        if t >= 0.945 and u >= max(reserve + 0.025, 0.58):
            return ResponseType.ACCEPT_OFFER
        if t >= 0.900 and u >= max(reserve + 0.040, 0.64):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    # ------------------------------------------------------------------
    # Public-ish model API. Different tournament code versions sometimes look
    # for a callable model or a method; exposing both is harmless.
    # ------------------------------------------------------------------
    def estimate_opponent_utility(self, outcome: Any) -> float:
        """Return an estimated opponent utility in [0, 1]."""
        return self._opponent_score(outcome)

    def opponent_utility(self, outcome: Any) -> float:
        return self._opponent_score(outcome)

    # ------------------------------------------------------------------
    # State and utilities
    # ------------------------------------------------------------------
    def _reset_session_state(self, keep_model_proxy: bool = False) -> None:
        self._outcomes: list[Any] = []
        self._outcome_utils: dict[Any, float] = {}
        self._sorted_outcomes: list[Any] = []
        self._min_u = 0.0
        self._max_u = 1.0
        self._best_offer: Any = None
        self._opponent_offers: list[Any] = []
        self._my_offers: list[Any] = []
        self._last_seen_offer: Any = object()
        self._issue_value_counts: list[defaultdict[Any, float]] = []
        self._issue_totals: list[float] = []
        if not keep_model_proxy:
            self.opponent_model = _OpponentModelProxy(self)

    def _u(self, outcome: Any) -> float:
        if outcome is None:
            return -1.0
        if outcome in self._outcome_utils:
            return self._outcome_utils[outcome]
        try:
            value = float(self.ufun(outcome))  # type: ignore[misc]
        except Exception:
            value = -1.0
        if math.isnan(value):
            value = -1.0
        self._outcome_utils[outcome] = value
        return value

    def _reserved_value(self) -> float:
        ufun = getattr(self, "ufun", None)
        for name in ("reserved_value", "reservation_value", "rv"):
            try:
                value = getattr(ufun, name)
                if value is not None:
                    return float(value)
            except Exception:
                pass
        return 0.0

    def _relative_time(self, state: Any) -> float:
        for name in ("relative_time", "relative_time_elapsed"):
            try:
                value = getattr(state, name)
                if value is not None:
                    return max(0.0, min(1.0, float(value)))
            except Exception:
                pass
        try:
            step = float(getattr(state, "step", 0))
            n_steps = float(getattr(self.nmi, "n_steps", 0))
            if n_steps > 0:
                return max(0.0, min(1.0, step / max(1.0, n_steps - 1.0)))
        except Exception:
            pass
        return 0.0

    def _as_tuple(self, outcome: Any) -> tuple[Any, ...]:
        if outcome is None:
            return tuple()
        if isinstance(outcome, tuple):
            return outcome
        if isinstance(outcome, list):
            return tuple(outcome)
        try:
            return tuple(outcome)
        except Exception:
            return (outcome,)

    # ------------------------------------------------------------------
    # Outcome cache
    # ------------------------------------------------------------------
    def _cache_outcomes(self) -> None:
        outcomes = list(self._enumerate_outcomes_limited())
        if not outcomes:
            self._outcomes = []
            self._sorted_outcomes = []
            self._best_offer = None
            return

        # Remove duplicate outcomes while preserving order.
        unique: list[Any] = []
        seen: set[Any] = set()
        for outcome in outcomes:
            key = self._safe_key(outcome)
            if key not in seen:
                unique.append(outcome)
                seen.add(key)
        self._outcomes = unique

        for outcome in self._outcomes:
            self._u(outcome)
        self._sorted_outcomes = sorted(self._outcomes, key=self._u, reverse=True)
        self._best_offer = self._sorted_outcomes[0] if self._sorted_outcomes else None
        values = [self._u(o) for o in self._outcomes]
        self._min_u = min(values) if values else 0.0
        self._max_u = max(values) if values else 1.0

    def _enumerate_outcomes_limited(self) -> Iterable[Any]:
        nmi = getattr(self, "nmi", None)
        outcome_space = getattr(nmi, "outcome_space", None)

        # Best case: finite enumeration supported by the outcome space.
        for method_name in ("enumerate", "enumerate_or_sample", "to_discrete"):
            method = getattr(outcome_space, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if method_name == "to_discrete" and hasattr(result, "enumerate"):
                    result = result.enumerate()
                count = 0
                for outcome in result:
                    yield outcome
                    count += 1
                    if count >= MAX_CACHED_OUTCOMES:
                        return
                if count > 0:
                    return
            except Exception:
                pass

        # Fallback: sample random outcomes from the NMI.
        random_outcome = getattr(nmi, "random_outcome", None)
        if callable(random_outcome):
            for _ in range(MAX_RANDOM_SAMPLES):
                try:
                    outcome = random_outcome()
                except Exception:
                    break
                if outcome is not None:
                    yield outcome

    def _safe_key(self, outcome: Any) -> Any:
        try:
            hash(outcome)
            return outcome
        except Exception:
            return repr(outcome)

    # ------------------------------------------------------------------
    # Bidding and acceptance logic
    # ------------------------------------------------------------------
    def _aspiration(self, t: float) -> float:
        reserve = self._reserved_value()
        # End target adjusts to reserve so that high reservation scenarios remain safe.
        end = max(reserve + 0.055, 0.58)
        start = max(end + 0.20, min(0.995, self._max_u - 0.002))
        # Boulware concession: high early, drops mainly late.
        return max(reserve + 0.015, end + (start - end) * (1.0 - t ** 3.2))

    def _select_offer(self, state: Any, record_choice: bool = True) -> Any:
        if not self._sorted_outcomes:
            self._cache_outcomes()
        if not self._sorted_outcomes:
            try:
                return self.nmi.random_outcome()
            except Exception:
                return None

        t = self._relative_time(state)
        target = self._aspiration(t)
        reserve = self._reserved_value()

        # Work with a band rather than only the best outcomes. This protects our
        # utility while avoiding a perfectly revealing sequence of top offers.
        candidates = [o for o in self._sorted_outcomes if self._u(o) >= target]
        if len(candidates) < 8:
            relaxed = max(reserve + 0.025, target - 0.08 - 0.10 * max(0.0, t - 0.70))
            candidates = [o for o in self._sorted_outcomes if self._u(o) >= relaxed]
        if not candidates:
            candidates = self._sorted_outcomes[: min(120, len(self._sorted_outcomes))]

        # Limit computation for very large spaces but keep enough diversity.
        if len(candidates) > 800:
            stride = max(1, len(candidates) // 800)
            candidates = candidates[::stride][:800]

        best = None
        best_score = -10**9
        for outcome in candidates:
            own = self._normalized_own(outcome)
            opp = self._opponent_score(outcome)
            diversity = self._diversity_score(outcome)
            repeat_penalty = self._repeat_penalty(outcome)
            reveal_penalty = self._reveal_penalty(outcome, target)

            # Early offers are more concealing/diverse; late offers become more
            # agreement-oriented. Opponent score helps reach agreements without
            # giving away exact private weights.
            own_w = 0.69 + 0.18 * t
            opp_w = 0.24 - 0.08 * t
            div_w = 0.08 * (1.0 - t)
            score = (
                own_w * own
                + opp_w * opp
                + div_w * diversity
                - 0.12 * repeat_penalty
                - 0.055 * reveal_penalty
                + self._tiny_noise(outcome, state)
            )
            if score > best_score:
                best_score = score
                best = outcome

        if best is None:
            best = self._best_offer
        if record_choice and best is not None:
            self._my_offers.append(best)
        return best

    def _normalized_own(self, outcome: Any) -> float:
        u = self._u(outcome)
        return max(0.0, min(1.0, (u - self._min_u) / max(EPS, self._max_u - self._min_u)))

    def _reveal_penalty(self, outcome: Any, target: float) -> float:
        # Penalize always sending the exact maximum and highly monotone utility
        # levels. We prefer outcomes just above the current target unless the
        # deadline is close.
        u = self._u(outcome)
        if self._max_u <= self._min_u + EPS:
            return 0.0
        too_high = max(0.0, u - max(target + 0.10, 0.92))
        return too_high / max(EPS, self._max_u - self._min_u)

    def _diversity_score(self, outcome: Any) -> float:
        if not self._my_offers:
            return 1.0
        ot = self._as_tuple(outcome)
        if not ot:
            return 0.5
        recent = self._my_offers[-5:]
        distances = []
        for past in recent:
            pt = self._as_tuple(past)
            m = max(len(ot), len(pt), 1)
            same = sum(1 for a, b in zip(ot, pt) if a == b)
            distances.append(1.0 - same / m)
        return max(0.0, min(1.0, sum(distances) / len(distances)))

    def _repeat_penalty(self, outcome: Any) -> float:
        if not self._my_offers:
            return 0.0
        if self._safe_key(outcome) == self._safe_key(self._my_offers[-1]):
            return 1.0
        count = sum(1 for o in self._my_offers[-8:] if self._safe_key(o) == self._safe_key(outcome))
        return min(1.0, 0.22 * count)

    def _tiny_noise(self, outcome: Any, state: Any) -> float:
        # Deterministic small perturbation to break ties without using global state.
        step = getattr(state, "step", 0)
        key = hash((self._safe_key(outcome), step, len(self._opponent_offers)))
        return ((key % 1009) / 1009.0 - 0.5) * 0.004

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------
    def _maybe_record_current_offer(self, state: Any) -> None:
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return
        key = self._safe_key(offer)
        if key == self._safe_key(self._last_seen_offer):
            return
        self._record_opponent_offer(offer)
        self._last_seen_offer = offer

    def _record_opponent_offer(self, offer: Any) -> None:
        if offer is None:
            return
        self._opponent_offers.append(offer)
        values = self._as_tuple(offer)
        if not self._issue_value_counts:
            self._issue_value_counts = [defaultdict(float) for _ in range(len(values))]
            self._issue_totals = [0.0 for _ in range(len(values))]
        if len(values) > len(self._issue_value_counts):
            extra = len(values) - len(self._issue_value_counts)
            self._issue_value_counts.extend(defaultdict(float) for _ in range(extra))
            self._issue_totals.extend(0.0 for _ in range(extra))

        # Recency weighting: later offers reveal more about acceptable values.
        weight = 1.0 + 0.03 * len(self._opponent_offers)
        for i, v in enumerate(values):
            self._issue_value_counts[i][v] += weight
            self._issue_totals[i] += weight

    def _opponent_score(self, outcome: Any) -> float:
        values = self._as_tuple(outcome)
        if not values:
            return 0.5
        if not self._issue_value_counts:
            # Before seeing any offer, prefer outcomes not too extreme for us.
            return 0.5
        scores = []
        for i, v in enumerate(values):
            if i >= len(self._issue_value_counts) or self._issue_totals[i] <= 0:
                scores.append(0.5)
                continue
            # Laplace smoothing so unseen values are not impossible.
            distinct = max(1, len(self._issue_value_counts[i]))
            c = self._issue_value_counts[i].get(v, 0.0)
            scores.append((c + 1.0) / (self._issue_totals[i] + distinct))
        raw = sum(scores) / len(scores)
        return max(0.0, min(1.0, raw))
