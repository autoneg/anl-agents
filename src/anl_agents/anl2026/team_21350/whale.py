from __future__ import annotations

import json
import os
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
from negmas import Outcome, ResponseType, SAOResponse, SAOState
from negmas.sao import SAONegotiator
from negmas.inout import pareto_frontier
from negmas.preferences import LambdaMultiFun, MappingUtilityFunction
from negmas.preferences.ops import nash_points


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class AgentParams:
    # -- Boulware concession -------------------------------------------------
    boulware_exponent: float = 16.2     # Tuned against 2024 leaders (was 17.5)
    pareto_weight: float = -0.2         # Anti-social: we bid more selfishly
    acceptance_margin: float = 0.98     # Accept at this fraction of bidding threshold
    deadline_fraction: float = 0.975    # Panic mode starts here
    panic_time: float = 0.995           # Accept above RV at any cost
    acceptance_exponent: float = 50.0   # Steepness of acceptance curve (UOAgent-style)

    # -- Opponent classifier -------------------------------------------------
    classify_after_n_offers: int = 8    # How many opponent offers before classifying
    selector_update_interval: int = 10  # Steps between tactic re-selection

    # -- Deception (ANL 2026 preparation) -----------------------------------
    deception_enabled: bool = True
    deception_utility_budget: float = 0.07   # Max utility we sacrifice for deception
    deception_start_time: float = 0.17       # Don't deceive in first 17%
    deception_end_time: float = 0.79         # Stop deceiving near deadline
    deception_min_high_impact_issues: int = 1  # Minimum issues to target

    # -- Panic endgame -------------------------------------------------------
    panic_min_margin: float = 0.07     # Min gap above RV required at panic time
    panic_best_fraction: float = 0.78  # Fraction of best-seen offer required at panic time

    # -- Opponent classifier thresholds --------------------------------------
    nash_min_gap_fraction: float = 0.04        # NashBidder: min gap as fraction of (max_u - rv)
    stubborn_variance_threshold: float = 0.0005  # FrequencyModel: variance cutoff for STUBBORN
    nash_recompute_interval: int = 11           # Steps between Nash re-computation

    # -- Deception detection -------------------------------------------------
    deception_coverage_threshold: float = 0.45  # Fraction of issue values opponent must use
    deception_jump_threshold: float = 0.087     # Min utility jump to flag volatility
    deception_range_threshold: float = 0.049    # Min utility range for jump check to apply

    # -- Misc ----------------------------------------------------------------
    min_outcomes_in_pool: int = 3       # Min candidate pool size
    concession_bonus_threshold: float = 0.03  # If opponent concedes this much, be lenient
    bid_band_width: float = 0.03        # Narrow band around target utility for bid selection

    # -- Refactoring additions (parameterized magic numbers) ------------------
    endgame_time: float = 0.975
    endgame_min_steps: int = 2
    best_offer_return_time: float = 0.98
    endgame_rv_increment: float = 0.02
    step_based_rv_low_threshold: float = 0.4
    step_based_rv_low_increment: float = 0.40
    step_based_rv_high_floor: float = 0.80
    step_based_softmax_multiplier: float = 5.0
    step_based_endgame_fraction: float = 0.15
    acceptor_panic_min_diff: float = 0.01
    acceptor_concession_bonus_multiplier: float = 0.97
    freq_model_maxlen: int = 20
    freq_model_min_offers_deceptive: int = 10
    freq_model_min_history_deceptive: int = 5
    freq_model_min_history_type: int = 4
    freq_model_random_variance_threshold: float = 0.05
    freq_model_random_slope_threshold: float = 0.1
    freq_model_conceder_slope_threshold: float = -0.15
    freq_model_linear_slope_threshold: float = -0.02
    decoy_n_samples: int = 15
    decoy_impact_threshold: float = 0.01
    decoy_phase1_time: float = 0.40
    decoy_phase2_time: float = 0.85
    decoy_phase3_time: float = 0.95
    decoy_phase3_duration: float = 0.10

    def __post_init__(self) -> None:
        assert 1.0 <= self.boulware_exponent <= 30.0, f"boulware_exponent={self.boulware_exponent}"
        assert -1.0 <= self.pareto_weight <= 1.0
        assert 0.5 <= self.acceptance_margin <= 1.0
        assert 0.5 <= self.deadline_fraction < self.panic_time <= 1.0
        assert 0.0 <= self.deception_start_time < self.deception_end_time <= 1.0
        assert 0.0 <= self.deception_utility_budget <= 0.30
        assert 0.0 <= self.panic_min_margin <= 0.5
        assert 0.0 <= self.panic_best_fraction <= 1.0
        assert 0.0 < self.nash_min_gap_fraction <= 0.5
        assert 0.0 < self.stubborn_variance_threshold <= 0.1
        assert 0.0 <= self.bid_band_width <= 0.5
        assert 0.0 < self.endgame_time <= 1.0
        assert self.endgame_min_steps >= 0
        assert 0.0 < self.best_offer_return_time <= 1.0
        assert 0.0 <= self.endgame_rv_increment <= 0.5
        assert 0.0 <= self.step_based_rv_low_threshold <= 1.0
        assert 0.0 <= self.step_based_rv_low_increment <= 1.0
        assert 0.0 <= self.step_based_rv_high_floor <= 1.0
        assert 0.0 < self.step_based_softmax_multiplier <= 100.0
        assert 0.0 <= self.step_based_endgame_fraction <= 1.0
        assert 0.0 <= self.acceptor_panic_min_diff <= 0.5
        assert 0.0 <= self.acceptor_concession_bonus_multiplier <= 2.0
        assert self.freq_model_maxlen > 0
        assert self.freq_model_min_offers_deceptive > 0
        assert self.freq_model_min_history_deceptive > 0
        assert self.freq_model_min_history_type > 0
        assert 0.0 <= self.freq_model_random_variance_threshold <= 1.0
        assert 0.0 <= self.freq_model_random_slope_threshold <= 5.0
        assert -5.0 <= self.freq_model_conceder_slope_threshold <= 0.0
        assert -5.0 <= self.freq_model_linear_slope_threshold <= 0.0
        assert self.decoy_n_samples > 0
        assert 0.0 <= self.decoy_impact_threshold <= 0.5
        assert 0.0 <= self.decoy_phase1_time < self.decoy_phase2_time < self.decoy_phase3_time <= 1.0
        assert 0.0 <= self.decoy_phase3_duration <= 1.0


def load_params(path: str | Path | None = None) -> AgentParams:
    """Load params from JSON file or env var, falling back to defaults."""
    # Env-var override for the file path
    if path is None:
        path = os.environ.get("ANL_AGENT_PARAMS_FILE", "")

    if path:
        try:
            with open(path) as f:
                data = json.load(f)
            return AgentParams(**{k: v for k, v in data.items() if k in AgentParams.__dataclass_fields__})
        except Exception:
            pass  # Fall through to defaults

    # Per-key env overrides (e.g. ANL_BOULWARE_EXPONENT=12.0)
    overrides: dict = {}
    for fname, ftype in AgentParams.__dataclass_fields__.items():
        env_key = f"ANL_{fname.upper()}"
        if env_key in os.environ:
            raw = os.environ[env_key]
            try:
                # Coerce to the field's type
                default_val = getattr(AgentParams(), fname)
                overrides[fname] = type(default_val)(raw)
            except Exception:
                pass

    return AgentParams(**overrides) if overrides else AgentParams()


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _aspiration(t: float, max_u: float, rv: float, e: float) -> float:
    """Boulware aspiration function: (max-rv)*(1-t^e) + rv."""
    return (max_u - rv) * (1.0 - t ** e) + rv


def _enumerate_outcomes(outcome_space, max_outcomes: int = 10_000) -> list[Outcome]:
    return list(outcome_space.enumerate_or_sample(levels=5, max_cardinality=max_outcomes))


def _compute_pareto_and_nash(
    outcomes: list[Outcome], own_ufun, opp_ufun
) -> tuple[list[Outcome], float]:
    """Return (pareto_outcomes_sorted_by_own_util_desc, nash_utility_for_us)."""
    ufuns = (own_ufun, opp_ufun)
    frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
    frontier_outcomes = [outcomes[i] for i in frontier_indices]

    nash = nash_points(ufuns, frontier_utils)
    if nash:
        nash_util = nash[0][0][0]  # our utility at Nash point
    else:
        # Fallback: midpoint between RV and max
        nash_util = (float(own_ufun.max()) + own_ufun.reserved_value) / 2.0

    # Sort pareto outcomes by own utility descending
    pareto_sorted = sorted(
        frontier_outcomes,
        key=lambda o: float(own_ufun(o)),
        reverse=True,
    )
    return pareto_sorted, nash_util


# ==============================================================================
# OPPONENT CLASSIFICATION
# ==============================================================================

class OpponentType(Enum):
    BOULWARE = auto()   # Holds firm, concedes slowly
    CONCEDER = auto()   # Concedes quickly early on
    LINEAR = auto()     # Steady concessions
    RANDOM = auto()     # High offer diversity, no trend
    STUBBORN = auto()   # Repeats same offer
    DECEPTIVE = auto()  # Varies issue values erratically to poison our model
    UNKNOWN = auto()    # Not enough data yet


# ==============================================================================
# TACTICS: BIDDING, ACCEPTANCE, OPPONENT MODELING, DECEPTION
# ==============================================================================

class BoulwareBidder:
    """
    Pure time-based Boulware concession without opponent model dependency.
    Concedes from max_utility DOWN TO reserved_value following t^e curve.
    Works on Pareto frontier outcomes (sorted by own utility).

    threshold(t) = (max_u - rv) * (1 - t^e) + rv
      At t=0: max_util (never concede)
      At t=1: rv (willing to accept anything above rv)
    """

    def setup(self, outcome_space, own_ufun, reserved_value: float, params: AgentParams,
              opp_ufun=None) -> None:
        self._ufun = own_ufun
        self._rv = reserved_value
        self._params = params
        self._max_util = float(own_ufun.max()) if hasattr(own_ufun, "max") else 1.0
        self._opp_ufun = opp_ufun

        outcomes = _enumerate_outcomes(outcome_space)
        self._all_outcomes = outcomes

        if opp_ufun is not None:
            pareto, nash_u = _compute_pareto_and_nash(outcomes, own_ufun, opp_ufun)
            self._pool = pareto if pareto else outcomes
            # Nash utility stored for NashBidder subclass use only
            self._nash_utility = float(np.clip(nash_u, reserved_value, self._max_util))
        else:
            # Without opponent model: sort by own utility
            self._pool = sorted(
                [o for o in outcomes if float(own_ufun(o)) >= reserved_value],
                key=lambda o: float(own_ufun(o)),
                reverse=True,
            )
            self._nash_utility = reserved_value + 0.5 * (self._max_util - reserved_value)

    def concession_threshold(self, t: float) -> float:
        """Return minimum own-utility we're willing to bid at time t.
        BoulwareBidder concedes all the way to RV (floor = rv).
        """
        return _aspiration(t, self._max_util, self._rv, self._params.boulware_exponent)

    def best_bid(self, state) -> Optional[Outcome]:
        t = getattr(state, "relative_time", 0.5)
        threshold = self.concession_threshold(t)

        # Panic: at deadline fraction, drop to RV
        if t >= self._params.deadline_fraction:
            threshold = self._rv

        candidates = [
            o for o in self._pool
            if float(self._ufun(o)) >= threshold
        ]

        if not candidates:
            # Soften threshold to find something above RV
            candidates = [
                o for o in self._pool
                if float(self._ufun(o)) >= self._rv
            ]

        if not candidates:
            return self._pool[0] if self._pool else None

        # Score candidates: own utility + pareto_weight x opp utility (if available)
        def score(o: Outcome) -> float:
            u = float(self._ufun(o))
            if self._opp_ufun is not None:
                u += self._params.pareto_weight * float(self._opp_ufun(o))
            return u

        scored = sorted(candidates, key=score, reverse=True)
        # Deterministic: always pick the top-scored outcome (UOAgent-style).
        return scored[0] if scored else None


class NashBidder(BoulwareBidder):
    """
    Nash-targeting Boulware: concedes from max_utility toward Nash utility (not all the way to rv).
    Recomputes Nash every N offers using the opponent model.

    threshold(t) = (max_u - nash_u) * (1 - t^e) + nash_u
      At t=0: max_util
      At t=1: nash_util (floor is Nash, not rv)
    """

    def setup(self, outcome_space, own_ufun, reserved_value: float, params: AgentParams,
              opp_ufun=None) -> None:
        super().setup(outcome_space, own_ufun, reserved_value, params, opp_ufun)
        self._step = 0
        self._recompute_interval = params.nash_recompute_interval
        self._outcome_space = outcome_space
        # Ensure minimum gap so agent always concedes from max toward nash
        min_gap = params.nash_min_gap_fraction * (self._max_util - reserved_value)
        if self._nash_utility > self._max_util - min_gap:
            self._nash_utility = self._max_util - min_gap
        # Concede toward self._nash_utility as the floor. The endgame strategy
        # will handle late-game agreements below this utility if needed.

    def _compute_t(self, state_or_t):
        """Compute normalized time/step fraction without side effects."""
        if isinstance(state_or_t, float):
            return state_or_t
        state = state_or_t
        n_steps = getattr(state, 'n_steps', None)
        if n_steps is not None and n_steps > 0:
            return min(1.0, self._step / n_steps)
        return getattr(state, 'relative_time', 0.5)

    def concession_threshold(self, state_or_t):
        """NashBidder concedes toward nash_utility (not all the way to rv).
        Supports both time-based (float) and step-based (state object) input."""
        t = self._compute_t(state_or_t)
        return _aspiration(t, self._max_util, self._nash_utility, self._params.boulware_exponent)

    def best_bid(self, state) -> Optional[Outcome]:
        # Increment step ONLY when actually bidding (not during acceptance preview)
        self._step += 1
        if self._opp_ufun is not None and (self._step - 1) % self._recompute_interval == 0:
            outcomes = _enumerate_outcomes(self._outcome_space)
            _, nash_u = _compute_pareto_and_nash(outcomes, self._ufun, self._opp_ufun)
            self._nash_utility = float(np.clip(
                nash_u, self._rv,
                self._max_util - self._params.nash_min_gap_fraction * (self._max_util - self._rv)
            ))
        # CRITICAL FIX: NashBidder must NOT drop threshold to RV at deadline.
        # BoulwareBidder's panic drop overrides our Nash floor, causing us to
        # bid terrible outcomes near the deadline. We override best_bid to
        # preserve the Nash floor.
        threshold = self.concession_threshold(state)

        candidates = [o for o in self._pool if float(self._ufun(o)) >= threshold]
        if not candidates:
            candidates = [o for o in self._pool if float(self._ufun(o)) >= self._rv]
        if not candidates:
            return self._pool[0] if self._pool else None

        def score(o: Outcome) -> float:
            u = float(self._ufun(o))
            if self._opp_ufun is not None:
                u += self._params.pareto_weight * float(self._opp_ufun(o))
            return u

        scored = sorted(candidates, key=score, reverse=True)
        # Deterministic: always pick the top-scored outcome (UOAgent-style).
        return scored[0] if scored else None

    def preview_best_bid(self, state) -> Optional[Outcome]:
        """Return what best_bid would return without incrementing step counter."""
        old_step = self._step
        try:
            return self.best_bid(state)
        finally:
            self._step = old_step

    def endgame_bid(self, state) -> Optional[Outcome]:
        """UOAgent-style endgame: offer an outcome just above opponent's estimated RV.

        Sorts by opponent utility descending, finds outcomes near their learned RV,
        then picks the one with highest social welfare (and thus highest own utility
        among equally generous offers).
        """
        if self._opp_ufun is None:
            return self.best_bid(state)
        outcomes = self._all_outcomes
        opp_rv = getattr(self._opp_ufun, 'reserved_value', 0.0)
        rational = [
            o for o in outcomes
            if float(self._ufun(o)) > self._rv
            and float(self._opp_ufun(o)) > opp_rv
        ]
        if not rational:
            return self.best_bid(state)
        # Sort by opponent utility descending (most generous to opponent first)
        rational.sort(key=lambda o: float(self._opp_ufun(o)), reverse=True)
        # Find outcomes near opponent's RV (just barely acceptable)
        opp_utils = [float(self._opp_ufun(o)) for o in rational]
        target = opp_rv + self._params.endgame_rv_increment
        # Find nearest index
        nearest_idx = min(range(len(opp_utils)), key=lambda i: abs(opp_utils[i] - target))
        # Build a small window around that index
        band = rational[max(0, nearest_idx - 2) : min(len(rational), nearest_idx + 3)]
        if not band:
            band = rational[:3]
        # Sort by social welfare first, then own utility
        band.sort(
            key=lambda o: (
                float(self._ufun(o)) + float(self._opp_ufun(o)),
                float(self._ufun(o)),
            ),
            reverse=True,
        )
        return band[0]

    def _social_welfare_score(self, o: Outcome) -> float:
        """Score by social welfare (own + opponent utility)."""
        u = float(self._ufun(o))
        if self._opp_ufun is not None:
            u += float(self._opp_ufun(o))
        return u


class StepBasedBidder:
    """Step-based concession inspired by UOAgent (ANL 2024 advantage winner).

    Key differences from time-based Boulware:
      1. Concession is based on step / n_steps (deterministic, not wall-clock time).
      2. Outcome pool is aggressively filtered to high-utility outcomes only
         (floor = max(RV, 0.80) or RV + 0.40, whichever is higher).
      3. At endgame, finds the best outcome for us that is just barely
         acceptable to the opponent (near their estimated RV).
    """

    def setup(self, outcome_space, own_ufun, reserved_value: float, params: AgentParams,
              opp_ufun=None) -> None:
        self._ufun = own_ufun
        self._rv = reserved_value
        self._params = params
        self._max_util = float(own_ufun.max()) if hasattr(own_ufun, "max") else 1.0
        self._opp_ufun = opp_ufun
        self._outcome_space = outcome_space
        self._step = 0

        outcomes = _enumerate_outcomes(outcome_space)
        self._all_outcomes = outcomes

        # UOAgent-style aggressive outcome filtering
        if reserved_value <= self._params.step_based_rv_low_threshold:
            self._under = reserved_value + self._params.step_based_rv_low_increment
        else:
            self._under = max(reserved_value, self._params.step_based_rv_high_floor)

        # Build pool: only outcomes >= under
        self._pool = sorted(
            [o for o in outcomes if float(own_ufun(o)) >= self._under],
            key=lambda o: float(own_ufun(o)),
            reverse=True,
        )

        # If filtering is too aggressive and pool is empty, fall back to RV
        if not self._pool:
            self._pool = sorted(
                [o for o in outcomes if float(own_ufun(o)) >= reserved_value],
                key=lambda o: float(own_ufun(o)),
                reverse=True,
            )
            self._under = reserved_value

    def concession_threshold(self, state) -> float:
        """Step-based aspiration: under + (max - under) * (1 - fraction^e)."""
        self._step += 1
        if isinstance(state, float):
            return _aspiration(state, self._max_util, self._under, self._params.boulware_exponent)
        n_steps = getattr(state, 'n_steps', None)
        if n_steps is None or n_steps <= 0:
            # Fallback to time-based if n_steps unknown
            t = getattr(state, 'relative_time', 0.5)
            return _aspiration(t, self._max_util, self._under, self._params.boulware_exponent)
        fraction = min(1.0, self._step / n_steps)
        return _aspiration(fraction, self._max_util, self._under, self._params.boulware_exponent)

    def best_bid(self, state) -> Optional[Outcome]:
        threshold = self.concession_threshold(state)

        candidates = [o for o in self._pool if float(self._ufun(o)) >= threshold]
        if not candidates:
            candidates = self._pool
        if not candidates:
            return self._all_outcomes[0] if self._all_outcomes else None

        def score(o: Outcome) -> float:
            u = float(self._ufun(o))
            if self._opp_ufun is not None:
                u += self._params.pareto_weight * float(self._opp_ufun(o))
            return u

        scored = sorted(candidates, key=score, reverse=True)
        pool = scored[: max(self._params.min_outcomes_in_pool, min(self._params.min_outcomes_in_pool, len(scored)))]
        scores = [score(o) for o in pool]
        weights = np.exp(np.array(scores) * self._params.step_based_softmax_multiplier)
        weights = weights / weights.sum()
        idx = np.random.choice(len(pool), p=weights)
        return pool[idx]

    def endgame_bid(self, state) -> Optional[Outcome]:
        """Find best outcome for us that opponent can just barely accept."""
        if self._opp_ufun is None:
            return self.best_bid(state)
        rational = [
            o for o in self._all_outcomes
            if float(self._ufun(o)) > self._rv
            and float(self._opp_ufun(o)) > getattr(self._opp_ufun, 'reserved_value', 0.0)
        ]
        if not rational:
            return self.best_bid(state)
        rational.sort(key=lambda o: float(self._opp_ufun(o)))
        n = max(1, int(len(rational) * self._params.step_based_endgame_fraction))
        near_rv = rational[:n]
        near_rv.sort(key=lambda o: (float(self._ufun(o)), float(self._ufun(o)) + float(self._opp_ufun(o))), reverse=True)
        return near_rv[0]


class ThresholdAcceptor:
    """
    Accept if opponent offer >= steep aspiration curve.
    Inspired by UOAgent: very strict early, relaxing only near deadline.
    """

    def __init__(self, params: AgentParams) -> None:
        self._params = params

    def should_accept(self, state, own_ufun, bidder, negotiator) -> bool:
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return False

        u = float(own_ufun(offer))
        rv = own_ufun.reserved_value
        t = getattr(state, "relative_time", 0.5)

        # Hard floor: never accept below RV
        if u <= rv:
            return False

        # Track best offer received
        if u > negotiator._best_received_util:
            negotiator._best_received_util = u
            negotiator._best_received_offer = offer

        # Panic time: defend against stalling opponents (e.g., ExploitAgent offering rv+0.001)
        if t >= self._params.panic_time:
            if t >= 0.999:  # absolute timeout failsafe -- never score zero
                return True
            min_floor = rv + self._params.panic_min_margin
            if negotiator._best_received_util > rv + self._params.acceptor_panic_min_diff:
                min_floor = max(min_floor, negotiator._best_received_util * self._params.panic_best_fraction)
            return u >= min_floor

        # Steep acceptance curve inspired by UOAgent:
        # accept if u >= rv + (max_u - rv) * (1 - t^acceptance_exponent)
        # This is much stricter early on and only relaxes near deadline.
        max_u = float(own_ufun.max()) if hasattr(own_ufun, 'max') else 1.0
        accept_exponent = getattr(self._params, 'acceptance_exponent', 50.0)
        asp = rv + (max_u - rv) * (1.0 - t ** accept_exponent)
        if u >= asp:
            return True

        # Fallback: standard margin-based acceptance near deadline
        if t >= self._params.deadline_fraction:
            threshold = bidder.concession_threshold(t) * self._params.acceptance_margin
            return u >= threshold

        # Concession bonus: if opponent just conceded significantly, be more lenient
        if negotiator._prev_opp_util is not None and u - negotiator._prev_opp_util >= self._params.concession_bonus_threshold:
            negotiator._prev_opp_util = u
            return u >= bidder.concession_threshold(t) * self._params.acceptance_margin * self._params.acceptor_concession_bonus_multiplier
        negotiator._prev_opp_util = u

        threshold = bidder.concession_threshold(t) * self._params.acceptance_margin
        return u >= threshold


class ACNextAcceptor:
    """
    Accept if opponent's offer is at least as good as what we'd propose next.
    Falls back to ThresholdAcceptor at deadline.
    """

    def __init__(self, params: AgentParams) -> None:
        self._params = params
        self._threshold_acceptor = ThresholdAcceptor(params)

    def should_accept(self, state, own_ufun, bidder, negotiator) -> bool:
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return False

        u = float(own_ufun(offer))
        rv = own_ufun.reserved_value

        if u <= rv:
            return False

        t = getattr(state, "relative_time", 0.5)
        if t >= self._params.deadline_fraction:
            return self._threshold_acceptor.should_accept(state, own_ufun, bidder, negotiator)

        # ACNext: accept if offer >= our next planned bid
        preview = getattr(bidder, 'preview_best_bid', bidder.best_bid)
        next_bid = preview(state)
        if next_bid is not None:
            next_u = float(own_ufun(next_bid))
            if u >= next_u:
                return True

        return self._threshold_acceptor.should_accept(state, own_ufun, bidder, negotiator)


class FrequencyModel:
    """
    Estimates opponent utility via issue-value frequency counting.
    Classifies opponent type via concession trend analysis.
    """

    def __init__(self) -> None:
        self._issue_names: list[str] = []
        self._freq: dict[str, dict] = {}  # {issue_name: {value: count}}
        self._total_offers = 0
        self._utility_history: deque = deque(maxlen=20)
        self._opp_ufun = None
        self._params: AgentParams = AgentParams()

    def setup(self, outcome_space, opp_ufun, *, params: Optional[AgentParams] = None) -> None:
        self._opp_ufun = opp_ufun
        self._outcome_space = outcome_space
        if params is not None:
            self._params = params
        self._utility_history = deque(maxlen=self._params.freq_model_maxlen)
        for issue in outcome_space.issues:
            name = issue.name
            self._issue_names.append(name)
            self._freq[name] = {}

    def update(self, state) -> None:
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return

        self._total_offers += 1

        # Update frequency counts
        if isinstance(offer, tuple):
            for i, val in enumerate(offer):
                if i < len(self._issue_names):
                    name = self._issue_names[i]
                    self._freq[name][val] = self._freq[name].get(val, 0) + 1

        # Track utility for type classification
        if self._opp_ufun is not None:
            u = float(self._opp_ufun(offer))
            self._utility_history.append((getattr(state, "relative_time", 0.0), u))

    def utility(self, outcome: Outcome) -> float:
        """Estimate opponent utility via frequency model."""
        if self._opp_ufun is not None:
            return float(self._opp_ufun(outcome))

        if self._total_offers == 0 or not isinstance(outcome, tuple):
            return 0.5

        score = 0.0
        n_issues = len(self._issue_names)
        for i, val in enumerate(outcome):
            if i < len(self._issue_names):
                name = self._issue_names[i]
                freq = self._freq[name].get(val, 0)
                score += freq / max(self._total_offers, 1)

        return score / max(n_issues, 1)

    def estimated_rv(self) -> float:
        """Estimate opponent reservation value from concession trend."""
        if len(self._utility_history) < 3:
            return 0.0
        utils = [u for _, u in self._utility_history]
        return max(0.0, min(utils) - self._params.endgame_rv_increment)

    def _is_deceptive(self) -> bool:
        """
        Detect whether the opponent is likely running a deception strategy.

        Two signals must both fire (AND) to minimise false positives:

        1. **High per-issue coverage ratio**: fraction of each issue's possible
           values that the opponent has actually offered.  A normal agent
           repeatedly offers the same preferred values (low coverage); a decoy
           deceiver cycles through many distinct values to confuse our frequency
           model (high coverage, threshold 0.5).

        2. **Non-monotonic utility jumps**: estimated-utility history shows at
           least one large consecutive swing (>0.12) over a meaningful range
           (>0.05), indicating erratic non-concession behaviour inconsistent
           with a normal Boulware/Conceder curve.
        """
        if self._total_offers < self._params.freq_model_min_offers_deceptive:
            return False

        # Signal 1: per-issue coverage ratio vs possible values.
        if not self._freq or not self._issue_names:
            return False
        os = getattr(self, "_outcome_space", None)
        if os is None:
            return False
        possible_per_issue = [len(list(iss.all)) for iss in os.issues]
        if not possible_per_issue:
            return False
        ratios = [
            len(self._freq.get(name, {})) / max(n_possible, 1)
            for name, n_possible in zip(self._issue_names, possible_per_issue)
        ]
        avg_coverage = sum(ratios) / len(ratios)
        high_diversity = avg_coverage > self._params.deception_coverage_threshold

        # Signal 2: large non-monotonic jumps in estimated utility
        if len(self._utility_history) < self._params.freq_model_min_history_deceptive:
            return False
        utils = [u for _, u in self._utility_history]
        diffs = [abs(utils[k + 1] - utils[k]) for k in range(len(utils) - 1)]
        max_jump = max(diffs)
        u_range = max(utils) - min(utils)
        # Require: at least one big jump AND meaningful overall range
        jump_volatile = max_jump > self._params.deception_jump_threshold and u_range > self._params.deception_range_threshold

        return high_diversity and jump_volatile

    def opponent_type(self) -> OpponentType:
        if len(self._utility_history) < self._params.freq_model_min_history_type:
            return OpponentType.UNKNOWN

        utils = [u for _, u in self._utility_history]

        # Check if stubborn: very low variance
        variance = float(np.var(utils))
        if variance < self._params.stubborn_variance_threshold:
            return OpponentType.STUBBORN

        # Deception check: high issue diversity + erratic utility jumps
        if self._is_deceptive():
            return OpponentType.DECEPTIVE

        # Fit linear trend
        times = [t for t, _ in self._utility_history]
        if len(times) < 2 or max(times) == min(times):
            return OpponentType.UNKNOWN
        slope = float(np.polyfit(times, utils, 1)[0])

        # Check randomness: high variance with no clear trend
        if variance > self._params.freq_model_random_variance_threshold and abs(slope) < self._params.freq_model_random_slope_threshold:
            return OpponentType.RANDOM

        if slope < self._params.freq_model_conceder_slope_threshold:
            return OpponentType.CONCEDER  # Utility dropping fast
        elif slope < self._params.freq_model_linear_slope_threshold:
            return OpponentType.LINEAR
        else:
            return OpponentType.BOULWARE  # Little change -> Boulware


class NullModel:
    """
    Returns 0.5 for all outcomes (ignore opponent offers).
    Used as anti-deception defense: don't let opponent corrupt our Nash computation.
    """

    def setup(self, outcome_space, opp_ufun, *, params: Optional[AgentParams] = None) -> None:
        pass

    def update(self, state) -> None:
        pass

    def utility(self, outcome: Outcome) -> float:
        return 0.5

    def estimated_rv(self) -> float:
        return 0.0

    def opponent_type(self) -> OpponentType:
        return OpponentType.UNKNOWN


class GroundTruthModel:
    """
    Uses the actual opponent utility function when provided by ANL.
    ANL 2024 typically provides opponent_ufun in private_info.
    """

    def __init__(self) -> None:
        self._opp_ufun = None
        self._freq_model = FrequencyModel()

    def setup(self, outcome_space, opp_ufun, *, params: Optional[AgentParams] = None) -> None:
        self._opp_ufun = opp_ufun
        self._freq_model.setup(outcome_space, opp_ufun, params=params)

    def update(self, state) -> None:
        self._freq_model.update(state)

    def utility(self, outcome: Outcome) -> float:
        if self._opp_ufun is not None:
            return float(self._opp_ufun(outcome))
        return self._freq_model.utility(outcome)

    def estimated_rv(self) -> float:
        if self._opp_ufun is not None and hasattr(self._opp_ufun, "reserved_value"):
            return self._opp_ufun.reserved_value
        return self._freq_model.estimated_rv()

    def opponent_type(self) -> OpponentType:
        return self._freq_model.opponent_type()


class NoDeception:
    """Identity deception: passes bids through unchanged. Baseline for A/B testing."""

    def setup(self, outcome_space, own_ufun, reserved_value: float, params: AgentParams) -> None:
        pass

    def transform_bid(self, candidate: Outcome, state, *, min_utility: float = 0.0) -> Outcome:
        return candidate

    def update(self, state) -> None:
        pass


class IssueDecoyDeceiver:
    """
    Misleads opponent's frequency model by substituting decoy values on high-impact issues.

    Upgraded to DAD (Dynamic Adaptive Deception):
    - Multi-outcome impact estimation: calculates average impact across sampled outcomes.
    - Zero-start preference poisoning: active from step 1 (t=0.0).
    - Phase-based dynamic budget: Phase 1 (t < 0.40) uses double budget, Phase 2 uses base,
      Phase 3 (t >= 0.85) linearly decays budget to 0.0 to guarantee convergence.
    - True best value hiding: completely hides true optimal values on high-impact issues.
    """

    def __init__(self) -> None:
        self._own_ufun = None
        self._rv = 0.0
        self._params: Optional[AgentParams] = None
        self._issue_names: list[str] = []
        self._all_values: list[list] = []  # per issue
        self._issue_impact: list[float] = []  # average utility range per issue
        self._decoy_values: list = []  # best decoy per issue (second-best average value)
        self._decoy_candidates: list[list] = []  # all non-best values per issue, best-decoy first
        self._is_high_impact: list[bool] = []
        self._offer_value_counts: list[dict] = []  # track decoy offer counts per issue

    def setup(self, outcome_space, own_ufun, reserved_value: float, params: AgentParams) -> None:
        self._own_ufun = own_ufun
        self._rv = reserved_value
        self._params = params

        issues = outcome_space.issues
        self._issue_names = [iss.name for iss in issues]
        self._all_values = [list(iss.all) for iss in issues]
        self._offer_value_counts = [{} for _ in issues]

        # Compute impact per issue: average range of utility contribution across sampled outcomes
        n_samples = params.decoy_n_samples
        sampled_outcomes = []
        try:
            sampled_outcomes = list(outcome_space.sample(n_samples))
        except Exception:
            pass
        if not sampled_outcomes:
            sampled_outcomes = [[list(iss.all)[0] for iss in issues]]

        self._issue_impact = []
        self._decoy_values = []
        self._decoy_candidates = []

        n_issues = len(issues)
        for i, issue in enumerate(issues):
            vals = list(issue.all)
            if not vals:
                self._issue_impact.append(0.0)
                self._decoy_values.append(None)
                self._decoy_candidates.append([])
                continue

            total_impact = 0.0
            value_utilities = {v: [] for v in vals}
            
            for outcome in sampled_outcomes:
                outcome_list = list(outcome)
                utilities_for_sample = []
                for v in vals:
                    test = tuple(outcome_list[:i] + [v] + outcome_list[i+1:])
                    try:
                        u = float(own_ufun(test))
                    except Exception:
                        u = 0.0
                    utilities_for_sample.append(u)
                    value_utilities[v].append(u)
                if utilities_for_sample:
                    total_impact += max(utilities_for_sample) - min(utilities_for_sample)
            
            avg_impact = total_impact / len(sampled_outcomes)
            self._issue_impact.append(avg_impact)

            # Sort values by their average utility contribution (descending: best to worst)
            avg_val_utils = [(sum(utils) / len(utils), v) for v, utils in value_utilities.items()]
            avg_val_utils.sort(reverse=True)

            # All non-best values as decoy candidates (strictly hides the true best value)
            candidates = [v for _, v in avg_val_utils[1:]]
            self._decoy_candidates.append(candidates)
            self._decoy_values.append(candidates[0] if candidates else avg_val_utils[0][1])

        # Classify issues: top 50% impact -> high-impact
        median_impact = float(np.median(self._issue_impact)) if self._issue_impact else 0.0
        self._is_high_impact = [
            imp >= median_impact and imp > self._params.decoy_impact_threshold
            for imp in self._issue_impact
        ]

        # Ensure at least min_issues are classified as high-impact
        n_high = sum(self._is_high_impact)
        if n_high < params.deception_min_high_impact_issues and n_issues > 0:
            max_idx = int(np.argmax(self._issue_impact))
            self._is_high_impact[max_idx] = True

    def transform_bid(self, candidate: Outcome, state, *, min_utility: float = 0.0) -> Outcome:
        """Replace values on high-impact issues with decoys, within a dynamic utility budget.

        min_utility: deception must not reduce our utility below this floor.
        """
        if not isinstance(candidate, tuple) or self._own_ufun is None:
            return candidate

        t = getattr(state, "relative_time", 0.5)
        p = self._params

        # Phase-based dynamic budget calculation (DAD)
        if t < p.decoy_phase1_time:
            budget = p.deception_utility_budget * 2.0
        elif t < p.decoy_phase2_time:
            budget = p.deception_utility_budget
        elif t < p.decoy_phase3_time:
            decay_fraction = (p.decoy_phase3_time - t) / p.decoy_phase3_duration
            budget = p.deception_utility_budget * max(0.0, decay_fraction)
        else:
            budget = 0.0

        if budget <= 1e-9:
            return candidate

        u_original = float(self._own_ufun(candidate))
        
        if u_original <= self._rv + 1e-9:
            return candidate  # Already at floor

        result = list(candidate)

        # Shuffle issue order to vary deception pattern
        issue_order = list(range(len(self._issue_names)))
        random.shuffle(issue_order)

        remaining_budget = budget

        for i in issue_order:
            if not self._is_high_impact[i]:
                continue
            candidates = self._decoy_candidates[i]
            if not candidates:
                continue

            # Weighted stochastic selection: penalize overused values for variety
            counts = self._offer_value_counts[i]
            weights = [1.0 / (counts.get(v, 0) + 1) for v in candidates]
            total = sum(weights)
            normed = [w / total for w in weights]
            decoy = random.choices(candidates, weights=normed, k=1)[0]

            if decoy == result[i]:
                continue

            # Compute utility cost of substitution
            test = tuple(result[:i] + [decoy] + result[i+1:])
            u_after = float(self._own_ufun(test))
            cost = float(self._own_ufun(tuple(result))) - u_after

            if cost <= remaining_budget and u_after >= max(self._rv, min_utility):
                result[i] = decoy
                remaining_budget -= cost
                counts[decoy] = counts.get(decoy, 0) + 1

        return tuple(result)

    def update(self, state) -> None:
        """Track our own offer history for adaptive deception pressure."""
        offer = getattr(state, "current_offer", None)
        if offer is None or not isinstance(offer, tuple):
            return
        for i, val in enumerate(offer):
            if i < len(self._offer_value_counts):
                counts = self._offer_value_counts[i]
                counts[val] = counts.get(val, 0) + 1


# ==============================================================================
# STRATEGY SELECTOR / BUNDLES
# ==============================================================================

@dataclass
class TacticBundle:
    bidder_cls: Type
    acceptor_cls: Type
    model_cls: Type
    deceiver_cls: Type


# ── Opponent-type → tactic table ─────────────────────────────────────────────
TACTIC_TABLE: dict[OpponentType, TacticBundle] = {
    OpponentType.BOULWARE: TacticBundle(
        NashBidder, ThresholdAcceptor, GroundTruthModel, IssueDecoyDeceiver
    ),
    OpponentType.CONCEDER: TacticBundle(
        BoulwareBidder, ACNextAcceptor, GroundTruthModel, IssueDecoyDeceiver
    ),
    OpponentType.LINEAR: TacticBundle(
        NashBidder, ThresholdAcceptor, GroundTruthModel, IssueDecoyDeceiver
    ),
    OpponentType.RANDOM: TacticBundle(
        BoulwareBidder, ThresholdAcceptor, NullModel, IssueDecoyDeceiver
    ),
    OpponentType.DECEPTIVE: TacticBundle(
        BoulwareBidder, ThresholdAcceptor, NullModel, IssueDecoyDeceiver
    ),
    OpponentType.STUBBORN: TacticBundle(
        BoulwareBidder, ThresholdAcceptor, GroundTruthModel, IssueDecoyDeceiver
    ),
    OpponentType.UNKNOWN: TacticBundle(
        NashBidder, ThresholdAcceptor, GroundTruthModel, IssueDecoyDeceiver
    ),
}


class TacticSelector:
    """
    Monitors opponent behavior, classifies type, selects appropriate tactic bundle.
    """

    def __init__(self, params: AgentParams, opp_ufun=None) -> None:
        self._params = params
        self._opp_ufun = opp_ufun
        self._current_type = OpponentType.UNKNOWN
        self._step = 0
        self._model = None

    def current_opponent_type(self, model) -> OpponentType:
        """Re-classify opponent using the current model's history."""
        self._step += 1
        if self._step % self._params.selector_update_interval != 0:
            return self._current_type

        if self._step < self._params.classify_after_n_offers:
            return OpponentType.UNKNOWN

        self._current_type = model.opponent_type()
        return self._current_type

    def select_bundle(self, opponent_type: OpponentType, has_opp_ufun: bool) -> TacticBundle:
        """Return the tactic bundle for this opponent type."""
        bundle = TACTIC_TABLE.get(opponent_type, TACTIC_TABLE[OpponentType.UNKNOWN])

        # If we don't have the ground truth ufun, downgrade GroundTruthModel → FrequencyModel
        if bundle.model_cls is GroundTruthModel and not has_opp_ufun:
            return TacticBundle(
                bundle.bidder_cls,
                bundle.acceptor_cls,
                FrequencyModel,
                bundle.deceiver_cls,
            )
        return bundle


# ==============================================================================
# MAIN NEGOTIATOR CLASS
# ==============================================================================

class WhaleNegotiator(SAONegotiator):
    """
    WhaleNegotiator is the single-class entrypoint for the ANL 2026 competition.
    It inherits from SAONegotiator and directly implements all components and the highly-tuned modular tactics.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = 0
        self._estimated_opp_rv = float('inf')
        self._best_received_util = -1.0
        self._best_received_offer = None
        self._prev_opp_util = None

    def on_preferences_changed(self, changes) -> None:
        assert self.ufun is not None
        # ANL often provides opponent ufun in private_info (Shochan's trick).
        # Guard against organizers patching this leak: fall back cleanly to FrequencyModel.
        try:
            self._opp_ufun = deepcopy(
                self.private_info.get("opponent_ufun", None) or getattr(self, "opponent_ufun", None)
            )
        except Exception:
            self._opp_ufun = None
        # Reset opponent's stated RV to 0 (gives us more negotiating power -- Shochan's trick)
        if self._opp_ufun is not None and hasattr(self._opp_ufun, "reserved_value"):
            try:
                self._opp_ufun.reserved_value = 0.0
            except Exception:
                self._opp_ufun = None  # Immutable ufun -- fall back to FrequencyModel

        self._params = load_params()
        os_ = self.ufun.outcome_space
        rv = self.ufun.reserved_value

        # Initialize components
        has_opp_ufun = self._opp_ufun is not None

        # Opponent model (ground truth if available, else frequency)
        if has_opp_ufun:
            self._history_model = GroundTruthModel()
            self._history_model.setup(os_, self._opp_ufun, params=self._params)
        else:
            self._history_model = FrequencyModel()
            self._history_model.setup(os_, None, params=self._params)
        self._model = self._history_model

        # Bidder (Nash seek is always used to find mutual agreements)
        opp_model_ufun = self._opp_ufun or MappingUtilityFunction(self._model.utility, reserved_value=0.0, outcome_space=os_)
        self._bidder = NashBidder()
        self._bidder.setup(os_, self.ufun, rv, self._params, opp_model_ufun)

        # Acceptor
        self._acceptor = ThresholdAcceptor(self._params)

        # Deceiver (IssueDecoyDeceiver if deception enabled, else NoDeception)
        if self._params.deception_enabled:
            self._deceiver = IssueDecoyDeceiver()
            self._deceiver.setup(os_, self.ufun, rv, self._params)
        else:
            self._deceiver = NoDeception()
            self._deceiver.setup(os_, self.ufun, rv, self._params)

        # Selector (for tactic hot-swapping)
        self._selector = TacticSelector(self._params, self._opp_ufun)
        self._step = 0
        self._estimated_opp_rv = float('inf')
        self._best_received_util = -1.0
        self._best_received_offer = None
        self._prev_opp_util = None

        # Trigger parent initialization to setup internal preference references
        super().on_preferences_changed(changes)

        # Synchronize private_info to share the opponent ufun if available
        if self._opp_ufun is not None:
            self.private_info["opponent_ufun"] = MappingUtilityFunction(self._model.utility, reserved_value=0.0, outcome_space=os_)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun is not None
        self._step += 1

        # Track best received offer (Shochan trick)
        offer = getattr(state, 'current_offer', None)
        if offer is not None:
            u = float(self.ufun(offer))
            if not hasattr(self, '_best_received_util') or u > getattr(self, '_best_received_util', -1.0):
                self._best_received_util = u
                self._best_received_offer = offer

        # Dynamic opponent RV learning (UOAgent trick)
        if offer is not None and self._opp_ufun is not None:
            opp_u = float(self._opp_ufun(offer))
            if opp_u >= 0.0 and opp_u < getattr(self, '_estimated_opp_rv', float('inf')):
                self._estimated_opp_rv = opp_u
                self._opp_ufun.reserved_value = float(opp_u)

        # Update opponent model
        self._history_model.update(state)
        self._deceiver.update(state)

        # Periodic tactic re-selection (ExploitAgent pattern)
        if self._step >= self._params.classify_after_n_offers:
            opp_type = self._selector.current_opponent_type(self._model)
            # Hot-swap tactic components if needed
            self._maybe_swap_tactics(opp_type)

        # Acceptance check
        if state.current_offer is not None:
            if self._acceptor.should_accept(state, self.ufun, self._bidder, self):
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        # Bidding
        t = getattr(state, "relative_time", 0.5)

        # Bidding: use endgame bid near deadline (opponent-centric), otherwise best_bid.
        n_steps = getattr(state, "n_steps", None)
        state_step = getattr(state, "step", self._step)
        is_endgame = t > self._params.endgame_time or (n_steps is not None and n_steps - state_step <= self._params.endgame_min_steps)
        if is_endgame and hasattr(self._bidder, 'endgame_bid'):
            raw_bid = self._bidder.endgame_bid(state)
        else:
            raw_bid = self._bidder.best_bid(state)

        if raw_bid is None:
            # Fallback: our best outcome
            raw_bid = self.ufun.best()

        # Shochan trick: if best received offer is better than our planned bid, offer it back
        if t > self._params.best_offer_return_time:
            best_offer = getattr(self, '_best_received_offer', None)
            if best_offer is not None:
                best_u = float(self.ufun(best_offer))
                planned_u = float(self.ufun(raw_bid))
                if best_u > planned_u and best_u > self.ufun.reserved_value:
                    raw_bid = best_offer

        # Deception transform: enforce concession threshold so deception never produces
        # an offer worse than what the bidder intended at this point in time.
        min_util = self._bidder.concession_threshold(t)
        final_bid = self._deceiver.transform_bid(raw_bid, state, min_utility=min_util)

        # Wrap the model's utility method so it's a NegMAS-compatible ufun for tournament score evaluation
        self.private_info["opponent_ufun"] = MappingUtilityFunction(self._model.utility, reserved_value=0.0, outcome_space=self.ufun.outcome_space)

        return SAOResponse(ResponseType.REJECT_OFFER, final_bid)

    def _maybe_swap_tactics(self, opp_type: OpponentType) -> None:
        """Hot-swap model and acceptor when opponent type changes."""
        bundle = self._selector.select_bundle(opp_type, self._opp_ufun is not None)

        # Swap opponent model if needed
        if not isinstance(self._model, bundle.model_cls):
            if bundle.model_cls is NullModel:
                self._model = NullModel()
            else:
                self._model = self._history_model

        # Swap acceptor if needed
        if not isinstance(self._acceptor, bundle.acceptor_cls):
            self._acceptor = bundle.acceptor_cls(self._params)

    def acceptance_strategy(self, state: SAOState) -> bool:
        if not hasattr(self, "_acceptor"):
            return False
        return self._acceptor.should_accept(state, self.ufun, self._bidder, self)

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        if not hasattr(self, "_bidder"):
            return None
        return self._bidder.best_bid(state)

    def update_opponent_model(self, state: SAOState) -> None:
        if hasattr(self, "_history_model"):
            self._history_model.update(state)
