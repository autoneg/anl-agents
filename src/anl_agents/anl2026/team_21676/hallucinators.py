"""
AgentNexus - ANL 2026 Negotiation Agent (Enhanced Version)
Team: Kevin, Gayathri, Diya

An advanced strategic negotiator with sophisticated opponent modeling,
adaptive negotiation strategies, and robust error handling.

Features:
- Selfish early-phase bidding with gradual concession
- Distribution-based opponent utility estimation
- Dynamic opponent issue weight adjustment
- Adaptive window-based offer selection
- Extreme conflict detection and rescue mechanisms
- Type-safe outcome handling
- Crash-proof response generation with a guaranteed-valid last resort
"""

# Build tag for entrypoint verification: bump this string whenever this file
# changes. The wrapper/submission class can use it to confirm tournament logs
# are importing this exact implementation rather than a stale cached copy.
AGENT_BUILD_TAG = "jupiter-2026-06-20-outcome-cap-v10"

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import pow
from statistics import median
from typing import Optional

from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.sao import ResponseType, SAOResponse, SAOState
from negmas.sao.components.acceptance import ACNext
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.negotiators.modular import BOANegotiator


class BOANeg(BOANegotiator):
    """Small local BOA negotiator base to avoid package-path issues on upload."""

    def __init__(self, *args, **kwargs):
        offering = TimeBasedOfferingPolicy()
        kwargs |= dict(
            acceptance=ACNext(offering),
            offering=offering,
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

@dataclass
class NegotiationConfig:
    """Configuration parameters for negotiation strategy."""
    
    # Utility targets
    EARLY_TARGET_UTILITY = 0.92
    EARLY_PHASE_THRESHOLD = 0.55
    MID_PHASE_THRESHOLD = 0.85
    LATE_TARGET_UTILITY = 0.75
    FLOOR_UTILITY_RATIO = 0.64
    # Restored 0.50 -> 0.55: tournament data showed the lowered floor traded
    # away too much captured utility against a field of competent rational
    # agents (Jupiter under-scored its firmer ancestors).  The survival-accept
    # path still protects against genuine erratic opponents without conceding
    # to everyone.
    EMERGENCY_UTILITY_RATIO = 0.55
    
    # Distribution model parameters
    WINDOW_SIZE = 4
    DISTRIBUTION_CHANGE_THRESHOLD = 0.32
    CONCESSION_DETECTION_THRESHOLD = 0.02
    DISTRIBUTION_UPDATE_WEIGHT = 0.08
    
    # Masking and offer selection
    EARLY_MASKING_SLACK = 0.035
    LATE_MASKING_SLACK = 0.070
    MASKING_PHASE_THRESHOLD = 0.55
    MIN_MASKED_WINDOW_SIZE = 5
    MAX_MASKED_WINDOW_SIZE = 17
    
    # Decoy signaling
    DECOY_SIGNAL_PHASE_THRESHOLD = 0.78
    CHEAP_ISSUE_FRACTION = 1/3
    EXPENSIVE_ISSUE_FRACTION = 1/3
    CHEAP_STABILITY_WEIGHT = 0.65
    EXPENSIVE_VARIETY_WEIGHT = 0.35
    
    # Conflict detection
    EXTREME_CONFLICT_THRESHOLD = 0.70
    EXTREME_CONFLICT_MIN_RATIO = 0.05
    STALL_DETECTION_MIN_INTERACTION = 5
    LOW_VALUE_THRESHOLD = 0.30
    STALL_URGENCY_THRESHOLD = 0.90
    RESCUE_URGENCY_THRESHOLD = 0.94
    RESCUE_FINAL_THRESHOLD = 0.97
    HARDLINER_ADVANTAGE_GUARD_THRESHOLD = 0.85
    HARDLINER_ADVANTAGE_MIRROR_MAX = 0.25
    HARDLINER_ADVANTAGE_CONCEDER_MAX = 0.12
    HARDLINER_ADVANTAGE_FLOOR_RATIO = 0.62
    
    # Acceptance criteria
    ACCOMBI_THRESHOLD = 0.99
    MIN_OFFER_HISTORY = 3
    MIN_CONCESSION_HISTORY = 4
    LATE_GAME_THRESHOLD = 0.97
    
    # Weight tuning over time (own, accept, pareto, masking)
    WEIGHT_EARLY = (0.74, 0.06, 0.04, 0.16)
    WEIGHT_MID = (0.60, 0.16, 0.10, 0.14)
    WEIGHT_LATE = (0.48, 0.28, 0.16, 0.08)
    # Emergency (t >= 0.94): moderate reduction of own-utility weight so
    # scoring begins converging toward opponent-reachable outcomes while still
    # leaving room to negotiate.  Not too extreme — there can be 5-10 steps
    # remaining where extracting value still matters.
    WEIGHT_EMERGENCY = (0.28, 0.44, 0.20, 0.08)
    EMERGENCY_WEIGHT_THRESHOLD = 0.94
    # Ultra-late (t >= 0.985): 1-2 steps left; closing is the only goal.
    # Own-utility weight is nearly zeroed; acceptance dominates.
    WEIGHT_ULTRA_LATE = (0.08, 0.68, 0.16, 0.08)
    ULTRA_LATE_THRESHOLD = 0.985
    
    # Opponent modeling
    VALUE_SCORE_POWER = 0.35
    SMOOTHING_FACTOR = 1
    DEFAULT_OPPONENT_UTILITY = 0.5
    DEFAULT_SIMILARITY = 0.5
    LATE_OFFER_MODEL_WEIGHT = 0.45
    REPEAT_OFFER_DAMPING = 0.48
    MIN_REPEATED_OFFER_WEIGHT = 0.18

    # Evidence confidence for opponent-responsive concession
    FORECAST_MIN_OFFERS = 4
    FORECAST_MAX_LOOKAHEAD = 0.35

    # Opponent-responsive concession
    RESPONSIVE_START_THRESHOLD = 0.45
    RESPONSIVE_MAX_HOLD = 0.055
    RESPONSIVE_MAX_CONCESSION = 0.045
    RESPONSIVE_CONFIDENCE_THRESHOLD = 0.30

    # ANL 2026-compatible Team 271 adaptations
    BEST_OFFER_REUSE_THRESHOLD = 0.985
    BEST_OFFER_TARGET_MARGIN = 0.98
    TWO_LEVEL_SELECTION_THRESHOLD = 0.60
    TWO_LEVEL_MIN_HISTORY = 4
    TWO_LEVEL_OWN_UTILITY_SLACK = 0.035
    TWO_LEVEL_KEEP_FRACTION = 0.60
    LOOKAHEAD_THRESHOLD = 0.75
    LOOKAHEAD_CONFIDENCE_THRESHOLD = 0.55
    LOOKAHEAD_TIEBREAK_WEIGHT = 0.025
    FINAL_TURN_TARGET_RELIEF = 0.025
    WEAK_MODEL_CONFIDENCE_THRESHOLD = 0.42
    LATE_RUFL_SELECTION_THRESHOLD = 0.86
    DEADLINE_RANGE_THRESHOLD = 0.94
    DEADLINE_RANGE_MARGIN = 0.03
    LATE_SAFE_FLOOR_RATIO = 0.70
    # Pushed 0.95 -> 0.97: let the smarter close-offer logic own the
    # 0.94-0.97 window; reserve the blunt emergency path for the final 3%.
    EMERGENCY_DEADLINE_START = 0.97
    HARD_SAFE_FLOOR_RATIO = 0.68
    EMERGENCY_REASONABLE_OPPONENT_SCORE = 0.34
    MODEL_TRUST_MIN_UNIQUE_OFFERS = 3
    MODEL_TRUST_MIN_DIVERSITY = 0.35
    MODEL_TRUST_MAX_REPEAT_SHARE = 0.70
    PHASE_ROTATION_STRIDE_EARLY = 7
    PHASE_ROTATION_STRIDE_LATE = 3
    BEST_OPPONENT_CLOSE_THRESHOLD = 0.94
    OPPONENT_ACCEPTANCE_CLOSE_THRESHOLD = 0.94
    OPPONENT_ACCEPTANCE_CLOSE_MIN_LIKELIHOOD = 0.82
    OPPONENT_ACCEPTANCE_CLOSE_MIN_OPPONENT_SCORE = 0.74
    OPPONENT_ACCEPTANCE_CLOSE_TARGET_MARGIN = 0.98
    LATE_REPETITION_RESCUE_THRESHOLD = 0.45

    # Percentile-based adaptive floor blending
    # Controls how much weight goes to the percentile floor vs the fixed-ratio floor.
    # 0.0 = pure ratio (old behavior), 1.0 = pure percentile.
    PERCENTILE_BLEND_WEIGHT = 0.55

    # --- Tier 2: rational acceptance ---------------------------------------
    # Accept when the offer is within `margin` of the best utility we forecast
    # the opponent will ever give us.  Prevents rejecting a good offer when
    # waiting cannot realistically yield anything better.
    RATIONAL_ACCEPT_MIN_TIME = 0.82
    RATIONAL_ACCEPT_MIN_CONFIDENCE = 0.68
    # Tightened 0.97 -> 0.98: only accept very close to the forecast best so we
    # do not sell low when the model is confident.
    RATIONAL_ACCEPT_MARGIN = 0.98

    # --- Tier 2: profile-adaptive scoring weights --------------------------
    # When the opponent model is confident, shift scoring weights by detected
    # opponent type.  Small shift, renormalized, never applied in ultra-late.
    PROFILE_WEIGHT_SHIFT = 0.06
    PROFILE_MIN_CONFIDENCE = 0.50

    # --- Tier 2: adaptive concession exponents -----------------------------
    # Stay Boulware longer (higher exponent) against hardliners/conceders who
    # do not reward early concession; default otherwise.
    EXP_DEFAULT_MID = 2.4
    EXP_DEFAULT_LATE = 1.4
    EXP_HARDLINER_MID = 3.2
    EXP_HARDLINER_LATE = 2.0
    EXP_CONCEDER_MID = 3.0
    EXP_CONCEDER_LATE = 1.8

    # --- Survival against erratic / "maniac" opponents ---------------------
    # In the final window, if the opponent is unpredictable (high offer
    # variance, no concession trend) or simply never going to give us better,
    # accept any deal a small margin above reservation rather than timing out
    # at the reservation value.  Tightly gated so normal play is unaffected.
    SURVIVAL_ACCEPT_THRESHOLD = 0.98
    SURVIVAL_MARGIN_FRACTION = 0.05
    ERRATIC_MIN_HISTORY = 6
    ERRATIC_VARIANCE_THRESHOLD = 0.18
    ERRATIC_CONFIDENCE_MAX = 0.45

    # --- Part 5: Pareto-frontier-restricted bidding ------------------------
    # Among bids at/above aspiration, prefer ones on the estimated Pareto
    # frontier (own vs estimated-opponent utility).  Captures the same own
    # utility but with higher acceptance — the main non-aggressive score lever.
    PARETO_BIDDING_THRESHOLD = 0.50      # activate after this relative time
    PARETO_MIN_HISTORY = 4              # need some opponent evidence first

    # --- Part 5: stability ------------------------------------------------
    # Fixed seed for outcome-space sampling so the rational-outcome set is
    # identical across runs on large domains (removes run-to-run bid drift).
    # NOTE: within a single negotiation relative_time is monotonic, so phase
    # boundaries cannot "flap" — hysteresis is unnecessary.  The frequency
    # opponent model already updates with recency damping, so a separate EMA
    # would be redundant.  Determinism (this seed + deterministic selection)
    # plus Pareto narrowing is the effective stability pass.
    OUTCOME_SAMPLE_SEED = 20260619
    # Cap the rational-outcome set on huge domains.  Enumerating + evaluating
    # the ufun over a million-outcome space takes ~200s per negotiation, which
    # both makes tournaments crawl and EXCEEDS the competition's per-negotiation
    # time limit (the agent would forfeit).  Above this cardinality we sample a
    # representative subset (seeded → deterministic) plus the true best outcome.
    MAX_OUTCOMES = 20000

    # --- Opponent-adaptive stance -----------------------------------------
    # Jupiter reads the opponent's behavior and picks a stance.  Against
    # EXPLOITABLE opponents (already conceding, not hard) it stays FIRM and
    # squeezes — no opponent-friendly bidding, no early rational-accept — so it
    # captures the value the firm base agent was capturing.  Against TOUGH or
    # ERRATIC opponents it engages the flexible/safety machinery (Pareto
    # bidding, rational-accept, survival net) because that is when softening
    # actually prevents a walk-away.
    EXPLOITABLE_CONCEDER_MIN = 0.55   # opponent is clearly conceding to us
    EXPLOITABLE_HARDLINER_MAX = 0.40  # ...and not behaving like a hardliner
    TOUGH_HARDLINER_MIN = 0.55        # opponent holds firm / low value for us
    TOUGH_THRESHOLD_MIN = 0.55        # ...or accepts only at a fixed bar


class DistributionAnalyzer:
    """Analyzes opponent offer distributions to estimate utility weights."""
    
    def __init__(self, config: NegotiationConfig):
        """Initialize the distribution analyzer."""
        self.config = config
        self._opponent_value_counts: dict[int, Counter] = defaultdict(Counter)
        self._opponent_issue_weights: list[float] = []
        self._last_distribution_update = 0
        self._opponent_offers = 0
    
    def initialize(self, n_issues: int, issue_values: list[set]) -> None:
        """Initialize weights for opponent issue importance."""
        self._opponent_issue_weights = (
            [1.0 / n_issues] * n_issues if n_issues > 0 else []
        )
    
    def update(self, offer: Outcome, weight: float = 1.0) -> None:
        """Record an opponent offer for distribution analysis."""
        self._opponent_offers += 1
        weight = max(0.0, weight)
        for issue_index, value in enumerate(offer):
            self._opponent_value_counts[issue_index][value] += weight
    
    def get_issue_weight(self, issue_index: int) -> float:
        """Get estimated importance weight for an issue."""
        if 0 <= issue_index < len(self._opponent_issue_weights):
            return self._opponent_issue_weights[issue_index]
        return 1.0 / max(len(self._opponent_issue_weights), 1)
    
    def should_update_weights(self, opponent_offers: int, window_size: int = 4) -> bool:
        """Check if enough offers received to update distribution model."""
        return (
            opponent_offers >= window_size * 2
            and opponent_offers - self._last_distribution_update >= window_size
            and len(self._opponent_issue_weights) > 0
        )
    
    def update_weights_from_concession(
        self,
        offers: list[Outcome],
        issue_values: list[set],
        relative_time: float,
    ) -> None:
        """Update issue weights based on opponent concession patterns."""
        window_size = self.config.WINDOW_SIZE
        if len(offers) < window_size * 2:
            return
        
        previous = offers[-window_size * 2:-window_size]
        current = offers[-window_size:]
        
        stable_issues = []
        conceded = False
        
        for issue_index in range(len(issue_values)):
            prev_dist = self._window_distribution(previous, issue_index, issue_values)
            curr_dist = self._window_distribution(current, issue_index, issue_values)
            
            movement = self._distribution_distance(prev_dist, curr_dist)
            if movement <= self.config.DISTRIBUTION_CHANGE_THRESHOLD:
                stable_issues.append(issue_index)
                continue
            
            prev_expected = self._expected_issue_value(prev_dist, issue_index)
            curr_expected = self._expected_issue_value(curr_dist, issue_index)
            
            if curr_expected < prev_expected - 0.02:
                conceded = True
        
        if stable_issues and conceded:
            update = self.config.DISTRIBUTION_UPDATE_WEIGHT * (1.0 - min(1.0, relative_time))
            for idx in stable_issues:
                self._opponent_issue_weights[idx] += update
            self._normalize_weights()
        
        self._last_distribution_update = self._opponent_offers
    
    def _window_distribution(
        self,
        offers: list[Outcome],
        issue_index: int,
        issue_values: list[set],
    ) -> dict[object, float]:
        """Calculate probability distribution of values in a window."""
        values = (
            issue_values[issue_index]
            if issue_index < len(issue_values)
            else set()
        )
        denominator = len(offers) + len(values)
        if denominator <= 0:
            return {}
        
        distribution = {}
        for value in values:
            count = sum(
                1
                for offer in offers
                if issue_index < len(offer) and offer[issue_index] == value
            )
            distribution[value] = (count + 1) / denominator
        return distribution
    
    def _distribution_distance(
        self,
        first: dict[object, float],
        second: dict[object, float],
    ) -> float:
        """Measure distance between two distributions (L1 distance)."""
        values = set(first) | set(second)
        return sum(
            abs(first.get(value, 0.0) - second.get(value, 0.0))
            for value in values
        )
    
    def _expected_issue_value(
        self,
        distribution: dict[object, float],
        issue_index: int,
    ) -> float:
        """Calculate expected opponent preference for an issue."""
        counts = self._opponent_value_counts[issue_index]
        if not counts:
            return 0.0
        
        max_count = max(counts.values(), default=0)
        if max_count <= 0:
            return 0.0
        
        return sum(
            probability * pow(counts[value] / max_count, self.config.VALUE_SCORE_POWER)
            for value, probability in distribution.items()
        )
    
    def _normalize_weights(self) -> None:
        """Normalize issue weights to sum to 1."""
        total = sum(self._opponent_issue_weights)
        if total <= 0:
            n = len(self._opponent_issue_weights)
            self._opponent_issue_weights = [1.0 / n] * n
            return
        self._opponent_issue_weights = [w / total for w in self._opponent_issue_weights]


class Hallucinators(BOANeg):
    """
    Advanced negotiation agent with sophisticated opponent modeling
    and adaptive bidding strategy.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the negotiation agent."""
        super().__init__(*args, **kwargs)
        self.config = NegotiationConfig()
        self.analyzer = DistributionAnalyzer(self.config)
        self._build_tag = AGENT_BUILD_TAG
        
        # Outcome management
        self._outcomes: list[tuple[float, Outcome]] = []
        self._issue_values: list[set] = []
        self._max_utility = 1.0
        self._issue_costs: list[float] = []
        self._frontier_cache: dict[str, list[tuple[float, Outcome]]] = {}
        self._percentile_cache: dict[str, float] = {}
        
        # Offer history
        self._last_offer: Optional[Outcome] = None
        self._recent_offers = deque(maxlen=18)
        self._own_offers = 0
        self._opponent_offer_history: deque[tuple[float, float, float, Outcome]] = deque(maxlen=18)
        self._opponent_offers = 0
        self._unique_opponent_offers: set[Outcome] = set()
        self._best_opponent_offer: Optional[Outcome] = None
        self._best_opponent_utility = float("-inf")
        self._opponent_offer_repetitions: Counter = Counter()
    
    def on_preferences_changed(self, changes) -> None:
        """Called when agent preferences are updated."""
        if self.ufun is None or self.nmi is None:
            return
        
        # Collect candidate outcomes.  On huge domains, enumerating and
        # evaluating the ufun over the whole space costs ~200s per negotiation
        # (exceeds the competition time limit), so above MAX_OUTCOMES we sample
        # a representative subset instead.  A fixed seed makes the sample (and
        # therefore all downstream bids) identical across runs, removing
        # run-to-run drift; the RNG state is restored afterward.
        import random as _random
        try:
            import numpy as _np
        except Exception:
            _np = None
        _rng_state = _random.getstate()
        _np_state = _np.random.get_state() if _np is not None else None
        # negmas sampling uses numpy's RNG, so seed both to make the sampled
        # outcome set (and all downstream bids) identical across runs.
        _random.seed(self.config.OUTCOME_SAMPLE_SEED)
        if _np is not None:
            _np.random.seed(self.config.OUTCOME_SAMPLE_SEED)
        try:
            cardinality = self.nmi.outcome_space.cardinality
            cap = self.config.MAX_OUTCOMES
            if isinstance(cardinality, (int, float)) and cardinality > cap:
                sampled_outcomes = list(
                    self.nmi.outcome_space.sample(
                        int(cap), with_replacement=False, fail_if_not_enough=False
                    )
                )
                # Guarantee our globally best outcome is present so our opening
                # aspiration is not capped below the true maximum by sampling.
                try:
                    best = self.ufun.best()
                    if best is not None:
                        sampled_outcomes.append(best)
                except Exception:
                    pass
            else:
                sampled_outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        finally:
            _random.setstate(_rng_state)
            if _np is not None and _np_state is not None:
                _np.random.set_state(_np_state)

        # Full per-issue value sets come from the issue definitions when
        # available (sampling could otherwise miss rare values), falling back to
        # whatever the sampled outcomes reveal.
        issue_values: list[set] = []
        try:
            for issue in self.nmi.outcome_space.issues:
                issue_values.append(set(issue.all))
        except Exception:
            issue_values = []

        outcomes = []
        for outcome in sampled_outcomes:
            utility = float(self.ufun(outcome))
            if utility > float(self.ufun.reserved_value):
                outcomes.append((utility, outcome))

            for i, value in enumerate(outcome):
                if i >= len(issue_values):
                    issue_values.append(set())
                issue_values[i].add(value)
        
        # Sort and store
        self._outcomes = sorted(
            outcomes,
            key=lambda item: (item[0], self._outcome_sort_key(item[1])),
            reverse=True,
        )
        self._issue_values = issue_values
        self._max_utility = self._outcomes[0][0] if self._outcomes else 1.0
        self._issue_costs = self._estimate_issue_costs()
        self._percentile_cache = self._build_percentile_cache()
        self._frontier_cache = self._build_frontier_cache()
        
        # Initialize distribution analyzer
        n_issues = len(issue_values)
        self.analyzer.initialize(n_issues, issue_values)
        
        # Provide opponent modeling utility function
        self.private_info["opponent_ufun"] = LambdaMultiFun(
            f=lambda outcome: self._estimated_opponent_utility(outcome)
        )

    def _build_frontier_cache(self) -> dict[str, list[tuple[float, Outcome]]]:
        """Precompute stable own-utility bands used by late bid selection."""
        if not self._outcomes:
            return {
                "top_utility": [],
                "safe_compromise": [],
                "masked_rotation": [],
            }

        top_utility = self._outcomes[0][0]
        top_band = [
            item
            for item in self._outcomes
            if top_utility - item[0] <= self._max_utility * self.config.EARLY_MASKING_SLACK
        ]
        safe_floor = self._adaptive_utility_floor(
            self.config.LATE_SAFE_FLOOR_RATIO, "p10"
        )
        safe_compromise = [
            item
            for item in self._outcomes
            if item[0] >= safe_floor
        ]
        masked_floor = max(
            top_utility - self._max_utility * self.config.LATE_MASKING_SLACK,
            safe_floor,
        )
        masked_rotation = [
            item
            for item in self._outcomes
            if item[0] >= masked_floor
        ]
        return {
            "top_utility": top_band or self._outcomes[:1],
            "safe_compromise": safe_compromise or self._outcomes[:],
            "masked_rotation": masked_rotation or top_band or self._outcomes[:1],
        }
    
    def _build_percentile_cache(self) -> dict[str, float]:
        """Compute utility percentile bands from the rational outcome space.

        Outcomes are pre-sorted descending, so index 0 is the best. A fraction f
        maps to the outcome at position int(f * n), giving the utility at the top-f
        percentile of rational outcomes above the reservation value.

        These percentiles adapt automatically to the scenario's utility distribution
        shape: sparse domains stay conservative; dense domains allow more concession.
        """
        reserved = float(self.ufun.reserved_value) if self.ufun is not None else 0.0
        if not self._outcomes:
            return {k: reserved for k in ("p01", "p05", "p10", "p20", "p35", "p50")}

        n = len(self._outcomes)

        def at_fraction(f: float) -> float:
            idx = min(n - 1, max(0, int(f * n)))
            return self._outcomes[idx][0]

        return {
            "p01": at_fraction(0.01),
            "p05": at_fraction(0.05),
            "p10": at_fraction(0.10),
            "p20": at_fraction(0.20),
            "p35": at_fraction(0.35),
            "p50": at_fraction(0.50),
        }

    def _adaptive_utility_floor(
        self,
        ratio: float,
        percentile_key: str,
        blend: float | None = None,
        relative_time: float = 0.0,
    ) -> float:
        """Return a floor that blends a fixed-ratio threshold with a percentile band.

        The blend prevents hardcoded absolute thresholds from over- or under-fitting
        to any specific scenario's utility distribution. A domain with many high-utility
        outcomes will yield a stricter floor from percentiles; a sparse domain will
        yield a looser one.

        A time-dependent minimum gain above reservation is enforced so the floor
        never collapses to near-reservation in sparse domains where percentile
        bands are very compressed:
          t < 0.97  → floor >= reserved + 0.20 * (max - reserved)
          0.97–0.99 → floor >= reserved + 0.15 * (max - reserved)
          t >= 0.99 → floor >= reserved + 0.10 * (max - reserved)

        Args:
            ratio: Fallback normalized ratio relative to max_utility (e.g. 0.68).
            percentile_key: One of p01/p05/p10/p20/p35/p50.
            blend: Weight on the percentile component (defaults to config value).
            relative_time: Current negotiation time, used for the minimum guard.
        """
        if blend is None:
            blend = self.config.PERCENTILE_BLEND_WEIGHT
        reserved = float(self.ufun.reserved_value) if self.ufun is not None else 0.0
        ratio_floor = self._max_utility * ratio
        percentile_floor = self._percentile_cache.get(percentile_key, ratio_floor)
        blended = blend * percentile_floor + (1.0 - blend) * ratio_floor

        # Time-dependent minimum gain above reservation.
        if relative_time >= 0.99:
            min_gain_fraction = 0.10
        elif relative_time >= 0.97:
            min_gain_fraction = 0.15
        else:
            min_gain_fraction = 0.20
        min_floor = reserved + min_gain_fraction * max(0.0, self._max_utility - reserved)

        return max(reserved, min_floor, blended)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Generate response to opponent's offer or provide counter-offer.

        This method is intentionally defensive: offer bookkeeping, acceptance
        logic, and candidate selection are isolated so an unusual domain or
        runtime bug cannot crash the negotiation or force an invalid response.
        """
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        relative_time = 0.0
        step = 0
        offer = None
        try:
            offer = state.current_offer
            relative_time = float(state.relative_time)
            step = int(state.step)
        except Exception:
            pass

        # Process opponent's offer and check acceptance. Failures here must
        # not block us from still proposing a counter-offer.
        if offer is not None:
            try:
                self._remember_opponent_offer(offer, relative_time)
            except Exception:
                pass
            try:
                if self._should_accept(offer, relative_time, step):
                    return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            except Exception:
                pass

        # Generate a counter-offer, falling through independent fallbacks so
        # one failing layer cannot take down the rest.
        counter_offer = None
        try:
            counter_offer = self._select_offer(relative_time, step)
        except Exception:
            counter_offer = None

        if counter_offer is None:
            try:
                counter_offer = self._safe_fallback_offer(relative_time)
            except Exception:
                counter_offer = None
        if counter_offer is None:
            try:
                counter_offer = self._absolute_fallback_offer()
            except Exception:
                counter_offer = None
        if counter_offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        try:
            self._record_own_offer(counter_offer)
        except Exception:
            pass
        return SAOResponse(ResponseType.REJECT_OFFER, counter_offer)

    def _absolute_fallback_offer(self) -> Optional[Outcome]:
        """Guaranteed-valid last resort if any outcome exists."""
        if self._outcomes:
            return self._outcomes[0][1]
        if self.nmi is not None:
            try:
                for outcome in self.nmi.outcome_space.enumerate_or_sample():
                    return outcome
            except Exception:
                return None
        return None

    def _safe_fallback_offer(self, relative_time: float = 0.0) -> Optional[Outcome]:
        """Return a valid high-own-utility offer if the strategic path fails."""
        if relative_time >= self.config.EMERGENCY_DEADLINE_START:
            deadline_offer = self._emergency_deadline_offer(relative_time, 0)
            if deadline_offer is not None:
                return deadline_offer
        if self._frontier_cache.get("safe_compromise"):
            return self._frontier_cache["safe_compromise"][0][1]
        if self._outcomes:
            return self._outcomes[0][1]
        return None
    
    def _should_accept(
        self,
        offer: Outcome,
        relative_time: float,
        step: int,
    ) -> bool:
        """Determine whether to accept opponent's offer."""
        assert self.ufun is not None
        
        offer_utility = float(self.ufun(offer))
        reserved = float(self.ufun.reserved_value)
        
        # Reject if below reservation value
        if offer_utility <= reserved:
            return False
        
        # Check for extreme conflict rescue
        if self._should_rescue_extreme_conflict(relative_time):
            compromise = self._extreme_conflict_offer()
            if (
                compromise is not None
                and offer_utility >= float(self.ufun(compromise)) * 0.98
            ):
                return True
        
        # Late-game acceptance heuristic
        if self._accombi_accept(offer_utility, relative_time):
            return True

        # NOTE: rational-acceptance was removed here.  Ablation against the firm
        # base showed it accepted forecast-low offers and caused us to cave to
        # hardliners (NoRationalAccept out-scored full Jupiter).  We now hold for
        # the aspiration target instead.

        # Target utility acceptance
        target = self._target_utility(relative_time, step)
        if offer_utility >= target:
            return True

        # At the true deadline, do not lose the best credible offer already
        # demonstrated by the opponent.
        if (
            self._is_last_chance(step, relative_time)
            and offer_utility >= self._best_opponent_utility
            and offer_utility >= target * self.config.BEST_OFFER_TARGET_MARGIN
        ):
            return True

        # Compare against a cheap fallback plan. Calling the full strategic
        # selector from inside acceptance can rescan large candidate spaces.
        planned = self._safe_fallback_offer()
        if planned is not None:
            planned_utility = float(self.ufun(planned))
            margin = self._counter_offer_margin(relative_time)
            if offer_utility >= planned_utility * margin:
                return True
        
        # Late-game emergency floor acceptance (shared with the base agent):
        # near the true deadline, accept offers that clear the protected floor.
        if (
            relative_time > self.config.LATE_GAME_THRESHOLD
            and offer_utility >= max(
                reserved,
                self._emergency_floor_utility(relative_time),
            )
        ):
            return True

        # Survival net — ERRATIC opponents ONLY.  Against a firm/rational
        # opponent, timing out to the reservation value beats accepting a weak
        # deadline offer (the firm base agent wins exactly this way), so the net
        # must NOT fire there.  Against a genuinely erratic opponent, grabbing a
        # deal a margin above reservation is worth more than a likely no-deal.
        if (
            self._opponent_is_erratic()
            and relative_time >= self.config.SURVIVAL_ACCEPT_THRESHOLD
            and offer_utility >= self._survival_floor_utility()
        ):
            return True

        return False

    def _rational_accept(self, offer_utility: float, relative_time: float) -> bool:
        """Accept when the offer is within a confidence-scaled margin of the
        best utility we forecast the opponent will offer before the deadline.

        Guards keep this conservative: only fires mid/late, requires a
        confident forecast, and the offer must clear the emergency floor.  This
        protects score (we only accept near the best we expect) while cutting
        the no-deal rate (we stop holding out for gains that will not come).
        """
        if relative_time < self.config.RATIONAL_ACCEPT_MIN_TIME:
            return False
        if len(self._opponent_offer_history) < self.config.FORECAST_MIN_OFFERS:
            return False

        # Opponent-adaptive: against an exploitable (still-conceding) opponent
        # do NOT settle early — hold out for the aspiration target and squeeze.
        # Rational-accept is for tough/erratic opponents where waiting will not
        # yield anything better.
        if self._opponent_stance() == "exploitable":
            return False

        forecast, confidence = self._opponent_deadline_forecast()
        if confidence < self.config.RATIONAL_ACCEPT_MIN_CONFIDENCE:
            return False

        # Tighter margin when confident, looser when uncertain.
        margin = self.config.RATIONAL_ACCEPT_MARGIN - (1.0 - confidence) * 0.04
        floor = self._emergency_floor_utility(relative_time)
        if offer_utility < floor:
            return False
        return offer_utility >= forecast * margin

    def _survival_floor_utility(self) -> float:
        """Minimum survivable utility: a small fixed margin above reservation."""
        reserved = float(self.ufun.reserved_value) if self.ufun is not None else 0.0
        span = max(0.0, self._max_utility - reserved)
        return reserved + self.config.SURVIVAL_MARGIN_FRACTION * span

    def _opponent_offer_variance(self) -> float:
        """Normalized standard deviation of opponent offer utility-for-us."""
        history = list(self._opponent_offer_history)
        if len(history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0
        utilities = [item[1] for item in history]
        mean = sum(utilities) / len(utilities)
        variance = sum((u - mean) ** 2 for u in utilities) / len(utilities)
        std = variance ** 0.5
        return min(1.0, std / self._max_utility) if self._max_utility else 0.0

    def _opponent_is_erratic(self) -> bool:
        """Detect messy/unpredictable opponents with no learnable concession.

        Erratic = jumpy offers (high variance), no improving concession trend,
        and a model we cannot trust (low forecast confidence).  Against such
        opponents holding out is futile, so the survival-accept net activates.
        """
        if len(self._opponent_offer_history) < self.config.ERRATIC_MIN_HISTORY:
            return False
        _, confidence = self._opponent_deadline_forecast()
        concession = self._opponent_concession_rate()
        variance = self._opponent_offer_variance()
        return (
            variance >= self.config.ERRATIC_VARIANCE_THRESHOLD
            and concession <= self.config.CONCESSION_DETECTION_THRESHOLD * self._max_utility
            and confidence < self.config.ERRATIC_CONFIDENCE_MAX
        )

    def _opponent_stance(self) -> str:
        """Classify the opponent into a stance that drives firm-vs-flexible play.

        Returns one of:
          "exploitable" - conceding and not hard: stay FIRM, squeeze value.
          "tough"       - hardliner or fixed-bar acceptor: engage flexible
                          (Pareto) bidding and rational-accept to reach a deal.
          "erratic"     - chaotic/unpredictable: rely on the survival net.
          "neutral"     - not enough evidence or mixed: balanced default.

        This is the core of opponent-adaptive behavior: the same agent plays
        like the firm base against pushovers and like a flexible compromiser
        only against opponents where holding firm would cause a walk-away.
        """
        if self._opponent_is_erratic():
            return "erratic"
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return "neutral"
        hardliner = self._hardliner_score()
        conceder = self._conceder_score()
        threshold = self._threshold_acceptor_score()
        if (
            conceder >= self.config.EXPLOITABLE_CONCEDER_MIN
            and hardliner <= self.config.EXPLOITABLE_HARDLINER_MAX
        ):
            return "exploitable"
        if (
            hardliner >= self.config.TOUGH_HARDLINER_MIN
            or threshold >= self.config.TOUGH_THRESHOLD_MIN
        ):
            return "tough"
        return "neutral"

    def _accombi_accept(self, offer_utility: float, relative_time: float) -> bool:
        """Accept if offer matches late-game window average."""
        if (
            relative_time < self.config.ACCOMBI_THRESHOLD
            or len(self._opponent_offer_history) < self.config.MIN_OFFER_HISTORY
        ):
            return False
        
        window_start = 2.0 * relative_time - 1.0
        window_utilities = [
            utility
            for time, utility, _, _ in self._opponent_offer_history
            if time >= window_start
        ]

        # Guarantee the window is large enough to be meaningful.  Falling
        # back to the full (potentially noisy) history when the time-window
        # returns too few entries produced unstable acceptance decisions.
        # Use the most recent MIN_OFFER_HISTORY entries instead.
        if len(window_utilities) < self.config.MIN_OFFER_HISTORY:
            window_utilities = [
                utility for _, utility, _, _ in self._opponent_offer_history
            ][-self.config.MIN_OFFER_HISTORY:]

        if not window_utilities:
            return False

        average = sum(window_utilities) / len(window_utilities)
        return offer_utility >= average
    
    def _target_utility(self, relative_time: float, step: int | None = None) -> float:
        """Calculate target utility for current phase."""
        assert self.ufun is not None
        
        t = max(0.0, min(1.0, relative_time))
        reserved = float(self.ufun.reserved_value)
        floor = self._adaptive_utility_floor(self.config.FLOOR_UTILITY_RATIO, "p20", relative_time=t)

        # Tier 2: adaptive concession exponents by opponent type.
        mid_exp, late_exp = self._concession_exponents()

        # Early phase: aggressive
        if t < self.config.EARLY_PHASE_THRESHOLD:
            base_target = max(floor, self._max_utility * self.config.EARLY_TARGET_UTILITY)

        # Mid phase: gradual concession
        elif t < self.config.MID_PHASE_THRESHOLD:
            concession = pow(
                (t - self.config.EARLY_PHASE_THRESHOLD) /
                (self.config.MID_PHASE_THRESHOLD - self.config.EARLY_PHASE_THRESHOLD),
                mid_exp
            )
            base_target = (
                self._max_utility * self.config.EARLY_TARGET_UTILITY
                - (self._max_utility * 0.17) * concession
            )
        else:
            # Late phase: final concession
            concession = pow(
                (t - self.config.MID_PHASE_THRESHOLD) /
                (1.0 - self.config.MID_PHASE_THRESHOLD),
                late_exp
            )
            base_target = (
                self._max_utility * self.config.LATE_TARGET_UTILITY
                - (self._max_utility * 0.11) * concession
            )

        base_target += self._responsive_concession_adjustment(t)
        if step is not None and self._is_last_chance(step, t):
            base_target -= self._max_utility * self.config.FINAL_TURN_TARGET_RELIEF
        return max(floor, min(self._max_utility, base_target))

    def _concession_exponents(self) -> tuple[float, float]:
        """Return (mid_exp, late_exp) for the concession curve by opponent type.

        Against hardliners and conceders — who do not reward early concession —
        we stay Boulware longer (higher exponents) to protect own utility.  Note
        this only shapes our *offers*; the survival-accept net still lets us
        close a deal if such an opponent does put a survivable offer on the
        table.  Falls back to defaults without enough evidence.
        """
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return (self.config.EXP_DEFAULT_MID, self.config.EXP_DEFAULT_LATE)

        hardliner = self._hardliner_score()
        concession_rate = self._opponent_concession_rate()
        if hardliner > 0.55:
            return (self.config.EXP_HARDLINER_MID, self.config.EXP_HARDLINER_LATE)
        if concession_rate > self.config.CONCESSION_DETECTION_THRESHOLD * 2:
            return (self.config.EXP_CONCEDER_MID, self.config.EXP_CONCEDER_LATE)
        return (self.config.EXP_DEFAULT_MID, self.config.EXP_DEFAULT_LATE)

    def _counter_offer_margin(self, relative_time: float) -> float:
        """Calculate acceptance margin for counter-offer."""
        t = max(0.0, min(1.0, relative_time))
        return 0.995 - 0.08 * t
    
    def _select_offer(self, relative_time: float, step: int) -> Optional[Outcome]:
        """Select best counter-offer for current phase."""
        if not self._outcomes:
            return None
        
        # Extreme conflict rescue
        if self._should_rescue_extreme_conflict(relative_time):
            return self._extreme_conflict_offer()

        target = self._target_utility(relative_time, step)

        emergency_offer = self._emergency_deadline_offer(relative_time, step)
        if emergency_offer is not None:
            return emergency_offer

        model_trustworthy = self._opponent_model_is_trustworthy()
        stance = self._opponent_stance()

        # _opponent_acceptance_close_offer searches for opponent-friendly bids
        # to clear their acceptance bar near the deadline.  Skip it against an
        # exploitable opponent — they are conceding toward us, so we hold firm
        # rather than hand them a friendly bid.  Its own internal guards handle
        # the tough/scripted cases where a reachable close offer matters.
        if stance != "exploitable":
            acceptance_close_offer = self._opponent_acceptance_close_offer(
                relative_time,
                target,
            )
            if acceptance_close_offer is not None:
                return acceptance_close_offer

        closeable_offer = self._best_opponent_close_offer(relative_time)
        if closeable_offer is not None:
            return closeable_offer

        if (
            self._is_last_chance(step, relative_time)
            and self._best_opponent_offer is not None
            and self._best_opponent_utility
            >= target * self.config.BEST_OFFER_TARGET_MARGIN
        ):
            return self._best_opponent_offer

        # Get candidate outcomes
        candidates = [(u, o) for u, o in self._outcomes if u >= target]
        
        if not candidates:
            floor = max(
                float(self.ufun.reserved_value) if self.ufun else 0.0,
                self._emergency_floor_utility(relative_time),
            )
            candidates = [(u, o) for u, o in self._outcomes if u >= floor]
        
        if not candidates:
            candidates = self._outcomes

        if model_trustworthy:
            candidates = self._two_level_candidates(candidates, relative_time)

        special_offer = self._phase_specific_offer(candidates, relative_time, step)
        if special_offer is not None:
            return special_offer
        
        # Score and rank candidates
        ranked = self._score_candidates(candidates, relative_time)
        
        # Select from masked window
        window = self._masked_window(ranked, relative_time)
        if not window:
            return self._outcomes[-1][1] if self._outcomes else None
        
        return window[0][2]

    def _phase_specific_offer(
        self,
        candidates: list[tuple[float, Outcome]],
        relative_time: float,
        step: int,
    ) -> Optional[Outcome]:
        """Clean RUFL-inspired phase policy over already-safe candidates."""
        if not candidates:
            return None

        confidence = self._opponent_model_confidence()
        model_trustworthy = self._opponent_model_is_trustworthy()

        # Against an exploitable opponent stay firm even in the late phase:
        # rotate the highest-own-utility bids rather than drifting to the
        # opponent-friendly frequency band.  They are conceding toward us, so
        # holding the top band captures more without risking the deal.
        if self._opponent_stance() == "exploitable":
            return self._rotate_high_own_utility(candidates, step, late=False)

        if relative_time >= self.config.DEADLINE_RANGE_THRESHOLD:
            deadline_offer = self._deadline_range_offer(candidates, relative_time)
            if deadline_offer is not None:
                return deadline_offer

        if relative_time >= self.config.LATE_RUFL_SELECTION_THRESHOLD:
            if (
                model_trustworthy
                and confidence >= self.config.WEAK_MODEL_CONFIDENCE_THRESHOLD
            ):
                return self._frequency_safe_band_offer(candidates, step)
            return self._rotate_high_own_utility(candidates, step, late=False)

        return self._rotate_high_own_utility(candidates, step, late=False)

    def _emergency_deadline_offer(
        self,
        relative_time: float,
        step: int,
    ) -> Optional[Outcome]:
        """Last-window recovery that avoids unsafe concessions and model traps."""
        if relative_time < self.config.EMERGENCY_DEADLINE_START:
            return None

        hard_floor = self._hard_safe_floor_utility(relative_time)
        if (
            self._best_opponent_offer is not None
            and self._best_opponent_utility >= hard_floor
        ):
            return self._best_opponent_offer

        safe_candidates = [
            (utility, outcome)
            for utility, outcome in self._outcomes
            if utility >= hard_floor
        ]
        if not safe_candidates:
            safe_candidates = self._outcomes[: max(1, min(5, len(self._outcomes)))]
        if not safe_candidates:
            return None

        reasonable = [
            (utility, outcome)
            for utility, outcome in safe_candidates
            if self._estimated_opponent_utility(outcome)
            >= self.config.EMERGENCY_REASONABLE_OPPONENT_SCORE
        ]
        if reasonable:
            safe_candidates = reasonable

        return max(
            safe_candidates,
            key=lambda item: (
                self._estimated_opponent_utility(item[1]),
                item[0],
                self._not_recent_exact_repeat(item[1]),
                self._outcome_sort_key(item[1]),
            ),
        )[1]
    
    def _score_candidates(
        self,
        candidates: list[tuple[float, Outcome]],
        relative_time: float,
    ) -> list[tuple[float, float, Outcome]]:
        """Score candidate outcomes across multiple dimensions."""
        ranked = []
        
        for utility, outcome in candidates:
            own_score = self._normalized_own_utility(utility)
            opponent_score = self._estimated_opponent_utility(outcome)
            acceptance_score = self._acceptance_likelihood(
                outcome, opponent_score, relative_time
            )
            pareto_score = self._pareto_score(own_score, opponent_score)
            masking_score = self._masking_score(outcome)
            deadline_score = self._deadline_score(relative_time, opponent_score)
            decoy_score = self._decoy_signal_score(outcome, relative_time)
            lookahead_score = self._one_step_lookahead_tiebreak(
                utility,
                acceptance_score,
                relative_time,
            )
            
            own_weight, accept_weight, pareto_weight, masking_weight = (
                self._weights(relative_time)
            )
            
            score = (
                own_weight * own_score
                + accept_weight * acceptance_score
                + pareto_weight * pareto_score
                + masking_weight * masking_score
                + 0.04 * deadline_score
                + 0.035 * decoy_score
                + self.config.LOOKAHEAD_TIEBREAK_WEIGHT * lookahead_score
            )
            
            ranked.append((score, utility, outcome))
        
        ranked.sort(
            key=lambda item: (item[0], item[1], self._outcome_sort_key(item[2])),
            reverse=True,
        )
        return ranked
    
    def _weights(self, relative_time: float) -> tuple[float, float, float, float]:
        """Get component weights based on negotiation phase.

        Three-tier late-game system:
          t < 0.94              → WEIGHT_LATE   (own 0.48, accept 0.28)
          0.94 <= t < 0.985    → WEIGHT_EMERGENCY (own 0.28, accept 0.44)
          t >= 0.985           → WEIGHT_ULTRA_LATE (own 0.08, accept 0.68)

        The moderate EMERGENCY step (not the extreme 0.12/0.58) preserves some
        self-utility pressure for the 5-10 steps before the final 1-2 steps,
        where ULTRA_LATE makes closing the absolute priority.
        """
        # Ultra-late: closing is the only goal — no profile shift, fixed weights.
        if relative_time >= self.config.ULTRA_LATE_THRESHOLD:
            return self.config.WEIGHT_ULTRA_LATE
        if relative_time >= self.config.EMERGENCY_WEIGHT_THRESHOLD:
            base = self.config.WEIGHT_EMERGENCY
        elif relative_time < self.config.EARLY_PHASE_THRESHOLD:
            base = self.config.WEIGHT_EARLY
        elif relative_time < self.config.MID_PHASE_THRESHOLD:
            base = self.config.WEIGHT_MID
        else:
            base = self.config.WEIGHT_LATE
        return self._profile_adjusted_weights(base)

    def _profile_adjusted_weights(
        self,
        base: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Tier 2: shift scoring weights by opponent type when confident.

        Hardliner → raise acceptance weight (we need them to accept, not just
        match our target).  Conceder → raise pareto weight (they will come to
        us; focus on joint quality).  Threshold acceptor → raise own weight
        (they accept at a fixed bar, so maximise ours).  Shifts are small and
        renormalized, so this nudges behavior without destabilizing it.
        """
        confidence = self._opponent_model_confidence()
        if confidence < self.config.PROFILE_MIN_CONFIDENCE:
            return base

        own_w, accept_w, pareto_w, mask_w = base
        shift = self.config.PROFILE_WEIGHT_SHIFT * confidence
        hardliner = self._hardliner_score()
        conceder = self._conceder_score()
        threshold = self._threshold_acceptor_score()

        if hardliner > 0.55:
            accept_w = min(1.0, accept_w + shift * hardliner)
            own_w = max(0.0, own_w - shift * hardliner * 0.5)
        elif conceder > 0.55:
            pareto_w = min(1.0, pareto_w + shift * conceder)
            mask_w = max(0.0, mask_w - shift * conceder * 0.5)
        elif threshold > 0.55:
            own_w = min(1.0, own_w + shift * threshold * 0.5)
            accept_w = max(0.0, accept_w - shift * threshold * 0.3)

        total = own_w + accept_w + pareto_w + mask_w
        if total > 0:
            own_w, accept_w, pareto_w, mask_w = (
                own_w / total, accept_w / total, pareto_w / total, mask_w / total
            )
        return (own_w, accept_w, pareto_w, mask_w)

    def _remember_opponent_offer(self, offer: Outcome, relative_time: float) -> None:
        """Record and analyze opponent's offer."""
        self._opponent_offers += 1
        
        # Update distribution analyzer. Earlier offers are usually closer to
        # the opponent's real preference; repeated identical offers are less
        # informative than genuinely new concessions.
        model_weight = self._opponent_model_update_weight(offer, relative_time)
        self.analyzer.update(offer, model_weight)
        
        # Record offer details
        own_utility = float(self.ufun(offer)) if self.ufun else 0.0
        opponent_self_score = self._estimated_opponent_utility(offer)
        self._opponent_offer_history.append(
            (relative_time, own_utility, opponent_self_score, offer)
        )
        if own_utility > self._best_opponent_utility:
            self._best_opponent_offer = offer
            self._best_opponent_utility = own_utility
        try:
            self._unique_opponent_offers.add(offer)
        except TypeError:
            self._unique_opponent_offers.add(self._outcome_sort_key(offer))
        
        # Update issue weights if conditions met
        if self.analyzer.should_update_weights(self._opponent_offers):
            offers = [item[3] for item in self._opponent_offer_history]
            self.analyzer.update_weights_from_concession(
                offers, self._issue_values, relative_time
            )

    def _opponent_model_update_weight(
        self,
        offer: Outcome,
        relative_time: float,
    ) -> float:
        """Weight how much one opponent offer should teach the value model."""
        key = self._outcome_sort_key(offer)
        previous_repetitions = self._opponent_offer_repetitions[key]
        self._opponent_offer_repetitions[key] += 1

        t = max(0.0, min(1.0, relative_time))
        time_weight = (
            1.0
            - (1.0 - self.config.LATE_OFFER_MODEL_WEIGHT) * t
        )
        repeat_weight = max(
            self.config.MIN_REPEATED_OFFER_WEIGHT,
            self.config.REPEAT_OFFER_DAMPING ** previous_repetitions,
        )
        return time_weight * repeat_weight
    
    def _record_own_offer(self, offer: Optional[Outcome]) -> None:
        """Record own generated offer."""
        if offer is None:
            return
        self._own_offers += 1
        self._last_offer = offer
        self._recent_offers.append(offer)
    
    def _estimated_opponent_utility(self, outcome: Outcome) -> float:
        """Estimate opponent's utility for an outcome."""
        if self.analyzer._opponent_offers == 0:
            # Cold-start: use mild negative correlation with own utility rather
            # than a flat 0.5.  In most domains own and opponent utilities are
            # inversely correlated, so high-utility outcomes for us are likely
            # low-utility for them.  This gives the early scoring a better
            # prior than treating every candidate as equally acceptable.
            if self.ufun is not None and self._max_utility > 0:
                own_u = float(self.ufun(outcome)) / self._max_utility
                return max(0.10, 0.70 - 0.30 * own_u)
            return self.config.DEFAULT_OPPONENT_UTILITY
        
        weighted_scores = []
        weights = []
        
        for issue_index, value in enumerate(outcome):
            counts = self.analyzer._opponent_value_counts[issue_index]
            total = sum(counts.values())
            if total == 0:
                continue
            
            n_values = max(1, len(self._issue_values[issue_index]))
            smoothed = (
                (counts[value] + self.config.SMOOTHING_FACTOR)
                / (total + n_values)
            )
            max_smoothed = (
                (max(counts.values()) + self.config.SMOOTHING_FACTOR)
                / (total + n_values)
            )
            
            raw_score = smoothed / max_smoothed if max_smoothed > 0 else smoothed
            value_score = pow(raw_score, self.config.VALUE_SCORE_POWER)
            issue_weight = self.analyzer.get_issue_weight(issue_index)
            
            weighted_scores.append(value_score)
            weights.append(issue_weight)
        
        if not weighted_scores:
            return self.config.DEFAULT_OPPONENT_UTILITY
        
        weight_sum = sum(weights)
        if weight_sum <= 0:
            return sum(weighted_scores) / len(weighted_scores)
        
        return sum(s * w for s, w in zip(weighted_scores, weights)) / weight_sum

    def _normalized_own_utility(self, utility: float) -> float:
        """Normalize own utility above reservation value to the range zero-one."""
        if self.ufun is None:
            return 0.0
        reserved = float(self.ufun.reserved_value)
        span = self._max_utility - reserved
        if span <= 0:
            return 1.0 if utility >= self._max_utility else 0.0
        return max(0.0, min(1.0, (utility - reserved) / span))

    def _two_level_candidates(
        self,
        candidates: list[tuple[float, Outcome]],
        relative_time: float,
    ) -> list[tuple[float, Outcome]]:
        """Two-level filtration: own utility first, then opponent-friendly.

        Level 1 keeps a band of high own-utility outcomes (within a small slack
        of the best available).  Level 2 restricts that band to the *estimated
        Pareto frontier* and ranks by opponent acceptance.  Pareto restriction
        is the key non-aggressive score lever: it removes outcomes that are
        worse for the opponent without being any better for us, so we keep the
        same own-utility while offering bids more likely to be accepted.
        """
        if (
            relative_time < self.config.TWO_LEVEL_SELECTION_THRESHOLD
            or len(self._opponent_offer_history) < self.config.TWO_LEVEL_MIN_HISTORY
            or len(candidates) <= 2
        ):
            return candidates

        best_utility = max(utility for utility, _ in candidates)
        safe_floor = best_utility - self._max_utility * self.config.TWO_LEVEL_OWN_UTILITY_SLACK
        safe = [(utility, outcome) for utility, outcome in candidates if utility >= safe_floor]
        if len(safe) <= 2:
            return safe or candidates

        stance = self._opponent_stance()

        # Against EXPLOITABLE opponents stay firm: keep the top own-utility
        # band and rank by own utility.  Do not engage opponent-friendly Pareto
        # bidding — a pushover does not require the concession, so offering it
        # just gives away value (this is what cost score vs the firm base).
        if stance == "exploitable":
            ranked = sorted(
                safe,
                key=lambda item: (item[0], self._outcome_sort_key(item[1])),
                reverse=True,
            )
            keep = max(2, int(len(ranked) * self.config.TWO_LEVEL_KEEP_FRACTION))
            return ranked[:keep]

        # TOUGH / NEUTRAL / ERRATIC: restrict to the estimated Pareto frontier
        # (reachable-for-opponent bids at no own-utility waste), then lead with
        # acceptance likelihood to clear the opponent's bar without lowering our
        # own-utility band — this is where flexibility actually buys a deal.
        if (
            relative_time >= self.config.PARETO_BIDDING_THRESHOLD
            and len(self._opponent_offer_history) >= self.config.PARETO_MIN_HISTORY
        ):
            frontier = self._pareto_frontier_candidates(safe)
            if len(frontier) >= 2:
                safe = frontier

        ranked = sorted(
            safe,
            key=lambda item: (
                self._acceptance_likelihood(
                    item[1], self._estimated_opponent_utility(item[1]), relative_time
                ),
                item[0],
                self._outcome_sort_key(item[1]),
            ),
            reverse=True,
        )
        keep = max(2, int(len(ranked) * self.config.TWO_LEVEL_KEEP_FRACTION))
        return ranked[:keep]

    def _pareto_frontier_candidates(
        self,
        candidates: list[tuple[float, Outcome]],
    ) -> list[tuple[float, Outcome]]:
        """Return the estimated-Pareto-efficient subset of candidates.

        Strict dominance in (own_utility, estimated_opponent_utility) space: an
        outcome is dropped only if another candidate is at least as good for us
        AND at least as good for the opponent, with at least one strict.  No
        own-utility slack — slack would let a lower-own bid dominate a
        higher-own one and collapse the frontier toward the opponent (i.e.
        cause over-concession).  This keeps our best bid AND opponent-friendly
        bids, removing only pure waste (bids worse for the opponent at no gain
        to us); the downstream acceptance ranking then chooses by phase.
        """
        if len(candidates) <= 2:
            return candidates

        scored = [
            (own, self._estimated_opponent_utility(outcome), outcome)
            for own, outcome in candidates
        ]
        frontier: list[tuple[float, Outcome]] = []
        for own, opp, outcome in scored:
            dominated = False
            for own2, opp2, _ in scored:
                if own2 >= own and opp2 >= opp and (own2 > own or opp2 > opp):
                    dominated = True
                    break
            if not dominated:
                frontier.append((own, outcome))
        return frontier or candidates

    def _opponent_model_confidence(self) -> float:
        """Estimate whether opponent-frequency signals are safe to trust."""
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0
        _, forecast_confidence = self._opponent_deadline_forecast()
        diversity = (
            len(self._unique_opponent_offers) / len(self._opponent_offer_history)
            if self._opponent_offer_history
            else 0.0
        )
        return max(0.0, min(1.0, 0.65 * forecast_confidence + 0.35 * diversity))

    def _opponent_model_is_trustworthy(self) -> bool:
        """Check whether the observed history is enough to trust frequency signals."""
        history = len(self._opponent_offer_history)
        if history < self.config.MIN_CONCESSION_HISTORY:
            return False
        unique = len(self._unique_opponent_offers)
        if unique < self.config.MODEL_TRUST_MIN_UNIQUE_OFFERS:
            return False
        diversity = unique / max(1, history)
        if diversity < self.config.MODEL_TRUST_MIN_DIVERSITY:
            return False
        max_repetition = max(self._opponent_offer_repetitions.values(), default=0)
        if max_repetition / max(1, self._opponent_offers) > self.config.MODEL_TRUST_MAX_REPEAT_SHARE:
            return False
        return self._opponent_model_confidence() >= self.config.RESPONSIVE_CONFIDENCE_THRESHOLD

    def _hard_safe_floor_utility(self, relative_time: float = 0.95) -> float:
        """Independent utility floor used for emergency deadline recovery.

        Near the true deadline the floor decays through progressively looser
        percentile bands so we can still close deals in sparse or asymmetric
        domains without permanently refusing every reachable offer.
        """
        if relative_time >= 0.98:
            pkey = "p50"
        elif relative_time >= 0.96:
            pkey = "p35"
        else:
            pkey = "p20"
        return self._adaptive_utility_floor(self.config.HARD_SAFE_FLOOR_RATIO, pkey, relative_time=relative_time)

    def _frequency_score(self, outcome: Outcome) -> float:
        """Score how much an outcome matches the opponent's repeated values."""
        score = 0.0
        for issue_index, value in enumerate(outcome):
            counts = self.analyzer._opponent_value_counts[issue_index]
            max_count = max(counts.values(), default=0.0)
            if max_count <= 0:
                continue
            score += self.analyzer.get_issue_weight(issue_index) * (
                counts[value] / max_count
            )
        return score

    def _not_recent_exact_repeat(self, outcome: Outcome) -> bool:
        """Prefer offers that are not exact recent repeats if alternatives exist."""
        return outcome not in self._recent_offers

    def _rotate_high_own_utility(
        self,
        candidates: list[tuple[float, Outcome]],
        step: int,
        late: bool,
    ) -> Outcome:
        """Rotate through high-own-utility candidates when model trust is weak."""
        if late:
            cached = self._frontier_candidates("masked_rotation", candidates)
            if cached:
                candidates = cached
        ordered = sorted(
            candidates,
            key=lambda item: (item[0], self._masking_score(item[1]), self._outcome_sort_key(item[1])),
            reverse=True,
        )
        top_utility = ordered[0][0]
        top_band = [
            item
            for item in ordered
            if top_utility - item[0] <= self._max_utility * self.config.EARLY_MASKING_SLACK
        ]
        pool = top_band or ordered[:max(1, min(7, len(ordered)))]
        non_repeats = [item for item in pool if self._not_recent_exact_repeat(item[1])]
        pool = non_repeats or pool
        stride = (
            self.config.PHASE_ROTATION_STRIDE_LATE
            if late
            else self.config.PHASE_ROTATION_STRIDE_EARLY
        )
        return pool[(step * stride) % len(pool)][1]

    def _frequency_safe_band_offer(
        self,
        candidates: list[tuple[float, Outcome]],
        step: int,
    ) -> Outcome:
        """Late game: choose opponent-friendly bids inside a safe utility band."""
        best_utility = max(utility for utility, _ in candidates)
        floor = max(
            best_utility - self._max_utility * self.config.LATE_MASKING_SLACK,
            self._adaptive_utility_floor(self.config.LATE_SAFE_FLOOR_RATIO, "p10"),
        )
        cached = self._frontier_candidates("safe_compromise", candidates)
        source = cached or candidates
        safe_band = [(utility, outcome) for utility, outcome in source if utility >= floor]
        if not safe_band:
            safe_band = candidates

        ranked = sorted(
            safe_band,
            key=lambda item: (
                self._frequency_score(item[1]),
                self._not_recent_exact_repeat(item[1]),
                item[0],
                self._outcome_sort_key(item[1]),
            ),
            reverse=True,
        )
        best_score = self._frequency_score(ranked[0][1])
        top = [item for item in ranked if self._frequency_score(item[1]) == best_score]
        non_repeats = [item for item in top if self._not_recent_exact_repeat(item[1])]
        top = non_repeats or top
        return top[(step * self.config.PHASE_ROTATION_STRIDE_LATE) % len(top)][1]

    def _frontier_candidates(
        self,
        band: str,
        candidates: list[tuple[float, Outcome]],
    ) -> list[tuple[float, Outcome]]:
        """Intersect a cached frontier band with the current safe candidates."""
        cached = self._frontier_cache.get(band, [])
        if not cached or not candidates:
            return []
        candidate_keys = {self._outcome_sort_key(outcome) for _, outcome in candidates}
        return [
            (utility, outcome)
            for utility, outcome in cached
            if self._outcome_sort_key(outcome) in candidate_keys
        ]

    def _deadline_range_offer(
        self,
        candidates: list[tuple[float, Outcome]],
        relative_time: float,
    ) -> Optional[Outcome]:
        """Search between best opponent offer and our threshold near deadline."""
        if self._best_opponent_offer is None or self._best_opponent_utility == float("-inf"):
            return None

        target = self._target_utility(relative_time)
        lower = max(
            self._best_opponent_utility,
            target - self._max_utility * self.config.DEADLINE_RANGE_MARGIN,
        )
        upper = max(target, lower)
        in_range = [
            (utility, outcome)
            for utility, outcome in candidates
            if lower <= utility <= upper
        ]
        if not in_range:
            return None

        return max(
            in_range,
            key=lambda item: (
                self._frequency_score(item[1]),
                self._not_recent_exact_repeat(item[1]),
                item[0],
                self._outcome_sort_key(item[1]),
            ),
        )[1]

    def _best_opponent_close_offer(
        self,
        relative_time: float,
    ) -> Optional[Outcome]:
        """Near deadline, repeat a good opponent bid instead of timing out."""
        if relative_time < self.config.BEST_OPPONENT_CLOSE_THRESHOLD:
            return None
        if self._best_opponent_offer is None:
            return None
        floor = self._adaptive_utility_floor(self.config.EMERGENCY_UTILITY_RATIO, "p35", relative_time=relative_time)
        if self._best_opponent_utility < floor:
            return None
        return self._best_opponent_offer

    def _opponent_acceptance_close_offer(
        self,
        relative_time: float,
        target: float,
    ) -> Optional[Outcome]:
        """Near timeout, prioritize bids likely to clear the opponent threshold."""
        if relative_time < self.config.OPPONENT_ACCEPTANCE_CLOSE_THRESHOLD:
            return None

        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return None

        if (
            self._best_opponent_utility != float("-inf")
            and self._best_opponent_utility
            >= target * self.config.OPPONENT_ACCEPTANCE_CLOSE_TARGET_MARGIN
        ):
            return None

        floor = self._adaptive_utility_floor(self.config.EMERGENCY_UTILITY_RATIO, "p35", relative_time=relative_time)
        candidates = [
            (utility, outcome)
            for utility, outcome in self._outcomes
            if utility >= floor
        ]
        if not candidates:
            return None

        acceptable = []
        for utility, outcome in candidates:
            opponent_score = self._estimated_opponent_utility(outcome)
            likelihood = self._acceptance_likelihood(
                outcome,
                opponent_score,
                relative_time,
            )
            if (
                likelihood >= self.config.OPPONENT_ACCEPTANCE_CLOSE_MIN_LIKELIHOOD
                or opponent_score
                >= self.config.OPPONENT_ACCEPTANCE_CLOSE_MIN_OPPONENT_SCORE
            ):
                score = (
                    0.52 * likelihood
                    + 0.28 * opponent_score
                    + 0.20 * self._normalized_own_utility(utility)
                )
                acceptable.append((score, utility, outcome))

        if not acceptable:
            return None

        acceptable.sort(
            key=lambda item: (
                item[0],
                self._not_recent_exact_repeat(item[2]),
                item[1],
                self._outcome_sort_key(item[2]),
            ),
            reverse=True,
        )
        return acceptable[0][2]

    def _one_step_lookahead_tiebreak(
        self,
        offer_utility: float,
        acceptance_probability: float,
        relative_time: float,
    ) -> float:
        """Estimate one accept-or-reject step only when evidence is reliable."""
        forecast, confidence = self._opponent_deadline_forecast()
        if (
            relative_time < self.config.LOOKAHEAD_THRESHOLD
            or confidence < self.config.LOOKAHEAD_CONFIDENCE_THRESHOLD
            or self.ufun is None
        ):
            return 0.0

        reserved = float(self.ufun.reserved_value)
        probability = max(0.0, min(1.0, acceptance_probability))
        continuation = max(reserved, min(self._max_utility, forecast))
        expected = probability * offer_utility + (1.0 - probability) * continuation
        return self._normalized_own_utility(expected)

    def _is_last_chance(self, step: int, relative_time: float) -> bool:
        """Detect the final practical response/proposal opportunity."""
        n_steps = getattr(self.nmi, "n_steps", None) if self.nmi is not None else None
        if isinstance(n_steps, int) and n_steps > 0:
            return step >= n_steps - 1
        return relative_time >= self.config.BEST_OFFER_REUSE_THRESHOLD

    def _pareto_score(self, own_score: float, opponent_score: float) -> float:
        """Calculate Pareto efficiency score."""
        nash_like = own_score * opponent_score
        welfare = 0.5 * own_score + 0.5 * opponent_score
        balance = 1.0 - abs(own_score - opponent_score)
        return 0.45 * nash_like + 0.35 * welfare + 0.20 * balance
    
    def _acceptance_likelihood(
        self,
        outcome: Outcome,
        opponent_score: float,
        relative_time: float,
    ) -> float:
        """Estimate likelihood opponent would accept this offer."""
        t = max(0.0, min(1.0, relative_time))
        similarity = self._similarity_to_opponent_offers(outcome)
        concession = self._opponent_concession_rate()
        hardliner = self._hardliner_score()
        
        patience_bonus = 0.20 * t
        hardliner_bonus = hardliner * similarity * (0.20 + 0.35 * t)
        conceder_penalty = max(0.0, concession) * (1.0 - t) * 0.12
        
        likelihood = (
            0.58 * opponent_score
            + 0.24 * similarity
            + patience_bonus
            + hardliner_bonus
            - conceder_penalty
        )
        return max(0.0, min(1.0, likelihood))
    
    def _opponent_concession_rate(self) -> float:
        """Estimate opponent's concession rate."""
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0
        
        history = list(self._opponent_offer_history)
        midpoint = len(history) // 2
        if midpoint == 0:
            return 0.0
        
        early = sum(item[1] for item in history[:midpoint]) / midpoint
        late_count = len(history) - midpoint
        if late_count == 0:
            return 0.0
        
        late = sum(item[1] for item in history[midpoint:]) / late_count
        return late - early

    def _opponent_deadline_forecast(self) -> tuple[float, float]:
        """Predict the best utility the opponent may offer by the deadline.

        This adapts Team 271's utility-fit idea without relying on SciPy curve
        fitting. Pairwise slopes provide a robust trend estimate, while offer
        diversity and covered time determine how much the forecast is trusted.
        """
        history = list(self._opponent_offer_history)
        if len(history) < self.config.FORECAST_MIN_OFFERS:
            best = max((item[1] for item in history), default=0.0)
            return best, 0.0

        slopes = []
        for index, (time_a, utility_a, _, _) in enumerate(history):
            for time_b, utility_b, _, _ in history[index + 1:]:
                elapsed = time_b - time_a
                if elapsed > 1e-6:
                    slopes.append((utility_b - utility_a) / elapsed)

        trend = median(slopes) if slopes else 0.0
        last_time, last_utility, _, _ = history[-1]
        remaining = min(
            max(0.0, 1.0 - last_time),
            self.config.FORECAST_MAX_LOOKAHEAD,
        )
        best_seen = max(item[1] for item in history)
        forecast = max(best_seen, last_utility + max(0.0, trend) * remaining)
        forecast = max(0.0, min(self._max_utility, forecast))

        time_span = max(0.0, history[-1][0] - history[0][0])
        diversity = len(self._unique_opponent_offers) / len(history)
        sample_confidence = min(1.0, len(history) / 10.0)
        confidence = (
            0.45 * sample_confidence
            + 0.30 * min(1.0, time_span / 0.5)
            + 0.25 * diversity
        )
        return forecast, max(0.0, min(1.0, confidence))

    def _responsive_concession_adjustment(self, relative_time: float) -> float:
        """Raise or lower aspiration in response to observed opponent behavior."""
        if (
            relative_time < self.config.RESPONSIVE_START_THRESHOLD
            or len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY
        ):
            return 0.0

        _, confidence = self._opponent_deadline_forecast()
        if confidence < self.config.RESPONSIVE_CONFIDENCE_THRESHOLD:
            return 0.0

        concession = self._opponent_concession_rate()
        normalized_concession = concession / self._max_utility if self._max_utility else 0.0
        hardliner = self._hardliner_score()
        urgency = max(
            0.0,
            (relative_time - self.config.RESPONSIVE_START_THRESHOLD)
            / (1.0 - self.config.RESPONSIVE_START_THRESHOLD),
        )

        # Improving opponent offers justify patience. Hardliners justify a
        # measured concession only when enough time has elapsed.
        hold = (
            min(1.0, max(0.0, normalized_concession) / 0.12)
            * self.config.RESPONSIVE_MAX_HOLD
            * confidence
        )
        concede = (
            hardliner
            * urgency
            * self.config.RESPONSIVE_MAX_CONCESSION
            * confidence
        )
        return self._max_utility * (hold - concede)
    
    def _hardliner_score(self) -> float:
        """Estimate how much opponent is a hardliner."""
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0
        
        concession = self._opponent_concession_rate()
        recent_utility = sum(
            item[1] for item in self._opponent_offer_history
        ) / len(self._opponent_offer_history)
        
        low_for_us = 1.0 - min(1.0, recent_utility / self._max_utility)
        no_concession = (
            1.0 if concession <= self.config.CONCESSION_DETECTION_THRESHOLD * self._max_utility
            else 0.0
        )
        return max(0.0, min(1.0, 0.65 * low_for_us + 0.35 * no_concession))

    def _threshold_acceptor_score(self) -> float:
        """Estimate whether opponent behaves like a fixed-threshold acceptor."""
        history = list(self._opponent_offer_history)
        if len(history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0

        self_values = [item[2] for item in history]
        unique_ratio = len(self._unique_opponent_offers) / len(history)
        concession = max(0.0, self._opponent_concession_rate() / self._max_utility)
        high_self_value = sum(1 for value in self_values if value >= 0.72) / len(self_values)
        low_variation = 1.0 - min(1.0, (max(self_values) - min(self_values)) / 0.45)
        repeated_or_scripted = 1.0 - min(1.0, unique_ratio)

        score = (
            0.42 * high_self_value
            + 0.28 * low_variation
            + 0.20 * repeated_or_scripted
            + 0.10 * (1.0 - min(1.0, concession / 0.15))
        )
        return max(0.0, min(1.0, score))

    def _conceder_score(self) -> float:
        """Estimate whether opponent offers are improving for us over time."""
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0

        concession = max(0.0, self._opponent_concession_rate() / self._max_utility)
        forecast, confidence = self._opponent_deadline_forecast()
        best_seen = max(item[1] for item in self._opponent_offer_history)
        forecast_gain = max(0.0, forecast - best_seen) / self._max_utility
        recent = self._recent_opponent_value_for_us() / self._max_utility

        score = (
            0.55 * min(1.0, concession / 0.18)
            + 0.25 * min(1.0, forecast_gain / 0.10) * confidence
            + 0.20 * min(1.0, recent)
        )
        return max(0.0, min(1.0, score))

    def _mirror_strategic_score(self) -> float:
        """Estimate whether opponent appears strategic and similar to us."""
        history = list(self._opponent_offer_history)
        if len(history) < self.config.MIN_CONCESSION_HISTORY:
            return 0.0

        unique_ratio = len(self._unique_opponent_offers) / len(history)
        mean_for_us = sum(item[1] for item in history) / len(history)
        middle_for_us = 1.0 - min(1.0, abs((mean_for_us / self._max_utility) - 0.55) / 0.55)
        model_confidence = self._opponent_model_confidence()
        not_hardliner = 1.0 - self._hardliner_score()
        not_threshold = 1.0 - self._threshold_acceptor_score()

        score = (
            0.30 * min(1.0, unique_ratio)
            + 0.25 * middle_for_us
            + 0.20 * model_confidence
            + 0.15 * not_hardliner
            + 0.10 * not_threshold
        )
        return max(0.0, min(1.0, score))

    def _opponent_profile(self) -> dict[str, float]:
        """Return passive behavior scores without changing strategy."""
        return {
            "threshold_acceptor_like": self._threshold_acceptor_score(),
            "hardliner_like": self._hardliner_score(),
            "mirror_strategic_like": self._mirror_strategic_score(),
            "conceder_like": self._conceder_score(),
        }
    
    def _similarity_to_opponent_offers(self, outcome: Outcome) -> float:
        """Calculate similarity to opponent's past offers."""
        if not self._opponent_offer_history:
            return self.config.DEFAULT_SIMILARITY
        
        best_similarity = 0.0
        for _, _, _, previous in self._opponent_offer_history:
            matches = sum(a == b for a, b in zip(outcome, previous))
            similarity = matches / len(outcome) if outcome else 0.0
            best_similarity = max(best_similarity, similarity)
        return best_similarity
    
    def _deadline_score(self, relative_time: float, opponent_score: float) -> float:
        """Calculate urgency-based deadline score."""
        if relative_time < self.config.MID_PHASE_THRESHOLD:
            return 0.0
        urgency = (relative_time - self.config.MID_PHASE_THRESHOLD) / (1.0 - self.config.MID_PHASE_THRESHOLD)
        return urgency * opponent_score
    
    def _masked_window(
        self,
        ranked: list[tuple[float, float, Outcome]],
        relative_time: float,
    ) -> list[tuple[float, float, Outcome]]:
        """Select window of similar-utility offers to obscure preferences."""
        if not ranked:
            return []
        
        top_utility = ranked[0][1]
        slack = (
            self.config.EARLY_MASKING_SLACK
            if relative_time < self.config.MASKING_PHASE_THRESHOLD
            else self.config.LATE_MASKING_SLACK
        )
        
        near = [
            item
            for item in ranked
            if top_utility - item[1] <= self._max_utility * slack
        ]
        
        if len(near) >= self.config.MIN_MASKED_WINDOW_SIZE:
            near.sort(
                key=lambda item: (self._masking_score(item[2]), item[0]),
                reverse=True,
            )
            return near[:min(self.config.MAX_MASKED_WINDOW_SIZE, len(near))]
        
        return ranked[:max(1, min(9, len(ranked)))]
    
    def _masking_score(self, outcome: Outcome) -> float:
        """Score how well outcome masks true preferences."""
        if self._last_offer is None:
            return 1.0
        
        issue_novelty = sum(a != b for a, b in zip(outcome, self._last_offer))
        issue_novelty = issue_novelty / len(outcome) if outcome else 0.0
        
        exact_repeat_penalty = 1.0 if outcome in self._recent_offers else 0.0
        value_reuse_penalty = self._recent_value_reuse(outcome)
        
        return (
            0.55 * issue_novelty
            + 0.30 * (1.0 - value_reuse_penalty)
            + 0.15 * (1.0 - exact_repeat_penalty)
        )
    
    def _estimate_issue_costs(self) -> list[float]:
        """Estimate how important each issue is to us."""
        if not self._outcomes or not self._issue_values:
            return []
        
        costs = []
        for issue_index in range(len(self._issue_values)):
            best_by_value = {}
            for utility, outcome in self._outcomes:
                if issue_index >= len(outcome):
                    continue
                value = outcome[issue_index]
                best_by_value[value] = max(utility, best_by_value.get(value, 0.0))
            
            if len(best_by_value) <= 1:
                costs.append(0.0)
                continue
            
            utility_range = max(best_by_value.values()) - min(best_by_value.values())
            costs.append(
                utility_range / self._max_utility if self._max_utility else 0.0
            )
        
        return costs
    
    def _decoy_signal_score(self, outcome: Outcome, relative_time: float) -> float:
        """Score how well outcome provides decoy signals."""
        if (
            self._last_offer is None
            or not self._issue_costs
            or outcome is None
            or relative_time > self.config.DECOY_SIGNAL_PHASE_THRESHOLD
            or len(outcome) < 2
        ):
            return 0.0
        
        indexed_costs = list(enumerate(self._issue_costs[:len(outcome)]))
        if not indexed_costs:
            return 0.0
        
        cheapest_count = max(1, int(len(indexed_costs) * self.config.CHEAP_ISSUE_FRACTION))
        expensive_count = max(1, int(len(indexed_costs) * self.config.EXPENSIVE_ISSUE_FRACTION))
        
        cheapest = {
            index
            for index, _ in sorted(indexed_costs, key=lambda item: item[1])[:cheapest_count]
        }
        expensive = {
            index
            for index, _ in sorted(indexed_costs, key=lambda item: item[1], reverse=True)[:expensive_count]
        }
        
        cheap_stability = sum(
            1 for index in cheapest if outcome[index] == self._last_offer[index]
        ) / len(cheapest)
        
        expensive_variety = sum(
            1 for index in expensive if outcome[index] != self._last_offer[index]
        ) / len(expensive)
        
        return (
            self.config.CHEAP_STABILITY_WEIGHT * cheap_stability
            + self.config.EXPENSIVE_VARIETY_WEIGHT * expensive_variety
        )
    
    def _recent_value_reuse(self, outcome: Outcome) -> float:
        """Calculate how many values are reused from recent offers."""
        if not self._recent_offers:
            return 0.0
        
        reused = 0
        total = 0
        for previous in self._recent_offers:
            for current_value, previous_value in zip(outcome, previous):
                total += 1
                if current_value == previous_value:
                    reused += 1
        
        return reused / total if total else 0.0
    
    def _should_rescue_extreme_conflict(self, relative_time: float) -> bool:
        """Check if we should switch to conflict resolution mode."""
        if not self._is_tiny_extreme_domain():
            return False

        if self._hardliner_advantage_guard(relative_time):
            return False
        
        if relative_time > self.config.RESCUE_FINAL_THRESHOLD:
            return True
        
        return (
            relative_time > self.config.STALL_URGENCY_THRESHOLD
            and self._negotiation_is_stalled()
        )

    def _hardliner_advantage_guard(self, relative_time: float) -> bool:
        """Avoid low compromise rescue in clear hardliner/extreme domains."""
        if relative_time < self.config.STALL_URGENCY_THRESHOLD:
            return False
        if not self._is_tiny_extreme_domain():
            return False
        if len(self._opponent_offer_history) < self.config.MIN_CONCESSION_HISTORY:
            return False
        return (
            self._hardliner_score()
            >= self.config.HARDLINER_ADVANTAGE_GUARD_THRESHOLD
            and self._mirror_strategic_score()
            <= self.config.HARDLINER_ADVANTAGE_MIRROR_MAX
            and self._conceder_score()
            <= self.config.HARDLINER_ADVANTAGE_CONCEDER_MAX
        )

    def _emergency_floor_utility(self, relative_time: float) -> float:
        """Return the protected late utility floor for the current profile.

        Blends a fixed-ratio floor with a percentile band that loosens as the
        deadline approaches: p20 → p35 → p50.  Hardliner domains keep a tighter
        floor (p20) to avoid giving away too much when we still hold leverage.
        """
        if self._hardliner_advantage_guard(relative_time):
            return self._adaptive_utility_floor(
                self.config.HARDLINER_ADVANTAGE_FLOOR_RATIO, "p20", relative_time=relative_time
            )
        if relative_time >= 0.98:
            pkey = "p50"
        elif relative_time >= 0.95:
            pkey = "p35"
        else:
            pkey = "p20"
        return self._adaptive_utility_floor(self.config.EMERGENCY_UTILITY_RATIO, pkey, relative_time=relative_time)
    
    def _is_tiny_extreme_domain(self) -> bool:
        """Check if this is an extremely constrained negotiation."""
        if len(self._outcomes) <= 3:
            return True
        
        high_utility = [
            u for u, _ in self._outcomes
            if u >= self._max_utility * self.config.EXTREME_CONFLICT_THRESHOLD
        ]
        return len(high_utility) <= max(
            2, int(self.config.EXTREME_CONFLICT_MIN_RATIO * len(self._outcomes))
        )
    
    def _extreme_conflict_offer(self) -> Optional[Outcome]:
        """Find compromise offer for extreme conflict situations."""
        if not self._outcomes:
            return None
        
        best_score = float("-inf")
        best_outcome = self._outcomes[-1][1]
        
        for utility, outcome in self._outcomes:
            own_score = utility / self._max_utility if self._max_utility else utility
            opponent_score = self._estimated_opponent_utility(outcome)
            acceptance_score = self._acceptance_likelihood(outcome, opponent_score, 1.0)
            balance = 1.0 - abs(own_score - opponent_score)
            mutual_floor = min(own_score, opponent_score)
            
            score = (
                0.45 * mutual_floor
                + 0.35 * balance
                + 0.20 * acceptance_score
            )
            
            if score > best_score:
                best_score = score
                best_outcome = outcome
        
        return best_outcome
    
    def _negotiation_is_stalled(self) -> bool:
        """Check if negotiation progress has stalled."""
        enough_interaction = (
            self._own_offers >= self.config.STALL_DETECTION_MIN_INTERACTION
            or self._opponent_offers >= self.config.STALL_DETECTION_MIN_INTERACTION
        )
        if not enough_interaction:
            return False
        
        concession = self._opponent_concession_rate()
        hardliner = self._hardliner_score()
        low_recent = self._recent_opponent_value_for_us() < (
            self._max_utility * self.config.LOW_VALUE_THRESHOLD
        )
        no_progress = concession <= (
            self._max_utility * self.config.CONCESSION_DETECTION_THRESHOLD
        )
        
        return low_recent or hardliner > 0.50 or no_progress
    
    def _recent_opponent_value_for_us(self) -> float:
        """Calculate average utility of recent opponent offers."""
        if not self._opponent_offer_history:
            return 0.0
        
        recent = list(self._opponent_offer_history)[-4:]
        return sum(item[1] for item in recent) / len(recent)
    
    def _outcome_sort_key(self, outcome: Optional[Outcome]) -> tuple:
        """Generate consistent sort key for outcomes (handles dict/tuple/list)."""
        if outcome is None:
            return ()
        if isinstance(outcome, dict):
            return tuple(
                f"{key!r}:{value!r}"
                for key, value in sorted(outcome.items())
            )
        return tuple(repr(value) for value in outcome)
