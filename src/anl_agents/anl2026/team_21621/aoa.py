import math
import random
from collections import defaultdict

from negmas.gb import GSmithFrequencyModel, GNashFrequencyModel, GHardHeadedFrequencyModel, GScalableBayesianModel, \
    GAgentLGModel
from negmas.gb.negotiators.cab import MAPNegotiator
from negmas.sao.components.acceptance import ACNext, ACTime, AnyAcceptancePolicy
from negmas.sao.components.offering import MiCROOfferingPolicy


class AOA007(MAPNegotiator):
    def __init__(self, *args, **kwargs):
        offering = MiCROOfferingPolicy()

        logical_acceptance = ACNext(offering)
        panic_acceptance = ACTime(tau=0.98)

        combined_acceptance = AnyAcceptancePolicy(
            strategies=[logical_acceptance, panic_acceptance]
        )

        kwargs |= dict(
            acceptance=combined_acceptance,
            offering=offering,
            models=[
                ExponentialDecayFrequencyModel(growth_factor=5.0),
                TimeScaledHardHeadedModel(max_learning_coef=0.9),
                AdaptiveScalableBayesianModel(),
                RuthlessAgentLGModel(),
                GSmithFrequencyModel(),
                GNashFrequencyModel()
            ],
            model_names=[
                "Freq_Late",
                "Stubborn_Late",
                "Bayes_Adaptive",
                "AgentLG_Ruthless",
                "Smith_Standard",
                "Nash_Baseline"
            ],
            acceptance_first=True,
        )

        super().__init__(*args, **kwargs)


class ExponentialDecayFrequencyModel(GSmithFrequencyModel):
    """
    Improves upon standard frequency models by giving exponentially more
    weight to offers made closer to the deadline.
    """

    def __init__(self, growth_factor=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.growth_factor = growth_factor
        self._weighted_issue_counts = {}

    def on_partner_proposal(self, state, partner_id, offer):
        # We MUST inform the base class so NegMAS internal flags (like is_ready) update properly
        if hasattr(super(), 'on_partner_proposal'):
            super().on_partner_proposal(state, partner_id, offer)

        if offer is None:
            return

        time = state.relative_time
        weight = math.exp(self.growth_factor * time)

        for issue_index, issue_value in enumerate(offer):
            if issue_index not in self._weighted_issue_counts:
                self._weighted_issue_counts[issue_index] = {}

            if issue_value not in self._weighted_issue_counts[issue_index]:
                self._weighted_issue_counts[issue_index][issue_value] = 0.0

            self._weighted_issue_counts[issue_index][issue_value] += weight

    def issue_value_utility(self, issue, value):
        """
        Safely calculates utility regardless of whether NegMAS passes the issue
        as an integer index, a string name, or the Issue object itself.
        """
        # Safely resolve the issue index
        if isinstance(issue, int):
            issue_idx = issue
        elif isinstance(issue, str):
            issue_idx = self.negotiator.nmi.issue_names.index(issue)
        else:
            issue_idx = self.negotiator.nmi.issues.index(issue)

        if issue_idx not in self._weighted_issue_counts:
            return 0.5  # Return default uncertainty if unseen

        value_counts = self._weighted_issue_counts[issue_idx]

        if value not in value_counts:
            return 0.0

        max_weight = max(value_counts.values())
        if max_weight == 0:
            return 0.0

        # Normalize the utility based on the most heavily weighted value
        return value_counts[value] / max_weight

    def __call__(self, offer) -> float:
        """
        Directly override the main evaluation call to ensure NegMAS uses
        our custom logic instead of falling back to the base class math.
        """
        if not offer:
            return 0.5

        utility = 0.0
        for issue_index, issue_value in enumerate(offer):
            utility += self.issue_value_utility(issue_index, issue_value)

        # Average the utilities across all issues
        return utility / len(offer) if len(offer) > 0 else 0.5


class TimeScaledHardHeadedModel(GHardHeadedFrequencyModel):
    """
    Extends HardHeaded model.
    Ignores stubbornness early in the negotiation (assuming it's a decoy),
    but heavily rewards stubbornness near the deadline.
    """

    def __init__(self, max_learning_coef=0.8, *args, **kwargs):
        # Start with a very low base learning coefficient
        kwargs['learning_coef'] = 0.05
        super().__init__(*args, **kwargs)
        self.base_coef = 0.05
        self.max_learning_coef = max_learning_coef

    def on_partner_proposal(self, state, partner_id, offer):
        # Dynamically scale the learning coefficient based on relative time
        time_factor = state.relative_time ** 2  # Exponential growth
        current_coef = self.base_coef + (self.max_learning_coef - self.base_coef) * time_factor

        # Override the base class coefficient just before it processes the offer
        self.learning_coef = current_coef

        # Let the base GHardHeaded logic do the heavy lifting
        if hasattr(super(), 'on_partner_proposal'):
            super().on_partner_proposal(state, partner_id, offer)


class AdaptiveScalableBayesianModel(GScalableBayesianModel):
    """
    Extends Scalable Bayesian.
    Adapts the learning rate based on time, cutting through early noise
    and aggressively locking into the opponent's late-game compromises.
    """

    def __init__(self, max_learning_rate=0.5, *args, **kwargs):
        # Start with a conservative learning rate
        kwargs['learning_rate'] = 0.05
        super().__init__(*args, **kwargs)
        self.base_lr = 0.05
        self.max_learning_rate = max_learning_rate

    def on_partner_proposal(self, state, partner_id, offer):
        # As time runs out, the opponent's offers become more rational.
        # We increase the learning rate to trust these later offers more.
        if state.relative_time > 0.8:
            self.learning_rate = self.max_learning_rate
        else:
            self.learning_rate = self.base_lr + (state.relative_time * 0.1)

        if hasattr(super(), 'on_partner_proposal'):
            super().on_partner_proposal(state, partner_id, offer)


class RuthlessAgentLGModel(GAgentLGModel):
    """
    Extends AgentLG.
    Instead of just rewarding unchanged issues, this model actively punishes
    (drastically reduces weight of) issues that bounce around too much.
    """

    def update(self, state, offer, partner_id):
        # First, let the base AgentLG logic run (it boosts unchanged weights)
        if hasattr(super(), 'update'):
            super().update(state, offer, partner_id)

        if offer is None or self._n_issues == 0:
            return

        # Now apply our ruthless punishment to changed issues
        if self._last_bid is not None:
            for i in range(self._n_issues):
                if offer[i] != self._last_bid[i]:
                    # The issue changed! Punish its weight heavily
                    self._issue_weights[i] *= 0.6  # 40% penalty for moving

            # Re-normalize weights after our custom penalty
            total_weight = sum(self._issue_weights.values())
            if total_weight > 0:
                for i in self._issue_weights:
                    self._issue_weights[i] /= total_weight


class AOA008(AOA007):
    """
    Inherits AOA007's full negotiation logic and adds iso-utility randomization
    to conceal preferences. Instead of always offering MiCRO's deterministic pick,
    it swaps it with a random outcome of similar self-utility — confusing frequency-based
    opponent models at near-zero cost to advantage.
    """

    def __init__(self, *args, epsilon=0.05, min_bucket_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon
        self._min_bucket_size = min_bucket_size
        self._utility_buckets: dict[int, list] = {}

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        if self.ufun is None or self.nmi is None:
            return
        self._build_utility_buckets()

    def _build_utility_buckets(self):
        """Pre-compute buckets of outcomes grouped by quantized utility."""
        buckets = defaultdict(list)
        reserved = float(self.ufun.reserved_value)

        for outcome in self.nmi.outcome_space.enumerate_or_sample():
            u = float(self.ufun(outcome))
            if u <= reserved:
                continue
            # Quantize utility into bins of width epsilon
            bucket_key = round(u / self._epsilon)
            buckets[bucket_key].append(outcome)

        # Merge small buckets with nearest neighbors
        self._utility_buckets = self._merge_small_buckets(dict(buckets))

    def _merge_small_buckets(self, buckets: dict[int, list]) -> dict[int, list]:
        """Merge buckets with fewer than min_bucket_size outcomes into neighbors."""
        if not buckets:
            return buckets

        sorted_keys = sorted(buckets.keys())

        for key in sorted_keys:
            if len(buckets.get(key, [])) < self._min_bucket_size:
                # Find the nearest neighbor bucket to merge into
                best_neighbor = None
                best_dist = float('inf')
                for other_key in sorted_keys:
                    if other_key == key:
                        continue
                    dist = abs(other_key - key)
                    if dist < best_dist and len(buckets.get(other_key, [])) > 0:
                        best_dist = dist
                        best_neighbor = other_key
                if best_neighbor is not None:
                    buckets[best_neighbor].extend(buckets[key])
                    buckets[key] = []

        # Remove empty buckets
        return {k: v for k, v in buckets.items() if v}

    def generate_proposal(self, state, dest=None):
        """Intercept MiCRO's offer and swap with a random iso-utility alternative."""
        offer = super().generate_proposal(state, dest=dest)
        if offer is None or self.ufun is None:
            return offer

        target_utility = float(self.ufun(offer))
        bucket_key = round(target_utility / self._epsilon)

        # Look up the bucket; fall back to original offer if no bucket found
        bucket = self._utility_buckets.get(bucket_key)
        if not bucket:
            return offer

        return random.choice(bucket)
