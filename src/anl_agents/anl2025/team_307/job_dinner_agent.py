"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Adaptive Strategy Team
*Authors* Improved UnifiedNegotiator focusing on utility maximization

This agent combines the best of simple effectiveness with strategic concession,
prioritizing actual agreements over perfect pattern matching.
"""

import itertools
from typing import Tuple
from negmas.outcomes import Outcome
from negmas import ResponseType
from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_agreement_at_index,
    get_current_negotiation_index,
    get_outcome_space_from_index,
    find_best_bid_in_outcomespace,
)

__all__ = ["ImprovedUnifiedNegotiator"]


class ImprovedUnifiedNegotiator(ANL2025Negotiator):
    """
    An improved negotiation agent that prioritizes getting valuable agreements
    through adaptive concession strategies rather than rigid pattern matching.

    Key improvements:
    1. Agreement-focused strategy (recognizes that deals > no deals)
    2. Simplified but effective concession patterns
    3. Dynamic adaptation based on negotiation progress
    4. Balanced approach between utility maximization and deal completion
    """

    def init(self):
        """Initialize with streamlined, effectiveness-focused approach."""
        # Core negotiation state
        self.current_neg_index = -1
        self.agreements = []
        self.target_bid = None

        # Negotiation tracking
        self.id_dict = {}
        set_id_dict(self)
        self.num_negotiations = len(self.id_dict)

        # Strategy parameters - tuned for effectiveness
        self.concession_speed = 0.3  # Moderate concession rate
        self.min_acceptable_ratio = 0.4  # Minimum utility ratio to accept
        self.time_pressure_start = 0.6  # When to start time-based concessions
        self.aggressive_threshold = 0.8  # When to become more aggressive

        # Opponent and context modeling
        self.opponent_utilities = []
        self.round_number = 0
        self.current_side_ufun = None
        self.best_known_utility = 0.0

        # Performance tracking
        self.successful_agreements = 0
        self.total_utility_achieved = 0.0

        # Pre-compute strategy if center agent
        if not is_edge_agent(self):
            self._initialize_center_strategy()

    def _initialize_center_strategy(self):
        """Initialize strategy for center agents with focus on utility maximization."""
        try:
            # Quick analysis of utility landscape
            self._analyze_utility_potential()
        except:
            # Fallback to simple strategy
            self.best_known_utility = 1.0

    def _analyze_utility_potential(self):
        """Fast analysis of utility potential without over-complication."""
        try:
            # Sample some combinations to understand utility range
            sample_outcomes = []
            for i in range(min(3, self.num_negotiations)):
                outcomes = get_outcome_space_from_index(self, i)
                if outcomes:
                    # Take best, middle, and a random outcome
                    sample_outcomes.append(
                        outcomes[:3] if len(outcomes) >= 3 else outcomes
                    )
                else:
                    sample_outcomes.append([None])

            if sample_outcomes:
                max_utility = 0.0
                for combo in itertools.product(*sample_outcomes):
                    try:
                        utility = self.ufun(combo)
                        max_utility = max(max_utility, utility)
                    except:
                        continue

                self.best_known_utility = max_utility
            else:
                self.best_known_utility = 1.0

        except:
            self.best_known_utility = 1.0

    def propose(self, negotiator_id: str, state: SAOState, dest: str = None) -> Outcome:
        """Generate proposals using utility-focused strategy with strategic concession."""
        # Handle negotiation transitions
        if did_negotiation_end(self):
            self._handle_negotiation_transition(negotiator_id)

        # Get time and context
        relative_time = self._get_relative_time(negotiator_id, state)

        # Generate proposal based on role and strategy
        if is_edge_agent(self):
            return self._propose_as_edge(negotiator_id, relative_time)
        else:
            return self._propose_as_center(negotiator_id, relative_time)

    def _propose_as_edge(self, negotiator_id: str, relative_time: float) -> Outcome:
        """Edge agent proposal strategy - aim high but concede strategically."""
        try:
            # Get best possible outcome
            _, best_outcome = self.ufun.extreme_outcomes()
            best_utility = self.ufun(best_outcome)

            # Early phase: aim for best
            if relative_time < 0.4:
                return best_outcome

            # Time-dependent concession strategy that learns how to make strategic concessions
            # Middle phase: slight concession
            elif relative_time < 0.7:
                if self.opponent_utilities:
                    # Find outcomes that give decent utility to both parties
                    acceptable_outcomes = [
                        outcome
                        for outcome, my_util, opp_util in self.opponent_utilities
                        if my_util >= 0.8 * best_utility
                    ]
                    if acceptable_outcomes:
                        return max(acceptable_outcomes, key=lambda x: self.ufun(x))

                return best_outcome

            # Late phase: more concession but stay above minimum
            else:
                min_acceptable = max(self.ufun.reserved_value, 0.5 * best_utility)

                if self.opponent_utilities:
                    viable_outcomes = [
                        outcome
                        for outcome, my_util, opp_util in self.opponent_utilities
                        if my_util >= min_acceptable
                    ]
                    if viable_outcomes:
                        # Choose outcome that maximizes joint utility among viable options
                        return max(
                            viable_outcomes,
                            key=lambda x: self.ufun(x)
                            + (
                                self.current_side_ufun(x)
                                if self.current_side_ufun
                                else 0
                            ),
                        )

                return best_outcome

        except:
            # Fallback
            return self.target_bid

    def _propose_as_center(self, negotiator_id: str, relative_time: float) -> Outcome:
        """Center agent proposal strategy - balance utility with agreement probability."""
        try:
            current_neg_idx = get_current_negotiation_index(self)

            # Get current best option
            best_contextual = self._find_best_contextual_outcome()

            if best_contextual is None:
                return None

            # Extract the bid for current negotiation
            if (
                isinstance(best_contextual, (list, tuple))
                and len(best_contextual) > current_neg_idx
            ):
                current_bid = best_contextual[current_neg_idx]
            else:
                current_bid = best_contextual

            # Strategic concessions in installments are more effective than single large concessions
            # Early phase: try for optimal outcomes
            if relative_time < 0.3:
                return current_bid

            # Middle phase: start strategic concession
            elif relative_time < self.time_pressure_start:
                return self._apply_moderate_concession(current_bid, relative_time)

            # Late phase: more aggressive concession to secure agreement
            else:
                return self._apply_aggressive_concession(current_bid, relative_time)

        except:
            return self.target_bid

    def _apply_moderate_concession(
        self, base_bid: Outcome, relative_time: float
    ) -> Outcome:
        """Apply moderate concession to increase agreement probability."""
        if not self.opponent_utilities:
            return base_bid

        try:
            # Find alternatives that give reasonable utility to opponent
            current_utility = self.ufun(self._construct_full_outcome(base_bid))
            min_utility = current_utility * 0.9  # Small concession

            viable_alternatives = [
                outcome
                for outcome, my_util, opp_util in self.opponent_utilities
                if my_util >= min_utility and opp_util > 0.3
            ]

            if viable_alternatives:
                # Choose one that maximizes opponent utility among viable options
                return max(
                    viable_alternatives,
                    key=lambda x: self.current_side_ufun(x)
                    if self.current_side_ufun
                    else 0,
                )

        except:
            pass

        return base_bid

    def _apply_aggressive_concession(
        self, base_bid: Outcome, relative_time: float
    ) -> Outcome:
        """Apply aggressive concession in late phase to secure agreements."""
        if not self.opponent_utilities:
            return base_bid

        try:
            # Time-dependent concession strategy maximizing expected utility without negotiation break-off
            # Calculate how much we can concede based on time pressure
            concession_factor = (relative_time - self.time_pressure_start) / (
                1.0 - self.time_pressure_start
            )

            current_utility = self.ufun(self._construct_full_outcome(base_bid))
            min_acceptable = max(
                self.best_known_utility * self.min_acceptable_ratio,
                current_utility * (1.0 - self.concession_speed * concession_factor),
            )

            # Find outcomes that meet minimum utility but maximize opponent satisfaction
            acceptable_outcomes = [
                outcome
                for outcome, my_util, opp_util in self.opponent_utilities
                if my_util >= min_acceptable
            ]

            if acceptable_outcomes:
                # Prioritize opponent utility to increase acceptance probability
                return max(
                    acceptable_outcomes,
                    key=lambda x: (
                        self.current_side_ufun(x) if self.current_side_ufun else 0,
                        self.ufun(self._construct_full_outcome(x)),
                    ),
                )

        except:
            pass

        return base_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str = None
    ) -> ResponseType:
        """Strategic acceptance decisions prioritizing utility and deal completion."""
        if did_negotiation_end(self):
            self._handle_negotiation_transition(negotiator_id)

        if state.current_offer is None:
            return ResponseType.REJECT_OFFER

        # Track opponent behavior
        self._update_opponent_model(state.current_offer)

        # Make acceptance decision
        return self._make_strategic_acceptance_decision(negotiator_id, state)

    def _make_strategic_acceptance_decision(
        self, negotiator_id: str, state: SAOState
    ) -> ResponseType:
        """Make acceptance decisions balancing utility with agreement probability."""
        offer = state.current_offer
        relative_time = self._get_relative_time(negotiator_id, state)

        if is_edge_agent(self):
            return self._edge_acceptance_strategy(offer, relative_time)
        else:
            return self._center_acceptance_strategy(offer, relative_time)

    def _edge_acceptance_strategy(
        self, offer: Outcome, relative_time: float
    ) -> ResponseType:
        """Edge agent acceptance with graduated thresholds."""
        try:
            offer_utility = self.ufun(offer)
            _, best_outcome = self.ufun.extreme_outcomes()
            best_utility = self.ufun(best_outcome)

            # Strategic acceptance thresholds that decrease over time to ensure deal completion
            # Dynamic threshold based on time
            if relative_time < 0.3:
                threshold = 0.95  # Very selective early
            elif relative_time < 0.6:
                threshold = 0.85  # Moderately selective
            elif relative_time < 0.8:
                threshold = 0.7  # More accepting
            else:
                threshold = 0.5  # Very accepting late

            target_utility = threshold * best_utility

            if offer_utility >= target_utility:
                return ResponseType.ACCEPT_OFFER

            # Emergency acceptance to avoid no deal
            if relative_time > 0.9 and offer_utility > self.ufun.reserved_value:
                return ResponseType.ACCEPT_OFFER

        except:
            pass

        return ResponseType.REJECT_OFFER

    def _center_acceptance_strategy(
        self, offer: Outcome, relative_time: float
    ) -> ResponseType:
        """Center agent acceptance focusing on overall utility impact."""
        try:
            # Calculate utility impact of accepting this offer
            utility_with_offer = self._calculate_utility_with_offer(offer)
            utility_without_offer = self._calculate_utility_without_offer()

            # Improvement-based acceptance
            improvement = utility_with_offer - utility_without_offer

            # Time-dependent acceptance thresholds
            if relative_time < 0.4:
                # Early: only accept if significant improvement
                min_improvement = self.best_known_utility * 0.1
            elif relative_time < 0.7:
                # Middle: accept smaller improvements
                min_improvement = self.best_known_utility * 0.05
            else:
                # Late: accept any positive improvement
                min_improvement = 0.01

            if improvement >= min_improvement:
                return ResponseType.ACCEPT_OFFER

            # Special case: if we have few agreements, be more accepting
            agreement_count = sum(1 for a in self.agreements if a is not None)
            if agreement_count < self.num_negotiations // 2 and improvement > 0:
                return ResponseType.ACCEPT_OFFER

        except:
            pass

        return ResponseType.REJECT_OFFER

    def _handle_negotiation_transition(self, negotiator_id: str):
        """Handle transition to new negotiation round."""
        self.round_number = len(self.finished_negotiators)

        # Update agreements
        prev_index = self.current_neg_index - 1
        if prev_index >= 0:
            try:
                agreement = get_agreement_at_index(self, prev_index)
                while len(self.agreements) <= prev_index:
                    self.agreements.append(None)
                self.agreements[prev_index] = agreement

                if agreement is not None:
                    self.successful_agreements += 1

            except:
                pass

        # Update strategy
        self._update_strategy_for_new_round(negotiator_id)

    def _update_strategy_for_new_round(self, negotiator_id: str):
        """Update strategy when starting new negotiation."""
        # Get side utility function for opponent modeling
        if not is_edge_agent(self):
            try:
                _, context = self.negotiators[negotiator_id]
                self.current_side_ufun = context.get("ufun")
            except:
                self.current_side_ufun = None

        # Update target bid
        self._update_target_bid()

        # Reset opponent model for new negotiation
        self.opponent_utilities = []

    def _update_target_bid(self):
        """Update target bid for current context."""
        try:
            if is_edge_agent(self):
                _, self.target_bid = self.ufun.extreme_outcomes()
            else:
                self.target_bid = find_best_bid_in_outcomespace(self)
        except:
            self.target_bid = None

    def _update_opponent_model(self, offer: Outcome):
        """Update opponent model with new offer information."""
        if not self.current_side_ufun or not offer:
            return

        try:
            my_utility = (
                self.ufun(self._construct_full_outcome(offer))
                if not is_edge_agent(self)
                else self.ufun(offer)
            )
            opp_utility = self.current_side_ufun(offer)

            self.opponent_utilities.append((offer, my_utility, opp_utility))

            # Keep only recent offers to avoid memory bloat
            if len(self.opponent_utilities) > 50:
                self.opponent_utilities = self.opponent_utilities[-30:]

        except:
            pass

    def _construct_full_outcome(self, single_offer: Outcome) -> Tuple:
        """Construct full outcome tuple for center agent utility calculation."""
        try:
            current_neg_idx = get_current_negotiation_index(self)
            full_outcome = list(self.agreements)

            # Pad with None until current index
            while len(full_outcome) <= current_neg_idx:
                full_outcome.append(None)

            # Set current offer
            full_outcome[current_neg_idx] = single_offer

            # Pad to full length
            while len(full_outcome) < self.num_negotiations:
                full_outcome.append(None)

            return tuple(full_outcome)
        except:
            return tuple([single_offer] + [None] * (self.num_negotiations - 1))

    def _calculate_utility_with_offer(self, offer: Outcome) -> float:
        """Calculate utility if offer is accepted."""
        try:
            return self.ufun(self._construct_full_outcome(offer))
        except:
            return 0.0

    def _calculate_utility_without_offer(self) -> float:
        """Calculate utility if no agreement is reached."""
        try:
            return self.ufun(self._construct_full_outcome(None))
        except:
            return 0.0

    def _find_best_contextual_outcome(self) -> Outcome:
        """Find best outcome in current context."""
        try:
            if is_edge_agent(self):
                _, best_outcome = self.ufun.extreme_outcomes()
                return best_outcome
            else:
                return find_best_bid_in_outcomespace(self)
        except:
            return self.target_bid

    def _get_relative_time(self, negotiator_id: str, state: SAOState) -> float:
        """Get negotiation progress as relative time."""
        try:
            nmi = self.negotiators[negotiator_id].negotiator.nmi
            if state.step == 0:
                return 0.0
            elif nmi.n_steps and state.step >= nmi.n_steps - 1:
                return 1.0
            return state.relative_time if state.relative_time is not None else 0.0
        except:
            return 0.0


# Test runner
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    run_a_tournament(ImprovedUnifiedNegotiator, small=True)
