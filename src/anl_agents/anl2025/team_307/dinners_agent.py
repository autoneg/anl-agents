"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

import itertools

from anl2025.negotiator import ANL2025Negotiator
from negmas import (
    ResponseType,
)

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_agreement_at_index,
)

__all__ = ["DinnersNegotiator"]

# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
# from helpers.helperfunctions import set_id_dict, ...


class DinnersNegotiator(ANL2025Negotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    This example agent aims for the absolute best bid available. As a center agent, it adapts its strategy after each negotiation, by aiming for the best bid GIVEN the previous outcomes.
    """

    """
       The most general way to implement an agent is to implement propose and respond.
       """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # print("init")

        # Initalize variables
        self.current_neg_index = -1
        self.agreements = []

        # Map negotiation index to negotiator id
        self.id_dict = {}
        set_id_dict(self)

        # For analyzing utility patterns
        self.best_pattern = None
        self.best_utility = float("-inf")
        self.num_negotiations = len(self.id_dict)

        # Precompute best utility combinations
        if not is_edge_agent(self):
            self._analyze_utility_patterns()

    def _get_possible_outcomes(self, neg_id):
        """Get all possible outcomes for a negotiation by id."""
        nmi = self.negotiators[neg_id].negotiator.nmi
        return list(nmi.outcome_space.enumerate_or_sample())

    def _get_progress(self, negotiator_id):
        """Get the current negotiation progress (0 to 1)."""
        nmi = self.negotiators[negotiator_id].negotiator.nmi
        return nmi.state.relative_time if nmi.state.relative_time is not None else 0

    def _analyze_utility_patterns(self):
        """
        Analyze utility patterns to find the best agreement pattern.
        """
        # Skip for edge agents
        if is_edge_agent(self):
            return

        # Get all possible outcomes for each negotiation
        outcome_spaces = []
        for i in range(self.num_negotiations):
            neg_id = self.id_dict.get(i)
            if neg_id:
                outcomes = self._get_possible_outcomes(neg_id)
                outcomes.append(None)  # Include possibility of no agreement
                outcome_spaces.append(outcomes)
            else:
                outcome_spaces.append([None])

        # For 3-negotiation scenarios, analyze patterns with different numbers of agreements
        if self.num_negotiations == 3:
            # Evaluate combinations grouping by number of agreements
            patterns_by_count = {0: [], 1: [], 2: [], 3: []}

            for combo in itertools.product(*outcome_spaces):
                # Count agreements
                agreement_count = sum(1 for outcome in combo if outcome is not None)

                # Calculate utility
                utility = self.ufun(combo)

                # Store by agreement count
                patterns_by_count[agreement_count].append((combo, utility))

            # Find best pattern for each agreement count
            best_by_count = {}
            for count, patterns in patterns_by_count.items():
                if patterns:
                    best_pattern = max(patterns, key=lambda x: x[1])
                    best_by_count[count] = best_pattern

            # Find overall best pattern
            if best_by_count:
                best_count, best_pattern = max(
                    best_by_count.items(), key=lambda x: x[1][1]
                )
                self.best_pattern = best_pattern[0]
                self.best_utility = best_pattern[1]
        else:
            # For scenarios with different number of negotiations
            # Simply find the best combination
            best_combo = None
            best_utility = float("-inf")

            for combo in itertools.product(*outcome_spaces):
                utility = self.ufun(combo)
                if utility > best_utility:
                    best_utility = utility
                    best_combo = combo

            self.best_pattern = best_combo
            self.best_utility = best_utility

    def _find_best_outcome(self, negotiator_id):
        """Find the best outcome for the current negotiation."""
        # For edge agents, find outcome with highest utility
        if is_edge_agent(self):
            best_outcome = None
            best_utility = float("-inf")

            for outcome in self._get_possible_outcomes(negotiator_id):
                utility = self.ufun(outcome)
                if utility > best_utility:
                    best_outcome = outcome
                    best_utility = utility

            return best_outcome

        # For center agents, follow the best pattern
        if self.best_pattern and self.current_neg_index < len(self.best_pattern):
            return self.best_pattern[self.current_neg_index]

        # If no pattern available, calculate best outcome in context
        context = list(self.agreements)
        context += [None] * (self.current_neg_index - len(context))

        best_outcome = None
        best_utility = float("-inf")

        # Try each possible outcome
        for outcome in self._get_possible_outcomes(negotiator_id):
            test_context = context.copy()
            test_context.append(outcome)

            # Pad with None
            while len(test_context) < self.num_negotiations:
                test_context.append(None)

            # Calculate utility
            utility = self.ufun(tuple(test_context))

            if utility > best_utility:
                best_outcome = outcome
                best_utility = utility

        # Try having no agreement
        test_context = context.copy()
        test_context.append(None)

        # Pad with None
        while len(test_context) < self.num_negotiations:
            test_context.append(None)

        # Calculate utility
        none_utility = self.ufun(tuple(test_context))

        if none_utility > best_utility:
            return None

        return best_outcome

    def propose(self, negotiator_id, state, dest=None):
        """Generate a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        if did_negotiation_end(self):
            self._update_agreements_if_needed()

        # For edge agents
        if is_edge_agent(self):
            return self._find_best_outcome(negotiator_id)

        # For center agents
        # Special case for 3-negotiation scenarios
        if self.num_negotiations == 3 and self.current_neg_index == 2:
            # Count agreements so far
            agreement_count = sum(1 for a in self.agreements if a is not None)
            if agreement_count >= 2:
                # Always end negotiation if we already have 2 agreements
                progress = self._get_progress(negotiator_id)
                if progress > 0.3:
                    return None

        # Find best outcome
        best_outcome = self._find_best_outcome(negotiator_id)

        # If best outcome is no agreement, end negotiation after early phase
        if best_outcome is None:
            progress = self._get_progress(negotiator_id)
            if progress > 0.3:
                return None

        return best_outcome

    def respond(self, negotiator_id, state, source=None):
        """Respond to a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        if did_negotiation_end(self):
            self._update_agreements_if_needed()

        # If no offer, reject
        if state.current_offer is None:
            return ResponseType.REJECT_OFFER

        # For edge agents
        if is_edge_agent(self):
            # Calculate utilities
            offer_utility = self.ufun(state.current_offer)
            best_outcome = self._find_best_outcome(negotiator_id)
            best_utility = self.ufun(best_outcome) if best_outcome else 0
            progress = self._get_progress(negotiator_id)

            # Accept if close to best or late in negotiation
            if offer_utility >= 0.95 * best_utility:
                return ResponseType.ACCEPT_OFFER

            if progress > 0.7 and offer_utility >= 0.8 * best_utility:
                return ResponseType.ACCEPT_OFFER

            if progress > 0.9 and offer_utility >= 0.7 * best_utility:
                return ResponseType.ACCEPT_OFFER
        else:
            # For center agents
            # Special case for 3-negotiation scenarios
            if self.num_negotiations == 3 and self.current_neg_index == 2:
                # Count agreements so far
                agreement_count = sum(1 for a in self.agreements if a is not None)
                if agreement_count >= 2:
                    # Always reject if we already have 2 agreements
                    return ResponseType.REJECT_OFFER

            # Check if offer matches best pattern
            if self.best_pattern and self.current_neg_index < len(self.best_pattern):
                if state.current_offer == self.best_pattern[self.current_neg_index]:
                    return ResponseType.ACCEPT_OFFER

            # Calculate utility of offer in context
            context = list(self.agreements)
            while len(context) < self.current_neg_index:
                context.append(None)

            test_context = context.copy()
            test_context.append(state.current_offer)

            # Pad with None
            while len(test_context) < self.num_negotiations:
                test_context.append(None)

            # Calculate utility of offer
            offer_utility = self.ufun(tuple(test_context))

            # Calculate utility with no agreement
            test_context = context.copy()
            test_context.append(None)

            # Pad with None
            while len(test_context) < self.num_negotiations:
                test_context.append(None)

            # Calculate utility
            none_utility = self.ufun(tuple(test_context))

            # If no agreement is better, reject
            if none_utility > offer_utility:
                return ResponseType.REJECT_OFFER

            # Check if this would be a good agreement
            progress = self._get_progress(negotiator_id)

            # Early phase: only accept if it matches best pattern
            if progress < 0.4:
                return ResponseType.REJECT_OFFER

            # Middle phase: accept if utility is good
            if progress < 0.7:
                # Calculate utility of best possible outcome
                best_outcome = self._find_best_outcome(negotiator_id)
                if best_outcome is not None:
                    test_context = context.copy()
                    test_context.append(best_outcome)

                    # Pad with None
                    while len(test_context) < self.num_negotiations:
                        test_context.append(None)

                    # Calculate utility
                    best_utility = self.ufun(tuple(test_context))

                    if offer_utility >= 0.9 * best_utility:
                        return ResponseType.ACCEPT_OFFER

            # Late phase: accept decent offers
            if progress > 0.7:
                if offer_utility > none_utility * 1.1:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _update_agreements_if_needed(self):
        """Update the agreements list if a negotiation has ended."""
        if did_negotiation_end(self):
            # Store the agreement from the just-ended negotiation
            prev_index = self.current_neg_index - 1
            if prev_index >= 0:
                agreement = get_agreement_at_index(self, prev_index)
                while len(self.agreements) <= prev_index:
                    self.agreements.append(None)
                self.agreements[prev_index] = agreement
                return True
        return False


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    # Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    # Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(DinnersNegotiator, small=True)
