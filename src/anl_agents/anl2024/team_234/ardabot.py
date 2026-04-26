"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random

from anl.anl2024.negotiators.base import ANLNegotiator

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["Ardabot"]


class Ardabot(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()

    partner_reserved_value = 0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer
        current_utility = self.ufun(offer) if offer else 0
        time_elapsed = state.relative_time
        remaining_time = 1 - time_elapsed

        # Basic threshold based on reservation value and time dynamics
        dynamic_factor = 2 - 1.5 * time_elapsed
        dynamic_threshold = dynamic_factor * self.ufun.reserved_value

        # Accept if the current offer exceeds the dynamically adjusted threshold
        if current_utility > dynamic_threshold:
            return True

        # Strategy adjustment near the deadline
        if remaining_time < 0.2:  # More willing to accept as deadline approaches
            # Accept if the current offer is close to the maximum observed utility
            if current_utility >= 0.9 * self.ufun.reserved_value:
                return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        current_offer = state.current_offer
        time_elapsed = state.relative_time

        if (
            not self.rational_outcomes
        ):  # It should ensure there are outcomes to choose from
            return None

        if current_offer is None:
            return random.choice(self.rational_outcomes)

            # Bid evaluation with dynamic weights
        my_weight = (
            1 - time_elapsed
        )  # The weight of our own benefit decreasing over time
        their_weight = (
            time_elapsed  # The weight of our own opponent's decreasing over time
        )

        best_offer = None
        best_combined_utility = -float("inf")

        for outcome in self.rational_outcomes:
            my_utility = self.ufun(outcome)
            their_utility = self.opponent_ufun(outcome) if self.opponent_ufun else 0
            combined_utility = my_utility * my_weight + their_utility * their_weight

            # update best offer
            if combined_utility > best_combined_utility:
                best_combined_utility = combined_utility
                best_offer = outcome

        return best_offer if best_offer else random.choice(self.rational_outcomes)

    def update_partner_reserved_value(self, state: SAOState):
        """
        Updates the estimated reservation value of the opponent based on the last offer received.
        Uses the utility function to evaluate the opponent's last offer and updates the reservation value
        if the received utility is lower than the currently stored value, indicating a possible lower boundary.
        """
        assert self.ufun and self.opponent_ufun

        # Check the last offer received
        if state.current_offer is not None:
            current_opponent_utility = self.opponent_ufun(state.current_offer)

            # Update the partner's reservation value if the new utility is lower
            if current_opponent_utility < self.partner_reserved_value:
                self.partner_reserved_value = current_opponent_utility

        # Optionally update rational outcomes based on the new estimated reservation value
        self.rational_outcomes = [
            outcome
            for outcome in self.rational_outcomes
            if self.opponent_ufun(outcome) > self.partner_reserved_value
        ]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Ardabot, small=True)
