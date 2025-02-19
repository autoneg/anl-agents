"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import math

from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["BidBot"]


class BidBot(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    min_partner_reserved_value = 1
    rational_outcomes = tuple()
    partner_reserved_value = 1
    opponent_opp_ufun_history = []
    window_size = 1

    ufun_in_opponent_offers = []
    opponent_ufun_in_opponent_offers = []

    previous_ufun = 0
    previous_opponent_ufun = 0

    my_current_target_utility = 1.0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)

        if self.nmi.outcome_space:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()
                # enumerates outcome space when finite, samples when infinite
                if self.ufun(_) > self.ufun.reserved_value
            ]
        else:
            # Handle the case when outcome space is empty gracefully
            self.rational_outcomes = []

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        # self.partner_reserved_value = self.ufun.reserved_value

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

        if len(self.opponent_opp_ufun_history) == 0:
            self.opponent_opp_ufun_history.append(1)
            self.opponent_opp_ufun_history.append(1)
        else:
            self.opponent_opp_ufun_history.append(
                self.opponent_ufun(offer)
            )  # history for opponet's op_ufun

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def interval_bidding_strategy(
        self, t, target_utility, interval=0.05, min_val=0.0, max_val=1
    ):
        valid_outcomes = list()
        distance = [abs(self.ufun(o) - target_utility) for o in self.rational_outcomes]
        for o in self.rational_outcomes:
            if abs(self.ufun(o) - target_utility) <= interval:
                valid_outcomes.append([o, self.opponent_ufun(o)])
        if len(valid_outcomes) == 0:  # if valid outcomes is empty
            distance = [
                abs(self.ufun(o) - target_utility) for o in self.rational_outcomes
            ]
            o = self.rational_outcomes[distance.index(min(distance))]
            return [o, self.opponent_ufun(o)]

        valid_outcomes = sorted(valid_outcomes, key=lambda x: x[1])
        index_ = round(((max_val - min_val) * t + min_val) * (len(valid_outcomes) - 1))
        return valid_outcomes[index_]

    def create_time_dependant_bidding(self, state, P1=1.0):
        t = state.relative_time  # Time as a fraction of the total negotiation time
        P0 = max(self.ufun(o) for o in self.rational_outcomes)  # maximum value
        P2 = self.ufun.reserved_value  # minimum value
        target_utility = min(
            ((1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2), 1
        )  # omit value bigger than 1
        target_utility = max(0, target_utility)  # omit negative value

        move = self.interval_bidding_strategy(t, target_utility)

        return move[0]

    def ac_next(self, opponent_offer, my_offer, a, b):
        return a * self.ufun(opponent_offer) + b >= self.ufun(my_offer)

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun
        a = 1
        b = 0
        offer = state.current_offer
        my_future_offer = self.bidding_strategy(state)

        return self.ac_next(offer, my_future_offer, a, b)

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        offer = state.current_offer

        if offer is not None:
            self.ufun_in_opponent_offers.append(self.ufun(offer))
            self.opponent_ufun_in_opponent_offers.append(self.opponent_ufun(offer))

        if len(self.opponent_ufun_in_opponent_offers) < 2:
            return self.create_time_dependant_bidding(state, 3)

        opponent_behavior = ""
        opponent_behavior_decision = 0

        temp = []
        for i in range(
            min(self.window_size, len(self.opponent_ufun_in_opponent_offers) - 1)
        ):
            temp.append(
                self.bidding_strategy_formula(
                    self.ufun_in_opponent_offers[-2 - i],
                    self.ufun_in_opponent_offers[-1 - i],
                    self.opponent_ufun_in_opponent_offers[-2 - i],
                    self.opponent_ufun_in_opponent_offers[-1 - i],
                    0,
                )
            )
        temp = temp[::-1]
        opponent_behavior_decision = sum(temp) / len(temp)

        if opponent_behavior_decision < 0:
            opponent_behavior = "Boulware"
        else:
            opponent_behavior = "Conceder"

        if opponent_behavior == "" or opponent_behavior == "Conceder":
            return self.create_time_dependant_bidding(
                state, 2 + opponent_behavior_decision
            )
        else:
            return self.create_bidding_tit_for_tat(state, state.relative_time)

    def bidding_strategy_formula(
        self, prev_ufun, ufun, prev_opponent_ufun, opponent_ufun, k=10
    ):
        return (k * (ufun - prev_ufun) - (opponent_ufun - prev_opponent_ufun)) / (k + 1)

    def gaussian_function(self, t, k):
        return (math.e ** (k * t) - 1) / (math.e**k - 1)

    def moving_gaussian_function(self, t, min_val, max_val, past_val, future_val):
        rng = max_val - min_val
        self.gauss_const = self.gauss_const + rng * (future_val - past_val)
        return self.gaussian_function(t, self.gauss_const)

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """

        assert self.ufun and self.opponent_ufun
        t = state.relative_time

        res_value_coef = self.gaussian_function(t, 5)

        opp_ufun = self.opponent_opp_ufun_history[-1]

        self.min_partner_reserved_value = min(self.min_partner_reserved_value, opp_ufun)
        self.partner_reserved_value = self.min_partner_reserved_value * res_value_coef

        rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]

        if len(rational_outcomes) > 0:
            self.rational_outcomes = rational_outcomes

    def create_bidding_tit_for_tat(self, state, t):
        maximum_offered_utility_by_opponent = max(self.opponent_ufun_in_opponent_offers)
        minimum_offered_utility_by_opponent = min(self.opponent_ufun_in_opponent_offers)
        min_utility_of_opponent_last_bids = min(self.ufun_in_opponent_offers[-5:])

        opponent_concession = (
            maximum_offered_utility_by_opponent - minimum_offered_utility_by_opponent
        )

        opponent_concede_factor = min(
            1,
            opponent_concession
            / (
                self.my_current_target_utility
                - min_utility_of_opponent_last_bids
                + 1e-12
            ),
        )

        my_concession = opponent_concede_factor * (1 - self.my_current_target_utility)
        my_current_target_utility = max(0.0, (1.0 - my_concession))
        my_current_target_utility = min(my_current_target_utility, 1.0)

        move = self.interval_bidding_strategy(t, my_current_target_utility)
        return move[0]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(BidBot, small=True)
