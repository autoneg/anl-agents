"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

from copy import deepcopy
import random

import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator

from negmas.outcomes import Outcome
from negmas.preferences import PresortingInverseUtilityFunction
from negmas.sao import ResponseType, SAOResponse, SAOState
from scipy.optimize import curve_fit

__all__ = ["Goldie"]


def aspiration_function(t, mx, rv, e):
    """A monotonically decreasing curve starting at mx (t=0) and ending at rv (t=1)"""
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class Goldie(ANLNegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize local variables
        self.worst_offer_utility: float = float("inf")
        self.sorter = None
        self._received, self._sent = set(), set()

        # Conceder
        self.concede_last_offer = None
        self.time_failed = False

        # Upper
        self.thresh = None
        self.last_offer = []
        self.upper = True

        # Modeling
        self.min_unique_utilities = 10
        self.e = 5.0
        self.stochasticity = 0.1
        self.opponent_times = []
        self.opponent_utilities = []
        self.past_opponent_rv = 0.0
        self.rational = []

    def on_preferences_changed(self, changes):
        self.private_info["opponent_ufun"] = deepcopy(self.opponent_ufun)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        # The main implementation of the MiCRO strategy
        assert self.ufun
        # initialize the sorter (This should better be done in on_negotiation_start() to allow for reuse but this is not needed in ANL)
        if self.sorter is None:
            # A sorter, sorts a ufun and can be used to get outcomes using their utiility
            self.sorter = PresortingInverseUtilityFunction(
                self.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            # Initialize the sorter. This is an O(nlog n) operation where n is the number of outcomes
            self.sorter.init()
        # get the current offer and prepare for rejecting it

        if self.ufun.reserved_value < 0.2:
            # The threshold at which we stop offer at the conceding mode
            self.time_thresh = (
                random.uniform(0.4, 0.5)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )
            # The last offer by the conceder
            self.concede_last_offer = (
                random.uniform(0.2, 0.3)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )
        elif self.ufun.reserved_value < 0.5:
            # The threshold at which we stop offer at the conceding mode
            self.time_thresh = (
                random.uniform(0.4, 0.5)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )
            # The last offer by the conceder
            self.concede_last_offer = (
                random.uniform(0.3, 0.4)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )
        else:
            # The threshold at which we stop offer at the conceding mode
            self.time_thresh = (
                random.uniform(0.4, 0.5)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )
            # The last offer by the conceder
            self.concede_last_offer = (
                random.uniform(0.3, 0.4)
                * (float(self.ufun(self.sorter.best())) - self.ufun.reserved_value)
                + self.ufun.reserved_value
            )

        if self.ufun.reserved_value < 0.1:
            self.thresh = self.ufun.reserved_value + random.uniform(0.26, 0.3)
        elif self.ufun.reserved_value < 0.3:
            self.thresh = self.ufun.reserved_value + random.uniform(0.21, 0.3)
        elif self.ufun.reserved_value > 0.8:
            self.thresh = self.ufun.reserved_value
        else:
            self.thresh = self.ufun.reserved_value + random.uniform(0.05, 0.1)

        while float(self.ufun(self.sorter.next_worse())) > self.thresh:
            self.sorter.next_worse()

        # Do the main MiCRO strategy
        if self.time_failed:
            # Offer from the opponent
            opponent_offer = state.current_offer
            # If I received something, check for acceptance
            if opponent_offer is not None:
                self._received.add(opponent_offer)
            # Determine if our agent should concede.
            # Find out my next offer and the acceptable offer
            concede = len(self._sent) <= len(self._received)
            # Determine our next offer
            # My next offer is either a conceding outcome if will_concede or sampled randomly from my past offers
            my_offer = self.sample_sent() if not concede else self.sorter.next_worse()
            # If there are no more offers left, the agent will not concede.
            # If I exhausted all my rational offers, do not concede
            if my_offer is None:
                concede, my_offer = False, self.sample_sent()
            else:
                my_utility = float(self.ufun(my_offer))
                if my_utility < self.ufun.reserved_value:
                    concede, my_offer = False, self.sample_sent()
            my_utility = float(self.ufun(my_offer))

            # Find my acceptable outcome. It will be None if I did not offer anything yet.
            acceptable_utility = self.worst_offer_utility if not concede else my_utility

            # The Acceptance Policy of MiCRO
            # accept if the offer is not worse than my acceptable offer if I am conceding or the best so far if I am not
            opponent_offer_utility = float(self.ufun(opponent_offer))
            if (
                opponent_offer is not None
                and opponent_offer_utility >= acceptable_utility
                and opponent_offer_utility >= self.ufun.reserved_value
            ):
                return SAOResponse(ResponseType.ACCEPT_OFFER, opponent_offer)
            # If I cannot find any offers, I know that there are NO rational outcomes in this negotiation for me and will end it.
            if my_offer is None:
                return SAOResponse(ResponseType.END_NEGOTIATION, None)
            # Offer my next-offer and record it
            self._sent.add(my_offer)
            self.worst_offer_utility = my_utility
            return SAOResponse(ResponseType.REJECT_OFFER, my_offer)

        # Do the conceding strategy and upper strategy
        else:
            self.update_reserved_value(state.current_offer, state.relative_time)

            opponent_offer = state.current_offer
            opponent_utility = float(self.ufun(opponent_offer))
            prob_upper = random.uniform(0, 1)
            if prob_upper > 0.7 and len(self.last_offer) > 0:
                my_offer = self.sorter.next_better()
            else:
                my_offer = self.sorter.next_worse()
            if my_offer is not None:
                self.last_offer.append(my_offer)
            if my_offer is None and len(self.last_offer) > 0:
                my_offer = random.choice(self.last_offer)
            else:
                my_offer = self.sorter.next_worse()

            my_utility = float(self.ufun(my_offer))

            # If the opponent offer is better than my offer, accept it
            if opponent_offer is not None and opponent_utility >= my_utility:
                return SAOResponse(ResponseType.ACCEPT_OFFER, opponent_offer)

            # Modeling acceptance criterion
            asp = aspiration_function(
                state.relative_time, 1.0, self.ufun.reserved_value, self.e
            )
            if opponent_offer is not None and opponent_utility >= asp:
                return SAOResponse(ResponseType.ACCEPT_OFFER, opponent_offer)
            # If my offer is better than the threshold, reject it
            if my_utility > self.time_thresh:
                return SAOResponse(ResponseType.REJECT_OFFER, my_offer)
            else:
                # If I failed to find an offer better than the threshold
                self.time_failed = True
                # Find the best offer that is worse than the threshold
                while float(self.ufun(my_offer)) > self.concede_last_offer:
                    my_offer = self.sorter.next_worse()
                return SAOResponse(ResponseType.REJECT_OFFER, my_offer)

    def sample_sent(self) -> Outcome | None:
        # Get an outcome from the set I sent so far (or my best if I sent nothing)
        if not len(self._sent):
            return None
        return random.choice(list(self._sent))

    def update_reserved_value(self, offer: Outcome | None, t: float):
        """Update the reserved value based on the opponent's offers"""

        assert self.opponent_ufun is not None

        if offer is None:
            return

        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self.past_opponent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
            self.past_opponent_rv = self.opponent_ufun.reserved_value
            self.opponent_ufun.reserved_value = optimal_vals[1]
        except Exception as e:
            _err, optimal_vals = f"{str(e)}", [None, None]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    run_a_tournament(Goldie, small=True)
