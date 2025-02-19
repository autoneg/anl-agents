import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["Nayesian"]


class Nayesian(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()

    partner_reserved_value = 0

    def __init__(self, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of our last estimate of the opponent reserved value
        self.oppnent_rv_range = [0, 1]
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        # self._rational: list[tuple[float, float, Outcome]] = []

    def on_preferences_changed(self, changes):
        """
        Called just after the ufun is set and before the negotiation starts.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        # self.opponent_ufun = self.private_info["opponent_ufun"]

        outcomes = self.nmi.outcome_space.enumerate_or_sample()

        self.ordered_outcomes = sorted(
            [
                (self.ufun(outcome), self.opponent_ufun(outcome), outcome)
                for outcome in outcomes
            ],
            key=lambda x: x[0],
            reverse=True,
        )  # sort from high to low according my utils

        self.rational_outcomes = []
        pair_utilities = []
        prod_utilities = []
        for o in self.ordered_outcomes:
            if o[0] > self.ufun.reserved_value:
                self.rational_outcomes.append(o[2])
                pair_utilities.append([o[0], o[1]])
                prod_utilities.append(o[0] * o[1])
        self.oppnent_rv_max = pair_utilities[-1][1]
        self.pair_utilities = np.array(pair_utilities)
        self.prod_utilities = np.array(prod_utilities)

        best_nash_outcome_i = np.argmax(self.prod_utilities)
        self.best_nash_outcome = self.rational_outcomes[best_nash_outcome_i]
        self.worst_nash_outcome = self.rational_outcomes[-1]

        self.best_nash_u = self.pair_utilities[best_nash_outcome_i, 0]
        self.worst_nash_u = self.pair_utilities[-1, 0]

        self.acceptable_utility = 1
        # print('ufun_initilize')

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
        # print('called', offer)
        self._update_nash(offer)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            # print('None')
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            # print('accept')
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def _update_nash(self, offer):
        # print('update')
        # update wrost nash according to opponent's bids
        if offer is None:
            return
        # update partner reservation value, first use the easist way
        # print(offer)
        opp_offer_u = self.opponent_ufun(offer)
        # print('ufun_cal')
        if opp_offer_u > self.oppnent_rv_max:
            return
        else:
            self.oppnent_rv_max = opp_offer_u
        self.offer_u_i = np.searchsorted(self.pair_utilities[:, 1], opp_offer_u)

        self.worst_nash_outcome = self.rational_outcomes[self.offer_u_i]

    def acceptance_strategy(self, state: SAOState) -> bool:
        # print('In acceptance')
        offered_u = self.ufun(state.current_offer)
        # print('received utility:', offered_u)
        if (offered_u >= self.acceptable_utility) or (offered_u >= self.best_nash_u):
            return True
        else:
            return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        # print('bidding_start')
        t = state.relative_time
        asp = (1 - self.best_nash_u) * (1.0 - np.power(t, 5)) + self.best_nash_u
        # print('asp:', asp)
        index = np.searchsorted(self.pair_utilities[:, 0][::-1], asp, side="right")
        # print('searched:', self.pair_utilities[:,0][::-1][index-1])
        index = self.pair_utilities[:, 0].size - index
        # print('indexted:', self.pair_utilities[:,0][index])
        # print('bidding_u:', self.ufun(self.rational_outcomes[index]))

        return self.rational_outcomes[index]


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    # if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
    run_a_tournament(Nayesian, small=True)
