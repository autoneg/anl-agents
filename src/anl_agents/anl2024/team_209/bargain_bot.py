"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.preferences import nash_points, pareto_frontier
from negmas.sao import ResponseType, SAOResponse, SAOState

from .acceptance_logic import should_accept_offer
from .bidding_strategy import BiddingStrategy
from .opponent_model import update_reserved_value

__all__ = ["BargainBot"]


class BargainBot(ANLNegotiator):
    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """

        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of our last estimate of the opponent reserved value
        self.past_opponent_rv = 1.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        # list to hold past RV predictions
        self.past_opponent_rv_list = []

        # Nash difference list to be used in RV prediction
        self.nash_diffs: list[float] = []
        # Nash Equilibrium might be nonexistent
        self.nash_exists = 0
        # Pareto Frontier might be nonexistent
        self.pareto_front_exists = 0
        # Pareto frontier distances list to be used in RV prediction
        self.pareto_distances: list[float] = []
        # in which phase of the negotiation we are?
        self.neg_phase = 0
        # in how long a negotaition we are
        self.step_class = 0
        # if window_size < 0 then we cant use NE features.
        self.effective_feature_mode = 0

        self.bidding_strategy = BiddingStrategy(self)

        # To use in the RV prediction;
        # Find the Pareto Frontier:
        assert self.ufun is not None and self.opponent_ufun is not None
        self.frontier_utils, self.frontier_outcomes = pareto_frontier(
            [self.ufun, self.opponent_ufun]
        )
        if self.frontier_utils == ():
            self.pareto_front_exists = 0
        else:
            self.pareto_front_exists = 1

        # Find Nash outcome but might be nonexistent:

        assert self.ufun is not None and self.opponent_ufun is not None
        try:
            self.nash_utils, self.nash_outcome = nash_points(
                [self.ufun, self.opponent_ufun],  # type: ignore
                self.frontier_utils,  # type: ignore
            )[0]
            self.nash_welfare = sum(self.nash_utils)
            self.nash_exists = 1
        except Exception:
            self.nash_exists = 0
            # if there's no nash equilibrium, go back to single feature schema for RV prediction
            self.rv_feature_mode = 0

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
            - Txhis is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        # update the opponent reserved value in self.opponent_ufun
        update_reserved_value(self, offer)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        # return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))
        assert self.opponent_ufun is not None
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            self.bidding_strategy.get_next_bid(
                state, self.opponent_ufun.reserved_value
            ),
        )

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun is not None
        current_offer_utlity = float(self.ufun(state.current_offer))
        return should_accept_offer(
            current_offer_utlity, self.reserved_value, state.relative_time
        )
        # return False


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
# if __name__ == "__main__":
#     # from helpers.runner import run_a_tournament
#     # from helpers.config_test import grid_serach

#     run_a_tournament(BargainBot, small=True, debug=False)
#     # grid_serach(AwesomeNegotiator)
