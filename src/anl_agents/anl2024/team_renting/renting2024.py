"""
ANAC:
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import math
import random
import time
import warnings
from collections import deque
from itertools import chain, zip_longest

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from negmas.outcomes import Outcome
from negmas.preferences import nash_points, pareto_frontier
from negmas.sao import ResponseType, SAOResponse, SAOState
from scipy.optimize import curve_fit

from .ReservationValuePredictor import ReservationValuePredictor

warnings.filterwarnings("ignore")

__all__ = ["AgentRenting2024"]


class AgentRenting2024(ANLNegotiator):
    def __init__(self, **kwargs):
        """
        Initializes variables/lists.
        """

        super().__init__(**kwargs)
        self.debug = False
        self.name = "AgentRenting2024"

        self.sorted_rational_outcome = None
        self.last_and_first_offer = None

        self.own_nash_point = None
        self.opp_nash_point = None

        self.own_past_offers = []  # Done by ourselves
        self.opp_past_offers = []  # Done by the opponent

        # Opponent modeling
        self.use_curve_fitting = True
        self.curve_fitting_c = 0.9

        self.use_neural_network = True
        self.reservation_value_predictor = ReservationValuePredictor()
        self.nn_certainty_threshold = 0.8
        self.nn_range_threshold = 0.2

        self.partner_reserved_value = None
        self.partner_reserved_value_range_width = None

        self.times_used = []
        self.opp_start = None

        self.last_bid = None
        self.flat_line = False

        # self.P = array('f', [0] * self.nmi.n_steps)
        # self.S = array('f', [0] * self.nmi.n_steps)
        # self.Final_policy = deque()

    rational_outcomes = tuple()

    def on_preferences_changed(self, changes):
        """
        ANAC: Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        -   Resets variables/lists that need to be reset when initializing the agent.
        -   Set up nash point
        """

        # Bools needed for our negotiation that need to reset with every new negotiation
        self.last_and_first_offer = None

        # Setting our and the opponent's nash points.
        # We found that this can, albeit very rarely, go wrong in big tournaments.
        # For lack of an alternative, we assign our reserved value in this case
        try:
            nash_pts = self.nash_points()

            self.own_nash_point = nash_pts[0][0][0]
            self.opp_nash_point = nash_pts[0][0][1]
        except Exception:
            self.own_nash_point = self.reserved_value
            self.opp_nash_point = self.reserved_value

        self.own_past_offers = []
        self.opp_past_offers = []
        self.times_used = []
        self.opp_start = None

        # Failsafe of ANAC: If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        # Making a list of all outcomes that are above our reservation value
        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            # ANAC: enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        self.sorted_rational_outcome = sorted(
            self.rational_outcomes, key=self.ufun, reverse=True
        )

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as us
        self.partner_reserved_value = self.ufun.reserved_value
        self.partner_reserved_value_range_width = 1

        self.last_bid = None
        self.flat_line = False

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Wrapper that captures info for the rv predictor
        """

        if state.current_offer is not None:
            # Add offer to list of opponent offers
            self.opp_past_offers.append(state.current_offer)

        if state.current_offer is not None:
            if self.opp_start is None:
                # because we couldn't measure the first opp response time, we use a default
                self.times_used.append(0.1)
            else:
                self.times_used.append(time.perf_counter() - self.opp_start)

        start = time.perf_counter()
        response = self.handle(state)
        self.times_used.append(time.perf_counter() - start)

        self.opp_start = time.perf_counter()

        if response.response is ResponseType.REJECT_OFFER:
            # Add offer to list of own offers
            self.own_past_offers.append(response.outcome)

        return response

    def handle(self, state: SAOState) -> SAOResponse:
        """
        ANAC:
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.
        """

        # Renaming the current offer
        offer = state.current_offer

        # Checking whether we have the last offer or not. This only happens the first round.
        # We have come to believe that the agent who does the first offer is also the last agent able to accept an offer
        if self.last_and_first_offer is None:
            if offer is None:
                self.last_and_first_offer = True
            else:
                self.last_and_first_offer = False

            # self.debug_log(f"last offer: {self.last_and_first_offer}")
            # self.debug_log(f"our reservation value: {self.ufun.reserved_value}")
            self.debug_log(f"Total steps: {self.nmi.n_steps}")

        # Update the reserved value of the opponent
        # See the function update_partner_reserved_value
        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if self.use_neural_network and state.step > math.floor(nsteps__ * 0.95):
            self.update_partner_reserved_value(state)

        # A failsafe for if something goes wrong. ANAC: if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Generate our counter-offer
        our_offer = self.bidding_strategy(state)

        # Use our counter-offer in our acceptance strategy to determine whether we accept the current offer
        if offer is not None:
            if self.acceptance_strategy(state, our_offer):
                if self.last_and_first_offer:
                    self.debug_log("BJARNE ACCEPTED OFFER")
                else:
                    self.debug_log("MICK ACCEPTED OFFER")

                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If we do not accept, propose our conter-offer
        return SAOResponse(ResponseType.REJECT_OFFER, our_offer)

    def acceptance_strategy(self, state: SAOState, our_offer) -> bool:
        """
        Function that returns a bool. Either we accept the offer or we reject it
        """

        # Failsafe setup by ANAC
        assert self.ufun

        # Renaming the current offer
        offer = state.current_offer

        # We accept the opponent's (current) offer if it gives us more utility than our counter-offer would give us
        # Return true if we want to accept the offer
        if self.ufun(our_offer) <= self.ufun(offer):
            return True

        # When we do not have the last offer and our last offer has been rejected, we accept the opponent's
        # last offer if it gives us more utility than our reservation value, as the alternative is no agreement

        elif not self.last_and_first_offer:
            nsteps__ = (
                self.nmi.n_steps
                if self.nmi.n_steps
                else int(self.nmi.state.time / self.nmi.state.relative_time + 0.5)
            )
            if state.step == nsteps__ - 1:
                self.debug_log("last step")
                self.debug_log(self.ufun(offer))

                if self.reserved_value <= self.ufun(offer):
                    self.debug_log("concede")
                    return True

        return False

    # Our bidding strategy function
    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Returns: The counteroffer as Outcome.
        """

        # Returning an offer depending on whether we have the last offer
        if self.last_and_first_offer:
            return self.last_offer_strategy(state)
        else:
            return self.not_last_offer_strategy(state)

    def last_offer_strategy(self, state: SAOState):
        """
        Strategy for if we have last offer
        Returns: The counteroffer
        """
        # If our neural network has predicted the opponent's reservation value with a specific accuracy
        # we start flatlining on our current offer

        if (
            self.use_neural_network
            and self.partner_reserved_value_range_width < self.nn_range_threshold
            and self.last_bid is not None
        ) or self.flat_line:
            self.flat_line = True

            # In the final step, we stop flatlining and offer an offer that we expect to be just above
            # the opponent's reservation value

            nsteps__ = (
                self.nmi.n_steps
                if self.nmi.n_steps
                else int(
                    (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                    + 0.5
                )
            )
            assert self.opponent_ufun and self.ufun
            if nsteps__ - 1 == state.step:
                # Find the best offer for us that is at least better for the opponent
                # than their RV, and which is better for us than our RV
                if self.partner_reserved_value > float(
                    self.opponent_ufun(self.last_bid)
                ):
                    final_bid = None
                    for bid in self.rational_outcomes:
                        if self.opponent_ufun(bid) > self.partner_reserved_value and (
                            self.ufun(bid) > self.ufun(final_bid) or final_bid is None
                        ):
                            final_bid = bid

                    # If we do not find such an offer, we offer the nash point
                    if final_bid is None:
                        if self.ufun(self.last_bid) > self.own_nash_point:
                            final_bid = None
                            for bid in self.rational_outcomes:
                                if self.ufun(bid) >= self.own_nash_point and (
                                    self.opponent_ufun(bid)
                                    > self.opponent_ufun(final_bid)
                                    or final_bid is None
                                ):
                                    final_bid = bid
                            if final_bid is None:
                                final_bid = self.sorted_rational_outcome[0]

                    self.debug_log(state.step)
                    self.debug_log("using NN bid")
                    self.debug_log("Bjarne")
                    self.debug_log(
                        ("Utility current offer:", self.ufun(state.current_offer))
                    )
                    self.debug_log(("Flat lining on", self.ufun(final_bid)))
                    self.debug_log(("opp RV:", self.partner_reserved_value))
                    self.debug_log(
                        ("accuracy:", self.partner_reserved_value_range_width)
                    )

                    return final_bid

                # If the opponent's predicted RV is lower than their utility of our current offer,
                # we also return this offer as our final offer
                else:
                    self.debug_log(state.step)
                    self.log("using NN bid")
                    self.debug_log("Bjarne")
                    self.debug_log(
                        ("Utility current offer:", self.ufun(state.current_offer))
                    )
                    self.debug_log(("Flat lining on", self.ufun(self.last_bid)))
                    self.debug_log(("opp RV:", self.partner_reserved_value))
                    self.debug_log(
                        ("accuracy:", self.partner_reserved_value_range_width)
                    )

                    return self.last_bid

            # Flatlining until we reach our final offer
            else:
                self.debug_log(state.step)
                self.debug_log("Bjarne")
                self.debug_log(
                    ("Utility current offer:", self.ufun(state.current_offer))
                )
                self.debug_log(("Flat lining on", self.ufun(self.last_bid)))

                return self.last_bid

        # If we are not (yet) able to predict the opponent's RV accurately enough,
        # we determine our offers depending on a concession curve
        else:
            # Variable assignments:

            nsteps__ = (
                self.nmi.n_steps
                if self.nmi.n_steps
                else int(
                    (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                    + 0.5
                )
            )
            total_t = (
                nsteps__ - 1
            )  # Total negotiation time. As the negotiation steps start at 0,
            # we substract 1 from the total amount of steps
            t = state.step  # The current step
            e = 0.02  # This variable is responsible for the pace of concession.
            # Boulware is < 1, Conceder is > 1, Linear == 1.

            if total_t < 100:
                e = 0.05

            # The maximum utility we can get from all possible offers
            max_utility = self.ufun(self.sorted_rational_outcome[0])

            # The minimum utility that we concede to
            min_utility = self.own_nash_point  # ToDo: test, nash point?

            datapoints = 5
            c = self.curve_fitting_c

            # After a certain amount of time, model the opponent's concession curve and
            # update the minimum utility that we concede to accordingly
            if self.use_curve_fitting and t > max(datapoints + 1, total_t * 0.9):
                min_utility = self.opponent_modeling(
                    t, total_t, datapoints, c, min_utility, max_utility
                )

            # Choose our utility that we want our offer to be above in this step, using our concession curve
            next_utility = self.concession_function(
                min_utility, max_utility, t, total_t, e
            )

            # Determine our counter-offer using next_utility
            next_bid = self.next_offer(next_utility)

            # Version 2: A deque with the optimal bids at each step of the negotiation

            # If Neural Network (NN) is not ready
            # transform deque[t] coordinate into an offer
            # next_bid = transformed deque[t]

            # If NN is ready
            # Reduce the outcome.space with our reservation value and the estimated reservation value of the opponent
            # Find the bid with the highest utility for us (best_bid)
            # next_bid = best_bid
            # Repeat this offer until the end of the negotiation

            if t == 0:
                self.debug_log(("Bjarne's RV:", self.reserved_value))

            self.last_bid = next_bid
            return next_bid  # You need to return an offer here that can be given to the opponent

    def not_last_offer_strategy(self, state: SAOState):
        """
        Strategy for if we do not have last offer. Uses a concession curve similarly to
        the one used by last offer strategy when it is not flatlining. Het neural network is instead
        used to adjust the minimum utility that we concede to.
        Returns: the counteroffer
        """

        min_utility = (
            self.own_nash_point
        )  # Here too we start by conceding to the nash point
        max_utility = self.ufun(self.sorted_rational_outcome[0])
        t = state.step
        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )

        total_t = (
            nsteps__ - 2
        )  # In the ANL tournament system, the person without the final offer does
        # one final offer which does not get accepted or rejected, hence the - 2 here
        e = 0.05  # Determines the pace of concession. Slightly more conceding than in last_offer
        datapoints = 5
        c = self.curve_fitting_c

        if t == 0:
            self.debug_log(f"Mick's utility from nash point: {min_utility}")
            self.debug_log(f"Mick's RV: {self.reserved_value}")

        # After a certain amount of time, model the opponent's concession curve and
        # update the minimum utility that we concede to accordingly
        if self.use_curve_fitting and t > max(datapoints + 1, total_t * 0.9):
            min_utility = self.opponent_modeling(
                t, total_t, datapoints, c, min_utility, max_utility
            )

        # The part underneath is initiated when the neural network predicts the opponent's RV with high enough accuracy
        if self.partner_reserved_value_range_width < self.nn_range_threshold:
            # We calculate the utility our opponent gets from the final offer we would make
            final_offer = None
            for offer in self.rational_outcomes:
                if self.ufun(offer) >= min_utility and (
                    self.opponent_ufun(offer) > self.opponent_ufun(final_offer)
                    or final_offer is None
                ):
                    final_offer = offer

            # If this utility is lower than their predicted RV, we find the best offer for us which is above their RV
            if self.partner_reserved_value > self.opponent_ufun(final_offer):
                opp_min_utility = self.partner_reserved_value
                new_final_offer = None
                for offer in self.rational_outcomes:
                    if self.opponent_ufun(offer) > opp_min_utility and (
                        self.ufun(offer) > self.ufun(new_final_offer)
                        or new_final_offer is None
                    ):
                        new_final_offer = offer
                if t == total_t:
                    self.debug_log("Mick: NN worked")

                # The minimum utility which we concede to gets updated to the utility of this new final offer
                min_utility = self.ufun(new_final_offer)

        # Choose our utility that we want our offer to be above in this step, using our concession curve
        next_utility = self.concession_function(min_utility, max_utility, t, total_t, e)

        # Determine our counter-offer using next_utility
        next_bid = self.next_offer(next_utility)

        return next_bid  # You need to return an offer here that can be given to the opponent

    def next_offer(self, next_utility):
        """Takes next_utility (from our concession curve). Returns our counter-offer for this step"""

        # Select from a list of rational offers the offer which gives us at least next_utility
        # and the highest possible utility to the opponent
        next_bid = None
        for bid in self.rational_outcomes:
            if self.ufun(bid) >= next_utility and (
                self.opponent_ufun(bid) > self.opponent_ufun(next_bid)
                or next_bid is None
            ):
                next_bid = bid

        # If such an offer does not exist, offer the best possible for us
        if next_bid is None:
            next_bid = self.sorted_rational_outcome[0]

        return next_bid

    def concession_function(self, min_utility, max_utility, t, total_t, e):
        """The function for our concession curve. Returns our minimum utility for the current step"""
        return min_utility + (1 - (t / total_t) ** (1 / e)) * (
            max_utility - min_utility
        )

    def opponent_modeling(self, t, total_t, datapoints, c, min_utility, max_utility):
        """
        Models the opponent's concession curve to update our min_utility if necessary.
        Returns min_utility
        """
        try:  # Failsafe against curve fitting failing.
            time_list = list(range(t - datapoints, t))  # Creating time datapoints
            opponent_offers_subset = self.opp_past_offers[
                t - 1 - datapoints : t - 1
            ]  # Creating a subset of the past
            # 'datapoints' amount of opp. offers
            u_opponent_subset = []
            for o in opponent_offers_subset:
                u_opponent_subset.append(
                    self.opponent_ufun(o)
                )  # Creating opponent utility datapoints
            u = self.opponent_final_offer(
                total_t, time_list, u_opponent_subset
            )  # This function returns our utility
            # from the opponent's final offer
            # If this utility (times constant c) is higher than the current
            # utility we are conceding to, we update min_utility
            if u * c > min_utility and u * c < max_utility:
                new_min_utility = u * c
                return new_min_utility
            else:
                return min_utility
        except Exception:
            return min_utility

    def opponent_final_offer(self, total_t, t_data, p_data):
        """
        Uses non-linear least squares curve fitting to predict the opponent's final offer.
        Returns: our utility from the opponent's final offer
        """
        # Finding the opponent's maximum utility
        max_utility = 0
        for o in self.nmi.outcome_space.enumerate_or_sample():
            if self.opponent_ufun(o) > max_utility:
                max_utility = self.opponent_ufun(o)

        # The function for the opponent's concession curve
        def model(t_o, r_o, e_o):
            return r_o + (1 - (t_o / total_t) ** (1 / e_o)) * (max_utility - r_o)

        # Perform curve fitting
        initial_guess = (0.5, 0.05)  # Initial guess for parameters r and e
        params, covariance = curve_fit(model, t_data, p_data, p0=initial_guess)

        # Extract the estimated parameters
        estimated_r, estimated_e = params

        # Rename the paramater related to the minimum utility the opponent concedes to
        r = estimated_r.item()

        # Find the opponent's final offer related to this minimum utility and determine what utility this gives us
        u_final_offer = 0
        for o in self.nmi.outcome_space.enumerate_or_sample():
            if self.opponent_ufun(o) >= r and (self.ufun(o) > u_final_offer):
                u_final_offer = self.ufun(o)

        return u_final_offer

    # A function that gets called every round to update the expected reservation value of our opponent
    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """

        # Failsafe setup by ANAC, we should keep this
        assert self.ufun
        assert self.opponent_ufun

        def gen_bid_history_with_time(ufuns, offers, times):
            offer_cycles_utils = []
            for offer in offers:
                for i, ufun in enumerate(ufuns):
                    offer_cycles_utils.append(ufun(offer))

            # A_bid_A_utl, A_bid_B_utl, B_bid_A_utl, B_bid_B_utl ...
            # B_bid_A_utl, B_bid_B_utl, A_bid_A_utl, A_bid_B_utl,

            n_full_rounds = min(
                math.floor(len(offer_cycles_utils) / 4), math.floor(len(times) / 2)
            )

            # make sure offers and times are in same shape
            offer_cycles_utils = offer_cycles_utils[: n_full_rounds * 4]
            time_used = times[: n_full_rounds * 2]

            # Constructing A data
            A_offer_cycles_utils = []
            if len(offer_cycles_utils) > 0:
                A_offer_cycles_utils = [
                    offer_cycles_utils[i : i + 4]
                    for i in range(0, len(offer_cycles_utils), 4)
                ]

            A_time_used = [time_used[i : i + 2] for i in range(0, len(time_used), 2)]

            # merge offers and times together
            A_bid_history = [
                offers + times
                for offers, times in zip(A_offer_cycles_utils, A_time_used)
            ]

            return A_bid_history

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        n_relevant_steps = max(100, math.floor(nsteps__ * 0.1))

        if self.last_and_first_offer is True:
            past_offers = list(
                chain.from_iterable(
                    zip_longest(
                        self.own_past_offers, self.opp_past_offers, fillvalue=None
                    )
                )
            )
        else:
            past_offers = list(
                chain.from_iterable(
                    zip_longest(
                        self.opp_past_offers, self.own_past_offers, fillvalue=None
                    )
                )
            )

        past_offers = [offer for offer in past_offers if offer is not None]

        opp_bid_history = [
            float(self.opponent_ufun(offer)) for offer in self.opp_past_offers
        ]

        max_utility = 0
        for o in self.nmi.outcome_space.enumerate_or_sample():
            if self.opponent_ufun(o) > max_utility:
                max_utility = self.opponent_ufun(o)

        def model(t_o, r_o, e_o):
            return r_o + (1 - (t_o / self.nmi.state.step) ** (1 / e_o)) * (
                max_utility - r_o
            )

        t_data = list(range(1, self.nmi.state.step + 1))[-5:]
        p_data = opp_bid_history[-5:]

        try:
            # Perform curve fitting
            initial_guess = (0.5, 0.05)  # Initial guess for parameters r and e
            params, covariance = curve_fit(model, t_data, p_data, p0=initial_guess)

            # Extract the estimated parameters
            estimated_r, estimated_e = params

            concession_end_estimate = estimated_r.item()
        except Exception:
            concession_end_estimate = min(p_data)

        if len(opp_bid_history) > 0:
            (
                (partner_reserved_value, partner_reserved_value_certainty),
                (prediction_range, spread_certainty),
            ) = self.reservation_value_predictor.predict(
                gen_bid_history_with_time(
                    [self.ufun, self.opponent_ufun], past_offers, self.times_used
                ),
                np.mean(opp_bid_history)
                if len(opp_bid_history) > 1
                else opp_bid_history[0],
                np.std(opp_bid_history),
                np.min(opp_bid_history),
                np.max(opp_bid_history),
                np.mean(np.diff(opp_bid_history[-n_relevant_steps:]))
                if len(opp_bid_history) > 1
                else 0,
                np.sum(np.diff(opp_bid_history[-n_relevant_steps:])),
                concession_end_estimate,
                # self.nmi.state.step,
                self.nmi.state.step / self.nmi.n_steps
                if self.nmi.n_steps
                else self.nmi.state.relative_time,
                # self.nmi.n_steps - self.nmi.state.step,
                self.opp_nash_point,
                certainty_minimum=self.nn_certainty_threshold,
            )

            predicted_partner_reserved_value = prediction_range[1]

            if (
                np.min(opp_bid_history) < predicted_partner_reserved_value
                or predicted_partner_reserved_value > self.opp_nash_point
            ):
                self.debug_log("NN made impossible prediction...")
                self.debug_log("min bid:", np.min(opp_bid_history))
                self.debug_log("predicted:", predicted_partner_reserved_value)
                self.debug_log("opp nash pt:", self.opp_nash_point)

                # Fix the rv to be within its possibilities
                predicted_partner_reserved_value = min(
                    max(np.min(opp_bid_history), predicted_partner_reserved_value),
                    self.opp_nash_point,
                )

            self.partner_reserved_value = predicted_partner_reserved_value
            self.partner_reserved_value_range_width = (
                prediction_range[1] - prediction_range[0]
            )

            if self.partner_reserved_value_range_width < self.nn_range_threshold:
                self.debug_log("rv prediction:", partner_reserved_value)
                self.debug_log(
                    "rv prediction certainty:", partner_reserved_value_certainty
                )
                self.debug_log("rv prediction range:", prediction_range)

    def nash_points(self) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        frontier, frontier_outcomes = self.pareto_frontier()

        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        nash_pts = nash_points(
            [self.ufun, self.opponent_ufun],
            frontier,
            outcome_space=self.nmi.outcome_space,
        )
        return tuple(
            (nash_utils, frontier_outcomes[indx]) for nash_utils, indx in nash_pts
        )

    def pareto_frontier(self) -> tuple[tuple[tuple[float, ...], ...], list[Outcome]]:
        if any(_ is None for _ in [self.ufun, self.opponent_ufun]):
            raise ValueError(
                "Some negotiators have no ufuns. Cannot calculate the pareto frontier"
            )

        outcomes = self.nmi.discrete_outcomes()
        results = pareto_frontier(
            [self.ufun, self.opponent_ufun],
            outcomes=outcomes,
            n_discretization=None,
            max_cardinality=float("inf"),
        )

        return results[0], [outcomes[_] for _ in results[1]]

    def QGCA(self) -> deque:
        "Select the best bids from the outcome space"
        policy = deque()
        L = []
        for outcome in self.nmi.outcome_space:
            L.append(self.ufun(outcome), self.opponent_ufun(outcome), 0)

        sorted_L = sorted(L, key=lambda x: x[2], reverse=True)

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        for k in range(1, nsteps__ + 1):
            d_prime = 0.0
            w_prime = ()
            for bid in self.nmi.outcome_space:
                sorted_L = sorted(L, key=lambda x: x[2], reverse=True)
                for outcome in sorted_L:
                    i = outcome[2]
                    dw = self.S[i - 1] + (1 - outcome[1]) * (
                        self.EU(len(policy), self.ufun.reserved_value)
                        - self.S[i - 1]
                        + self.P[i] * (outcome[0] * outcome[1]) * outcome[1]
                    )
                    if dw >= d_prime:
                        d_prime = dw
                        w_prime = bid
            policy.append(w_prime)
            self.P[i] = self.P[i - 1] * (1 - w_prime[1])
            self.S[i] = self.S[i - 1] + w_prime[0] * w_prime[1] * self.P[i]
            # L(w) = increase the count for each w < w_prime
            for offer in sorted_L:
                if offer[0] + offer[1] < w_prime[0] + w_prime[1]:
                    offer[2] += 1

        return policy

    def PBS(self, policy_1) -> deque:
        """Sort and find the best permutation"""
        policy = policy_1

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        for r in range(1, nsteps__ + 1):
            org_policy = policy
            for k in range(0, nsteps__):
                if self.swap_check_neighbor(policy, k) > 0:
                    policy = self.swap(policy, k, k + 1)
            if policy == org_policy:
                return policy
        return policy

    def PA(self, policy_2) -> deque:
        "Randomly modify an outcome of the policy and check if it leads to"
        "a higher expected utility, if not, accept with a lower chance"
        policy = policy_2
        outcomespace = [
            (self.ufun(outcome), self.opponent_ufun(outcome))
            for outcome in self.nmi.outcome_space
        ]

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        for r in range(nsteps__):
            s = random.randint(0, nsteps__)
            selected_offer = policy[s]
            reduced_os = outcomespace.remove(selected_offer)
            random.shuffle(reduced_os)
            replacement_offer = reduced_os[0]
            delta = self.replace_check(policy, replacement_offer, s)

            if delta > 0 or random.randint(0, nsteps__) > math.exp(-delta / (0.05 * r)):
                policy = self.replace(policy, s, replacement_offer)

            for i in range(0, nsteps__ + 1):
                if i != s:
                    if self.swap_check(policy, i, s) > 0:
                        policy = self.swap(policy, i, s)

        return policy

    def EU(self, policy_length, reservation_value):
        "Calculate the expected value of a series of bids"
        return self.S[policy_length - 2] + reservation_value * self.P[policy_length - 1]

    def swap_check_neighbor(self, policy, k):
        "Calculate the change in EU when swapping index k with k+1"
        return (
            self.P[k]
            * self.opponent_ufun(policy[k])
            * self.opponent_ufun(policy[k + 1])
            * (self.ufun(policy[k + 1]) - self.ufun(policy[k]))
        )

    def swap_check(self, policy, i, j):
        "Calculate the change in EU when swapping index i with j"
        first_part = (
            self.ufun(policy[i])
            * self.opponent_ufun(policy[i])
            * (
                self.P[j]
                * (
                    1
                    - self.opponent_ufun(policy[j])
                    / (1 - self.opponent_ufun(policy[i]))
                    - self.P[i]
                )
            )
        )
        second_part = (
            self.ufun(policy[j])
            * self.opponent_ufun(policy[j])
            * (self.P[i] - self.P[j])
        )
        third_part = (
            (self.S[j - 1] - self.S[i])
            * (self.opponent_ufun(policy[i]) - self.opponent_ufun(policy[j]))
            / (1 - self.opponent_ufun(policy[i]))
        )
        return first_part + second_part + third_part

    def swap(self, policy, i, j):
        "Swap the offers at index i & j"
        policy[i], policy[j] = policy[j], policy[i]
        return policy

    def replace_check(self, policy, w, i):
        "Check if replacing the offer at i with w yields a better EU"
        first_part = self.P[i] * (self.ufun(w) * self.opponent_ufun(w)) - (
            self.ufun(policy[i]) * self.opponent_ufun(policy[i])
        )

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        second_part = (self.S[nsteps__] - self.S[i]) * (
            self.opponent_ufun(policy[i])
            - self.opponent(policy[w]) / (1 - self.opponent_ufun(policy[i]))
        )
        return first_part + second_part

    def replace(self, policy, i, replacement_offer):
        "replace the offer at index i with the replacement offer"
        policy[i] = replacement_offer
        return policy

    def debug_log(self, *p):
        if self.debug:
            self.log(*p)

    def log(self, *p):
        pass  # print(*p)
