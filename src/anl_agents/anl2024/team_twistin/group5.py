"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Twister Team
*Authors*
Ali Kargin (a.kargin@students.uu.nl),
Sam Leurink (s.h.leurink@students.uu.nl),
Marc Overbeek (m.overbeek2@students.uu.nl),
Sacha Vucinec (s.i.vucinec@students.uu.nl),
Pieter van der Werff (p.e.vanderwerff@uu.nl)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import math
import random
from operator import itemgetter

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from matplotlib import pyplot as plt
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["Group5"]


def safelog(x, *args, **kwargs):
    if x < 1e-10:
        return -3000.0
    return math.log(x, *args, **kwargs)


class Group5(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    pareto_frontier = list()
    partner_reserved_value = None
    opponent_offer_history = None
    opponent_offer_history2 = None

    # Reservation value estimation
    deadline = None
    opponent_rv_range = None
    n_estimation_areas = None
    estimation_areas = None
    plot_opponents_bidding_curve = False

    last_offer = None
    opponent_last_offer = None
    last_util = 0
    first_util = 0
    first_bidder = False

    # Hyper parameter which defines the minimum increase for our pretend reservation
    mu = 0.0

    # Hyper parameter which defines the concession curve (The lower the number the steeper the concession at the end)
    psi = 0.5

    # Hyper parameter which defines the confidence rate we have in the opponent rv estimation
    rho = 0.05

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
            (self.ufun(_), self.opponent_ufun(_), self.utilityScore(_, False), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        if callable(self.mu) and self.mu.__name__ == "<lambda>":
            self.mu = self.mu(self.ufun.reserved_value)

        self.create_pareto_front()

        self.rational_outcomes.sort(key=itemgetter(2), reverse=True)

        # Reset class-level variables between negotiations
        self.partner_reserved_value = 0
        self.opponent_offer_history = list()
        self.opponent_offer_history2 = list()

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.deadline = nsteps__  # Only takes into account a step limit, not time limit. Should it be expanded?
        self.opponent_rv_range = [0.0, 1.0]
        self.n_estimation_areas = 100
        self.estimation_areas = list()
        self.first_bidder = self.nmi.negotiator_index(self.id) == 0

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

        # Create the bid once.
        next_offer = self.bidding_strategy(state)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state, next_offer):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, next_offer)

    def create_pareto_front(self):
        """
        Function to calculate the pareto frontier to make sure our final bids are decently pareto optimal.
        Also usefull in calculating our max pretend reservation.

        Note that we will never have to remove an element because of the fact that the lists are sorted beforehand
        """
        sorted_list = sorted(self.rational_outcomes, key=itemgetter(1), reverse=True)
        sorted_list = sorted(sorted_list, key=itemgetter(0), reverse=True)
        for own_util, opponent_util, calc_util, outcome in sorted_list:
            if len(self.pareto_frontier) == 0:
                self.pareto_frontier.append(
                    (own_util, opponent_util, calc_util, outcome)
                )
            else:
                pareto_optimal = True
                for entry in self.pareto_frontier:
                    if own_util < entry[0] and opponent_util < entry[1]:
                        pareto_optimal = False
                        break
                if pareto_optimal:
                    self.pareto_frontier.append(
                        (own_util, opponent_util, calc_util, outcome)
                    )

    def acceptance_strategy(self, state: SAOState, nextOffer: Outcome) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer

        """
        This part keeps track of all the offers that were recieved so far.
        """
        self.opponent_offer_history2.append(self.ufun(offer))

        """
        Reject offer if no offer has been made.
        """
        if offer is None:
            return False

        """
        This part rejects any offer that is smaller then our reservation value.
        We take 0.5 * mu to ensure we don't take the bare minimum.
        """
        OpponentGain = self.opponent_ufun(offer)
        NextOpponentGain = self.opponent_ufun(nextOffer)

        if self.ufun(offer) < (self.ufun.reserved_value + (self.mu / 2)):
            return False

        """
        because the last part rejects all the offers that are really bad,
        it is safe to accept the last offer that is made in the negotiation.
        """

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if nsteps__ == state.step:
            return True

        """
        This part accepts any offer that is extremely beneficial for us.
        Like if this is why an offer is accepted it has to be because our opponent made a mistake.
        Always accept if our gain is big enough to where we get enough value out of the deal compared to the number of opponents in the tournament.
        """
        # if (self.ufun(offer) - self.ufun.reserved_value) > ((OpponentGain - self.partner_reserved_value) * AmountOfOpponents):
        #    return True

        """
        This part accepts any offer that would be better then the one we were going to send in the next bidding round,
        and our opponent does not gain more then us from it
        """
        if (self.ufun(offer) > self.ufun(nextOffer)) & (
            (self.ufun(nextOffer) - self.ufun(offer)) > NextOpponentGain - OpponentGain
        ):
            return True

        """
        This part only kicks in at the last 10% of the negotiation.
        It will reject any offer that is not in the top of offers that we have recieved so far.
        This window will shrink as the negotiation goes on.
        """

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        time_remaining = nsteps__ - state.step
        if state.relative_time > 0.9:
            for x in self.opponent_offer_history2[-time_remaining:]:
                if x > self.ufun(offer):
                    return False
            return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        # If this is the first step of the negotiation, offer the starting bid.
        bid = None
        if self.last_offer is None:
            bid = self.starting_bid()
        else:
            # Calculate our pretend reservation value
            pretend_reservation = self.calculate_reservation(state)

            # Calculate the concession rate based on the current state
            concession = self.calculate_concession(state, pretend_reservation)

            # Calculate the opponent concession rate with our calculated expected concession
            final_concession = self.calculate_final_concession(state, concession)

            # Select bid based on the concession
            bid = self.choose_bid(final_concession)

        # Set last offer and opponent last offer as variables for the next bid.
        self.last_offer = bid
        self.opponent_last_offer = state.current_offer

        # Return the bid
        return bid

    def calculate_reservation(self, state: SAOState) -> int:
        """
        This function is used to calculate the pretend reservation. It takes the expected opponent
        reservation value and checks in the pareto front what our top pretend reservation can be.
        This pretend reservation isn't calculated at the start but during every offer, so the pretend
        reservation gets adjusted as we go.
        """

        # Calculate opponent reservation value and find corresponding point in pareto frontier
        reservation_opp_res = self.partner_reserved_value + self.rho
        wRO = min(self.pareto_frontier, key=lambda x: abs(reservation_opp_res - x[1]))

        # Calculate own minimum artificial reservation value and find corresponding point in pareto frontier
        reservation_with_mu = self.ufun.reserved_value + self.mu
        wMU = min(self.pareto_frontier, key=lambda x: abs(reservation_with_mu - x[0]))

        # Compare the utility scores of points in the pareto front and return the best one.
        if wRO[2] > wMU[2]:
            return wRO[2]
        else:
            return wMU[2]

    def calculate_concession(self, state: SAOState, reservation: int) -> int:
        """
        This function calculates the concession rate according to the concession curve without
        the opponent concession discount.
        """

        # Adjusted time function

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if self.first_bidder:
            time = state.step / nsteps__
        else:
            time = (state.step + 1) / nsteps__

        # Calculate the concession rate based on function described in paper
        power = 1 / self.psi
        util_expected = reservation + (self.first_util - reservation) * (
            1 - pow(time, power)
        )
        concession_rate = self.last_util - util_expected

        # Make sure concessions are positive
        return max(concession_rate, 0)

    def calculate_final_concession(self, state: SAOState, concession: int) -> int:
        """
        This function calculates the concession discount according to opponent concessions.
        Returns the finalized concession with discount included.
        """

        # Adjusted time function

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if self.first_bidder:
            time = state.step / nsteps__
        else:
            time = (state.step + 1) / nsteps__

        # Return concession if lower part of ratio function is 0
        if concession == 0 or time == 1:
            return concession

        # Calculate opponent concession and check the ratio as described in paper
        opponent_concession = self.utilityScore(
            self.opponent_last_offer, True
        ) - self.utilityScore(state.current_offer, True)
        ratio_t = (opponent_concession + 0.01) / (concession * (1 - time))
        if ratio_t >= 1:
            return 1 * concession
        else:
            return max(ratio_t * concession, 0)

    def choose_bid(self, concession: int) -> Outcome:
        """
        This function chooses the bid based on the finalized concession and previous utility.
        """

        # Update last utility
        self.last_util = self.last_util - concession

        # Choose closest bid in rational outcomes
        bid = min(self.rational_outcomes, key=lambda x: abs(self.last_util - x[2]))

        return bid[3]

    def utilityScore(self, outcome: Outcome, opponent: bool) -> float:
        """
        Returns the utility score of an offer, used determining the starting bid and
        concession rate.

        Returns: The utility score as float.
        """
        if opponent:
            return self.opponent_ufun(outcome)

            # Possible new utility function
            return 2 * self.opponent_ufun(outcome) + (1 - self.ufun(outcome))
        else:
            return self.ufun(outcome)

            # Possible new utility function
            return 2 * self.ufun(outcome) + (1 - self.opponent_ufun(outcome))

    def starting_bid(self):
        """
        Generates the opening bid. For this bid, we choose the outcome with the lowest possible opponent utility.
        If multiple outcomes yield the minimum opponent utility, offer the one among those outcomes that has the
        highest own utility.

        Returns: The opening offer as Outcome.
        """
        # First, define the list of all outcomes that are rational for us and sort them in order of descending own utility.
        # This is done to break ties between outcomes with the same utilityScore.
        outcomeList = [
            (my_util, _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
        ]
        outcomeList.sort(key=itemgetter(0))

        # Then, return the outcome with the highest utilityScore as the starting bid.
        startingBid = max(outcomeList, key=lambda x: self.utilityScore(x[1], False))[1]

        # Set starting util and last util for first bid.
        self.last_util = self.first_util = self.utilityScore(startingBid, False)
        return startingBid

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        # Add opponent's current offer to offer history
        if offer is None:
            self.opponent_offer_history.append(None)
        else:
            self.opponent_offer_history.append(self.opponent_ufun(offer))

        if state.step == 0:
            return

        # Initialize estimation areas
        if len(self.estimation_areas) == 0:
            area_boundaries = np.linspace(
                *self.opponent_rv_range, num=self.n_estimation_areas + 1
            )
            lower_boundary = area_boundaries[0]
            for upper_boundary in area_boundaries[1:]:
                self.estimation_areas.append(
                    {"lower": lower_boundary, "upper": upper_boundary, "mse": 1}
                )
                lower_boundary = upper_boundary

        # Filter out any non-offer rounds from the offer history, so it can be used for regression analysis properly.
        # This also sets the maximum utility as the first offer and disregards any previous offers.
        proper_offer_history = []
        max_offer = 0
        for i, offer in enumerate(self.opponent_offer_history):
            if offer is not None:
                if offer >= max_offer:
                    proper_offer_history = [(i, offer)]
                    max_offer = offer
                else:
                    proper_offer_history.append((i, offer))

        # Stop if there is not enough information in the offer history
        if len(proper_offer_history) == 0:
            return

        # Regression analysis
        # Step 1: select a random reservation point X_i(t_x, rv_x) in each estimation area.
        for area in self.estimation_areas:
            rv_x = random.random() * (area["upper"] - area["lower"]) + area["lower"]
            area["estimation"] = rv_x
            t_x = self.deadline

            # Step 2: using each point chosen in step 1, calculate the regression line rl based on the opponent's
            # offer history.
            _, offer0 = proper_offer_history[0]

            if offer0 <= rv_x or len(proper_offer_history) <= 1:
                continue

            # Calculate regression coefficient (see equation 5 and 6 in the Bayesian Learning paper). Here I
            # assume there is a mistake in the paper, I use t_i instead of state.step.
            eq6_top = sum(
                [
                    safelog((offer0 - offer_i) / (offer0 - rv_x)) * safelog(t_i / t_x)
                    for t_i, offer_i in proper_offer_history[1:]
                ]
            )
            eq6_bottom = sum(
                [safelog(t_i / t_x) ** 2 for t_i, _ in proper_offer_history[1:]]
            )
            b = eq6_top / eq6_bottom

            # Step 3: based on the calculated regression line, the buyer can calculate the fitted offers for each round.
            fitted_offers = [
                (t, offer0 + (0 if t == 0 else ((rv_x - offer0) * (t / t_x) ** b)))
                for t in range(proper_offer_history[0][0], state.step + 1)
            ]

            # Step 4: Use mean squared error to determine how good of a fit this estimation is.
            mse = 0
            historical_offers_idx = 0
            for i, t in enumerate(range(fitted_offers[0][0], fitted_offers[-1][0])):
                if t < proper_offer_history[historical_offers_idx][0]:
                    continue
                elif t > proper_offer_history[historical_offers_idx][0]:
                    pass  # print('This shouldn\'t happen...')
                else:
                    mse += (
                        proper_offer_history[historical_offers_idx][1]
                        - fitted_offers[i][1]
                    ) ** 2
                    historical_offers_idx += 1

            area["mse"] = mse / len(proper_offer_history)
            if self.plot_opponents_bidding_curve:
                area["fitted"] = fitted_offers

        # Find the estimation area with the smallest mse, aka the best estimation.
        max_area = min(self.estimation_areas, key=itemgetter("mse"))
        self.partner_reserved_value = max_area["estimation"]

        if not self.plot_opponents_bidding_curve or state.step < (0.9 * self.deadline):
            return

        _, ax = plt.subplots()

        line1_x = [point[0] for point in proper_offer_history]
        line1_y = [point[1] for point in proper_offer_history]
        ax.plot(line1_x, line1_y, label="Offer history")

        line2_x = [point[0] for point in max_area["fitted"]]
        line2_y = [point[1] for point in max_area["fitted"]]
        ax.plot(line2_x, line2_y, label="Fitted offers")

        ax.plot(
            self.deadline,
            max_area["estimation"],
            marker=".",
            label=f'Reservation Value estimation ({max_area["estimation"]})',
            markersize=15,
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Opponent's utility")
        ax.set_title(f'Offer history vs Fitted offers (MSE = {max_area["mse"]})')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, self.deadline)
        ax.legend()
        plt.show()


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))

    from helpers.runner import run_a_tournament

    run_a_tournament(Group5)
