"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random
import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator
import math

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState
from negmas.preferences import pareto_frontier

__all__ = ["AwesomeNegotiator", "IngoNegotiator"]


class AwesomeNegotiator(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    opponent_history = []
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
        self.opponent_history.append(offer)

        # Never accept a proposal that is either empty or worse than or equal to our reservation value
        if self.ufun(offer) <= self.ufun.reserved_value or offer is None:
            return False

        # We consider the last 20% of offers as recent
        recent_utilities = [
            self.ufun(o)
            for o in self.opponent_history[round(-0.2 * len(self.opponent_history)) :]
        ]
        # We consider the last 3% of the negotation as the final offers, or the final round in smaller negotations

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        end_game = state.relative_time >= 0.97 or state.step + 1 == nsteps__

        # Accept a proposal under the following conditions:
        # The offer is better than the one that we are about to propose or the negotation is almost finished
        # The offer is better than the average of the recently made offers
        counter_offer = self.bidding_strategy(state)
        # alfa = max(recent_utilities)
        alfa = sum(recent_utilities) / len(recent_utilities)

        if (self.ufun(offer) > self.ufun(counter_offer) or end_game) and self.ufun(
            offer
        ) >= alfa:
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        sorted_outcomes = sorted(
            self.rational_outcomes,
            # Base the value of outcomes on a combination of our utility and our opponent's utility
            key=lambda o: 0.7 * self.ufun(o) + 0.3 * self.opponent_ufun(o),
            reverse=True,
        )
        # Propose every outcome in order of overall utility
        return sorted_outcomes[state.step % len(sorted_outcomes)]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(offer)) / 2

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]


class IngoNegotiator(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0  # From Tomas

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

        self.endgame_percentage = 0.03

        self.Min_value = 0
        self.Max_value = 1

        # self.reserved_value
        self.partner_reserved_value = 0
        self.opp_previous_offer = None
        self.own_previous_offer = None

        self.prev_offer = None
        self.offer = None
        self.prev_offer_opp = None
        self.offer_opp = None

        self.forgiveness_count = 0

        self.RV_granularity = 20

        # Calculate the size of each subrange
        subrange_size = (self.Max_value - self.Min_value) / self.RV_granularity

        # Create a list to store the averages of sub-ranges
        self.partner_reserved_value_range = []

        def float_range(start, stop, step):
            result = []
            while start < stop:
                result.append(start)
                start += step
            return result

        # Calculate the average for each sub-range
        for i in range(self.RV_granularity):
            start = self.Min_value + i * subrange_size
            end = start + subrange_size
            sub_range_values = list(float_range(start, end, 1 / self.RV_granularity))
            average = np.round(sum(sub_range_values) / len(sub_range_values), 2)
            self.partner_reserved_value_range.append(average)

        self.initial_proposal = None
        self.initial_proposal_opp = None

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.total_steps = nsteps__

        self.partner_rv_prob = [0] * self.RV_granularity
        self.partner_rv_prob[0] = 1
        self.partner_rv_prob_prev = None

        self.concession_rate_opp = 1
        self.concession_rate = 1
        self.current_step = 0

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
        self.prev_offer = self.offer
        self.offer = offer

        if self.initial_proposal_opp is None:
            self.opponents_initial_proposal = offer

        if self.initial_proposal is None:
            self.initial_proposal = self.bidding_strategy(state)

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

        self.opponent_history.append(offer)
        # We consider the last 20% of offers as recent
        recent_opponent_utilities = [
            self.ufun(o)
            for o in self.opponent_history[round(-0.2 * len(self.opponent_history)) :]
        ]
        # We consider the last 3% of the negotation as the final offers

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        end_game = state.relative_time >= 0.97 or state.step + 1 == nsteps__

        # Never accept a proposal that is worse than our reservation value
        if self.ufun(offer) < self.ufun.reserved_value:
            return False

        # Accept a proposal under the following conditions:
        # The offer is better than the one that we are about to propose or the negotation is almost finished
        # The offer is better than the average of the recently made offers
        counter_offer = self.bidding_strategy(state)
        # alfa = max(recent_opponent_utilities)
        alfa = sum(recent_opponent_utilities) / len(recent_opponent_utilities)

        if (self.ufun(offer) > self.ufun(counter_offer) or end_game) and self.ufun(
            offer
        ) >= alfa:
            return True
        return False

    def pareto_calculations(self):
        pareto_list = pareto_frontier([self.ufun, self.opponent_ufun])
        return pareto_list

    def closest_point(self, nodes, node):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

    def current_phase(self, state):
        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        percentage_complete = state.step / nsteps__
        if percentage_complete < 0.79:
            return 0
        elif percentage_complete < 0.97:
            return 1
        else:
            return 1

    def calc_lambda(self, time: int, proposal, prev_proposal, role="S"):
        base = time / (time - 1)
        if role == "S":
            ratio = proposal - self.initial_proposal
            ratio /= prev_proposal - self.initial_proposal
            return np.log(ratio) / np.log(base)
        else:
            ratio = proposal - self.initial_proposal_opp
            ratio /= prev_proposal - self.initial_proposal_opp
            return np.log(ratio) / np.log(base)

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        our_offers = []
        # Hyper parameters
        modifier = 0.5

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        chaos_threshold = 0.3 + (state.step / nsteps__)
        # Get all possible offers
        pareto_points = self.pareto_calculations()

        for i in self.rational_outcomes:
            if self.ufun(i) >= self.reserved_value:
                our_offers.append(
                    (
                        self.ufun(i),
                        self.opponent_ufun(i),
                        self.rational_outcomes.index(i),
                    )
                )

        if (
            self.current_phase(state=state) == 0
        ):  # early game basic tit for tat with chaos
            if self.opp_previous_offer is None:  # First offer
                offer_val = max(our_offers, key=lambda item: item[0])
                index = offer_val[2]
                self.own_previous_offer = self.rational_outcomes[index]
            elif random.random() > chaos_threshold:  # Chaos action
                offer_val = max(our_offers, key=lambda item: item[0])
                index = offer_val[2]
            else:  # tit for tat game
                if (
                    self.opponent_ufun(self.opp_previous_offer)
                    - self.opponent_ufun(state.current_offer)
                    > 0
                ):  # see if opponent is nice
                    self.forgiveness_count += 1
                else:
                    self.forgiveness_count = 0
                # calculate their movement
                tit_for_tat_dif = (
                    self.opponent_ufun(self.opp_previous_offer)
                    - self.opponent_ufun(state.current_offer)
                ) * modifier
                reverse_tit_for_tat_dif = (
                    self.ufun(self.opp_previous_offer) - self.ufun(state.current_offer)
                ) * modifier

                if (
                    self.forgiveness_count >= 3
                ):  # if defection for 3 turns in a row forgive (dont move)
                    offer_val = our_offers[
                        self.closest_point(
                            [LL[:2] for LL in our_offers],
                            (
                                self.ufun(self.own_previous_offer),
                                self.opponent_ufun(self.own_previous_offer),
                            ),
                        )
                    ]
                    self.forgiveness_count = 0
                else:  # match their movement
                    offer_val = our_offers[
                        self.closest_point(
                            [LL[:2] for LL in our_offers],
                            (
                                self.ufun(self.own_previous_offer) - tit_for_tat_dif,
                                self.opponent_ufun(self.own_previous_offer)
                                - reverse_tit_for_tat_dif,
                            ),
                        )
                    ]
                index = offer_val[2]
                self.own_previous_offer = self.rational_outcomes[index]

            self.opp_previous_offer = state.current_offer

            return self.rational_outcomes[index]
        elif self.current_phase(state=state) == 1:  # Phase 2 dance on pareto
            # get only the points close to pareto
            percentage_range = 0.3
            pareto_points = pareto_frontier([self.ufun, self.opponent_ufun])
            pareto_area_points = self.calculate_distances_to_pareto(self, pareto_points)
            pareto_area_points = pareto_area_points[
                : int(len(pareto_area_points) * percentage_range)
            ]
            for i in pareto_area_points:
                if i[0] >= self.reserved_value:
                    our_offers.append((i[0], i[1], i[3]))

            # tit for tat game
            # Should we add chaos here?
            if (
                self.opponent_ufun(self.opp_previous_offer)
                - self.opponent_ufun(state.current_offer)
                > 0
            ):  # see if opponent is nice
                self.forgiveness_count += 1
            else:
                self.forgiveness_count = 0
            # calculate their movement
            tit_for_tat_dif = (
                self.opponent_ufun(self.opp_previous_offer)
                - self.opponent_ufun(state.current_offer)
            ) * modifier
            reverse_tit_for_tat_dif = (
                self.ufun(self.opp_previous_offer) - self.ufun(state.current_offer)
            ) * modifier

            if (
                self.forgiveness_count >= 3
            ):  # if defection for 3 turns in a row forgive (dont move)
                offer_val = our_offers[
                    self.closest_point(
                        [LL[:2] for LL in our_offers],
                        (
                            self.ufun(self.own_previous_offer),
                            self.opponent_ufun(self.own_previous_offer),
                        ),
                    )
                ]
                self.forgiveness_count = 0
            else:  # match their movement
                offer_val = our_offers[
                    self.closest_point(
                        [LL[:2] for LL in our_offers],
                        (
                            self.ufun(self.own_previous_offer) - tit_for_tat_dif,
                            self.opponent_ufun(self.own_previous_offer)
                            - reverse_tit_for_tat_dif,
                        ),
                    )
                ]
            index = offer_val[2]
            self.own_previous_offer = self.rational_outcomes[index]

            self.opp_previous_offer = state.current_offer

            return self.rational_outcomes[index]
        elif self.current_phase(state=state) == 2:  # Aim for nash
            # print(nash_points([self.ufun,self.opponent_ufun],pareto_points[0],ranges=[(0.7,1),(0.6,1)]))
            # Learn how this function works
            pass
        # TODO
        # - Basic TitforTat (X)
        # - Only on or near pareto (X)
        # - Add chaos (X)
        # - Never offer below res value (X)
        # - Implement forgiveness (/)
        # - Add phases ()
        # - Clean up code (move functions to seperate file in helpers folder) ()

    def proposal(self, state: SAOState, role="s"):
        if role == "s":
            alpha = 1
            self.offer = self.prev_offer + np.power(-1, alpha) * np.power(
                1 / (self.total_steps(state.step - 1)), self.concession_rate
            ) * np.abs(self.reserved_value - self.prev_offer)
            self.prev_offer = self.offer
            return self.offer

        else:
            pass  # print("We should never be here")
            raise ValueError

    def update_partner_reserved_value(self) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        self.current_step += 1
        if self.current_step == 1:
            self.new_proposal = self.ufun(self.initial_proposal)

        if self.current_step == 2:
            self.probability_rv_opp = [0] * self.num_of_hyp
            self.probability_rv_opp[0] = 1

        if self.current_step >= 3:
            # compute opponent lambda using (7)
            log_base = self.current_step / (self.current_step - 1)
            ratio = self.opponent_ufun(self.opp_offer) - self.opponent_ufun(
                self.opponents_initial_proposal
            )
            ratio /= (
                self.opponent_ufun(self.prev_opp_offer)
                - self.opponent_ufun(self.opponents_initial_proposal)
                + 1e-6
            )
            self.lambda_opp = np.log(np.max([ratio, 1])) / np.log(log_base)

            # compute discount factor using (10)

            nsteps__ = (
                self.nmi.n_steps
                if self.nmi.n_steps
                else int(
                    (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                    + 0.5
                )
            )
            self.discount_ratio = (self.current_step / nsteps__) ** self.lambda_opp

            # Compute Prob(proposal opp at t | res val of opp i ) using (9) for all i
            probability_proposal_given_rv = []
            for opponents_reserved_value in self.opponents_reserved_value:
                numerator = np.abs(
                    (
                        self.opponent_ufun(self.opp_offer)
                        - self.opponent_ufun(self.opponents_initial_proposal)
                    )
                    / self.discount_ratio
                    - opponents_reserved_value
                    + self.opponent_ufun(self.opponents_initial_proposal)
                )
                denominator = (
                    self.opponent_ufun(self.opp_offer)
                    - self.opponent_ufun(self.opponents_initial_proposal)
                ) / self.discount_ratio
                probability_proposal_given_rv.append(1 - numerator / denominator)

            # Compute P (RP i opp | P t opp) using (11) for all i
            prob_sum = 0
            # eliminate RV that are greater than their offer
            for i in range(self.num_of_hyp):
                if (
                    self.opponent_ufun(self.opp_offer)
                    < self.opponents_reserved_value[i]
                ):
                    probability_proposal_given_rv[i] = 0
                prob_sum += (
                    self.probability_rv_opp[i] * probability_proposal_given_rv[i]
                )

            # compute all rv probabilities given proposal
            probability_rv_given_proposal = []
            for i in range(self.num_of_hyp):
                probability_rv_given_proposal.append(
                    (self.probability_rv_opp[i] * probability_proposal_given_rv[i])
                    / prob_sum
                )

            # Set P_{t-1}(RP opp i) = P(rv opp i | prob opp t-1)
            self.probability_rv_opp = probability_rv_given_proposal

            # compute RP opp at t using (12)
            opponents_reserved_value = 0
            for i in range(self.num_of_hyp):
                opponents_reserved_value += (
                    probability_rv_given_proposal[i] * self.opponents_reserved_value[i]
                )
            self.partner_reserved_value = opponents_reserved_value

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]

    # Given a list of points, returns list of tuples (util_x, util_y, distance_to_paretto_front) sorted from close to far
    def calculate_distances_to_pareto(self, state, pareto_list):
        pareto_list = sorted(pareto_list[0], key=lambda x: x[0])
        norm = np.linalg.norm
        least_lines = []
        line_segments = []
        points_with_distance = []

        a = [self.ufun(item) for item in self.rational_outcomes]
        b = [self.opponent_ufun(item) for item in self.rational_outcomes]
        c = [item for item in range(len(self.rational_outcomes))]
        points = list(zip(a, b, c))

        # Create a set of line that form the front
        p = pareto_list
        for x in range(len(p) - 1):
            line_segments.append((p[x], p[x + 1]))

        # For each point, check to what line it had the minimum distance
        for point in points:
            least_distance_to_segment = math.inf
            least_distance_line = [point, None]

            for line in line_segments:
                p1 = np.asarray(line[0])
                p2 = np.asarray(line[1])
                p3 = np.asarray(point[:2])
                v = p2 - p1
                w = p3 - p1
                t = np.dot(w, v) / np.dot(v, v)
                t = np.clip(t, 0, 1)
                p4 = p1 + t * v
                distance_from_segment = norm(p3 - p4)
                if distance_from_segment < least_distance_to_segment:
                    least_distance_to_segment = distance_from_segment
                    least_distance_line[1] = p4
            points_with_distance.append(
                (point[0], point[1], least_distance_to_segment, point[2])
            )
            least_lines.append(least_distance_line)

        points_with_distance.sort(key=lambda x: x[2])
        # print(points_with_distance)
        return points_with_distance
        # plotter(line_segments, points, least_lines)
