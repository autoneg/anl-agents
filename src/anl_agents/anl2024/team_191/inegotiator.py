"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import copy
import math
import random

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
import scipy
from negmas.outcomes import Outcome
from negmas.preferences import nash_points, pareto_frontier
from negmas.sao import ResponseType, SAOResponse, SAOState


__all__ = ["INegotiator"]


class INegotiator(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        _ = changes
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            # enumerates outcome space when finite, samples when infinite
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]

        self.partner_reserved_value = 0
        self.own_previous_offer = None

        self.forgiveness_count = 0
        self.phase_2_flag = (
            False  # flag to only add the pareto points once to the list of offers
        )
        self.initial_proposal = None
        self.opponents_initial_proposal = None
        self.our_offers = []
        self.opp_offer = None
        self.prev_opp_offer = None
        self.concession_rate = 1
        self.new_proposal = None

        min_number_hyp = 10

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if nsteps__ is not None:
            max_number_hyp = np.min([int(nsteps__ / 5), 100])
        else:
            max_number_hyp = 100

        self.num_of_hyp = np.max([min_number_hyp, max_number_hyp])
        segments = np.linspace(0, 1, num=self.num_of_hyp + 1)
        self.opponents_reserved_value = [
            (segments[i] + segments[i + 1]) / 2 for i in range(len(segments) - 1)
        ]
        self.total_time = None
        self.current_step = 0
        self.probability_rv_opp = []
        self.opponent_history = []
        self.bidding_initializations()

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
        self.prev_opp_offer = self.opp_offer
        self.opp_offer = offer

        if self.opponents_initial_proposal is None:
            self.opponents_initial_proposal = offer

        self.update_partner_reserved_value_2(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        our_bid = self.bidding_strategy(state)
        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state, our_bid):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        if self.initial_proposal is None:
            self.initial_proposal = our_bid
        return SAOResponse(ResponseType.REJECT_OFFER, our_bid)

    def acceptance_strategy(self, state: SAOState, counter_offer) -> bool:
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

        # We consider the last 3% of the negotation as the final offers, or the final round in smaller negotations

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        end_game = self.current_phase(state) == 2 or state.step + 1 == nsteps__
        # In case we get significantly more utility than our reservation value in the end game, we accept
        if (
            end_game
            and (self.ufun(offer) > self.ufun.reserved_value + 0.3)
            or (self.ufun(offer) > self.ufun.reserved_value * 3)
        ):
            return True

        # We consider the last 20% of offers as recent
        recent_utilities = []

        for previous_offer in self.opponent_history[
            int(np.floor(0.8 * len(self.opponent_history))) :
        ]:
            recent_utilities.append(self.ufun(previous_offer))

        # Accept a proposal under the following conditions:
        # The offer is better than the one that we are about to propose or the negotation is almost finished
        # The offer is better than the average of the recently made offers
        alfa = sum(recent_utilities) / len(recent_utilities)
        if (self.ufun(offer) > self.ufun(counter_offer) or end_game) and self.ufun(
            offer
        ) >= alfa:
            return True
        return False

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
        steps_left = nsteps__ - state.step
        min_endgame_steps = 5
        if percentage_complete < 0.89 and steps_left > min_endgame_steps:
            return 0
        elif percentage_complete < 0.97 and steps_left > min_endgame_steps:
            return 1
        else:
            return 2

    # Given a list of points, returns list of tuples (util_x, util_y, distance_to_paretto_front) sorted from close to far
    def calculate_distances_to_pareto(self, pareto_list):
        pareto_list = sorted(pareto_list[0], key=lambda x: x[0])
        norm = np.linalg.norm
        least_lines = []
        line_segments = []
        points_with_distance = []
        assert self.ufun and self.opponent_ufun

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
        return points_with_distance

    def bidding_initializations(self):
        assert self.ufun and self.opponent_ufun
        for i in self.rational_outcomes:
            self.our_offers.append(
                (self.ufun(i), self.opponent_ufun(i), self.rational_outcomes.index(i))
            )
        pareto_points = pareto_frontier([self.ufun, self.opponent_ufun])
        self.pareto_area_points = self.calculate_distances_to_pareto(pareto_points)
        # self.pareto_area_points = self.pareto_area_points[:len(pareto_points)]
        self.pareto_area_points = [i for i in self.pareto_area_points if i[2] == 0.0]

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # start_time = time.time()
        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        # Hyper parameters
        if not hasattr(self, "hyper"):
            self.hyper = None

        if self.hyper is None:
            # phase 1 + 2 (percentage of matching concession) (larger seems to be better, 0.6 in test)
            modifier = 0.6
            # phase 1 (1 to never chaos 0 to always chaos) (no correlation found in test)
            chaos_threshold = 0.7
            # phase 2 (minimum amount of concession) (larger the better, 0.04 in test)
            min_concession_rate = 0.04
        else:
            modifier = self.hyper[0]
            chaos_threshold = self.hyper[1]
            min_concession_rate = self.hyper[2]

        # Get all possible offers

        # Variable init
        assert self.ufun and self.opponent_ufun
        opp_prev_opp_offer_val = float(self.opponent_ufun(self.own_previous_offer))
        own_prev_offer_val = float(self.ufun(self.own_previous_offer))

        # calculate their movement
        opp_prev_offer_val = float(self.opponent_ufun(self.prev_opp_offer))
        opp_current_offer_val = float(self.opponent_ufun(state.current_offer))
        opp_concession = opp_prev_offer_val - opp_current_offer_val

        own_prev_opp_offer_val = float(self.ufun(self.prev_opp_offer))
        own_curr_offer_val = float(self.ufun(state.current_offer))
        opp_concession_for_us = own_prev_opp_offer_val - own_curr_offer_val

        if opp_concession > 0:
            opp_concession *= modifier
        if opp_concession_for_us < 0:
            opp_concession_for_us *= modifier

        # early game basic tit for tat with chaos
        if self.current_phase(state=state) == 0:
            if self.prev_opp_offer is None:  # First offer
                offer_val = max(self.our_offers, key=lambda item: item[0])
                index = offer_val[2]
                self.own_previous_offer = self.rational_outcomes[index]
            elif random.random() > chaos_threshold:  # Chaos action
                offer_val = max(self.our_offers, key=lambda item: item[0])
                index = offer_val[2]
            else:  # tit for tat game
                # see if opponent is nice
                if opp_prev_offer_val - opp_current_offer_val > 0:
                    self.forgiveness_count += 1
                else:
                    self.forgiveness_count = 0
                # if defection for 3 turns in a row forgive (dont move)
                if self.forgiveness_count >= 3:
                    offer_val = self.our_offers[
                        self.closest_point(
                            [L[:2] for L in self.our_offers],
                            (own_prev_offer_val, opp_prev_opp_offer_val),
                        )
                    ]
                    self.forgiveness_count = 0
                else:  # match their movement
                    offer_val = self.our_offers[
                        self.closest_point(
                            [L[:2] for L in self.our_offers],
                            (
                                own_prev_offer_val - opp_concession,
                                opp_prev_opp_offer_val - opp_concession_for_us,
                            ),
                        )
                    ]
                index = offer_val[2]
                self.own_previous_offer = self.rational_outcomes[index]

            self.prev_opp_offer = state.current_offer
            # print(f"Phase 1 took {time.time()-start_time}")
            return self.rational_outcomes[index]
        elif self.current_phase(state=state) == 1:
            # Phase 2 dance on pareto
            # tit for tat game

            # get only the points close to pareto
            if self.phase_2_flag is False:
                self.our_offers = []
                # we want our first offer of phase 2 to be the largest pareto point
                max_vers = 0
                for i in self.pareto_area_points:
                    curr_util = self.ufun(self.rational_outcomes[i[3]])
                    if max_vers == 0:
                        # init the variable
                        max_vers = (i[0], i[1], i[3])
                    if curr_util >= self.reserved_value:
                        # we add it to our available offers since its above our reservation value
                        if curr_util > self.ufun(self.rational_outcomes[max_vers[2]]):
                            # new biggest pareto point
                            max_vers = (i[0], i[1], i[3])
                        self.our_offers.append((i[0], i[1], i[3]))
                if self.our_offers == []:
                    # if no pareto points are above our reserve value, call again to get an answer
                    self.phase_2_flag = True
                    return self.bidding_strategy(state)
                self.phase_2_flag = True
                index = max_vers[2]  # type: ignore
                self.own_previous_offer = self.rational_outcomes[index]

                self.prev_opp_offer = state.current_offer
                # print(f"Phase 2 took {time.time()-start_time}")

                return self.rational_outcomes[index]

            # see if opponent is nice
            if opp_prev_offer_val - opp_current_offer_val > 0:
                self.forgiveness_count += 1
            else:
                self.forgiveness_count = 0

            # calculate their movement history
            if all(i == self.opponent_history[-1] for i in self.opponent_history[-5:]):
                opp_concession += min_concession_rate
                opp_concession_for_us -= min_concession_rate

            # if defection for 3 turns in a row forgive (dont move)
            available_points = [
                offers[:2]
                for offers in self.our_offers
                if offers != self.own_previous_offer
            ]

            if self.forgiveness_count >= 3:
                ideal_point = (own_prev_offer_val, opp_prev_opp_offer_val)
                self.forgiveness_count = 0
            else:
                # match their movement
                ideal_point = (
                    own_prev_offer_val - opp_concession,
                    opp_prev_opp_offer_val - opp_concession_for_us,
                )

            # get the closest to ideal point
            offer_val = self.our_offers[
                self.closest_point(available_points, ideal_point)
            ]
            index = offer_val[2]

            # update local vars
            self.own_previous_offer = self.rational_outcomes[index]
            self.prev_opp_offer = state.current_offer

            return self.rational_outcomes[index]
        elif self.current_phase(state=state) == 2:
            # Aim for nash
            pareto_points = pareto_frontier([self.ufun, self.opponent_ufun])
            # Get final outcome space
            final_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()
                if self.ufun(_) > self.ufun.reserved_value
                and self.opponent_ufun(_) > self.partner_reserved_value
            ]
            if len(final_outcomes) == 0:
                # rare, but happens if RV est is out of wack
                self.partner_reserved_value = 0.25  # Possible temp fix
                final_outcomes = [
                    _
                    for _ in self.nmi.outcome_space.enumerate_or_sample()
                    if self.ufun(_) > self.ufun.reserved_value
                    and self.opponent_ufun(_) > self.partner_reserved_value
                ]
            self.our_offers = []
            for i in final_outcomes:
                self.our_offers.append(
                    (self.ufun(i), self.opponent_ufun(i), final_outcomes.index(i))
                )
            # get nash points
            assert self.ufun and self.opponent_ufun
            nash = nash_points(
                [self.ufun, self.opponent_ufun],  # type: ignore
                pareto_points[0],
                ranges=[(self.reserved_value, 1), (self.partner_reserved_value, 1)],
            )
            temp_opp_rv = self.partner_reserved_value
            while len(nash) == 0 and temp_opp_rv > 0:
                # try to find any nash point if none are found by reducing the opp. RV incrementally
                temp_opp_rv -= 0.01
                nash = nash_points(
                    [self.ufun, self.opponent_ufun],  # type: ignore
                    pareto_points[0],
                    ranges=[(self.reserved_value, 1), (temp_opp_rv, 1)],
                )
            if len(nash) == 0:
                # Go back to modified phase 2
                opp_concession = max(opp_concession, 0)
                opp_concession_for_us = min(opp_concession_for_us, 0)

                offer_val = self.our_offers[
                    self.closest_point(
                        [offer[:2] for offer in self.our_offers],
                        (
                            own_prev_offer_val - opp_concession,
                            opp_prev_opp_offer_val - opp_concession_for_us,
                        ),
                    )
                ]

                index = offer_val[2]

                self.own_previous_offer = self.rational_outcomes[index]
                self.prev_opp_offer = state.current_offer
                return self.rational_outcomes[index]
            else:
                # get number of turns left

                nsteps__ = (
                    self.nmi.n_steps
                    if self.nmi.n_steps
                    else int(
                        (self.nmi.state.time + 1e-6)
                        / (self.nmi.state.relative_time + 1e-6)
                        + 0.5
                    )
                )
                if nsteps__ is None:
                    num_of_turns_left = min(
                        state.step * self.nmi.time_limit / state.time,
                        self.nmi.n_outcomes,
                    )
                else:
                    num_of_turns_left = nsteps__ - state.step

                # move along pareto towards the nash to end there in the last step
                move_distance = float(self.ufun(self.own_previous_offer))
                # nudge towards the nash point
                nudge = np.abs(nash[0][0][0] - move_distance) / num_of_turns_left

                if move_distance > nash[0][0][0]:
                    move_distance -= nudge

                elif move_distance < nash[0][0][0]:
                    move_distance += nudge

                inverse_move_distance = float(
                    self.opponent_ufun(self.own_previous_offer)
                )
                nudge = (
                    np.abs(nash[0][0][1] - inverse_move_distance) / num_of_turns_left
                )

                if inverse_move_distance > nash[0][0][1]:
                    inverse_move_distance -= nudge

                elif inverse_move_distance < nash[0][0][1]:
                    inverse_move_distance += nudge

                offer_val = self.closest_point(
                    [offer[:2] for offer in self.our_offers],
                    (move_distance, inverse_move_distance),
                )
                return final_outcomes[offer_val]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        self.current_step += 1
        if self.current_step == 1:
            self.new_proposal = self.ufun(self.initial_proposal)

        if self.current_step == 2:
            # initialize rv probabilities with bias towards 0
            self.probability_rv_opp = [1 / self.num_of_hyp] * self.num_of_hyp
            self.probability_rv_opp[0] += 0.4
            self.probability_rv_opp = np.array(self.probability_rv_opp) / np.sum(
                self.probability_rv_opp
            )
            self.probability_rv_opp = self.probability_rv_opp.tolist()

        if self.current_step > 3:
            # compute opponent lambda using (7)
            log_base = self.current_step / (self.current_step - 1)

            opp_offer_val = float(self.opponent_ufun(self.opp_offer))
            opp_init_val = float(self.opponent_ufun(self.opponents_initial_proposal))
            opp_prev_offer_val = float(self.opponent_ufun(self.prev_opp_offer))

            if opp_offer_val > opp_init_val:
                opp_offer_val = opp_init_val
            if opp_offer_val > opp_prev_offer_val:
                opp_offer_val = opp_prev_offer_val
            if opp_prev_offer_val > opp_init_val:
                opp_prev_offer_val = opp_init_val

            ratio = opp_offer_val - opp_init_val
            if (opp_prev_offer_val - opp_init_val == 0) or (
                opp_prev_offer_val - opp_offer_val == 0
            ):
                lambda_opp = 10**4
            else:
                ratio /= opp_prev_offer_val - opp_init_val  # + error_term
                lambda_opp = np.log(np.max([ratio, 1])) / np.log(log_base)

            if ratio == 0:
                lambda_opp = 10**4
            if lambda_opp > 2:
                lambda_opp = min([lambda_opp, 10**4])

            # compute discount factor using (10)
            if self.nmi.n_steps is not None:
                discount_ratio = np.power(
                    self.current_step / self.nmi.n_steps, lambda_opp
                )
            else:
                discount_ratio = np.power(state.relative_time, lambda_opp)
            if discount_ratio < 1e-6 or math.isnan(discount_ratio):
                discount_ratio = 1e-6

            # Compute Prob(proposal opp at t | res val of opp i ) using (9) for all i

            if opp_offer_val == opp_init_val:
                # Define parameters
                mean = opp_offer_val
                std_dev = (
                    0.1  # Adjust std_dev to control the spread of the distribution
                )
                num_buckets = self.num_of_hyp

                # Calculate bucket boundaries
                bucket_boundaries = np.linspace(0, 1, num_buckets + 1)

                # Calculate probabilities for each bucket
                probabilities = np.zeros(num_buckets)
                for i in range(num_buckets):
                    lower_bound = bucket_boundaries[i]
                    upper_bound = bucket_boundaries[i + 1]
                    probability = scipy.stats.norm.cdf(  # type: ignore
                        upper_bound, mean, std_dev
                    ) - scipy.stats.norm.cdf(lower_bound, mean, std_dev)  # type: ignore
                    probabilities[i] = probability

                probabilities /= sum(probabilities)
                probability_proposal_given_rv = probabilities
            else:
                probability_proposal_given_rv = np.array([])
                for opponents_reserved_value in self.opponents_reserved_value:
                    if discount_ratio == 1e-6:
                        adjustment = 0
                    else:
                        adjustment = np.max(
                            np.abs(
                                (opp_init_val - opp_offer_val) / discount_ratio
                                - np.abs(opponents_reserved_value - opp_init_val)
                            ),
                            0,
                        )

                    chance_adjustment = 1 - np.abs(
                        adjustment / ((opp_init_val - opp_offer_val) / discount_ratio)
                    )
                    probability_proposal_given_rv = np.append(
                        probability_proposal_given_rv, chance_adjustment
                    )

            probability_proposal_given_rv /= sum(probability_proposal_given_rv)
            if (np.array(probability_proposal_given_rv) < 0).any():
                probability_proposal_given_rv += -1 * np.min(
                    probability_proposal_given_rv
                )
                probability_proposal_given_rv /= np.sum(probability_proposal_given_rv)
            else:
                probability_proposal_given_rv /= np.sum(probability_proposal_given_rv)

            # Compute P (RP i opp | P t opp) using (11) for all i
            prob_sum = 0
            # eliminate RV that are greater than their offer
            for i in range(self.num_of_hyp):
                if opp_offer_val < self.opponents_reserved_value[i]:
                    probability_proposal_given_rv[i] = 0

            probability_proposal_given_rv /= np.sum(probability_proposal_given_rv)
            for i in range(self.num_of_hyp):
                prob_sum += (
                    self.probability_rv_opp[i] * probability_proposal_given_rv[i]
                )
            # compute all rv probabilities given proposal
            if prob_sum == 0:
                prob_sum = 1
            probability_rv_given_proposal = (
                np.array(
                    # BUG still gives runtime errors not a crash tho
                    np.multiply(self.probability_rv_opp, probability_proposal_given_rv)
                )
                / prob_sum
            )

            # Set P_{t-1}(RP opp i) = P(rv opp i | prob opp t-1)
            self.probability_rv_opp = probability_rv_given_proposal.tolist()

            # compute RP opp at t using (12)
            self.partner_reserved_value = np.dot(
                probability_rv_given_proposal, self.opponents_reserved_value
            )

    def update_partner_reserved_value_2(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        self.current_step += 1
        if self.current_step == 1:
            self.new_proposal = self.ufun(self.initial_proposal)
            # initialize rv probabilities with possible bias
            self.probability_prev = [1 / self.num_of_hyp] * self.num_of_hyp
            self.probability_prev[0] += 0.3
            self.probability_prev = np.array(self.probability_prev) / np.sum(
                self.probability_prev
            )
            return

        opp_offer_val = float(self.opponent_ufun(self.opp_offer))
        mean = 0.6 * opp_offer_val

        # Adjust std_dev to control the spread of the distribution
        # making it dynamic might not be optimal, need to test
        if self.nmi.n_steps is not None:
            std_dev = np.max(
                np.array(
                    [
                        (self.nmi.n_steps - self.current_step) / self.nmi.n_steps - 0.6,
                        0.1,
                    ]
                )
            )
        else:
            std_dev = np.max(np.array([(1 - state.relative_time) - 0.6, 0.1]))

        # Calculate bucket boundaries
        bucket_boundaries = np.linspace(0, 1, self.num_of_hyp + 1)

        # Calculate probabilities for each bucket
        probabilities = copy.deepcopy(self.probability_prev)

        index = 0
        for index, value in enumerate(probabilities):
            if opp_offer_val < self.opponents_reserved_value[index]:
                # it should never be below, and the ones 20% closest to it as a percentage of number of hypothesis
                # will also be penalized in proportion to how close they're to caught val
                for x in range(int(np.round(self.num_of_hyp / 10))):
                    if index >= x:
                        # the penalty is maximal near to the nulled out value, but diminishes
                        probabilities[index - x] /= self.num_of_hyp / 10 - x
                    else:
                        # no further we can penalize without making an error
                        break
                probabilities[index] = 0

        scale_additions = scipy.stats.norm.cdf(1, mean, std_dev)  # type: ignore
        # scale_additions = 1
        for i in range(self.num_of_hyp):
            if probabilities[index] == 0:
                break
            lower_bound = bucket_boundaries[i]
            upper_bound = bucket_boundaries[i + 1]

            probability = scipy.stats.norm.cdf(  # type: ignore
                upper_bound, mean, std_dev
            ) - scipy.stats.norm.cdf(lower_bound, mean, std_dev)  # type: ignore
            probability /= scale_additions
            probabilities[i] += probability

        if np.sum(probabilities) <= 0:
            # reset the eval completely if probs get negative or zero, should never happen.
            self.probability_prev = [1 / self.num_of_hyp] * self.num_of_hyp
            self.probability_prev[0] += 0.3
            self.probability_prev = np.array(self.probability_prev) / np.sum(
                self.probability_prev
            )
            probabilities = 1
        self.probability_prev /= np.sum(probabilities)

        # final calc
        self.partner_reserved_value = np.dot(
            self.probability_prev, self.opponents_reserved_value
        )

        # check for suspiciously high RVs
        if self.partner_reserved_value > 0.6:
            # we are being duped by a hardhead, reset the estimator and set the RV to 0.25
            self.probability_prev = [1 / self.num_of_hyp] * self.num_of_hyp
            self.probability_prev[0] += 0.3
            self.probability_prev = np.array(self.probability_prev) / np.sum(
                self.probability_prev
            )
            self.partner_reserved_value = 0.25

        # if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
