"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import math
from copy import deepcopy
import random

import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState


__all__ = ["TAKAgent"]


class TAKAgent(ANLNegotiator):
    rational_outcomes = tuple()
    negotiation_round: int = 0
    negotiation_duration: int = 0

    partner_reserved_value = 0

    current_bid = None
    my_bids = np.array([])

    opponent_utilities = np.array([])
    opponent_concession_rates = np.array([])

    threshold = 0
    previous_offer = None
    concession_rate = 0

    nash_bid = None
    nash_offer_times = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threshold = 2 * self.reserved_value

        self.opponent_utilities = np.array([])
        self.opponent_concession_rates = np.array([])

        self.previous_offer = None

        self.concession_rate = 0

    def opponent_concession_rate(self):
        window_size = 5
        average_concession_rate = 0
        if len(self.opponent_concession_rates) >= window_size:
            recent_concession_rates = self.opponent_concession_rates[-window_size:]
            average_concession_rate = sum(recent_concession_rates) / len(
                recent_concession_rates
            )
        return average_concession_rate

    def opponent_is_conceding_fast(self, our_concession_rate):
        opponent_concession_rate = self.opponent_concession_rate()
        return opponent_concession_rate > our_concession_rate

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        self.private_info["opponent_ufun"] = deepcopy(self.opponent_ufun)

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]

        self.partner_reserved_value = self.ufun.reserved_value

        assert self.opponent_ufun is not None
        self.opponent_ufun.reserved_value = self.ufun.reserved_value

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.negotiation_duration = nsteps__

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

        if state.step == 0:
            self.nash_bid = self.find_nash_product_bid()

        self.update_partner_reserved_value(state)
        self.negotiation_round += 1

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def find_nash_product_bid(self) -> Outcome:
        best_bid = None
        max_product = float("-inf")

        for bid in self.rational_outcomes:
            agent_utility = self.ufun(bid)
            opponent_utility = self.opponent_ufun(bid)
            product = agent_utility * opponent_utility
            if product > max_product:
                max_product = product
                best_bid = bid
        return best_bid

    def exp_threshold(self, initial_threshold, current_round, total_rounds):
        start_rate = 0.001
        end_rate = 0.0005
        min_threshold = self.reserved_value
        decreasing_rate = (
            start_rate + (end_rate - start_rate) * current_round / total_rounds
        )
        current_threshold = initial_threshold * math.exp(
            -decreasing_rate * current_round
        )
        return max(min_threshold, current_threshold)

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun
        current_offer = state.current_offer
        remaining_rounds = self.negotiation_duration - self.negotiation_round

        if state.step == 0:
            self.threshold = min(0.99, self.ufun(self.nash_bid) * 1.5)

        if state.step > 5:
            ocr = self.opponent_concession_rate()
            if ocr < 0.001:
                self.threshold = self.threshold
            else:
                self.threshold = self.exp_threshold(
                    self.threshold, self.negotiation_round, self.negotiation_duration
                )
        else:
            self.threshold = self.exp_threshold(
                self.threshold, self.negotiation_round, self.negotiation_duration
            )

        if self.ufun(current_offer) > self.threshold:
            return True

        if self.ufun(current_offer) > self.ufun(self.current_bid) and state.step > 0:
            return True

        elif remaining_rounds < 2 and self.ufun(current_offer) > self.reserved_value:
            return True

        else:
            return False

    def update_my_history(self):
        self.my_bids = np.append(self.my_bids, self.current_bid)

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        if not self.rational_outcomes:
            return None
        else:
            num_bids = state.step
            if num_bids == 0:
                self.current_bid = max(self.rational_outcomes, key=self.ufun)
                self.update_my_history()
                return self.current_bid

            if num_bids <= 5:
                self.concession_rate = 0.02
                current_utility = self.ufun(self.current_bid) - (
                    self.ufun(self.current_bid) * self.concession_rate
                )
                found_bid = min(
                    self.rational_outcomes,
                    key=lambda x: abs(self.ufun(x) - current_utility),
                )
                self.current_bid = found_bid
                self.update_my_history()
                return self.current_bid

            if random.random() < 0.1:
                valid_offers = [
                    offer
                    for offer in self.rational_outcomes
                    if self.ufun(offer) > self.threshold
                ]
                if valid_offers:
                    rand_offer = random.choice(valid_offers)
                    return rand_offer

            else:
                opponent_concession_rate = self.opponent_concession_rate()

                if self.opponent_is_conceding_fast(self.concession_rate):
                    self.concession_rate = opponent_concession_rate
                else:
                    self.concession_rate = opponent_concession_rate * 0.8

                my_bid_utility = self.ufun(self.current_bid) * (
                    1 - self.concession_rate
                )
                closest_bid = min(
                    self.rational_outcomes,
                    key=lambda x: abs(self.ufun(x) - my_bid_utility),
                )

                if (
                    self.ufun(closest_bid) < self.ufun(self.nash_bid)
                    and self.nash_offer_times > 0
                ):
                    self.nash_offer_times -= 1
                    self.current_bid = self.nash_bid
                    self.update_my_history()
                    return self.current_bid

                if opponent_concession_rate < 0.005:
                    self.concession_rate += 0.001
                    new_utility = self.ufun(self.current_bid) * (
                        1 - self.concession_rate
                    )
                    new_closest_bid = min(
                        self.rational_outcomes,
                        key=lambda x: abs(self.ufun(x) - new_utility),
                    )
                    self.current_bid = new_closest_bid
                    self.update_my_history()
                    return self.current_bid

                self.current_bid = closest_bid
                self.update_my_history()
                return self.current_bid

    def update_partner_reserved_value(self, state: SAOState) -> None:
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if offer is None:
            return False

        opponent_utility = float(self.opponent_ufun(offer))
        self.opponent_utilities = np.append(self.opponent_utilities, opponent_utility)

        if len(self.opponent_utilities) > 1:
            previous_utility = self.opponent_utilities[-2]
            current_utility = self.opponent_utilities[-1]
            concession_rate = (previous_utility - current_utility) / previous_utility
            self.opponent_concession_rates = np.append(
                self.opponent_concession_rates, concession_rate
            )

        reservation_val_of_our_agent = self.ufun.reserved_value
        prior_mean = reservation_val_of_our_agent
        prior_std = np.std(self.opponent_utilities)

        if prior_std == 0:
            self.partner_reserved_value = reservation_val_of_our_agent

        else:
            likelihood_std = 1.0
            posterior_precision = 1.0 / (
                1.0 / (prior_std**2) + 1.0 / (likelihood_std**2)
            )
            posterior_mean = posterior_precision * (
                prior_mean / (prior_std**2) + opponent_utility / (likelihood_std**2)
            )
            self.partner_reserved_value = posterior_mean

        self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]
