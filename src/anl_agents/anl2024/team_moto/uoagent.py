"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Team moto
*Authors* Hirotada Matsumoto: h-matsumoto@katfuji.lab.tuat.ac.jp
*Affiliation* Tokyo University of Agriculture and Technology
*Country* Japan

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
ver 1.1.1
GRAND_FINAL
"""

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["UOAgent"]


class UOAgent(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()

    outcome_list = []

    partner_reserved_value = 1.0
    step = 0
    relative_time = 0.0
    under = 1.0

    def getNearestValue(self, list, num):
        """
        概要: リストからある値に最も近い値を返却する関数
        @param list: データ配列
        @param num: 対象値
        @return 対象値に最も近い値
        """

        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        idx = np.abs(np.asarray(list) - num).argmin()
        return idx

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
        """
        if self.ufun.reserved_value <= 0.4:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()
                if self.ufun.reserved_value + 0.35 < self.ufun(_)
            ]

        if not self.rational_outcomes:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
                if self.ufun(_) > self.ufun.reserved_value and self.ufun(_) > 0.75
            ]
        """

        if self.ufun.reserved_value <= 0.4:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()
                if self.ufun.reserved_value + 0.40 <= self.ufun(_)
            ]

        else:
            # if not self.rational_outcomes:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
                if self.ufun(_) >= self.ufun.reserved_value and self.ufun(_) >= 0.80
            ]

        if not self.rational_outcomes:
            self.rational_outcomes = [
                _
                for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
                if self.ufun(_) >= self.ufun.reserved_value
            ]

        self.outcome_list = []
        self.step = 0
        self.relative_time = 0.0

        if self.ufun.reserved_value > 0.85:
            self.under = self.ufun.reserved_value
        else:
            self.under = 0.85

        assert self.ufun is not None and self.opponent_ufun is not None
        for _ in self.rational_outcomes:
            self.outcome_list.append(
                (
                    _,
                    float(self.ufun(_)) + float(self.opponent_ufun(_)),
                    float(self.ufun(_)),
                    float(self.opponent_ufun(_)),
                )
            )

        self.outcome_list = sorted(self.outcome_list, key=lambda x: x[2], reverse=True)

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        # self.partner_reserved_value = self.ufun.reserved_value
        self.partner_reserved_value = 1.0

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
        self.relative_time = state.relative_time

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

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

        if self.nmi.n_steps is not None and self.nmi.n_steps - self.step == 1:
            if self.ufun(offer) > self.ufun.reserved_value:
                return True

        if self.ufun(offer) > (
            1.0 - (1.0 - self.ufun.reserved_value) * (self.relative_time**50)
        ):
            self.step = self.step + 1
            return True

        self.step = self.step + 1
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # offer, social_utility, my_utility, opponent_utility = self.outcome_list[0]

        if self.relative_time > 0.975 or (
            self.nmi.n_steps is not None and self.nmi.n_steps - self.step == 1
        ):
            last_list = sorted(self.outcome_list, key=lambda x: x[3], reverse=True)
            listing = map(lambda x: x[3], last_list)
            ufunlist = list(listing)
            num1 = self.getNearestValue(ufunlist, self.partner_reserved_value + 0.02)
            num2 = self.getNearestValue(ufunlist, self.partner_reserved_value)
            offer_list = sorted(
                last_list[num1 : num2 + 1], key=lambda x: (x[1], x[2]), reverse=True
            )
            offer, social_utility, my_utility, opponent_utility = offer_list[0]
        else:
            """
            listing = map(lambda x: x[2], self.outcome_list)
            num1 = self.getNearestValue(ufunlist, rand)
            num2 = self.getNearestValue(ufunlist, rand-0.02)
            offer_list = sorted(self.outcome_list[num1:num2+1], key=lambda x:(x[1], x[2]), reverse=True)
            offer, social_utility, my_utility, opponent_utility = offer_list[0]
            """
            nsteps__ = (
                self.nmi.n_steps
                if self.nmi.n_steps
                else int(
                    (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                    + 0.5
                )
            )
            # assert self.nmi.n_steps is not None
            value = 1.0 - (1.0 - self.under) * ((self.step / nsteps__) ** 5)
            listing = map(lambda x: x[2], self.outcome_list)
            ufunlist = list(listing)
            num1 = self.getNearestValue(ufunlist, value + 0.02)
            num2 = self.getNearestValue(ufunlist, value - 0.02)
            offer_list = sorted(
                self.outcome_list[num1 : num2 + 1],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )

            offer, social_utility, my_utility, opponent_utility = offer_list[0]
        return offer

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if (
            self.opponent_ufun(offer) < self.partner_reserved_value
            and self.opponent_ufun(offer) >= 0.01
        ):
            self.partner_reserved_value = float(self.opponent_ufun(offer))


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(UOAgent, small=False)
