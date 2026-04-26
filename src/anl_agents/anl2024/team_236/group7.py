"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["Group7"]

# burada rakibin yaptığı tekliflerde yine rakibin utility'si saklanacak


class Group7(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_offer_utility_list = []
        self.rational_outcomes = tuple()
        self.partner_reserved_value = 0
        self.last_bid = None

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
        assert self.ufun

        offer = state.current_offer

        current_offer_utility = self.ufun(offer)
        opponent_offer_utility = self.opponent_ufun(offer)

        # first phase: accept only if we have better utility than the opponent and if its higher than our reserved value
        if state.relative_time <= 0.33:
            return (
                current_offer_utility > opponent_offer_utility
                and current_offer_utility > (1.5 * self.ufun.reserved_value)
            )

        # second phase:  accept if we have higher utiliy than our reserved value and we have better utiliy than our opponent
        elif state.relative_time <= 0.66:
            if (
                current_offer_utility >= 1.0 * self.ufun.reserved_value
                and opponent_offer_utility < current_offer_utility
            ):
                return True

        # late phase: Accept if we have better offer on our hand then what we'd likely to get with next offer
        elif state.relative_time <= 0.95:
            if self.is_next_bid_worse(offer, state):
                return True

        else:
            return self.ufun(offer) > self.ufun.reserved_value

        return False

    def is_next_bid_worse(self, current_offer, state: SAOState) -> bool:
        next_bid = self.bidding_strategy(state)

        next_bid_utility = self.ufun(next_bid)
        current_offer_utility = self.ufun(current_offer)

        # if the next bids utility is lower than the current offers utility then it's worse
        return next_bid_utility < current_offer_utility

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        assert self.ufun

        relative_time = state.relative_time  # bu anladığım kadarıyla 0 ile 1 arasında

        # bunları rastgele verdim değiştirip test edebiliriz
        phase_1_end = 0.33
        phase_2_end = 0.66
        # arada hata verdiği için bunu ekledim
        if len(self.rational_outcomes) == 0:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the current phase
        if relative_time <= phase_1_end:
            phase = 1
        elif phase_1_end < relative_time <= phase_2_end:
            phase = 2
        else:
            phase = 3

        if phase == 1:
            # Offer bid with maximum utility for us and minimum for the opponent
            return max(
                self.rational_outcomes,
                key=lambda x: self.ufun(x) - self.opponent_ufun(x),
            )

        elif phase == 2:
            # Offer bid by conceding from our utility and increasing the opponent's
            return self.conceding_bid2(state)

        # phase3 checks opponents behaviour
        else:
            if self.is_opponent_conceding(state):
                return self.conceding_bid(state, favor_opponent=True)
            else:
                return self.conceding_bid(state, favor_opponent=False)

    def conceding_bid2(self, state: SAOState) -> Outcome:
        relative_time = state.relative_time
        concession_rate = self.calculate_concession_rate(relative_time)

        # Sort rational outcomes by our utility, descending
        sorted_outcomes = sorted(self.rational_outcomes, key=self.ufun, reverse=True)

        # The index at which we start our concession
        start_index = int(concession_rate * len(sorted_outcomes))

        result = sorted_outcomes[start_index]
        # opponent reservation değerini burada kullanıyoruz unutmayalım bunu!!!
        for outcome in sorted_outcomes[start_index:]:
            if self.ufun(outcome) > self.opponent_ufun(outcome):
                result = outcome
                break
        # If no outcome, bir şey değişmemiş oluyor işte yine result result'tır

        for outcome in sorted_outcomes:  # nice move yaptırıcam
            # bizim utility'miz yüzde beşten az değişcek fakat rakibinki artacak
            if (self.ufun(outcome) > 0.95 * self.ufun(result)) and (
                self.opponent_ufun(outcome) > self.opponent_ufun(result)
            ):
                result = outcome
        return result

    def calculate_concession_rate(self, relative_time: float) -> float:
        return relative_time / 10  # This is a simple linear concession rate

    def is_opponent_conceding(self, state: SAOState) -> bool:
        if state.current_offer is None:
            return False  # No offer yet

        current_opponent_utility = self.opponent_ufun(state.current_offer)

        if len(self.opponent_offer_utility_list) >= 5:
            average_opponent_utility = 0
            amount_regarded_in_average = int(
                len(self.opponent_offer_utility_list) * 0.3
            )  # ilk yüzde yetmişini ortalama hesabına dahil etmeyeceğim.

            for offer_utility in self.opponent_offer_utility_list[
                :amount_regarded_in_average
            ]:
                average_opponent_utility = average_opponent_utility + offer_utility
            average_opponent_utility = (
                average_opponent_utility / amount_regarded_in_average
            )  # şimdiye kadar rakibin sunduğu tekliflerde onun utility'sinin ortalaması

            for offer_utility in self.opponent_offer_utility_list[
                -5:
            ][
                ::-1
            ]:  # son beş teklifteki davranışa bakıyorum, herhangi biri istediğim aralıktaysa taviz veriyor sayacağım
                if (
                    offer_utility > current_opponent_utility
                    or offer_utility < average_opponent_utility * 0.9
                ):
                    return True  # yani, eğer şu ankinden az bir teklifte bulunuyorsa veya şimdiye kadarki tekliflerinin ortalamasının yüzde doksanından az bir teklifte bulunuyrosa taviz veriyordur.
        return False

    def conceding_bid(self, state: SAOState, favor_opponent: bool = False) -> Outcome:
        relative_time = state.relative_time

        concession_rate = self.calculate_concession_rate(relative_time)

        sorted_by_our_utility = sorted(
            self.rational_outcomes, key=self.ufun, reverse=True
        )

        sorted_by_their_utility = sorted(
            self.rational_outcomes, key=self.opponent_ufun, reverse=True
        )

        if favor_opponent:
            # we are trying to find a bid which is good for the opponent and we are willing to offer

            best_for_us = sorted_by_our_utility[
                int(concession_rate * len(sorted_by_our_utility))
            ]

            for outcome in sorted_by_their_utility:
                if (
                    self.ufun(outcome) >= self.ufun(best_for_us)
                    and self.opponent_ufun(outcome) > self.partner_reserved_value
                ):
                    best_for_us = outcome
                    break
            # opponent da bizim gibiyse diye bunu koydum ama bu olmasa daha iyi de olabilir emin değilim "self.opponent_ufun(outcome) > self.ufun(outcome)"
            # If we can't find such an outcome, we offer the best for us

            return best_for_us

        else:
            # When not favoring the opponent, we search for a bid that gives us a decreasing utility
            # but still above the opponent's utility. We reduce our utility as time goes by.
            for outcome in sorted_by_our_utility:
                if self.opponent_ufun(
                    outcome
                ) >= self.partner_reserved_value and self.ufun(
                    outcome
                ) > self.opponent_ufun(outcome):
                    return outcome
            # If no such outcome is found, we fall back to the least concessionary bid that respects the opponent's reservation value

            return sorted_by_our_utility[-1]  # listedeki son eleman demek

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """

        assert self.ufun and self.opponent_ufun

        self.partner_reserved_value = self.reserved_value

        offer = state.current_offer

        self.opponent_offer_utility_list.append(
            float(self.opponent_ufun(offer))
        )  # listeye son teklifin rakipteki utility'sini eklemiş olduk

        if offer is not None:
            current_opponent_utility = self.opponent_ufun(offer)

            if current_opponent_utility > self.partner_reserved_value:
                # If the opponent's current offer is higher, slightly increase the reserved value

                self.partner_reserved_value = self.partner_reserved_value * 1.05
            else:
                # If the opponent's current offer is lower, slightly decrease the reserved value
                self.partner_reserved_value = self.partner_reserved_value * 0.8


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group7, small=False)
