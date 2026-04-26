import random

import numpy as np
from negmas.sao import ResponseType, SAOResponse, SAOState

from anl.anl2024.negotiators.base import ANLNegotiator
from scipy.optimize import curve_fit

__all__ = ["Group6"]


class Group6(ANLNegotiator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.offer_history = []
        self.time_history = []
        self.utility_history = []

    def on_preferences_changed(self, changes) -> None:
        if self.ufun is None:
            return
        self.best_offer__ = self.ufun.best()
        self.rational_outcomes = [
            o
            for o in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(o) > self.ufun.reserved_value
        ]

    def opponent_model(self):
        if not self.offer_history:
            return None
        # Estimate the opponent's strategy (simple curve fitting here)
        try:
            params, _ = curve_fit(
                self.aspiration_function,
                self.time_history,
                self.utility_history,
                maxfev=1000,
            )
            return params
        except Exception:
            return None

    def aspiration_function(self, t, max_util, min_util, exp):
        """Aspiration level adjusts over time from max_util to min_util."""
        return (max_util - min_util) * (1 - np.power(t, exp)) + min_util

    def acceptance_strategy(self, current_offer, current_time):
        if not current_offer:
            return False
        assert self.ufun is not None and self.opponent_ufun is not None
        offer_utility = self.ufun(current_offer)
        # Dynamic aspiration level based on negotiation time
        aspiration_level = self.aspiration_function(
            current_time, self.ufun(self.best_offer__), self.ufun.reserved_value, 1.2
        )
        return offer_utility >= aspiration_level

    def bidding_strategy(self, current_time):
        aspiration_level = self.aspiration_function(
            current_time, self.ufun(self.best_offer__), self.ufun.reserved_value, 1.2
        )
        potential_outcomes = [
            o for o in self.rational_outcomes if self.ufun(o) >= aspiration_level
        ]
        if not potential_outcomes:
            potential_outcomes = self.rational_outcomes
        return random.choice(potential_outcomes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        current_offer = state.current_offer
        current_time = state.relative_time
        if current_offer:
            self.offer_history.append(current_offer)
            self.time_history.append(current_time)
            self.utility_history.append(self.ufun(current_offer))

        if self.acceptance_strategy(current_offer, current_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, current_offer)
        counter_offer = self.bidding_strategy(current_time)
        return SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
