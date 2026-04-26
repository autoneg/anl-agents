from scipy.optimize import curve_fit
from negmas.sao import SAOResponse
from negmas import ResponseType
from copy import deepcopy
from anl.anl2024.negotiators.base import ANLNegotiator
from negmas import Outcome
import numpy as np

__all__ = ["HardChaosNegotiator"]


def aspiration_function(t, mx, rv, e):
    """A monotonically decreasing curve starting at mx (t=0) and ending at rv (t=1)"""
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class HardChaosNegotiator(ANLNegotiator):
    def __init__(
        self, *args, e=8.0, **kwargs
    ):  # Increase the exponent to make the negotiation tougher
        super().__init__(*args, **kwargs)
        self.e = e
        self._rational = []
        self.last_offer = None
        self.opponent_times = []
        self.opponent_utilities = []
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.private_info["opponent_ufun"] = deepcopy(self.opponent_ufun)
        self.best_offer__ = self.ufun.best()

    def update_opponent_model(self, offer, relative_time):
        assert self.opponent_ufun is not None
        if offer is not None:
            self.opponent_utilities.append(float(self.opponent_ufun(offer)))
            self.opponent_times.append(relative_time)

    def update_reserved_value(self):
        assert self.opponent_ufun is not None
        if (
            not self.opponent_utilities or len(self.opponent_utilities) < 3
        ):  # Require more data points for a reliable estimate
            return
        try:
            optimal_vals, _ = curve_fit(
                lambda t, e, rv: (max(self.opponent_utilities) - rv)
                * (1.0 - np.power(t, e))
                + rv * 0.9,  # Adjust for a potentially lower reservation value
                self.opponent_times,
                self.opponent_utilities,
                bounds=(
                    (0.5, min(self.opponent_utilities) * 0.9),
                    (8.0, max(self.opponent_utilities)),
                ),  # Tighten bounds based on a stronger assumption
            )
            self.opponent_ufun.reserved_value = optimal_vals[1]
        except Exception:
            pass

    def aspiration_function(self, t):
        return (
            1.0 - np.power(t, self.e)
        ) + 0.1  # Start with a higher aspiration and end lower

    def generate_offer(self, relative_time) -> Outcome | None:
        assert self.ufun is not None
        if not self._rational:
            self.populate_rational_outcomes()
        aspiration_level = (
            self.aspiration_function(relative_time) + self.ufun.reserved_value
        )
        for outcome, my_util, opp_util in self._rational:
            if my_util >= aspiration_level:
                return outcome
        return self.best_offer__

    def populate_rational_outcomes(self):
        assert self.opponent_ufun is not None
        assert self.ufun is not None
        self._rational = [
            (outcome, self.ufun(outcome), self.opponent_ufun(outcome))
            for outcome in self.nmi.outcome_space.enumerate()  # type: ignore
            if self.ufun(outcome) > self.ufun.reserved_value
        ]
        self._rational.sort(key=lambda x: -x[2])  # type: ignore

    def is_acceptable(self, offer, relative_time) -> bool:
        assert self.ufun is not None
        best_outcome_utility = self.ufun(self.best_offer__)
        current_aspiration = (
            self.aspiration_function(relative_time)
            * (best_outcome_utility - self.ufun.reserved_value)
            + self.ufun.reserved_value
        )
        return (
            self.ufun(offer) >= current_aspiration + 0.05
        )  # Require a slight improvement over the aspiration level

    def __call__(self, state):
        self.update_opponent_model(state.current_offer, state.relative_time)
        self.update_reserved_value()
        if self.is_acceptable(state.current_offer, state.relative_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self.generate_offer(state.relative_time)
            )
