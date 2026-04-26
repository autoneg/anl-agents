from negmas.common import PreferencesChange
from copy import deepcopy

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from negmas.sao import SAOResponse
from negmas import Outcome, ResponseType
from negmas.preferences import nash_points, pareto_frontier
from scipy.optimize import curve_fit

__all__ = ["CARCAgent"]


def aspiration_function(t, mx, rv, e, c):
    return (mx - rv) * (1.0 - c * np.power(t, e)) + rv


class CARCAgent(ANLNegotiator):
    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 5.0,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        **kwargs,
    ):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.e = e
        self.min_unique_utilities = min_unique_utilities
        self.stochasticity = stochasticity
        self.opponent_times: list[float] = []
        self.opponent_utilities: list[float] = []
        self._past_oppnent_rv = 0.0
        self._rational: list[tuple[float, float, Outcome]] = []
        self._type_name = "CARCAgent"
        self._enable_logging = enable_logging
        self._nash_util = 0.0
        self._opponent_nash_util = 0.0

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        assert (
            self.ufun is not None
            and self.opponent_ufun is not None
            and self.ufun.outcome_space is not None
        )
        self.private_info["opponent_ufun"] = deepcopy(self.opponent_ufun)
        ufuns = (self.ufun, self.opponent_ufun)
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        frontier_utils, _ = pareto_frontier(ufuns, outcomes)  # type: ignore
        nash_point = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash_point:
            self._nash_util = nash_point[0][0][0]
            self._opponent_nash_util = nash_point[0][0][1]
        else:
            self._nash_util = (self.ufun.minmax()[0] + self.ufun.minmax()[1]) / 2.0
            self._opponent_nash_util = (
                self.opponent_ufun.minmax()[0] + self.opponent_ufun.minmax()[1]
            ) / 2.0

        self.opponent_ufun.reserved_value = self.ufun.reserved_value

        self._rational = sorted(
            [
                (my_util, opp_util, _)
                for _ in self.nmi.outcome_space.enumerate_or_sample(
                    levels=10, max_cardinality=100_000
                )
                if (my_util := float(self.ufun(_)))
                >= max(self.ufun.reserved_value, self._nash_util * 0.8)
                and (opp_util := float(self.opponent_ufun(_)))
                <= max(self.opponent_ufun.reserved_value, self._opponent_nash_util)
                * 1.2
            ],
        )
        self.best_offer__ = self.ufun.best()

    def __call__(self, state):
        # update the opponent reserved value in self.opponent_ufun
        self.update_reserved_value(state.current_offer, state.relative_time)
        # run the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state.current_offer, state.relative_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        # call the offering strategy
        return SAOResponse(
            ResponseType.REJECT_OFFER, self.generate_offer(state.relative_time)
        )

    def generate_offer(self, relative_time) -> Outcome:
        if not self._rational:
            return self.best_offer__

        asp = aspiration_function(relative_time, 1.0, 0.0, 1.0, 1.0)
        asp1 = aspiration_function(relative_time, 1.0, 0.0, 2.0, 0.8)
        max_rational = len(self._rational) - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))
        max_indx = max(0, min(max_rational, int(asp1 * max_rational)))
        indx = (
            np.random.randint(min_indx, max_indx) if max_indx > min_indx else max_indx
        )
        outcome = self._rational[indx][-1]
        # if relative_time > 0.3:
        #     assert self.opponent_ufun is not None
        #     if self.opponent_ufun(outcome) < self.opponent_ufun.reserved_value:
        #         self.generate_offer(relative_time)
        return outcome

    def is_acceptable(self, offer, relative_time) -> bool:
        """The acceptance strategy"""
        # If there is no offer, there is nothing to accept
        if offer is None or self._rational is None:
            return False

        asp = aspiration_function(relative_time, 1.0, 0.0, 1.0, 1.0)
        max_rational = len(self._rational) - 1
        indx = max(0, min(max_rational, int(asp * max_rational)))
        outcome = self._rational[indx][-1]

        assert self.ufun is not None
        if self.ufun(offer) >= self.ufun(outcome):
            return True
        elif relative_time > 0.98 and self.ufun(offer) > self.ufun.reserved_value:
            return True
        else:
            return False

    def update_reserved_value(self, offer, relative_time):
        """Learns the reserved value of the partner"""
        assert self.opponent_ufun is not None

        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(relative_time)

        tmp_times = [_ for idx, _ in enumerate(self.opponent_times) if idx % 2 == 0]
        tmp_opponent_utilities = [
            _ for idx, _ in enumerate(self.opponent_utilities) if idx % 2 == 0
        ]

        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e, 1.0
                ),
                tmp_times,
                tmp_opponent_utilities,
                bounds=bounds,
            )
            if relative_time >= 0.4 and len(self.opponent_utilities) > 10:
                self._past_oppnent_rv = self.opponent_ufun.reserved_value
                self.opponent_ufun.reserved_value = optimal_vals[1] * (
                    np.exp(self.opponent_utilities[-1] / self.opponent_utilities[-2])
                    - 0.5
                )
            else:
                self._past_oppnent_rv = self.opponent_ufun.reserved_value
                self.opponent_ufun.reserved_value = optimal_vals[1]
        except Exception as e:
            err, optimal_vals = f"{str(e)}", [None, None]

        # log my estimate
        if self._enable_logging:
            self.nmi.log_info(
                self.id,
                dict(
                    estimated_rv=self.opponent_ufun.reserved_value,
                    n_unique=n_unique,
                    opponent_utility=self.opponent_utilities[-1],
                    estimated_exponent=optimal_vals[0],
                    estimated_max=self.opponent_utilities[0],
                    error=err,
                ),
            )
