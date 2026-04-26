"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random

from negmas import Outcome
from negmas.negotiators import Controller
from negmas.preferences import pareto_frontier
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences
from negmas.sao.common import SAOState
from negmas.situated import Agent
from scipy.optimize import curve_fit

from .base import BaseAgent

__all__ = ["AgentNyan"]


class AgentNyan(BaseAgent):
    def __init__(
        self,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        type_name: str | None = None,
        can_propose: bool = True,
        **kwargs,
    ):
        super().__init__(
            preferences, ufun, name, parent, owner, id, type_name, can_propose, **kwargs
        )
        self.opponent_minimum_util = 1.0
        self.pareto: list[tuple[float, float, Outcome]] = []
        self.rational_outcomes: list[Outcome] = []
        self.e = 0.5
        self.strategy_change_time = 0.5
        self.opponent_minimum_utils_plot: list[tuple[float, float]] = []

    def strategy(self, state: SAOState):
        if state.relative_time < self.strategy_change_time:
            return "TimeDependent"
        return "Advantage"

    def on_preferences_changed(self, changes) -> None:
        """
        Called when the ufun is set and on any change to the ufun.

        Remarks:
            - Can optionally be used for initializing your agent.

        """
        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        pareto_frontier_utils, pareto_frontier_indices = pareto_frontier(
            [self.my_ufun_typesafe, self.opponent_ufun_typesafe], outcomes
        )
        assert len(pareto_frontier_indices) == len(pareto_frontier_utils)
        self.pareto: list[tuple[float, float, Outcome]] = [
            (
                pareto_frontier_utils[i][0],
                pareto_frontier_utils[i][1],
                outcomes[pareto_frontier_indices[i]],
            )
            for i in range(len(pareto_frontier_indices))
        ]  # pareto[i][0]: my utility, pareto[i][1]: opponent utility, pareto[i][2]: outcome
        self.pareto.sort(key=lambda x: x[0], reverse=True)
        self.rational_outcomes = [
            o
            for _, _, o in self.pareto
            if self.my_ufun_typesafe(o) > self.my_ufun_typesafe.reserved_value
        ]
        self.opponent_worst_outcome__ = None
        self.best_offer__ = None

    def should_accept(self, state: SAOState) -> bool:
        # self.opponent_minimum_util = min(
        #     self.opponent_minimum_util,
        #     float(self.opponent_ufun_typesafe(state.current_offer)),
        # )
        # update opponent minimum utility
        offer = state.current_offer
        opponent_util = float(self.opponent_ufun_typesafe(offer))
        if self.opponent_minimum_util > opponent_util:
            self.opponent_minimum_util = opponent_util
            self.opponent_minimum_utils_plot.append(
                (state.relative_time, opponent_util)
            )

        if offer is None:
            return False
        if offer not in self.rational_outcomes:
            return False
        if self.strategy(state) == "TimeDependent":
            return self.my_ufun_typesafe(offer) >= self.target_utility(state)
        return self.advantage_delta(state, offer) >= self.advantage_delta_threshold(
            state
        )

    def advantage_delta(self, state: SAOState, offer: Outcome) -> float:
        my_advantage = (
            self.my_ufun_typesafe(offer) - self.my_ufun_typesafe.reserved_value
        )
        expected_opponent_advantage = self.opponent_ufun_typesafe(
            offer
        ) - self.expected_opponent_reserved_value(state)
        return float(my_advantage) - float(expected_opponent_advantage)

    def advantage_delta_threshold(self, state: SAOState) -> float:
        reserved_value_delta = (
            self.my_ufun_typesafe.reserved_value
            - self.expected_opponent_reserved_value(state)
        )
        if reserved_value_delta > 0:
            # 0.0 -> -reserved_value_delta
            return -reserved_value_delta * state.relative_time
        else:
            # reserved_value_delta -> 0.0
            return reserved_value_delta * (1 - state.relative_time)

    def get_first_offer(self) -> Outcome:
        if self.best_offer__ is None:
            self.best_offer__ = self.my_ufun_typesafe.best()
        return self.best_offer__

    def get_candidate_offers(self, state: SAOState) -> list[Outcome]:
        target_utility = self.target_utility(state)
        if self.strategy(state) == "TimeDependent":
            return [
                o
                for o in self.rational_outcomes
                if self.my_ufun_typesafe(o) >= target_utility
            ]
        else:
            advantage_delta_threshold = self.advantage_delta_threshold(state)
            return [
                o
                for o in self.rational_outcomes
                if self.advantage_delta(state, o) >= advantage_delta_threshold
                and self.my_ufun_typesafe(o) >= target_utility
            ]

    def get_offer(self, state: SAOState) -> Outcome:
        candidates = self.get_candidate_offers(state)
        if len(candidates) == 0:
            return self.get_first_offer()
        return random.choice(candidates)

    def target_utility(self, state: SAOState) -> float:
        min = self.my_ufun_typesafe.reserved_value
        max = 1.0
        return min + (max - min) * ((1 - state.relative_time) ** self.e)

    def expected_opponent_reserved_value(self, state: SAOState) -> float:
        # if state.relative_time == 0.0:
        #     return 1.0
        # diminished_util = 1.0 - self.opponent_minimum_util
        # v = 1.0 / state.relative_time
        # worst_outcome = self.opponent_ufun_typesafe.worst()
        # return max(
        #     float(self.opponent_ufun_typesafe(worst_outcome)), 1.0 - diminished_util * v
        # )
        if len(self.opponent_minimum_utils_plot) < 2:
            return self.opponent_minimum_util  # fallback

        def curve(x, a, b):
            # return a * x**2 + b * x + c
            return b + (1.0 - b) * ((1 - x) ** a)

        try:
            popt, pocv = curve_fit(
                curve,
                [x for x, _ in self.opponent_minimum_utils_plot],
                [y for _, y in self.opponent_minimum_utils_plot],
                p0=[1.0, 0.5],
                maxfev=10000,
                bounds=([0.0, 0.0], [2.0, 1.0]),
            )
            result = curve(1.0, *popt)
        except Exception:
            result = self.opponent_minimum_util  # fallback
        if self.opponent_worst_outcome__ is None:
            self.opponent_worst_outcome__ = self.opponent_ufun_typesafe.worst()
        if result < self.opponent_minimum_util:
            return self.clamp(
                result,
                float(self.opponent_ufun_typesafe(self.opponent_worst_outcome__)),
                1.0,
            )
        else:
            return self.opponent_minimum_util

    def clamp(self, value: float, minval: float, maxval: float) -> float:
        return max(min(value, maxval), minval)
