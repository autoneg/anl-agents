from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from negmas import (
    Outcome,
    ResponseType,
    SAOResponse,
    SAOState,
    nash_points,
    pareto_frontier,
)
from copy import deepcopy
from scipy.optimize import curve_fit


__all__ = ["Shochan"]


def aspiration_function(t, mx, rv, e):
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class Shochan(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 17.5,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.fe = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        self.my_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []
        self.predict = []
        self.my_utilities: list[float] = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        self.lasttime = 1.0
        self.diffmean = 0.01
        self.pat = 0.95
        self.g1 = 0
        self.g2 = 0
        self.g3 = 0
        self.g4 = 0
        self.mode = 0
        self.plus = 0.10

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        assert self.opponent_ufun is not None
        self.best_offer__ = self.ufun.best()
        self.private_info["opponent_ufun"] = deepcopy(self.opponent_ufun)
        self.opponent_ufun.reserved_value = 0.0
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        diff = []
        for i in range(len(self.opponent_times)):
            diff.append(self.opponent_times[i] - self.my_times[i])
        # diff = self.opponent_times - self.my_times
        if len(diff) == 0:
            diff_mean = 0.01
        else:
            diff_mean = sum(diff) / len(diff)

        self.diff_mean = diff_mean

        asp = aspiration_function(
            state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.fe
        )

        self.e = self.fe + (1.0 - asp) * 100

        self.my_times.append(state.relative_time)
        if self.is_acceptable(state):
            if (state.step) == 0:
                one_step = 0.0001
            else:
                one_step = (state.relative_time) / (state.step)
            if self.ufun(state.current_offer) >= self.ufun.reserved_value + self.plus:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        if (state.step) == 0:
            one_step = 0.0001
        else:
            one_step = (state.relative_time) / (state.step)
        lasttime = (1.0 // one_step) * one_step
        self.lasttime = lasttime
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun and self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        opp_frontier_utils = [_[1] for _ in frontier_utils]
        # print(my_frontier_utils)
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            self.nash = nash[0][0][0]
            self.nasho = frontier_outcomes[nash[0][1]]

        else:
            self.nash = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = self.ufun.reserved_value
        self.y2 = self.ufun.reserved_value

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            ave_nash = 1.0
            if len(my_frontier_utils) != 0:
                ave_nash = 0.0
                min_nash = my_frontier_utils[0] + opp_frontier_utils[0]
                for i in frontier_utils:
                    ave_nash = ave_nash + i[0] + i[1]
                    if min_nash > i[0] + i[1]:
                        # print(min_nash)
                        # print(i[0] + i[1])
                        min_nash = i[0] + i[1]
                ave_nash = ave_nash / len(my_frontier_utils)
                # print(min_nash)
            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]
            self._outcomes2 = [w for u, w in zip(my_frontier_utils, frontier_outcomes)]

            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1

            self._opprational = sorted(
                [  # type: ignore
                    (opp_util, my_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )

            self._rational2 = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1

            self._opprational2 = sorted(
                [  # type: ignore
                    (opp_util, my_util, _)
                    for _ in outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )

            y1 = self._rational2[-1][0]
            x1 = self._rational2[-1][1]
            y2 = self._opprational2[-1][1]
            x2 = self._opprational2[-1][0]
            difx = x2 - x1
            dify = y1 - y2
            self.y2 = y2
            if difx - dify >= 0.2:
                self.mode = 1
            if self.nmi.n_steps is not None and self.nmi.n_steps <= 50:
                self.mode = 1
            # print(self.mode)
            # x1 = int(x1*100)/100
            # x2 = int(x2*100)/100
            # y1 = int(y1*100)/100
            # y2 = int(y2*100)/100
            # if(nash):
            #     print("nash")
            #     print(nash)

            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        max_rational = len(self._rational) - 1
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)

        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1

        border = self.ufun.reserved_value

        myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)

        myut = self.nash
        myut = self.ufun.reserved_value + 0.1
        # myut = self.ufun.reserved_value
        # if(state.step + 1 == self.nmi.n_steps or state.relative_time + 2 * one_step > 1.0):
        if state.relative_time + 3 * one_step > 1.0:
            # print("num")
            # print([len(self.opponent_utilities),len(self.my_utilities)])
            # print(self.nmi.n_steps)
            # print(state.relative_time)
            opmin = sorted(self.opponent_utilities)[0]
            opmax = sorted(self.opponent_utilities)[-1]
            opmin2 = sorted(self.opponent_utilities)[1]
            target = (
                opmax
                - (opmax - opmin) * (state.relative_time) / (self.opponent_times[-1])
            )
            opmin - (opmin2 - opmin)

            indop = len(self._opprational2) - 1
            if nash:
                outcome4 = self.nasho
            else:
                outcome4 = self._best

            myut = self.ufun.reserved_value + 0.1
            tttt = []
            ttttt = []
            # print(myut)
            while indop != 0:
                if myut <= self._opprational2[indop][1]:
                    myut = self._opprational2[indop][1]
                    outcome = self._opprational2[indop][2]
                    tttt.append(self._opprational2[indop][1])
                    outcome4 = outcome
                nextidx = max(indop - 1, 0)
                ttttt.append(self._opprational2[indop][1])
                if self._opprational2[nextidx][0] >= target:
                    indop = nextidx
                else:
                    break

            # print("myut")
            # print(self.ufun.reserved_value)
            # print(myut)
            # print(target)
            # print(target2)
            # self.opponent_ufun.reserved_value = max(target - 0.1,0)
            # # self.opponent_ufun.reserved_value = max(target - 0.1,0)
            # ufuns2 = (self.ufun, self.opponent_ufun)
            # # list all outcomes
            # outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
            # frontier_utils, frontier_indices = pareto_frontier(ufuns2, outcomes) # type: ignore
            # nash2 = nash_points(ufuns2, frontier_utils)  # type: ignore
            # # nash2 =
            # if(nash):
            #     print("nash")
            #     print(nash)
            # if(nash2):
            #     print("nash2")
            #     print(nash2)

            if state.step <= 50:
                if self.ufun(self._best) > self.ufun.reserved_value + self.plus:
                    outcome = self._best
                else:
                    outcome = self.nasho
                # if(self.ufun(self._nasho) >= self.ufun(self._best))

            else:
                if nash:
                    outcome = self.nasho
                    if self.ufun(self._best) >= self.ufun.reserved_value + self.plus:
                        if self.ufun(self._best) <= self.ufun(outcome4):
                            outcome = outcome4
                        else:
                            # print("aaa")
                            outcome = self._best
                        # if(len(self.opponent_utilities) == len(self.my_utilities)):
                        if self.opponent_utilities[-1] < self.opponent_utilities[-2]:
                            # print([len(self.opponent_utilities),len(self.my_utilities)])
                            # print(self.nmi.n_steps)
                            if len(self.opponent_utilities) > len(self.my_utilities):
                                outcome = outcome4
                            else:
                                outcome = self.nasho
                else:
                    # print("nothing")
                    if self.ufun(self._best) >= self.ufun.reserved_value:
                        outcome = self._best
                        if self.ufun(self._best) <= self.ufun(outcome4):
                            outcome = outcome4
                    else:
                        outcome = self._opprational[self.nidx][2]
                        outcome = outcome4

            # if(self.ufun(self._best) > self.ufun.reserved_value):
            #     outcome = self._best
        else:
            if self.best_offer__ is None:
                self.best_offer__ = self.ufun.best()
            outcome = self.best_offer__
            border = self.ufun.reserved_value
            border = self.y2
            if nash:
                border = max(border, self.ufun(self.nasho))
            if self.mode == 1:
                myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)
                indmy = max_rational
                while indmy != 0:
                    nextidx = max(indmy - 1, 0)
                    if self._rational[nextidx][0] >= myasp:
                        indmy = nextidx
                    else:
                        break

                indx = indmy
                outcome = self._rational[indx][-1]

        assert self.ufun
        self.my_utilities.append(float(self.ufun(outcome)))
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)
        # return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False
        if (state.step) == 0:
            one_step = 0.0001
        else:
            one_step = (state.relative_time) / (state.step)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        self.opponent_times.append(state.relative_time)
        assert self.opponent_ufun
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))

        # Find the current aspiration level
        myasp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        aspiration_function(
            state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
        )

        pat = self.pat * self.lasttime
        border = self.ufun.reserved_value
        if state.relative_time >= pat:
            myasp = aspiration_function(
                pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe
            )
            ratio = (myasp - self.ufun.reserved_value) / (
                (self.lasttime - pat) * (self.lasttime - pat)
            )
            xd = (state.relative_time / self.lasttime) - pat
            y = myasp - (ratio * xd * xd)
            border = max(border, y)
        else:
            border = self.ufun.reserved_value

        myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

        if state.relative_time + 1 * one_step > 1.0:
            # if(state.step + 1 == self.nmi.n_steps or state.relative_time + 1 * one_step > 1.0):
            # print(self.predict)
            # print("last")
            # print([len(self.opponent_utilities),len(self.my_utilities)])
            # print(self.nmi.n_steps)
            # print(state.relative_time)
            if float(self.ufun(offer)) >= self.ufun.reserved_value + self.plus:
                if self.opponent_utilities[-1] <= self.opponent_utilities[-2]:
                    return True
                if len(self.opponent_utilities) > len(self.my_utilities):
                    return True

        return float(self.ufun(offer)) >= myasp


class Shochan_base75(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 7.5,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None
        # self.most

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base50(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 5.0,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base100(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 10.0,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base125(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 12.5,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base150(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 15.0,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base175(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 17.5,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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


class Shochan_base200(ANLNegotiator):
    """A simple negotiator that uses curve fitting to learn the reserved value.

    Args:
        min_unique_utilities: Number of different offers from the opponent before starting to
                              attempt learning their reserved value.
        e: The concession exponent used for the agent's offering strategy
        stochasticity: The level of stochasticity in the offers.
        enable_logging: If given, a log will be stored  for the estimates.

    Remarks:

        - Assumes that the opponent is using a time-based offering strategy that offers
          the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
          the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
          concession rate and $r$ is the reserved value we want to learn.
        - After it receives offers with enough different utilities, it starts finding the optimal values
          for $e$ and $r$.
        - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
          based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
          the same concession curve defined above to concede over an ordered list of these outcomes.
        - Is this better than using the same concession curve on the outcome space without even trying to learn
          the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
        - Note that the way we check for availability of enough data for training is based on the uniqueness of
          the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
          is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
          but very close utilities because or rounding errors (or genuine very small differences). Such differences should
          be ignored.
        - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
          with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
          we have MORE negotiation power when the partner has LOWER utility.
    """

    def __init__(
        self,
        *args,
        min_unique_utilities: int = 10,
        e: float = 20.0,
        stochasticity: float = 0.1,
        enable_logging: bool = False,
        nash_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.stochasticity = stochasticity
        # keeps track of times at which the opponent offers
        self.opponent_times: list[float] = []
        # keeps track of opponent utilities of its offers
        self.opponent_utilities: list[float] = []
        # keeps track of the our last estimate of the opponent reserved value
        self._past_oppnent_rv = 0.0
        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []
        self._enable_logging = enable_logging
        self.preoffer = []

        self._outcomes: list[Outcome] = []
        self._min_acceptable = float("inf")
        self._nash_factor = nash_factor
        self._best: Outcome = None  # type: ignore
        self.nidx = 0
        assert self.opponent_ufun
        self.opponent_ufun.reserved_value = 0.0
        self.best_offer__ = None

    def on_preferences_changed(self, changes):
        assert self.ufun is not None
        self.best_offer__ = self.ufun.best()
        return super().on_preferences_changed(changes)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        if self.is_acceptable(state):
            # print(self.ufun(state.current_offer))
            # print(self.opponent_ufun(state.current_offer))
            # print(self.opponent_ufun.reserved_value)
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        else:
            self.preoffer.append(
                (
                    self.ufun(state.current_offer),
                    self.opponent_ufun(state.current_offer),
                )
            )
        # The offering strategy

        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        assert self.ufun.outcome_space
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)  # type: ignore
        frontier_outcomes = [outcomes[_] for _ in frontier_indices]
        my_frontier_utils = [_[0] for _ in frontier_utils]
        nash = nash_points(ufuns, frontier_utils)  # type: ignore
        if nash:
            # find my utility at the Nash Bargaining Solution.
            my_nash_utility = nash[0][0][0]
        else:
            my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        # Set the acceptable utility limit
        self._min_acceptable = my_nash_utility * self._nash_factor
        self._min_acceptable = self.ufun.reserved_value
        # self._min_acceptable = 0
        # Set the set of outcomes to offer from

        # self._outcomes = [ # type: ignore
        #     w
        #     for u, w in zip(my_frontier_utils, frontier_outcomes)
        #     if u >= self._min_acceptable
        # ]

        # We only update our estimate of the rational list of outcomes if it is not set or
        # there is a change in estimated reserved value
        if (
            not self._rational
            or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
        ):
            # The rational set of outcomes sorted dependingly according to our utility function
            # and the opponent utility function (in that order).
            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self.nmi.outcome_space.enumerate_or_sample(
            #             levels=10, max_cardinality=100_000
            #         )
            #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > self.opponent_ufun.reserved_value
            #     ],
            # )
            a = set([])
            b = set([])
            ori_outcomes = []
            tmp = sorted(
                [  # type: ignore
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [  # type: ignore
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],
                reverse=True,
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if tmp[i][0] not in a:
                    a.add(tmp[i][0])
                    if tmp[i][0] >= self.ufun.reserved_value:
                        ori_outcomes.append(tmp[i][-1])

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if tmp2[i][0] not in b:
                    b.add(tmp2[i][0])
                    if tmp2[i][0] >= self.ufun.reserved_value:
                        ori_outcomes2.append(tmp2[i][-1])
                        # print([tmp[i][0],tmp[i][1]])
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [  # type: ignore
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]

            # self._rational = sorted(
            #     [ # type: ignore
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [  # type: ignore
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_))) > 0
                ],
            )
            self.nidx = len(self._rational) - 1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer__)
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        while self.nidx != 0:
            nextidx = max(self.nidx - 1, 0)
            if self._rational[nextidx][0] > asp:
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx

        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max(0, min(int(min_indx + s * n_rational), max_rational))
        max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx

        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if (
            self.ufun(self._best) > self.ufun(outcome)
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best

        if (
            state.relative_time > 0.98
            and self.ufun(self._best) > self.ufun.reserved_value
        ):
            outcome = self._best
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def is_acceptable(self, state: SAOState) -> bool:
        # The acceptance strategy
        assert self.ufun and self.opponent_ufun
        # get the offer from the mechanism state
        offer = state.current_offer
        # If there is no offer, there is nothing to accept
        if offer is None:
            return False

        # print("offer")
        # print(offer)

        if self._best is None:
            self._best = offer
        else:
            if self.ufun(offer) > self.ufun(self._best):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif self.ufun(offer) == self.ufun(self._best):
                if self.opponent_ufun(offer) < self.opponent_ufun(self._best):
                    self._best = offer

        # Find the current aspiration level
        asp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        # accept if the utility of the received offer is higher than
        # the current aspiration

        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, state: SAOState):
        # Learns the reserved value of the partner
        assert self.opponent_ufun is not None
        # extract the current offer from the state
        offer = state.current_offer
        if offer is None:
            return
        # save to the list of utilities received from the opponent and their times
        self.opponent_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(state.relative_time)

        # If we do not have enough data, just assume that the opponent
        # reserved value is zero
        n_unique = len(set(self.opponent_utilities))
        if n_unique < self.min_unique_utilities:
            self._past_oppnent_rv = 0.0
            self.opponent_ufun.reserved_value = 0.0
            return
        # Use curve fitting to estimate the opponent reserved value
        # We assume the following:
        # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
        # - The opponent never offers outcomes lower than their reserved value which means
        #   that their rv must be no higher than the worst outcome they offered for themselves.
        bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
        err = ""
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponent_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponent_utilities,
                bounds=bounds,
            )
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
