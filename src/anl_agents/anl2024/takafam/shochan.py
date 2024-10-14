import random

import numpy as np
from negmas import nash_points, pareto_frontier
from negmas import Outcome, ResponseType, SAONegotiator, SAOResponse, SAOState
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

__all__ = ["Shochan"]


def aspiration_function(t, mx, rv, e):
    return (mx - rv) * (1.0 - np.power(t, e)) + rv

class Shochan(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        self.lasttime = 1.0
        self.diffmean = 0.01
        self.pat = 0.95
        self.g1 = 0
        self.g2 = 0
        self.g3 = 0
        self.g4 = 0
        self.mode = 0
        self.plus = 0.10
        

    def __call__(self, state: SAOState) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        # update the opponent reserved value in self.opponent_ufun
        # self.update_reserved_value(state)
        # rune the acceptance strategy and if the offer received is acceptable, accept it
        diff = []
        for i in range(len(self.opponent_times)):
            diff.append(self.opponent_times[i] - self.my_times[i])
        # diff = self.opponent_times - self.my_times
        if(len(diff)==0):
            diff_mean=0.01
        else:
            diff_mean = sum(diff) / len(diff)
        
        self.diff_mean = diff_mean

        asp = aspiration_function(state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
        
        self.e = self.fe + (1.0 - asp) * 100

        self.my_times.append(state.relative_time)
        if self.is_acceptable(state):
            if((state.step)==0):
                one_step = 0.0001
            else:
                one_step = (state.relative_time) / (state.step)
            if(self.ufun(state.current_offer) >= self.ufun.reserved_value + self.plus):
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
                

        else:
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        if((state.step)==0):
            one_step = 0.0001
        else:
            one_step = (state.relative_time) / (state.step)
        lasttime = (1.0 // one_step) * one_step
        self.lasttime = lasttime
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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
            if(len(my_frontier_utils)!=0):
                ave_nash = 0.0
                min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
                for i in frontier_utils:
                    ave_nash = ave_nash + i[0] + i[1]
                    if(min_nash > i[0] + i[1]):
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
            self._outcomes2 = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
            ]     

            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1

            self._opprational = sorted(
                [
                    (opp_util, my_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )

            self._rational2 = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1

            self._opprational2 = sorted(
                [
                    (opp_util, my_util, _)
                    for _ in outcomes
                    if (my_util := float(self.ufun(_))) > 0
                    # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )

            y1 = self._rational2[-1][0]
            x1 = self._rational2[-1][1]
            y2 = self._opprational2[-1][1]
            x2 = self._opprational2[-1][0]
            difx = x2 - x1
            dify = y1 - y2
            self.y2 = y2
            if(difx - dify >= 0.2):
                self.mode = 1
            if(self.nmi.n_steps <= 50):
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
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        

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
        if(state.relative_time + 3 * one_step > 1.0):
            # print("num")
            # print([len(self.opponent_utilities),len(self.my_utilities)])
            # print(self.nmi.n_steps)
            # print(state.relative_time)
            opmin = sorted(self.opponent_utilities)[0]
            opmax = sorted(self.opponent_utilities)[-1]
            opmin2 = sorted(self.opponent_utilities)[1]
            target = opmax - (opmax - opmin) * (state.relative_time) / (self.opponent_times[-1]) 
            target2 = opmin - (opmin2 - opmin)

            indop = len(self._opprational2) - 1
            if(nash):
                outcome4 = self.nasho
            else:
                outcome4 = self._best

            myut = self.ufun.reserved_value + 0.1
            tttt = []
            ttttt = []
            # print(myut)
            while(indop!=0):
                if(myut <= self._opprational2[indop][1]):
                    myut = self._opprational2[indop][1]
                    outcome = self._opprational2[indop][2]
                    tttt.append(self._opprational2[indop][1])
                    outcome4 = outcome
                nextidx = max(indop-1, 0)
                ttttt.append(self._opprational2[indop][1])
                if(self._opprational2[nextidx][0] >= target):
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
            # frontier_utils, frontier_indices = pareto_frontier(ufuns2, outcomes)
            # nash2 = nash_points(ufuns2, frontier_utils)  # type: ignore
            # # nash2 = 
            # if(nash):
            #     print("nash")
            #     print(nash)
            # if(nash2):
            #     print("nash2")
            #     print(nash2)


            if(state.step <= 50):
                if(self.ufun(self._best) > self.ufun.reserved_value + self.plus):
                    outcome = self._best
                else:
                    outcome = self.nasho
                # if(self.ufun(self._nasho) >= self.ufun(self._best))
                
                
            else:
                if(nash):
                    outcome = self.nasho
                    if(self.ufun(self._best) >= self.ufun.reserved_value  + self.plus):
                        if(self.ufun(self._best) <= self.ufun(outcome4)):
                            outcome = outcome4
                        else:
                            # print("aaa")
                            outcome = self._best
                        # if(len(self.opponent_utilities) == len(self.my_utilities)):
                        if(self.opponent_utilities[-1] < self.opponent_utilities[-2]):
                            # print([len(self.opponent_utilities),len(self.my_utilities)])
                            # print(self.nmi.n_steps)
                            if(len(self.opponent_utilities) > len(self.my_utilities)):
                                outcome = outcome4
                            else:
                                outcome = self.nasho
                else:
                    # print("nothing")
                    if(self.ufun(self._best) >= self.ufun.reserved_value):
                        outcome = self._best
                        if(self.ufun(self._best) <= self.ufun(outcome4)):
                            outcome = outcome4
                    else:
                        outcome = self._opprational[self.nidx][2]
                        outcome = outcome4

            # if(self.ufun(self._best) > self.ufun.reserved_value):
            #     outcome = self._best
        else:
            outcome = self.ufun.best()
            border = self.ufun.reserved_value
            border = self.y2
            if(nash):
                border = max(border,self.ufun(self.nasho))
            if(self.mode==1):
                myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)
                indmy = max_rational
                while(indmy!=0):
                    nextidx = max(indmy-1 , 0)
                    if(self._rational[nextidx][0] >= myasp):
                        indmy = nextidx
                    else:
                        break
                
                indx = indmy
                outcome = self._rational[indx][-1]

        self.my_utilities.append(self.ufun(outcome))
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
        if((state.step)==0):
            one_step = 0.0001
        else:
            one_step = (state.relative_time) / (state.step)
        
        
        if self._best is None:
            self._best = offer
        else:
            if(self.ufun(offer) > self.ufun(self._best)):
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
                    self._best = offer

        self.opponent_times.append(state.relative_time)
        self.opponent_utilities.append(self.opponent_ufun(offer))
                    
        # Find the current aspiration level
        myasp = aspiration_function(
            state.relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        opasp = aspiration_function(
            state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
        )
        ans = False

        pat = self.pat * self.lasttime
        border = self.ufun.reserved_value
        if(state.relative_time >= pat):
            myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
            ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
            xd = (state.relative_time / self.lasttime) - pat
            y = myasp - (ratio*xd*xd) 
            border = max(border,y)
        else:
            border = self.ufun.reserved_value



        myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

        if(state.relative_time + 1 * one_step > 1.0):
        # if(state.step + 1 == self.nmi.n_steps or state.relative_time + 1 * one_step > 1.0):
            # print(self.predict)
            # print("last")
            # print([len(self.opponent_utilities),len(self.my_utilities)])
            # print(self.nmi.n_steps)
            # print(state.relative_time)
            if(float(self.ufun(offer)) >= self.ufun.reserved_value + self.plus):
                if(self.opponent_utilities[-1] <= self.opponent_utilities[-2]):
                    return True
                if(len(self.opponent_utilities) > len(self.my_utilities)):
                    return True
                # myasp = 0.0

        # if(self.nmi.n_steps <= 50):
        #     if(float(self.ufun(offer)) > self.ufun.reserved_value):
        # accept if the utility of the received offer is higher than
        # the current aspiration

        # return ans
        return (float(self.ufun(offer)) >= myasp)


# ?代目 3rd
# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """
#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 17.5,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.fe = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         self.my_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []
#         self.predict = []
#         self.my_utilities: list[float] = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 0.0
#         self.lasttime = 1.0
#         self.diffmean = 0.01
#         self.pat = 0.95
#         self.g1 = 0
#         self.g2 = 0
#         self.g3 = 0
#         self.g4 = 0
#         self.mode = 0
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         diff = []
#         for i in range(len(self.opponent_times)):
#             diff.append(self.opponent_times[i] - self.my_times[i])
#         # diff = self.opponent_times - self.my_times
#         if(len(diff)==0):
#             diff_mean=0.01
#         else:
#             diff_mean = sum(diff) / len(diff)
        
#         self.diff_mean = diff_mean

#         asp = aspiration_function(state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
        
#         self.e = self.fe + (1.0 - asp) * 100

#         self.my_times.append(state.relative_time)
#         if self.is_acceptable(state):
#             if((state.step)==0):
#                 one_step = 0.0001
#             else:
#                 one_step = (state.relative_time) / (state.step)
#             if(self.ufun(state.current_offer) > self.ufun.reserved_value + 0.1):
#                 return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
                

#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         if((state.step)==0):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
#         lasttime = (1.0 // one_step) * one_step
#         self.lasttime = lasttime
#         # The offering strategy
            
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             self.nash = nash[0][0][0]
#             self.nasho = frontier_outcomes[nash[0][1]]

#         else:
#             self.nash = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         self._min_acceptable = self.ufun.reserved_value
#         self.y2 = self.ufun.reserved_value
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             # The rational set of outcomes sorted dependingly according to our utility function
#             # and the opponent utility function (in that order).
#             ave_nash = 1.0
#             if(len(my_frontier_utils)!=0):
#                 ave_nash = 0.0
#                 min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
#                 for i in frontier_utils:
#                     ave_nash = ave_nash + i[0] + i[1]
#                     if(min_nash > i[0] + i[1]):
#                         # print(min_nash)
#                         # print(i[0] + i[1])
#                         min_nash = i[0] + i[1]
#                 ave_nash = ave_nash / len(my_frontier_utils)
                
#                 # print(min_nash)


#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]     
#             self._outcomes2 = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#             ]     

#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )

#             self._rational2 = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational2 = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )

#             y1 = self._rational2[-1][0]
#             x1 = self._rational2[-1][1]
#             y2 = self._opprational2[-1][1]
#             x2 = self._opprational2[-1][0]
#             difx = x2 - x1
#             dify = y1 - y2
#             self.y2 = y2
#             if(difx - dify >= 0.2):
#                 self.mode = 1
#             if(self.nmi.n_steps <= 50):
#                 self.mode = 1
#             # print(self.mode)
#             x1 = int(x1*100)/100
#             x2 = int(x2*100)/100
#             y1 = int(y1*100)/100
#             y2 = int(y2*100)/100

        
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         max_rational = len(self._rational) - 1
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        

#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
#         # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
#         n_rational = len(self._rational)
#         max_rational = n_rational - 1

#         border = self.ufun.reserved_value



#         myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)



#         myut = self.nash
#         myut = self.ufun.reserved_value + 0.1
#         # myut = self.ufun.reserved_value 
#         # if(state.step + 1 == self.nmi.n_steps or state.relative_time + 2 * one_step > 1.0):
#         if(state.relative_time + 3 * one_step > 1.0):
#             opmin = sorted(self.opponent_utilities)[0]
#             opmax = sorted(self.opponent_utilities)[-1]
#             opmin2 = sorted(self.opponent_utilities)[1]
#             target = opmax - (opmax - opmin) * (state.relative_time) / (self.opponent_times[-1]) 
#             # target = opmin - (opmin2 - opmin)

#             indop = len(self._opprational2) - 1
#             if(nash):
#                 outcome4 = self.nasho
#             else:
#                 outcome4 = self.best

#             myut = self.ufun.reserved_value + 0.1
#             tttt = []
#             ttttt = []
#             # print(myut)
#             while(indop!=0):
#                 if(myut <= self._opprational2[indop][1]):
#                     myut = self._opprational2[indop][1]
#                     outcome = self._opprational2[indop][2]
#                     tttt.append(self._opprational2[indop][1])
#                     outcome4 = outcome
#                 nextidx = max(indop-1, 0)
#                 ttttt.append(self._opprational2[indop][1])
#                 if(self._opprational2[nextidx][0] >= target):
#                     indop = nextidx
#                 else:
#                     break

#             if(state.step <= 50):
#                 if(self.ufun(self._best) >= self.ufun.reserved_value):
#                     outcome = self._best
#                 else:
#                     outcome = self.nasho
#                 # if(self.ufun(self._nasho) >= self.ufun(self._best))
                
                
#             else:
#                 if(nash):
#                     outcome = self.nasho
#                     if(self.ufun(self._best) >= self.ufun.reserved_value):
#                         if(self.ufun(self._best) <= self.ufun(outcome4)):
#                             outcome = outcome4
#                         else:
#                             outcome = self._best
#                         # if(len(self.opponent_utilities) == len(self.my_utilities)):
#                         if(self.opponent_utilities[-1] < self.opponent_utilities[-2]):
#                             outcome = self.nasho
#                 else:
#                     # print("nothing")
#                     if(self.ufun(self._best) >= self.ufun.reserved_value):
#                         outcome = self._best
#                         if(self.ufun(self._best) <= self.ufun(outcome4)):
#                             outcome = outcome4
#                     else:
#                         outcome = self._opprational[self.nidx][2]
#                         outcome = outcome4

#             # if(self.ufun(self._best) > self.ufun.reserved_value):
#             #     outcome = self._best
#         else:
#             outcome = self.ufun.best()
#             border = self.ufun.reserved_value
#             border = self.y2
#             if(nash):
#                 border = max(border,self.ufun(self.nasho))
#             if(self.mode==1):
#                 myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)
#                 indmy = max_rational
#                 while(indmy!=0):
#                     nextidx = max(indmy-1 , 0)
#                     if(self._rational[nextidx][0] >= myasp):
#                         indmy = nextidx
#                     else:
#                         break
                
#                 indx = indmy
#                 outcome = self._rational[indx][-1]

#         self.my_utilities.append(self.ufun(outcome))
#         return SAOResponse(ResponseType.REJECT_OFFER, outcome)
#         # return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
#         if((state.step)==0):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
        
        
#         if self._best is None:
#             self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer

#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
                    
#         # Find the current aspiration level
#         myasp = aspiration_function(
#             state.relative_time, 1.0, self.ufun.reserved_value, self.e
#         )
#         opasp = aspiration_function(
#             state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
#         )
#         ans = False

#         pat = self.pat * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
#             ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)
#         else:
#             border = self.ufun.reserved_value



#         myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

#         if(state.relative_time + 1 * one_step > 1.0):
#         # if(state.step + 1 == self.nmi.n_steps or state.relative_time + 1 * one_step > 1.0):
#             # print(self.predict)
#             if(float(self.ufun(offer)) > self.ufun.reserved_value):
#                 if(self.opponent_utilities[-1] <= self.opponent_utilities[-2]):
#                     return True
#                 # myasp = 0.0

#         # if(self.nmi.n_steps <= 50):
#         #     if(float(self.ufun(offer)) > self.ufun.reserved_value):
#         # accept if the utility of the received offer is higher than
#         # the current aspiration

#         # return ans
#         return (float(self.ufun(offer)) >= myasp)
    

# ?代目 2nd2nd
# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """
#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 17.5,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.fe = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         self.my_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []
#         self.predict = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 0.0
#         self.lasttime = 1.0
#         self.diffmean = 0.01
#         self.pat = 0.95
#         self.g1 = 0
#         self.g2 = 0
#         self.g3 = 0
#         self.g4 = 0
#         self.mode = 0
#         # self.opmin = 1.0
#         # self.most
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         diff = []
#         for i in range(len(self.opponent_times)):
#             diff.append(self.opponent_times[i] - self.my_times[i])
#         # diff = self.opponent_times - self.my_times
#         if(len(diff)==0):
#             diff_mean=0.01
#         else:
#             diff_mean = sum(diff) / len(diff)
        
#         self.diff_mean = diff_mean

#         asp = aspiration_function(state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
        
#         self.e = self.fe + (1.0 - asp) * 100

#         self.my_times.append(state.relative_time)
#         if self.is_acceptable(state):
#             # print(self.predict)
#             # print(self.ufun(state.current_offer))
#             # print(self.opponent_ufun(state.current_offer))
#             # print(self.opponent_ufun.reserved_value)
#             # print(len(self.my_times))
#             if((state.step)==0):
#                 one_step = 0.0001
#             else:
#                 one_step = (state.relative_time) / (state.step)
#             # print(state.step)
#             # print(one_step)
#             # print(self.my_times)
#             # print(self.my_times[-1])
#             # print(len(self.opponent_times))
#             # print(self.opponent_times)
#             # print(self.opponent_utilities)
#             # print(self.opponent_times[-1])
#             # print(state.step)
#             # # print(state.time)
#             # print(state.relative_time)
#             # print(diff_mean)
#             # print(self.nmi.n_step)
#             return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         if((state.step)==0):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
#         lasttime = (1.0 // one_step) * one_step
#         self.lasttime = lasttime
#         # The offering strategy
            
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             # print(nash)
#             self.nash = nash[0][0][0]
#             self.nasho = frontier_outcomes[nash[0][1]]
#             # print(self.ufun(self.nasho))
#             # print(self.opponent_ufun(self.nasho))

#         else:
#             self.nash = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         # self._min_acceptable = my_nash_utility * self._nash_factor
#         self._min_acceptable = self.ufun.reserved_value
#         # self._min_acceptable = 0
#         # Set the set of outcomes to offer from
#         # self._outcomes = [
#         #     w
#         #     for u, w in zip(my_frontier_utils, frontier_outcomes)
#         #     if u >= self._min_acceptable
#         # ]        
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             # The rational set of outcomes sorted dependingly according to our utility function
#             # and the opponent utility function (in that order).
#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self.nmi.outcome_space.enumerate_or_sample(
#             #             levels=10, max_cardinality=100_000
#             #         )
#             #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > self.opponent_ufun.reserved_value
#             #     ],
#             # )
#             ave_nash = 1.0
#             if(len(my_frontier_utils)!=0):
#                 ave_nash = 0.0
#                 min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
#                 for i in frontier_utils:
#                     ave_nash = ave_nash + i[0] + i[1]
#                     if(min_nash > i[0] + i[1]):
#                         # print(min_nash)
#                         # print(i[0] + i[1])
#                         min_nash = i[0] + i[1]
#                 ave_nash = ave_nash / len(my_frontier_utils)
                
#                 # print(min_nash)


#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]     
#             self._outcomes2 = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#             ]     

#             # if(len(my_frontier_utils)!=0):
#             #     for _ in outcomes:
#             #         if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= min_nash):
#             #             if( _ not in self._outcomes):
#             #                 self._outcomes.append(_)

#             # if(len(my_frontier_utils)!=0):
#             #     for _ in outcomes:
#             #         if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= 1.0):
#             #             if( _ not in self._outcomes):
#             #                 self._outcomes.append(_)
            
#                     # if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= ave_nash - 0.1):
#                     #     if( _ not in self._outcomes):
#                     #         self._outcomes.append(_)


#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self._outcomes
#             #         if (my_util := float(self.ufun(_))) > 0
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > 0
#             #     ],
#             # )
#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )

#             # ufuns = (self.ufun, self.opponent_ufun)
#             # print(self.ufun)
#             # print(self.opponent_ufun)
#             self._rational2 = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational2 = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )

#             y1 = self._rational2[-1][0]
#             x1 = self._rational2[-1][1]
#             y2 = self._opprational2[-1][1]
#             x2 = self._opprational2[-1][0]
#             difx = x2 - x1
#             dify = y1 - y2
#             if(difx - dify >= 0.2):
#                 self.mode = 1
#             # print(self.mode)
#             x1 = int(x1*100)/100
#             x2 = int(x2*100)/100
#             y1 = int(y1*100)/100
#             y2 = int(y2*100)/100

#             # print(len(self._outcomes))
#             # print([x1,y1])
#             # print([x2,y2])

#             # for o in outcomes:
#             #     x = float(self.ufun(o))
#             #     x = int(self.ufun(o)*100)/100
#             #     y = float(self.opponent_ufun(o))
#             #     y = int(self.opponent_ufun(o)*100)/100
#             #     tmp1 = (x - x1) * (y1 - y2)
#             #     tmp2 = (y - y2) * (x2 - x1)
#             #     if(tmp1 > tmp2):
#             #         tmp3 = (x2 - x) * (y1 - y2)
#             #         tmp4 = (y - y2) * (x2 - x1)
#             #         if(tmp3 < tmp4):
#             #             self.g1 = self.g1 + 1
#             #         elif(tmp3 > tmp4):
#             #             self.g2 = self.g2 + 1
#             #     elif(tmp1 < tmp2):
#             #         tmp3 = (x2 - x) * (y1 - y2)
#             #         tmp4 = (y - y2) * (x2 - x1)
#             #         if(tmp3 < tmp4):
#             #             self.g4 = self.g4 + 1
#             #         elif(tmp3 > tmp4):
#             #             self.g3 = self.g3 + 1

#             # print([self.g1,self.g2,self.g3,self.g4])


        
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         max_rational = len(self._rational) - 1
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        

#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
#         # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
#         n_rational = len(self._rational)
#         max_rational = n_rational - 1
#         # min_indx = max(0, min(max_rational, int(asp * max_rational)))

#         # asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
#         # while(self.nidx!=0):
#         #     nextidx = max(self.nidx-1 , 0)
#         #     if(self._rational[nextidx][0] > asp):
#         #         self.nidx = nextidx
#         #     else:
#         #         break

#         # min_indx = self.nidx
        
#         # pat = 0.95
#         # pat = self.pat * self.lasttime
#         border = self.ufun.reserved_value
#         # if(state.relative_time >= pat):
#         #     myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
#         #     ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#         #     xd = (state.relative_time / self.lasttime) - pat
#         #     y = myasp - (ratio*xd*xd) 
#         #     # y = aspiration_function(xd,self.lasttime - xd, myasp, self.ufun.reserved_value, self.e)
#         #     # myasp = y
#         #     border = max(border,y)
#         #     # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#         #     # indmy = max_rational
#         #     # while(self.nidx!=0):
#         #     #     nextidx = max(indmy-1 , 0)
#         #     #     if(self._rational[nextidx][0] >= y):
#         #     #         indmy = nextidx
#         #     #     else:
#         #     #         break
#         #     # outcome = self._best
#         # else:
#         #     # tmp = 0.0
#         #     if(state.relative_time > 1.1):
#         #         popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#         #         tmp = popt[1]
#         #         # print(state.relative_time)
#         #         # print(popt)
#         #         indop = len(self._opprational) - 1 
#         #         while(indop!=0):
#         #             nextidx = max(indop-1, 0)
#         #             if(self._opprational[nextidx][0] >= tmp):
#         #                 indop = nextidx
#         #             else:
#         #                 break

#         #         pre_rv = self._opprational[self.nidx][1]

#         #         # self.opponent_ufun.reserved_value = pre_rv
#         #         border = max(border,pre_rv)
#         #     else:
#         #         border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)

#         # if(state.step < 25):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 5.0)
#         # elif(state.step < 75):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 7.5)
#         # elif(state.step < 150):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 10)
#         # else:
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)


#         # myasp = aspiration_function(state.relative_time, 1.0, border, self.e)


#         # indmy = max_rational
#         # while(indmy!=0):
#         #     nextidx = max(indmy-1 , 0)
#         #     if(self._rational[nextidx][0] >= myasp):
#         #         indmy = nextidx
#         #     else:
#         #         break
        
#         # indx = indmy
#         # outcome = self._rational[indx][-1]



#         myut = self.nash
#         myut = self.ufun.reserved_value + 0.1
#         # myut = self.ufun.reserved_value 
#         if(state.relative_time + 2 * one_step > 1.0):
#             opmin = sorted(self.opponent_utilities)[0]
#             # print("-----------------------")
#             # print(opmin)
#             indop = len(self._opprational) - 1 
#             while(indop!=0):
#                 if(myut <= self._opprational[indop][1]):
#                     myut = self._opprational[indop][1]
#                     outcome = self._opprational[indop][2]
#                 nextidx = max(indop-1, 0)
#                 if(self._opprational[nextidx][0] >= opmin):
#                     indop = nextidx
#                 else:
#                     break

#             pre_rv = self._opprational[self.nidx][1]
#             # print(pre_rv)

#             # self.opponent_ufun.reserved_value = pre_rv
#             border = max(border,pre_rv)
#             if(state.step <= 50):
#                 if(self.ufun(self._best) >= self.ufun.reserved_value):
#                     outcome = self._best
#                 else:
#                     outcome = self.nasho
#                 # if(self.ufun(self._nasho) >= self.ufun(self._best))
                
                
#             else:
#                 if(nash):
#                     outcome = self.nasho
#                     if(self.ufun(self._best) >= self.ufun(self.nasho)):
#                         outcome = self._best
#                 else:
#                     # print("nothing")
#                     if(self.ufun(self._best) >= self.ufun.reserved_value):
#                         outcome = self._best
#                     else:
#                         outcome = self._opprational[self.nidx][2]
#             # print(self.ufun(outcome))
#             # print(self.opponent_ufun(outcome))
#             # print(self.predict)
#             # if(self.ufun(self._best) > self.ufun.reserved_value):
#             #     outcome = self._best
#         else:
#             outcome = self.ufun.best()
#             border = self.ufun.reserved_value
#             if(self.mode==1):
#                 myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)
#                 indmy = max_rational
#                 while(indmy!=0):
#                     nextidx = max(indmy-1 , 0)
#                     if(self._rational[nextidx][0] >= myasp):
#                         indmy = nextidx
#                     else:
#                         break
                
#                 indx = indmy
#                 outcome = self._rational[indx][-1]
            
#         return SAOResponse(ResponseType.REJECT_OFFER, outcome)
#         # return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
#         if((state.step)==0):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
        
#         # print("offer")
#         # print(offer)
#         # print(negmas.outcomes.outcome2dict(offer))
        
#         if self._best is None:
#             self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 # print(self.ufun(offer))
#                 # print(self.ufun(self._best))
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer

#         # print(offer)

#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
                    
#         # Find the current aspiration level
#         myasp = aspiration_function(
#             state.relative_time, 1.0, self.ufun.reserved_value, self.e
#         )
#         opasp = aspiration_function(
#             state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
#         )
#         ans = False

#         pat = self.pat * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
#             ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)

#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(indop!=0):
#                     nextidx = max(indop-1, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

#         # if(state.step < 25):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 5.0)
#         # elif(state.step < 75):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 7.5)
#         # elif(state.step < 150):
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, 10)
#         # else:
#         #     myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)

#         # myasp = aspiration_function(state.relative_time, 1.0, border, self.e)


#         # if(state.relative_time >= 0.97):
#         #     ans = (float(self.ufun(offer)) >= myasp)
#         # else:
#         #     asp = max(myasp,opasp)
#         #     ans = (float(self.ufun(offer)) >= asp)

#         if(state.relative_time + one_step > 1.0):
#             # print(self.predict)
#             if(float(self.ufun(offer)) > self.ufun.reserved_value):
#                 if(self.opponent_utilities[-1] <= self.opponent_utilities[-2]):
#                     return True
#                 # myasp = 0.0


#         # accept if the utility of the received offer is higher than
#         # the current aspiration

#         # return ans
#         return (float(self.ufun(offer)) >= myasp)


# ?代目
# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """
#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 17.5,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.fe = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         self.my_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []
#         self.predict = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 0.0
#         self.lasttime = 1.0
#         self.diffmean = 0.01
#         self.pat = 0.95
#         self.nash = 0.5
#         self.nasho: Outcome = None
#         # self.opmin = 1.0
#         # self.most
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         diff = []
#         for i in range(len(self.opponent_times)):
#             diff.append(self.opponent_times[i] - self.my_times[i])
#         # diff = self.opponent_times - self.my_times
#         if(len(diff)==0):
#             diff_mean=0.01
#         else:
#             diff_mean = sum(diff) / len(diff)
        
#         self.diff_mean = diff_mean

#         asp = aspiration_function(state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
        
#         self.e = self.fe + (1.0 - asp) * 100

#         self.my_times.append(state.relative_time)

#         # if(state.timedout):
#         #     print(state.relative_time)

#         # if(state.started):
#         #     print("###")
#         #     print(state.last_negotiator)
#         #     print("##")

#         if self.is_acceptable(state):
#             # print(self.predict)
#             # print(self.ufun(state.current_offer))
#             # print(self.opponent_ufun(state.current_offer))
#             # print(self.opponent_ufun.reserved_value)
#             # print(len(self.my_times))
#             if((state.step)==0):
#                 one_step = 0.0001
#             else:
#                 # one_step = (state.relative_time) / (state.step)
#                 one_step = (self.my_times[-1] - self.my_times[1]) / (state.step - 1)
#                 # one_step3 = (self.my_times[-1] - self.my_times[0]) / (state.step)
#             # print(state.step)
#             # print(state.relative_time)
#             # print(state.timedout)
#             # print(self.my_times[1])
#             # print(self.my_times[0])
#             # print(one_step)
#             # print(one_step2)
#             # print(one_step3)

#             # print(self.my_times)
#             # print(self.my_times[-1])
#             # print(len(self.opponent_times))
#             # print(self.opponent_times)
#             # print(self.opponent_utilities)
#             # print(self.opponent_times[-1])
#             # print(state.step)
#             # # print(state.time)
#             # print(state.relative_time)
#             # print(diff_mean)
#             # print(state.current_proposer)
#             # print(state.last_negotiator)
#             # print(state.started)

#             return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         if((state.step)==0 or (state.step)==1):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
#             one_step = (self.my_times[-1] - self.my_times[1]) / (state.step - 1)
#         lasttime = (1.0 // one_step) * one_step
#         self.lasttime = lasttime
#         # The offering strategy
            
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             # print(nash)
#             self.nash = nash[0][0][0]
#             # print("333")
#             # print(self.nash)
#             # print(nash)
#             # # print(nash[0])
#             # # print(nash[0][0])
#             # # print(nash[0][0][0])
#             # print(nash[0][1])
#             # print(outcomes[nash[0][1]])
#             # print("33")
#             self.nasho = outcomes[nash[0][1]]
#         else:
#             self.nash = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         self._min_acceptable = self.nash * self._nash_factor
#         self._min_acceptable = self.ufun.reserved_value
#         # self._min_acceptable = 0
#         # Set the set of outcomes to offer from
#         # self._outcomes = [
#         #     w
#         #     for u, w in zip(my_frontier_utils, frontier_outcomes)
#         #     if u >= self._min_acceptable
#         # ]        
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             # The rational set of outcomes sorted dependingly according to our utility function
#             # and the opponent utility function (in that order).
#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self.nmi.outcome_space.enumerate_or_sample(
#             #             levels=10, max_cardinality=100_000
#             #         )
#             #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > self.opponent_ufun.reserved_value
#             #     ],
#             # )
#             ave_nash = 1.0
#             if(len(my_frontier_utils)!=0):
#                 ave_nash = 0.0
#                 min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
#                 for i in frontier_utils:
#                     ave_nash = ave_nash + i[0] + i[1]
#                     if(min_nash > i[0] + i[1]):
#                         # print(min_nash)
#                         # print(i[0] + i[1])
#                         min_nash = i[0] + i[1]
#                 ave_nash = ave_nash / len(my_frontier_utils)
#                 # print(min_nash)


#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]     

#             # if(len(my_frontier_utils)!=0):
#             #     for _ in outcomes:
#             #         if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= min_nash):
#             #             if( _ not in self._outcomes):
#             #                 self._outcomes.append(_)

#             # if(len(my_frontier_utils)!=0):
#             #     for _ in outcomes:
#             #         if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= 1.0):
#             #             if( _ not in self._outcomes):
#             #                 self._outcomes.append(_)
            
#                     # if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= ave_nash - 0.1):
#                     #     if( _ not in self._outcomes):
#                     #         self._outcomes.append(_)


#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self._outcomes
#             #         if (my_util := float(self.ufun(_))) > 0
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > 0
#             #     ],
#             # )
#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         max_rational = len(self._rational) - 1
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        

#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
#         # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
#         n_rational = len(self._rational)
#         max_rational = n_rational - 1
#         # min_indx = max(0, min(max_rational, int(asp * max_rational)))

#         # asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
#         # while(self.nidx!=0):
#         #     nextidx = max(self.nidx-1 , 0)
#         #     if(self._rational[nextidx][0] > asp):
#         #         self.nidx = nextidx
#         #     else:
#         #         break

#         # min_indx = self.nidx
        
#         pat = 0.95
#         pat = self.pat * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
#             ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             # y = aspiration_function(xd,self.lasttime - xd, myasp, self.ufun.reserved_value, self.e)
#             # myasp = y
#             border = max(border,y)
#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(indop!=0):
#                     nextidx = max(indop-1, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time, 1.0, border, self.fe)
#         indmy = max_rational
#         while(indmy!=0):
#             nextidx = max(indmy-1 , 0)
#             if(self._rational[nextidx][0] >= myasp):
#                 indmy = nextidx
#             else:
#                 break
        
#         indx = indmy
#         outcome = self._rational[indx][-1]
#         myut = self.ufun.reserved_value + 0.15
#         # print(state.relative_time + 1 * one_step)
#         if(state.relative_time + 2 * one_step >= 1.0):
#             opmin = sorted(self.opponent_utilities)[0]
#             # print("-----------------------")
#             # print(state.relative_time + 2 * one_step)
#             # print(state.relative_time)
#             # print(one_step)
#             # print(self.lasttime)
#             # print(opmin)
#             indop = len(self._opprational) - 1 
#             while(indop!=0):
#                 if(myut <= self._opprational[indop][1]):
#                     myut = self._opprational[indop][1]
#                     outcome = self._opprational[indop][2]
#                 nextidx = max(indop-1, 0)
#                 if(self._opprational[nextidx][0] >= opmin):
#                     indop = nextidx
#                 else:
#                     break

#             pre_rv = self._opprational[self.nidx][1]

#             # self.opponent_ufun.reserved_value = pre_rv
#             border = max(border,pre_rv)
#             # if nash:
#             #     # find my utility at the Nash Bargaining Solution.
#             #     print(nash)
#                 # my_nash_utility = nash[0][0][0]
#             if(state.step <= 50):
#                 if((self.ufun(self._best) >= self.ufun.reserved_value) and (self.ufun(self._best) >= self.nash)):
#                     # if(self.ufun(self._best) >= self.nash):
#                     outcome2 = self._best
#                 else:
#                     outcome2 = self.nasho
#                 if(self.ufun(outcome) <=  self.ufun(outcome2)):
#                     outcome = outcome2

                    
#             # print(self.ufun(outcome))
#             # print(self.opponent_ufun(outcome))
#             # outcome = self._best
#             # print(self.predict)
#             # if(self.ufun(self._best) > self.ufun.reserved_value):
#             #     outcome = self._best
#         else:
#             outcome = self.ufun.best()

            
#         return SAOResponse(ResponseType.REJECT_OFFER, outcome)
#         # return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
#         if((state.step)==0 or (state.step)==1):
#             one_step = 0.0001
#         else:
#             # one_step = (state.relative_time) / (state.step)
#             one_step = (self.opponent_times[-1] - self.opponent_times[1]) / (state.step - 1)
#             # one_step3 = (self.opponent_times[-1] - self.opponent_times[0]) / (state.step)

        
#         # print("offer")
#         # print(offer)
#         # print(negmas.outcomes.outcome2dict(offer))
#         # print(self.opponent_times[0])
#         # print(self.opponent_times[1])
        
#         if self._best is None:
#             if(self.ufun(offer) > self.ufun.reserved_value):
#                 self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 # print(self.ufun(offer))
#                 # print(self.ufun(self._best))
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer

                    
#         # Find the current aspiration level
#         myasp = aspiration_function(
#             state.relative_time, 1.0, self.ufun.reserved_value, self.e
#         )
#         opasp = aspiration_function(
#             state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
#         )
#         ans = False

#         pat = self.pat * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.fe)
#             ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)

#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(indop!=0):
#                     nextidx = max(indop-1, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time, 1.0, border, self.e)


#         # if(state.relative_time >= 0.97):
#         #     ans = (float(self.ufun(offer)) >= myasp)
#         # else:
#         #     asp = max(myasp,opasp)
#         #     ans = (float(self.ufun(offer)) >= asp)

#         if(state.relative_time + 2 * one_step > 1.0):
#             # print(self.predict)
#             if(float(self.ufun(offer)) > self.ufun.reserved_value):
#                 if(self.opponent_utilities[-1] <= self.opponent_utilities[-2]):
#                     return True
#                     # return False
#                 # myasp = 0.0


#         # accept if the utility of the received offer is higher than
#         # the current aspiration

#         # return ans
#         return (float(self.ufun(offer)) >= myasp)



# ?代目
# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """
#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 17.5,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         self.my_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 0.0
#         self.lasttime = 1.0
#         self.diffmean = 0.01
        
#         # self.most
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         diff = []
#         for i in range(len(self.opponent_times)):
#             diff.append(self.opponent_times[i] - self.my_times[i])
#         # diff = self.opponent_times - self.my_times
#         if(len(diff)==0):
#             diff_mean=0.01
#         else:
#             diff_mean = sum(diff) / len(diff)
        
#         self.diff_mean = diff_mean

#         self.my_times.append(state.relative_time)
#         if self.is_acceptable(state):
#             # print(self.ufun(state.current_offer))
#             # print(self.opponent_ufun(state.current_offer))
#             # print(self.opponent_ufun.reserved_value)
#             # print(len(self.my_times))
#             # print(self.my_times[0])
#             # print(self.my_times[-1])
#             # print(len(self.opponent_times))
#             # print(self.opponent_times[0])
#             # print(self.opponent_times[-1])
#             # print(state.step)
#             # # print(state.time)
#             # print(state.relative_time)
#             # print(diff_mean)
#             return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         if((state.step)==0):
#             one_step = 0.0001
#         else:
#             one_step = (state.relative_time) / (state.step)
#         lasttime = (1.0 // one_step) * one_step
#         self.lasttime = lasttime

#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             my_nash_utility = nash[0][0][0]
#         else:
#             my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         self._min_acceptable = my_nash_utility * self._nash_factor
#         self._min_acceptable = self.ufun.reserved_value
#         # self._min_acceptable = 0
#         # Set the set of outcomes to offer from
#         # self._outcomes = [
#         #     w
#         #     for u, w in zip(my_frontier_utils, frontier_outcomes)
#         #     if u >= self._min_acceptable
#         # ]        
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             if(len(my_frontier_utils)!=0):
#                 min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
#                 for i in frontier_utils:
#                     if(min_nash > i[0] + i[1]):
#                         # print(min_nash)
#                         # print(i[0] + i[1])
#                         min_nash = i[0] + i[1]
#                 # print(min_nash)
            


#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]     

#             if(len(my_frontier_utils)!=0):
#                 for _ in outcomes:
#                     if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= min_nash):
#                         if( _ not in self._outcomes):
#                             self._outcomes.append(_)


#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self._outcomes
#             #         if (my_util := float(self.ufun(_))) > 0
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > 0
#             #     ],
#             # )
#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         max_rational = len(self._rational) - 1
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        

#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
#         # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
#         n_rational = len(self._rational)
#         max_rational = n_rational - 1
#         pat = 0.95 * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.e)
#             ratio = (myasp - self.ufun.reserved_value) / ((lasttime - pat)*(lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)

#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(indop!=0):
#                     nextidx = max(indop, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time / self.lasttime, 1.0, border, self.e)
#         indmy = max_rational
#         while(indmy!=0):
#             nextidx = max(indmy-1 , 0)
#             if(self._rational[nextidx][0] >= myasp):
#                 indmy = nextidx
#             else:
#                 break
        
#         indx = indmy
#         outcome = self._rational[indx][-1]

#         if(state.relative_time + one_step > 1.0):
#             if(self.ufun(self._best) > self.ufun.reserved_value):
#                 outcome = self._best
#         # print(state.step)

#         return SAOResponse(ResponseType.REJECT_OFFER, outcome)
#         # return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
        
#         # print("offer")
#         # print(offer)
        
#         if self._best is None:
#             self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 # print(self.ufun(offer))
#                 # print(self.ufun(self._best))
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer

#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
                    
#         # Find the current aspiration level
#         myasp = aspiration_function(
#             state.relative_time / self.lasttime, 1.0, self.ufun.reserved_value, self.e
#         )
#         opasp = aspiration_function(
#             state.relative_time / self.lasttime, 1.0, self.opponent_ufun.reserved_value, self.e
#         )
#         ans = False

#         pat = 0.95 * self.lasttime
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat / self.lasttime, 1.0, self.ufun.reserved_value, self.e)
#             ratio = (myasp - self.ufun.reserved_value) / ((self.lasttime - pat)*(self.lasttime - pat))
#             xd = (state.relative_time / self.lasttime) - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)

#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(indop!=0):
#                     nextidx = max(indop, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time / self.lasttime, 1.0, border, self.e)


#         # if(state.relative_time >= 0.97):
#         #     ans = (float(self.ufun(offer)) >= myasp)
#         # else:
#         #     asp = max(myasp,opasp)
#         #     ans = (float(self.ufun(offer)) >= asp)


#         # accept if the utility of the received offer is higher than
#         # the current aspiration
#         if(self.opponent_times[-1] + self.diff_mean > 1.0):
#             if(self.ufun(offer) > self.ufun.reserved_value):
#                 return True

#         # return ans
#         return (float(self.ufun(offer)) >= myasp)
















    # def update_reserved_value(self, state: SAOState):
    #     # Learns the reserved value of the partner
    #     assert self.opponent_ufun is not None
    #     # extract the current offer from the state
    #     offer = state.current_offer
    #     if offer is None:
    #         return
    #     # save to the list of utilities received from the opponent and their times
    #     self.opponent_utilities.append(float(self.opponent_ufun(offer)))
    #     self.opponent_times.append(state.relative_time)

    #     # If we do not have enough data, just assume that the opponent
    #     # reserved value is zero
    #     n_unique = len(set(self.opponent_utilities))
    #     if n_unique < self.min_unique_utilities:
    #         self._past_oppnent_rv = 0.0
    #         self.opponent_ufun.reserved_value = 0.0
    #         return
    #     # Use curve fitting to estimate the opponent reserved value
    #     # We assume the following:
    #     # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
    #     # - The opponent never offers outcomes lower than their reserved value which means
    #     #   that their rv must be no higher than the worst outcome they offered for themselves.
    #     bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
    #     err = ""
    #     try:
    #         optimal_vals, _ = curve_fit(
    #             lambda x, e, rv: aspiration_function(
    #                 x, self.opponent_utilities[0], rv, e
    #             ),
    #             self.opponent_times,
    #             self.opponent_utilities,
    #             bounds=bounds,
    #         )
    #         self._past_oppnent_rv = self.opponent_ufun.reserved_value
    #         self.opponent_ufun.reserved_value = optimal_vals[1]
    #     except Exception as e:
    #         err, optimal_vals = f"{str(e)}", [None, None]

    #     # log my estimate
    #     if self._enable_logging:
    #         self.nmi.log_info(
    #             self.id,
    #             dict(
    #                 estimated_rv=self.opponent_ufun.reserved_value,
    #                 n_unique=n_unique,
    #                 opponent_utility=self.opponent_utilities[-1],
    #                 estimated_exponent=optimal_vals[0],
    #                 estimated_max=self.opponent_utilities[0],
    #                 error=err,
    #             ),
    #         )

# 2代目
# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """

#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 17.5,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 0.0
#         # self.most
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         if self.is_acceptable(state):
#             # print(self.ufun(state.current_offer))
#             # print(self.opponent_ufun(state.current_offer))
#             # print(self.opponent_ufun.reserved_value)
#             return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
            
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             my_nash_utility = nash[0][0][0]
#         else:
#             my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         self._min_acceptable = my_nash_utility * self._nash_factor
#         self._min_acceptable = self.ufun.reserved_value
#         # self._min_acceptable = 0
#         # Set the set of outcomes to offer from

      
#         # self._outcomes = [
#         #     w
#         #     for u, w in zip(my_frontier_utils, frontier_outcomes)
#         #     if u >= self._min_acceptable
#         # ]        
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             # or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             # The rational set of outcomes sorted dependingly according to our utility function
#             # and the opponent utility function (in that order).
#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self.nmi.outcome_space.enumerate_or_sample(
#             #             levels=10, max_cardinality=100_000
#             #         )
#             #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > self.opponent_ufun.reserved_value
#             #     ],
#             # )

#             # a = set([])
#             # b = set([])
#             # ori_outcomes = []
#             # tmp = sorted(
#             #     [
#             #         (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
#             #         for _ in outcomes
#             #     ],reverse=True
#             # )

#             # ori_outcomes2 = []
#             # tmp2 = sorted(
#             #     [
#             #         (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
#             #         for _ in outcomes
#             #     ],reverse=True
#             # )
#             # # print(tmp[20])
#             # # i = 0
#             # # print("---------------------------------------------------")
#             # for i in range(len(tmp)):
#             #     # print(i)
#             #     # print(tmp[i])
#             #     if(tmp[i][0] not in a):
#             #         a.add(tmp[i][0])
#             #         if(tmp[i][0] >= self.ufun.reserved_value):
#             #             ori_outcomes.append(tmp[i][-1]) 

#             # for i in range(len(tmp2)):
#             #     # print(i)
#             #     # print(tmp[i])
#             #     if(tmp2[i][0] not in b):
#             #         b.add(tmp2[i][0])
#             #         if(tmp2[i][0] >= self.ufun.reserved_value):
#             #             ori_outcomes2.append(tmp2[i][-1]) 
#             #             # print([tmp[i][0],tmp[i][1]])    
#             #             # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             # c = set(ori_outcomes)
#             # for i in ori_outcomes2:
#             #     c.add(i)
#             # all_outcomes = list(c)
#             # self._outcomes = all_outcomes
#             # print(my_frontier_utils)
#             if(len(my_frontier_utils)!=0):
#                 min_nash = my_frontier_utils[0]+ opp_frontier_utils[0]
#                 for i in frontier_utils:
#                     if(min_nash > i[0] + i[1]):
#                         # print(min_nash)
#                         # print(i[0] + i[1])
#                         min_nash = i[0] + i[1]
#                 # print(min_nash)
            


#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]     

#             if(len(my_frontier_utils)!=0):
#                 for _ in outcomes:
#                     if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= min_nash):
#                         if( _ not in self._outcomes):
#                             self._outcomes.append(_)


#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self._outcomes
#             #         if (my_util := float(self.ufun(_))) > 0
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > 0
#             #     ],
#             # )
#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1

#             self._opprational = sorted(
#                 [
#                     (opp_util, my_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > 0
#                     # if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         max_rational = len(self._rational) - 1
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         # asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
#         # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
#         # n_rational = len(self._rational)
#         # min_indx = max(0, min(max_rational, int(asp * max_rational)))

#         min_indx = self.nidx
        
#         # find current stochasticity which goes down from the set level to zero linearly
#         # s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
#         # find the index of the maximum utility we require based on stochasticity (going down over time)
#         # max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
#         # max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
#         # offer an outcome in the selected range
#         # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

#         # print("outcome")
#         # print(outcome)
#         # if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
#         #     outcome = self._best
#         # if(state.relative_time > 0.95 and self.ufun(self._best) > self.ufun.reserved_value)
#         pat = 0.95
#         border = self.ufun.reserved_value
#         if(state.relative_time >= pat):
#             myasp = aspiration_function(pat, 1.0, self.ufun.reserved_value, self.e)
#             ratio = (myasp - self.ufun.reserved_value) / ((1.0 - pat)*(1.0 - pat))
#             xd = state.relative_time - pat
#             y = myasp - (ratio*xd*xd) 
#             border = max(border,y)

#             # asp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= y):
#             #         indmy = nextidx
#             #     else:
#             #         break
#             # outcome = self._best
#         else:
#             # tmp = 0.0
#             if(state.relative_time > 1.1):
#                 popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 1000.0]))
#                 tmp = popt[1]
#                 # print(state.relative_time)
#                 # print(popt)
#                 indop = len(self._opprational) - 1 
#                 while(self.nidx!=0):
#                     nextidx = max(indop, 0)
#                     if(self._opprational[nextidx][0] >= tmp):
#                         indop = nextidx
#                     else:
#                         break

#                 pre_rv = self._opprational[self.nidx][1]

#                 # self.opponent_ufun.reserved_value = pre_rv
#                 border = max(border,pre_rv)
#             else:
#                 border = self.ufun.reserved_value

#             # myasp = aspiration_function(state.relative_time, 1.0, rv, self.e)
#             # indmy = max_rational
#             # while(self.nidx!=0):
#             #     nextidx = max(indmy-1 , 0)
#             #     if(self._rational[nextidx][0] >= myasp):
#             #         indmy = nextidx
#             #     else:
#             #         break


#         myasp = aspiration_function(state.relative_time, 1.0, border, self.e)
#         indmy = max_rational
#         while(self.nidx!=0):
#             nextidx = max(indmy-1 , 0)
#             if(self._rational[nextidx][0] >= myasp):
#                 indmy = nextidx
#             else:
#                 break
        
#         indx = indmy
#         outcome = self._rational[indx][-1]

#         # return SAOResponse(ResponseType.REJECT_OFFER, outcome)
#         return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
        
#         # print("offer")
#         # print(offer)
        
#         if self._best is None:
#             self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 # print(self.ufun(offer))
#                 # print(self.ufun(self._best))
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer

#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
                    
#         # Find the current aspiration level
#         myasp = aspiration_function(
#             state.relative_time, 1.0, self.ufun.reserved_value, self.e
#         )
#         opasp = aspiration_function(
#             state.relative_time, 1.0, self.opponent_ufun.reserved_value, self.e
#         )
#         ans = False


#         # if(state.relative_time >= 0.97):
#         #     ans = (float(self.ufun(offer)) >= myasp)
#         # else:
#         #     asp = max(myasp,opasp)
#         #     ans = (float(self.ufun(offer)) >= asp)


#         # accept if the utility of the received offer is higher than
#         # the current aspiration

#         # return ans
#         return (float(self.ufun(offer)) >= myasp)

#     # def update_reserved_value(self, state: SAOState):
#     #     # Learns the reserved value of the partner
#     #     assert self.opponent_ufun is not None
#     #     # extract the current offer from the state
#     #     offer = state.current_offer
#     #     if offer is None:
#     #         return
#     #     # save to the list of utilities received from the opponent and their times
#     #     self.opponent_utilities.append(float(self.opponent_ufun(offer)))
#     #     self.opponent_times.append(state.relative_time)

#     #     # If we do not have enough data, just assume that the opponent
#     #     # reserved value is zero
#     #     n_unique = len(set(self.opponent_utilities))
#     #     if n_unique < self.min_unique_utilities:
#     #         self._past_oppnent_rv = 0.0
#     #         self.opponent_ufun.reserved_value = 0.0
#     #         return
#     #     # Use curve fitting to estimate the opponent reserved value
#     #     # We assume the following:
#     #     # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
#     #     # - The opponent never offers outcomes lower than their reserved value which means
#     #     #   that their rv must be no higher than the worst outcome they offered for themselves.
#     #     bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
#     #     err = ""
#     #     try:
#     #         optimal_vals, _ = curve_fit(
#     #             lambda x, e, rv: aspiration_function(
#     #                 x, self.opponent_utilities[0], rv, e
#     #             ),
#     #             self.opponent_times,
#     #             self.opponent_utilities,
#     #             bounds=bounds,
#     #         )
#     #         self._past_oppnent_rv = self.opponent_ufun.reserved_value
#     #         self.opponent_ufun.reserved_value = optimal_vals[1]
#     #     except Exception as e:
#     #         err, optimal_vals = f"{str(e)}", [None, None]

#     #     # log my estimate
#     #     if self._enable_logging:
#     #         self.nmi.log_info(
#     #             self.id,
#     #             dict(
#     #                 estimated_rv=self.opponent_ufun.reserved_value,
#     #                 n_unique=n_unique,
#     #                 opponent_utility=self.opponent_utilities[-1],
#     #                 estimated_exponent=optimal_vals[0],
#     #                 estimated_max=self.opponent_utilities[0],
#     #                 error=err,
#     #             ),
#     #         )

# class Shochan(SAONegotiator):
#     """A simple negotiator that uses curve fitting to learn the reserved value.

#     Args:
#         min_unique_utilities: Number of different offers from the opponent before starting to
#                               attempt learning their reserved value.
#         e: The concession exponent used for the agent's offering strategy
#         stochasticity: The level of stochasticity in the offers.
#         enable_logging: If given, a log will be stored  for the estimates.

#     Remarks:

#         - Assumes that the opponent is using a time-based offering strategy that offers
#           the outcome at utility $u(t) = (u_0 - r) - r \\exp(t^e)$ where $u_0$ is the utility of
#           the first offer (directly read from the opponent ufun), $e$ is an exponent that controls the
#           concession rate and $r$ is the reserved value we want to learn.
#         - After it receives offers with enough different utilities, it starts finding the optimal values
#           for $e$ and $r$.
#         - When it is time to respond, RVFitter, calculates the set of rational outcomes **for both agents**
#           based on its knowledge of the opponent ufun (given) and reserved value (learned). It then applies
#           the same concession curve defined above to concede over an ordered list of these outcomes.
#         - Is this better than using the same concession curve on the outcome space without even trying to learn
#           the opponent reserved value? Maybe sometimes but empirical evaluation shows that it is not in general.
#         - Note that the way we check for availability of enough data for training is based on the uniqueness of
#           the utility of offers from the opponent (for the opponent). Given that these are real values, this approach
#           is suspect because of rounding errors. If two outcomes have the same utility they may appear to have different
#           but very close utilities because or rounding errors (or genuine very small differences). Such differences should
#           be ignored.
#         - Note also that we start assuming that the opponent reserved value is 0.0 which means that we are only restricted
#           with our own reserved values when calculating the rational outcomes. This is the best case scenario for us because
#           we have MORE negotiation power when the partner has LOWER utility.
#     """

#     def __init__(
#         self,
#         *args,
#         min_unique_utilities: int = 10,
#         e: float = 15.0,
#         stochasticity: float = 0.1,
#         enable_logging: bool = False,
#         nash_factor=0.1,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.min_unique_utilities = min_unique_utilities
#         self.e = e
#         self.stochasticity = stochasticity
#         # keeps track of times at which the opponent offers
#         self.opponent_times: list[float] = []
#         # keeps track of opponent utilities of its offers
#         self.opponent_utilities: list[float] = []
#         # keeps track of the our last estimate of the opponent reserved value
#         self._past_oppnent_rv = 0.0
#         # keeps track of the rational outcome set given our estimate of the
#         # opponent reserved value and our knowledge of ours
#         self._rational: list[tuple[float, float, Outcome]] = []
#         self._enable_logging = enable_logging
#         self.preoffer = []

#         self._outcomes: list[Outcome] = []
#         self._min_acceptable = float("inf")
#         self._nash_factor = nash_factor
#         self._best: Outcome = None  # type: ignore
#         self.nidx = 0
#         self.opponent_ufun.reserved_value = 25.0
#         # self.most
        

#     def __call__(self, state: SAOState) -> SAOResponse:
#         assert self.ufun and self.opponent_ufun
#         # update the opponent reserved value in self.opponent_ufun
#         # self.update_reserved_value(state)
#         # rune the acceptance strategy and if the offer received is acceptable, accept it
#         if self.is_acceptable(state):
#             # print(self.opponent_times)
#             # print(self.opponent_utilities)
#             # print(self.opponent_ufun(state.current_offer))
#             # print(self.opponent_ufun.reserved_value)
#             # xdata = 
#             # print(self.opponent_)
#             popt, pcov = curve_fit(aspiration_function, self.opponent_times, self.opponent_utilities,bounds=(0, [1.0, 1.0, 100.0]))
#             # print(state.current_proposer_agent)
#             # print(state.new_offerer_agents)
#             # print(self.annotation)
#             # print(self.capabilities)
#             # print(self.crisp_ufun)
#             # print(self.id)
#             # print(self.name)
#             # print(self.owner)
#             # print(self.parent)
#             # print(self.preferences)
#             # print(self.private_info)
#             # print(self.prob_ufun)
#             # print(self.reserved_outcome)
#             # print(self.short_type_name)
#             # print(self.type_name)
#             # print(self.ufun)
#             # print(self.uuid)
#             print(state.relative_time)
#             print(popt)
#             # plt.plot(self.opponent_times, aspiration_function(self.opponent_times, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#             # plt.xlabel('x')
#             # plt.ylabel('y')
#             # plt.legend()
#             # plt.savefig("/root/negmas/")  
#             # plt.show()
#             return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
#         else:
#             self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
#         # The offering strategy
            
#         # nash
#         ufuns = (self.ufun, self.opponent_ufun)
#         # list all outcomes
#         outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
#         # print(outcomes)
#         # find the pareto-front and the nash point
#         frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
#         frontier_outcomes = [outcomes[_] for _ in frontier_indices]
#         # print(frontier_outcomes)
#         # print(frontier_utils)
#         my_frontier_utils = [_[0] for _ in frontier_utils]
#         opp_frontier_utils = [_[1] for _ in frontier_utils]
#         # print(my_frontier_utils)
#         # print(opp_frontier_utils)
#         nash = nash_points(ufuns, frontier_utils)  # type: ignore
#         if nash:
#             # find my utility at the Nash Bargaining Solution.
#             my_nash_utility = nash[0][0][0]
#         else:
#             my_nash_utility = 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
#         # Set the acceptable utility limit
#         self._min_acceptable = my_nash_utility * self._nash_factor
#         self._min_acceptable = self.ufun.reserved_value
#         # self._min_acceptable = 0
#         # Set the set of outcomes to offer from

      
#         # self._outcomes = [
#         #     w
#         #     for u, w in zip(my_frontier_utils, frontier_outcomes)
#         #     if u >= self._min_acceptable
#         # ]        
    
#         # We only update our estimate of the rational list of outcomes if it is not set or
#         # there is a change in estimated reserved value
#         if (
#             not self._rational
#             or abs(self.opponent_ufun.reserved_value - self._past_oppnent_rv) > 1e-3
#         ):
#             # The rational set of outcomes sorted dependingly according to our utility function
#             # and the opponent utility function (in that order).
#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self.nmi.outcome_space.enumerate_or_sample(
#             #             levels=10, max_cardinality=100_000
#             #         )
#             #         if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > self.opponent_ufun.reserved_value
#             #     ],
#             # )
#             a = set([])
#             b = set([])
#             ori_outcomes = []
#             tmp = sorted(
#                 [
#                     (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
#                     for _ in outcomes
#                 ],reverse=True
#             )

#             ori_outcomes2 = []
#             tmp2 = sorted(
#                 [
#                     (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
#                     for _ in outcomes
#                 ],reverse=True
#             )
#             # print(tmp[20])
#             # i = 0
#             # print("---------------------------------------------------")
#             for i in range(len(tmp)):
#                 # print(i)
#                 # print(tmp[i])
#                 if(tmp[i][0] not in a):
#                     a.add(tmp[i][0])
#                     if(tmp[i][0] >= self.ufun.reserved_value):
#                         ori_outcomes.append(tmp[i][-1]) 

#             for i in range(len(tmp2)):
#                 # print(i)
#                 # print(tmp[i])
#                 if(tmp2[i][0] not in b):
#                     b.add(tmp2[i][0])
#                     if(tmp2[i][0] >= self.ufun.reserved_value):
#                         ori_outcomes2.append(tmp2[i][-1]) 
#                         # print([tmp[i][0],tmp[i][1]])    
#                         # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             c = set(ori_outcomes)
#             for i in ori_outcomes2:
#                 c.add(i)
#             all_outcomes = list(c)
#             self._outcomes = all_outcomes
#             # print()

#             # min_nash = my_frontier_utils[0]+opp_frontier_utils[0]
#             # for i in frontier_utils:
#             #     if(min_nash > i[0] + i[1]):
#             #         print(min_nash[0]+min_nash[1])
#             #         print(i[0] + i[1])
#             #         min_nash = i
#                 # print(min_nash)
            

#             self._outcomes = [
#                 w
#                 for u, w in zip(my_frontier_utils, frontier_outcomes)
#                 if u >= self.ufun.reserved_value
#             ]
#             # print(self._outcomes)

#             # for _ in outcomes:
#             #     if (float(self.ufun(_)) + float(self.opponent_ufun(_)) >= min_nash[0] + min_nash[1]):
#             #         if( _ not in self._outcomes):
#             #             self._outcomes.append(_)


#             # self._rational = sorted(
#             #     [
#             #         (my_util, opp_util, _)
#             #         for _ in self._outcomes
#             #         if (my_util := float(self.ufun(_))) > 0
#             #         and (opp_util := float(self.opponent_ufun(_)))
#             #         > 0
#             #     ],
#             # )
   
#             self._rational = sorted(
#                 [
#                     (my_util, opp_util, _)
#                     for _ in self._outcomes
#                     if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
#                     and (opp_util := float(self.opponent_ufun(_)))
#                     > 0
#                 ],
#             )
#             self.nidx = len(self._rational)-1
#             # print(self.nidx)
#         # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
#         # then just revert to offering our top offer
#         if not self._rational:
#             return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
#         # find our aspiration level (value between 0 and 1) the higher the higher utility we require
#         asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)


#         n_rational = len(self._rational)
#         max_rational = n_rational - 1
#         min_indx = max(0, min(max_rational, int(asp * max_rational)))

#         asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
#         while(self.nidx!=0):
#             nextidx = max(self.nidx-1 , 0)
#             if(self._rational[nextidx][0] > asp):
#                 self.nidx = nextidx
#             else:
#                 break

#         # kaiki
#         # if self.opponent_times == [] or self.opponent_utilities == []:
#         #     popt, pcov = curve_fit(aspiration_function, self.opponent_times , self.opponent_utilities,bounds=(0, [1.0, 1.0, 100.0]))
#         #     self.opponent_ufun.reserved_value = popt[1] - 0.10

#         # best_utill = 0
#         # for _ in frontier_utils:
#         #     my_util = _[0]
#         #     opp_util = _[1]
#         #     if(opp_util >= self.opponent_ufun.reserved_value):
#         #         best_utill = max(best_utill,my_util)
#         # asp = aspiration_function(state.relative_time, 1.0, best_utill, self.e)

#         # self.nidx = max_rational
#         # while(self.nidx!=0):
#         #     nextidx = max(self.nidx-1 , 0)
#         #     if(self._rational[nextidx][0] > asp):
#         #         self.nidx = nextidx
#         #     else:
#         #         break

#         min_indx = self.nidx
        
#         # find current stochasticity which goes down from the set level to zero linearly
#         s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
#         # find the index of the maximum utility we require based on stochasticity (going down over time)
#         max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
#         max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
#         # offer an outcome in the selected range
#         # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

#         indx = self.nidx
        
#         outcome = self._rational[indx][-1]
#         # print("outcome")
#         # print(outcome)
#         if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
#             outcome = self._best

#         if(state.relative_time > 0.99 and self.ufun(self._best) > self.ufun.reserved_value):
#             outcome = self._best
#         return SAOResponse(ResponseType.REJECT_OFFER, outcome)

#     def is_acceptable(self, state: SAOState) -> bool:
#         # The acceptance strategy
#         assert self.ufun and self.opponent_ufun
#         # get the offer from the mechanism state
#         offer = state.current_offer
#         # If there is no offer, there is nothing to accept
#         if offer is None:
#             return False
#         self.opponent_times.append(state.relative_time)
#         self.opponent_utilities.append(self.opponent_ufun(offer))
        
#         # print("offer")
#         # print(offer)
        
#         if self._best is None:
#             self._best = offer
#         else:
#             if(self.ufun(offer) > self.ufun(self._best)):
#                 # print(self.ufun(offer))
#                 # print(self.ufun(self._best))
#                 self._best = offer
#             elif(self.ufun(offer) == self.ufun(self._best)):
#                 if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
#                     self._best = offer
                    
#         # Find the current aspiration level
#         asp = aspiration_function(
#             state.relative_time, 1.0, self.ufun.reserved_value, self.e
#         )
#         # accept if the utility of the received offer is higher than
#         # the current aspiration

#         return float(self.ufun(offer)) >= asp

#     def update_reserved_value(self, state: SAOState):
#         # Learns the reserved value of the partner
#         assert self.opponent_ufun is not None
#         # extract the current offer from the state
#         offer = state.current_offer
#         if offer is None:
#             return
#         # save to the list of utilities received from the opponent and their times
#         self.opponent_utilities.append(float(self.opponent_ufun(offer)))
#         self.opponent_times.append(state.relative_time)

#         # If we do not have enough data, just assume that the opponent
#         # reserved value is zero
#         n_unique = len(set(self.opponent_utilities))
#         if n_unique < self.min_unique_utilities:
#             self._past_oppnent_rv = 0.0
#             self.opponent_ufun.reserved_value = 0.0
#             return
#         # Use curve fitting to estimate the opponent reserved value
#         # We assume the following:
#         # - The opponent is using a concession strategy with an exponent between 0.2, 5.0
#         # - The opponent never offers outcomes lower than their reserved value which means
#         #   that their rv must be no higher than the worst outcome they offered for themselves.
#         bounds = ((0.2, 0.0), (5.0, min(self.opponent_utilities)))
#         err = ""
#         try:
#             optimal_vals, _ = curve_fit(
#                 lambda x, e, rv: aspiration_function(
#                     x, self.opponent_utilities[0], rv, e
#                 ),
#                 self.opponent_times,
#                 self.opponent_utilities,
#                 bounds=bounds,
#             )
#             self._past_oppnent_rv = self.opponent_ufun.reserved_value
#             self.opponent_ufun.reserved_value = optimal_vals[1]
#         except Exception as e:
#             err, optimal_vals = f"{str(e)}", [None, None]

#         # log my estimate
#         if self._enable_logging:
#             self.nmi.log_info(
#                 self.id,
#                 dict(
#                     estimated_rv=self.opponent_ufun.reserved_value,
#                     n_unique=n_unique,
#                     opponent_utility=self.opponent_utilities[-1],
#                     estimated_exponent=optimal_vals[0],
#                     estimated_max=self.opponent_utilities[0],
#                     error=err,
#                 ),
#             )





class Shochan_base75(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base50(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base100(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base125(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base150(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base175(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

class Shochan_base200(SAONegotiator):
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
        self.opponent_ufun.reserved_value = 0.0
        # self.most
        

    def __call__(self, state: SAOState) -> SAOResponse:
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
            self.preoffer.append((self.ufun(state.current_offer),self.opponent_ufun(state.current_offer)))
        # The offering strategy
            
        # nash
        ufuns = (self.ufun, self.opponent_ufun)
        # list all outcomes
        outcomes = list(self.ufun.outcome_space.enumerate_or_sample())
        # find the pareto-front and the nash point
        frontier_utils, frontier_indices = pareto_frontier(ufuns, outcomes)
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

      
        # self._outcomes = [
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
            #     [
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
                [
                    (float(self.ufun(_)), float(self.opponent_ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )

            ori_outcomes2 = []
            tmp2 = sorted(
                [
                    (float(self.opponent_ufun(_)), float(self.ufun(_)), _)
                    for _ in outcomes
                ],reverse=True
            )
            # print(tmp[20])
            # i = 0
            # print("---------------------------------------------------")
            for i in range(len(tmp)):
                # print(i)
                # print(tmp[i])
                if(tmp[i][0] not in a):
                    a.add(tmp[i][0])
                    if(tmp[i][0] >= self.ufun.reserved_value):
                        ori_outcomes.append(tmp[i][-1]) 

            for i in range(len(tmp2)):
                # print(i)
                # print(tmp[i])
                if(tmp2[i][0] not in b):
                    b.add(tmp2[i][0])
                    if(tmp2[i][0] >= self.ufun.reserved_value):
                        ori_outcomes2.append(tmp2[i][-1]) 
                        # print([tmp[i][0],tmp[i][1]])    
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            c = set(ori_outcomes)
            for i in ori_outcomes2:
                c.add(i)
            all_outcomes = list(c)
            self._outcomes = all_outcomes
            # print()

            self._outcomes = [
                w
                for u, w in zip(my_frontier_utils, frontier_outcomes)
                if u >= self.ufun.reserved_value
            ]     


            # self._rational = sorted(
            #     [
            #         (my_util, opp_util, _)
            #         for _ in self._outcomes
            #         if (my_util := float(self.ufun(_))) > 0
            #         and (opp_util := float(self.opponent_ufun(_)))
            #         > 0
            #     ],
            # )
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self._outcomes
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                    and (opp_util := float(self.opponent_ufun(_)))
                    > 0
                ],
            )
            self.nidx = len(self._rational)-1
            # print(self.nidx)
        # If there are no rational outcomes (i.e. our estimate of the opponent rv is very wrogn),
        # then just revert to offering our top offer
        if not self._rational:
            return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())
        # find our aspiration level (value between 0 and 1) the higher the higher utility we require
        asp = aspiration_function(state.relative_time, 1.0, 0.0, self.e)
        # find the index of the rational outcome at the aspiration level (in the rational set of outcomes)
        n_rational = len(self._rational)
        max_rational = n_rational - 1
        min_indx = max(0, min(max_rational, int(asp * max_rational)))

        asp = aspiration_function(state.relative_time, 1.0, self.ufun.reserved_value, self.e)
        while(self.nidx!=0):
            nextidx = max(self.nidx-1 , 0)
            if(self._rational[nextidx][0] > asp):
                self.nidx = nextidx
            else:
                break

        min_indx = self.nidx
        
        # find current stochasticity which goes down from the set level to zero linearly
        s = aspiration_function(state.relative_time, self.stochasticity, 0.0, 1.0)
        # find the index of the maximum utility we require based on stochasticity (going down over time)
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        max_indx = max(0, min(int(min_indx + s * n_rational), max_rational))
        # offer an outcome in the selected range
        # indx = random.randint(min_indx, max_indx) if min_indx != max_indx else min_indx

        indx = self.nidx
        
        outcome = self._rational[indx][-1]
        # print("outcome")
        # print(outcome)
        if(self.ufun(self._best) > self.ufun(outcome) and self.ufun(self._best) > self.ufun.reserved_value):
            outcome = self._best

        if(state.relative_time > 0.98 and self.ufun(self._best) > self.ufun.reserved_value):
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
            if(self.ufun(offer) > self.ufun(self._best)):
                # print(self.ufun(offer))
                # print(self.ufun(self._best))
                self._best = offer
            elif(self.ufun(offer) == self.ufun(self._best)):
                if(self.opponent_ufun(offer) < self.opponent_ufun(self._best)):
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

