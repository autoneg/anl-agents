"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Knights of the NegotiOtters
*Authors* Xintong Wang: xintong.wang@rutgers.edu; Tri-an Nguyen: tdn39@scarletmail.rutgers.edu; Garrett Seo: garrett.seo@rutgers.edu

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from collections import defaultdict
import math
from random import random, choice, randint
import statistics
from negmas import SAONMI, PolyAspiration, Outcome

from .helpers.helperfunctions import *
from .game import GameEnvironment, AbstractedOutcomeGameState
# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.

from anl2025.negotiator import ANL2025Negotiator
from anl2025.ufun import MaxCenterUFun, LinearCombinationCenterUFun, MeanSMCenterUFun
from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing import Tuple


__all__ = [
    "NaiveAgent",
    "LookAheadCurrentUtilityNaive",
    "LookAheadRolloutsNaive",
    "NoAgreement",
    "ConcedingAgent",
    "ConcedingLookAheadAgent",
    "LookAheadCurrentUtilityNaiveSoftMaxAdj",
    "NewBoulwareLookAhead",
    "BoulwareLookAhead",
    "UtilityFitAgent",
    "NewerUtilityFitLookAheadAgent",
    "RUFL",
    "NewishUtilityFitLookAheadAgent",
    "UtilityFitLookAheadAgent",
    "AbinesLookAheadAgent",
]


class NewNegotiator(ANL2025Negotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    This example agent aims for the absolute best bid available. As a center agent, it adapts its strategy after each negotiation, by aiming for the best bid GIVEN the previous outcomes.
    """

    """
       The most general way to implement an agent is to implement propose and respond.
       """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # print("init")

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).

        Remarks:
            - You can use the negotiator_id to identify what side negotiator is currently proposing. This id is stable within a negotiation.
        """
        # If the negotiation has ended, update the strategy. The subnegotiator may of may not have found an agreement: this affects the strategy for the rest of the negotiation.
        if did_negotiation_end(self):
            self._update_strategy()

        bid = get_target_bid_at_current_index(self)

        # if you want to end the negotiation, return None
        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """Responds to the given partner (source) using the side negotiator (negotiator_id).

        Remarks:
            - negotiator_id is the ID of the side negotiator representing this agent.
            - source: is the ID of the partner.
            - the mapping from negotiator_id to source is stable within a negotiation.

        """
        if did_negotiation_end(self):
            self._update_strategy()

        # This agent is very stubborn: it only accepts an offer if it is EXACTLY the target bid it wants to have.
        if state.current_offer is get_target_bid_at_current_index(self):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
        # You can also return ResponseType.END_NEGOTIATION to end the negotiation.

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""
        # if your current role is the edge agent, use the strategy as if your centeragent is in it's last subnegotiation.
        # In this case, just get the best bid from the utility function.
        if is_edge_agent(self):
            # note that the edge utility function has a slightly different structure than a center utility function.
            _, best_bid = self.ufun.extreme_outcomes()
        else:
            # get the best bid from the outcomes that are still possible to achieve.
            best_bid = find_best_bid_in_outcomespace(self)

        self.target_bid = best_bid
        # print(self.target_bid)


class NaiveAgent(ANL2025Negotiator):
    """
    At propose, greedily finds best bid in the subnegotation via the sideufun.
    Filters by opponent bids and reservation value.
    Concedes linearly from this filtered space.

    At respond, follows the same logic as propose, but
    accepts if the opponent's offer is better than the target bid.
    """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # print("init")

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        bid = None
        nmi = self.negotiators[negotiator_id][0].nmi
        current_negotation_index = get_current_negotiation_index(self)
        ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[current_negotation_index]
        )
        subnegotation_trace = nmi.trace

        total_steps = nmi.n_steps
        current_step = state.step  # think it is originally 0 indexed

        outcome_space = ufun.outcome_space.enumerate_or_sample()
        reservation_value = ufun.reserved_value

        opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]

        util_of_best_opponent_bid = 0
        for opponent_bid in opponent_bids:
            if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                util_of_best_opponent_bid = ufun.eval(opponent_bid)

        outcome_space = [
            x
            for x in outcome_space
            if ufun.eval(x) >= util_of_best_opponent_bid
            and ufun.eval(x) > reservation_value
        ]
        outcome_space.sort(key=ufun.eval, reverse=True)

        # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
        step_percentage = current_step / (total_steps)

        if outcome_space:
            target_index = math.ceil(step_percentage * len(outcome_space))
            target_index = min(
                target_index, len(outcome_space) - 1
            )  # make sure it is not out of bounds in case step_percentage is 1
            bid = outcome_space[target_index]

        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        target_bid = None
        nmi = self.negotiators[negotiator_id][0].nmi
        current_negotation_index = get_current_negotiation_index(self)
        ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[current_negotation_index]
        )
        subnegotation_trace = nmi.trace

        total_steps = nmi.n_steps
        current_step = state.step  # think it is originally 0 indexed

        outcome_space = ufun.outcome_space.enumerate_or_sample()
        reservation_value = ufun.reserved_value

        opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]

        if current_step == total_steps - 1:  # last step
            if ufun.eval(state.current_offer) > reservation_value:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        outcome_space = ufun.outcome_space.enumerate_or_sample()

        util_of_best_opponent_bid = 0
        for opponent_bid in opponent_bids:
            if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                util_of_best_opponent_bid = ufun.eval(opponent_bid)

        outcome_space = [
            x
            for x in outcome_space
            if ufun.eval(x) >= util_of_best_opponent_bid
            and ufun.eval(x) > reservation_value
        ]
        outcome_space.sort(key=ufun.eval, reverse=True)

        # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
        step_percentage = (current_step + 1) / (
            total_steps
        )  # don't have to accept the best offer, willing to concede

        if outcome_space:
            target_index = math.ceil(step_percentage * len(outcome_space))
            target_index = min(target_index, len(outcome_space) - 1)
            target_bid = outcome_space[target_index]

        target_util = ufun.eval(target_bid) if target_bid is not None else math.inf

        if ufun.eval(state.current_offer) >= target_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


class LookAheadCurrentUtilityNaive(ANL2025Negotiator):
    """
    Agent that uses a lookahead strategy to estimated expected utilities for an action.
    It's done with the current utility.
    Concedes linearly with the deadline.
    """

    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if not is_edge_agent(self) and self.lookahead_required:
            bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            if state.step == 0:
                # If this is the first step, we need to perform the look ahead
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

            # if the current step
            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                bid = filtered_children[target_index].history[-1]

        else:
            bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed
            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value
            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_index = 0
                bid = outcome_space[target_index]

        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if not is_edge_agent(self) and self.lookahead_required:
            target_bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            # If you are first to respond, you need to perform look ahead
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
            if not has_made_offer:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            opponent_current_offer = current_game_state.get_child_from_action(
                state.current_offer
            )

            # Because this is last step, maybe just use ufun eval
            if current_step == total_steps - 1:  # last step
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede
            child_of_target_bid = None
            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                child_of_target_bid = filtered_children[target_index]

                if (
                    self.value_dictionary[opponent_current_offer]
                    >= self.value_dictionary[child_of_target_bid]
                ):
                    return ResponseType.ACCEPT_OFFER
            else:
                # If there are no filtered children, just accept the offer
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER
        else:
            target_bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value

            if current_step == total_steps - 1:  # last step
                if ufun.eval(state.current_offer) > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_bid = outcome_space[target_index]

            target_util = ufun.eval(target_bid) if target_bid is not None else math.inf

            if ufun.eval(state.current_offer) >= target_util:
                return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_uniform_distribution_on_outcomes(
        self, num_outcomes: int
    ) -> NDArray[np.float64]:
        """
        Returns a uniform probability distribution over the number of children.
        """
        return np.full(num_outcomes, 1.0 / num_outcomes, dtype=np.float64)

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class LookAheadRolloutsNaive(ANL2025Negotiator):
    """
    Agent that uses a lookahead strategy to estimated expected utilities for an action.
    It's done with rolling out and aggregating the utilities.
    Concedes linearly with the deadline.
    """

    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            # Calculate depth and number of rollouts for lookahead
            self.depth, self.num_rollouts = self.get_maximum_depth_limit_and_rollouts()

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if not is_edge_agent(self) and self.lookahead_required:
            bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            if state.step == 0:
                # If this is the first step, we need to perform the look ahead
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                    num_samples=self.num_rollouts,
                )

            # if the current step
            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                bid = filtered_children[target_index].history[-1]

        else:
            bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed
            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value
            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_index = 0
                bid = outcome_space[target_index]

        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if not is_edge_agent(self) and self.lookahead_required:
            target_bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            # If you are first to respond, you need to perform look ahead
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
            if not has_made_offer:
                # If this is the first step, we need to perform the look ahead
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                    num_samples=self.num_rollouts,
                )

            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            opponent_current_offer = current_game_state.get_child_from_action(
                state.current_offer
            )

            # Because this is last step, maybe just use ufun eval
            if current_step == total_steps - 1:  # last step
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede
            child_of_target_bid = None
            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                child_of_target_bid = filtered_children[target_index]

                if (
                    self.value_dictionary[opponent_current_offer]
                    >= self.value_dictionary[child_of_target_bid]
                ):
                    return ResponseType.ACCEPT_OFFER
            else:
                # If there are no filtered children, just accept the offer
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER
        else:
            target_bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value

            if current_step == total_steps - 1:  # last step
                if ufun.eval(state.current_offer) > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_bid = outcome_space[target_index]

            target_util = ufun.eval(target_bid) if target_bid is not None else math.inf

            if ufun.eval(state.current_offer) >= target_util:
                return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self,
        state: AbstractedOutcomeGameState,
        discount: float,
        depth_limit: int,
        num_samples: int = 100,
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_rollouts(
                state, num_samples
            )
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [
                self.lookahead(child, discount, depth_limit, num_samples)
                for child in children
            ]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def estimate_with_rollouts(
        self, state: AbstractedOutcomeGameState, num_samples: int
    ) -> float:
        current_utility = state.get_current_utility()
        utilities = np.empty(num_samples)

        # Take a sample of the future outcomes
        starting_agreement = list(state.history)

        for sample_index in range(num_samples):
            sampled_agreement = starting_agreement.copy()
            for i in range(
                state.get_current_negotiation_index(), len(self.negotiators)
            ):
                # Sample a random outcome from the possible outcomes
                sampled_outcome = choice(self.outcome_space_at_subnegotiation[i])
                sampled_agreement.append(sampled_outcome)

            utilities[sample_index] = self.ufun(sampled_agreement)

        distribution = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=utilities,
            temperature=1,
            nsteps=self.get_nsteps_for_negotation_index(
                get_current_negotiation_index(self)
            ),
        )

        return np.dot(utilities, distribution)

    def get_uniform_distribution_on_outcomes(
        self, num_outcomes: int
    ) -> NDArray[np.float64]:
        """
        Returns a uniform probability distribution over the number of children.
        """
        return np.full(num_outcomes, 1.0 / num_outcomes, dtype=np.float64)

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit_and_rollouts(self) -> Tuple[int, int]:
        """
        Returns the maximum depth limit for lookahead and rollouts
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)
        num_calls = 0

        while True:
            new_num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                new_num_calls += calls_in_lookahead * 100

            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            new_num_calls += calls_in_normal_search
            if new_num_calls <= self.max_num_calls:
                current_depth_limit += 1
                num_calls = new_num_calls
            else:
                break

        # Given the current depth limit, calculate the number of rollouts
        rollout_multiplier = self.max_num_calls / num_calls
        num_rollouts = int(rollout_multiplier * 100)

        return current_depth_limit - 1, num_rollouts


class NoAgreement(ANL2025Negotiator):
    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # print("init")

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # if not is_edge_agent(self):
        #     print("The reserved value is: ")
        #     print(self.reserved_value)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        return None

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        return ResponseType.REJECT_OFFER


class ConcedingAgent(ANL2025Negotiator):
    """ """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # print("init")

        # Initalize variables
        self.current_util = 0  # current realized util, hold across neg

        self.current_neg_index = -1
        self.current_ufun = None
        self.target_bid = None
        self.max_util = float("inf")

        self.best_oppo_bid = None
        self.best_oppo_bid_util = 0
        self.potential_bids = []
        self.potential_bids_util = []
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float("inf")
        self.next_bid_util = float("inf")
        self.util_difference_median = 0

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.debug = False

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if did_negotiation_end(self):
            self._update_strategy()

        nmi = self.negotiators[negotiator_id][0].nmi

        # utility cannot be improved by any further neg
        if not self.potential_bids:
            return None

        # Concede as a function of
        # (1) index of subnegotiation: self.finished_negotiators vs. len(self.negotiators) vs. self.unfinished_negotiators
        # (2) progress within a subneg: state.step vs. nmi.n_steps
        # (3) whether opponent is conceding
        # (4) concede_degree: how far is the util of current bid to max(self.current_util, self.best_oppo_bid_util)

        perc_subnegs = float(
            (len(self.finished_negotiators) + 1) / len(self.negotiators)
        )

        # if self.debug:
        #     print(perc_subnegs)
        #     print(state.relative_time)

        # TODO: Concede differently for center and edge
        first_time_bid = self.current_bid_util == float("inf")
        self.current_bid_util = self.current_ufun.eval(
            self.potential_bids[self.conceding_index]
        )

        if (
            not first_time_bid
            and self.current_bid_util > self.best_oppo_bid_util
            and self.conceding_index < len(self.potential_bids) - 1
        ):
            self.next_bid_util = self.current_ufun.eval(
                self.potential_bids[self.conceding_index + 1]
            )
            if self.next_bid_util > self.best_oppo_bid_util:
                if self.current_bid_util - self.next_bid_util == 0:
                    util_diff_multiplier = 4
                else:
                    util_diff_multiplier = min(
                        4,
                        float(
                            self.util_difference_median
                            / (self.current_bid_util - self.next_bid_util)
                        ),
                    )

                not_too_aggressive = (
                    float((self.conceding_index + 1) / len(self.potential_bids))
                    < state.relative_time
                )
                relative_standard = self.conceding_index - len(set(self.oppo_bids)) < 3
                # TODO: check on util normalization
                concede_degree = self.next_bid_util - max(
                    self.current_util, self.best_oppo_bid_util
                )

                # NOTE: my very heuristic strategy
                if (
                    relative_standard
                    and not_too_aggressive
                    and state.relative_time * concede_degree * util_diff_multiplier
                    > 0.5 * (1 - perc_subnegs)
                ):  # * self.current_util:
                    self.conceding_index += 1

                    # if not is_edge_agent(self) and self.debug:
                    #     print(is_edge_agent(self), perc_subnegs)
                    #     print(state.relative_time)
                    #     print("concede degree {}, self {}, oppo {}".format(concede_degree, self.conceding_index, len(set(self.oppo_bids))))
                    #     print("{} Conceding to {}: util {} diff {} index {}".format(self, self.potential_bids[self.conceding_index], self.next_bid_util, self.current_bid_util - self.next_bid_util, self.conceding_index))

        # Last chance to propose (TODO: -1 for center; -2 for edge; or according to randomization)
        if (
            is_edge_agent(self) and state.step == nmi.n_steps - 2
        ) or state.step == nmi.n_steps - 1:
            temp_reserve = (
                self.current_ufun.reserved_value if is_edge_agent(self) else 0
            )
            if self.best_oppo_bid_util > max(self.current_util, temp_reserve):
                # if self.debug:
                #     print("{} ConcedingAgent LAST propose {}: util {}".format(self, self.best_oppo_bid, self.current_ufun.eval(self.best_oppo_bid)))
                return self.best_oppo_bid
            else:
                # return None
                return self.potential_bids[0]

        # if self.debug:
        #     print("{} ConcedingAgent propose {}".format(self, self.potential_bids[self.conceding_index]))

        self.current_bid_util = self.current_ufun.eval(
            self.potential_bids[self.conceding_index]
        )
        return self.potential_bids[self.conceding_index]

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if did_negotiation_end(self):
            self._update_strategy()

        # utility cannot be improved by any further neg
        if not self.potential_bids:
            return ResponseType.END_NEGOTIATION

        nmi = self.negotiators[negotiator_id][0].nmi
        current_offer_util = self.current_ufun.eval(state.current_offer)

        # Last chance to respond (NOTE: edge only, center has one more chance to propose)
        # TODO: adapt based on whether last to propose
        if is_edge_agent(self) and state.step == nmi.n_steps - 1:
            if current_offer_util > max(
                self.current_util, self.current_ufun.reserved_value
            ):
                # if self.debug:
                #     print("{} ConcedingAgent respond last step: accept".format(self))
                return ResponseType.ACCEPT_OFFER
            else:
                # if self.debug:
                #     print("{} ConcedingAgent respond last step: reject".format(self))
                return ResponseType.REJECT_OFFER

        self.oppo_bids.append(state.current_offer)

        if current_offer_util >= self.best_oppo_bid_util:
            self.best_oppo_bid_util = current_offer_util
            self.best_oppo_bid = state.current_offer

            if current_offer_util >= self.current_bid_util:
                # if self.debug:
                #     print("{} ConcedingAgent respond: accept".format(self))
                return ResponseType.ACCEPT_OFFER

        # if self.debug:
        #     print("{} ConcedingAgent respond: reject".format(self))

        return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""
        if self.current_neg_index > 0:
            last_agreement = get_agreement_at_index(self, self.current_neg_index - 1)
            last_ufun = self.ufun.side_ufuns()[self.current_neg_index - 1]
            last_neg_util = last_ufun.eval(last_agreement)
            self.current_util = max(last_neg_util, self.current_util)
            # if self.debug:
            #     print("LAST Agreement: {}, current_util {}".format(last_agreement, self.current_util))

        # reset
        self.best_oppo_bid_util = 0
        self.best_oppo_bid = None
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float("inf")
        self.next_bid_util = float("inf")
        self.util_difference_median = 0

        # filter considered bids
        self.current_ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[self.current_neg_index]
        )
        # NOTE: this is myopic, make sure that future neg wont decrease self.current_util
        # TODO: this eligible bids needs to be adapted based on utility function
        # TODO: We might need to treat reserve value differently for center and edge
        temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0
        eligible_bids = [
            x
            for x in self.current_ufun.outcome_space.enumerate_or_sample()
            if self.current_ufun.eval(x) > max(self.current_util, temp_reserve)
        ]
        eligible_bids.sort(
            key=self.current_ufun.eval, reverse=True
        )  # TODO: myopic, needs to be sort by expected util from game tree
        self.potential_bids = eligible_bids
        self.potential_bids_util = [
            self.current_ufun.eval(x) for x in self.potential_bids
        ]

        if self.current_neg_index == 0:
            self.max_util = self.potential_bids_util[0]

        if len(self.potential_bids) > 4:  # NOTE: I chose some arbitrary threshold
            diffs = [
                self.potential_bids_util[i] - self.potential_bids_util[i + 1]
                for i in range(len(self.potential_bids_util) - 2)
            ]
            self.util_difference_median = statistics.median(diffs)
        else:
            self.util_difference_median = 0.005
        # pdb.set_trace()


class ConcedingLookAheadAgent(ANL2025Negotiator):
    def init(self):
        # Initalize variables
        self.current_util = 0  # current realized util, hold across neg

        self.current_neg_index = -1
        self.current_ufun = None
        self.target_bid = None
        self.max_util = float("inf")

        self.best_oppo_bid = None
        self.best_oppo_bid_util = 0
        self.potential_bids = []
        self.potential_bids_util = []
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float("inf")
        self.next_bid_util = float("inf")
        self.util_difference_median = 0

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.debug = False

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # Perform Lookahead if needed
        if not is_edge_agent(self) and self.lookahead_required:
            if state.step == 0:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

        if did_negotiation_end(self):
            self._update_strategy()

        nmi = self.negotiators[negotiator_id][0].nmi

        # utility cannot be improved by any further neg
        if not self.potential_bids:
            return None

        # Concede as a function of
        # (1) index of subnegotiation: self.finished_negotiators vs. len(self.negotiators) vs. self.unfinished_negotiators
        # (2) progress within a subneg: state.step vs. nmi.n_steps
        # (3) whether opponent is conceding
        # (4) concede_degree: how far is the util of current bid to max(self.current_util, self.best_oppo_bid_util)

        perc_subnegs = float(
            (len(self.finished_negotiators) + 1) / len(self.negotiators)
        )

        # if self.debug:
        #     # print(perc_subnegs)
        #     # print(state.relative_time)

        # TODO: Concede differently for center and edge
        first_time_bid = self.current_bid_util == float("inf")

        if not is_edge_agent(self) and self.lookahead_required:
            child_from_current_bid = self.get_current_gamestate().get_child_from_action(
                self.potential_bids[self.conceding_index]
            )
            self.current_bid_util = self.value_dictionary[child_from_current_bid]
        else:
            self.current_bid_util = self.current_ufun.eval(
                self.potential_bids[self.conceding_index]
            )

        if (
            not first_time_bid
            and self.current_bid_util > self.best_oppo_bid_util
            and self.conceding_index < len(self.potential_bids) - 1
        ):
            if not is_edge_agent(self) and self.lookahead_required:
                child_from_next_bid = (
                    self.get_current_gamestate().get_child_from_action(
                        self.potential_bids[self.conceding_index + 1]
                    )
                )
                self.next_bid_util = self.value_dictionary[child_from_next_bid]
            else:
                self.next_bid_util = self.current_ufun.eval(
                    self.potential_bids[self.conceding_index + 1]
                )

            if self.next_bid_util > self.best_oppo_bid_util:
                if self.current_bid_util - self.next_bid_util == 0:
                    util_diff_multiplier = 4
                else:
                    util_diff_multiplier = min(
                        4,
                        float(
                            self.util_difference_median
                            / (self.current_bid_util - self.next_bid_util)
                        ),
                    )

                not_too_aggressive = (
                    float((self.conceding_index + 1) / len(self.potential_bids))
                    < state.relative_time
                )
                relative_standard = self.conceding_index - len(set(self.oppo_bids)) < 3
                # TODO: check on util normalization
                concede_degree = self.next_bid_util - max(
                    self.current_util, self.best_oppo_bid_util
                )

                # NOTE: my very heuristic strategy
                if (
                    relative_standard
                    and not_too_aggressive
                    and state.relative_time * concede_degree * util_diff_multiplier
                    > 0.5 * (1 - perc_subnegs)
                ):  # * self.current_util:
                    self.conceding_index += 1

                    # if not is_edge_agent(self) and self.debug:
                    #     print(is_edge_agent(self), perc_subnegs)
                    #     print(state.relative_time)
                    #     print("concede degree {}, self {}, oppo {}".format(concede_degree, self.conceding_index, len(set(self.oppo_bids))))
                    #     print("{} Conceding to {}: util {} diff {} index {}".format(self, self.potential_bids[self.conceding_index], self.next_bid_util, self.current_bid_util - self.next_bid_util, self.conceding_index))

        # Last chance to propose (TODO: -1 for center; -2 for edge; or according to randomization)
        if (
            is_edge_agent(self) and state.step == nmi.n_steps - 2
        ) or state.step == nmi.n_steps - 1:
            temp_reserve = (
                self.current_ufun.reserved_value if is_edge_agent(self) else 0
            )
            if self.best_oppo_bid_util > max(self.current_util, temp_reserve):
                # if self.debug:
                #     print("{} ConcedingAgent LAST propose {}: util {}".format(self, self.best_oppo_bid, self.current_ufun.eval(self.best_oppo_bid)))
                return self.best_oppo_bid
            else:
                # return None
                return self.potential_bids[0]

        # if self.debug:
        #     print("{} ConcedingAgent propose {}".format(self, self.potential_bids[self.conceding_index]))

        if not is_edge_agent(self) and self.lookahead_required:
            child_from_current_bid = self.get_current_gamestate().get_child_from_action(
                self.potential_bids[self.conceding_index]
            )
            self.current_bid_util = self.value_dictionary[child_from_current_bid]
        else:
            self.current_bid_util = self.current_ufun.eval(
                self.potential_bids[self.conceding_index]
            )

        return self.potential_bids[self.conceding_index]

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        nmi = self.negotiators[negotiator_id][0].nmi

        # Perform Lookahead if needed
        if not is_edge_agent(self) and self.lookahead_required:
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)

            if not has_made_offer:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

        if did_negotiation_end(self):
            self._update_strategy()

        # utility cannot be improved by any further neg
        if not self.potential_bids:
            return ResponseType.END_NEGOTIATION

        if not is_edge_agent(self) and self.lookahead_required:
            child_current_offer = self.get_current_gamestate().get_child_from_action(
                state.current_offer
            )
            current_offer_util = self.value_dictionary[child_current_offer]
        else:
            current_offer_util = self.current_ufun.eval(state.current_offer)

        # Last chance to respond (NOTE: edge only, center has one more chance to propose)
        # TODO: adapt based on whether last to propose
        if is_edge_agent(self) and state.step == nmi.n_steps - 1:
            if current_offer_util > max(
                self.current_util, self.current_ufun.reserved_value
            ):
                # if self.debug:
                #     print("{} ConcedingAgent respond last step: accept".format(self))
                return ResponseType.ACCEPT_OFFER
            else:
                # if self.debug:
                #     print("{} ConcedingAgent respond last step: reject".format(self))
                return ResponseType.REJECT_OFFER

        self.oppo_bids.append(state.current_offer)

        if current_offer_util >= self.best_oppo_bid_util:
            self.best_oppo_bid_util = current_offer_util
            self.best_oppo_bid = state.current_offer

            if current_offer_util >= self.current_bid_util:
                # if self.debug:
                #     print("{} ConcedingAgent respond: accept".format(self))
                return ResponseType.ACCEPT_OFFER

        # if self.debug:
        #     print("{} ConcedingAgent respond: reject".format(self))

        return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""
        if self.current_neg_index > 0:
            if not is_edge_agent(self) and self.lookahead_required:
                # Get current expected utility of last agreement
                current_gamestate = self.get_current_gamestate()
                self.current_util = max(
                    self.value_dictionary[current_gamestate], self.current_util
                )
            else:
                last_agreement = get_agreement_at_index(
                    self, self.current_neg_index - 1
                )
                last_ufun = self.ufun.side_ufuns()[self.current_neg_index - 1]
                last_neg_util = last_ufun.eval(last_agreement)
                self.current_util = max(last_neg_util, self.current_util)
                # if self.debug:
                #     print("LAST Agreement: {}, current_util {}".format(last_agreement, self.current_util))

        # reset
        self.best_oppo_bid_util = 0
        self.best_oppo_bid = None
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float("inf")
        self.next_bid_util = float("inf")
        self.util_difference_median = 0

        # filter considered bids
        self.current_ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[self.current_neg_index]
        )
        # NOTE: this is myopic, make sure that future neg wont decrease self.current_util
        # TODO: this eligible bids needs to be adapted based on utility function
        # TODO: We might need to treat reserve value differently for center and edge
        temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0

        # center agent uses lookahead to calculate expected utilities
        if not is_edge_agent(self) and self.lookahead_required:
            # Get the current gamestate
            current_gamestate = self.get_current_gamestate()

            children = current_gamestate.get_children()

            eligible_children = [
                child
                for child in children
                if self.value_dictionary[child] > max(self.current_util, temp_reserve)
            ]

            # Sort the eligible bids based on their expected utility
            eligible_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)
            self.potential_bids = [child.history[-1] for child in eligible_children]
            self.potential_bids_util = [
                self.value_dictionary[child] for child in eligible_children
            ]

        else:
            eligible_bids = [
                x
                for x in self.current_ufun.outcome_space.enumerate_or_sample()
                if self.current_ufun.eval(x) > max(self.current_util, temp_reserve)
            ]
            eligible_bids.sort(
                key=self.current_ufun.eval, reverse=True
            )  # TODO: myopic, needs to be sort by expected util from game tree
            self.potential_bids = eligible_bids
            self.potential_bids_util = [
                self.current_ufun.eval(x) for x in self.potential_bids
            ]

        if self.current_neg_index == 0:
            self.max_util = self.potential_bids_util[0]

        if len(self.potential_bids) > 4:  # NOTE: I chose some arbitrary threshold
            diffs = [
                self.potential_bids_util[i] - self.potential_bids_util[i + 1]
                for i in range(len(self.potential_bids_util) - 2)
            ]
            self.util_difference_median = statistics.median(diffs)
        else:
            self.util_difference_median = 0.005
        # pdb.set_trace()

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class LookAheadCurrentUtilityNaiveSoftMaxAdj(ANL2025Negotiator):
    """
    Agent that uses a lookahead strategy to estimated expected utilities for an action.
    It's done with the current utility.
    Concedes linearly with the deadline.
    """

    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if not is_edge_agent(self) and self.lookahead_required:
            bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            if state.step == 0:
                # If this is the first step, we need to perform the look ahead
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

            # if the current step
            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                bid = filtered_children[target_index].history[-1]

        else:
            bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed
            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value
            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = current_step / (total_steps)

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_index = 0
                bid = outcome_space[target_index]

        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if not is_edge_agent(self) and self.lookahead_required:
            target_bid = None
            current_game_state = self.get_current_gamestate()
            children = current_game_state.get_children()
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            # If you are first to respond, you need to perform look ahead
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
            if not has_made_offer:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

            subnegotation_trace = nmi.trace

            children_of_opponent_bids = [
                current_game_state.get_child_from_action(x[1])
                for x in subnegotation_trace
                if x[0] != negotiator_id
            ]
            reservation_value = self.ufun.reserved_value

            util_of_best_opponent_bid = 0
            for child_opponent_bid in children_of_opponent_bids:
                if (
                    self.value_dictionary[child_opponent_bid]
                    > util_of_best_opponent_bid
                ):
                    util_of_best_opponent_bid = self.value_dictionary[
                        child_opponent_bid
                    ]

            opponent_current_offer = current_game_state.get_child_from_action(
                state.current_offer
            )

            # Because this is last step, maybe just use ufun eval
            if current_step == total_steps - 1:  # last step
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            filtered_children = [
                child
                for child in children
                if self.value_dictionary[child] > util_of_best_opponent_bid
                and self.value_dictionary[child] > reservation_value
            ]

            filtered_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede
            child_of_target_bid = None
            if filtered_children:
                target_index = math.ceil(step_percentage * len(filtered_children))
                target_index = min(
                    target_index, len(filtered_children) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                child_of_target_bid = filtered_children[target_index]

                if (
                    self.value_dictionary[opponent_current_offer]
                    >= self.value_dictionary[child_of_target_bid]
                ):
                    return ResponseType.ACCEPT_OFFER
            else:
                # If there are no filtered children, just accept the offer
                if self.value_dictionary[opponent_current_offer] > reservation_value:
                    return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER
        else:
            target_bid = None
            outcome_space = []
            current_negotation_index = get_current_negotiation_index(self)
            nmi = self.negotiators[negotiator_id][0].nmi
            total_steps = nmi.n_steps
            current_step = state.step  # think it is originally 0 indexed

            subnegotation_trace = nmi.trace

            opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]
            ufun = (
                self.ufun
                if is_edge_agent(self)
                else self.ufun.side_ufuns()[current_negotation_index]
            )
            reservation_value = ufun.reserved_value

            if current_step == total_steps - 1:  # last step
                if ufun.eval(state.current_offer) > reservation_value:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

            outcome_space = ufun.outcome_space.enumerate_or_sample()

            util_of_best_opponent_bid = 0
            for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

            outcome_space = [
                x
                for x in outcome_space
                if ufun.eval(x) >= util_of_best_opponent_bid
                and ufun.eval(x) > reservation_value
            ]
            outcome_space.sort(key=ufun.eval, reverse=True)

            # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
            step_percentage = (current_step + 1) / (
                total_steps
            )  # don't have to accept the best offer, willing to concede

            if outcome_space:
                target_index = math.ceil(step_percentage * len(outcome_space))
                target_index = min(
                    target_index, len(outcome_space) - 1
                )  # make sure it is not out of bounds in case step_percentage is 1
                target_bid = outcome_space[target_index]

            target_util = ufun.eval(target_bid) if target_bid is not None else math.inf

            if ufun.eval(state.current_offer) >= target_util:
                return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=100000.0,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1000000000.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        adj_expected_values = np.copy(children_expected_values)
        for child in range(len(children_expected_values)):
            if children_expected_values[child] < current_parent_utility:
                adj_expected_values[child] = (
                    current_parent_utility  # will just select None if parent selected, we don't want this to negatively weight the parent node
                )

        # Apply softmax to positive children
        scaled_values = adj_expected_values / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = exp_values / total_exp

        return probabilities

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_uniform_distribution_on_outcomes(
        self, num_outcomes: int
    ) -> NDArray[np.float64]:
        """
        Returns a uniform probability distribution over the number of children.
        """
        return np.full(num_outcomes, 1.0 / num_outcomes, dtype=np.float64)

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class NewBoulwareLookAhead(ANL2025Negotiator):
    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

        self._curve = PolyAspiration(1.0, "boulware")
        self._inverter: dict[str, bool] = dict()
        self._best: list[Outcome] = None  # type: ignore
        self._mx: float = 1.0
        self._mn: float = 0.0
        self._deltas = (0.1, 0.2, 0.4, 0.8, 1)
        self._best_margin = 1e-8
        self.reject_exactly_as_reserved = False

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        nmi = self.negotiators[negotiator_id][0].nmi

        if not is_edge_agent(self) and self.lookahead_required and state.step == 0:
            # If this is the first step, we need to perform the look ahead
            self.value_dictionary = {}
            self.lookahead(
                self.get_current_gamestate(),
                0.9,
                depth_limit=get_current_negotiation_index(self) + self.depth,
            )

        if did_negotiation_end(self):
            self._update_strategy()

        self.ensure_inverter(
            negotiator_id
        )  # need to change this to use the value dictionary
        level = self.calc_level(nmi, state, normalized=False)

        outcome = None
        if not self.potential_bids:
            return None
        for d in self._deltas:
            d_scaled = (
                d * (self._mx - self._mn) if (self._mx - self._mn) > 1e-9 else d * 0.01
            )

            mx = min(self._mx, level + d_scaled)

            borderline_bid_index = self.find_borderline_bid_index(level)

            if borderline_bid_index is None:
                continue

            if self.potential_bids_util[borderline_bid_index] < mx:
                upper_bound = borderline_bid_index - 1
                while upper_bound >= 0 and self.potential_bids_util[upper_bound] < mx:
                    upper_bound -= 1
                outcome_index = randint(upper_bound + 1, borderline_bid_index)
                outcome = self.potential_bids[outcome_index]
                break

        if outcome is None:
            if not self._best:
                return None
            return choice(self._best)

        return outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        nmi = self.negotiators[negotiator_id][0].nmi

        if not is_edge_agent(self) and self.lookahead_required:
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)

            if not has_made_offer:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

        if did_negotiation_end(self):
            self._update_strategy()

        self.ensure_inverter(
            negotiator_id
        )  # need to change this to use the value dictionary
        level = self.calc_level(nmi, state, normalized=False)

        if not self.potential_bids:
            return ResponseType.END_NEGOTIATION

        if not is_edge_agent(self) and self.lookahead_required:
            current_game_state = self.get_current_gamestate()
            current_offer_util = self.value_dictionary[
                current_game_state.get_child_from_action(state.current_offer)
            ]  # type: ignore
        else:
            current_offer_util = (
                self.current_ufun.eval(state.current_offer)
                if is_edge_agent(self)
                else self.current_ufun.eval(state.current_offer)
            )  # type: ignore

        if (self.reject_exactly_as_reserved and level >= current_offer_util) or (
            not self.reject_exactly_as_reserved and level > current_offer_util
        ):
            # If the offer is better than the level, accept it
            return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""

        # filter considered bids
        self.current_ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[self.current_neg_index]
        )
        # NOTE: this is myopic, make sure that future neg wont decrease self.current_util
        # TODO: this eligible bids needs to be adapted based on utility function
        # TODO: We might need to treat reserve value differently for center and edge
        temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0  # type: ignore

        temp_reserve = self.get_current_utility()
        # center agent uses lookahead to calculate expected utilities
        if not is_edge_agent(self) and self.lookahead_required:
            # Get the current gamestate
            current_gamestate = self.get_current_gamestate()

            children = current_gamestate.get_children()

            non_Child = current_gamestate.get_child_from_action(None)  # type: ignore

            # eligible_children = [child for child in children if self.value_dictionary[child] > max(self.value_dictionary[non_Child], temp_reserve)]
            eligible_children = [
                child
                for child in children
                if self.value_dictionary[child]
                > max(self.value_dictionary[non_Child], temp_reserve)
            ]

            # Sort the eligible bids based on their expected utility
            eligible_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)
            self.potential_bids = [child.history[-1] for child in eligible_children]
            self.potential_bids_util = [
                self.value_dictionary[child] for child in eligible_children
            ]

        else:
            eligible_bids = [
                x
                for x in self.current_ufun.outcome_space.enumerate_or_sample()
                if self.current_ufun.eval(x)
                > max(self.current_ufun.eval(None), temp_reserve)
            ]  # type: ignore
            # if self.current_ufun.eval(None) != temp_reserve:
            #     index = get_current_negotiation_index(self)
            #     print("current negotiation index", index)
            #     nmi = get_nmi_from_index(self, index)
            #     print("nmi trace", nmi.trace)
            #     print("side utility reserve value?", self.current_ufun.reserved_value)
            #     print("none", self.current_ufun.eval(None), temp_reserve, self.get_current_agreements())
            # assert self.current_ufun.eval(None) == temp_reserve
            eligible_bids = [
                x
                for x in self.current_ufun.outcome_space.enumerate_or_sample()
                if self.current_ufun.eval(x) > temp_reserve
            ]  # type: ignore
            eligible_bids.sort(key=self.current_ufun.eval, reverse=True)  # type: ignore # TODO: myopic, needs to be sort by expected util from game tree
            self.potential_bids = eligible_bids
            self.potential_bids_util = [
                self.current_ufun.eval(x) for x in self.potential_bids
            ]  # type: ignore
        # if not is_edge_agent(self):
        #     print(self.potential_bids, self.potential_bids_util)

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        for side in self.negotiators.keys():
            if side == negotiator_id:
                continue
            if side in self._inverter:
                del self._inverter[side]

    def ensure_inverter(self, negotiator_id):
        if self._inverter.get(negotiator_id) is None:
            self._mx = (
                float(self.potential_bids_util[0]) if self.potential_bids_util else 1.0
            )
            self._mn = (
                float(self.potential_bids_util[-1]) if self.potential_bids_util else 0.0
            )
            self._best = []
            threshold = max(
                self._mn, self._mx - self._best_margin
            )  # already filtered out outcomes worse than None in self._update_strategy
            for i in range(len(self.potential_bids_util)):
                if self.potential_bids_util[i] >= threshold:
                    self._best.append(self.potential_bids[i])
                else:
                    break
            self._inverter[negotiator_id] = True

    def find_borderline_bid_index(self, level: float) -> int:
        start = 0
        end = len(self.potential_bids_util) - 1
        mid = (
            3 * (start + end) // 4
        )  # start at 3/4 of the list, because more likely will be greater than half
        while start <= end:
            mid_bid_util = self.potential_bids_util[mid]
            if mid_bid_util == level or (
                mid_bid_util > level and mid_bid_util - self._best_margin < level
            ):  # consideration: doesn't really account for ties well
                return mid
            elif mid_bid_util < level:
                start = mid + 1
            else:
                end = mid - 1
            mid = (start + end) // 2
        return None  # type: ignore

    def calc_level(self, nmi: SAONMI, state: SAOState, normalized: bool):
        if state.step == 0:
            level = 1.0
        elif (
            # not self.reject_exactly_as_reserved
            # and
            nmi.n_steps is not None and state.step >= nmi.n_steps - 1
        ):
            level = 0.0
        else:
            level = self._curve.utility_at(state.relative_time)
        if not normalized:
            level = level * (self._mx - self._mn) + self._mn
        return level

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
            non_child = self.get_current_gamestate().get_child_from_action(None)
            utility = self.value_dictionary[non_child]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class BoulwareLookAhead(ANL2025Negotiator):
    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

        self._curve = PolyAspiration(1.0, "boulware")
        self._inverter: dict[str, bool] = dict()
        self._best: list[Outcome] = None  # type: ignore
        self._mx: float = 1.0
        self._mn: float = 0.0
        self._deltas = (0.1, 0.2, 0.4, 0.8, 1)
        self._best_margin = 1e-8
        self.reject_exactly_as_reserved = False

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        nmi = self.negotiators[negotiator_id][0].nmi

        if not is_edge_agent(self) and self.lookahead_required and state.step == 0:
            # If this is the first step, we need to perform the look ahead
            self.value_dictionary = {}
            self.lookahead(
                self.get_current_gamestate(),
                0.9,
                depth_limit=get_current_negotiation_index(self) + self.depth,
            )

        if did_negotiation_end(self):
            self._update_strategy()

        self.ensure_inverter(
            negotiator_id
        )  # need to change this to use the value dictionary
        level = self.calc_level(nmi, state, normalized=False)

        outcome = None
        if not self.potential_bids:
            return None
        for d in self._deltas:
            d_scaled = (
                d * (self._mx - self._mn) if (self._mx - self._mn) > 1e-9 else d * 0.01
            )

            mx = min(self._mx, level + d_scaled)

            borderline_bid_index = self.find_borderline_bid_index(level)

            if borderline_bid_index is None:
                continue

            if self.potential_bids_util[borderline_bid_index] < mx:
                upper_bound = borderline_bid_index - 1
                while upper_bound >= 0 and self.potential_bids_util[upper_bound] < mx:
                    upper_bound -= 1
                outcome_index = randint(upper_bound + 1, borderline_bid_index)
                outcome = self.potential_bids[outcome_index]
                break

        if outcome is None:
            if not self._best:
                return None
            return choice(self._best)

        return outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        nmi = self.negotiators[negotiator_id][0].nmi

        if not is_edge_agent(self) and self.lookahead_required:
            has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)

            if not has_made_offer:
                self.value_dictionary = {}
                self.lookahead(
                    self.get_current_gamestate(),
                    0.9,
                    depth_limit=get_current_negotiation_index(self) + self.depth,
                )

        if did_negotiation_end(self):
            self._update_strategy()

        self.ensure_inverter(
            negotiator_id
        )  # need to change this to use the value dictionary
        level = self.calc_level(nmi, state, normalized=False)

        if not self.potential_bids:
            return ResponseType.END_NEGOTIATION

        if not is_edge_agent(self) and self.lookahead_required:
            current_game_state = self.get_current_gamestate()
            current_offer_util = self.value_dictionary[
                current_game_state.get_child_from_action(state.current_offer)
            ]  # type: ignore
        else:
            current_offer_util = (
                self.current_ufun.eval(state.current_offer)
                if is_edge_agent(self)
                else self.current_ufun.eval(state.current_offer)
            )  # type: ignore

        if (self.reject_exactly_as_reserved and level >= current_offer_util) or (
            not self.reject_exactly_as_reserved and level > current_offer_util
        ):
            # If the offer is better than the level, accept it
            return ResponseType.REJECT_OFFER

        return ResponseType.ACCEPT_OFFER

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""

        # filter considered bids
        self.current_ufun = (
            self.ufun
            if is_edge_agent(self)
            else self.ufun.side_ufuns()[self.current_neg_index]
        )
        # NOTE: this is myopic, make sure that future neg wont decrease self.current_util
        # TODO: this eligible bids needs to be adapted based on utility function
        # TODO: We might need to treat reserve value differently for center and edge
        temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0  # type: ignore

        # center agent uses lookahead to calculate expected utilities
        if not is_edge_agent(self) and self.lookahead_required:
            # Get the current gamestate
            current_gamestate = self.get_current_gamestate()

            children = current_gamestate.get_children()

            non_Child = current_gamestate.get_child_from_action(None)  # type: ignore

            eligible_children = [
                child
                for child in children
                if self.value_dictionary[child]
                > max(self.value_dictionary[non_Child], temp_reserve)
            ]

            # Sort the eligible bids based on their expected utility
            eligible_children.sort(key=lambda x: self.value_dictionary[x], reverse=True)
            self.potential_bids = [child.history[-1] for child in eligible_children]
            self.potential_bids_util = [
                self.value_dictionary[child] for child in eligible_children
            ]

        else:
            eligible_bids = [
                x
                for x in self.current_ufun.outcome_space.enumerate_or_sample()
                if self.current_ufun.eval(x)
                > max(self.current_ufun.eval(None), temp_reserve)
            ]  # type: ignore
            eligible_bids.sort(key=self.current_ufun.eval, reverse=True)  # type: ignore # TODO: myopic, needs to be sort by expected util from game tree
            self.potential_bids = eligible_bids
            self.potential_bids_util = [
                self.current_ufun.eval(x) for x in self.potential_bids
            ]  # type: ignore

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        for side in self.negotiators.keys():
            if side == negotiator_id:
                continue
            if side in self._inverter:
                del self._inverter[side]

    def ensure_inverter(self, negotiator_id):
        if self._inverter.get(negotiator_id) is None:
            self._mx = (
                float(self.potential_bids_util[0]) if self.potential_bids_util else 1.0
            )
            self._mn = (
                float(self.potential_bids_util[-1]) if self.potential_bids_util else 0.0
            )
            self._best = []
            threshold = max(
                self._mn, self._mx - self._best_margin
            )  # already filtered out outcomes worse than None in self._update_strategy
            for i in range(len(self.potential_bids_util)):
                if self.potential_bids_util[i] >= threshold:
                    self._best.append(self.potential_bids[i])
                else:
                    break
            self._inverter[negotiator_id] = True

    def find_borderline_bid_index(self, level: float) -> int:
        start = 0
        end = len(self.potential_bids_util) - 1
        mid = (
            3 * (start + end) // 4
        )  # start at 3/4 of the list, because more likely will be greater than half
        while start <= end:
            mid_bid_util = self.potential_bids_util[mid]
            if mid_bid_util == level or (
                mid_bid_util > level and mid_bid_util - self._best_margin < level
            ):  # consideration: doesn't really account for ties well
                return mid
            elif mid_bid_util < level:
                start = mid + 1
            else:
                end = mid - 1
            mid = (start + end) // 2
        return None  # type: ignore

    def calc_level(self, nmi: SAONMI, state: SAOState, normalized: bool):
        if state.step == 0:
            level = 1.0
        elif (
            # not self.reject_exactly_as_reserved
            # and
            nmi.n_steps is not None and state.step >= nmi.n_steps - 1
        ):
            level = 0.0
        else:
            level = self._curve.utility_at(state.relative_time)
        if not normalized:
            level = level * (self._mx - self._mn) + self._mn
        return level

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class UtilityFitAgent(ANL2025Negotiator):
    def init(self):
        self.debug = True

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.outcome_space_at_subnegotiation = []
        if not is_edge_agent(self):
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

        # Create the game environment
        self.environment = GameEnvironment(
            center_ufun=self.ufun,
            n_edges=len(self.negotiators),
            outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
        )

        # Variables that must be initalized at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = 0  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.utilities_and_time_from_villain = (
            None  # utilities and relative time of villain's offers
        )
        self.e_value = None  # e value of the utility model. How do the villains concede based on time
        self.unique_proposals_from_villain = None  # number of unique proposals from the villain | used to determine if the curve fit has enough data to be used
        self.maximum_utility = (
            None  # predicted maximum utility the hero can get from the villain's offers
        )
        self.weak_curve_fit_index = None  # When the curve fit is weak, we will use this index to determine which offer to propose next.
        # This is used to ensure that we do not propose the same offer over and over again when the curve fit is weak.

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True
        utility = self.ufun(current_outcome)
        if current_agreements:
            if any_agreement:
                return utility
            else:
                return self.ufun.reserved_value
        else:
            return self.ufun.reserved_value

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # We are first to act, and also last to propose
            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            # self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            if self.current_negotiation_index > 0:
                last_agreement = get_agreement_at_index(
                    self, self.current_negotiation_index - 1
                )
                last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index - 1]
                last_neg_util = last_ufun.eval(last_agreement)
                self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                utility_of_outcome = self.current_ufun(outcome)
                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        # No better outcomes
        if len(self.filtered_outcome_space) == 0:
            return None

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            # TODO: Change this? Maybe just return some rotation of the top offers?
            return self.filtered_outcome_space[0][0]

        # TODO: This ideally should not happen, fix
        # Sometimes acceptance threshold is very low if opponent's offers are low and curve fit is weak
        # In this case, just return the best outcome for the hero
        if next_bid not in self.filtered_outcome_space_dict:
            return self.filtered_outcome_space[0][0]

        # If the offer from the villain is greater than the  predicted concession of the villain
        # Then we just return their offer. HOWEVER, this is just for now, I may only want to offer
        # their offer if it's the last step to propose

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if (
                self.normalize_utility(self.best_offer_from_villain_and_utiility[1])
                >= acceptance_threshold - 0.01
            ):
                return self.best_offer_from_villain_and_utiility[0]

        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
        if not has_made_offer:  # We are first to respond, so we do not propose last
            self.last_to_propose = False
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            # self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # self.current_utility = 0
            if self.current_negotiation_index > 0:
                last_agreement = get_agreement_at_index(
                    self, self.current_negotiation_index - 1
                )
                last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index - 1]
                last_neg_util = last_ufun.eval(last_agreement)
                self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        # End negotiation if there's no outcome better for you
        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        # Update offers from villain and the best one
        current_villain_offer = state.current_offer
        current_villain_offer_utility = self.current_ufun(current_villain_offer)
        if current_villain_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_villain_offer,
                current_villain_offer_utility,
            )

        # Update the utilities and time from villain's offers
        if self.last_to_propose:
            # If you propose last, then their step is -1 relative to yours when they offerred
            time_they_proposed = (state.step - 1 + 1) / (self.total_steps + 1)
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, time_they_proposed)
            )
        else:
            # if you respond last, then their step is the same as yours
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, state.relative_time)
            )

        self.unique_proposals_from_villain.add(current_villain_offer)

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        # TODO: Maybe only accept if it's above a certain threshold of the reservation value
        # If we are last to respond, just accept offer if better than our reservation value
        if not self.last_to_propose:
            if (
                state.step == self.total_steps - 1
            ):  # the last step for responding if we respond last is n-1
                if (
                    self.current_ufun(current_villain_offer)
                    > self.reservation_value + 0.1
                ):
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        # Reject if offer not even above
        # TODO: What if all of their offers are not in the filtered outcome space?
        if current_villain_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        # Check if the current offer is acceptable
        # TODO: We might always want to reject just to see if we can get a higher concession...
        # For now, just always reject offer
        if (
            current_villain_offer_utility
            >= self.filtered_outcome_space_dict[next_bid] - 0.01
            if next_bid
            else False
        ):
            if (self.last_to_propose and state.step >= self.total_steps - 1) or (
                not self.last_to_propose and state.step >= self.total_steps - 2
            ):
                return ResponseType.REJECT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def utility_model(self, time, e, max_utility):
        """
        Utility model based on time, where we predict the e value of the villain and maximum utility the hero can reach.
        """
        if self.is_edge_agent:
            # FOR SOME REASON, setting this equal to 1 gives the best results. Will need to investigate this later
            max_utility = 1
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.reservation_value),
                ),
                max_utility,
            )
        else:
            # The utility either follows a time based concession model, or the maximum utility
            # This is why np.minimum is used, between the time based and the maximum utility we can achieve
            # the time based concession model is denoted from np.power(time, e)
            # This is vertically shifted by the utility of the first offer from the villain,
            # or the current utility of the hero, whichever is greater
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.current_utility),
                ),
                max_utility,
            )

    def compute_a(self, e) -> float:
        """
        This a value is used to interpolate the maximum utility of the hero based on the villain's concession.
        If the villain concedes a lot, then the maximum utility of the hero is closer to the villain's concession
        Otherwise, it is closer to the hero's maximum utility.

        Closer to 0 means the maximum utility will be closer to the villain's best offer.
        """
        if e <= 1:
            return 0
        else:
            return np.log(e) / np.log(17)

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """
        Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept.
        We will calculate this acceptance threshold by modelling the opponent's offers and seeing how much utility their offer gives.
        If the utility of their offers increase, this means they are conceding. All utilities are normalized to be between 0 and 1.

        We will assume they model their concession as a time based strategy

        The idea is to predict the utility they will concede to at t = 1 or the final timestep. This will be our acceptance threshold
        """

        if state.step == 0:
            return 1
        # plt.clf()

        # Normalize the utilities from the villain's offers
        normalized_villain_utilities = np.array(
            [
                self.normalize_utility(utility)
                for utility, time in self.utilities_and_time_from_villain
            ]
        )
        villain_times = np.array(
            [time for _, time in self.utilities_and_time_from_villain]
        )

        # print(self.utilities_and_time_from_villain)
        # plt.plot(villain_times, normalized_villain_utilities, 'ro', label='Observed')

        normalized_hero_maximum_utility = self.normalize_utility(
            self.filtered_outcome_space[0][1]
        )
        normalized_best_utility_from_villain = self.normalize_utility(
            self.best_offer_from_villain_and_utiility[1]
        )

        # Returns a value between 0 and 1. 0 means that the upperbound for the max utility will be closer to
        #   the villain's best offer, 1 means that the upperbound will be the hero's maximum utility
        a = self.compute_a(self.e_value)

        # Setting the bounds of the maximum utility for the curve fit
        # Setting the bound to 0.5 is a huge assumption, but it helps if the curve fit predicts the maximum utility to be much lower
        # I did this because when against a villain that never concedes, the curve fit will predict a very low maximum utility which is not ideal
        low_bound_max_utility = max(
            min(
                normalized_best_utility_from_villain,
                normalized_hero_maximum_utility - 0.001,
            ),
            0.5,
        )
        # The value of a is used here to bound the maximum utility
        hi_bound_max_utility = max(
            normalized_hero_maximum_utility * a
            + normalized_best_utility_from_villain * (1 - a),
            0.55,
        )
        if hi_bound_max_utility == low_bound_max_utility:
            hi_bound_max_utility += 0.01

        predicted_parameters, _ = curve_fit(
            self.utility_model,
            villain_times,
            normalized_villain_utilities,
            # p0=[self.e_value, self.maximum_utility],
            bounds=[(0.01, low_bound_max_utility), (17, hi_bound_max_utility)],
            maxfev=10000,
        )
        e_value = predicted_parameters[0]
        max_utility = predicted_parameters[1]
        # print("e value: ", e_value, "rv", reservation_value, "max utility", max_utility)

        # This was originally here to interpolate the e value, but currently not used. Would have also been used to interpolate the maximum utility
        gamma = 0
        self.e_value = gamma * self.e_value + (1 - gamma) * e_value
        self.maximum_utility = max_utility
        # Obtain the last time step they will offer/respond to something
        last_time_step = (self.total_steps - 1 + 1) / (self.total_steps + 1)

        # In the case that they are very boulware, or not conceding, we set last time step to 1, or simply the predicted max utility
        # This is because the last_time_step may be too "low", especially when n_steps is small. Then it will predict a very low utility.
        # Let me know if this is confusing
        if self.e_value > 10:
            last_time_step = 1
        # last_time_step = 1
        predicted_utility = self.utility_model(
            last_time_step, self.e_value, max_utility
        )
        # print(hero_maximum_utility * predicted_utility, self.best_offer_from_villain_and_utiility[1])

        # t_plot = np.linspace(0, 1, 100)
        # u_plot = self.utility_model(t_plot, self.e_value, max_utility)
        # plt.plot(t_plot, u_plot, 'b-', label=f'Fitted curve')
        # print("step", state.step, self.interpolated_reservation_value)
        # print("predicted utility", predicted_utility, "thingie returned", hero_maximum_utility * predicted_utility)

        # When curve fit is weak, the minimum proportion of maximum utility to offer
        # Idea is to set a high acceptance threshold, so we do not accept offers that are too low
        weak_curve_fit_factor = max(0.8, predicted_utility)

        # Number of unique offers from the villain to determine if we can use the curve fit
        # Not really if this is a good number... will need to fine tune this
        minimum_unique_offers = max(10, 0.75 * self.total_steps)

        # If at last couple of timesteps, use the predicted utility regardless of the number of unique offers
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            return normalized_hero_maximum_utility * predicted_utility

        # Otherwise, if the curve fit is weak, we will always set a high acceptance threshold, based on the weak curve fit factor
        # if len(self.unique_proposals_from_villain) < minimum_unique_offers:
        if len(self.utilities_and_time_from_villain) < minimum_unique_offers:
            acceptance_threshold = (
                weak_curve_fit_factor * normalized_hero_maximum_utility
            )
            # print("neg index", self.current_negotiation_index, "unique proposals", self.unique_proposals_from_villain, "acceptance_threshold", acceptance_threshold, "best utility", self.filtered_outcome_space[0][1])
            return acceptance_threshold

        return normalized_hero_maximum_utility * predicted_utility

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent using the acceptance threshold.

        There are 3 "regions" that we can be in based on time.

        In the aggressive region, based on stop aggressive timestep, we will offer bids that are above the acceptance threshold.
        After the aggressive region, we will offer bids that are at the acceptance threshold.
        On the last couple of steps (or last timestep to propose), we will offer bids that are between the acceptance threshold and the best offer from the villain.
        """
        # Discounting for updating of item count
        # This should be < 1, but I was just fine tuning things. Will test on lower values
        eta = 0.8

        # timestep to when to stop offering bid above point
        stop_aggressive_timestep = 0.9

        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")
        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        # In the last step to propose, propose an offer in between the best offer from the villain (slightly higher than this) and the acceptance threshold
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # If best offer is close to acceptance threshold, just return it. Without this line, often result in no agreements reached
            if (
                acceptance_threshold - 0.1
                <= normalized_best_offer_from_villain_and_utility[1]
            ):
                return normalized_best_offer_from_villain_and_utility[0]

            # Find offers in the range of acceptance threshold and best offer from villain
            fscore = 0
            for outcome, utility in self.normalized_filtered_outcome_space:
                # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                if (
                    utility
                    > normalized_best_offer_from_villain_and_utility[1] * 0.5
                    + acceptance_threshold * 0.5
                    and utility <= acceptance_threshold
                ):
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    fscore = 0
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore

                # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                elif utility < normalized_best_offer_from_villain_and_utility[1]:
                    break
            # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
            if len(outcome_space_above_threshold) == 0:
                return normalized_best_offer_from_villain_and_utility[0]

        # elif state.relative_time >= stop_aggressive_timestep:
        #     # Find offers in the range of acceptance threshold
        #     fscore = 0
        #     for outcome, utility in self.normalized_filtered_outcome_space:
        #         # Maybe we can expand this range? Idk how helpful it will be
        #         if utility >= acceptance_threshold - 0.01 and utility <= acceptance_threshold:
        #             outcome_space_above_threshold.append(outcome)
        #             # Calculate the fscore of the outcome
        #             fscore = 0
        #             for index, issue in enumerate(outcome):
        #                 indexed_issue = str(index) + str(issue)
        #                 fscore += self.discounted_item_counter[indexed_issue]

        #             if fscore == outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore[0].append(outcome)

        #             if fscore > outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore = [outcome], fscore
        #         else:
        #             break
        #     # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
        #     if len(outcome_space_above_threshold) == 0:
        #         for outcome, utility in self.normalized_filtered_outcome_space:
        #             if utility < acceptance_threshold:
        #                 outcomes_and_highest_fscore[0].append(outcome)
        #                 break
        else:
            for outcome, utility in self.normalized_filtered_outcome_space:
                fscore = 0
                if utility >= acceptance_threshold:
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore
                else:
                    break

        # If we are last step to propose, or we are greater than aggressive timestep, return random outcome from highest fscore outcomes
        if (
            (self.last_to_propose and state.step >= self.total_steps - 1)
            or (not self.last_to_propose and state.step >= self.total_steps - 2)
            or state.relative_time >= stop_aggressive_timestep
        ):
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            # elif len(outcomes_and_highest_fscore[0]) == 1:
            #     return self.best_offer_from_villain_and_utiility[0]
            else:
                # print(len(outcomes_and_highest_fscore[0]), outcomes_and_highest_fscore[1])
                random_choice = choice(outcomes_and_highest_fscore[0])

                # print("next bid", random_choice, "utility", self.normalized_filtered_outcome_space_dict[random_choice], "acceptance threshold", acceptance_threshold)
                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # We return some bid above the acceptance threshold, but we rotate through the bids to ensure we do not propose the same bid over and over again
                # Make it look like we are doing some kind of conceding
                next_bid = outcome_space_above_threshold[
                    -1 - self.weak_curve_fit_index % len(outcome_space_above_threshold)
                ]
                self.weak_curve_fit_index += 1
                return next_bid

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)


class NewerUtilityFitLookAheadAgent(ANL2025Negotiator):
    def init(self):
        self.debug = True
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Variables that must be initalized at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = 0  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.utilities_and_time_from_villain = (
            None  # utilities and relative time of villain's offers
        )
        self.e_value = None  # e value of the utility model. How do the villains concede based on time
        self.unique_proposals_from_villain = None  # number of unique proposals from the villain | used to determine if the curve fit has enough data to be used
        self.maximum_utility = (
            None  # predicted maximum utility the hero can get from the villain's offers
        )
        self.weak_curve_fit_index = None  # When the curve fit is weak, we will use this index to determine which offer to propose next.
        # This is used to ensure that we do not propose the same offer over and over again when the curve fit is weak.

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        self.hero_bids = None

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

            # Dictionary to store expected values of outcomes for lookahead
            self.value_dictionary = None

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
            non_child = self.get_current_gamestate().get_child_from_action(None)
            utility = self.value_dictionary[non_child]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            # We are first to act, and also last to propose
            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            self.current_utility = self.get_current_utility()

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            # Observe the None outcome (from current utility) and see if it's min or max
            if self.current_utility < self.minimum_outcome_utility:
                self.minimum_outcome_utility = self.current_utility
            if self.current_utility > self.maximum_outcome_utility:
                self.maximum_outcome_utility = self.current_utility

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

            self.hero_bids = []

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        # No better outcomes
        if len(self.filtered_outcome_space) == 0:
            return None

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            # TODO: Change this? Maybe just return some rotation of the top offers?
            self.hero_bids.append(self.filtered_outcome_space[0][0])
            return self.filtered_outcome_space[0][0]

        # TODO: This ideally should not happen, fix
        # Sometimes acceptance threshold is very low if opponent's offers are low and curve fit is weak
        # In this case, just return the best outcome for the hero
        if next_bid not in self.filtered_outcome_space_dict:
            self.hero_bids.append(self.filtered_outcome_space[0][0])
            return self.filtered_outcome_space[0][0]

        # If the offer from the villain is greater than the  predicted concession of the villain
        # Then we just return their offer. HOWEVER, this is just for now, I may only want to offer
        # their offer if it's the last step to propose

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if (
                self.normalize_utility(self.best_offer_from_villain_and_utiility[1])
                >= acceptance_threshold - 0.01
            ):
                if (
                    self.best_offer_from_villain_and_utiility[0]
                    in self.filtered_outcome_space
                ):
                    self.hero_bids.append(self.best_offer_from_villain_and_utiility[0])
                    return self.best_offer_from_villain_and_utiility[0]

        self.hero_bids.append(next_bid)
        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
        if not has_made_offer:  # We are first to respond, so we do not propose last
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = False
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # self.current_utility = 0
            # if self.current_negotiation_index > 0:
            #     last_agreement = get_agreement_at_index(self, self.current_negotiation_index-1)
            #     last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index-1]
            #     last_neg_util = last_ufun.eval(last_agreement)
            #     self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            # Observe the None outcome and see if it's min or max
            if self.current_utility < self.minimum_outcome_utility:
                self.minimum_outcome_utility = self.current_utility
            if self.current_utility > self.maximum_outcome_utility:
                self.maximum_outcome_utility = self.current_utility

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

            self.hero_bids = []

        # End negotiation if there's no outcome better for you
        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        # Update offers from villain and the best one
        current_villain_offer = state.current_offer

        if self.use_lookahead():
            current_villain_offer_utility = self.value_dictionary[
                self.get_current_gamestate().get_child_from_action(
                    current_villain_offer
                )
            ]
        else:
            current_villain_offer_utility = self.current_ufun(current_villain_offer)

        if current_villain_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_villain_offer,
                current_villain_offer_utility,
            )

        # Update the utilities and time from villain's offers
        if self.last_to_propose:
            # If you propose last, then their step is -1 relative to yours when they offerred
            time_they_proposed = (state.step - 1 + 1) / (self.total_steps + 1)
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, time_they_proposed)
            )
        else:
            # if you respond last, then their step is the same as yours
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, state.relative_time)
            )

        self.unique_proposals_from_villain.add(current_villain_offer)

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        if state.step == self.total_steps - 1 and not self.last_to_propose:
            if (
                current_villain_offer in self.hero_bids
            ):  # and self.filtered_outcome_space_dict[current_villain_offer] > self.best_offer_from_villain_and_utiility[1]:
                # If we have already proposed this offer, then we will accept it
                return ResponseType.ACCEPT_OFFER

        # TODO: Maybe only accept if it's above a certain threshold of the reservation value
        # If we are last to respond, just accept offer if better than our reservation value
        if not self.last_to_propose:
            if (
                state.step == self.total_steps - 1
            ):  # the last step for responding if we respond last is n-1
                if current_villain_offer_utility > self.reservation_value + 0.1:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        # Reject if offer not even above
        # TODO: What if all of their offers are not in the filtered outcome space?
        if current_villain_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        # Check if the current offer is acceptable
        # TODO: We might always want to reject just to see if we can get a higher concession...
        # For now, just always reject offer
        if (
            current_villain_offer_utility
            >= self.filtered_outcome_space_dict[next_bid] - 0.01
            if next_bid
            else False
        ):
            if (self.last_to_propose and state.step >= self.total_steps - 1) or (
                not self.last_to_propose and state.step >= self.total_steps - 1
            ):
                return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def utility_model(self, time, e, max_utility):
        """
        Utility model based on time, where we predict the e value of the villain and maximum utility the hero can reach.
        """
        if self.is_edge_agent:
            # FOR SOME REASON, setting this equal to 1 gives the best results. Will need to investigate this later
            max_utility = 1
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.reservation_value),
                ),
                max_utility,
            )
        else:
            # The utility either follows a time based concession model, or the maximum utility
            # This is why np.minimum is used, between the time based and the maximum utility we can achieve
            # the time based concession model is denoted from np.power(time, e)
            # This is vertically shifted by the utility of the first offer from the villain,
            # or the current utility of the hero, whichever is greater
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.current_utility),
                ),
                max_utility,
            )

    def compute_a(self, e) -> float:
        """
        This a value is used to interpolate the maximum utility of the hero based on the villain's concession.
        If the villain concedes a lot, then the maximum utility of the hero is closer to the villain's concession
        Otherwise, it is closer to the hero's maximum utility.

        Closer to 0 means the maximum utility will be closer to the villain's best offer.
        """
        if e <= 1:
            return 0
        else:
            return np.log(e) / np.log(17)

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """
        Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept.
        We will calculate this acceptance threshold by modelling the opponent's offers and seeing how much utility their offer gives.
        If the utility of their offers increase, this means they are conceding. All utilities are normalized to be between 0 and 1.

        We will assume they model their concession as a time based strategy

        The idea is to predict the utility they will concede to at t = 1 or the final timestep. This will be our acceptance threshold
        """

        # TODO: Remove this?
        if state.step == 0:
            return 1

        # if self.is_edge_agent:
        #     return 0.99
        # plt.clf()

        # Normalize the utilities from the villain's offers
        normalized_villain_utilities = np.array(
            [
                self.normalize_utility(utility)
                for utility, time in self.utilities_and_time_from_villain
            ]
        )
        villain_times = np.array(
            [time for _, time in self.utilities_and_time_from_villain]
        )

        # print(self.utilities_and_time_from_villain)
        # plt.plot(villain_times, normalized_villain_utilities, 'ro', label='Observed')

        normalized_hero_maximum_utility = self.normalize_utility(
            self.filtered_outcome_space[0][1]
        )
        normalized_best_utility_from_villain = self.normalize_utility(
            self.best_offer_from_villain_and_utiility[1]
        )

        # Returns a value between 0 and 1. 0 means that the upperbound for the max utility will be closer to
        #   the villain's best offer, 1 means that the upperbound will be the hero's maximum utility
        a = self.compute_a(self.e_value)

        # Setting the bounds of the maximum utility for the curve fit
        # Setting the bound to 0.5 is a huge assumption, but it helps if the curve fit predicts the maximum utility to be much lower
        # I did this because when against a villain that never concedes, the curve fit will predict a very low maximum utility which is not ideal
        low_bound_max_utility = max(
            min(
                normalized_best_utility_from_villain,
                normalized_hero_maximum_utility - 0.001,
            ),
            0.5,
            self.normalize_utility(self.current_utility),
        )
        # The value of a is used here to bound the maximum utility
        if (
            self.e_value <= 7
            or abs(self.maximum_utility - normalized_best_utility_from_villain) <= 0.2
        ):
            hi_bound_max_utility = (
                normalized_hero_maximum_utility * a
                + normalized_best_utility_from_villain * (1 - a)
            )
        else:
            hi_bound_max_utility = 1

        if hi_bound_max_utility <= low_bound_max_utility:
            hi_bound_max_utility = max(low_bound_max_utility + 0.1, 1)

        predicted_parameters, _ = curve_fit(
            self.utility_model,
            villain_times,
            normalized_villain_utilities,
            # p0=[self.e_value, self.maximum_utility],
            bounds=[(0.01, low_bound_max_utility), (17, hi_bound_max_utility)],
            maxfev=10000,
        )
        e_value = predicted_parameters[0]
        max_utility = predicted_parameters[1]
        # print("e value: ", e_value, "rv", reservation_value, "max utility", max_utility)

        # This was originally here to interpolate the e value, but currently not used. Would have also been used to interpolate the maximum utility
        gamma = 0
        self.e_value = gamma * self.e_value + (1 - gamma) * e_value
        self.maximum_utility = max_utility
        # Obtain the last time step they will offer/respond to something
        last_time_step = (self.total_steps - 1 + 1) / (self.total_steps + 1)

        # In the case that they are very boulware, or not conceding, we set last time step to 1, or simply the predicted max utility
        # This is because the last_time_step may be too "low", especially when n_steps is small. Then it will predict a very low utility.
        # Let me know if this is confusing
        if self.e_value > 7:
            last_time_step = 1

        # TODO: Check if this is always a good idea
        # last_time_step = 1
        predicted_utility = self.utility_model(
            last_time_step, self.e_value, max_utility
        )
        # print(hero_maximum_utility * predicted_utility, self.best_offer_from_villain_and_utiility[1])

        # t_plot = np.linspace(0, 1, 100)
        # u_plot = self.utility_model(t_plot, self.e_value, max_utility)
        # plt.plot(t_plot, u_plot, 'b-', label=f'Fitted curve')
        # print("step", state.step, self.interpolated_reservation_value)
        # print("predicted utility", predicted_utility, "thingie returned", hero_maximum_utility * predicted_utility)

        # When curve fit is weak, the minimum proportion of maximum utility to offer
        # Idea is to set a high acceptance threshold, so we do not accept offers that are too low
        weak_curve_fit_factor = max(
            0.8, self.maximum_utility
        )  # TODO: change this to maximum utility?

        # Number of unique offers from the villain to determine if we can use the curve fit
        # Not really if this is a good number... will need to fine tune this
        minimum_unique_offers = max(
            10, 0.75 * self.total_steps
        )  # TODO: 0.75, maybe just always keep minimum_unique_offers high?

        # If at last couple of timesteps, use the predicted utility regardless of the number of unique offers
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # if normalized_hero_maximum_utility * predicted_utility <= self.normalize_utility(self.current_utility):
            #     print("normalized_hero_maximum_utility", normalized_hero_maximum_utility, "predicted_utility", predicted_utility, "current utility", self.normalize_utility(self.current_utility))
            # assert normalized_hero_maximum_utility * predicted_utility > self.normalize_utility(self.current_utility)
            return normalized_hero_maximum_utility * predicted_utility

        # Otherwise, if the curve fit is weak, we will always set a high acceptance threshold, based on the weak curve fit factor
        # if len(self.unique_proposals_from_villain) < minimum_unique_offers:
        if len(self.utilities_and_time_from_villain) < minimum_unique_offers:
            acceptance_threshold = (
                weak_curve_fit_factor * normalized_hero_maximum_utility
            )
            # print("neg index", self.current_negotiation_index, "unique proposals", self.unique_proposals_from_villain, "acceptance_threshold", acceptance_threshold, "best utility", self.filtered_outcome_space[0][1])
            # if acceptance_threshold <= self.normalize_utility(self.current_utility):
            #     print("lowbound", low_bound_max_utility, "maximum utility", self.maximum_utility, "acceptance trheshodl", acceptance_threshold, "current uitlty", self.normalize_utility(self.current_utility))
            # assert acceptance_threshold > self.normalize_utility(self.current_utility)
            return acceptance_threshold

        return normalized_hero_maximum_utility * predicted_utility

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent using the acceptance threshold.

        There are 3 "regions" that we can be in based on time.

        In the aggressive region, based on stop aggressive timestep, we will offer bids that are above the acceptance threshold.
        After the aggressive region, we will offer bids that are at the acceptance threshold.
        On the last couple of steps (or last timestep to propose), we will offer bids that are between the acceptance threshold and the best offer from the villain.
        """
        # Discounting for updating of item count
        # This should be < 1, but I was just fine tuning things. Will test on lower values
        eta = 0.25

        # timestep to when to stop offering bid above point
        stop_aggressive_timestep = 0.9

        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                self.discounted_item_counter[indexed_issue] += state.relative_time

                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")
        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        # In the last step to propose, propose an offer in between the best offer from the villain (slightly higher than this) and the acceptance threshold
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # If best offer is close to acceptance threshold, just return it. Without this line, often result in no agreements reached
            if (
                acceptance_threshold - 0.1
                <= normalized_best_offer_from_villain_and_utility[1]
            ):
                if (
                    normalized_best_offer_from_villain_and_utility[0]
                    in self.normalized_filtered_outcome_space_dict
                ):
                    return normalized_best_offer_from_villain_and_utility[0]

            # Find offers in the range of acceptance threshold and best offer from villain
            fscore = 0
            for outcome, utility in self.normalized_filtered_outcome_space:
                # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                if (
                    self.e_value <= 7
                    or abs(
                        self.maximum_utility
                        - normalized_best_offer_from_villain_and_utility[1]
                    )
                    <= 0.2
                ):
                    if (
                        utility
                        >= normalized_best_offer_from_villain_and_utility[1] * 0.5
                        + acceptance_threshold * 0.5
                        and utility <= acceptance_threshold
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        fscore = 0
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore

                    # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                    elif utility < normalized_best_offer_from_villain_and_utility[1]:
                        break
                else:
                    fscore = 0
                    if (
                        utility
                        >= acceptance_threshold * 0.75
                        + normalized_best_offer_from_villain_and_utility[1] * 0.25
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore
                    else:
                        break

            # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
            if len(outcome_space_above_threshold) == 0:
                if (
                    normalized_best_offer_from_villain_and_utility[0]
                    not in self.normalized_filtered_outcome_space_dict
                ):
                    # Just find the highest fscore in the outcome space
                    for outcome, utility in self.normalized_filtered_outcome_space:
                        fscore = 0
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore
                else:
                    return normalized_best_offer_from_villain_and_utility[0]
        else:
            for outcome, utility in self.normalized_filtered_outcome_space:
                fscore = 0
                if utility >= acceptance_threshold:
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore
                else:
                    break

        # If we are last step to propose, or we are greater than aggressive timestep, return random outcome from highest fscore outcomes
        if (
            (self.last_to_propose and state.step >= self.total_steps - 1)
            or (not self.last_to_propose and state.step >= self.total_steps - 2)
            or state.relative_time >= stop_aggressive_timestep
        ):
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            else:
                # print(len(outcomes_and_highest_fscore[0]), outcomes_and_highest_fscore[1])
                random_choice = choice(outcomes_and_highest_fscore[0])
                # if self.normalized_filtered_outcome_space_dict[random_choice] > acceptance_threshold:
                # print("next bid", random_choice, "utility", self.normalized_filtered_outcome_space_dict[random_choice], "acceptance threshold", acceptance_threshold)

                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # We return some bid above the acceptance threshold, but we rotate through the bids to ensure we do not propose the same bid over and over again
                # Make it look like we are doing some kind of conceding
                next_bid = outcome_space_above_threshold[
                    -1 - self.weak_curve_fit_index % len(outcome_space_above_threshold)
                ]
                self.weak_curve_fit_index += 1

                return next_bid

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=10000,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        # temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class RUFL(ANL2025Negotiator):
    def init(self):
        self.debug = True
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Variables that must be initalized at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = 0  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.utilities_and_time_from_villain = (
            None  # utilities and relative time of villain's offers
        )
        self.e_value = None  # e value of the utility model. How do the villains concede based on time
        self.unique_proposals_from_villain = None  # number of unique proposals from the villain | used to determine if the curve fit has enough data to be used
        self.maximum_utility = (
            None  # predicted maximum utility the hero can get from the villain's offers
        )
        self.weak_curve_fit_index = None  # When the curve fit is weak, we will use this index to determine which offer to propose next.
        # This is used to ensure that we do not propose the same offer over and over again when the curve fit is weak.

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 25_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

            # Dictionary to store expected values of outcomes for lookahead
            self.value_dictionary = None

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value
        #     else:
        #         return self.ufun.reserved_value
        # else:
        #     return self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # if not is_edge_agent(self):
        #     for side_ufun in self.ufun.side_ufuns():
        #         print("negotiation index", self.current_negotiation_index, side_ufun)

        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            # We are first to act, and also last to propose
            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            self.current_utility = self.get_current_utility()

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        # No better outcomes
        if len(self.filtered_outcome_space) == 0:
            return None

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            # TODO: Change this? Maybe just return some rotation of the top offers?
            return self.filtered_outcome_space[0][0]

        # TODO: This ideally should not happen, fix
        # Sometimes acceptance threshold is very low if opponent's offers are low and curve fit is weak
        # In this case, just return the best outcome for the hero
        if next_bid not in self.filtered_outcome_space_dict:
            return self.filtered_outcome_space[0][0]

        # If the offer from the villain is greater than the  predicted concession of the villain
        # Then we just return their offer. HOWEVER, this is just for now, I may only want to offer
        # their offer if it's the last step to propose

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if (
                self.normalize_utility(self.best_offer_from_villain_and_utiility[1])
                >= acceptance_threshold - 0.01
            ):
                if (
                    self.best_offer_from_villain_and_utiility[0]
                    in self.normalized_filtered_outcome_space_dict
                ):
                    return self.best_offer_from_villain_and_utiility[0]

        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
        if not has_made_offer:  # We are first to respond, so we do not propose last
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = False
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = nmi
            self.total_steps = self.nmi.n_steps

            self.current_utility = self.get_current_utility()

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        # End negotiation if there's no outcome better for you
        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        # Update offers from villain and the best one
        current_villain_offer = state.current_offer

        if self.use_lookahead():
            current_villain_offer_utility = self.value_dictionary[
                self.get_current_gamestate().get_child_from_action(
                    current_villain_offer
                )
            ]
        else:
            current_villain_offer_utility = self.current_ufun(current_villain_offer)

        if current_villain_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_villain_offer,
                current_villain_offer_utility,
            )

        # Update the utilities and time from villain's offers
        if self.last_to_propose:
            # If you propose last, then their step is -1 relative to yours when they offerred
            time_they_proposed = (state.step - 1 + 1) / (self.total_steps + 1)
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, time_they_proposed)
            )
        else:
            # if you respond last, then their step is the same as yours
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, state.relative_time)
            )

        self.unique_proposals_from_villain.add(current_villain_offer)

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        # Reject if offer not even above
        # TODO: What if all of their offers are not in the filtered outcome space?
        if current_villain_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        # TODO: Maybe only accept if it's above a certain threshold of the reservation value
        # If we are last to respond, just accept offer if better than our reservation value
        if not self.last_to_propose:
            if (
                state.step == self.total_steps - 1
            ):  # the last step for responding if we respond last is n-1
                if current_villain_offer_utility > self.reservation_value + 0.1:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        # Check if the current offer is acceptable
        # TODO: We might always want to reject just to see if we can get a higher concession...
        # For now, just always reject offer
        if (
            current_villain_offer_utility
            >= self.filtered_outcome_space_dict[next_bid] - 0.01
            if next_bid
            else False
        ):
            if (self.last_to_propose and state.step >= self.total_steps - 1) or (
                not self.last_to_propose and state.step >= self.total_steps - 2
            ):
                return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def utility_model(self, time, e, max_utility):
        """
        Utility model based on time, where we predict the e value of the villain and maximum utility the hero can reach.
        """
        if self.is_edge_agent:
            # FOR SOME REASON, setting this equal to 1 gives the best results. Will need to investigate this later
            max_utility = 1
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.reservation_value),
                ),
                max_utility,
            )
        else:
            # The utility either follows a time based concession model, or the maximum utility
            # This is why np.minimum is used, between the time based and the maximum utility we can achieve
            # the time based concession model is denoted from np.power(time, e)
            # This is vertically shifted by the utility of the first offer from the villain,
            # or the current utility of the hero, whichever is greater
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.current_utility),
                ),
                max_utility,
            )

    def compute_a(self, e) -> float:
        """
        This a value is used to interpolate the maximum utility of the hero based on the villain's concession.
        If the villain concedes a lot, then the maximum utility of the hero is closer to the villain's concession
        Otherwise, it is closer to the hero's maximum utility.

        Closer to 0 means the maximum utility will be closer to the villain's best offer.
        """
        if e <= 1:
            return 0
        else:
            return np.log(e) / np.log(17)

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """
        Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept.
        We will calculate this acceptance threshold by modelling the opponent's offers and seeing how much utility their offer gives.
        If the utility of their offers increase, this means they are conceding. All utilities are normalized to be between 0 and 1.

        We will assume they model their concession as a time based strategy

        The idea is to predict the utility they will concede to at t = 1 or the final timestep. This will be our acceptance threshold
        """

        if state.step == 0:
            return 1

        # if self.is_edge_agent:
        #     return 0.99
        # plt.clf()

        # Normalize the utilities from the villain's offers
        normalized_villain_utilities = np.array(
            [
                self.normalize_utility(utility)
                for utility, time in self.utilities_and_time_from_villain
            ]
        )
        villain_times = np.array(
            [time for _, time in self.utilities_and_time_from_villain]
        )

        # print(self.utilities_and_time_from_villain)
        # plt.plot(villain_times, normalized_villain_utilities, 'ro', label='Observed')

        normalized_hero_maximum_utility = self.normalize_utility(
            self.filtered_outcome_space[0][1]
        )
        normalized_best_utility_from_villain = self.normalize_utility(
            self.best_offer_from_villain_and_utiility[1]
        )

        # Returns a value between 0 and 1. 0 means that the upperbound for the max utility will be closer to
        #   the villain's best offer, 1 means that the upperbound will be the hero's maximum utility
        a = self.compute_a(self.e_value)

        # Setting the bounds of the maximum utility for the curve fit
        # Setting the bound to 0.5 is a huge assumption, but it helps if the curve fit predicts the maximum utility to be much lower
        # I did this because when against a villain that never concedes, the curve fit will predict a very low maximum utility which is not ideal
        low_bound_max_utility = max(
            min(
                normalized_best_utility_from_villain,
                normalized_hero_maximum_utility - 0.001,
            ),
            0.5,
            self.current_utility,
        )
        # The value of a is used here to bound the maximum utility

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if self.e_value >= 6:
                if (
                    abs(self.maximum_utility - normalized_best_utility_from_villain)
                    < 0.15
                ):
                    hi_bound_max_utility = low_bound_max_utility + 0.02
                else:
                    hi_bound_max_utility = max(
                        normalized_hero_maximum_utility * a
                        + normalized_best_utility_from_villain * (1 - a),
                        0.6,
                    )
            else:
                if (
                    abs(self.maximum_utility - normalized_best_utility_from_villain)
                    < 0.15
                ):
                    hi_bound_max_utility = low_bound_max_utility + 0.02
                hi_bound_max_utility = low_bound_max_utility + 0.02
        else:
            hi_bound_max_utility = 1

        if self.is_edge_agent:
            hi_bound_max_utility = max(hi_bound_max_utility, 0.8)

        if hi_bound_max_utility <= low_bound_max_utility:
            hi_bound_max_utility = low_bound_max_utility + 0.02

        # if self.e_value >= 6:
        #     if abs(self.maximum_utility - normalized_hero_maximum_utility) < 0.15 and ((self.last_to_propose and state.step >= self.total_steps - 1) or (not self.last_to_propose and state.step >= self.total_steps - 2)):
        #         hi_bound_max_utility = low_bound_max_utility + 0.02
        #     else:
        #         hi_bound_max_utility = max(normalized_hero_maximum_utility*a + normalized_best_utility_from_villain*(1-a), 0.6)
        # else:
        #     hi_bound_max_utility = max(normalized_hero_maximum_utility*a + normalized_best_utility_from_villain*(1-a), 0.6)

        # if hi_bound_max_utility <= low_bound_max_utility:
        #     hi_bound_max_utility = low_bound_max_utility + 0.02

        predicted_parameters, _ = curve_fit(
            self.utility_model,
            villain_times,
            normalized_villain_utilities,
            # p0=[self.e_value, self.maximum_utility],
            bounds=[(0.01, low_bound_max_utility), (17, hi_bound_max_utility)],
            maxfev=10000,
        )
        e_value = predicted_parameters[0]
        max_utility = predicted_parameters[1]
        # print("e value: ", e_value, "rv", reservation_value, "max utility", max_utility)

        # This was originally here to interpolate the e value, but currently not used. Would have also been used to interpolate the maximum utility
        gamma = 0
        self.e_value = gamma * self.e_value + (1 - gamma) * e_value
        self.maximum_utility = max_utility
        # Obtain the last time step they will offer/respond to something
        last_time_step = (self.total_steps - 1 + 1) / (self.total_steps + 1)

        # In the case that they are very boulware, or not conceding, we set last time step to 1, or simply the predicted max utility
        # This is because the last_time_step may be too "low", especially when n_steps is small. Then it will predict a very low utility.
        # Let me know if this is confusing
        if self.e_value > 6:
            last_time_step = 1
        # last_time_step = 1
        predicted_utility = self.utility_model(
            last_time_step, self.e_value, max_utility
        )
        # print(hero_maximum_utility * predicted_utility, self.best_offer_from_villain_and_utiility[1])

        # t_plot = np.linspace(0, 1, 100)
        # u_plot = self.utility_model(t_plot, self.e_value, max_utility)
        # plt.plot(t_plot, u_plot, 'b-', label=f'Fitted curve')
        # print("step", state.step, self.interpolated_reservation_value)
        # print("predicted utility", predicted_utility, "thingie returned", hero_maximum_utility * predicted_utility)

        # When curve fit is weak, the minimum proportion of maximum utility to offer
        # Idea is to set a high acceptance threshold, so we do not accept offers that are too low
        weak_curve_fit_factor = max(0.8, predicted_utility)

        # Number of unique offers from the villain to determine if we can use the curve fit
        # Not really if this is a good number... will need to fine tune this
        minimum_unique_offers = max(10, 0.95 * self.total_steps)

        # If at last couple of timesteps, use the predicted utility regardless of the number of unique offers
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            return max(
                normalized_hero_maximum_utility * predicted_utility,
                self.current_utility + 0.05,
            )

        # Otherwise, if the curve fit is weak, we will always set a high acceptance threshold, based on the weak curve fit factor
        # if len(self.unique_proposals_from_villain) < minimum_unique_offers:
        if len(self.utilities_and_time_from_villain) < minimum_unique_offers:
            acceptance_threshold = (
                weak_curve_fit_factor * normalized_hero_maximum_utility
            )
            # print("neg index", self.current_negotiation_index, "unique proposals", self.unique_proposals_from_villain, "acceptance_threshold", acceptance_threshold, "best utility", self.filtered_outcome_space[0][1])
            return max(acceptance_threshold, self.current_utility + 0.05)

        return max(
            normalized_hero_maximum_utility * predicted_utility,
            self.current_utility + 0.05,
        )

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent using the acceptance threshold.

        There are 3 "regions" that we can be in based on time.

        In the aggressive region, based on stop aggressive timestep, we will offer bids that are above the acceptance threshold.
        After the aggressive region, we will offer bids that are at the acceptance threshold.
        On the last couple of steps (or last timestep to propose), we will offer bids that are between the acceptance threshold and the best offer from the villain.
        """
        # Discounting for updating of item count
        # This should be < 1, but I was just fine tuning things. Will test on lower values
        eta = 0.8

        # timestep to when to stop offering bid above point
        stop_aggressive_timestep = 0.9

        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")
        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        # if self.is_edge_agent:
        #     return self.filtered_outcome_space[0][0]

        # In the last step to propose, propose an offer in between the best offer from the villain (slightly higher than this) and the acceptance threshold
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # If best offer is close to acceptance threshold, just return it. Without this line, often result in no agreements reached
            if (
                acceptance_threshold - 0.1
                <= normalized_best_offer_from_villain_and_utility[1]
            ):
                if (
                    normalized_best_offer_from_villain_and_utility[0]
                    in self.normalized_filtered_outcome_space_dict
                ):
                    return normalized_best_offer_from_villain_and_utility[0]

            # Find offers in the range of acceptance threshold and best offer from villain
            if self.e_value >= 6:
                for outcome, utility in self.normalized_filtered_outcome_space:
                    # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                    if (
                        utility
                        > normalized_best_offer_from_villain_and_utility[1] * 0.25
                        + acceptance_threshold * 0.75
                        and utility <= acceptance_threshold
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        fscore = 0
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore

                    # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                    elif utility < normalized_best_offer_from_villain_and_utility[1]:
                        break
            else:
                for outcome, utility in self.normalized_filtered_outcome_space:
                    # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                    if (
                        utility
                        > normalized_best_offer_from_villain_and_utility[1] * 0.75
                        + acceptance_threshold * 0.25
                        and utility <= acceptance_threshold
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        fscore = 0
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore

                    # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                    elif utility < normalized_best_offer_from_villain_and_utility[1]:
                        break
            # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
            if len(outcome_space_above_threshold) == 0:
                if (
                    normalized_best_offer_from_villain_and_utility[0]
                    in self.normalized_filtered_outcome_space_dict
                ):
                    return normalized_best_offer_from_villain_and_utility[0]
                else:
                    for outcome, utility in self.normalized_filtered_outcome_space:
                        if utility > 0.5:
                            outcome_space_above_threshold.append(outcome)
                            fscore = 0
                            for index, issue in enumerate(outcome):
                                indexed_issue = str(index) + str(issue)
                                fscore += self.discounted_item_counter[indexed_issue]

                            if fscore == outcomes_and_highest_fscore[1]:
                                outcomes_and_highest_fscore[0].append(outcome)

                            if fscore > outcomes_and_highest_fscore[1]:
                                outcomes_and_highest_fscore = [outcome], fscore
        # elif state.relative_time >= stop_aggressive_timestep:
        #     # Find offers in the range of acceptance threshold
        #     fscore = 0
        #     for outcome, utility in self.normalized_filtered_outcome_space:
        #         # Maybe we can expand this range? Idk how helpful it will be
        #         if utility >= acceptance_threshold - 0.01 and utility <= acceptance_threshold:
        #             outcome_space_above_threshold.append(outcome)
        #             # Calculate the fscore of the outcome
        #             fscore = 0
        #             for index, issue in enumerate(outcome):
        #                 indexed_issue = str(index) + str(issue)
        #                 fscore += self.discounted_item_counter[indexed_issue]

        #             if fscore == outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore[0].append(outcome)

        #             if fscore > outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore = [outcome], fscore
        #         else:
        #             break
        #     # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
        #     if len(outcome_space_above_threshold) == 0:
        #         for outcome, utility in self.normalized_filtered_outcome_space:
        #             if utility < acceptance_threshold:
        #                 outcomes_and_highest_fscore[0].append(outcome)
        #                 break
        else:
            for outcome, utility in self.normalized_filtered_outcome_space:
                fscore = 0
                if utility >= acceptance_threshold:
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore
                else:
                    break

        # If we are last step to propose, or we are greater than aggressive timestep, return random outcome from highest fscore outcomes
        if (
            (self.last_to_propose and state.step >= self.total_steps - 1)
            or (not self.last_to_propose and state.step >= self.total_steps - 2)
            or state.relative_time >= stop_aggressive_timestep
        ):
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            # elif len(outcomes_and_highest_fscore[0]) == 1:
            #     return self.best_offer_from_villain_and_utiility[0]
            else:
                # print(len(outcomes_and_highest_fscore[0]), outcomes_and_highest_fscore[1])
                random_choice = choice(outcomes_and_highest_fscore[0])
                # best_outcome = min(outcomes_and_highest_fscore[0], key=lambda x: self.normalized_filtered_outcome_space_dict[x])
                # if self.normalized_filtered_outcome_space_dict[random_choice] > acceptance_threshold:
                # print("next bid", random_choice, "utility", self.normalized_filtered_outcome_space_dict[random_choice], "acceptance threshold", acceptance_threshold)
                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # We return some bid above the acceptance threshold, but we rotate through the bids to ensure we do not propose the same bid over and over again
                # Make it look like we are doing some kind of conceding
                next_bid = outcome_space_above_threshold[
                    -1 - self.weak_curve_fit_index % len(outcome_space_above_threshold)
                ]
                self.weak_curve_fit_index += 1
                return next_bid

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=100,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class NewishUtilityFitLookAheadAgent(ANL2025Negotiator):
    def init(self):
        self.debug = True
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Variables that must be initalized at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = 0  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.utilities_and_time_from_villain = (
            None  # utilities and relative time of villain's offers
        )
        self.e_value = None  # e value of the utility model. How do the villains concede based on time
        self.unique_proposals_from_villain = None  # number of unique proposals from the villain | used to determine if the curve fit has enough data to be used
        self.maximum_utility = (
            None  # predicted maximum utility the hero can get from the villain's offers
        )
        self.weak_curve_fit_index = None  # When the curve fit is weak, we will use this index to determine which offer to propose next.
        # This is used to ensure that we do not propose the same offer over and over again when the curve fit is weak.

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

            # Dictionary to store expected values of outcomes for lookahead
            self.value_dictionary = None

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value
        #     else:
        #         return self.ufun.reserved_value
        # else:
        #     return self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # if not is_edge_agent(self):
        #     for side_ufun in self.ufun.side_ufuns():
        #         print("negotiation index", self.current_negotiation_index, side_ufun)

        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            # We are first to act, and also last to propose
            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # if self.current_negotiation_index > 0:
            #     last_agreement = get_agreement_at_index(self, self.current_negotiation_index-1)
            #     last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index-1]
            #     last_neg_util = last_ufun.eval(last_agreement)
            #     self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        # No better outcomes
        if len(self.filtered_outcome_space) == 0:
            return None

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            # TODO: Change this? Maybe just return some rotation of the top offers?
            return self.filtered_outcome_space[0][0]

        # TODO: This ideally should not happen, fix
        # Sometimes acceptance threshold is very low if opponent's offers are low and curve fit is weak
        # In this case, just return the best outcome for the hero
        if next_bid not in self.filtered_outcome_space_dict:
            return self.filtered_outcome_space[0][0]

        # If the offer from the villain is greater than the  predicted concession of the villain
        # Then we just return their offer. HOWEVER, this is just for now, I may only want to offer
        # their offer if it's the last step to propose

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if (
                self.normalize_utility(self.best_offer_from_villain_and_utiility[1])
                >= acceptance_threshold - 0.01
            ):
                return self.best_offer_from_villain_and_utiility[0]

        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
        if not has_made_offer:  # We are first to respond, so we do not propose last
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = False
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # self.current_utility = 0
            # if self.current_negotiation_index > 0:
            #     last_agreement = get_agreement_at_index(self, self.current_negotiation_index-1)
            #     last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index-1]
            #     last_neg_util = last_ufun.eval(last_agreement)
            #     self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        # End negotiation if there's no outcome better for you
        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        # Update offers from villain and the best one
        current_villain_offer = state.current_offer

        if self.use_lookahead():
            current_villain_offer_utility = self.value_dictionary[
                self.get_current_gamestate().get_child_from_action(
                    current_villain_offer
                )
            ]
        else:
            current_villain_offer_utility = self.current_ufun(current_villain_offer)

        if current_villain_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_villain_offer,
                current_villain_offer_utility,
            )

        # Update the utilities and time from villain's offers
        if self.last_to_propose:
            # If you propose last, then their step is -1 relative to yours when they offerred
            time_they_proposed = (state.step - 1 + 1) / (self.total_steps + 1)
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, time_they_proposed)
            )
        else:
            # if you respond last, then their step is the same as yours
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, state.relative_time)
            )

        self.unique_proposals_from_villain.add(current_villain_offer)

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        # TODO: Maybe only accept if it's above a certain threshold of the reservation value
        # If we are last to respond, just accept offer if better than our reservation value
        if not self.last_to_propose:
            if (
                state.step == self.total_steps - 1
            ):  # the last step for responding if we respond last is n-1
                if current_villain_offer_utility > self.reservation_value + 0.1:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        # Reject if offer not even above
        # TODO: What if all of their offers are not in the filtered outcome space?
        if current_villain_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        # Check if the current offer is acceptable
        # TODO: We might always want to reject just to see if we can get a higher concession...
        # For now, just always reject offer
        if (
            current_villain_offer_utility
            >= self.filtered_outcome_space_dict[next_bid] - 0.01
            if next_bid
            else False
        ):
            if (self.last_to_propose and state.step >= self.total_steps - 1) or (
                not self.last_to_propose and state.step >= self.total_steps - 2
            ):
                return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def utility_model(self, time, e, max_utility):
        """
        Utility model based on time, where we predict the e value of the villain and maximum utility the hero can reach.
        """
        if self.is_edge_agent:
            # FOR SOME REASON, setting this equal to 1 gives the best results. Will need to investigate this later
            max_utility = 1
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.reservation_value),
                ),
                max_utility,
            )
        else:
            # The utility either follows a time based concession model, or the maximum utility
            # This is why np.minimum is used, between the time based and the maximum utility we can achieve
            # the time based concession model is denoted from np.power(time, e)
            # This is vertically shifted by the utility of the first offer from the villain,
            # or the current utility of the hero, whichever is greater
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.current_utility),
                ),
                max_utility,
            )

    def compute_a(self, e) -> float:
        """
        This a value is used to interpolate the maximum utility of the hero based on the villain's concession.
        If the villain concedes a lot, then the maximum utility of the hero is closer to the villain's concession
        Otherwise, it is closer to the hero's maximum utility.

        Closer to 0 means the maximum utility will be closer to the villain's best offer.
        """
        if e <= 1:
            return 0
        else:
            return np.log(e) / np.log(7)

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """
        Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept.
        We will calculate this acceptance threshold by modelling the opponent's offers and seeing how much utility their offer gives.
        If the utility of their offers increase, this means they are conceding. All utilities are normalized to be between 0 and 1.

        We will assume they model their concession as a time based strategy

        The idea is to predict the utility they will concede to at t = 1 or the final timestep. This will be our acceptance threshold
        """

        # TODO: Remove this?
        if state.step == 0:
            return 1

        # if self.is_edge_agent:
        #     return 0.99
        # plt.clf()

        # Normalize the utilities from the villain's offers
        normalized_villain_utilities = np.array(
            [
                self.normalize_utility(utility)
                for utility, time in self.utilities_and_time_from_villain
            ]
        )
        villain_times = np.array(
            [time for _, time in self.utilities_and_time_from_villain]
        )

        # print(self.utilities_and_time_from_villain)
        # plt.plot(villain_times, normalized_villain_utilities, 'ro', label='Observed')

        normalized_hero_maximum_utility = self.normalize_utility(
            self.filtered_outcome_space[0][1]
        )
        normalized_best_utility_from_villain = self.normalize_utility(
            self.best_offer_from_villain_and_utiility[1]
        )

        # Returns a value between 0 and 1. 0 means that the upperbound for the max utility will be closer to
        #   the villain's best offer, 1 means that the upperbound will be the hero's maximum utility
        a = self.compute_a(self.e_value)

        # Setting the bounds of the maximum utility for the curve fit
        # Setting the bound to 0.5 is a huge assumption, but it helps if the curve fit predicts the maximum utility to be much lower
        # I did this because when against a villain that never concedes, the curve fit will predict a very low maximum utility which is not ideal
        low_bound_max_utility = max(
            min(
                normalized_best_utility_from_villain,
                normalized_hero_maximum_utility - 0.001,
            ),
            0.5,
        )
        # The value of a is used here to bound the maximum utility
        if self.e_value <= 7:
            hi_bound_max_utility = (
                normalized_hero_maximum_utility * a
                + normalized_best_utility_from_villain * (1 - a)
            )
        else:
            hi_bound_max_utility = 1

        if hi_bound_max_utility <= low_bound_max_utility:
            hi_bound_max_utility = low_bound_max_utility + 0.01

        predicted_parameters, _ = curve_fit(
            self.utility_model,
            villain_times,
            normalized_villain_utilities,
            # p0=[self.e_value, self.maximum_utility],
            bounds=[(0.01, low_bound_max_utility), (17, hi_bound_max_utility)],
            maxfev=10000,
        )
        e_value = predicted_parameters[0]
        max_utility = predicted_parameters[1]
        # print("e value: ", e_value, "rv", reservation_value, "max utility", max_utility)

        # This was originally here to interpolate the e value, but currently not used. Would have also been used to interpolate the maximum utility
        gamma = 0
        self.e_value = gamma * self.e_value + (1 - gamma) * e_value
        self.maximum_utility = max_utility
        # Obtain the last time step they will offer/respond to something
        last_time_step = (self.total_steps - 1 + 1) / (self.total_steps + 1)

        # In the case that they are very boulware, or not conceding, we set last time step to 1, or simply the predicted max utility
        # This is because the last_time_step may be too "low", especially when n_steps is small. Then it will predict a very low utility.
        # Let me know if this is confusing
        if self.e_value > 10:
            last_time_step = 1

        # TODO: Check if this is always a good idea
        last_time_step = 1
        predicted_utility = self.utility_model(
            last_time_step, self.e_value, max_utility
        )
        # print(hero_maximum_utility * predicted_utility, self.best_offer_from_villain_and_utiility[1])

        # t_plot = np.linspace(0, 1, 100)
        # u_plot = self.utility_model(t_plot, self.e_value, max_utility)
        # plt.plot(t_plot, u_plot, 'b-', label=f'Fitted curve')
        # print("step", state.step, self.interpolated_reservation_value)
        # print("predicted utility", predicted_utility, "thingie returned", hero_maximum_utility * predicted_utility)

        # When curve fit is weak, the minimum proportion of maximum utility to offer
        # Idea is to set a high acceptance threshold, so we do not accept offers that are too low
        weak_curve_fit_factor = max(
            0.8, predicted_utility
        )  # TODO: change this to maximum utility?

        # Number of unique offers from the villain to determine if we can use the curve fit
        # Not really if this is a good number... will need to fine tune this
        minimum_unique_offers = max(
            10, 0.75 * self.total_steps
        )  # TODO: 0.75, maybe just always keep minimum_unique_offers high?

        # If at last couple of timesteps, use the predicted utility regardless of the number of unique offers
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            return normalized_hero_maximum_utility * predicted_utility

        # Otherwise, if the curve fit is weak, we will always set a high acceptance threshold, based on the weak curve fit factor
        # if len(self.unique_proposals_from_villain) < minimum_unique_offers:
        if len(self.utilities_and_time_from_villain) < minimum_unique_offers:
            acceptance_threshold = (
                weak_curve_fit_factor * normalized_hero_maximum_utility
            )
            # print("neg index", self.current_negotiation_index, "unique proposals", self.unique_proposals_from_villain, "acceptance_threshold", acceptance_threshold, "best utility", self.filtered_outcome_space[0][1])
            return acceptance_threshold

        return normalized_hero_maximum_utility * predicted_utility

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent using the acceptance threshold.

        There are 3 "regions" that we can be in based on time.

        In the aggressive region, based on stop aggressive timestep, we will offer bids that are above the acceptance threshold.
        After the aggressive region, we will offer bids that are at the acceptance threshold.
        On the last couple of steps (or last timestep to propose), we will offer bids that are between the acceptance threshold and the best offer from the villain.
        """
        # Discounting for updating of item count
        # This should be < 1, but I was just fine tuning things. Will test on lower values
        eta = 0.8

        # timestep to when to stop offering bid above point
        stop_aggressive_timestep = 0.9

        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                # self.discounted_item_counter[indexed_issue] += (state.relative_time)

                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")
        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        # if self.is_edge_agent:
        #     return self.filtered_outcome_space[0][0]

        # In the last step to propose, propose an offer in between the best offer from the villain (slightly higher than this) and the acceptance threshold
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # If best offer is close to acceptance threshold, just return it. Without this line, often result in no agreements reached
            if (
                acceptance_threshold - 0.1
                <= normalized_best_offer_from_villain_and_utility[1]
            ):
                return normalized_best_offer_from_villain_and_utility[0]

            # Find offers in the range of acceptance threshold and best offer from villain
            fscore = 0
            for outcome, utility in self.normalized_filtered_outcome_space:
                # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                if self.e_value <= 7:
                    if (
                        utility
                        > normalized_best_offer_from_villain_and_utility[1] * 0.5
                        + acceptance_threshold * 0.5
                        and utility <= acceptance_threshold
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        fscore = 0
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore

                    # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                    elif utility < normalized_best_offer_from_villain_and_utility[1]:
                        break
                else:
                    fscore = 0
                    if (
                        utility
                        >= acceptance_threshold * 0.75
                        + normalized_best_offer_from_villain_and_utility[1] * 0.25
                    ):
                        outcome_space_above_threshold.append(outcome)
                        # Calculate the fscore of the outcome
                        for index, issue in enumerate(outcome):
                            indexed_issue = str(index) + str(issue)
                            fscore += self.discounted_item_counter[indexed_issue]

                        if fscore == outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore[0].append(outcome)

                        if fscore > outcomes_and_highest_fscore[1]:
                            outcomes_and_highest_fscore = [outcome], fscore
                    else:
                        break

            # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
            if len(outcome_space_above_threshold) == 0:
                return normalized_best_offer_from_villain_and_utility[0]
        else:
            for outcome, utility in self.normalized_filtered_outcome_space:
                fscore = 0
                if utility >= acceptance_threshold:
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore
                else:
                    break

        # If we are last step to propose, or we are greater than aggressive timestep, return random outcome from highest fscore outcomes
        if (
            (self.last_to_propose and state.step >= self.total_steps - 1)
            or (not self.last_to_propose and state.step >= self.total_steps - 2)
            or state.relative_time >= stop_aggressive_timestep
        ):
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            # elif len(outcomes_and_highest_fscore[0]) == 1:
            #     return self.best_offer_from_villain_and_utiility[0]
            else:
                # print(len(outcomes_and_highest_fscore[0]), outcomes_and_highest_fscore[1])
                random_choice = choice(outcomes_and_highest_fscore[0])

                # print("next bid", random_choice, "utility", self.normalized_filtered_outcome_space_dict[random_choice], "acceptance threshold", acceptance_threshold)
                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # We return some bid above the acceptance threshold, but we rotate through the bids to ensure we do not propose the same bid over and over again
                # Make it look like we are doing some kind of conceding
                next_bid = outcome_space_above_threshold[
                    -1 - self.weak_curve_fit_index % len(outcome_space_above_threshold)
                ]
                self.weak_curve_fit_index += 1
                return next_bid

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=10000,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        # temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class UtilityFitLookAheadAgent(ANL2025Negotiator):
    def init(self):
        self.debug = True
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Variables that must be initalized at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = 0  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.utilities_and_time_from_villain = (
            None  # utilities and relative time of villain's offers
        )
        self.e_value = None  # e value of the utility model. How do the villains concede based on time
        self.unique_proposals_from_villain = None  # number of unique proposals from the villain | used to determine if the curve fit has enough data to be used
        self.maximum_utility = (
            None  # predicted maximum utility the hero can get from the villain's offers
        )
        self.weak_curve_fit_index = None  # When the curve fit is weak, we will use this index to determine which offer to propose next.
        # This is used to ensure that we do not propose the same offer over and over again when the curve fit is weak.

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

            # Dictionary to store expected values of outcomes for lookahead
            self.value_dictionary = None

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value
        #     else:
        #         return self.ufun.reserved_value
        # else:
        #     return self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # if not is_edge_agent(self):
        #     for side_ufun in self.ufun.side_ufuns():
        #         print("negotiation index", self.current_negotiation_index, side_ufun)

        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            # We are first to act, and also last to propose
            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # if self.current_negotiation_index > 0:
            #     last_agreement = get_agreement_at_index(self, self.current_negotiation_index-1)
            #     last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index-1]
            #     last_neg_util = last_ufun.eval(last_agreement)
            #     self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        # No better outcomes
        if len(self.filtered_outcome_space) == 0:
            return None

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            # TODO: Change this? Maybe just return some rotation of the top offers?
            return self.filtered_outcome_space[0][0]

        # TODO: This ideally should not happen, fix
        # Sometimes acceptance threshold is very low if opponent's offers are low and curve fit is weak
        # In this case, just return the best outcome for the hero
        if next_bid not in self.filtered_outcome_space_dict:
            return self.filtered_outcome_space[0][0]

        # If the offer from the villain is greater than the  predicted concession of the villain
        # Then we just return their offer. HOWEVER, this is just for now, I may only want to offer
        # their offer if it's the last step to propose

        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            if (
                self.normalize_utility(self.best_offer_from_villain_and_utiility[1])
                >= acceptance_threshold - 0.01
            ):
                return self.best_offer_from_villain_and_utiility[0]

        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)
        if not has_made_offer:  # We are first to respond, so we do not propose last
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = False
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = nmi
            self.total_steps = self.nmi.n_steps

            # FOR SOME REASON THIS PRODUCES BAD RESULTS. Either the side utilities are not being set correctly, or the issue w/ the multiissue is changing the utilities
            # For some reason, the current utility is not being set, so it just filters out all possible negotiations in the space, leading to no negotiations
            self.current_utility = self.get_current_utility()

            # Using this from Xintong's conceding agent for now...
            # self.current_utility = 0
            # if self.current_negotiation_index > 0:
            #     last_agreement = get_agreement_at_index(self, self.current_negotiation_index-1)
            #     last_ufun = self.ufun.side_ufuns()[self.current_negotiation_index-1]
            #     last_neg_util = last_ufun.eval(last_agreement)
            #     self.current_utility = max(last_neg_util, self.current_utility)

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize list of offers from villain and utilities
            self.utilities_and_time_from_villain = []

            # Intialize the e value and maximum utility that will be used to predict the villain's concession
            self.e_value = 17
            self.maximum_utility = 0.99

            self.unique_proposals_from_villain = set()

            self.weak_curve_fit_index = 0

        # End negotiation if there's no outcome better for you
        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        # Update offers from villain and the best one
        current_villain_offer = state.current_offer

        if self.use_lookahead():
            current_villain_offer_utility = self.value_dictionary[
                self.get_current_gamestate().get_child_from_action(
                    current_villain_offer
                )
            ]
        else:
            current_villain_offer_utility = self.current_ufun(current_villain_offer)

        if current_villain_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_villain_offer,
                current_villain_offer_utility,
            )

        # Update the utilities and time from villain's offers
        if self.last_to_propose:
            # If you propose last, then their step is -1 relative to yours when they offerred
            time_they_proposed = (state.step - 1 + 1) / (self.total_steps + 1)
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, time_they_proposed)
            )
        else:
            # if you respond last, then their step is the same as yours
            self.utilities_and_time_from_villain.append(
                (current_villain_offer_utility, state.relative_time)
            )

        self.unique_proposals_from_villain.add(current_villain_offer)

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        # TODO: Maybe only accept if it's above a certain threshold of the reservation value
        # If we are last to respond, just accept offer if better than our reservation value
        if not self.last_to_propose:
            if (
                state.step == self.total_steps - 1
            ):  # the last step for responding if we respond last is n-1
                if current_villain_offer_utility > self.reservation_value + 0.1:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        # Reject if offer not even above
        # TODO: What if all of their offers are not in the filtered outcome space?
        if current_villain_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        # Check if the current offer is acceptable
        # TODO: We might always want to reject just to see if we can get a higher concession...
        # For now, just always reject offer
        if (
            current_villain_offer_utility
            >= self.filtered_outcome_space_dict[next_bid] - 0.01
            if next_bid
            else False
        ):
            if (self.last_to_propose and state.step >= self.total_steps - 1) or (
                not self.last_to_propose and state.step >= self.total_steps - 2
            ):
                return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        pass

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def utility_model(self, time, e, max_utility):
        """
        Utility model based on time, where we predict the e value of the villain and maximum utility the hero can reach.
        """
        if self.is_edge_agent:
            # FOR SOME REASON, setting this equal to 1 gives the best results. Will need to investigate this later
            max_utility = 1
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.reservation_value),
                ),
                max_utility,
            )
        else:
            # The utility either follows a time based concession model, or the maximum utility
            # This is why np.minimum is used, between the time based and the maximum utility we can achieve
            # the time based concession model is denoted from np.power(time, e)
            # This is vertically shifted by the utility of the first offer from the villain,
            # or the current utility of the hero, whichever is greater
            return np.minimum(
                (np.power(time, e))
                + np.maximum(
                    self.normalize_utility(self.utilities_and_time_from_villain[0][0]),
                    self.normalize_utility(self.current_utility),
                ),
                max_utility,
            )

    def compute_a(self, e) -> float:
        """
        This a value is used to interpolate the maximum utility of the hero based on the villain's concession.
        If the villain concedes a lot, then the maximum utility of the hero is closer to the villain's concession
        Otherwise, it is closer to the hero's maximum utility.

        Closer to 0 means the maximum utility will be closer to the villain's best offer.
        """
        if e <= 1:
            return 0
        else:
            return np.log(e) / np.log(17)

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """
        Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept.
        We will calculate this acceptance threshold by modelling the opponent's offers and seeing how much utility their offer gives.
        If the utility of their offers increase, this means they are conceding. All utilities are normalized to be between 0 and 1.

        We will assume they model their concession as a time based strategy

        The idea is to predict the utility they will concede to at t = 1 or the final timestep. This will be our acceptance threshold
        """

        if state.step == 0:
            return 1

        # if self.is_edge_agent:
        #     return 0.99
        # plt.clf()

        # Normalize the utilities from the villain's offers
        normalized_villain_utilities = np.array(
            [
                self.normalize_utility(utility)
                for utility, time in self.utilities_and_time_from_villain
            ]
        )
        villain_times = np.array(
            [time for _, time in self.utilities_and_time_from_villain]
        )

        # print(self.utilities_and_time_from_villain)
        # plt.plot(villain_times, normalized_villain_utilities, 'ro', label='Observed')

        normalized_hero_maximum_utility = self.normalize_utility(
            self.filtered_outcome_space[0][1]
        )
        normalized_best_utility_from_villain = self.normalize_utility(
            self.best_offer_from_villain_and_utiility[1]
        )

        # Returns a value between 0 and 1. 0 means that the upperbound for the max utility will be closer to
        #   the villain's best offer, 1 means that the upperbound will be the hero's maximum utility
        a = self.compute_a(self.e_value)

        # Setting the bounds of the maximum utility for the curve fit
        # Setting the bound to 0.5 is a huge assumption, but it helps if the curve fit predicts the maximum utility to be much lower
        # I did this because when against a villain that never concedes, the curve fit will predict a very low maximum utility which is not ideal
        low_bound_max_utility = max(
            min(
                normalized_best_utility_from_villain,
                normalized_hero_maximum_utility - 0.001,
            ),
            0.5,
        )
        # The value of a is used here to bound the maximum utility
        hi_bound_max_utility = max(
            normalized_hero_maximum_utility * a
            + normalized_best_utility_from_villain * (1 - a),
            0.55,
        )
        if hi_bound_max_utility == low_bound_max_utility:
            hi_bound_max_utility += 0.01

        predicted_parameters, _ = curve_fit(
            self.utility_model,
            villain_times,
            normalized_villain_utilities,
            # p0=[self.e_value, self.maximum_utility],
            bounds=[(0.01, low_bound_max_utility), (17, hi_bound_max_utility)],
            maxfev=10000,
        )
        e_value = predicted_parameters[0]
        max_utility = predicted_parameters[1]
        # print("e value: ", e_value, "rv", reservation_value, "max utility", max_utility)

        # This was originally here to interpolate the e value, but currently not used. Would have also been used to interpolate the maximum utility
        gamma = 0
        self.e_value = gamma * self.e_value + (1 - gamma) * e_value
        self.maximum_utility = max_utility
        # Obtain the last time step they will offer/respond to something
        last_time_step = (self.total_steps - 1 + 1) / (self.total_steps + 1)

        # In the case that they are very boulware, or not conceding, we set last time step to 1, or simply the predicted max utility
        # This is because the last_time_step may be too "low", especially when n_steps is small. Then it will predict a very low utility.
        # Let me know if this is confusing
        if self.e_value > 10:
            last_time_step = 1
        # last_time_step = 1
        predicted_utility = self.utility_model(
            last_time_step, self.e_value, max_utility
        )
        # print(hero_maximum_utility * predicted_utility, self.best_offer_from_villain_and_utiility[1])

        # t_plot = np.linspace(0, 1, 100)
        # u_plot = self.utility_model(t_plot, self.e_value, max_utility)
        # plt.plot(t_plot, u_plot, 'b-', label=f'Fitted curve')
        # print("step", state.step, self.interpolated_reservation_value)
        # print("predicted utility", predicted_utility, "thingie returned", hero_maximum_utility * predicted_utility)

        # When curve fit is weak, the minimum proportion of maximum utility to offer
        # Idea is to set a high acceptance threshold, so we do not accept offers that are too low
        weak_curve_fit_factor = max(0.8, predicted_utility)

        # Number of unique offers from the villain to determine if we can use the curve fit
        # Not really if this is a good number... will need to fine tune this
        minimum_unique_offers = max(10, 0.75 * self.total_steps)

        # If at last couple of timesteps, use the predicted utility regardless of the number of unique offers
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            return normalized_hero_maximum_utility * predicted_utility

        # Otherwise, if the curve fit is weak, we will always set a high acceptance threshold, based on the weak curve fit factor
        # if len(self.unique_proposals_from_villain) < minimum_unique_offers:
        if len(self.utilities_and_time_from_villain) < minimum_unique_offers:
            acceptance_threshold = (
                weak_curve_fit_factor * normalized_hero_maximum_utility
            )
            # print("neg index", self.current_negotiation_index, "unique proposals", self.unique_proposals_from_villain, "acceptance_threshold", acceptance_threshold, "best utility", self.filtered_outcome_space[0][1])
            return acceptance_threshold

        return normalized_hero_maximum_utility * predicted_utility

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent using the acceptance threshold.

        There are 3 "regions" that we can be in based on time.

        In the aggressive region, based on stop aggressive timestep, we will offer bids that are above the acceptance threshold.
        After the aggressive region, we will offer bids that are at the acceptance threshold.
        On the last couple of steps (or last timestep to propose), we will offer bids that are between the acceptance threshold and the best offer from the villain.
        """
        # Discounting for updating of item count
        # This should be < 1, but I was just fine tuning things. Will test on lower values
        eta = 1.5

        # timestep to when to stop offering bid above point
        stop_aggressive_timestep = 0.9

        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")
        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        # if self.is_edge_agent:
        #     return self.filtered_outcome_space[0][0]

        # In the last step to propose, propose an offer in between the best offer from the villain (slightly higher than this) and the acceptance threshold
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            # If best offer is close to acceptance threshold, just return it. Without this line, often result in no agreements reached
            if (
                acceptance_threshold - 0.1
                <= normalized_best_offer_from_villain_and_utility[1]
            ):
                return normalized_best_offer_from_villain_and_utility[0]

            # Find offers in the range of acceptance threshold and best offer from villain
            fscore = 0
            for outcome, utility in self.normalized_filtered_outcome_space:
                # The minimum utility is interpolated between the best offer from the villain and the acceptance threshold
                if (
                    utility
                    > normalized_best_offer_from_villain_and_utility[1] * 0.5
                    + acceptance_threshold * 0.5
                    and utility <= acceptance_threshold
                ):
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    fscore = 0
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore

                # the outcome space is sorted, so if utility is less than best offer from villain, we can break
                elif utility < normalized_best_offer_from_villain_and_utility[1]:
                    break
            # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
            if len(outcome_space_above_threshold) == 0:
                return normalized_best_offer_from_villain_and_utility[0]

        # elif state.relative_time >= stop_aggressive_timestep:
        #     # Find offers in the range of acceptance threshold
        #     fscore = 0
        #     for outcome, utility in self.normalized_filtered_outcome_space:
        #         # Maybe we can expand this range? Idk how helpful it will be
        #         if utility >= acceptance_threshold - 0.01 and utility <= acceptance_threshold:
        #             outcome_space_above_threshold.append(outcome)
        #             # Calculate the fscore of the outcome
        #             fscore = 0
        #             for index, issue in enumerate(outcome):
        #                 indexed_issue = str(index) + str(issue)
        #                 fscore += self.discounted_item_counter[indexed_issue]

        #             if fscore == outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore[0].append(outcome)

        #             if fscore > outcomes_and_highest_fscore[1]:
        #                 outcomes_and_highest_fscore = [outcome], fscore
        #         else:
        #             break
        #     # if there are no outcomes here, give the outcome where acceptance threshold is ceiling of its utility
        #     if len(outcome_space_above_threshold) == 0:
        #         for outcome, utility in self.normalized_filtered_outcome_space:
        #             if utility < acceptance_threshold:
        #                 outcomes_and_highest_fscore[0].append(outcome)
        #                 break
        else:
            for outcome, utility in self.normalized_filtered_outcome_space:
                fscore = 0
                if utility >= acceptance_threshold:
                    outcome_space_above_threshold.append(outcome)
                    # Calculate the fscore of the outcome
                    for index, issue in enumerate(outcome):
                        indexed_issue = str(index) + str(issue)
                        fscore += self.discounted_item_counter[indexed_issue]

                    if fscore == outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore[0].append(outcome)

                    if fscore > outcomes_and_highest_fscore[1]:
                        outcomes_and_highest_fscore = [outcome], fscore
                else:
                    break

        # If we are last step to propose, or we are greater than aggressive timestep, return random outcome from highest fscore outcomes
        if (
            (self.last_to_propose and state.step >= self.total_steps - 1)
            or (not self.last_to_propose and state.step >= self.total_steps - 2)
            or state.relative_time >= stop_aggressive_timestep
        ):
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            # elif len(outcomes_and_highest_fscore[0]) == 1:
            #     return self.best_offer_from_villain_and_utiility[0]
            else:
                # print(len(outcomes_and_highest_fscore[0]), outcomes_and_highest_fscore[1])
                random_choice = choice(outcomes_and_highest_fscore[0])

                # print("next bid", random_choice, "utility", self.normalized_filtered_outcome_space_dict[random_choice], "acceptance threshold", acceptance_threshold)
                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # We return some bid above the acceptance threshold, but we rotate through the bids to ensure we do not propose the same bid over and over again
                # Make it look like we are doing some kind of conceding
                next_bid = outcome_space_above_threshold[
                    -1 - self.weak_curve_fit_index % len(outcome_space_above_threshold)
                ]
                self.weak_curve_fit_index += 1
                return next_bid

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


class AbinesLookAheadAgent(ANL2025Negotiator):
    def init(self):
        self.debug = True

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Intialize variables used at every subnegotiation
        self.is_edge_agent = is_edge_agent(self)
        self.last_to_propose: bool = None  # Are we last to act?
        self.current_negotiation_index = None
        self.current_ufun = None  # utiilty function of the current subnegotiation
        self.nmi = None
        self.total_steps = None
        self.current_utility = None  # utility of all agreements from before
        self.reservation_value = None
        self.filtered_outcome_space = None  # outcome space of the current subnegotiation above current utility | SORTED!
        self.filtered_outcome_space_dict = (
            None  # dictionary of above, outcome -> utility
        )
        self.best_offer_from_villain_and_utiility = (
            None,
            None,
        )  # best offer from the villain and utility
        self.non_exploitation_point = None

        self.item_counter = (
            None  # Counts the indexed items that have been offered by the villain
        )
        self.discounted_item_counter = None  # Discounts the count from above

        # Determine if Lookahead is needed depending on utility function
        not_required_ufuns = [
            MaxCenterUFun,
            LinearCombinationCenterUFun,
            MeanSMCenterUFun,
        ]
        self.lookahead_required = True
        if type(self.ufun) in not_required_ufuns:
            self.lookahead_required = False

        if not is_edge_agent(self):
            # Make a list that maps index for a subnegotiation to sampled outcomes from the utility function.
            # This is to avoid calling the enumerate_or_sample function every time we need to get the outcomes.
            self.outcome_space_at_subnegotiation = []
            for i in range(len(self.negotiators)):
                suboutcomes = get_outcome_space_from_index(self, i)
                self.outcome_space_at_subnegotiation.append(suboutcomes)

            # Create the game environment
            self.environment = GameEnvironment(
                center_ufun=self.ufun,
                n_edges=len(self.negotiators),
                outcome_space_at_subnegotiation=self.outcome_space_at_subnegotiation,
            )

            # Max calls to ufun
            self.max_num_calls = 30_000_000

            #  Calculate depth for lookahead
            self.depth = self.get_maximum_depth_limit()

            # Dictionary to store expected values of outcomes for lookahead
            self.value_dictionary = None

    def get_current_utility(self) -> float:
        if is_edge_agent(self):
            return self.ufun.reserved_value

        current_agreements = self.get_current_agreements()
        current_outcome = [None for _ in range(get_number_of_subnegotiations(self))]
        any_agreement = False
        for index, agreement in enumerate(current_agreements):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True

        if self.use_lookahead():
            utility = self.value_dictionary[self.get_current_gamestate()]
            non_child = self.get_current_gamestate().get_child_from_action(None)
            utility = self.value_dictionary[non_child]
        else:
            utility = self.ufun.eval(current_outcome)

        if current_agreements:
            if any_agreement:
                return utility

        return utility if self.use_lookahead() else self.ufun.reserved_value

    def use_lookahead(self) -> bool:
        return not is_edge_agent(self) and self.lookahead_required

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).

        Remarks:
            - You can use the negotiator_id to identify what side negotiator is currently proposing. This id is stable within a negotiation.
        """
        # First time in subnegotiation
        if (
            state.step == 0 and state.current_offer is None
        ):  # If you respond first, propose step is still 0 so an extra check
            # We are first to act, and also last to propose

            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            self.current_utility = self.get_current_utility()

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize non exploitation point
            self.non_exploitation_point = None

            self.old_acceptance_threshold = None

        # print("proposing at step", state.step, "negotiator id", negotiator_id)
        # If last step of negotiation, return best offer proposed by the villain
        # TODO, if we are last to respond, this is not reached. also need to check
        if (self.last_to_propose and state.step >= self.total_steps - 1) or (
            not self.last_to_propose and state.step >= self.total_steps - 2
        ):
            return self.best_offer_from_villain_and_utiility[0]

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=False
        )

        if not next_bid:
            # This means there are no outcomes above the acceptance threshold
            # Just return the best outcome for the hero
            if len(self.filtered_outcome_space) == 0:
                # Terminate if no outcomes available

                return None
            return self.filtered_outcome_space[0][0]

        if (
            self.best_offer_from_villain_and_utiility[1] > acceptance_threshold
            or self.best_offer_from_villain_and_utiility[1]
            > self.filtered_outcome_space_dict[next_bid]
        ):
            return self.best_offer_from_villain_and_utiility[0]

        return next_bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # First time in subnegotiation
        nmi = self.negotiators[negotiator_id][0].nmi

        has_made_offer = any(neg_id == negotiator_id for neg_id, _ in nmi.trace)

        if not has_made_offer:  # We are first to act, and also last to propose
            # Perform lookahead if needed
            if self.use_lookahead():
                if state.step == 0:
                    self.value_dictionary = {}
                    self.lookahead(
                        self.get_current_gamestate(),
                        0.9,
                        depth_limit=get_current_negotiation_index(self) + self.depth,
                    )

            self.last_to_propose = True
            self.current_negotiation_index = get_current_negotiation_index(self)
            self.current_ufun = (
                self.ufun
                if self.is_edge_agent
                else self.ufun.side_ufuns()[self.current_negotiation_index]
            )
            self.nmi = self.negotiators[negotiator_id][0].nmi
            self.total_steps = self.nmi.n_steps

            self.current_utility = self.get_current_utility()

            self.reservation_value = (
                self.current_ufun.reserved_value
                if self.is_edge_agent
                else self.ufun.reserved_value
            )

            # Store the current utilities of every outcome that's greater than your current utility or reservation value
            current_outcome_space = (
                self.current_ufun.outcome_space.enumerate_or_sample()
            )

            # Find minimum and maximum utility to normalize the utilities
            # Normalization is done so that offers can be compared across different subnegotiations with different utility ranges
            self.minimum_outcome_utility = float("inf")
            self.maximum_outcome_utility = float("-inf")
            self.filtered_outcome_space = []
            for outcome in current_outcome_space:
                if self.use_lookahead():
                    utility_of_outcome = self.value_dictionary[
                        self.get_current_gamestate().get_child_from_action(outcome)
                    ]
                else:
                    utility_of_outcome = self.current_ufun(outcome)

                if utility_of_outcome < self.minimum_outcome_utility:
                    self.minimum_outcome_utility = utility_of_outcome
                if utility_of_outcome > self.maximum_outcome_utility:
                    self.maximum_outcome_utility = utility_of_outcome

                if (
                    utility_of_outcome > self.current_utility
                    and utility_of_outcome > self.reservation_value
                ):
                    self.filtered_outcome_space.append((outcome, utility_of_outcome))

            self.filtered_outcome_space.sort(key=lambda x: x[1], reverse=True)
            self.filtered_outcome_space_dict = dict(self.filtered_outcome_space)

            # Normalize the filtered outcome space
            self.normalized_filtered_outcome_space = []
            for outcome, utility in self.filtered_outcome_space:
                normalized_utility = self.normalize_utility(utility)
                self.normalized_filtered_outcome_space.append(
                    (outcome, normalized_utility)
                )
            self.normalized_filtered_outcome_space_dict = dict(
                self.normalized_filtered_outcome_space
            )

            # print("subnegotiation", self.current_negotiation_index, "maximum utility", self.filtered_outcome_space[0][1], "current utility", self.current_utility, "reservation value", self.reservation_value)

            # Initialize the counter of items in issues for next bid
            self.item_counter = defaultdict(int)
            self.discounted_item_counter = defaultdict(float)

            # Initialize the best offer from the villain and utility
            self.best_offer_from_villain_and_utiility = None, float("-inf")

            # Initialize non exploitation point
            self.non_exploitation_point = None

            self.old_acceptance_threshold = None

        if len(self.filtered_outcome_space) == 0:
            return ResponseType.END_NEGOTIATION

        acceptance_threshold = self.acceptance_threshold(negotiator_id, state)
        # print("step", state.step, "acceptance_threshold", acceptance_threshold)
        next_bid = self.get_next_bid(
            negotiator_id, state, acceptance_threshold, responding=True
        )

        if not next_bid:
            return ResponseType.REJECT_OFFER

        current_offer = state.current_offer
        # If last step of negotiation, return best offer proposed by the villain
        # TODO: we are never last to respond, so this is never reached. Need to check if we are last or not
        if state.step == self.total_steps:
            if self.current_ufun.eval(current_offer) > self.reservation_value:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        if current_offer not in self.filtered_outcome_space_dict:
            return ResponseType.REJECT_OFFER

        current_offer_utility = self.filtered_outcome_space_dict[current_offer]
        if current_offer_utility > self.best_offer_from_villain_and_utiility[1]:
            self.best_offer_from_villain_and_utiility = (
                current_offer,
                current_offer_utility,
            )

        # Check if the current offer is acceptable
        if current_offer_utility > acceptance_threshold or (
            current_offer_utility > self.filtered_outcome_space_dict[next_bid]
            if next_bid
            else False
        ):
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def normalize_utility(self, utility: float) -> float:
        """
        Normalizes the utility to be between 0 and 1.
        """
        if self.maximum_outcome_utility > self.minimum_outcome_utility:
            return (utility - self.minimum_outcome_utility) / (
                self.maximum_outcome_utility - self.minimum_outcome_utility
            )
        else:
            return 1

    def acceptance_threshold(self, negotiator_id: str, state: SAOState) -> float:
        """Returns the acceptance threshold of the agent. This is the minimum utility that the agent is willing to accept."""

        discount = 0.8
        # Calculate non exploitation point

        # determines the way the value of the non exploitation point changes w/ respect to the discount factor
        # This is only used in the initialization of the non exploitation point

        # Closer to 0, this will set nonexploitation point to 1. Have to set this pretty large to get the non exploitation point to be less than 1
        beta = 5

        # deterimines the way the value of the non exploitation point changes w/ respect to concessive degree
        # < 1 to make the point more delayed or greater,
        gamma = 5

        # weighing factor to ajudst concenssion degree
        # Value between 0 and 1
        weighting_factor = 0.1

        # Calculate the concessive degree
        # This is the ratio of new outcomes proposed to the total number of outcomes
        # This will be a value between 0 and 1
        proposed_outcomes = set()
        num_outcomes_proposed = 0
        for neg_id, outcome in self.nmi.trace:
            if neg_id != negotiator_id:
                proposed_outcomes.add(outcome)
                num_outcomes_proposed += 1
        concession_degree = (
            len(proposed_outcomes) / num_outcomes_proposed
            if num_outcomes_proposed > 0
            else None
        )

        # Calculate the non exploitation point
        if concession_degree is not None:
            self.non_exploitation_point = (
                self.non_exploitation_point
                + weighting_factor
                * (1 - self.non_exploitation_point)
                * concession_degree**gamma
            )
        else:
            # set minimum exploitation point
            minimum_exploitation_point = 0.1
            self.non_exploitation_point = (
                minimum_exploitation_point
                + (1 - minimum_exploitation_point) * discount**beta
            )

        acceptance_threshold = float("inf")

        # Find the maximum utility of the current outcome space
        if len(self.filtered_outcome_space) != 0:
            # determines the way the acceptance threshold approaches the minimum (max_utility * discount **(1-t))
            normalized_hero_maximum_utility = self.normalize_utility(
                self.filtered_outcome_space[0][1]
            )

            # Set alpha = 1 to linear, < 1 to concede, > 1 to not
            alpha = 10

            relative_time = state.relative_time

            # Calculate the acceptance threshold
            if relative_time < self.non_exploitation_point:
                acceptance_threshold = (
                    normalized_hero_maximum_utility
                    - (
                        normalized_hero_maximum_utility
                        - normalized_hero_maximum_utility
                        * discount ** (1 - self.non_exploitation_point)
                    )
                    * (relative_time / self.non_exploitation_point) ** alpha
                )
            else:
                acceptance_threshold = discount ** (1 - relative_time)
                # if self.old_acceptance_threshold is None:
                #     self.old_acceptance_threshold = normalized_hero_maximum_utility * discount**(1 - relative_time)
                #     acceptance_threshold = self.old_acceptance_threshold
                # else:
                #     acceptance_threshold = self.old_acceptance_threshold

        return acceptance_threshold

    def get_next_bid(
        self,
        negotiator_id: str,
        state: SAOState,
        acceptance_threshold: float,
        responding: bool,
    ) -> Outcome | None:
        """
        Returns the next bid of the agent.
        """

        # Discounting for updating of item count
        eta = 1.5

        # Set the minimum greedy rate
        minimum_greedy = 1
        maximum_greedy = 1

        # set the rate of concession for exploring
        iota = 0.5
        # Update the counter of items in issues if responding
        if responding:
            current_offer = state.current_offer
            for index, issue in enumerate(current_offer):
                indexed_issue = str(index) + str(issue)
                self.discounted_item_counter[indexed_issue] = (
                    self.discounted_item_counter[indexed_issue]
                    + eta ** self.item_counter[indexed_issue]
                )
                self.item_counter[indexed_issue] += 1

        outcome_space_above_threshold = []
        outcomes_and_highest_fscore = [], float("-inf")

        normalized_best_offer_from_villain_and_utility = (
            self.best_offer_from_villain_and_utiility[0],
            self.normalize_utility(self.best_offer_from_villain_and_utiility[1]),
        )

        for outcome, utility in self.normalized_filtered_outcome_space:
            fscore = 0
            if utility >= acceptance_threshold:
                outcome_space_above_threshold.append(outcome)
                # Calculate the fscore of the outcome
                for index, issue in enumerate(outcome):
                    indexed_issue = str(index) + str(issue)
                    fscore += self.discounted_item_counter[indexed_issue]

                if fscore == outcomes_and_highest_fscore[1]:
                    outcomes_and_highest_fscore[0].append(outcome)

                if fscore > outcomes_and_highest_fscore[1]:
                    outcomes_and_highest_fscore = [outcome], fscore
            else:
                break
        greedy_rate = minimum_greedy + (
            maximum_greedy - state.relative_time ** (1 / iota)
        ) * (maximum_greedy - minimum_greedy)

        r = random()
        if r < greedy_rate:
            if len(outcomes_and_highest_fscore[0]) == 0:
                return None
            else:
                random_choice = choice(outcomes_and_highest_fscore[0])
                return random_choice
        else:
            if len(outcome_space_above_threshold) == 0:
                return None
            else:
                # Get a random outcome from the outcome space above the acceptance threshold
                random_choice = choice(outcome_space_above_threshold)
                return random_choice

    def get_current_gamestate(self) -> AbstractedOutcomeGameState:
        # Obtain all histories/states
        current_agreements = self.get_current_agreements()
        if current_agreements:
            return AbstractedOutcomeGameState(current_agreements, self.environment)
        else:
            return AbstractedOutcomeGameState(tuple(), self.environment)

    def get_current_agreements(self) -> tuple[Outcome]:
        agreements = []
        for index in range(len(self.finished_negotiators)):
            agreements.append(get_agreement_at_index(self, index))
        return tuple(agreements)

    def lookahead(
        self, state: AbstractedOutcomeGameState, discount: float, depth_limit: int
    ) -> float:
        if state.is_terminal():
            self.value_dictionary[state] = state.get_current_utility()
            return self.value_dictionary[state]

        if state in self.value_dictionary:
            return self.value_dictionary[state]

        current_depth = state.get_current_negotiation_index()
        if current_depth == depth_limit:
            self.value_dictionary[state] = self.estimate_with_current_utility(state)
            return self.value_dictionary[state]

        current_utility = state.get_current_utility()
        children = state.get_children()
        current_index = state.get_current_negotiation_index()
        nsteps = self.get_nsteps_for_negotation_index(current_index)

        # Obtain expected values of all children
        children_expected_values = np.array(
            [self.lookahead(child, discount, depth_limit) for child in children]
        )

        # Get the probability distribution on outcomes
        distribution_on_outcomes = self.get_probability_distribution_on_outcomes(
            current_parent_utility=current_utility,
            children_expected_values=children_expected_values,
            temperature=1,
            nsteps=nsteps,
        )

        # Compute the dot product of children_expected_values and distribution_on_outcomes
        expected_future_utility = np.dot(
            children_expected_values, distribution_on_outcomes
        )

        value = current_utility + discount * (expected_future_utility - current_utility)

        self.value_dictionary[state] = value
        return value

    def get_probability_distribution_on_outcomes(
        self,
        current_parent_utility: float,
        children_expected_values: NDArray[np.float64],
        temperature: float = 1.0,
        nsteps: int = 1,
    ) -> NDArray[np.float64]:
        """
        Returns a probability distribution over the number of children.
        The probability of each child is softmaxed by its expected value, scaled by a temperature parameter.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        # Scale temperature by number of nsteps
        temperature = temperature / np.log(nsteps + 1)

        # Get the indices of children that are better than the current parent utility
        # Always include None as a possible outcome
        better_than_parent = children_expected_values > current_parent_utility
        include_none = (
            np.arange(len(children_expected_values))
            == len(children_expected_values) - 1
        )
        worthy_indices = np.logical_or(better_than_parent, include_none)

        worthy_children = children_expected_values[worthy_indices]

        # Means only None agreement is worthy
        if len(worthy_children) == 1:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[-1] = 1.0  # Assign probability 1 to None
            return probabilities

        # Apply softmax to positive children
        scaled_values = worthy_children / temperature
        exp_values = np.exp(
            scaled_values - np.max(scaled_values)
        )  # Subtract max for numerical stability
        total_exp = np.sum(exp_values)
        if total_exp == 0:
            # I don't think this is possible
            raise ValueError(
                "Total expected value is zero, cannot compute probabilities."
            )
        else:
            probabilities = np.zeros_like(children_expected_values)
            probabilities[worthy_indices] = exp_values / total_exp

        return probabilities

    def get_nsteps_for_negotation_index(self, index: int) -> int:
        """
        Returns the number of steps for the given negotiation index.
        """
        nsteps_list = [
            get_nmi_from_index(self, i).n_steps for i in range(len(self.negotiators))
        ]
        return nsteps_list[index]

    def estimate_with_current_utility(self, state: AbstractedOutcomeGameState) -> float:
        return state.get_current_utility()

    def get_maximum_depth_limit(self) -> int:
        """
        Returns the maximum depth limit for lookahead
        """
        current_depth_limit = 1
        total_subnegotiations = get_number_of_subnegotiations(self)

        while True:
            num_calls = 0
            times_lookahead_calls = total_subnegotiations - current_depth_limit

            # No need to use lookahead if outcome space is small
            if times_lookahead_calls < 0:
                break

            for i in range(times_lookahead_calls):
                calls_in_lookahead = 1

                for neg_index in range(i, i + current_depth_limit):
                    calls_in_lookahead *= len(
                        self.outcome_space_at_subnegotiation[neg_index]
                    )

                num_calls += calls_in_lookahead
            calls_in_normal_search = 1

            for i in range(times_lookahead_calls, total_subnegotiations):
                calls_in_normal_search *= len(self.outcome_space_at_subnegotiation[i])
            num_calls += calls_in_normal_search
            if num_calls <= self.max_num_calls:
                current_depth_limit += 1
            else:
                break

        # Subtract 1 because last increase was not valid
        return current_depth_limit - 1


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament
    # Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    # Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(NewNegotiator, small=True)
