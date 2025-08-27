"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from collections import Counter
import itertools
from negmas.outcomes import Outcome


# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
# from helpers.helperfunctions import set_id_dict, ...

import random
from anl2025.negotiator import ANL2025Negotiator
from anl2025.ufun import MaxCenterUFun, LinearCombinationCenterUFun
from negmas.sao.controllers import SAOState
from negmas import SAONMI
from negmas import (
    ResponseType,
)


# CONSTANTS
ASPIRATION_TYPE_BOULWARE = 0.125

from negmas import Aspiration

__all__ = ["StarGold15"]


class AdjustableConvexAspiration(Aspiration):
    def __init__(self, min_y_value=0.5, min_point_x_value=0.5, y_start_point=1):
        super().__init__()
        self.m = min_y_value
        self.min_point_at = min_point_x_value
        self.original_min_y = self.m
        self.y_start_point = y_start_point

    def utility_at(self, t: float) -> float:
        a = (self.y_start_point - self.m) / (self.min_point_at**2)
        return a * (t - self.min_point_at) ** 2 + self.m


#### all functions below are hulp functions for the functions above. They are not necessary to implement the agent, but they can be useful." ####


def set_id_dict(self):
    """Creates a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
    This dictionary allows us to find the right id for easy access to further information about the specific negotiation."""
    for id in self.negotiators.keys():
        index = self.negotiators[id].context["index"]
        self.id_dict[index] = id


def did_negotiation_end(self):
    """The function finished_negotiators checks how many threads are finished. If that number changes, then the next negotiation has started."""
    if self.current_neg_index != len(self.finished_negotiators):
        self.current_neg_index = len(self.finished_negotiators)
        return True
    return False


def get_current_negotiation_index(self):
    """Returns the index of the current negotiation. An index 0 means the first negotiation in the sequence, index 1 the second, etc."""
    return len(self.finished_negotiators)


def get_agreement_at_index(self, index):
    """The nmi is the negotiator mechanism interface, available for each subnegotiation.
    Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids."""
    nmi = get_nmi_from_index(self, index)
    agreement = nmi.state.agreement
    # print(agreement)
    return agreement


def get_outcome_space_from_index(self, index):
    """This function returns the outcome space of the subnegotiation with the given index."""
    outcomes = self.ufun.outcome_spaces[index].enumerate_or_sample()
    #  print(outcomes)

    # Each subnegotiation can also end in disagreement aka outcome None. Therefore, we add this to all outcomes.
    outcomes.append(None)

    return outcomes


def get_number_of_subnegotiations(self):
    """Returns the total number of (sub)negotiations that the agent is involved in. For edge agents, this is 1."""
    return len(self.negotiators)


def cartesian_product(arrays):
    """This function returns the cartesian product of the given arrays."""
    cartesian_product = list(itertools.product(*arrays))
    #  print(cartesian_product)
    return cartesian_product


def is_edge_agent(self):
    """Returns True if the agent is an edge agent, False otherwise, then the agent is a center agent."""
    if self.id.__contains__("edge"):
        return True
    return False


def all_possible_bids_with_agreements_fixed(self):
    """This function returns all the bids that are still possible to achieve, given the agreements that were made in the previous negotiations."""

    # Once a negotiation has ended, the bids in the previous negotiations cannot be changed.
    # Therefore, this function helps to construct all the bids that can still be achieved, fixing the agreements of the previous negotiations.

    # If the agent is an edge agent, there is just one bid to be made, so we can just return the outcome space of the utility function.
    # Watch out, the structure of the outcomes for an edge agent is different from for a center agent.

    if is_edge_agent(self):
        return self.ufun.outcome_space.enumerate_or_sample()

    possible_outcomes = []
    neg_index = get_current_negotiation_index(self)

    # As the previous agreements are fixed, these are added first.
    for i in range(neg_index):
        possible_outcomes.append(get_agreement_at_index(self, i))

    temp_all_possible_outcomes = []
    temp_ufun_all_possible_outcomes = []
    for outcome, ufun_value in zip(
        self.all_possible_outcomes, self.ufun_all_possible_outcomes
    ):
        if all(
            bid == outcome[bid_index] for bid_index, bid in enumerate(possible_outcomes)
        ):
            temp_all_possible_outcomes.append(outcome)
            temp_ufun_all_possible_outcomes.append(ufun_value)

    self.all_possible_outcomes = temp_all_possible_outcomes
    self.ufun_all_possible_outcomes = temp_ufun_all_possible_outcomes


def get_target_bid_at_current_index(self):
    """Returns the bid for the current subnegotiation, with the target_bid as source."""
    index = get_current_negotiation_index(self)
    # An edge agents bid is only one bid, not a tuple of bids. Therefore, we can just return the target bid.
    if is_edge_agent(self):
        return self.target_bid

    return self.target_bid[index]


def get_nmi_from_index(self, index):
    """
    This function returns the nmi of the subnegotiation with the given index.
    The nmi is the negotiator mechanism interface per subnegotiation. Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids.
    """
    negotiator_id = get_negid_from_index(self, index)
    return self.negotiators[negotiator_id].negotiator.nmi


def get_negid_from_index(self, index):
    """This function returns the negotiator id of the subnegotiation with the given index."""
    return self.id_dict[index]


"""
This agents compromises more and properly utilizes nash equilibrium near the end of a sub-negotiation,
for propose, the agent goes uses a last-resort mechanism to propose the best bid that the rival proposed.

for calcuating utility of lambda utility functions, I use a low bound of the value of propose_cruve divided by 2,
this is in order to not count extremely low utility bids as valid bids.
"""


DEBUG = False
PROPOSABLE_BIDS_MIN_SIZE = 0.7


class StarGold15(ANL2025Negotiator):
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

        # I replaced the Aspiration used by the timeBased negotiator with a exponential one
        # so that we could set the curve of the level calculation (which i use to set the concession level) to concede slow
        # at the beginning and fast at the end.
        self.propose_bids_by_utility = {}
        self.nash_bids = {}

        if not is_edge_agent(self):
            self.max_reduce_bottom_utility_per_neg = 0.4
            self.respond_curve = AdjustableConvexAspiration(0.6, 0.6)
            self.propose_curve = AdjustableConvexAspiration(0.5, 0.8)
            self.all_possible_outcomes = []
            for i in range(0, get_number_of_subnegotiations(self)):
                self.all_possible_outcomes.append(get_outcome_space_from_index(self, 0))

            self.all_possible_outcomes = cartesian_product(self.all_possible_outcomes)
            # the line below is for initializing the ufun weights
            self.ufun(self.all_possible_outcomes[0])

            self.all_possible_outcomes.sort(key=lambda x: self.ufun(x), reverse=True)
            self.all_possible_outcomes = [
                outcome
                for outcome in self.all_possible_outcomes
                if self.ufun(outcome) > 0.0
            ]
            self.ufun_all_possible_outcomes = [
                self.ufun(outcome) for outcome in self.all_possible_outcomes
            ]
            # we will keep track of all the bids that the opponent proposed to us in order to fallback to the nash equilibrium (basically assumming that the rival is intellegent)
            self.offers_received = {}

            if DEBUG:
                pass  # print(f"ufun type is max: {isinstance(self.ufun, MaxCenterUFun)}")
                pass  # print(f"ufun type is linear: {isinstance(self.ufun, LinearCombinationCenterUFun)}")
                if isinstance(self.ufun, LinearCombinationCenterUFun):
                    pass  # print(f"ufun weights: {self.ufun._weights}")

        else:
            # edge agent setup
            self.respond_curve = AdjustableConvexAspiration(0.5, 1)
            self.propose_curve = AdjustableConvexAspiration(0.5, 1)

        self.do_none = False

    def propose_last_resort(self, negotiator_id: str, state: SAOState):
        negotiator_id = get_negid_from_index(self, self.current_neg_index)
        _, cntxt = self.negotiators[negotiator_id]
        ufun = cntxt["ufun"]
        best_counter_offer = None
        best_counter_offer_value = 0.0
        for offer in self.offers_received[negotiator_id]:
            if (
                ufun(offer) > ufun.max() * 0.3
                and ufun(offer) > best_counter_offer_value
            ):
                best_counter_offer = offer
                best_counter_offer_value = ufun(offer)

        if best_counter_offer is not None:
            return best_counter_offer

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).

        Remarks:
            - You can use the negotiator_id to identify what side negotiator is currently proposing. This id is stable within a negotiation.
        """
        # If the negotiation has ended, update the strategy. The subnegotiator may of may not have found an agreement: this affects the strategy for the rest of the negotiation.

        if did_negotiation_end(self):
            self.last_neg_index = self.current_neg_index
            self._update_strategy(state)

        if not is_edge_agent(self):
            current_bid = self.propose_center(negotiator_id, state, dest)
            # if the current bid is None, it means that we gain nothing from "helping" our rival to get better
            nmi: SAONMI = self.negotiators[negotiator_id].negotiator.nmi
            if current_bid != None and state.relative_time >= 0.98:
                if self.propose_last_resort(negotiator_id, state) is not None:
                    current_bid = self.propose_last_resort(negotiator_id, state)
            return current_bid

        else:
            return self.propose_edge(negotiator_id, state, dest)

    def propose_edge(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ):
        if self.proposeable_bids == [None]:
            return None

        nmi: SAONMI = self.negotiators[negotiator_id].negotiator.nmi

        amount_of_bids = len(self.proposeable_bids)
        bid_index = (nmi.n_steps // amount_of_bids) * state.step

        base_steps = nmi.n_steps // amount_of_bids
        remainder = nmi.n_steps % amount_of_bids

        # Compute the step ranges for each value
        boundaries = []
        current = 0
        for i in range(amount_of_bids):
            count = base_steps + (1 if i < remainder else 0)
            boundaries.append((current, current + count))
            current += count

        # Find which range the step falls into
        for i, (start, end) in enumerate(boundaries):
            if start <= state.step < end:
                try:
                    return (
                        self.proposeable_bids[i]
                        if len(self.proposeable_bids[i]) == 1
                        else self.proposeable_bids[i]
                    )
                except:
                    self.update_strategy_center()
                    return (
                        self.proposeable_bids[i]
                        if len(self.proposeable_bids[i]) == 1
                        else self.proposeable_bids[i]
                    )

    def propose_center(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ):
        level = self.propose_curve.utility_at(state.relative_time)
        if self.propose_bids_by_utility == {}:
            self.update_strategy_center()

        # check if we should just do nothing
        if self.do_none:
            return None

        return self.propose_edge(negotiator_id, state, dest)

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
            if not is_edge_agent(self):
                self.last_neg_index = self.current_neg_index
                self._update_strategy(state)

            elif is_edge_agent(self):
                # if the negotiation has ended, we need to update the strategy for the edge agent
                self.update_strategy_edge(state)
                self.last_neg_index = self.current_neg_index

        if not is_edge_agent(self):
            final_response = self.respond_center(negotiator_id, state, source)
            # if the current response is None, it means that we gain nothing from "helping" our rival to get better
            nmi: SAONMI = self.negotiators[negotiator_id].negotiator.nmi
            if (
                final_response != None
                and final_response == ResponseType.REJECT_OFFER
                and state.relative_time >= 0.98
            ):
                # if its a center agent, we have a super fallback for the last step
                self.estimate_opponent_ufun(negotiator_id)
                self.compute_nash_point_center(negotiator_id)
                if state.current_offer == self.nash_bids[negotiator_id]:
                    # if its the last step, we save as a final resort the nash equilibrium:
                    return ResponseType.ACCEPT_OFFER

            return final_response

        else:
            return self.respond_edge(negotiator_id, state, source)

    def respond_center(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ):
        if self.propose_bids_by_utility == {}:
            self.update_strategy_center()

        if self.do_none:
            return ResponseType.END_NEGOTIATION

        level = self.respond_curve.utility_at(state.relative_time)

        # collecting the rival's offers
        if negotiator_id not in self.offers_received:
            self.offers_received[negotiator_id] = []

        self.offers_received[negotiator_id].append(state.current_offer)

        # now we have the best outcome, we need to know how much utility we could individually get from each option there.
        best_bid_utility = max(self.propose_bids_by_utility.values())
        # we accept only bids that are the best bid or are close to it (the closeness decends as time passes using the `level`)
        acceptable_bids = [
            bid
            for bid in self.propose_bids_by_utility.keys()
            if self.propose_bids_by_utility[bid] >= best_bid_utility * level
        ]

        neg_index = get_current_negotiation_index(self)
        should_prefer_none_vs_current_offer = False

        for index, outcome in enumerate(self.all_possible_outcomes):
            if outcome[neg_index:].count(None) == len(outcome[neg_index:]):
                should_prefer_none_vs_current_offer = True
                break
            if state.current_offer == outcome[neg_index]:
                should_prefer_none_vs_current_offer = False
                break

        if (
            state.current_offer in acceptable_bids
            and not should_prefer_none_vs_current_offer
        ):
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def respond_edge(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ):
        level_reduction = 0

        try:
            self.max_reduce_bottom_utility_per_neg = 0.2
            a = self.crisp_ufun.n_edges - state.n_acceptances
            level_reduction = self.remaining_scaled_flipped_edge()
        except:
            pass
        level = self.respond_curve.utility_at(state.relative_time)
        self.negotiators[negotiator_id].negotiator

        all_possible_bids = all_possible_bids_with_agreements_fixed(self)
        all_possible_bids.sort(key=lambda x: self.ufun(x), reverse=True)
        best_bid_utility = self.ufun(all_possible_bids[0])
        self.negotiators[negotiator_id].negotiator
        ordered_bids = sorted(
            all_possible_bids, key=lambda x: self.ufun(x), reverse=True
        )
        # we accept only bids that are the best bid or are close to it (the closeness decends as time passes using the `level`)
        acceptable_bids = [
            bid
            for bid in ordered_bids
            if self.ufun(bid) >= best_bid_utility * (level + level_reduction)
        ]

        if state.current_offer in acceptable_bids:
            return ResponseType.ACCEPT_OFFER

        else:
            return ResponseType.REJECT_OFFER

    def estimate_opponent_ufun(self, negotiator_id: str):
        """
        Builds a naive opponent utility estimator using frequency analysis.
        """
        offers = self.offers_received[negotiator_id]
        _, context = self.negotiators[negotiator_id]
        possible_bids = context["ufun"].outcome_space.enumerate_or_sample()

        if not offers:
            self.opponent_scores = {
                negotiator_id: {possible_bid: 1.0 for possible_bid in possible_bids}
            }
            return

        freq = Counter(offers)
        max_freq = max(freq.values())
        self.opponent_scores = {}
        self.opponent_scores[negotiator_id] = {}

        for bid in possible_bids:
            self.opponent_scores[negotiator_id][bid] = (
                freq[bid] / max_freq if max_freq > 0 else 0.1
            )

    def compute_nash_point_center(self, negotiator_id: str):
        """
        Finds the Nash outcome maximizing self_utility * opponent_estimate
        """

        best_bid = None
        best_product = -1
        minimal_utility_ratio = 0.5
        if isinstance(self.ufun, LinearCombinationCenterUFun):
            if self.ufun._weights[self.current_neg_index] == max(self.ufun._weights):
                minimal_utility_ratio = 0.2

        max_bid_utility = max(self.propose_bids_by_utility.values())
        for possible_bid, bid_utility in self.propose_bids_by_utility.items():
            u_opp = self.opponent_scores[negotiator_id].get(possible_bid, 0.01)
            # we will try to find the best bid that maximizes the product of our utility and the opponent's estimated utility
            # but a little bit better for us
            if bid_utility < minimal_utility_ratio * max_bid_utility:
                continue

            product = bid_utility * u_opp
            if product > best_product:
                best_product = product
                best_bid = possible_bid

        self.nash_bids[negotiator_id] = best_bid

    def ufun_that_checks_future(self, bid):
        """
        extended to also calculate recursively using probabilities of future outcomes.
        """

        # we need to know if its a max center ufun here
        if isinstance(self.ufun, MaxCenterUFun):
            negotiator_id = get_negid_from_index(self, self.current_neg_index)
            _, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]
            bid_value = ufun(bid)
            for i in range(self.current_neg_index):
                prev_negotiator_id = get_negid_from_index(self, self.current_neg_index)
                _, cntxt = self.negotiators[prev_negotiator_id]
                prev_ufun = cntxt["ufun"]
                if prev_ufun(get_agreement_at_index(self, i)) > bid_value:
                    return 0.0

            return bid_value

        if isinstance(self.ufun, LinearCombinationCenterUFun):
            negotiator_id = get_negid_from_index(self, self.current_neg_index)
            _, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]

            return ufun(bid)

        else:
            bid_accomulated_utility = 0.0
            amount_of_possible_outcomes = 0.0
            current_neg_index = get_current_negotiation_index(self)

            minimal_utility_ratio = self.propose_curve.m / 2
            total_outcomes_with_minimal_ufun = 0
            while total_outcomes_with_minimal_ufun == 0:
                # we want to make sure that we have at least one outcome with a minimal utility ratio
                total_outcomes_with_minimal_ufun = len(
                    [
                        1
                        for outcome in self.all_possible_outcomes
                        if self.ufun(outcome) >= minimal_utility_ratio * self.ufun.max()
                    ]
                )
                minimal_utility_ratio *= 0.9

            for outcome, ufun_value in zip(
                self.all_possible_outcomes, self.ufun_all_possible_outcomes
            ):
                # I check if the bid is a part of an outcome that is acheivable, also that this bid is not already in the outcome
                if bid == outcome[current_neg_index]:
                    if (
                        self.ufun(outcome)
                        >= minimal_utility_ratio * self.ufun_all_possible_outcomes[0]
                    ):
                        bid_accomulated_utility += self.ufun(outcome)
                        amount_of_possible_outcomes += 1

            if amount_of_possible_outcomes == 0:
                return 0.0

            # Normalize count by total possible outcomes (avoid divide by zero)
            normalized_count = (
                amount_of_possible_outcomes / total_outcomes_with_minimal_ufun
            )
            bid_accomulated_utility = (
                bid_accomulated_utility / amount_of_possible_outcomes
            )

            weight_utility = 0.6
            weight_count = 1.0 - weight_utility
            return (
                float(bid_accomulated_utility) * weight_utility
                + normalized_count * weight_count
            )

    def get_future_value(self, bid, n_samples: int = 200):
        """
        uses monte carlo sampling to estimate the utility of a bid in the future sub-negotiations.
        Approximate E[ufun] if we choose `bid` now, by sampling `n_samples`
        random completions of the future subnegotiations (uniformly).
        """

        # 1) Identify where we are
        idx = get_current_negotiation_index(self)
        total_subs = get_number_of_subnegotiations(self)

        # 2) Collect the fixed prefix of earlier agreements
        prefix = []
        for j in range(idx):
            prefix.append(get_agreement_at_index(self, j))

        # 3) If this is the last sub-neg, return direct ufun
        if idx == total_subs - 1:
            full = tuple(prefix + [bid])
            return float(self.ufun(full))

        # 4) Build outcome spaces for each future sub-neg
        future_spaces = [
            get_outcome_space_from_index(self, j) for j in range(idx + 1, total_subs)
        ]

        # 5) Monte Carlo sampling
        total_val = 0.0
        for _ in range(n_samples):
            # Randomly pick one outcome in each future sub-neg
            sample_tail = [random.choice(space) for space in future_spaces]
            full_outcome = tuple(prefix + [bid] + sample_tail)
            total_val += float(self.ufun(full_outcome))

        # 6) Return the average over all samples
        return total_val / n_samples

    def remaining_scaled_flipped(self) -> float:
        """
        0.0 at the first sub-negotiation
        0.2 when all sub-negotiations have been exhausted
        (linear in-between, independent of how many there are)
        """
        total = max(1, get_number_of_subnegotiations(self))  # avoid /0
        remaining = max(0, total - get_current_negotiation_index(self))

        # remaining/total ∈ [0,1]     ← 1 at start, 0 at the end
        # 1 - (…)      ∈ [0,1]        ← 0 at start, 1 at the end
        return self.max_reduce_bottom_utility_per_neg * (1 - remaining / total)

    def remaining_scaled_flipped_edge(self, state: SAOState) -> float:
        """
        0.0 at the first sub-negotiation
        0.2 when all sub-negotiations have been exhausted
        (linear in-between, independent of how many there are)
        """
        total = max(1, self.crisp_ufun.n_edges - 1)  # avoid /0
        remaining = max(0, total - state.n_acceptances - 1)

        # remaining/total ∈ [0,1]     ← 1 at start, 0 at the end
        # 1 - (…)      ∈ [0,1]        ← 0 at start, 1 at the end
        return self.max_reduce_bottom_utility_per_neg * (1 - remaining / total)

    def update_strategy_center(self) -> None:
        self.do_none = False
        all_possible_bids_with_agreements_fixed(self)

        if isinstance(self.ufun, LinearCombinationCenterUFun):
            self.max_reduce_bottom_utility_per_neg = 0.2
            # if its a linear combination we need to compromise between sub-negotiations in order to get a lot of accomulative value
            # for a weight between 0 and 1, we set the propose curve to be more aggressive, we want it to be more aggressive if the weight
            # is relatively high, and less aggressive if the weight is relatively low compared to the other weights.
            weight = self.ufun._weights[self.current_neg_index]
            self.propose_curve = AdjustableConvexAspiration(
                0.45 + 0.5 * weight - self.remaining_scaled_flipped(), 0.95
            )
            self.respond_curve = AdjustableConvexAspiration(
                0.5 + 0.5 * weight - self.remaining_scaled_flipped(),
                0.95,
                y_start_point=1.0 - weight * 0.3,
            )

        elif isinstance(self.ufun, MaxCenterUFun):
            self.max_reduce_bottom_utility_per_neg = 0.4
            # if its a max center ufun, we want to be more aggressive in the beginning and less aggressive in the end
            # since we only need ONE good bid for a good score
            self.propose_curve = AdjustableConvexAspiration(
                1.0 - self.remaining_scaled_flipped(), 0.95
            )
            self.respond_curve = AdjustableConvexAspiration(
                1.0 - self.remaining_scaled_flipped(), 0.95
            )

        else:
            self.propose_curve.m = (
                self.propose_curve.original_min_y - self.remaining_scaled_flipped()
            )
            self.respond_curve.m = (
                self.respond_curve.original_min_y - self.remaining_scaled_flipped()
            )

        self.propose_bids_by_utility = {}

        for outcome in self.all_possible_outcomes:
            # we look only on bids in the current negotiation index and forward, because those are the only bids that we can possibly get now.
            for bid in outcome[self.current_neg_index :]:
                if bid not in self.propose_bids_by_utility and bid is not None:
                    self.propose_bids_by_utility[bid] = 0

        # TODO Change here for changing the bids that we propose
        for bid in self.propose_bids_by_utility.keys():
            self.propose_bids_by_utility[bid] = self.ufun_that_checks_future(bid)

        self.propose_bids_by_utility = {
            bid: value
            for bid, value in self.propose_bids_by_utility.items()
            if value > 0.0
        }
        if self.propose_bids_by_utility == {}:
            self.do_none = True
            return

        best_bid_utility_value = max(self.propose_bids_by_utility.values())
        level = self.propose_curve.m

        self.proposeable_bids = [
            bid
            for bid in self.propose_bids_by_utility.keys()
            if self.propose_bids_by_utility[bid] >= best_bid_utility_value * level
        ]
        self.proposeable_bids.sort(
            key=lambda x: self.propose_bids_by_utility[x], reverse=True
        )

        # truncanting the proposeable bids to a minimum size if there are ennough bids, we shrink it to a smaller size of the the total available bids.
        if (
            len(self.proposeable_bids)
            > len(self.propose_bids_by_utility) * PROPOSABLE_BIDS_MIN_SIZE
        ):
            self.proposeable_bids = self.proposeable_bids[
                : round(
                    float(len(self.propose_bids_by_utility)) * PROPOSABLE_BIDS_MIN_SIZE
                )
            ]

        self.proposeable_bids = (
            self.proposeable_bids[::-1]
            + [self.proposeable_bids[0]]
            + [self.proposeable_bids[0]]
        )

        if DEBUG:
            pass  # print(f"bids by utility: {self.propose_bids_by_utility.items()}")
            pass  # print(f"Proposeable bids: {self.proposeable_bids}")

        neg_index = get_current_negotiation_index(self)

        # we check if we should prefer None over the any offer
        for outcome, outcome_ufun in zip(
            self.all_possible_outcomes, self.ufun_all_possible_outcomes
        ):
            if outcome_ufun < self.ufun_all_possible_outcomes[0] * 0.92:
                # the outcomes on all_possible_outcomes are sorted by utility, so if we find an outcome that is less than 95% of the best outcome, we can stop checking
                return
            if (
                outcome_ufun >= self.ufun_all_possible_outcomes[0] * 0.92
                and outcome[neg_index:].count(None) == len(outcome[neg_index:])
                and outcome[neg_index:].count(None) != 0
            ):
                self.do_none = True
                return

    def update_strategy_edge(self, state):
        negotiator_id = list(self.negotiators.keys())[0]
        nmi: SAONMI = self.negotiators[negotiator_id].negotiator.nmi

        all_possible_bids = all_possible_bids_with_agreements_fixed(self)
        ordered_bids = sorted(
            all_possible_bids, key=lambda x: self.ufun(x), reverse=True
        )
        best_bid_utility = self.ufun(ordered_bids[0])

        level_reduction = 0

        try:
            self.max_reduce_bottom_utility_per_neg = 0.2
            a = self.crisp_ufun.n_edges - state.n_acceptances
            level_reduction = self.remaining_scaled_flipped_edge(state)
        except:
            pass

        self.negotiators[negotiator_id].negotiator

        level = 0.5
        # we propose only bids that are the best bid or are close to it (the closeness decends as time passes using the `level`)
        self.proposeable_bids = [
            bid
            for bid in ordered_bids
            if self.ufun(bid) >= best_bid_utility * (level + level_reduction)
        ]
        self.proposeable_bids = self.proposeable_bids[::-1] + [self.proposeable_bids[0]]

    def _update_strategy(self, state) -> None:
        if not is_edge_agent(self):
            return self.update_strategy_center()

        if is_edge_agent(self):
            return self.update_strategy_edge(state)
