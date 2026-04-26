"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* J(Jacob) E(Eyal) E(Eyal) M(Mark)- Jeem negotiator
Mark Vexler <markvexler@hotmail.com> Eyal Elboim <eyal.elboim1@gmail.com> Jacob Gigron <jacob.o.gidron@gmail.com>
Eyal Seckbach <eyal.seck@gmail.com>
"""

from negmas.outcomes import Outcome
from negmas.negotiators.controller import TState
from negmas.sao.controllers import SAOState
from negmas import ResponseType
import anl2025.ufun
from anl2025.negotiator import ANL2025Negotiator
from collections import Counter
import itertools
from random import random

# Propose defines
# This is the minimum factor for bets utility, the agent doesn't compromise below this
EDGE_PROPOSE_MINIMUM_FACTOR = 0.6
EDGE_PROPOSE_MINIMUM_FACTOR_ACCUMULATE_GAME = 0.7
CENTER_PROPOSE_MINIMUM_FACTOR = 0.7
EDGE_RESPONSE_MINIMUM_FACTOR_FOR_LAST_ROUND = 0.15

# Dedicated values for MAX game
# Center MAX parameters
CENTER_PROPOSE_MINIMUM_FACTOR_FOR_MAX_BEGINNING = 0.9
CENTER_PROPOSE_MINIMUM_FACTOR_FOR_MAX_END = 0.65
CENTER_PROPOSE_MINIMUM_FACTOR_FOR_MAX_LAST = 0.3
CENTER_PROPOSE_MINIMUM_STAGES_BEFORE_LOW_THRESHOLD = 3

# This is the minimum steps(rounds) before allowing to compromise for utility
EDGE_MINIMUM_STEPS_FOR_DISCOUNT = 0.5
CENTER_MINIMUM_STEPS_FOR_DISCOUNT = 0.5
# This is the minimum stages(negotiations) before allowing to compromise for utility, edge has no meaning here
EDGE_MINIUM_STAGES_FOR_DISCOUNT = 0
CENTER_MINIUM_STAGES_FOR_DISCOUNT = 0.6
UTILITY_QUANTIZATION_RESOLUTION = 100

MAX_OUTCOMES_TO_EVALUATE = 1000000

__all__ = ["JeemNegotiator"]


class JeemNegotiator(ANL2025Negotiator):
    def init(self):
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        self.id_dict = {}
        for id in self.negotiators.keys():
            index = self.negotiators[id].context["index"]
            self.id_dict[index] = id

        self.is_edge = self.id.__contains__("edge")
        self.number_of_negotiations = len(self.negotiators)
        self.utility_type = type(self.ufun)

        # The best utility the agent can get currently
        self.best_stage_utility = -1
        # The current utility the agent already has
        self.current_utility = -1
        # A list contains pair of utility, list of bids that achieve this utility
        self.stage_utility_to_bids = {}
        # A list of bids the agent is willing to propose in round
        self.bids_to_suggest_in_this_round = []
        # Best utility if current negotiation ends with None
        self.skip_current_negotiation_utility = -1
        # Check if each negotiation stand by itself
        if isinstance(
            self.ufun,
            (
                anl2025.ufun.UtilityCombiningCenterUFun,
                anl2025.ufun.SingleAgreementSideUFunMixin,
            ),
        ):
            self.is_single_negotiation = True
        else:
            self.is_single_negotiation = False

        if self.is_edge:
            self.propose_minimum_factor = EDGE_PROPOSE_MINIMUM_FACTOR
            if self.is_single_negotiation:
                self.propose_minimum_factor += (
                    EDGE_PROPOSE_MINIMUM_FACTOR_ACCUMULATE_GAME
                )
            self.minimum_steps_for_discount = EDGE_MINIMUM_STEPS_FOR_DISCOUNT
            self.minimum_stages_for_discount = EDGE_MINIUM_STAGES_FOR_DISCOUNT
        else:
            self.propose_minimum_factor = CENTER_PROPOSE_MINIMUM_FACTOR
            self.minimum_steps_for_discount = CENTER_MINIMUM_STEPS_FOR_DISCOUNT
            self.minimum_stages_for_discount = CENTER_MINIUM_STAGES_FOR_DISCOUNT

    def on_negotiation_start(self, negotiator_id: str, state: SAOState) -> None:
        self.current_neg_index = len(self.finished_negotiators)
        self.number_of_steps = self.negotiators[negotiator_id].negotiator.nmi.n_steps
        self.negotiator_id = negotiator_id

        # Calculate how much utility the agent gained so far
        self.current_utility, _ = self.calculate_current_utility()

        # Get a list of pairs: utility, and outcomes that gain this utility
        updated_outcomes = self.get_possible_bids_with_agreements_fixed()

        # Get a list of pairs: utility and a set of bids in this negotiation that can achieve this utility
        self.stage_utility_to_bids, self.skip_current_negotiation_utility = (
            self.bids_list_for_each_utility(updated_outcomes)
        )

        self.best_stage_utility = self.stage_utility_to_bids[0][0]
        self.bids_to_suggest_in_this_round = []

    def on_round_start(self, negotiator_id: str, state: TState) -> None:
        super().on_round_start(negotiator_id, state)

        # There are still proposals: continue with them
        if len(self.bids_to_suggest_in_this_round) != 0:
            return

        # No point to continue: better to not propose anything
        if self.best_stage_utility <= self.current_utility:
            return

        # Nothing is in the list of proposal: prepare new one. The list will be the proposals for this cycle. there
        # might be few cycles in each round, in each cycle the agent compromise more on the utility
        # Calculate which utilities to consider in proposal
        discount_utility = self.discount_utility(state)

        # Calculate how much the agent is willing to compromise on the best utility it can achieve
        if self.is_single_negotiation:
            min_gain = self.best_stage_utility - self.current_utility
            min_gain *= discount_utility
            min_gain = max(
                min_gain,
                (self.best_stage_utility - self.current_utility)
                * self.propose_minimum_factor,
            )
            utility_to_consider_threshold = self.current_utility + min_gain
        else:
            utility_to_consider_threshold = self.best_stage_utility * discount_utility
            # The agent shall not compromise more than a minimum percentage of the best utility it can achieve
            utility_to_consider_threshold = max(
                utility_to_consider_threshold,
                self.propose_minimum_factor * self.best_stage_utility,
            )

        # For each utility which is higher than the discounted best utility, take all bids as an option including bids
        # that already participated. Dont repeat candidates in each cycle
        bids_to_suggest_in_this_round_set = set()

        for future_utility, counter, bid in self.stage_utility_to_bids:
            # Take the bids only:
            # 1. The future utility is at least as the utility the agent is willing to compromise on
            # 2. The future utility is better than the current utility
            # 3. The future utility is at least the same as future utility if selecting None in this negotiation
            if (
                future_utility >= utility_to_consider_threshold
                and future_utility > self.current_utility
                and future_utility >= self.skip_current_negotiation_utility
            ):
                if bid is not None and bid not in bids_to_suggest_in_this_round_set:
                    # Add both to list and to set: set is to not repeat candidates and list is to pop according to
                    # the order of inserting- higher utility first and counter is second priority
                    self.bids_to_suggest_in_this_round.append(bid)
                    bids_to_suggest_in_this_round_set.add(bid)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # If there are no bids to propose end negotiation and send None
        if len(self.bids_to_suggest_in_this_round) == 0:
            return None
        else:
            # Propose the first in the list- higher utility first
            bid_to_propose = self.bids_to_suggest_in_this_round.pop(0)
            return bid_to_propose

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        nmi = self.get_nmi_from_index(self.current_neg_index)

        # Go over all bids suggested before and accept if the current offer is one of them
        if nmi is not None:
            trace = nmi.trace
            my_bids = [
                bid[1]
                for bid in trace
                if bid[0] == negotiator_id and bid[1] is not None
            ]
            if state.current_offer in my_bids:
                return ResponseType.ACCEPT_OFFER

        # Go over future proposals and accept if the current offer is one of them
        if state.current_offer in self.bids_to_suggest_in_this_round:
            return ResponseType.ACCEPT_OFFER

        utility_with_bid, bid_gain = self.calculate_bid_gain(state.current_offer)

        # in the last round of the edge just compromise if the bid gives any additional utility over no agreement at all
        if self.is_edge:
            if (
                state.step + 1 == self.number_of_steps
                and self.current_utility < utility_with_bid
            ):
                return ResponseType.ACCEPT_OFFER

        # If it is MAX negotiation and we are in last round
        # For center: if we are in the last negotiation and the gain is still significant- accept proposal
        if (
            not self.is_edge
            and self.ufun.type_name == "anl2025.ufun.MaxCenterUFun"
            and state.step + 1 == self.number_of_steps
            and self.current_utility < utility_with_bid
        ):
            if not self.is_edge:
                if (
                    self.current_neg_index + 1 == self.number_of_negotiations
                    and self.current_utility == 0
                    and bid_gain
                    >= (
                        self.best_stage_utility
                        * CENTER_PROPOSE_MINIMUM_FACTOR_FOR_MAX_LAST
                    )
                ):
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    # This function calculate the utility achieved so far- calculating the utility if the rest of the negotiations are
    # None. For edge is simply the utility of each bid
    def calculate_current_utility(self):
        if self.is_edge:
            return self.ufun(None), []

        # Collect all agreements so far
        current_list = []
        for i in range(self.current_neg_index):
            current_list.append(self.get_agreement_at_index(i))
        # Put None in the future negotiation including the current
        for i in range(self.current_neg_index, self.number_of_negotiations):
            current_list.append(None)

        return self.ufun(tuple(current_list)), current_list

    # Calculate the added value to the utility assuming the bid is accepted. Not used by the edge agent
    def calculate_bid_gain(self, bid):
        if self.is_edge:
            utility_with_bid = self.ufun(bid)
            return self.ufun(bid), utility_with_bid - self.current_utility

        # Collect all agreements so far
        current_list = []
        for i in range(self.current_neg_index):
            current_list.append(self.get_agreement_at_index(i))
        # Add the bid to current agreements
        current_list.append(bid)
        # Assume future negotiations end in None
        for i in range(self.current_neg_index + 1, self.number_of_negotiations):
            current_list.append(None)

        utility_with_bid = self.ufun(tuple(current_list))

        return utility_with_bid, utility_with_bid - self.current_utility

    # This function give a discount on the utility the agent is trying to achieve: calculates how much the agent is
    # willing to compromise
    def discount_utility(self, state):
        # Check if the round is too early to compromise
        if (state.step + 1) / self.number_of_steps < self.minimum_steps_for_discount:
            return 1

        # Allow compromising in the later step(round)- the discount factor for the utility the agent try to achieve
        # is decreasing with the step, meaning compromising on smaller utility
        discount = self.propose_minimum_factor + (1 - self.propose_minimum_factor) * (
            (self.number_of_steps - (state.step + 1)) / self.number_of_steps
        )

        if not self.is_edge and not self.is_single_negotiation:
            # Check if the stage is too early to compromise
            if (
                self.current_neg_index + 1
            ) / self.number_of_negotiations < self.minimum_stages_for_discount:
                return 1

            # Allow compromising in the later stages(negotiations)- the discount factor for the utility the agent
            # try to achieve is decreasing with the stage, meaning compromising on smaller utility
            discount += (
                self.propose_minimum_factor
                + (1 - self.propose_minimum_factor)
                * (self.number_of_negotiations - (self.current_neg_index + 1))
                / self.number_of_negotiations
            )
            # For center agents the discount is dependent on a combination between the current round and current
            # negotiation
            discount /= 2

        return discount

    # This function return a list of pairs: utility and a set of bids that achieve this bid
    # In addition it also returns the maximum utility achievable if this negotiation fails
    # The utility is quantized into buckets ain order to save big sorting: the sorting is done only at the end
    # on the buckets list
    def bids_list_for_each_utility(self, updated_outcomes):
        counter_of_bids_for_each_utility = Counter()
        skip_current_negotiation_utility = -1

        for bid in updated_outcomes:
            # Find the correct bucket
            utility = round(self.ufun(bid), 2)

            # For edge just add the utility
            if self.is_edge:
                counter_of_bids_for_each_utility[(utility, bid)] += 1
            else:
                counter_of_bids_for_each_utility[
                    (utility, bid[self.current_neg_index])
                ] += 1

            # In addition, also calculate the potential utility if this negotiation fails
            if not self.is_edge and bid[self.current_neg_index] is None:
                if skip_current_negotiation_utility < utility:
                    skip_current_negotiation_utility = utility

        # Sort the bucket utility and counters
        sorted_utility_list = sorted(
            [
                [utility_bid_tuple[0], counter, utility_bid_tuple[1]]
                for utility_bid_tuple, counter in counter_of_bids_for_each_utility.items()
            ],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        return sorted_utility_list, skip_current_negotiation_utility

    # Calculate all still possible outcomes (the negotiation in the past are fixed)
    def all_possible_bids_with_agreements_fixed(self):
        neg_index = self.current_neg_index
        possible_outcomes = []

        # As the previous agreements are fixed, these are added first.
        for i in range(neg_index):
            possible_outcomes.append([self.get_agreement_at_index(i)])

        # The coming negotiations (with index higher than the current negotiation index) can still be anything,
        # so we just list all possibilities there.
        for i in range(neg_index, self.number_of_negotiations):
            possible_outcomes.append(self.get_outcome_space_from_index(i))

        # The cartesian product constructs the combinations of all possible outcomes.
        adapted_outcomes = list(itertools.product(*possible_outcomes))

        return adapted_outcomes

    def get_possible_bids_with_agreements_fixed(self):
        if self.is_edge:
            return self.ufun.outcome_space.enumerate_or_sample()

        if self.is_single_negotiation:
            return self.get_possible_bids_only_for_current_negotiation()

        possible_bids = len(self.ufun.outcome_spaces[0].enumerate_or_sample())
        possible_outcomes_length = possible_bids ** (
            self.number_of_negotiations - self.current_neg_index
        )

        if possible_outcomes_length < MAX_OUTCOMES_TO_EVALUATE:
            return self.all_possible_bids_with_agreements_fixed()

        current_agreements = []

        # As the previous agreements are fixed, these are added first.
        for i in range(self.current_neg_index):
            current_agreements.append(self.get_agreement_at_index(i))

        possible_outcomes = []
        for j in range(MAX_OUTCOMES_TO_EVALUATE):
            outcome = current_agreements.copy()

            for i in range(self.current_neg_index, self.number_of_negotiations):
                bid = random.choice(self.get_outcome_space_from_index(i))
                outcome.append(bid)

            possible_outcomes.append(tuple(outcome))

        return possible_outcomes

    def get_possible_bids_only_for_current_negotiation(self):
        current_agreements = []

        for i in range(self.current_neg_index):
            current_agreements.append(self.get_agreement_at_index(i))

        possible_bids_for_this_round = self.get_outcome_space_from_index(
            self.current_neg_index
        )
        possible_outcomes = []

        for bid in possible_bids_for_this_round:
            outcome = current_agreements.copy()
            outcome.append(bid)

            for i in range(self.current_neg_index + 1, self.number_of_negotiations):
                outcome.append(None)

            possible_outcomes.append(tuple(outcome))
        return possible_outcomes

    # This function returns the possible bids to suggest
    def get_outcome_space_from_index(self, index):
        outcomes = self.ufun.outcome_spaces[index].enumerate_or_sample()
        outcomes.append(None)

        return outcomes

    def get_nmi_from_index(self, index):
        if self.is_edge:
            negotiator_id = self.negotiator_id
        else:
            negotiator_id = self.id_dict[index]
        return self.negotiators[negotiator_id].negotiator.nmi

    # Return the agreement at specific negotiation
    def get_agreement_at_index(self, index):
        nmi = self.get_nmi_from_index(index)
        agreement = nmi.state.agreement
        return agreement
