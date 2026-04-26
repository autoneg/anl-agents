"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from negmas.outcomes import Outcome

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    get_target_bid_at_current_index,
    is_edge_agent,
    find_best_bid_in_outcomespace,
    all_possible_bids_with_agreements_fixed,
    get_outcome_space_from_index,
    get_current_negotiation_index,
    get_agreement_at_index,
)

# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
# from helpers.helperfunctions import set_id_dict, ...
from anl2025.ufun import SideUFun
from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

__all__ = ["JobHunterNegotiator"]


class JobHunterNegotiator(ANL2025Negotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    This example agent aims for the absolute best bid available. As a center agent, it adapts its strategy after each negotiation, by aiming for the best bid GIVEN the previous outcomes.
    """

    """
       The most general way to implement an agent is to implement propose and respond.
       """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.min_val_idx_acceptable = 0.35
        self.rel_t_for_agreements = 0.3
        self.is_debugging = False
        self.can_improve = True
        self.conssession_exp = 1 / 4
        self.max_cases_to_compute = 10e4
        self.is_mcuf = (
            not is_edge_agent(self)
        ) and self.preferences.short_type_name == "MCUF"
        self.can_compute_all_pos = self.can_all_possib_be_computed()
        self.options_by_utilities = []

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).

        Remarks:
            - You can use the negotiator_id to identify what side negotiator is currently proposing. This id is stable within a negotiation.
        """
        # If the negotiation has ended, update the strategy. The subnegotiator may of may not have found an agreement: this affects the strategy for the rest of the negotiation.
        if did_negotiation_end(self):
            self.start_new_round(negotiator_id)
            self._update_strategy()

        self.c_step_ = state.step

        if self.can_compute_all_pos:  # updates on start_new_round
            if not self.can_improve:
                return None

        # Get relative time (0 at start, 1 at deadline)
        relative_time = self.get_relative_time(negotiator_id, state)

        # Generate a bid based on time
        bid = self.generate_bid_with_concession(negotiator_id, relative_time)

        # if you want to end the negotiation, return None
        self.my_print(
            "propose {0} to {1} at step {2}.".format(bid, negotiator_id, self.c_step_)
        )
        return bid

    def get_offer_util_idx(self, offer):
        for i in range(len(self.options_by_opponent_utilities)):
            if self.options_by_opponent_utilities[i][3] == offer:
                return i

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
            self.start_new_round(negotiator_id)
            self._update_strategy()

        if self.can_compute_all_pos:
            if self.does_offer_not_improve_utility(state.current_offer):
                return ResponseType.REJECT_OFFER

        # This agent is very stubborn: it only accepts an offer if it is EXACTLY the target bid it wants to have.
        # if state.current_offer is get_target_bid_at_current_index(self):
        if (
            self.c_round_ == 0
            and state.current_offer is get_target_bid_at_current_index(self)
        ) or (self.c_round_ > 0 and self.improvement_by_offer(state.current_offer) > 0):
            self.my_print("offer {0} accepted.".format(state.current_offer))
            return ResponseType.ACCEPT_OFFER

        self.my_print("offer {0} rejected.".format(state.current_offer))
        return ResponseType.REJECT_OFFER
        # You can also return ResponseType.END_NEGOTIATION to end the negotiation.

    def does_offer_not_improve_utility_mcuf(self, offer):
        current_outcomes = self.get_prev_agreements()
        no_deals_with_next_negs = [None] * (self.n_neg - (self.neg_idx + 1))
        offer_util = self.ufun(current_outcomes + [offer] + no_deals_with_next_negs)
        return self.cur_util < offer_util

    def does_offer_not_improve_utility(self, offer):
        if self.is_mcuf:
            self.does_offer_not_improve_utility_mcuf(offer)
        return False

    def get_prev_agreements(self):
        neg_index = get_current_negotiation_index(self)
        return [get_agreement_at_index(self, i) for i in range(neg_index)]

    def start_new_round(self, negotiator_id):
        _, cntxt = self.negotiators[negotiator_id]
        self.op_ufun: SideUFun = cntxt["ufun"]
        self.curr_op_nmi = self.get_nmi_from_id(negotiator_id)
        self.c_round_ = len(self.finished_negotiators)
        self.can_compute_all_pos = self.can_all_possib_be_computed()
        self.neg_idx = get_current_negotiation_index(self)
        self.n_neg = len(self.negotiators)

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""
        # if your current role is the edge agent, use the strategy as if your centeragent is in it's last subnegotiation.
        # In this case, just get the best bid from the utility function.
        if is_edge_agent(self):
            # note that the edge utility function has a slightly different structure than a center utility function.
            _, best_bid = self.ufun.extreme_outcomes()
            all_possible = self.get_possibilities_edge()
            utils = [(outcome, self.ufun(outcome), outcome) for outcome in all_possible]
            self.order_utilities(utils)
            self.can_improve = True
        else:
            if self.is_mcuf:
                all_possible = self.get_outcome_space()
                utils = [
                    (outcome, self.ufun(outcome), outcome[self.c_round_])
                    for outcome in all_possible
                ]
                self.order_utilities(utils)
                self.calc_cur_util_mcuf()

            self.can_improve = self.can_improve_state()
            best_bid = self.find_best_bid()

        self.target_bid = best_bid
        # print(self.target_bid)

    def calc_cur_util_mcuf(self):
        if self.c_round_ > 0:
            # the utility of the option with current deal as None
            self.cur_util = next(
                option[1]
                for option in self.options_by_utilities
                if option[0][self.c_round_] == None
            )
        else:
            self.cur_util = 0

    def can_all_possib_be_computed(self):
        if not self.preferences.outcome_space.is_finite():
            return False
        if is_edge_agent(self):
            return True
        n_possib_left = 1
        neg_index = get_current_negotiation_index(self)
        n_neg = len(self.negotiators)
        for i in range(neg_index, n_neg):
            if not self.is_mcuf:
                n_possib_left = n_possib_left * len(
                    get_outcome_space_from_index(self, neg_index)
                )
            else:
                n_possib_left = n_possib_left + len(
                    get_outcome_space_from_index(self, neg_index)
                )
        return n_possib_left <= self.max_cases_to_compute

    def can_improve_state_mcuf(self):
        op_by_ut = self.options_by_utilities
        n_offers = len(op_by_ut)
        if n_offers > 0 and op_by_ut[0][1] != op_by_ut[n_offers - 1][1]:
            return True
        return False

    def can_improve_state(self):
        if self.is_mcuf:
            return self.can_improve_state_mcuf()
        return True

    def get_possibilities_edge(self) -> list[Outcome | None]:
        return get_outcome_space_from_index(self, 0)

    def calc_outcome_space_mcuf(self):
        current_outcomes = self.get_prev_agreements()
        bids = get_outcome_space_from_index(self, self.neg_idx)
        no_deals_with_next_negs = [None] * (self.n_neg - (self.neg_idx + 1))

        return [current_outcomes + [bid] + no_deals_with_next_negs for bid in bids]

    def get_outcome_space(self):
        # get outcome space for general case and narrowed outcome space for max center
        if self.is_mcuf:
            return self.calc_outcome_space_mcuf()
        return all_possible_bids_with_agreements_fixed(self)

    def find_best_bid(self):
        best_bid = find_best_bid_in_outcomespace(self)

        current_outcomes = self.get_prev_agreements()

        last_bid = best_bid[0]
        neg_index = get_current_negotiation_index(self)
        if neg_index != 0:
            last_bid = current_outcomes[neg_index - 1]

        for i in range(len(self.negotiators) - neg_index):
            current_outcomes.append(last_bid)

        best_bid_u = self.ufun(best_bid)
        current_outcomes_u = self.ufun(current_outcomes)
        if self.ufun(best_bid) <= self.ufun(current_outcomes):
            best_bid = current_outcomes
        return best_bid

    def order_utilities(self, utilities):
        self.options_by_utilities = sorted(utilities, key=lambda x: x[1])

    def improvement_by_offer(self, offer):
        deals_with_offer = self.get_prev_agreements()
        for i in range(len(self.active_negotiators)):
            deals_with_offer.append(offer)

        util_of_offer = self.ufun(deals_with_offer)
        if self.cur_util >= util_of_offer:
            return 0
        return util_of_offer - self.cur_util

    def get_relative_time(self, negotiator_id: str, state: SAOState):
        nmi = self.get_nmi_from_id(negotiator_id)
        if state.step == 0:
            return 1.0
        elif nmi.n_steps is not None and state.step >= nmi.n_steps - 1:
            return 1
        return state.relative_time

    def get_possibilities(self) -> list[list[Outcome | None]]:
        """All option bids for current round, each option represented as full set with the previous deals
        inserted and next deals equals to bid."""
        return self.get_outcome_space()
        # TODO: in case of != Max -> calc distribution for all possib of min(1000/len(bids), n_neg-neg_index) following indexes and the rest as None

    def find_bid_with_utility_level(
        self, concession_factor, possible_by_utilities: list[Outcome]
    ):
        worst_acceptable = self.min_val_idx_acceptable
        normalized_conc_fact = concession_factor * worst_acceptable + worst_acceptable
        possible_index = int((len(possible_by_utilities) - 1) * normalized_conc_fact)
        return possible_by_utilities[possible_index][0]

    def min_max_offer(self):
        pos_by_ut = self.options_by_utilities
        best_bid_util = pos_by_ut[len(pos_by_ut) - 1][1]
        maxs = filter(lambda x: x[1] == best_bid_util, pos_by_ut)
        maxs_with_side_utility = [
            (t[0], t[1], self.op_ufun(t[0][self.c_round_])) for t in maxs
        ]
        maxs_by_min = sorted(maxs_with_side_utility, key=lambda t: t[2], reverse=True)
        min_max_offer = maxs_by_min[0][0]
        return min_max_offer

    def generate_bid_with_concession(self, negotiator_id, relative_time):
        """Generate a bid based on concession strategy."""
        # Boulware strategy: concede slowly at first, then faster
        concession_factor = pow(
            relative_time, self.conssession_exp
        )  # Adjust exponent for concession speed

        # Start with best bid for us
        if is_edge_agent(self):
            worst_bid, best_bid = self.ufun.extreme_outcomes()
            return best_bid
        else:
            possible_by_utilities = self.options_by_utilities

            if not self.can_improve:
                return None

            best_bid = self.min_max_offer()[self.c_round_]
            # return best_bid

        # As time progresses, be willing to accept worse bids
        if relative_time > self.rel_t_for_agreements:
            return best_bid

        # In final stages, consider concession
        return self.find_bid_with_utility_level(
            concession_factor, possible_by_utilities
        )[self.c_round_]

    def get_nmi_from_id(self, negotiators_id):
        """Get negotiator mechanism interface from negotiator ID."""
        return self.negotiators[negotiators_id].negotiator.nmi

    def my_print(self, str):
        if self.is_debugging:
            pass  # print(str)


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    # Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    # Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(JobHunterNegotiator, small=True)
