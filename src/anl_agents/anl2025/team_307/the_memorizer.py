"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from negmas.outcomes import Outcome
import numpy
from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_agreement_at_index,
    get_outcome_space_from_index,
    get_current_negotiation_index,
)

from anl2025.negotiator import ANL2025Negotiator
from anl2025.ufun import SideUFun, CenterUFun
from negmas import ResponseType

max_samples = 30

__all__ = ["TheMemorizer"]


class MockAdapter:
    def init(self):
        self.can_compute_all_pos = False
        self.can_improve = True

    def update_between_rounds(self, agent):
        return

    def does_offer_not_improve_utility(self, agent, offer):
        return False

    def can_all_possib_be_computed(self):
        return False


class McufAdapter:
    # can improve
    # can calc all posib
    # does_offer_not_improve_utility
    def init(self, agent):
        self.max_cases_to_compute = 10e4
        self.can_compute_all_pos = self.can_all_possib_be_computed(agent)
        self.ufun = agent.ufun
        self.cur_util = 0
        self.can_improve = True
        self.is_debugging = True
        # self.utilities = [None] * len(self.negotiators)

    def update_between_rounds(self, agent):
        self.c_round_ = len(agent.finished_negotiators)
        self.can_compute_all_pos = self.can_all_possib_be_computed(agent)
        self.neg_idx = get_current_negotiation_index(agent)
        self.n_neg = len(agent.negotiators)
        if is_edge_agent(agent):
            all_possible = self.get_possibilities_edge(agent)
            utils = [(outcome, self.ufun(outcome), outcome) for outcome in all_possible]
            self.order_utilities(utils)
            self.can_improve = True
        else:
            all_possible = self.get_outcome_space(agent)
            utils = [
                (outcome, self.ufun(outcome), outcome[self.c_round_])
                for outcome in all_possible
            ]
            self.order_utilities(utils)
            self.calc_cur_util_mcuf()

            self.can_improve = self.can_improve_state()

    def order_utilities(self, utilities):
        self.options_by_utilities = sorted(utilities, key=lambda x: x[1])

    def does_offer_not_improve_utility(self, agent, offer):
        if not is_edge_agent(agent):
            current_outcomes = self.get_prev_agreements(agent)
            no_deals_with_next_negs = [None] * (self.n_neg - (self.neg_idx + 1))
            offer_util = self.ufun(current_outcomes + [offer] + no_deals_with_next_negs)
            return self.cur_util < offer_util
        return False

    def get_prev_agreements(self, agent):
        neg_index = get_current_negotiation_index(agent)
        return [get_agreement_at_index(agent, i) for i in range(neg_index)]

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

    def can_all_possib_be_computed(self, agent):
        if not agent.preferences.outcome_space.is_finite():
            return False
        if is_edge_agent(agent):
            return True
        n_possib_left = 1
        neg_index = get_current_negotiation_index(agent)
        n_neg = len(agent.negotiators)
        for i in range(neg_index, n_neg):
            n_possib_left = n_possib_left + len(
                get_outcome_space_from_index(agent, neg_index)
            )
            # if not self.is_mcuf:
            #    n_possib_left = n_possib_left * len(get_outcome_space_from_index(agent, neg_index))
            # else:
        return n_possib_left <= self.max_cases_to_compute

    def can_improve_state_mcuf(self):
        op_by_ut = self.options_by_utilities
        n_offers = len(op_by_ut)
        if n_offers > 0 and op_by_ut[0][1] != op_by_ut[n_offers - 1][1]:
            return True
        return False

    def can_improve_state(self):
        # if self.is_mcuf:
        return self.can_improve_state_mcuf()
        # return True

    def get_possibilities_edge(self, agent) -> list[Outcome | None]:
        return get_outcome_space_from_index(agent, 0)

    def get_outcome_space(self, agent):
        # get outcome space for general case and narrowed outcome space for max center
        # if self.is_mcuf:
        # else:
        # return all_possible_bids_with_agreements_fixed(self)
        # return self.calc_outcome_space_mcuf()
        current_outcomes = self.get_prev_agreements(agent)
        bids = get_outcome_space_from_index(agent, self.neg_idx)
        no_deals_with_next_negs = [None] * (self.n_neg - (self.neg_idx + 1))

        return [current_outcomes + [bid] + no_deals_with_next_negs for bid in bids]

    # def calc_outcome_space_mcuf(self):

    def get_possibilities(self) -> list[list[Outcome | None]]:
        """All option bids for current round, each option represented as full set with the previous deals
        inserted and next deals equals to bid."""
        return self.get_outcome_space()
        # TODO: in case of != Max -> calc distribution for all possib of min(1000/len(bids), n_neg-neg_index) following indexes and the rest as None


class TheMemorizer(ANL2025Negotiator):
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
        self.agreements = []
        self.samples = None
        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None
        self.last_neg = ""
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)
        self.agreements = []
        self.best_pattern = None
        self.best_utility = float("-inf")
        self.num_negotiations = len(self.id_dict)
        self.trace_by_neg = {}
        self.current_offer = None
        self.rejection_counts = {}  # {negotiator_id: {outcome: [i_rejected, opponent_rejected]}}
        is_mcuf = (
            self.preferences.short_type_name == "MCUF"
        )  # and (not is_edge_agent(self))
        self.adapter = McufAdapter()
        self.adapter.init(self)
        if not is_mcuf or not self.adapter.can_compute_all_pos:
            self.adapter = MockAdapter()
            self.adapter.init()

        self.is_debugging = False
        # self.utilities = [None] * len(self.negotiators)

    def _get_possible_outcomes(self, neg_id):
        """Get all possible outcomes for a negotiation by id."""

        nmi = self.negotiators[neg_id].negotiator.nmi
        ufun = self.negotiators[neg_id].negotiator.ufun
        if self.samples is not None:
            return self.samples
        all_outcomes = list(nmi.outcome_space.enumerate_or_sample())
        valid_outcomes = []
        if len(all_outcomes) > 1000:
            self.samples = all_outcomes
            return all_outcomes
        # print(type(all_outcomes[0][0]))
        max_samples = 1000
        # return scored[:max_samples]

        for o in all_outcomes:
            try:
                _ = ufun(o)
                valid_outcomes.append(o)
            except Exception:
                continue  # skip malformed offers

        scored = sorted(valid_outcomes, key=ufun, reverse=True)
        self.samples = scored[:max_samples]
        # print(self.samples)
        return scored[:max_samples]

        # print(len(all_outcomes))

    def _get_progress(self, negotiator_id):
        """Get the current negotiation progress (0 to 1)."""
        nmi = self.negotiators[negotiator_id].negotiator.nmi
        return nmi.state.relative_time if nmi.state.relative_time is not None else 0

    def _find_best_outcome(self, negotiator_id, dict_outcome_space, ufun):
        """Find the best outcome for the current negotiation."""
        # For edge agents, find outcome with highest utility
        """
                the leverage is calculated by the negotiation number.
                Further along the way, the edge negotiator should hold more leverage. 

        """
        self.leverage = int(negotiator_id[-1]) + 1
        if is_edge_agent(self):
            best_outcome = None
            best_utility = float("-inf")
            self.pattern_outcomes = {}
            outcomes = self._get_possible_outcomes(negotiator_id)
            outcomes.append(self.current_offer)
            for outcome in outcomes:
                try:
                    i_rejected = dict_outcome_space[negotiator_id][outcome][1]
                    opp_rejected = dict_outcome_space[negotiator_id][outcome][2]
                except:
                    i_rejected = opp_rejected = 0

                #   tuple(str(int(outcome[0]) + (0.1 * i_rejected) - (0.1 * opp_rejected)))
                utility = (
                    (ufun(outcome))
                    + (0.000002 * i_rejected)
                    - (0.000002 * opp_rejected)
                )
                if outcome is None:
                    utility *= 0.75
                self.pattern_outcomes[outcome] = utility
                if utility > best_utility:
                    best_outcome = outcome
                    best_utility = utility

            return best_outcome, best_utility

        context = self.agreements.copy()
        len_ctxt = len(context)
        context += [(None, None)]  # * (len(self.negotiators) - len(context))
        self.pattern_outcomes = {}
        best_outcome = None
        best_utility = float("-inf")

        # Try each possible outcome + furute theoretic outcomes.
        outcomes = self._get_possible_outcomes(negotiator_id)
        outcomes.append(self.current_offer)
        for outcome in outcomes:
            if outcome is None:
                continue
            test_context = context.copy()
            test_context[int(negotiator_id[1])] = outcome
            # rest_combs = itertools.combinations(self._get_possible_outcomes(negotiator_id), len(self.negotiators.keys()) - (len_ctxt + 1))

            try:
                i_rejected = dict_outcome_space[negotiator_id][outcome][1]
                opp_rejected = dict_outcome_space[negotiator_id][outcome][2]
            except:
                i_rejected = opp_rejected = 0
            # print(opp_rejected)
            sum_utility = 0
            num_utility = 0
            combs_list = []
            remaining = (
                len(self.negotiators) - len(self.finished_negotiators) - 1
            )  # (len_ctxt + 1)

            sampled_outcomes = self._get_possible_outcomes(negotiator_id)
            # SAMPLE_SIZE = min(20, len(sampled_outcomes))
            level = self._get_progress(negotiator_id)
            fake_rest = [None] * remaining
            test_context_comb = test_context + fake_rest
            avg_util_inter = 0
            sum_util_inter = 0

            utility = self.ufun(test_context_comb) - (
                0.05 * level * opp_rejected * (pow(10, -(1 * self.leverage - 1)))
            )
            avg_util = utility  # = avg_util_inter = sum_util_inter / len(test_context)

            if outcome is None:
                avg_util *= 0.75
            self.pattern_outcomes[outcome] = avg_util

            if avg_util > best_utility:
                best_outcome = outcome
                best_utility = avg_util

        # Try having no agreement
        test_context = context.copy()
        test_context += [
            None for i in range(len(self.negotiators.keys()) - (len_ctxt + 1))
        ]

        # none_utility = ufun(test_context)

        # if none_utility > best_utility:
        #     return None, 0
        return best_outcome, best_utility

    def calc_dict(self, negotiator_id, nmi, ufun, level):
        if is_edge_agent(self):
            pass
        if negotiator_id not in self.rejection_counts:
            self.rejection_counts[negotiator_id] = {}

        trace = nmi.extended_trace
        existing_outcomes = self.rejection_counts[negotiator_id]

        # Process only the NEW offers in the trace
        prev_len = len(existing_outcomes)
        for event in trace[prev_len:]:  # assumes trace is append-only
            offer = event[2]
            is_mine = event[1] == negotiator_id

            if offer not in existing_outcomes:
                existing_outcomes[offer] = [0, 0]  # [i_rejected, opponent_rejected]

            if is_mine:
                existing_outcomes[offer][1] += 1  # opponent rejected mine
            else:
                existing_outcomes[offer][0] += 1  # I rejected theirs

        # Now build dict_outcome_space from rejection counts
        dict_outcome_space = {}
        outcomes = self._get_possible_outcomes(negotiator_id)

        for o in outcomes:
            i_rej, opp_rej = existing_outcomes.get(o, [0, 0])
            dict_outcome_space[o] = [level, i_rej, opp_rej]

        self.trace_by_neg[negotiator_id] = dict_outcome_space
        return dict_outcome_space

    def propose(self, negotiator_id, state, dest=None):
        if negotiator_id.startswith("s"):
            pass
        if negotiator_id != self.last_neg and self.last_neg != "":
            self.agreements.append(self.last_proposal)
        """Generate a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        if did_negotiation_end(self):
            self._update_agreements_if_needed()
            self.adapter.update_between_rounds(self)

        if self.adapter.can_compute_all_pos:  # updates on start_new_round
            if self.adapter.c_round_ > 0 and not self.adapter.can_improve:
                # self.my_print("{0} propose None to {1} at step {2}.".format("edge" if is_edge_agent(self) else "center", negotiator_id, state.relative_time))
                return None

        self.cur_state = state
        negotiator, cntxt = self.negotiators[negotiator_id]
        nmi = negotiator.nmi
        level = self._get_progress(negotiator_id)
        ufun: SideUFun = cntxt["ufun"]
        if not is_edge_agent(self):
            ufun: CenterUFun = cntxt["ufun"]
        step = state.step
        current_offer = state.current_offer
        my_offers = []
        oponnent_offers = []

        # A dictionary that states for each outcome how many times it was proposed and rejected by each negotiator
        dict_outcome_space = self.calc_dict(negotiator_id, nmi, ufun, level)

        if is_edge_agent(self):
            best_outcome, best_utility = self._find_best_outcome(
                negotiator_id, self.trace_by_neg[negotiator_id], ufun
            )
            # print(f'{self.id} proposed {best_outcome} to {dest}')
            # self.my_print("{0} propose {1} to {2} at step {3}.".format("edge" if is_edge_agent(self) else "center", best_outcome, negotiator_id, state.relative_time))
            return best_outcome
        # Find best outcome
        best_outcome, best_utility = self._find_best_outcome(
            negotiator_id, self.trace_by_neg, ufun
        )

        # print(f'{self.id} proposed {best_outcome} to {dest}')
        self.last_proposal = best_outcome
        self.last_neg = negotiator_id

        # if int(negotiator_id[-1]) < 2 and level > 0.3:
        # return(None)
        # self.my_print("{0} propose {1} to {2} at step {3}.".format("edge" if is_edge_agent(self) else "center", best_outcome, negotiator_id, state.relative_time))
        return best_outcome

    def respond(self, negotiator_id, state, source=None):
        if negotiator_id.startswith("s"):
            pass
        """Respond to a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        if did_negotiation_end(self):
            self._update_agreements_if_needed()
            self.adapter.update_between_rounds()

        # If no offer, reject
        self.cur_state = state
        if state.current_offer is None:
            # print(f'{self.id} responds REJECT')
            # self.my_print("{0}: offer {1} rejected (None).".format("e" if is_edge_agent(self) else "c", state.current_offer))
            return ResponseType.REJECT_OFFER
        # print(f'{self.id} recieves {state.current_offer}')

        if self.adapter.can_compute_all_pos:
            if self.adapter.does_offer_not_improve_utility(self, state.current_offer):
                # self.my_print("{0}: offer {1} rejected (mcuf).".format("e" if is_edge_agent(self) else "c", state.current_offer))
                return ResponseType.REJECT_OFFER

        negotiator, cntxt = self.negotiators[negotiator_id]
        nmi = negotiator.nmi
        level = self._get_progress(negotiator_id)
        ufun: SideUFun = cntxt["ufun"]
        if not is_edge_agent(self):
            ufun: CenterUFun = cntxt["ufun"]
        step = state.step
        current_offer = state.current_offer
        my_offers = []
        oponnent_offers = []
        # Same dict as in propose
        dict_outcome_space = self.calc_dict(negotiator_id, nmi, ufun, level)

        best_outcome, best_utility = self._find_best_outcome(
            negotiator_id, self.trace_by_neg, ufun
        )

        offer_utility = self.pattern_outcomes[current_offer]
        all_utilities = list(self.pattern_outcomes.values())
        mean_utility = numpy.mean(all_utilities)
        progress = self._get_progress(negotiator_id)
        agent_type_factor = 1 if is_edge_agent(self) else 1.2

        # Variance adjustment — higher std => lower z
        std_utility = numpy.std(all_utilities)
        std_utility = max(std_utility, 1e-5)  # avoid division by zero

        # Normalize std_utility against mean to make it scale-invariant
        std_ratio = std_utility / (numpy.mean(all_utilities) + 1e-5)
        # print(agent_type_factor)
        base_z = 3 * ((1 - progress) * (agent_type_factor))

        # Final z: reduced further as variance increases (e.g., z ~ 1/std)
        z = base_z / (1 + 5 * (std_ratio))  # 20 is a tuning hyperparameter
        # z = max(-5, z)
        if not is_edge_agent(self):
            pass
        if offer_utility > (mean_utility + (z * std_utility)):
            # self.my_print("{0}: {1}, {2}, {3}, {4}".format(negotiator_id, level, offer_utility, (mean_utility + (z * std_utility)), best_utility))

            # self.my_print("{0}: offer {1} accepted.".format("e" if is_edge_agent(self) else "c", state.current_offer))
            return ResponseType.ACCEPT_OFFER

        if (offer_utility / best_utility) < 0.15:
            # self.my_print('good enough')

            # self.my_print("{0}: offer {1} rejected.".format("e" if is_edge_agent(self) else "c", state.current_offer))
            return ResponseType.ACCEPT_OFFER

        # self.my_print("{0}: offer {1} rejected.".format("e" if is_edge_agent(self) else "c", state.current_offer))
        return ResponseType.REJECT_OFFER

    def _update_agreements_if_needed(self):
        """Update the agreements list if a negotiation has ended."""
        if did_negotiation_end(self):
            # Store the agreement from the just-ended negotiation
            prev_index = self.current_neg_index - 1
            if prev_index >= 0:
                agreement = get_agreement_at_index(self, prev_index)
                while len(self.agreements) <= prev_index:
                    self.agreements.append(None)
                self.agreements[prev_index] = agreement
                return True
        return False

    # def my_print(self, str):
    #   if self.is_debugging:
    #      print(str)
