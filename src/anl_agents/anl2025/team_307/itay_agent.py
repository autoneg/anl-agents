"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

import numpy
from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_agreement_at_index,
)

from anl2025.negotiator import ANL2025Negotiator
from anl2025.ufun import SideUFun, CenterUFun
from negmas import ResponseType

max_samples = 30

__all__ = ["ItayNegotiator"]


class ItayNegotiator(ANL2025Negotiator):
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
                    (ufun(outcome)) + (0.000001 * i_rejected) - (0.00002 * opp_rejected)
                )
                if outcome is None:
                    utility *= 0.75
                self.pattern_outcomes[outcome] = utility
                if utility > best_utility:
                    best_outcome = outcome
                    best_utility = utility

            return best_outcome, best_utility

        context = self.agreements.copy()
        # print(context, negotiator_id)
        len_ctxt = len(context)
        # if len(context) == 0:
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

            # fake_rest = random.choices(sampled_outcomes, k=remaining)
            fake_rest = [None] * remaining
            test_context_comb = test_context + fake_rest
            avg_util_inter = 0
            sum_util_inter = 0
            # for tc in test_context:
            # print(test_context_comb)
            # print(test_context)

            # test_context = [tc[0] for tc in test_context]

            utility = self.ufun(test_context_comb) - (
                0.05 * level * opp_rejected * (pow(10, -(1 * self.leverage - 1)))
            )
            # sum_util_inter += utility
            avg_util = utility  # = avg_util_inter = sum_util_inter / len(test_context)
            # sum_utility += avg_util_inter
            # for _ in range(SAMPLE_SIZE):
            #     fake_rest = random.choices(sampled_outcomes, k=remaining)
            #     test_context_comb = test_context + fake_rest
            #     avg_util_inter = 0
            #     sum_util_inter = 0
            #     for tc in test_context_comb:
            #         utility = ufun(tc) - (0.005 * level * opp_rejected * (pow(10, -(3 * self.leverage))))
            #         sum_util_inter += utility
            #     avg_util_inter = sum_util_inter / len(test_context_comb)
            #     sum_utility += avg_util_inter
            # avg_util = sum_utility / SAMPLE_SIZE
            if outcome is None:
                avg_util *= 0.75
            self.pattern_outcomes[outcome] = avg_util

            # Calculate the average theoretic rest of negotiation utility for the outcome
            # for c in rest_combs:
            #     test_context_comb = test_context.copy() + list(c)
            #     utility = ((8 / self.leverage) * self.ufun(tuple(test_context_comb))) + (0.001 * i_rejected) - ((0.01 * opp_rejected) * (1/self.leverage))
            #     sum_utility += utility
            #     num_utility += 1
            # avg_util = sum_utility / num_utility
            # self.pattern_outcomes[outcome] = avg_util

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
        # if negotiator_id != self.last_neg and self.last_neg != '':
        # self.agreements.append(self.last_proposal)
        """Generate a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        # print('hi1', self.current_neg_index, self.finished_negotiators)
        self._update_agreements_if_needed()

        self.cur_state = state
        negotiator, cntxt = self.negotiators[negotiator_id]
        nmi = negotiator.nmi
        level = self._get_progress(negotiator_id)
        ufun: SideUFun = self.ufun  # cntxt["ufun"]
        if not is_edge_agent(self):
            ufun: CenterUFun = self.ufun  # cntxt['ufun']
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
        return best_outcome

    def respond(self, negotiator_id, state, source=None):
        if negotiator_id.startswith("s"):
            pass
        """Respond to a proposal in the negotiation."""
        # Check if negotiation has ended and update strategy
        # print('hi', self.current_neg_index, self.finished_negotiators)
        self._update_agreements_if_needed()

        # If no offer, reject
        self.cur_state = state
        if state.current_offer is None:
            # print(f'{self.id} responds REJECT')
            return ResponseType.REJECT_OFFER
        # print(f'{self.id} recieves {state.current_offer}')
        negotiator, cntxt = self.negotiators[negotiator_id]
        nmi = negotiator.nmi
        level = self._get_progress(negotiator_id)
        ufun: SideUFun = self.ufun  # cntxt["ufun"]
        if not is_edge_agent(self):
            ufun: CenterUFun = self.ufun  # cntxt['ufun']
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
        agent_type_factor = 1 if is_edge_agent(self) else 1

        # Variance adjustment — higher std => lower z
        std_utility = numpy.std(all_utilities)
        std_utility = max(std_utility, 1e-5)  # avoid division by zero

        # Normalize std_utility against mean to make it scale-invariant
        std_ratio = std_utility / (numpy.mean(all_utilities) + 1e-5)
        # print(agent_type_factor)
        base_z = agent_type_factor * 3 * (1 - progress)

        # Final z: reduced further as variance increases (e.g., z ~ 1/std)
        z = base_z / (1 + 5 * (std_ratio))  # 20 is a tuning hyperparameter
        # z = max(-5, z)
        # if negotiator_id.startswith('e3'):
        #     print(negotiator_id, ':  received', current_offer, 'at', level, 'utility', offer_utility, 'mean', (mean_utility ), 'best_utility',  best_utility)
        if not is_edge_agent(self):
            pass
        if offer_utility > (mean_utility + (z * std_utility)):
            return ResponseType.ACCEPT_OFFER
        if (offer_utility / best_utility) < 0.15:
            # print('good enough')
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _update_agreements_if_needed(self):
        """Update the agreements list if a negotiation has ended."""
        if did_negotiation_end(self):
            # print('yo')
            # Store the agreement from the just-ended negotiation
            # self.current_neg_index = int(self.current_neg_index[1])
            prev_index = self.current_neg_index - 1
            # print(self.current_neg_index)
            if prev_index >= 0:
                agreement = get_agreement_at_index(self, prev_index)
                # print(self.current_neg_index)
                # print(self.agreements)
                while len(self.agreements) <= prev_index:
                    # print('APPEND_NONE')
                    self.agreements.append(None)
                # print('APPEND_PREV')
                self.agreements[prev_index] = agreement
                # print(self.agreements)
                return True
        return False
