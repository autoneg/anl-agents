"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
import itertools
import math
from negmas.outcomes import Outcome
import statistics
import random
import pdb

from helpers.helperfunctions import *
#be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOController, SAOState
from negmas import (
    DiscreteCartesianOutcomeSpace,
    ExtendedOutcome,
    ResponseType, CategoricalIssue,
)


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
        #print("init")

        #Initalize variables
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
        """Update the strategy of the agent after a negotiation has ended.
               """
        # if your current role is the edge agent, use the strategy as if your centeragent is in it's last subnegotiation.
        # In this case, just get the best bid from the utility function.
        if is_edge_agent(self):
            # note that the edge utility function has a slightly different structure than a center utility function.
            _, best_bid = self.ufun.extreme_outcomes()
        else:

            #get the best bid from the outcomes that are still possible to achieve.
            best_bid = find_best_bid_in_outcomespace(self)

        self.target_bid = best_bid
        #print(self.target_bid)


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
        #print("init")

        #Initalize variables
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
        ufun = self.ufun if is_edge_agent(self) else self.ufun.side_ufuns()[current_negotation_index]
        subnegotation_trace = nmi.trace

        total_steps = nmi.n_steps
        current_step = state.step # think it is originally 0 indexed

        outcome_space = ufun.outcome_space.enumerate_or_sample()
        reservation_value = ufun.reserved_value
        
        opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]

        util_of_best_opponent_bid = 0
        for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

        outcome_space = [x for x in outcome_space if ufun.eval(x) >= util_of_best_opponent_bid and ufun.eval(x) > reservation_value]
        outcome_space.sort(key = ufun.eval, reverse=True)
        
        # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
        step_percentage = current_step / (total_steps)

        if outcome_space:
            target_index = math.ceil(step_percentage * len(outcome_space))
            target_index = min(target_index, len(outcome_space)-1) # make sure it is not out of bounds in case step_percentage is 1
            bid = outcome_space[target_index]

        return bid

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:

        target_bid = None
        nmi = self.negotiators[negotiator_id][0].nmi
        current_negotation_index = get_current_negotiation_index(self)
        ufun = self.ufun if is_edge_agent(self) else self.ufun.side_ufuns()[current_negotation_index]
        subnegotation_trace = nmi.trace

        total_steps = nmi.n_steps
        current_step = state.step # think it is originally 0 indexed

        outcome_space = ufun.outcome_space.enumerate_or_sample()
        reservation_value = ufun.reserved_value

        opponent_bids = [x[1] for x in subnegotation_trace if x[0] != negotiator_id]

        if current_step == total_steps - 1: # last step
            if ufun.eval(state.current_offer) > reservation_value:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        outcome_space = ufun.outcome_space.enumerate_or_sample()

        util_of_best_opponent_bid = 0
        for opponent_bid in opponent_bids:
                if ufun.eval(opponent_bid) > util_of_best_opponent_bid:
                    util_of_best_opponent_bid = ufun.eval(opponent_bid)

        outcome_space = [x for x in outcome_space if ufun.eval(x) >= util_of_best_opponent_bid and ufun.eval(x) > reservation_value]
        outcome_space.sort(key = ufun.eval, reverse=True)
        
        # calculate the number of steps left in the subnegotatioon and pick from reduced outcome space accordingly (I think current_step is 0 indexed)
        step_percentage = (current_step+1) / (total_steps) # don't have to accept the best offer, willing to concede

        if outcome_space:
            target_index = math.ceil(step_percentage * len(outcome_space))
            target_index = min(target_index, len(outcome_space)-1)
            target_bid = outcome_space[target_index]

        target_util = ufun.eval(target_bid) if target_bid is not None else math.inf

        if ufun.eval(state.current_offer) >= target_util:
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER

class ConcedingAgent(ANL2025Negotiator):
    """
    """
    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        #print("init")

        #Initalize variables
        self.current_util = 0 # current realized util, hold across neg

        self.current_neg_index = -1
        self.current_ufun = None
        self.target_bid = None
        self.max_util = float('inf')

        self.best_oppo_bid = None
        self.best_oppo_bid_util = 0
        self.potential_bids = []
        self.potential_bids_util = []
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float('inf')
        self.next_bid_util = float('inf')
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

        perc_subnegs = float((len(self.finished_negotiators)+1) / len(self.negotiators))

        if self.debug:
            pass # print(perc_subnegs)
            pass # print(state.relative_time)

        # TODO: Concede differently for center and edge
        first_time_bid = self.current_bid_util == float('inf')
        self.current_bid_util = self.current_ufun.eval(self.potential_bids[self.conceding_index])

        if not first_time_bid and self.current_bid_util > self.best_oppo_bid_util and self.conceding_index < len(self.potential_bids)-1:
            self.next_bid_util = self.current_ufun.eval(self.potential_bids[self.conceding_index+1])
            if self.next_bid_util > self.best_oppo_bid_util:
                if self.current_bid_util - self.next_bid_util == 0:
                    util_diff_multiplier = 4
                else: 
                    util_diff_multiplier = min(4, float(self.util_difference_median / (self.current_bid_util - self.next_bid_util)))

                not_too_aggressive = float((self.conceding_index+1) / len(self.potential_bids)) < state.relative_time
                relative_standard = self.conceding_index-len(set(self.oppo_bids)) < 3
                #TODO: check on util normalization
                concede_degree = self.next_bid_util - max(self.current_util, self.best_oppo_bid_util)

                # NOTE: my very heuristic strategy
                if relative_standard and not_too_aggressive and \
                    state.relative_time * concede_degree * util_diff_multiplier > 0.5 * (1-perc_subnegs): #* self.current_util:
                    self.conceding_index += 1

                    if not is_edge_agent(self) and self.debug:
                        pass # print(is_edge_agent(self), perc_subnegs)
                        pass # print(state.relative_time)
                        pass # print("concede degree {}, self {}, oppo {}".format(concede_degree, self.conceding_index, len(set(self.oppo_bids))))
                        pass # print("{} Conceding to {}: util {} diff {} index {}".format(self, self.potential_bids[self.conceding_index], self.next_bid_util, self.current_bid_util - self.next_bid_util, self.conceding_index))

        # Last chance to propose (TODO: -1 for center; -2 for edge; or according to randomization)
        if (is_edge_agent(self) and state.step == nmi.n_steps-2) or \
            state.step == nmi.n_steps-1:
            temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0
            if self.best_oppo_bid_util > max(self.current_util, temp_reserve):
                if self.debug:
                    pass # print("{} ConcedingAgent LAST propose {}: util {}".format(self, self.best_oppo_bid, self.current_ufun.eval(self.best_oppo_bid)))
                return self.best_oppo_bid
            else:
                # return None
                return self.potential_bids[0]

        if self.debug:
            pass # print("{} ConcedingAgent propose {}".format(self, self.potential_bids[self.conceding_index]))
        
        self.current_bid_util = self.current_ufun.eval(self.potential_bids[self.conceding_index])
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
        if is_edge_agent(self) and state.step == nmi.n_steps-1:
            if  current_offer_util > max(self.current_util, self.current_ufun.reserved_value):
                if self.debug:
                    pass # print("{} ConcedingAgent respond last step: accept".format(self))
                return ResponseType.ACCEPT_OFFER
            else:
                if self.debug:
                    pass # print("{} ConcedingAgent respond last step: reject".format(self))
                return ResponseType.REJECT_OFFER
        
        self.oppo_bids.append(state.current_offer)

        if current_offer_util >= self.best_oppo_bid_util:
            self.best_oppo_bid_util = current_offer_util
            self.best_oppo_bid = state.current_offer
        
            if current_offer_util >= self.current_bid_util:
                if self.debug:
                    pass # print("{} ConcedingAgent respond: accept".format(self))
                return ResponseType.ACCEPT_OFFER
        
        if self.debug:
            pass # print("{} ConcedingAgent respond: reject".format(self))

        return ResponseType.REJECT_OFFER
    
    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended.
               """
        if self.current_neg_index > 0:
            last_agreement = get_agreement_at_index(self, self.current_neg_index-1)
            last_ufun = self.ufun.side_ufuns()[self.current_neg_index-1]
            last_neg_util = last_ufun.eval(last_agreement)
            self.current_util = max(last_neg_util, self.current_util)
            pass # print("LAST Agreement: {}, current_util {}".format(last_agreement, self.current_util))

        # reset
        self.best_oppo_bid_util = 0
        self.best_oppo_bid = None
        self.oppo_bids = []

        self.conceding_index = 0
        self.current_bid_util = float('inf')
        self.next_bid_util = float('inf')
        self.util_difference_median = 0

        # filter considered bids
        self.current_ufun = self.ufun if is_edge_agent(self) else self.ufun.side_ufuns()[self.current_neg_index]
        # NOTE: this is myopic, make sure that future neg wont decrease self.current_util
        # TODO: this eligible bids needs to be adapted based on utility function
        # TODO: We might need to treat reserve value differently for center and edge
        temp_reserve = self.current_ufun.reserved_value if is_edge_agent(self) else 0
        eligible_bids = [x for x in self.current_ufun.outcome_space.enumerate_or_sample() if self.current_ufun.eval(x) > max(self.current_util, temp_reserve)]
        eligible_bids.sort(key = self.current_ufun.eval, reverse=True) # TODO: myopic, needs to be sort by expected util from game tree
        self.potential_bids = eligible_bids
        self.potential_bids_util = [self.current_ufun.eval(x) for x in self.potential_bids]

        if self.current_neg_index == 0:
            self.max_util = self.potential_bids_util[0]

        if len(self.potential_bids) > 4: # NOTE: I chose some arbitrary threshold
            diffs = [self.potential_bids_util[i] - self.potential_bids_util[i+1] for i in range(len(self.potential_bids_util) - 2)]
            self.util_difference_median = statistics.median(diffs)
        else:
            self.util_difference_median = 0.005
        # pdb.set_trace()


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament
    #Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    #Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    from anl2025.scenario import MultidealScenario
    import pathlib
    from anl2025 import (
    make_multideal_scenario,
    make_dinners_scenario,
    make_job_hunt_scenario,
    make_target_quantity_scenario,
    run_session,
    load_example_scenario,
    anl2025_tournament,
    Boulware2025,
    Linear2025,
    )

    target_quantity_path = pathlib.Path("../official_test_scenarios/TargetQuantity_example")
    dinners_path = pathlib.Path("../official_test_scenarios/dinners")
    job_hunt_path = pathlib.Path("../official_test_scenarios/job_hunt_target")

    target_quantity = MultidealScenario.from_folder(target_quantity_path)
    dinners = MultidealScenario.from_folder(dinners_path)
    job_hunt = MultidealScenario.from_folder(job_hunt_path)

    # competitors = [Linear2025, Boulware2025]
    # results = run_session(center_type = ConcedingAgent, edge_types = competitors, scenario = job_hunt, verbose=True, nsteps=10)
    # print(f"Center Utility: {results.center_utility}\nEdge Utilities: {results.edge_utilities}")
    # print(f"Agreement: {results.agreements}")

    # pdb.set_trace()

    scenariosbig = (
        # [make_dinners_scenario() for _ in range(50)] 
        [make_job_hunt_scenario() for _ in range(50)] 
        # [make_target_quantity_scenario() for _ in range(50)] 
    )

    results = anl2025_tournament(
        scenariosbig, n_jobs=-1, competitors=(ConcedingAgent, Boulware2025, Linear2025), verbose=True
    )
    
    for key in results.final_scores: 
        pass # print(f"Agent name: {key} with center score: {results.final_scoresC[key] / results.center_count[key]}, edge score: {results.final_scoresE[key] / results.edge_count[key]}")
        
    pass # print(results.weighted_average)
