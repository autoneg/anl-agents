"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team*
Team 273
*Authors*
Loes Peters: loes.peters@cwi.nl

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from collections import defaultdict
import random
from negmas.outcomes import Outcome

from .helpers.helperfunctions import (
    set_id_dict,
    get_current_negotiation_index,
    is_edge_agent,
    all_possible_bids_with_agreements_fixed,
    find_best_bid_in_outcomespace,
    get_nmi_from_index,
)
# Be careful: When running directly from this file, change the relative import to an absolute import.
# When submitting, use relative imports.
# from helpers.helperfunctions import (set_id_dict, did_negotiation_end, get_current_negotiation_index, \
#     get_agreement_at_index, get_outcome_space_from_index, get_number_of_subnegotiations, \
#     is_edge_agent, all_possible_bids_with_agreements_fixed, find_best_bid_in_outcomespace, \
#     get_target_bid_at_current_index, get_nmi_from_index, get_negid_from_index, )

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState
from negmas import ResponseType, PolyAspiration

__all__ = ["ProbaBot"]


class ProbaBot(ANL2025Negotiator):  # Possibly change this name later!
    """
    Your agent code. This is the ONLY class you need to implement.
    """

    _n = []
    _side_utility_dict = {}
    _prev_offers = []
    _seen_inequalities = []

    def init(self):
        """
        Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts.
        """

        # Initalize variables
        self.current_neg_index = -1

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of
        # the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        # Initialize inverse utility function for edge agent
        if is_edge_agent(self):
            self._inv = self.ufun.invert()
        # Initialize aspiration function
        self._asp = PolyAspiration(1.0, 3)

        # Initialize side utilities dictionary
        if not is_edge_agent(self):
            self._side_utility_dict = defaultdict(float)

    def on_negotiation_start(self, negotiator_id: str, state: SAOState) -> None:
        """
        Executed at the start of every (sub)negotiation.
        """

        nmi = get_nmi_from_index(self, get_current_negotiation_index(self))
        self._n = [
            nmi.issues[i].cardinality for i in range(len(nmi.issues))
        ]  # list containing the number of values per issue
        self._prev_offers = []
        self._seen_inequalities = []

        if not is_edge_agent(self):
            self._side_utility_dict = self.new_side_utility()

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """
        Proposes to the given partner (dest) using the side negotiator (negotiator_id).
        """

        previous_offers = self._prev_offers  # list of previous offers made

        if not previous_offers:  # no previous offers yet
            if is_edge_agent(self):  # edge agent
                outcome = find_best_bid_in_outcomespace(self)
            else:  # center agent
                utility_dict = self._side_utility_dict
                max_util = max(utility_dict.values())
                best_bids = [
                    key for key, val in utility_dict.items() if val > max_util - 1e-6
                ]
                outcome = random.choice(best_bids)
            self._prev_offers.append(outcome)
            return outcome

        if is_edge_agent(self):  # edge agent
            self._min, self._max = self._inv.minmax()
            a = (self._max - self._min) * self._asp.utility_at(
                state.relative_time
            ) + self._min
            possible_outcomes_not_final = self._inv.some(
                (a - 1e-6, self._max + 1e-6), False
            )
        else:  # center agent
            new_side_utilities = self._side_utility_dict
            self._max = max(new_side_utilities.values())
            self._min = min(new_side_utilities.values())

            if (
                new_side_utilities[None] >= self._max - 1e-6
                and new_side_utilities[None] > 0
            ):
                return None

            a = (self._max - self._min) * self._asp.utility_at(
                state.relative_time
            ) + self._min
            possible_outcomes_not_final = [
                bid for bid, value in new_side_utilities.items() if value >= a - 1e-6
            ]

        if not possible_outcomes_not_final:  # list of bids with utility >= a is empty
            if is_edge_agent(self):  # edge agent
                outcome = find_best_bid_in_outcomespace(self)
            else:  # center agent
                utility_dict = self._side_utility_dict
                max_util = max(utility_dict.values())
                best_bids = [
                    key for key, val in utility_dict.items() if val > max_util - 1e-6
                ]
                outcome = random.choice(best_bids)
            if outcome != None:
                for prev_offer in previous_offers:
                    inequality = []
                    for i in range(len(self._n)):
                        if outcome[i] == prev_offer[i]:
                            inequality.append(0)
                        else:
                            inequality.append((prev_offer[i], outcome[i]))
                    if inequality not in self._seen_inequalities:
                        self._seen_inequalities.append(inequality)
            self._prev_offers.append(outcome)
            return outcome

        possible_outcomes = [
            bid for bid in possible_outcomes_not_final if bid not in previous_offers
        ]

        if (
            not possible_outcomes
        ):  # tuple of bids not in previous_offers with utility >= a is empty
            possible_outcomes = possible_outcomes_not_final

        # which bid in possible_outcomes gives away least amount of information
        best_outcomes = []
        for bid in possible_outcomes:
            if bid != None:
                inequality = []
                for i in range(len(self._n)):
                    if bid[i] == previous_offers[-1][i]:
                        inequality.append(0)
                    else:
                        inequality.append((previous_offers[-1][i], bid[i]))
                if inequality in self._seen_inequalities:
                    best_outcomes.append(bid)

        if (
            not best_outcomes
        ):  # list of best outcomes (giving away least amount of info) is empty
            outcome = random.choice(possible_outcomes)
            if outcome != None:
                for prev_offer in previous_offers:
                    inequality = []
                    for i in range(len(self._n)):
                        if outcome[i] == prev_offer[i]:
                            inequality.append(0)
                        else:
                            inequality.append((prev_offer[i], outcome[i]))
                    self._seen_inequalities.append(inequality)
        else:
            outcome = random.choice(best_outcomes)

        self._prev_offers.append(outcome)
        return outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """
        Responds to the given partner (source) using the side negotiator (negotiator_id).
        """

        offer = state.current_offer

        if is_edge_agent(self):  # edge agent
            self._min, self._max = self._inv.minmax()
            a = (self._max - self._min) * self._asp.utility_at(
                state.relative_time
            ) + self._min

            if self.ufun.eval(None) >= self._max - 1e-6:
                return ResponseType.END_NEGOTIATION
            elif self.ufun.eval(offer) >= a:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        # center agent
        new_side_utilities = self._side_utility_dict
        self._max = max(new_side_utilities.values())
        self._min = min(new_side_utilities.values())
        a = (self._max - self._min) * self._asp.utility_at(
            state.relative_time
        ) + self._min

        if (
            new_side_utilities[None] >= self._max - 1e-6
            and new_side_utilities[None] > 0
        ):
            return ResponseType.END_NEGOTIATION
        elif new_side_utilities[offer] >= a:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def opponent_model(self, outcome) -> float:
        """
        Estimates opponent's utility function of outcome, where outcome is a complete outcome
        over all subnegotiations.
        """

        return 1

    def new_side_utility(self) -> dict:
        """
        Computes new side utilities based on how likely future outcomes are.
        Returns: a dictionary with for each outcome in the current negotiation a float
        indicating the expected achievable center utility (but probabilities are not normalized).
        """

        possible_bids = all_possible_bids_with_agreements_fixed(self)
        index = get_current_negotiation_index(self)

        side_utility_dict = defaultdict(float)

        for bid in possible_bids:
            if is_edge_agent(self):
                side_utility_dict[bid] += float(self.opponent_model(bid)) * float(
                    self.ufun.eval(bid)
                )
            else:
                side_utility_dict[bid[index]] += float(
                    self.opponent_model(bid)
                ) * float(self.ufun.eval(bid))

        return side_utility_dict


# If you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
# if __name__ == "__main__":
#     from helpers.runner import run_a_tournament
#     #Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
#     #Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!
#
#     run_a_tournament(ProbaBot, small=True, nologs=True)


# if __name__ == "__main__":
#     from anl2025 import anl2025_tournament, make_multideal_scenario
#     from anl2025.negotiator import Conceder2025, Boulware2025, Linear2025, Random2025
#     import pathlib
#     from anl2025.scenario import MultidealScenario
#     from Code_for_tutorials2025.Tutorial_visualization import visualize
#     from anl2025 import run_session
#     import time
#     from negmas.helpers import humanize_time
#     from agentloes_v0 import ProbaBotV0
#
#     centeragent = ProbaBot
#     edgeagents = [
#         Boulware2025,
#         Linear2025,
#         # Conceder2025
#         # Random2025
#         # ProbaBot,
#         ProbaBotV0,
#         Boulware2025,
#         # Boulware2025,
#     ]
#
#     path = pathlib.Path("../official_test_scenarios/TargetQuantity_example")
#     path = pathlib.Path("../official_test_scenarios/dinners")
#     path = pathlib.Path("../official_test_scenarios/job_hunt_target")
#
#     load_from_path = True
#     if load_from_path:
#         scenario = (MultidealScenario.from_file(path)
#                     if path.is_file()
#                     else MultidealScenario.from_folder(path))
#     else:
#         scenario = make_multideal_scenario(nedges=4, nissues=3, nvalues=3)
#
#     start = time.perf_counter()
#     results = run_session(
#         scenario=scenario,
#         center_type=centeragent,
#         edge_types=edgeagents,  # type: ignore
#         nsteps=20,
#         output=None
#     )
#     print(f"Finished in {humanize_time(time.perf_counter() - start)}")
#
#     print(f"Center utility: {results.center_utility}")
#     print(f"Edge Utilities: {results.edge_utilities}")
#     print(f"Agreement: {results.agreements}")
#
#     # visualize(results)


if __name__ == "__main__":
    from anl2025 import anl2025_tournament, make_multideal_scenario
    from anl2025.negotiator import Conceder2025, Boulware2025, Linear2025

    scenarios = [
        make_multideal_scenario(nedges=3, nissues=3, nvalues=5) for _ in range(1)
    ]
    competitors = [ProbaBot, Boulware2025, Linear2025, Conceder2025]

    results = anl2025_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_jobs=-1,
        path=None,
        n_repetitions=4,
        n_steps=20,
    )

    pass  # print(results.final_scores)
    pass  # print(f'Center agent: {results.final_scoresC}')
    pass  # print(f'Edge agents: {results.final_scoresE}')
