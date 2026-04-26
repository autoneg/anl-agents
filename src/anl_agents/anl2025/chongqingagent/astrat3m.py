"""
**Submitted to ANAC 2025 Automated Negotiation League**
*ChongqingAgent*
*1. Yunfei Wang <cqnuwyf@126.com>  ,2. Siqi Chen <siqichen@cqjtu.edu.cn>  *

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from negmas.outcomes import Outcome

from .helpers.concessionAnalyzer import ConcessionAnalyzer

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_current_negotiation_index,
    get_sorted_outcomes_with_utility,
    # edge_td3_model,
    deal_state,
    get_state,
    inverse_utility_with_outcomes,
    create_combined_list,
    edge_predict_is_high,
    select_from_top_5_percent,
    center_bid_select,
    center_find_bid_5best_of_one,
    based_compomise_select_bid,
)


from anl2025.negotiator import ANL2025Negotiator

from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

import numpy as np

__all__ = ["Astrat3m"]


class Astrat3m(ANL2025Negotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    This optimized agent uses a dynamic concession strategy and learns from the opponent's behavior.
    """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""

        self.current_neg_index = -1

        self.target_bid = None

        self.id_dict = {}

        set_id_dict(self)

        self.agent_name = "Astrat3m"
        self.sorted_utilities, self.my_Agent_max_u = None, None

        self.edge_partner_history = []
        self.edge_own_history = []
        self.exception_values = 0.92
        self.edge_high_scene = False

        self.edge_policy_loader, self.edge_svm_model = None, None
        self.edge_policy_loader, self.edge_svm_model = None, None

        self.select_bid_index = 0

        self.selectd_curbid_count = 0

        self.previous_bid_index = None
        self.center_bid_history = []
        self.center_exception_values = 0.9
        self.compromise_step = 0
        (
            self.center_sorted_utilities,
            self.center_my_Agent_max_u,
            self.check_compromise_bid_list,
        ) = None, None, None
        self.neg_index = get_current_negotiation_index(self)
        self.oppo_bid_history = []
        self.analyzer = ConcessionAnalyzer(threshold=0.03)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id)."""

        if did_negotiation_end(self):
            self._update_strategy()
        # bid = get_target_bid_at_current_index(self)
        bid = None

        # print(self.sorted_utilities, self.my_Agent_max_u)
        # sorted_utilities, my_Agent_max_u = self.sorted_utilities, self.my_Agent_max_u

        if is_edge_agent(self):
            if self.sorted_utilities == None:
                self.sorted_utilities, self.my_Agent_max_u = (
                    get_sorted_outcomes_with_utility(self)
                )

            # print(sorted_utilities, my_Agent_max_u)

            if self.edge_high_scene == False:
                if state.relative_time >= 1:
                    self.edge_high_scene = True

                if (
                    len(self.edge_own_history) > 10
                    and len(self.edge_partner_history) > 10
                    and state.relative_time >= 1
                ):
                    combined_listcombined_list = create_combined_list(
                        self, self.edge_own_history, self.edge_partner_history
                    )

                    is_high_result = False
                    try:
                        is_high_result = edge_predict_is_high(
                            self, combined_listcombined_list
                        )
                    except Exception:
                        self.edge_high_scene = False

                    if is_high_result:
                        self.edge_high_scene = True

            action_utility = 1.0
            deal_history = deal_state(
                self, self.edge_own_history, self.edge_partner_history
            )

            if len(deal_history) > 0:
                if self.edge_high_scene == True:
                    try:
                        current_state = get_state(self, deal_history)
                        action_utility = (
                            self.edge_policy_loader.get_action(current_state)[0] + 0.05
                        )

                        if action_utility <= 0.15:
                            action_utility = 0.25
                        if np.mean(self.edge_partner_history[-1]) >= 0.7:
                            action_utility += 0.08
                            if action_utility < self.edge_partner_history[-1]:
                                action_utility = self.edge_partner_history[-1] + 0.2
                    except Exception:
                        action_utility = select_from_top_5_percent(
                            self, self.sorted_utilities
                        )

                else:
                    action_utility = select_from_top_5_percent(
                        self, self.sorted_utilities
                    )

            action_utility = min(1.0, action_utility)

            self.edge_own_history.append(float(action_utility))

            bid = inverse_utility_with_outcomes(
                self, action_utility, self.sorted_utilities, self.my_Agent_max_u
            )

        else:
            # bid = None
            if self.neg_index != get_current_negotiation_index(self):
                (
                    self.sorted_utilities,
                    self.my_Agent_max_u,
                    self.check_compromise_bid_list,
                ) = get_sorted_outcomes_with_utility(self)
                # self._update_strategy()
                self.neg_index = get_current_negotiation_index(self)

            if self.sorted_utilities == None:
                (
                    self.sorted_utilities,
                    self.my_Agent_max_u,
                    self.check_compromise_bid_list,
                ) = get_sorted_outcomes_with_utility(self)
                # print(self.sorted_utilities)

            # old_V
            # if state.relative_time >= 0.98:
            #     if self.selectd_curbid_count >= 4 and self.compromise_step <= 2:
            #         self.selectd_curbid_count = 0
            #         self.compromise_step += 1
            #         self.select_bid_index += 1
            #         if self.previous_bid_index == None:
            #             self.previous_bid_index = self.select_bid_index - 1
            #             # if self.compromise_step >= 3:
            #             #     bid = center_bid_select(self, self.select_bid_index-1)
            #             bid = center_bid_select(self,self.select_bid_index,self.sorted_utilities,  self.my_Agent_max_u)
            #
            #         else:
            #             bid = center_bid_select(self, 0,self.sorted_utilities,  self.my_Agent_max_u)
            # return bid

            if len(self.oppo_bid_history[-5:]) >= 5:
                for u in self.oppo_bid_history[-5:]:
                    # print(u)
                    self.analyzer.update_history(u)
                # print(self.oppo_bid_history[-5:])
                prob, checks = self.analyzer.get_last_step_concession_probability()
                # print(prob)

                bid = based_compomise_select_bid(
                    prob, self.check_compromise_bid_list, self.oppo_bid_history[-1]
                )
                # print(bid)
                # else:
                #     bid = center_bid_select(self, self.select_bid_index,self.sorted_utilities,  self.my_Agent_max_u)
                #     self.selectd_curbid_count += 1
                # bid = get_target_bid_at_current_index(self)
                # bid = None
                # all_possible_bids = all_possible_bids_with_agreements_fixed(self)
                # center_get_current_id_bids(self)
                # print(curr)
                if bid not in self.center_bid_history:
                    self.center_bid_history.append(bid)
            else:
                # bid = center_bid_select(self, self.select_bid_index)

                bid = center_find_bid_5best_of_one(self, self.sorted_utilities)

            if bid not in self.center_bid_history:
                self.center_bid_history.append(bid)

        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """Responds to the given partner (source) using the side negotiator (negotiator_id)."""

        if did_negotiation_end(self):
            self._update_strategy()

        oppo_offer = state.current_offer

        if is_edge_agent(self):
            # print(self.ufun.reserved_value, self.edge_own_history[-1:], self.ufun(state.current_offer), state.relative_time, oppo_offer)
            if oppo_offer is None:
                return ResponseType.REJECT_OFFER

            if self.sorted_utilities == None:
                self.sorted_utilities, self.my_Agent_max_u = (
                    get_sorted_outcomes_with_utility(self)
                )
            sorted_utilities, my_Agent_max_u = (
                self.sorted_utilities,
                self.my_Agent_max_u,
            )
            self.edge_partner_history.append(
                (self.ufun(oppo_offer) + 0.00001) / (my_Agent_max_u + 0.00001)
            )

            if (self.ufun(oppo_offer) + 0.00001) / (
                my_Agent_max_u + 0.00001
            ) >= self.exception_values:
                return ResponseType.ACCEPT_OFFER

            if len(self.edge_own_history) >= 3 and len(self.edge_partner_history) >= 3:
                if (self.ufun(oppo_offer) + 0.00001) / (my_Agent_max_u + 0.00001) >= (
                    (
                        self.edge_own_history[-1]
                        + self.edge_own_history[-2]
                        + self.edge_own_history[-3]
                    )
                    / 3
                ) - 0.05:
                    return ResponseType.ACCEPT_OFFER

            # if len(self.edge_own_history) > 10 and len(self.edge_partner_history) > 10:
            #     combined_listcombined_list = create_combined_list(self, self.edge_own_history,self.edge_partner_history)
            #
            #     try:
            #         is_high_result = edge_predict_is_high(self, combined_listcombined_list)
            #         if state.relative_time >= 0.95 and self.ufun.reserved_value <= self.ufun(state.current_offer) and is_high_result:
            #
            #             print(is_high_result)
            #             recent_oppo_bids = self.edge_partner_history[-6:]
            #             recent_own_bids = self.edge_own_history[-6:]
            #             if len(set(recent_oppo_bids)) <= 1 and len(set(recent_own_bids)) <= 1:
            #                 return ResponseType.ACCEPT_OFFER
            #             else:
            #                 return ResponseType.REJECT_OFFER
            #     except Exception:
            #         return ResponseType.ACCEPT_OFFER

            if (
                state.relative_time >= 0.9
                and self.ufun.reserved_value is not None
                and self.ufun.reserved_value <= self.ufun(state.current_offer)
            ):
                recent_oppo_bids = self.edge_partner_history[-5:]
                # recent_own_bids = self.edge_own_history[-5:]

                # both_stable = len(set(recent_oppo_bids)) <= 1 and len(set(recent_own_bids)) <= 1
                both_stable = len(set(recent_oppo_bids)) <= 1

                if state.relative_time >= 0.93 and both_stable:
                    return ResponseType.ACCEPT_OFFER

                    # if state.relative_time >= 0.98:
                    # self.center_sorted_utilities, self.center_my_Agent_max_u,self.check_compromise_bid_list = None, None,None
                    return ResponseType.ACCEPT_OFFER

            if state.relative_time >= 0.98:
                # self.center_sorted_utilities, self.center_my_Agent_max_u,self.check_compromise_bid_list = None, None,None
                return ResponseType.ACCEPT_OFFER

        else:
            # print("111")

            # if state.current_offer is get_target_bid_at_current_index(self):
            #     return ResponseType.ACCEPT_OFFER
            if len(self.check_compromise_bid_list):
                index_ch_last = 0
                for index_ch, check_offer in enumerate(self.check_compromise_bid_list):
                    index_ch_last = index_ch
                    if oppo_offer == check_offer[1]:
                        self.oppo_bid_history.append(check_offer[0])
                        break
                if index_ch_last == len(self.check_compromise_bid_list) - 1:
                    self.oppo_bid_history.append(self.check_compromise_bid_list[-1][0])

            if self.sorted_utilities == None:
                (
                    self.sorted_utilities,
                    self.my_Agent_max_u,
                    self.check_compromise_bid_list,
                ) = get_sorted_outcomes_with_utility(self)

            # sorted_utilities, my_Agent_max_u = self.sorted_utilities, self.my_Agent_max_u

            if (
                state.current_offer in self.center_bid_history
                or state.current_offer
                == center_bid_select(
                    self,
                    self.select_bid_index + 1,
                    self.sorted_utilities,
                    self.my_Agent_max_u,
                )
            ):
                # if state.current_offer in self.center_bid_history:
                (
                    self.center_sorted_utilities,
                    self.center_my_Agent_max_u,
                    self.check_compromise_bid_list,
                ) = None, None, None
                return ResponseType.ACCEPT_OFFER

            # if find_bid_bigger_than_exception_valeue(self,state.current_offer):
            #     return ResponseType.ACCEPT_OFFER

            # if source not in self.opponent_history:
            #     self.opponent_history[source] = []
            # self.opponent_history[source].append(state.current_offer)

            if state.relative_time >= 0.99:
                (
                    self.center_sorted_utilities,
                    self.center_my_Agent_max_u,
                    self.check_compromise_bid_list,
                ) = None, None, None
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""

        if is_edge_agent(self):
            _, best_bid = self.ufun.extreme_outcomes()

        else:
            best_bid = None
            self.select_bid_index = 0
            self.selectd_curbid_count = 0
            self.previous_bid_index = None
            self.center_bid_history = []
            (
                self.center_sorted_utilities,
                self.center_my_Agent_max_u,
                self.check_compromise_bid_list,
            ) = None, None, None
            self.compromise_step = 0
            # print(self.center_sorted_utilities)

        self.target_bid = best_bid


if __name__ == "__main__":
    # runcode:
    # conda activate py311
    # set TF_ENABLE_ONEDNN_OPTS = 0
    # python -m myagent.astrat3m

    from .helpers.runner import run_a_tournament
    # from .helpers.test_runner import run_test

    # run_test(Astrat3m)
    #
    run_a_tournament(Astrat3m, small=True)
