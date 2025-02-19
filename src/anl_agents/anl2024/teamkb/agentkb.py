"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* TeamKB
*Authors* KOBAYASHI Yuji(y-kobayashi@katfuji.lab.tuat.ac.jp)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random
import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator

from negmas import pareto_frontier
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["AgentKB"]


class AgentKB(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()

    # ステップ数
    step = 0

    # 対戦相手の留保価格は0に設定
    partner_reserved_value = 0

    def on_preferences_changed(self, changes):
        """
        Called just after the ufun is set and before the negotiation starts.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        self.step = 0  # ステップ数の宣言

        self.m1 = 11
        self.n1 = 2
        self.m2 = 1.02
        self.n2 = 1.0

        self.para_a = 10
        self.para_rate = 1 / 7

        self.t_thre = 0.8

        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        # 有限の場合は「self.nmi.outcome_space.enumerate_or_sample()」にて留保価格以上の全てのBidを列挙
        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # 留保価格以上のBidにおけるパレートフロントをリストに格納
        self.ufuns = (self.ufun, self.opponent_ufun)
        self.frontier_utils, self.frontier_indices = pareto_frontier(
            self.ufuns, self.rational_outcomes
        )
        self.frontier_outcomes = [
            self.rational_outcomes[_] for _ in self.frontier_indices
        ]
        self.my_frontier_utils = [_[0] for _ in self.frontier_utils]
        self.best_offer__ = self.ufun.best()

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        # self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """

        self.step += 1

        offer = state.current_offer

        # self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer

        # 超妥協フェーズ
        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if self.step == nsteps__:
            if offer in self.rational_outcomes:
                com_val = self.make_threshold_depend_on_reservation(
                    m=self.m2, n=self.n2
                )
                if self.ufun(offer) >= com_val:
                    return True

        # 妥協フェーズ
        elif state.relative_time >= self.t_thre:
            if offer in self.frontier_outcomes:
                target_util = self.sigmoid_func(state.relative_time)
                if self.ufun(offer) > target_util:
                    return True

        # 高望みフェーズ
        elif state.relative_time < self.t_thre:
            if offer in self.frontier_outcomes:
                com_val = self.make_threshold_depend_on_reservation(
                    m=self.m1, n=self.n1
                )
                if self.ufun(offer) > com_val:
                    return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using parter_ufun, which is not used yet.

        # 超妥協フェーズ
        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if self.step == nsteps__:
            com_val = self.make_threshold_depend_on_reservation(m=self.m2, n=self.n2)

            choose_bid = self.get_target_bid(com_val, self.frontier_outcomes)

            if not choose_bid:
                choose_bid = self.best_offer__

            return choose_bid

        # 妥協フェーズ
        elif state.relative_time >= self.t_thre:
            # 一定の確率で効用値の高いBidを提案
            if random.random() < 0.2:
                sort_frontier_outcomes = self.sort_bids_reverse(self.frontier_outcomes)

                if len(sort_frontier_outcomes) > 3:
                    choose_bids = sort_frontier_outcomes[:3]
                else:
                    choose_bids = self.sort_bids_reverse(self.rational_outcomes)[:3]
                    if not choose_bids:
                        return self.best_offer__

                return random.choice(choose_bids)

            # 妥協したBidを提案
            else:
                target_util = self.sigmoid_func(state.relative_time)
                choose_bid = self.get_target_bid(target_util, self.frontier_outcomes)
                return choose_bid

        # 高望みフェーズ
        elif state.relative_time < self.t_thre:
            com_val = self.make_threshold_depend_on_reservation(m=self.m1, n=self.n1)

            sort_frontier_outcomes = self.sort_bids_reverse(self.frontier_outcomes)

            # エラー対策
            if not sort_frontier_outcomes:
                return self.best_offer__

            choose_bids = [_ for _ in sort_frontier_outcomes if self.ufun(_) > com_val]

            if not choose_bids:
                if len(sort_frontier_outcomes) > 3:
                    choose_bids = sort_frontier_outcomes[:3]
                else:
                    choose_bids = self.sort_bids_reverse(self.rational_outcomes)[:3]

            return random.choice(choose_bids)

    # 自身の留保価格に応じて妥協する閾値の決定
    def make_threshold_depend_on_reservation(self, m, n):
        rv = self.ufun.reserved_value
        com_val = (1 / m) * np.arctan(n * (rv - 1)) + 1
        return com_val

    # Bidを降順にソート
    # ソートしたいBidのリストを入力したら、ソートしたBidが出力される
    def sort_bids_reverse(self, bids):
        _rational = sorted(
            [
                (my_util, opp_util, _)
                for _ in bids
                if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                and (opp_util := float(self.opponent_ufun(_)))
                > self.opponent_ufun.reserved_value
            ],
            reverse=True,
        )

        sort_outcomes = []
        for i in range(len(_rational)):
            sort_outcomes.append(_rational[i][2])

        return sort_outcomes

    # ターゲットの効用値付近のBidを出力する
    def get_target_bid(self, target_util, bids):
        _rational = sorted(
            [
                (my_util, opp_util, _)
                for _ in bids
                if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                and (opp_util := float(self.opponent_ufun(_)))
                > self.opponent_ufun.reserved_value
            ],
        )

        for i in range(len(_rational)):
            if _rational[i][0] > target_util:
                return _rational[i][2]

        return None

    # シグモイド関数実装部分
    # 時刻を入力すると、妥協すべきBidの効用値を出力
    def sigmoid_func(self, t):
        high = self.make_threshold_depend_on_reservation(self.m1, self.n1)
        low = self.make_threshold_depend_on_reservation(self.m2, self.n2)

        h_thre = high
        l_thre = self.para_rate * high + (1 - self.para_rate) * low

        width = 1 - self.t_thre
        based_sigmoid = (
            -1 / (1 + np.exp(-self.para_a * ((1 / width) * (t - self.t_thre) - 0.5)))
            + 1
        )
        target_util = based_sigmoid * (h_thre - l_thre) + l_thre

        return target_util


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    # run_a_tournament(AgentKB, small=False)
    run_a_tournament(
        AgentKB,
        n_repetitions=5,
        n_outcomes=100,
        n_scenarios=5,
        debug=False,
        nologs=False,
        small=False,
    )
