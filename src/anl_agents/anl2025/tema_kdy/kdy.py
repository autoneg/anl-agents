"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Tokyo University of Agriculture and Technology
*Authors* Yuji Kobayashi, yuji3koba@gmail.com

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from negmas.outcomes import Outcome

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    build_sorted_bid_util_dict,
    all_possible_bids_with_agreements_fixed,
    get_propose_bid_at_current_index,
    get_estimate_total_steps,
    get_accept_bid_at_current_index,
    job_hunt_get_propose_bid_at_current_index,
    job_hunt_get_accept_bid_at_current_index,
)
# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
# from helpers.helperfunctions import set_id_dict, ...

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

from collections import defaultdict

from .opponent_model import SimpleOpponentModel  # Simple Oppnent Modelをインポート

from anl2025.ufun import MaxCenterUFun

__all__ = ["KDY"]


class KDY(ANL2025Negotiator):
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

        # 譲歩曲線のパラメータ
        self.e = 0.5

        # 戦略変更を行うトリガーの時刻
        self.propose_t_triger1 = 0.75
        self.respond_t_triger1 = 0.75

        # 提案戦略におけるフェーズごとの効用閾値
        self.propose_early_phase_util_threshold = 0.85
        self.propose_late_phase_util_threshold = 0.50

        # 受入戦略におけるフェーズごとの閾値
        self.accept_early_phase_util_threshold = 0.85
        self.accept_late_phase_util_threshold = 0.50

        # 提案戦略におけるフェーズごとの効用閾値
        self.for_job_late_phase_util_threshold = 0.70

        # センターエージェントにおける受入戦略でのインデックスごとの時間の閾値
        self.t_center_accept_upper_threshold = 0.90
        self.t_center_accept_lower_threshold = 0.80  # change from 0.80 to 0.90

        # job hunt用 センターエージェントにおける受入戦略でのインデックスごとの時間の閾値
        self.for_job_hunting_t_center_accept_upper_threshold = 0.90
        self.for_job_hunting_t_center_accept_lower_threshold = (
            0.90  # change from 0.80 to 0.90
        )

        # センターエージェントにおける受入戦略でのインデックスごとのステップ数の閾値
        self.step_center_bigger_rest_threshold = 10  # change from 10 to 5
        self.step_center_smaller_rest_threshold = 5

        # job hunt用 センターエージェントにおける受入戦略でのインデックスごとのステップ数の閾値
        self.for_job_hunting_step_center_bigger_rest_threshold = 5
        self.for_job_hunting_step_center_smaller_rest_threshold = 5

        # 合計ステップ数を調べる際に使用するフラグ
        self.search_sum_steps_flag = 0

        # 推定合計ステップ数を格納する変数
        self.estimated_total_steps = 0

        # センターエージェントの提案戦略に用いる重み係数
        self.alpha = 0.2  # change from 0.1 to 0.2

        # 簡易頻度モデルにて提示する上位n個のbid
        self.top_k = 8

        # opponent modelの格納を行う時刻tの閾値
        self.get_op_model_t_threshold = 0.7

        # ラストフェーズに行く際に、残り何ステップで移行するかを定義
        self.last_phase_step_num = 1

        # estimated total steps を得るためにひとまず必要な閾値
        self.step_switch_threshold = 0.55

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
        if did_negotiation_end(
            self
        ):  # 正規化時刻のrelative_timeと現時刻のスッテプ数を引数に渡す
            self._update_strategy()

        # 1回のみ合計ステップ数を調べて変数に格納
        if state.relative_time >= 0.5 and self.search_sum_steps_flag == 0:
            self.estimated_total_steps = get_estimate_total_steps(self, state)

        # 相手が今までに提案した上位n(self.top_k)個のbidを取得
        op_model_bids = self.op_model.most_frequent_bids(self.top_k)

        ##########################################################################
        if isinstance(self.ufun, MaxCenterUFun):  # job hunting系の場合
            bid = job_hunt_get_propose_bid_at_current_index(
                self, state.relative_time, state.step, op_model_bids, state
            )
        else:
            bid = get_propose_bid_at_current_index(
                self, state.relative_time, state.step, op_model_bids, state
            )
        ##########################################################################

        # bid = get_propose_bid_at_current_index(self, state.relative_time, state.step, op_model_bids, state)

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

        if state.relative_time >= self.get_op_model_t_threshold:
            # 毎ターン state を受け取って update
            self.op_model.update(state)

        ##########################################################################
        if isinstance(self.ufun, MaxCenterUFun):  # job hunting系の場合
            response = job_hunt_get_accept_bid_at_current_index(
                self, state.relative_time, state.step, state
            )
        else:
            response = get_accept_bid_at_current_index(
                self, state.relative_time, state.step, state
            )
        ##########################################################################

        # response = get_accept_bid_at_current_index(self, state.relative_time, state.step, state)

        if response == "Accept":
            return ResponseType.ACCEPT_OFFER
        elif response == "Reject":
            return ResponseType.REJECT_OFFER
        else:  # 例外処理用
            return ResponseType.REJECT_OFFER

    def _update_strategy(self) -> None:
        """現時点で有効なすべてのビッドとその効用を辞書にして返す (ただし、valueでソートされている)"""
        self.search_sum_steps_flag = 0  # 新しいサブエージェントとの交渉になった際に合計ステップ数を調べるフラグをリセット
        self.estimated_total_steps = (
            0  # 新しいサブエージェントとの交渉になった際に合計ステップ数をリセット
        )

        # 初期化
        self.op_model = SimpleOpponentModel()

        if is_edge_agent(self):  # エッジエージェントの場合
            outcomes = self.ufun.outcome_space.enumerate_or_sample(
                max_cardinality=10000
            )  # アウトカム空間を取得
            self.sorted_bid_util_dict = build_sorted_bid_util_dict(self, outcomes)

            self.target_bid = next(iter(self.sorted_bid_util_dict))
            # print(f"Edge agent target_bid: {self.target_bid}")

        else:  # センターエージェントの場合
            updated_outcomes = all_possible_bids_with_agreements_fixed(
                self
            )  # 可能なアウトカム空間を取得
            self.sorted_bid_util_dict = build_sorted_bid_util_dict(
                self, updated_outcomes
            )
            # print(f"Center agent sorted_bid_util_dict: {self.sorted_bid_util_dict}")

            self.target_bid = next(iter(self.sorted_bid_util_dict))
            # print(f"Center agent target_bid: {self.target_bid}")


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_tournament
    # Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    # Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    # results = run_negotiation(KDY)
    # visualize(results)

    # 最終的に集計する辞書（初期化）
    all_results_dict = defaultdict(float)

    for i in range(1):
        results = run_tournament(KDY)

    #     results_dict = {k:f'{v:.3f}' for k,v in sorted(results.weighted_average.items(), key=lambda x: x[1], reverse=True)}
    #     # print(f"Results for KDY: {results_dict}")
    #     for key, value in results_dict.items():
    #         all_results_dict[key] += float(value)

    # print("-----------------------------")
    # print("-----------------------------")
    # print("-----------------------------")
    # # 値で降順にソート
    # sorted_all_results_dict = sorted(all_results_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_all_results_dict)
