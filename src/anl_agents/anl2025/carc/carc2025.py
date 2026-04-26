"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* CARC
*Authors*
    Tianzi Ma:mtz982437365@gmail.com;
    Hongji Xiong:xionghj@stu.hit.edu.cn;
    Ruoke Wang:24S151159@stu.hit.edu.cn;
    Xuan Wang:wangxuan@cs.hitsz.edu.cn;
    Yulin Wu:yulinwu@cs.hitsz.edu.cn;

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from negmas.outcomes import Outcome

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_current_negotiation_index,
    get_outcome_space_from_index,
    get_number_of_subnegotiations,
    all_possible_bids_with_agreements_fixed,
    get_agreement_at_index,
)
# be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
# from helpers.helperfunctions import set_id_dict, ...

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

# from .opponent_tracker import OpponentTracker
from .alpha import AlphaMode
import math
import copy
import random

__all__ = ["CARC2025"]


class CARC2025(ANL2025Negotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    This example agent aims for the absolute best bid available. As a center agent, it adapts its strategy after each negotiation, by aiming for the best bid GIVEN the previous outcomes.
    """

    """
       The most general way to implement an agent is to implement propose and respond.
       """

    def init(self):
        """Executed when the agent is created. In ANL2025, all agents are initialized before the tournament starts."""
        ## # print("init")

        # Initalize variables
        self.current_neg_index = -1
        self.target_bid = None

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.get_details()

        # 存储当前情况下所有的最佳出价
        self.possible_best_bids = []

        # 字典  存储所有的出价对应的效用值
        self.outcome_util_dict = {}

        self.concede_index = 0

        # 当前index下对下一个人的出价的希望
        self.current_index_pref_list = []

        # # 超参数
        # self.config=SimpleNamespace(

        #     # 寻找最优解范围的误差
        #     FIND_BEST_BID_TOLERANCE__IN_EDGE    = 0.05,   # 边缘寻找最优时允许的误差
        #     FIND_BEST_BID_TOLERANCE__IN_CENTER  = 0.05,   # 中心寻找最优时允许的误差

        #     # propose的让步内容
        #     TIME_THRESHOLD__FOR_PROPOSE         = 0.9,    # propose从何时开始让步
        #     UTILITY_WEIGHT__ALGO = lambda u: round(math.exp(5*u),3), # 随机选时的权重算法

        #     # respond center中的让步内容
        #     TIME_THRESHOLD__FOR_RESPOND_CENTER  = 0.9,    # respond center从何时开始让步
        #     CONCEDE__FOR_LINEAR = lambda x: 1.0-x    ,    # linear中 对weight较大的 如何让步

        #     # respond edge中的让步内容
        #     TIME_THRESHOLD__FOR_RESPOND_EDGE    = 0.9,    # respond edge从何时开始让步
        #     LOOSE__MIN_U__FOR_EDGE              = 0.60,   # min_u是max_u的多少倍
        #     MYID__ALGO = lambda x:(2/math.pi)*math.atan(x), # 将自己的自然数id映射到[0,1)
        #     CONCEDE_FACTOR__FOR_EDGE            = 0.5,    # 让步因子

        #     # center随时间放宽下限的内容
        #     LOOSE__MIN_U                        = 0.60,   # min_u是max_u的多少倍
        #     RELATIVE_SUBNEG__ALGO = lambda x: 1-0.5*x,    # 第几个sub neg对谈判的影响
        #     CONCEDE_FACTOR                      = 0.20,   # 让步因子

        # )

        self.opponent_pref_list = []

        self.op_bids_record = []
        self.my_bids_record = []

    def is_center(self) -> bool:
        """我是中心谈判者吗？"""
        return not is_edge_agent(self)

    def is_edge(self) -> bool:
        """我是边缘谈判者吗？"""
        return is_edge_agent(self)

    def get_details(self):
        return

    def get_ufun_name(self):
        return self.ufun.__class__.__name__

    def get_relative_subnegs(self):
        # 谈完多少子谈判了？ [0,1]
        assert self.is_center()
        index = get_current_negotiation_index(self)
        nums = get_number_of_subnegotiations(self)
        return float(index) / float(nums)

    def __get_all_best_bids_im_edge(self):
        # 当我是边缘代理...
        updated_outcomes = all_possible_bids_with_agreements_fixed(self)

        # 允许误差

        all_bests = []
        utilities = []

        bids_dic = {bid: self.ufun(bid) for bid in updated_outcomes}

        maxvalue = max(bids_dic.values())

        tolerance = max(0.05, 0.25 * maxvalue)

        for bid, utility in bids_dic.items():
            if abs(utility - maxvalue) <= tolerance:
                all_bests.append(bid)

        def bid_priority(bid):
            return (-1.0 * self.ufun(bid),)

        sorted_bids = sorted(all_bests, key=bid_priority)
        self.edge_candidate_utility = [self.ufun(x) for x in sorted_bids]
        return sorted_bids

    def get_all_best_bids(self):
        # 如果自己是边缘
        if self.is_edge():
            return self.__get_all_best_bids_im_edge()

        # 如果自己是中心

        """
            STEP 1
            一次看多少步？-linear&maxcenter -> 1
                            -其他 -> 所有 并逐渐剪枝
        """
        # 如果自己是linear 或者maxcenter 每一步都只看一步就够
        if (
            self.get_ufun_name() == "LinearCombinationCenterUFun"
            or self.get_ufun_name() == "MaxCenterUFun"
        ):
            # if False:

            cbid = self.get_curr_bids()
            index = get_current_negotiation_index(self)

            self.outcome_util_dict = {}

            if self.get_ufun_name() == "LinearCombinationCenterUFun":
                for val in get_outcome_space_from_index(self, index):
                    # cbid2=copy.deepcopy(cbid)
                    cbid2 = [None] * get_number_of_subnegotiations(self)
                    cbid2[index] = val
                    self.outcome_util_dict[tuple(cbid2)] = self.ufun(cbid2)

            else:
                for val in get_outcome_space_from_index(self, index):
                    cbid2 = copy.deepcopy(cbid)
                    cbid2[index] = val
                    self.outcome_util_dict[tuple(cbid2)] = self.ufun(cbid2)

        else:
            if self.current_neg_index == 0:
                # 如果dict还是空的话(第一次谈判开始前) 暴力初始化
                updated_outcomes = all_possible_bids_with_agreements_fixed(self)
                self.outcome_util_dict = {
                    o: self.ufun(o) for o in updated_outcomes if o is not None
                }
            else:
                # 如果不是第一次（即我们已经有了self.outcome_util_dict） 就更新一下self.outcome_util_dict
                index = self.current_neg_index  # 当前正在谈判的编号
                curr_bids = self.get_curr_bids()  # 当前已经达成的协议
                prefix = tuple(curr_bids[:index])
                self.outcome_util_dict = {
                    bid: u
                    for bid, u in self.outcome_util_dict.items()
                    if bid[:index] == prefix
                }

        # # print(f"子谈判{self.current_neg_index}下，解空间大小为",len(self.outcome_util_dict))
        # # print("self.outcome_util_dict:",self.outcome_util_dict)

        """
            STEP 2 
            允许误差是多少
        """
        # outcome util dict --> all bests

        maxvalue = max(self.outcome_util_dict.values())

        # 允许误差 分类讨论
        if self.get_ufun_name() == "LinearCombinationCenterUFun":
            tolerance = max(0.05, 0.75 * maxvalue)
        elif self.get_ufun_name() == "MeanSMCenterUFun":
            if maxvalue < 1.0:
                tolerance = max(0.05, 0.25 * maxvalue)
            else:
                tolerance = max(0.05, 0.05 * maxvalue)
        elif self.get_ufun_name == "LambdaCenterUFun":
            tolerance = max(0.05, 0.10 * maxvalue)
        else:
            tolerance = max(0.05, 0.25 * maxvalue)

        """
            STEP 3
            候选集的选取
        """
        all_bests = []

        # 候选集 分类讨论
        if self.get_ufun_name() == "MaxCenterUFun":
            for bid, utility in self.outcome_util_dict.items():
                if abs(utility - maxvalue) <= tolerance and utility > self.ufun(
                    self.get_curr_bids()
                ):
                    all_bests.append(bid)
        elif self.get_ufun_name() == "LinearCombinationCenterUFun":
            # 改动：所有的都加进去

            this_weight = self.ufun._weights[get_current_negotiation_index(self)]
            # print(f"maxvalue {round(maxvalue,3)}, tolerance:{round(tolerance,3)}, this_weight={round(this_weight,3)}")
            for bid, utility in self.outcome_util_dict.items():
                # print(f"bid {bid}, utility:{round(utility,3)}")
                # # # print("用ufuns里面的",self.ufun.ufuns[index])
                # if abs(utility-maxvalue) / this_weight <=tolerance :
                #     all_bests.append(bid)
                all_bests.append(bid)
        else:
            for bid, utility in self.outcome_util_dict.items():
                # # print(bid,utility,maxvalue)
                if abs(utility - maxvalue) <= tolerance:
                    all_bests.append(bid)

        """
            STEP 4 
            将候选集排序
        """

        index = get_current_negotiation_index(self)

        def compute_bid_optimism_values(bid, index):
            """
            对于一个给定的 bid（至少到 index），计算：
            - pessimistic_value: 当前维度之后全为 None 时的效用
            - expected_value: 后缀填入所有可能出价后的平均效用
            - optimistic_value: 后缀填入所有可能出价后的最大效用
            """
            # 配置参数：过滤掉后缀效用过低的候选
            UTILITY_THRESHOLD_RATIO = 0.6

            prefix = tuple(bid[: index + 1])  # 当前子谈判位置已填

            # 特殊处理：如果是最后一个子谈判，则bid已经完整
            if index == get_number_of_subnegotiations(self) - 1:
                full_bid = tuple(bid)
                util = self.outcome_util_dict.get(full_bid, self.ufun(full_bid))
                return util, util, util

            matching_utilities = []
            pessimistic_value = None

            for full_bid, utility in self.outcome_util_dict.items():
                if full_bid[: index + 1] == prefix:
                    if all(x is None for x in full_bid[index + 1 :]):
                        # 找到悲观值
                        pessimistic_value = utility

                    # 将效用与阈值比较
                    partial_bid = tuple(full_bid[: index + 1])
                    partial_util = self.outcome_util_dict.get(partial_bid)
                    if partial_util is None:
                        partial_util = self.ufun(
                            list(partial_bid)
                            + [None] * (get_number_of_subnegotiations(self) - index - 1)
                        )
                    if utility >= UTILITY_THRESHOLD_RATIO * partial_util * 0:
                        matching_utilities.append(utility)

            # 没有任何匹配项的防御性处理
            if not matching_utilities:
                expected_value = 0.0
                optimistic_value = 0.0
            else:
                expected_value = sum(matching_utilities) / len(matching_utilities)
                optimistic_value = max(matching_utilities)

            # 有可能没有全None的情况，悲观值就用最低值
            if pessimistic_value is None:
                pessimistic_value = (
                    min(matching_utilities) if matching_utilities else 0.0
                )

            return pessimistic_value, expected_value, optimistic_value

        def bid_priority(bid):
            incomplete_utility = self.ufun_with_incomplete_bids(bid, index)

            # 当前以及未来维度的 None 数量（成功谈判越少越好）
            none_count = sum(1 for i in range(index, len(bid)) if bid[i] is None)

            # index+1 到末尾中 None 出现得越晚越好
            none_positions = [i for i in range(index + 1, len(bid)) if bid[i] is None]
            latest_none_index = max(none_positions) if none_positions else -1

            return (-incomplete_utility, -none_count, -latest_none_index)

        def bid_priority_for_lambda(bid):
            pess, exp, opt = compute_bid_optimism_values(bid, index)

            rel = self.get_relative_subnegs()
            w_pess = 0.5 * (1 - rel)
            w_exp = 0.5
            w_opt = 0.5 * rel

            # 加权总分（越大越好）
            score = w_pess * pess + w_exp * exp + w_opt * opt

            # 我们希望从高到低排序，因此取负
            return -score

        # ====== 排序逻辑 ======

        if False:
            # if self.get_ufun_name() == "LambdaCenterUFun":
            cbid = self.get_curr_bids()
            local_bids = []

            for val in get_outcome_space_from_index(self, index):
                bid = list(cbid)
                bid[index] = val
                bid = tuple(bid)
                local_bids.append(bid)

            # print(f'local_bids {local_bids}')
            # 计算得分并存储 (负的，因为排序时是负得分)
            sorted_bids = [(bid, -bid_priority_for_lambda(bid)) for bid in local_bids]

            # 排序
            sorted_bids.sort(key=lambda x: -x[1])  # 越大越好

            # 保存结果
            self.sorted_best_bids = [bid for bid, score in sorted_bids]
            self.sorted_best_bids_with_score = sorted_bids
            sorted_bids = self.sorted_best_bids

            # print("=== LambdaCenter 排序完毕，三值评估 ===")
            # print(f"{'Rank':<4} {'Bid':<35} {'悲观值':<8} {'期望值':<8} {'乐观值':<8} {'加权得分':<10}")
            for i, bid in enumerate(sorted_bids):  # 可加 [:20] 限制打印数量
                pess, exp, opt = compute_bid_optimism_values(bid, index)
                rel = self.get_relative_subnegs()
                w_pess = 0.5 * (1 - rel)
                w_exp = 0.5
                w_opt = 0.5 * rel
                score = w_pess * pess + w_exp * exp + w_opt * opt
                # print(f"{i+1:<4} {str(bid):<35} {pess:<8.3f} {exp:<8.3f} {opt:<8.3f} {score:<10.3f}")
            # print("=======================================")

        elif (
            self.get_ufun_name() == "LinearCombinationCenterUFun"
            or self.get_ufun_name() == "MeanSMCenterUFun"
            or self.get_ufun_name() == "MaxCenterUFun"
            or self.get_ufun_name() == "LambdaCenterUFun"
        ):
            # 对所有理想最优的结果进行排序，最悲观最优的靠前
            sorted_bids = sorted(all_bests, key=bid_priority)
            self.sorted_best_bids = sorted_bids
        else:
            sorted_bids = sorted(all_bests, key=bid_priority)
            self.sorted_best_bids = sorted_bids

        # 当前子谈判上的优先出价列表（用于 respond 阶段判断）
        seen = set()
        self.current_index_pref_list = []
        self.current_index_thresholds = []  # 新增：每个出价值对应的下限效用值

        for bid in sorted_bids:
            item = bid[index]
            if item not in seen:
                seen.add(item)
                self.current_index_pref_list.append(item)

                utility1 = self.ufun_with_incomplete_bids(
                    bid, index - 1
                )  # 这里特殊！！意思是当前谈判也是None的话
                utility2 = self.ufun_with_incomplete_bids(bid, index)
                self.current_index_thresholds.append([utility1, utility2])

        # if self.get_ufun_name() != 'LambdaCenterUFun':
        #     for bid in sorted_bids:
        #         item = bid[index]
        #         if item not in seen:
        #             seen.add(item)
        #             self.current_index_pref_list.append(item)

        #             utility1 = self.ufun_with_incomplete_bids(bid, index-1) #这里特殊！！意思是当前谈判也是None的话
        #             utility2 = self.ufun_with_incomplete_bids(bid, index)
        #             self.current_index_thresholds.append([utility1,utility2])
        # else:
        #     for bid,score in self.sorted_best_bids_with_score:
        #         item = bid[index]
        #         if item not in seen:
        #             seen.add(item)
        #             self.current_index_pref_list.append(item)

        #             utility1 = self.ufun_with_incomplete_bids(bid, index-1) #这里特殊！！意思是当前谈判也是None的话
        #             utility2 = score
        #             self.current_index_thresholds.append([utility1,utility2])

        if self.get_ufun_name() == "LinearCombinationCenterUFun":
            # 制作严格的候选集
            pref = [
                x[1] for x in self.current_index_thresholds
            ]  # pref是对应的存储utility的list
            top_util = self.current_index_thresholds[0][1]  # 当前最优出价的效用
            candidates = [
                (item, util)
                for item, util in zip(self.current_index_pref_list, pref)
                if util >= 0.60 * top_util
            ]
            # 为每个候选出价加入小扰动，用于打乱排序但保持总体趋势
            disturbed_candidates = []
            for item, util in candidates:
                # 添加小扰动（-0.05 ~ +0.05倍数）
                noise = util * random.uniform(-0.05, 0.05)
                score = util + noise
                disturbed_candidates.append((item, util, score))

            # 按扰动后的分数降序排列，提取前50个
            disturbed_candidates.sort(key=lambda x: x[2], reverse=True)
            selected_candidates = disturbed_candidates[:50]

            # 结果（保留真实效用值）以供后续使用
            self.index_top50_candidates = [
                (item, util) for item, util, _ in selected_candidates
            ]
            self.index_top50_pointer = 0
            # print(f" self.index_top50_candidates:{ self.index_top50_candidates}")

        # print(f"===================子谈判{get_current_negotiation_index(self)}=====")
        # # print("outcome_util_dict",self.outcome_util_dict)
        # print("误差",tolerance)
        # print("优先出价列表", self.current_index_pref_list)
        # print("对应效用下限列表：",[x[1] for x in self.current_index_thresholds] )
        # print("当前最理想的max：",maxvalue)
        # print("========================")

        # assert False

        return all_bests

    def get_curr_bids(self):
        # 工具函数 当前所有达成的bids的tuple  ， 当前正在谈的 是None
        index = get_current_negotiation_index(self)
        candidate = [None] * get_number_of_subnegotiations(self)

        for j in range(get_number_of_subnegotiations(self)):
            if j < index:
                candidate[j] = get_agreement_at_index(self, j)  # 已经谈完的

        return candidate

    def ufun_with_incomplete_bids(self, bid, index) -> float:
        """
        计算在未来子谈判全部失败（即后续 index+1 之后都为 None）情况下的效用。

        Parameters:
            bid (tuple): 完整的 bid
            index (int): 当前谈判索引
        Returns:
            float: 对应的效用值
        """

        # 替换 index+1 之后的值为 None，保留前面的值
        bid_list = list(bid)
        for i in range(index + 1, len(bid_list)):
            bid_list[i] = None

        # # print("^^^^^^^^^^^^^^^^^^^^^^^^")
        # # print(f"bid:{bid}, u:{self.ufun(bid)}")
        # # print(f"bid with none:{bid_list}, u:{self.ufun(tuple(bid_list))}")
        # # print("^^^^^^^^^^^^^^^^^^^^^^^^")

        return self.ufun(tuple(bid_list))

    def _propose_special_cases(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ):
        # 特殊情况

        # 如果使用的是 MaxCenterUFun，提议时要找到能让总效用增加的

        if len(self.current_index_thresholds) == 0:
            return None

        index = get_current_negotiation_index(self)

        # for bid,u in zip(self.current_index_pref_list,self.current_index_thresholds):
        #     # print(bid,u)
        # # print("------------------")

        # self.current_index_pref_list是我偏好的出价（有顺序的list）
        pref = [
            x[1] for x in self.current_index_thresholds
        ]  # pref是对应的存储utility的list

        top_util = self.current_index_thresholds[0][1]  # 当前最优出价的效用

        if top_util < self.ufun(self.get_curr_bids()):
            return None

        candidates = [
            (item, util)
            for item, util in zip(self.current_index_pref_list, pref)
            if util >= 0.95 * top_util
        ]

        if len(candidates) == 1:
            # print(f"[PROPOSE] role={'center' if self.is_center() else 'edge'}, bid={candidates[0][0]}, time={round(state.relative_time, 3)}")
            return candidates[0][0]

        # ========= 权重设置方式 =========
        # 使用 softmax-like 权重（增强区分度但保持概率）
        utility_weight = lambda u: round(math.exp(5 * u), 3)

        weights = [utility_weight(util) for (_, util) in candidates]
        total = sum(weights)
        probs = [w / total for w in weights]

        items = [item for (item, _) in candidates]
        chosen = random.choices(items, weights=probs, k=1)[0]
        # # print(f"From {candidates} \n\t choose {chosen}")
        # print(f"[PROPOSE] role={'center' if self.is_center() else 'edge'}, bid={chosen}, time={round(state.relative_time, 3)}")
        return chosen

    def _propose_when_linear(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ):
        index = get_current_negotiation_index(self)
        t = state.relative_time  # [0, 1]

        time_threshold = 0.9

        time_threshold2 = 0.5

        # 如果时间还早，直接出最优解
        if t <= time_threshold2:
            # 从linear的候选集
            if hasattr(self, "index_top50_candidates") and hasattr(
                self, "index_top50_pointer"
            ):
                if self.index_top50_pointer < len(self.index_top50_candidates):
                    offer = self.index_top50_candidates[self.index_top50_pointer][0]
                    self.index_top50_pointer += 1
                    return offer

            # self.current_index_pref_list是我偏好的出价（有顺序的list）
            pref = [
                x[1] for x in self.current_index_thresholds
            ]  # pref是对应的存储utility的list
            top_util = self.current_index_thresholds[0][1]  # 当前最优出价的效用

            if t < time_threshold2 / 2.0:
                candidates = [
                    (item, util)
                    for item, util in zip(self.current_index_pref_list, pref)
                    if util >= 0.85 * top_util and item is not None and util != 0.0
                ]
            else:
                candidates = [
                    (item, util)
                    for item, util in zip(self.current_index_pref_list, pref)
                    if util >= 0.65 * top_util and item is not None and util != 0.0
                ]

            # # print(f"candidates:{candidates}")
            if len(candidates) == 1:
                return candidates[0][0]

            # ========= 权重设置方式 =========
            utility_weight = lambda u: u
            # utility_weight = lambda u: math.log(u)

            weights = [utility_weight(util) for (_, util) in candidates]
            total = sum(weights)
            probs = [w / total for w in weights]

            items = [item for (item, _) in candidates]
            chosen = random.choices(items, weights=probs, k=1)[0]

            # # print(f"From {candidates} \n\t choose {chosen}")
            # print(f"[PROPOSE] role=centre, bid={chosen}, time={round(state.relative_time, 3)}")

            return chosen

        # 否则开始让步（从最优逐渐允许更差的出价）
        # 调用对手建模
        # if len(self.opponent_pref_list)==0:
        #     self.opponent_pref_list = \
        #         [self.opponent_tracker.predicted_oppo_ufun(x) for x in self.current_index_pref_list]
        if len(self.opponent_pref_list) == 0:
            self.opponent_pref_list = []
            index = get_current_negotiation_index(self)
            curr_bid = self.get_curr_bids()

            for x in self.current_index_pref_list:
                bid = curr_bid.copy()
                bid[index] = x
                try:
                    u = self.opponent_tracker.predicted_oppo_ufun(tuple(bid))
                    # # print(f"[DEBUG] predicted_oppo_ufun({tuple(bid)}) = {u:.4f}")
                except Exception:
                    # # print(f"[WARNING] Failed to evaluate predicted_oppo_ufun({tuple(bid)}): {e}")
                    u = 0.0
                self.opponent_pref_list.append(u)

        # alpha = math.exp(-2 * (t))  # 我方偏好权重，越大越保守
        alpha = 1 - 2 * math.sqrt(-1 * t * t + t)
        best_score = -1e9
        candidates = []
        scores = [
            alpha * o + (1 - alpha) * t[1]
            for t, o in zip(self.current_index_thresholds, self.opponent_pref_list)
        ]
        best_score = max(scores)

        # 找出接近最优分数的候选 bid 及其对应权重（得分）
        candidates_with_weights = [
            (bid, round(score, 4))
            for bid, score in zip(self.current_index_pref_list, scores)
        ]

        # 分别取出 bid 和对应权重
        candidates, weights = zip(*candidates_with_weights)

        # 按权重随机选择一个 bid
        if sum(weights) <= 0:
            chosen = random.choice(candidates)
        else:
            chosen = random.choices(candidates, weights=weights, k=1)[0]

        # print(f"From {candidates_with_weights} \n\t choose {chosen}")
        # print(f"[PROPOSE] role=centre, 让步中, bid={chosen}, time={round(state.relative_time, 3)}")

        return chosen

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

        # 边缘代理：直接提最优
        if is_edge_agent(self):
            chosen = random.choices(
                self.possible_best_bids, weights=self.edge_candidate_utility
            )[0]
            # return random.choice(self.possible_best_bids) if random.random()<0.5 else self.possible_best_bids[0]
            # print(f"[PROPOSE] role=edge, bid={chosen}, time={round(state.relative_time, 3)}")
            return chosen

        mybid = self._propose_center(negotiator_id, state, dest)
        self.my_bids_record.append(mybid)
        return mybid

    def _propose_center(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        # 特殊情况
        if self.get_ufun_name() == "MaxCenterUFun":
            special_case_output = self._propose_special_cases(
                negotiator_id, state, dest
            )
            return special_case_output

        # 如果是linear的话
        if self.get_ufun_name() == "LinearCombinationCenterUFun":
            return self._propose_when_linear(negotiator_id, state, dest)

        # ===== 中心代理提案逻辑 =====
        index = get_current_negotiation_index(self)
        t = state.relative_time  # [0, 1]

        time_threshold = 0.9

        time_threshold2 = 0.5

        # 如果时间还早，直接出最优解
        if t <= time_threshold2:
            # self.current_index_pref_list是我偏好的出价（有顺序的list）
            pref = [
                x[1] for x in self.current_index_thresholds
            ]  # pref是对应的存储utility的list
            top_util = self.current_index_thresholds[0][1]  # 当前最优出价的效用
            candidates = [
                (item, util)
                for item, util in zip(self.current_index_pref_list, pref)
                if util >= 0.95 * top_util
            ]

            if len(candidates) == 1:
                return candidates[0][0]

            # ========= 权重设置方式 =========
            # 使用 softmax-like 权重（增强区分度但保持概率）
            utility_weight = lambda u: round(math.exp(5 * u), 3)

            weights = [utility_weight(util) for (_, util) in candidates]
            total = sum(weights)
            probs = [w / total for w in weights]

            items = [item for (item, _) in candidates]
            chosen = random.choices(items, weights=probs, k=1)[0]

            # # print(f"From {candidates} \n\t choose {chosen}")

            # print(f"[PROPOSE] role=centre, bid={chosen}, time={round(state.relative_time, 3)}")

            return chosen

        # 否则开始让步（从最优逐渐允许更差的出价）

        # 调用对手建模
        # if len(self.opponent_pref_list)==0:
        #     self.opponent_pref_list = \
        #         [self.opponent_tracker.predicted_oppo_ufun(x) for x in self.current_index_pref_list]
        if len(self.opponent_pref_list) == 0:
            self.opponent_pref_list = []
            index = get_current_negotiation_index(self)
            curr_bid = self.get_curr_bids()

            for x in self.current_index_pref_list:
                bid = curr_bid.copy()
                bid[index] = x
                try:
                    u = self.opponent_tracker.predicted_oppo_ufun(tuple(bid))
                    # # print(f"[DEBUG] predicted_oppo_ufun({tuple(bid)}) = {u:.4f}")
                except Exception:
                    # # print(f"[WARNING] Failed to evaluate predicted_oppo_ufun({tuple(bid)}): {e}")
                    u = 0.0000000001
                self.opponent_pref_list.append(u)

        alpha = math.exp(-2 * (t))

        best_score = -1e9
        candidates = []

        scores = [
            alpha * t[1] + (1 - alpha) * o
            for t, o in zip(self.current_index_thresholds, self.opponent_pref_list)
        ]
        best_score = max(scores)

        # 找出接近最优分数的候选 bid 及其对应权重（得分）
        candidates_with_weights = [
            (bid, round(score, 4))
            for bid, score in zip(self.current_index_pref_list, scores)
        ]

        # 分别取出 bid 和对应权重
        candidates, weights = zip(*candidates_with_weights)

        # 按权重随机选择一个 bid
        if sum(weights) <= 0:
            chosen = random.choice(candidates)
        else:
            chosen = random.choices(candidates, weights=weights, k=1)[0]

        # print(f"From {candidates_with_weights} \n\t choose {chosen}")
        # print(f"[PROPOSE] role=centre, 让步中, bid={chosen}, time={round(state.relative_time, 3)}")

        return chosen

    def _respond_edge(self, state, offer):
        if self.ufun(offer) < self.reserved_value:
            return ResponseType.REJECT_OFFER

        # 边缘代理让步
        t = state.relative_time  # 当前时间 [0.0, 1.0]

        time_threshold = 0.9

        if t < time_threshold:
            if offer in self.possible_best_bids:
                # print(f"[RESPOND] edge | time={round(state.relative_time, 3)} | offer={offer} | decision=ACCEPT")
                return ResponseType.ACCEPT_OFFER
            else:
                # print(f"[RESPOND] edge | time={round(state.relative_time, 3)} | offer={offer} | decision=REJECT")
                return ResponseType.REJECT_OFFER

        t = (t - time_threshold) / (1 - time_threshold)

        # 我是第几个谈判的？ ps 我不知道总共有几个
        my_id = next(iter(self.id_dict)) + 1

        max_u = self.ufun(self.possible_best_bids[0])
        min_u = max_u * 0.6 * (2 / math.pi) * math.atan(my_id)

        threshold = min_u + (max_u - min_u) * (1 - pow(t, 0.5))

        # # print(self.ufun(offer),threshold,my_id,max_u,min_u,t)
        if self.ufun(offer) >= threshold:
            # print(f"[RESPOND] edge | time={round(state.relative_time, 3)} | offer={offer} u:{self.ufun(offer)} >=threshold{ threshold} | decision=ACCEPT")
            return ResponseType.ACCEPT_OFFER

        # print(f"[RESPOND] edge | time={round(state.relative_time, 3)} | offer={offer} u:{self.ufun(offer)} <threshold { threshold} | decision=REJECT")
        return ResponseType.REJECT_OFFER

    def _respond_center_when_linear(self, state, offer):
        t = state.relative_time  # [0,1]
        time_threshold = 0.9
        time_threshold2 = 0.5

        index = self.current_neg_index

        # current_index_pref_list
        cbid = [None] * get_number_of_subnegotiations(self)
        cbid[index] = offer
        offer_util = self.ufun(cbid)

        # 时间 < 0.5 时
        if t < time_threshold2:
            if offer_util >= self.current_index_thresholds[0][1] * (0.90):
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 效用高 decision=ACCEPT")
                return ResponseType.ACCEPT_OFFER
            else:
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 效用低 decision=REJECT")
                return ResponseType.REJECT_OFFER

        # 时间 < 0.9 时
        if t <= time_threshold:
            # 从90%让步到60%
            if offer_util >= self.current_index_thresholds[0][1] * (
                (51.0 - 30 * t) / 40.0
            ):
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 效用高 decision=ACCEPT")
                return ResponseType.ACCEPT_OFFER
            else:
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 效用低 decision=REJECT")
                return ResponseType.REJECT_OFFER

        # 时间 >= 0.9，指数让步接受更差的出价

        # ===== 指数让步部分 =====
        max_u = self.current_index_thresholds[0][1]
        threshold = self.threshold_loose_on_time(max_u, t, time_threshold)

        # 如果是线性加，我们要找到weights来看当前谈判是否重要 越重要越要让步
        this_weight = self.ufun._weights[get_current_negotiation_index(self)]
        threshold = threshold * (1.0 - this_weight)

        # print("-----------respond center linear 随时间让步中-----------------")
        # print("Received offer" ,offer, offer_util)
        # print("relative time",t)
        # print("threshold",threshold)
        # print(f"possible max:{self.current_index_thresholds[0]}, possible min:{self.current_index_thresholds[-1]}")
        # print("----------------------------")

        if offer_util >= threshold:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def _respond_center(self, state, offer):
        index = get_current_negotiation_index(self)

        # 特殊情况 如果使用的是 MaxCenterUFun，一旦某个子博弈达成了最大化效用的出价，该子博弈就已经“完成任务”，后续的子博弈如果不是效用最大值就应直接拒绝或不主动提议。
        if self.get_ufun_name() == "MaxCenterUFun" and self.is_center():
            index = get_current_negotiation_index(self)
            cbid = self.get_curr_bids()
            cbid2 = self.get_curr_bids()
            cbid2[index] = offer
            if self.ufun(cbid2) <= self.ufun(cbid):
                # 如果新协议不能增加总收益，则严格直接拒绝
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | decision=REJECT")
                return ResponseType.REJECT_OFFER

        # linear
        if self.get_ufun_name() == "LinearCombinationCenterUFun":
            return self._respond_center_when_linear(state, offer)

        # === 最后一轮时：直接基于完整效用排序判断是否接受 ===
        num_subnegs = get_number_of_subnegotiations(self)
        index = get_current_negotiation_index(self)

        # 构造完整 offer，判断当前效用
        offer_tuple = tuple(self.get_curr_bids())
        offer_list = list(offer_tuple)
        offer_list[index] = offer
        complete_offer = tuple(offer_list)
        offer_util = self.ufun(complete_offer)

        # 仅在最后一个 subneg 才触发全局接受判断
        if index == num_subnegs - 1:
            prefix = tuple(self.get_curr_bids()[:index])
            # print(f"[RESPOND-FINAL] prefix: {prefix}")

            # 找到所有与当前 prefix 匹配的完整 bid（只查一次）
            matched_bids = {
                bid: u
                for bid, u in self.outcome_util_dict.items()
                if bid[:index] == prefix
            }
            # # print("------------529--matched_bids------------------")
            # # print(f"matched_bids: {matched_bids}")

            if not matched_bids:
                # print("[RESPOND-FINAL] No matched bids found!")
                return ResponseType.REJECT_OFFER

            sorted_by_real_util = sorted(matched_bids.items(), key=lambda x: -x[1])
            utilities = [u for _, u in sorted_by_real_util]

            # 设置接受门槛（前 10% 为强接受，后期可扩展）
            top_k = max(1, int(len(utilities) * 0.10))  # 至少保留一个
            top_threshold = utilities[top_k - 1]  # 第 top_k 大的效用值

            # 可选：加入 relative time 控制
            try:
                t = state.relative_time
            except:
                t = 0.0

            # 动态接受比例控制（early：只接受 top 10%；late：接受 top 50%）
            if t < 0.9:
                threshold_util = top_threshold
            else:
                extended_top_k = max(1, int(len(utilities) * 0.50))
                threshold_util = utilities[extended_top_k - 1]

            # 输出调试信息
            # print(f"[RESPOND-FINAL] matched={len(matched_bids)} bids")
            # print(f"[RESPOND-FINAL] complete_offer={complete_offer}, util={offer_util:.4f}, threshold={threshold_util:.4f}, t={t:.2f}")

            if offer_util >= threshold_util:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

        t = state.relative_time  # [0,1]

        time_threshold = 0.9

        # 时间 < 0.9 时
        if t < time_threshold:
            # if offer in self.current_index_pref_list:
            #     # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 在我方偏好集中 decision=ACCEPT")
            #     # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
            #     return ResponseType.ACCEPT_OFFER
            # else:
            cbid = self.get_curr_bids()
            cbid[index] = offer
            if self.ufun(cbid) >= 0.95 * self.current_index_thresholds[0][1]:
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 不在我方偏好集中，但效用高 decision=ACCEPT")
                # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
                return ResponseType.ACCEPT_OFFER
            else:
                # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 不在我方偏好集中，且效用低 decision=REJECT")
                # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
                return ResponseType.REJECT_OFFER

            # if self.get_ufun_name() != "LambdaCenterUFun":
            #     if offer in self.current_index_pref_list:
            #         # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 在我方偏好集中 decision=ACCEPT")
            #         # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
            #         return ResponseType.ACCEPT_OFFER
            #     else:
            #         cbid=self.get_curr_bids()
            #         cbid[index]=offer
            #         if self.ufun(cbid)>=self.current_index_thresholds[-1][1]:
            #             # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 不在我方偏好集中，但效用高 decision=ACCEPT")
            #             # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
            #             return ResponseType.ACCEPT_OFFER
            #         else:
            #             # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 不在我方偏好集中，且效用低 decision=REJECT")
            #             # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
            #             return ResponseType.REJECT_OFFER
            # else:

            #     for bid,score in zip(self.current_index_pref_list,self.current_index_thresholds):
            #         if bid == offer:
            #             if score[1]>0.95*self.current_index_thresholds[0][1]:
            #                 # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | LAMBDA decision=ACCEPT")
            #                 return ResponseType.ACCEPT_OFFER
            #             else:
            #                 # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | LAMBDA decision=REJECT")
            #                 return ResponseType.REJECT_OFFER
            #     # print("LAMBDA: offer not in pref list?")
            #     return ResponseType.REJECT_OFFER

        # 时间 >= 0.9，指数让步接受更差的出价

        # 如果在我们的备选list里面，直接成交？
        if offer in self.current_index_pref_list:
            # print(f"[RESPOND] center | time={round(state.relative_time, 3)} | offer={offer} | 在我方偏好集中 decision=ACCEPT")
            # print(f"[RESPOND2] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
            return ResponseType.ACCEPT_OFFER

        # ===== 指数让步部分 =====
        max_u = self.current_index_thresholds[0][1]
        threshold = self.threshold_loose_on_time(max_u, t, time_threshold)

        try:
            pos = self.current_index_pref_list.index(offer)
            offer_util = self.current_index_thresholds[pos][1]
        except ValueError:
            # 不在列表里，计算效用
            candidate = self.get_curr_bids()
            candidate[index] = offer
            offer_util = self.ufun(candidate)

        # print("-----------respond center 随时间让步中-----------------")
        # print("Received offer" ,offer, offer_util)
        # print("relative time",t)
        # print("threshold",threshold)
        # print(f"possible max:{self.current_index_thresholds[0]}, possible min:{self.current_index_thresholds[-1]}")
        # print(f"[RESPOND3] complete_offer={complete_offer}, util={offer_util:.4f}, t={t:.2f}")
        # print("----------------------------")

        if offer_util >= threshold:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

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

        offer = state.current_offer
        # # print("Receive offer:",offer)

        # if offer is None:
        #     return ResponseType.REJECT_OFFER

        t = state.relative_time  # 当前时间 [0.0, 1.0]

        if is_edge_agent(self):
            return self._respond_edge(state, offer)
        else:
            self.op_bids_record.append(offer)
            self.opponent_tracker.record_oppo_offer(offer, t)
            return self._respond_center(state, offer)
        # You can also return ResponseType.END_NEGOTIATION to end the negotiation.

    def threshold_loose_on_time(self, max_u, t, time_threshold):
        """
        根据时间适当放宽要求
        umax: 效用上限
        time: 相对时间[0,1]
        """
        # 目前方案

        # 最低效用阈值
        # 如果不是Linear的话随着整体谈判进行而逐渐降低 i.e. 前紧后松
        min_u = max_u * 0.6
        # if not self.get_ufun_name == "LinearCombinationCenterUFun":
        #     min_u = \
        #         min_u * (1-0.5*self.get_relative_subnegs())

        t = (t - time_threshold) / (1 - time_threshold)

        threshold = min_u + (max_u - min_u) * (1 - pow(t, 0.4))
        # threshold = min_u + (max_u - min_u) * ((1 - pow(t, 5)))
        return threshold

    def _update_strategy(self) -> None:
        """Update the strategy of the agent after a negotiation has ended."""

        if is_edge_agent(self):
            self.possible_best_bids = self.get_all_best_bids()
        else:
            # get the best bid from the outcomes that are still possible to achieve.
            # 只在get_all_best_bids函数中更新
            self.possible_best_bids = self.get_all_best_bids()

            # self.target_bid = self.possible_best_bids[0]
            self.concede_index = 0

            # 对下一个对手建模

            op_outcome_space = self.ufun._outcome_spaces[self.current_neg_index]
            op_issues = op_outcome_space.issues

            issue_value_info = {issue.name: issue.values for issue in op_issues}

            self.opponent_tracker = AlphaMode(
                my_ufun=self.ufun,
                our_name="Me",
                opp_name="Op",
                safe_d_path=None,
                repeat=None,
                opponent_ufun=None,
                negotiator_index=0,
                curr_neg_index=self.current_neg_index,
                issue_value_info=issue_value_info,
                op_issues=op_issues,
            )

            self.opponent_pref_list = []

            self.op_bids_record = []
            self.my_bids_record = []

            # # print(self.ufun.__dict__)
            # # print("WEIGHTS",self.ufun._weights)


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    # Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    # Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(CARC2025, small=False)
