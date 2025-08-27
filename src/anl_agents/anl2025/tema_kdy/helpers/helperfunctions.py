#### all functions below are hulp functions for the functions above. They are not necessary to implement the agent, but they can be useful." ####

from negmas.sao.controllers import SAOState
import itertools

# Add necessary imports
from negmas.outcomes import Outcome
import random

from collections import defaultdict

from anl2025.ufun import MaxCenterUFun


def set_id_dict(self):
    """Creates a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
    This dictionary allows us to find the right id for easy access to further information about the specific negotiation."""
    for id in self.negotiators.keys():
        index = self.negotiators[id].context['index']
        self.id_dict[index] = id


def did_negotiation_end(self):
    """The function finished_negotiators checks how many threads are finished. If that number changes, then the next negotiation has started."""
    if self.current_neg_index != len(self.finished_negotiators):
        self.current_neg_index = len(self.finished_negotiators)
        return True
    return False

def get_current_negotiation_index(self):
    """Returns the index of the current negotiation. An index 0 means the first negotiation in the sequence, index 1 the second, etc."""
    return len(self.finished_negotiators)

def get_agreement_at_index(self, index):
    """The nmi is the negotiator mechanism interface, available for each subnegotiation.
    Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids."""
    nmi = get_nmi_from_index(self, index)
    agreement = nmi.state.agreement
    # print(agreement)
    return agreement

def get_outcome_space_from_index(self, index):
    """This function returns the outcome space of the subnegotiation with the given index."""
    outcomes = self.ufun.outcome_spaces[index].enumerate_or_sample()
    #  print(outcomes)

    # Each subnegotiation can also end in disagreement aka outcome None. Therefore, we add this to all outcomes.
    outcomes.append(None)

    return outcomes

def get_number_of_subnegotiations(self):
    """Returns the total number of (sub)negotiations that the agent is involved in. For edge agents, this is 1."""
    return len(self.negotiators)

def cartesian_product(arrays):
    """This function returns the cartesian product of the given arrays."""
    cartesian_product = list(itertools.product(*arrays))
    #  print(cartesian_product)
    return cartesian_product

def is_edge_agent(self):
    """Returns True if the agent is an edge agent, False otherwise, then the agent is a center agent."""
    if self.id.__contains__("edge"):
        return True
    return False

def all_possible_bids_with_agreements_fixed(self):
    """ This function returns all the bids that are still possible to achieve, given the agreements that were made in the previous negotiations."""

    # Once a negotiation has ended, the bids in the previous negotiations cannot be changed.
    # Therefore, this function helps to construct all the bids that can still be achieved, fixing the agreements of the previous negotiations.

    # If the agent is an edge agent, there is just one bid to be made, so we can just return the outcome space of the utility function.
    # Watch out, the structure of the outcomes for an edge agent is different from for a center agent.

    if is_edge_agent(self):
        return self.ufun.outcome_space.enumerate_or_sample()

    possible_outcomes = []
    neg_index = get_current_negotiation_index(self)

    #As the previous agreements are fixed, these are added first.
    for i in range(neg_index):
        possible_outcomes.append([get_agreement_at_index(self,i)])
        # print(possible_outcomes)

    # The coming negotiations (with index higher than the current negotiation index) can still be anything, so we just list all possibilities there.
    n = get_number_of_subnegotiations(self)
    for i in range(neg_index, n):
        possible_outcomes.append(get_outcome_space_from_index(self,i))
    # print(possible_outcomes)

    #The cartesian product constructs the combinations of all possible outcomes.
    adapted_outcomes = cartesian_product(possible_outcomes)
    # print(adapted_outcomes)
    return adapted_outcomes

def find_best_bid_in_outcomespace(self):
    """Fixing previous agreements, this functions returns the best bid that can still be achieved."""
    # get outcome space with all bids with fixed agreements
    updated_outcomes = all_possible_bids_with_agreements_fixed(self)

    # this function is also in self.ufun.extreme_outcomes()
    mn, mx = float("inf"), float("-inf")
    worst, best = None, None
    for o in updated_outcomes:
        u = self.ufun(o)
        if u < mn:
            worst, mn = o, u # worstはbidで, mnは効用
        if u > mx:
            best, mx = o, u # bestはbidで, mxは効用
    # print(best)   
    return best


def get_target_bid_at_current_index(self):
    """ Returns the bid for the current subnegotiation, with the target_bid as source.
    """
    index = get_current_negotiation_index(self)
    #An edge agents bid is only one bid, not a tuple of bids. Therefore, we can just return the target bid.
    if is_edge_agent(self):
        return self.target_bid
    
    return self.target_bid[index]


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

#########################
# New Function
def for_job_hunting_center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold):
    phase = "Middle Phase"
    
    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 時間依存に基づいて、閾値を算出
    t = state.relative_time
    U_max = self.accept_early_phase_util_threshold
    U_min = self.for_job_late_phase_util_threshold # change

    if 1 - time_threshold == 0:
        high_threshold = U_min
    else:
        high_threshold = max(U_min, U_max - (U_max - U_min) * (((t - time_threshold)/(1 - time_threshold))**(1/self.e)))

    # あるインデックスのbidとその効用値の格納されたタプルのリストについて、閾値以上のもののみを抽出
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)

    # エラー対策
    if not high_bid_util_tuple_list:
        return one_of_most_bid[index]
    else:
        propose_bid = get_propose_bid_of_center_agent(self, high_bid_util_tuple_list)
        return propose_bid
#########################

#########################
# New Function
def job_hunt_get_propose_bid_at_current_index(self, relative_time, step, op_model_bids, state):
    """ Returns the bid for the current subnegotiation, with the target_bid as source.
    """
    one_of_most_bid = next(iter(self.sorted_bid_util_dict))

    # エッジエージェントの場合
    if is_edge_agent(self):
        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            propose_bid = edge_early_phase_propose_bid(self, one_of_most_bid)
            return propose_bid
        
        # estimate stepを受け取った後の処理
        else:
            # 残りn回になったらlate phaseへ突入
            # self.last_phase_step_num
            if step < (self.estimated_total_steps - self.last_phase_step_num):
                phase = "Early Phase"
                propose_bid = edge_early_phase_propose_bid(self, one_of_most_bid)
                return propose_bid

            # 残りn回以外なら、early phaseの作業を行う
            else:
                phase = "Last Phase"
                # 頻出モデルに基づいて、該当するbidの自身の効用値を計算し、効用値が大きい順で辞書を返したものを変数に格納
                sorted_bid_util_dict = build_sorted_bid_util_dict_from_op_model_bids(op_model_bids, self.ufun, self.top_k)
                high_utility_bids = filter_high_utility_bids(sorted_bid_util_dict, self.propose_late_phase_util_threshold)
                high_bid_util_tuple_list = list(high_utility_bids.items())
                
                # エラー対策
                if not high_bid_util_tuple_list:
                    return one_of_most_bid
                else:
                    # 頻度モデルにおけるトップいくつかのbidの中で(閾値以上)、最も自身の効用値の高いbidを取得し、return
                    best_bid_in_op_model = next(iter(high_utility_bids), one_of_most_bid)
                    return best_bid_in_op_model

    # センターエージェントの場合
    else:
        index = get_current_negotiation_index(self) # 現在の交渉インデックスを取得
        total_edge_agents = get_number_of_subnegotiations(self)

        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
            return propose_bid
        
        # estimate stepを受け取った後の処理
        else:
            # 推定ステップ数が100以下の場合は、時間をベースにmiddle phaseへ突入
            if self.estimated_total_steps <= 100:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.for_job_hunting_t_center_accept_upper_threshold
                lower = self.for_job_hunting_t_center_accept_lower_threshold 
                time_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))
                
                if state.relative_time < time_threshold: 
                    phase = "Early Phase"
                    propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
                    return propose_bid
                else:
                    phase = "Middle Phase"
                    propose_bid = for_job_hunting_center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold)
                    return propose_bid
            
            # 推定ステップ数が100より大きい場合は、ステップ数をベースにmiddle phaseへ突入
            else:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.for_job_hunting_step_center_bigger_rest_threshold
                lower = self.for_job_hunting_step_center_smaller_rest_threshold 
                step_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))

                rest_step = (self.estimated_total_steps - 1) -  step
                
                if rest_step > step_threshold: 
                    phase = "Early Phase"
                    propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
                    return propose_bid
                else:
                    phase = "Middle Phase"
                    time_threshold = (step + 1)/self.estimated_total_steps
                    propose_bid = for_job_hunting_center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold)
                    return propose_bid
#########################


#########################
# New Function
def job_hunt_get_accept_bid_at_current_index(self, relative_time, step, state):
    bid = state.current_offer

    # エッジエージェントの場合
    if is_edge_agent(self):
        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            if self.ufun(bid) >= self.accept_early_phase_util_threshold:
                return  "Accept"
            return "Reject"
        
        # estimate stepを受け取った後の処理
        else:
            # 残りn回になったらlate phaseへ突入
            # self.last_phase_step_num
            if step < (self.estimated_total_steps - self.last_phase_step_num):
                phase = "Early Phase"
                if self.ufun(bid) >= self.accept_early_phase_util_threshold:
                    return  "Accept"
                return "Reject"

            # 残りn回以外なら、early phaseの作業を行う
            else:
                phase = "Late Phase"
                if self.ufun(bid) >= self.accept_late_phase_util_threshold:
                    return "Accept"
                return "Reject"
    
    # センターエージェントの場合
    else:
        index = get_current_negotiation_index(self) # 現在の交渉インデックスを取得
        total_edge_agents = get_number_of_subnegotiations(self)
        bid = state.current_offer

        # estimate stepを受け取るまでの間はearly phaseでの作業を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            response = center_early_phase_response(self, index, bid)
            return response
        
        # estimate stepを受け取った後の処理
        else:
            # 推定ステップ数が100以下の場合は、時間をベースにmiddle phaseへ突入
            if self.estimated_total_steps <= 100:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.for_job_hunting_t_center_accept_upper_threshold
                lower = self.for_job_hunting_t_center_accept_lower_threshold 
                time_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))
                
                if state.relative_time < time_threshold: 
                    phase = "Early Phase"
                    response = center_early_phase_response(self, index, bid)
                    return response
                else:
                    phase = "Middle Phase"
                    response = for_job_hunting_center_middle_phase_response(self, index, state, time_threshold, bid)
                    return response
            
            # 推定ステップ数が100より大きい場合は、ステップ数をベースにmiddle phaseへ突入
            else:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.for_job_hunting_step_center_bigger_rest_threshold
                lower = self.for_job_hunting_step_center_smaller_rest_threshold 
                step_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))

                rest_step = (self.estimated_total_steps - 1) -  step
                
                if rest_step > step_threshold: 
                    phase = "Early Phase"
                    response = center_early_phase_response(self, index, bid)
                    return response
                else:
                    phase = "Middle Phase"
                    time_threshold = (step + 1)/self.estimated_total_steps
                    response = for_job_hunting_center_middle_phase_response(self, index, state, time_threshold, bid)
                    return response
#########################

#########################
# New Function
def for_job_hunting_center_middle_phase_response(self, index, state, time_threshold, bid):
    phase = "Middle Phase"
    
    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 時間依存に基づいて、閾値を算出
    t = state.relative_time
    U_max = self.accept_early_phase_util_threshold
    U_min = self.for_job_late_phase_util_threshold

    if 1 - time_threshold == 0:
        high_threshold = U_min
    else:
        high_threshold = max(U_min, U_max - (U_max - U_min) * (((t - time_threshold)/(1 - time_threshold))**(1/self.e)))

    # あるインデックスのbidとその効用値の格納されたタプルのリストについて、閾値以上のもののみを抽出
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)

    sorted_bid_count_dict = get_sorted_bid_count_dict(self, high_bid_util_tuple_list)

    if bid in sorted_bid_count_dict:
        return "Accept"
    return "Reject"
#########################

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

#########################
# New Function
def get_propose_bid_at_current_index(self, relative_time, step, op_model_bids, state):
    """ Returns the bid for the current subnegotiation, with the target_bid as source.
    """
    one_of_most_bid = next(iter(self.sorted_bid_util_dict))

    # エッジエージェントの場合
    if is_edge_agent(self):
        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            propose_bid = edge_early_phase_propose_bid(self, one_of_most_bid)
            return propose_bid
        
        # estimate stepを受け取った後の処理
        else:
            # 残りn回になったらlate phaseへ突入
            # self.last_phase_step_num
            if step < (self.estimated_total_steps - self.last_phase_step_num):
                phase = "Early Phase"
                propose_bid = edge_early_phase_propose_bid(self, one_of_most_bid)
                return propose_bid

            # 残りn回以外なら、early phaseの作業を行う
            else:
                phase = "Last Phase"
                # 頻出モデルに基づいて、該当するbidの自身の効用値を計算し、効用値が大きい順で辞書を返したものを変数に格納
                sorted_bid_util_dict = build_sorted_bid_util_dict_from_op_model_bids(op_model_bids, self.ufun, self.top_k)
                high_utility_bids = filter_high_utility_bids(sorted_bid_util_dict, self.propose_late_phase_util_threshold)
                high_bid_util_tuple_list = list(high_utility_bids.items())
                
                # エラー対策
                if not high_bid_util_tuple_list:
                    return one_of_most_bid
                else:
                    # 頻度モデルにおけるトップいくつかのbidの中で(閾値以上)、最も自身の効用値の高いbidを取得し、return
                    best_bid_in_op_model = next(iter(high_utility_bids), one_of_most_bid)
                    return best_bid_in_op_model

    # センターエージェントの場合
    else:
        index = get_current_negotiation_index(self) # 現在の交渉インデックスを取得
        total_edge_agents = get_number_of_subnegotiations(self)

        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
            return propose_bid
        
        # estimate stepを受け取った後の処理
        else:
            # 推定ステップ数が100以下の場合は、時間をベースにmiddle phaseへ突入
            if self.estimated_total_steps <= 100:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.t_center_accept_upper_threshold
                lower = self.t_center_accept_lower_threshold 
                time_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))
                
                if state.relative_time < time_threshold: 
                    phase = "Early Phase"
                    propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
                    return propose_bid
                else:
                    phase = "Middle Phase"
                    propose_bid = center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold)
                    return propose_bid
            
            # 推定ステップ数が100より大きい場合は、ステップ数をベースにmiddle phaseへ突入
            else:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.step_center_bigger_rest_threshold
                lower = self.step_center_smaller_rest_threshold 
                step_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))

                rest_step = (self.estimated_total_steps - 1) -  step
                
                if rest_step > step_threshold: 
                    phase = "Early Phase"
                    propose_bid = center_early_phase_propose_bid(self, index, one_of_most_bid)
                    return propose_bid
                else:
                    phase = "Middle Phase"
                    time_threshold = (step + 1)/self.estimated_total_steps
                    propose_bid = center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold)
                    return propose_bid
#########################


#########################
# New Function
def edge_early_phase_propose_bid(self, one_of_most_bid):
    phase = "Early Phase"

    high_utility_bids = filter_high_utility_bids(self.sorted_bid_util_dict, self.propose_early_phase_util_threshold)
    high_bid_util_tuple_list = list(high_utility_bids.items())

    # エラー対策
    if not high_bid_util_tuple_list:
        return one_of_most_bid
    else:
        propose_bid = get_propose_bid_of_edge_agent(self, high_bid_util_tuple_list)
        return propose_bid
#########################


#########################
# New Function
def get_propose_bid_of_edge_agent(self, high_bid_util_tuple_list):
    sorted_bid_count_dict = get_sorted_bid_count_dict(self, high_bid_util_tuple_list)

    # 確率に変換
    total = sum(sorted_bid_count_dict.values())
    prob_bid_count_dict = {k: v / total for k, v in sorted_bid_count_dict.items()}

    # 効用値に応じたbidの重みを計算したdictを作成
    weight_dict = calculate_weight_dict(self, high_bid_util_tuple_list)

    # 掛け算することで重みを加えたbidの確率分布をがvalueである辞書を作成
    add_weight_prob_bid_dict = {
        k: prob_bid_count_dict[k] * weight_dict.get(k, 1.0)
        for k in prob_bid_count_dict
    }

    # valueで降順にソート
    add_weight_prob_bid_dict = dict(
        sorted(add_weight_prob_bid_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # 合計を1に正規化
    total = sum(add_weight_prob_bid_dict.values())
    add_weight_prob_bid_dict = {
        k: v / total for k, v in add_weight_prob_bid_dict.items()
    }

    # キーとその確率をリストに変換
    keys = list(add_weight_prob_bid_dict.keys())
    weights = list(add_weight_prob_bid_dict.values())
    # ランダムに1つ選ぶ
    bid = random.choices(keys, weights=weights, k=1)[0]
    return bid
#########################


#########################
# New Function
def center_early_phase_propose_bid(self, index, one_of_most_bid):
    phase = "Early Phase"

    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 上記のタプルのリストについて、閾値以上のもののみを抽出
    high_threshold = self.accept_early_phase_util_threshold
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)
    # print(high_bid_util_tuple_list)

    # エラー対策
    if not high_bid_util_tuple_list:
        return one_of_most_bid[index]
    else:
        propose_bid = get_propose_bid_of_center_agent(self, high_bid_util_tuple_list)
        return propose_bid
#########################


#########################
# New Function
def center_middle_phase_propose_bid(self, index, one_of_most_bid, state, time_threshold):
    phase = "Middle Phase"
    
    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 時間依存に基づいて、閾値を算出
    t = state.relative_time
    U_max = self.accept_early_phase_util_threshold
    U_min = self.accept_late_phase_util_threshold

    if 1 - time_threshold == 0:
        high_threshold = U_min
    else:
        high_threshold = max(U_min, U_max - (U_max - U_min) * (((t - time_threshold)/(1 - time_threshold))**(1/self.e)))

    # あるインデックスのbidとその効用値の格納されたタプルのリストについて、閾値以上のもののみを抽出
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)

    # エラー対策
    if not high_bid_util_tuple_list:
        return one_of_most_bid[index]
    else:
        propose_bid = get_propose_bid_of_center_agent(self, high_bid_util_tuple_list)
        return propose_bid
#########################


#########################
# New Function
def get_propose_bid_of_center_agent(self, high_bid_util_tuple_list):
    sorted_bid_count_dict = get_sorted_bid_count_dict(self, high_bid_util_tuple_list)

    # 確率に変換
    total = sum(sorted_bid_count_dict.values())
    prob_bid_count_dict = {k: v / total for k, v in sorted_bid_count_dict.items()}

    # 効用値に応じたbidの重みを計算したdictを作成
    weight_dict = calculate_weight_dict(self, high_bid_util_tuple_list)

    # 掛け算することで重みを加えたbidの確率分布をがvalueである辞書を作成
    add_weight_prob_bid_dict = {
        k: prob_bid_count_dict[k] * weight_dict.get(k, 1.0)
        for k in prob_bid_count_dict
    }

    # valueで降順にソート
    add_weight_prob_bid_dict = dict(
        sorted(add_weight_prob_bid_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # 合計を1に正規化
    total = sum(add_weight_prob_bid_dict.values())
    add_weight_prob_bid_dict = {
        k: v / total for k, v in add_weight_prob_bid_dict.items()
    }

    # キーとその確率をリストに変換
    keys = list(add_weight_prob_bid_dict.keys())
    weights = list(add_weight_prob_bid_dict.values())
    # ランダムに1つ選ぶ
    bid = random.choices(keys, weights=weights, k=1)[0]
    return bid
#########################


#########################
# New Function
def get_accept_bid_at_current_index(self, relative_time, step, state):
    bid = state.current_offer

    # エッジエージェントの場合
    if is_edge_agent(self):
        # estimate stepを受け取るまでの間はearly phaseでの提案を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            if self.ufun(bid) >= self.accept_early_phase_util_threshold:
                return  "Accept"
            return "Reject"
        
        # estimate stepを受け取った後の処理
        else:
            # 残りn回になったらlate phaseへ突入
            # self.last_phase_step_num
            if step < (self.estimated_total_steps - self.last_phase_step_num):
                phase = "Early Phase"
                if self.ufun(bid) >= self.accept_early_phase_util_threshold:
                    return  "Accept"
                return "Reject"

            # 残りn回以外なら、early phaseの作業を行う
            else:
                phase = "Late Phase"
                if self.ufun(bid) >= self.accept_late_phase_util_threshold:
                    return "Accept"
                return "Reject"
    
    # センターエージェントの場合
    else:
        index = get_current_negotiation_index(self) # 現在の交渉インデックスを取得
        total_edge_agents = get_number_of_subnegotiations(self)
        bid = state.current_offer

        # estimate stepを受け取るまでの間はearly phaseでの作業を行う
        if state.relative_time < self.step_switch_threshold: 
            phase = "Early Phase"
            response = center_early_phase_response(self, index, bid)
            return response
        
        # estimate stepを受け取った後の処理
        else:
            # 推定ステップ数が100以下の場合は、時間をベースにmiddle phaseへ突入
            if self.estimated_total_steps <= 100:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.t_center_accept_upper_threshold
                lower = self.t_center_accept_lower_threshold 
                time_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))
                
                if state.relative_time < time_threshold: 
                    phase = "Early Phase"
                    response = center_early_phase_response(self, index, bid)
                    return response
                else:
                    phase = "Middle Phase"
                    response = center_middle_phase_response(self, index, state, time_threshold, bid)
                    return response
            
            # 推定ステップ数が100より大きい場合は、ステップ数をベースにmiddle phaseへ突入
            else:
                # インデックスごとに譲歩し出す時間を定義するので、その際用いる閾値を以下で計算
                upper = self.step_center_bigger_rest_threshold
                lower = self.step_center_smaller_rest_threshold 
                step_threshold = upper - (upper - lower) * (index / (total_edge_agents - 1))

                rest_step = (self.estimated_total_steps - 1) -  step
                
                if rest_step > step_threshold: 
                    phase = "Early Phase"
                    response = center_early_phase_response(self, index, bid)
                    return response
                else:
                    phase = "Middle Phase"
                    time_threshold = (step + 1)/self.estimated_total_steps
                    response = center_middle_phase_response(self, index, state, time_threshold, bid)
                    return response
#########################


#########################
# New Function
def center_early_phase_response(self, index, bid):
    phase = "Early Phase"

    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 上記のタプルのリストについて、閾値以上のもののみを抽出
    high_threshold = self.accept_early_phase_util_threshold
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)

    sorted_bid_count_dict = get_sorted_bid_count_dict(self, high_bid_util_tuple_list)

    if bid in sorted_bid_count_dict:
        return "Accept"
    return "Reject"
#########################


#########################
# New Function
def center_middle_phase_response(self, index, state, time_threshold, bid):
    phase = "Middle Phase"
    
    # センターエージェント用で、該当するインデックスのbidとその効用値のタプルのリストを取得
    for_center_bid_util_touple_list = get_bid_util_touple_list_for_center(self, index)

    # 時間依存に基づいて、閾値を算出
    t = state.relative_time
    U_max = self.accept_early_phase_util_threshold
    U_min = self.accept_late_phase_util_threshold

    if 1 - time_threshold == 0:
        high_threshold = U_min
    else:
        high_threshold = max(U_min, U_max - (U_max - U_min) * (((t - time_threshold)/(1 - time_threshold))**(1/self.e)))

    # あるインデックスのbidとその効用値の格納されたタプルのリストについて、閾値以上のもののみを抽出
    high_bid_util_tuple_list = filter_high_utility_tuple(self, for_center_bid_util_touple_list, high_threshold)

    sorted_bid_count_dict = get_sorted_bid_count_dict(self, high_bid_util_tuple_list)

    if bid in sorted_bid_count_dict:
        return "Accept"
    return "Reject"
#########################


#########################
def calculate_weight_dict(self, sorted_list):
    """
    valueで降順にソート済みのタプルリストを受け取り、
    各keyの平均順位に基づいて重み付き辞書を返す。

    Parameters:
        sorted_list (list of tuples): [('key', value), ...] （value降順でソート済み）
        alpha (float): 差分に対する重み係数（デフォルトは0.1）

    Returns:
        dict: keyに対する重み値の辞書
    """
    # 同順位を考慮したランク付け
    rank_dict = defaultdict(list)
    rank = 1
    prev_value = None
    actual_rank = 0

    for k, v in sorted_list:
        actual_rank += 1
        if v != prev_value:
            rank = actual_rank
            prev_value = v
        rank_dict[k].append(rank)

    # 平均順位の計算
    avg_rank_dict = {k: sum(ranks)/len(ranks) for k, ranks in rank_dict.items()}

    # 平均値との差分
    mean_rank = sum(avg_rank_dict.values()) / len(avg_rank_dict)
    diff_dict = {k: mean_rank - v for k, v in avg_rank_dict.items()}

    # 重み付け
    weight_dict = {k: 1 + self.alpha * diff for k, diff in diff_dict.items()}
    
    return weight_dict
#########################


#########################
# New Function
def get_bid_util_touple_list_for_center(self, index):
    # センターエージェント用の、全ての可能なbidの効用値を格納するリストを作成
    for_center_util_list = list(self.sorted_bid_util_dict.values())

    # センターエージェント用の、全ての可能なbidにおけるindexに格納されたbidを格納するリスト
    for_center_bid_list = []
    for_center_all_bid_list = list(self.sorted_bid_util_dict.keys())
    for i in range(len(for_center_all_bid_list)):
        for_center_bid_list.append(for_center_all_bid_list[i][index])

    # センターエージェント用の、あるインデックスにおける[(bid, util), (bid, util), ...]のリストを作成
    for_center_bid_util_touple_list = list(zip(for_center_bid_list, for_center_util_list))

    return for_center_bid_util_touple_list
#########################


#########################
# New Function
def filter_high_utility_tuple(self, for_center_bid_util_touple_list, threshold):
    high_bid_util_tuple_list = []
    for i in range(len(for_center_bid_util_touple_list)):
        if for_center_bid_util_touple_list[i][1] >= threshold:
            high_bid_util_tuple_list.append(for_center_bid_util_touple_list[i])
    
    return high_bid_util_tuple_list
#########################


#########################
# New Function
def get_sorted_bid_count_dict(self, high_bid_util_tuple_list):
    key_counts = {}
    for k, _ in high_bid_util_tuple_list:
        if k in key_counts:
            key_counts[k] += 1
        else:
            key_counts[k] = 1

    sorted_bid_count_dict = dict(sorted(key_counts.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_bid_count_dict
#########################


#########################
# New Function
def filter_high_utility_bids(bid_util_dict: dict, threshold: float) -> dict:
    """
    指定された閾値以上の効用値を持つ bid のみを抽出した辞書を返す。

    Args:
        bid_util_dict (dict): {bid: utility} 形式の辞書
        threshold (float): 閾値（これ以上の効用値のbidを抽出）

    Returns:
        dict: {bid: utility}形式の抽出済み辞書
    """
    return {
        bid: utility
        for bid, utility in bid_util_dict.items()
        if utility >= threshold
    }
#########################


#########################
# New Function
def build_sorted_bid_util_dict(self, outcomes) -> dict[Outcome, float]:
    """現時点で有効なすべてのビッドとその効用を辞書にして返す (ただし、valueでソートされている)
    """ 
    # 辞書に格納し、その後ソートする
    bid_util_dict = {}
    for outcome in outcomes:
        util = self.ufun(outcome)
        bid_util_dict[outcome] = util
    sorted_bid_util_dict = dict(sorted(bid_util_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_bid_util_dict
#########################


#########################
# New Function
def get_estimate_total_steps(self, state: SAOState):
    # 一回のみ合計ステップ数を調べて変数に格納
    self.search_sum_steps_flag += 1
    estimated_total_steps = int(state.step / state.relative_time) + 1
    # print(f"Estimated total steps: {estimated_total_steps}") # Debugging output to see the estimated total steps
    return estimated_total_steps
#########################


#########################
# New Function
def build_sorted_bid_util_dict_from_op_model_bids(op_model_bids, ufun, top_k):
    """
    頻出モデルに基づいて、該当するbidの自身の効用値を計算し、効用値が大きい順で辞書を返す関数

    Args:
        op_model_bids: List of tuples (bid, frequency)
        ufun: 効用関数（通常 self.ufun を渡す）
        top_k: 最大何個のbidを使うか（デフォルトは5）

    Returns:
        効用値で降順にソートされた {bid: utility} の辞書
    """
    top_bids = op_model_bids[:top_k]
    bid_util_dict = {bid: ufun(bid) for bid, _ in top_bids}
    sorted_bid_util_dict = dict(sorted(bid_util_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_bid_util_dict
#########################


# This function returns the nmi of the subnegotiation with the given index.
def get_nmi_from_index(self, index):
    """
    This function returns the nmi of the subnegotiation with the given index.
    The nmi is the negotiator mechanism interface per subnegotiation. Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids.
    """
    negotiator_id = get_negid_from_index(self,index)
    return self.negotiators[negotiator_id].negotiator.nmi

def get_negid_from_index(self, index):
    """ This function returns the negotiator id of the subnegotiation with the given index. """
    return self.id_dict[index]