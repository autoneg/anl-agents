### 使用ufun7作为预测对手效用函数的算法
### 主要解决预测对手下一步出价的（1）我方效用函数下的（2）预测的对手效用函数下的 出价效用值
## 现在的问题是太乱了有很多贝叶斯都不一定能用到，所以要删一删



import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


from negmas.preferences import LinearAdditiveUtilityFunction, TableFun
from negmas.outcomes import CategoricalIssue
from pathlib import Path
import numpy as np
from negmas.preferences import pareto_frontier
import numpy as np
from scipy.interpolate import interp1d
import copy

class OpponentTracker:

    def __init__(self, my_ufun, our_name, opp_name, safe_d_path, repeat,opponent_ufun, negotiator_index,
                 curr_neg_index, # 这是第几个edge?
                 issue_value_info,
                 op_issues,
                 ):
        """
        初始化对手出价跟踪器
        :param my_ufun: 我方的效用函数（用于计算对手出价的效用值）
        """
 
        self.my_ufun = my_ufun
        self.opponent_offers = []  # 记录对手所有的出价
        self.opponent_utilities = []  # 记录每个出价在我方效用函数下的效用值
        self.self_offers = []  # 记录我方所有的出价
        self.self_utilities = []  # 记录我方出价在我方效用函数下的效用值

        self.last_predicted_oppo_utility = None  # 记录上一步预测的效用值
        self.predicted_opponent_utilities = []  # 记录所有预测的效用值
        self.predicted_opponent_utilities.append(self.last_predicted_oppo_utility)
        self.opponent_real_utilities = opponent_ufun # 对手的效用函数(真实）)
        self.current_real_oppo_utility = None # 记录当下对手真实的效用值
        self.predcited_offers = [] #记录预测效用值后反推出的offer集合
        self.history_real_next_offer_utility = []
        self.history_avg_predicted_offers_utility = []
        self.history_closest_offer_utility = []
        self.history_avg_diff = []
        self.history_closest_diff = []
        self.history_real_next_offer_utility.append(None)
        self.history_avg_predicted_offers_utility.append(None)
        self.history_closest_offer_utility.append(None)
        self.history_avg_diff.append(None)
        self.history_closest_diff.append(None)
        self.is_diverse_opponent = True
        self.offer_space = list(self.my_ufun.outcome_space.enumerate_or_sample())
        # 在 __init__ 中提前计算我方固定效用缓存
        # self.my_util_cache = {
        #     offer: float(self.my_ufun(offer))
        #     for offer in self.offer_space
        # }

        ###这部分放在预测那里写就好
        if negotiator_index == 0:
            ##我方是先手、对手是后手
            ##预测的时候 时序index是使用相同的
            pass 
        else:
            ##我方是后手、对手是先手 对手比我方多出一个
            ##预测的时候 对手回应的应该是我方Index-1的offer
            pass 
        


        # 存储这些变量以便命名文件
        self.our_name = our_name
        self.opp_name = opp_name
        self.safe_d_path = safe_d_path
        self.repeat = repeat

        self.method_my = "dynamic_anchor_regression"   
        self.method_oppo = "dynamic_anchor_regression"  

        self.likelihood_method = 'trend'
        #'stepwise', 'regression', 'exception',trend
        
        # ## print(self.my_ufun.weights)
        # ## print(self.my_ufun.issues)
        # ## print(self.my_ufun.values)
        # ## print(self.my_ufun.weights)
        # ## print(self.my_ufun.issues)
        # ## print(self.my_ufun.issues[0].values)

        # 1. 构建对手的权重（初始假设：平均权重）
        self.issue_value_info=issue_value_info
        # 上面的，形如 {'i1': ['v1', 'v2', 'v3', 'v4'], 
        # 'i2': ['v1', 'v2', 'v3', 'v4'], 
        # 'i3': ['v1', 'v2', 'v3', 'v4']}
        self.num_issues = len(self.issue_value_info)
        self.oppo_weights = [round(1.0 / self.num_issues, 5)] * self.num_issues

        # 2. 议题保持相同
        self.oppo_issues = copy.deepcopy(op_issues)

        
        # 3. 假设对手对所有议题的权重是平均的
        # self.oppo_values = [1 / self.num_issues] * self.num_issues
        # self.oppo_values={
        #     issue: 1 / self.num_issues
        #     for issue in self.issue_value_info
        # }
        
        # self.oppo_values = {}
        # for issue, values in self.issue_value_info.items():
        #     n = len(values)
        #     avg_value = 1.0 / n if n > 0 else 0.0
        #     # 构造一个 dict：每个取值的初始效用为 avg_value
        #     value_util_dict = {v: avg_value for v in values}
        #     # 创建 TableFunction 或 TableUtilityFunction
        #     self.oppo_values[issue] = TableFun(value_util_dict)
        
        self.oppo_values = []
        for issue, values in self.issue_value_info.items():
            n = len(values)
            avg_value = 1.0 / n if n > 0 else 0.0
            # 构造一个 dict：每个取值的初始效用为 avg_value
            value_util_dict = {v: avg_value for v in values}
            # 创建 TableFunction 或 TableUtilityFunction
            self.oppo_values.append(TableFun(value_util_dict))

        # 4. 创建预测对手效用函数
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )


        # 验证一下输出结果
        # print("Predicted Oppo Weights:\n", self.predicted_oppo_ufun._weights)
        # print("\nPredicted Oppo Issues:\n", self.predicted_oppo_ufun.issues)
        # print("\nPredicted Oppo Issue Values:")
        # for val in self.predicted_oppo_ufun.values:
        #     print(val.mapping)
        # # assert False
        self.history_real_next_offer_utility = []
        self.history_predicted_next_offer_utility = []
        self.history_utility_diff = []

        # 添加存储预测效用和真实效用的列表
        self.predicted_my_space_utilities = []  # 预测的我方空间效用
        self.real_my_space_utilities = []       # 真实的我方空间效用
        self.predicted_oppo_space_utilities = [] # 预测的对手空间效用
        self.real_oppo_space_utilities = []      # 真实的对手空间效用
        self.counter = 0
        self.relative_time = 0
        # self.my_util_cache = {
        #     offer: float(self.my_ufun(offer))
        #     for offer in self.offer_space
        # }

        # self.predicted_pareto = None 
        
        # print("对手建模成功")
        
        # print("Predicted Oppo Values:\n", self.oppo_values)

    def record_oppo_offer(self, offer,relative_time):
        self.relative_time = relative_time
        """
        记录对手的出价
        """
        # Add the opponent's offer to the list
        self.opponent_offers.append(offer)
        
        # Calculate utilities
        # opponent_utility = float(self.my_ufun(offer))
        # self.opponent_utilities.append(opponent_utility)

        # 更新贝叶斯模型
        self.update()

            
        
        # 如果我们已经有了预测，则记录真实效用与预测的对比
        if len(self.opponent_offers) > 1 and len(self.predicted_my_space_utilities) > 0:
            # 记录当前出价在我方效用空间的实际效用值
            # self.real_my_space_utilities.append(opponent_utility)
            
            # 记录在对手效用空间的实际效用值
            real_oppo_utility = float(self.opponent_real_utilities(offer))
            self.real_oppo_space_utilities.append(real_oppo_utility)

        
        # # 打印对手出价的我方效用值
        # try:
        #     utils = float(self.my_ufun(offer))
        #     # max_utils = float(self.my_ufun(self.best_offer))
        #     ## print(f"对手出价的我方效用值: {utils:.6f},")
        # except:
        #     print("无法计算对手出价的效用值")
        
        # # 如果可能，打印对手出价的对手效用值
        # try:
        #     if self.opponent_real_utilities is not None:
        #         opponent_utils = float(self.opponent_real_utilities(offer))
        #         ## print(f"对手出价的对手效用值(真实): {opponent_utils:.6f}")
        #     elif self.opponent_utility_function is not None:
        #         opponent_utils = float(self.opponent_utility_function(offer))
        #         ## print(f"对手出价的对手效用值(预测): {opponent_utils:.6f}")
        # except Exception as e:
        #     print(f"无法计算对手出价的效用值，错误信息：{e}")
       


    def update(self, sigma=0.1):

        ## print("开始ufun7")
        # 检查观测数量
        if len(self.opponent_offers) < 20:
            
            return

        offers_history = []
        for offer in self.opponent_offers:
            try:
                indices = [issue.values.index(choice) for issue, choice in zip(self.oppo_issues, offer)]
                offers_history.append(indices)
            except ValueError as e:
                # print(f"[WARNING] Skipping invalid offer {offer}: {e}")
                continue  # 跳过非法值

        offers_history = np.array(offers_history)
        if len(offers_history) == 0:
            # print("[WARNING] No valid offers in opponent history. Skipping update().")
            return


        onehot_history = []
        for offer in offers_history:
            offer_onehot = []
            for i, val in enumerate(offer):
                num_choices = len(self.oppo_issues[i].values)
                onehot = [0] * num_choices
                onehot[val] = 1
                offer_onehot.extend(onehot)
            onehot_history.append(offer_onehot)
        onehot_history = np.array(onehot_history)

        predicted_ufun_vector = []
        for issue_weight, issue_vals in zip(self.oppo_weights, self.oppo_values):
            # for option in self.oppo_values[issue_vals].mapping.values():
            for option in issue_vals.mapping.values():
                predicted_ufun_vector.append(issue_weight * option)

        newest_bid = onehot_history[-1]
        newest_util = np.dot(predicted_ufun_vector, newest_bid)

        # === Likelihood1: 高值区似然 ===
        trend_len = int(min(len(onehot_history), max(10, 0.05 * len(onehot_history))))
        recent_offers = onehot_history[-trend_len:]
        recent_utils = [np.dot(predicted_ufun_vector, offer) for offer in recent_offers]
        quantile = 90 if self.relative_time < 0.3 else 85
        q = np.percentile(recent_utils, quantile)
        distance = max(0.0001, abs(newest_util - q))
        likelihood1 = np.exp(- (distance ** 2) / (2 * sigma * sigma))

        # === Likelihood2: 我方offer1被拒，对手提出offer2，则u2应大于u1 ===
        likelihood2 = 1.0
        if len(self.self_offers) >= 1 and len(self.opponent_offers) >= 2:
            offer1 = self.self_offers[-1]
            offer2 = self.opponent_offers[-1]
            try:
                u1_oppo = float(self.predicted_oppo_ufun(offer1))
                u2_oppo = float(self.predicted_oppo_ufun(offer2))
                delta = u2_oppo - u1_oppo
                likelihood2 = 1.0 / (1.0 + np.exp(-10 * delta))
            except Exception as e:
                # print(f"[异常] Likelihood2计算失败: {e}")
                pass

        posterior = likelihood1 * likelihood2
        update_strength = posterior

        # === 强化机制1：自己最近50个offer vs 对手当前offer ===
        if len(self.self_offers) >= 1:
            for my_offer in self.self_offers[-50:]:
                try:
                    u1_self = float(self.my_ufun(my_offer))
                    u2_self = float(self.my_ufun(self.opponent_offers[-1]))
                    if abs(u1_self - u2_self) <= 0.05:
                        diff_index = None
                        for i in range(len(my_offer)):
                            if my_offer[i] != self.opponent_offers[-1][i]:
                                if diff_index is not None:
                                    diff_index = None
                                    break
                                diff_index = i
                        if diff_index is not None:
                            option = self.opponent_offers[-1][diff_index]
                            current_val = self.oppo_values[diff_index].mapping[option]
                            boosted_val = current_val + 0.2 * (1.0 - current_val)
                            self.oppo_values[diff_index].mapping[option] = round(0.8 * current_val + 0.2 * boosted_val, 5)
                            ## print(f"[强化逻辑1] 对选项 '{option}' 进行增强更新: {self.oppo_values[diff_index].mapping[option]:.4f}")
                except Exception as e:
                    # print(f"[异常] 强化逻辑1执行失败: {e}")
                    pass

        # === Early Prior: 仅在early阶段执行一次 ===
        if not hasattr(self, "early_prior_applied") and self.relative_time < 0.1:
            early_len = max(3, int(0.1 * len(self.opponent_offers)))
            early_offers = offers_history[:early_len]
            for issue_idx, issue in enumerate(self.oppo_issues):
                choices = early_offers[:, issue_idx]
                choice_counts = np.bincount(choices, minlength=len(issue.values))
                choice_freq = choice_counts / choice_counts.sum()
                for val_idx, val in enumerate(issue.values):
                    if choice_freq[val_idx] > 0.2:
                        self.oppo_values[issue_idx].mapping[val] = 0.9
            self.early_prior_applied = True
            ## print("[信息] Early Prior已应用")

        # === 更新议题权重 ===
        stability = np.std(offers_history, axis=0)
        stability_weights = np.exp(-stability)
        self.oppo_weights = stability_weights / stability_weights.sum()

        # === 更新选项值 ===
        for issue_idx, issue in enumerate(self.oppo_issues):
            choices = offers_history[:, issue_idx]
            choice_counts = np.bincount(choices, minlength=len(issue.values))
            choice_freq = choice_counts / choice_counts.sum()

            for val_idx, val in enumerate(issue.values):
                current_val = self.oppo_values[issue_idx].mapping[val]
                delta = choice_freq[val_idx] - current_val
                raw_val = current_val + update_strength * delta
                raw_val = min(max(raw_val, 0.0), 1.0)
                max_growth = 0.15
                adjusted_val = min(raw_val, current_val + max_growth)
                smooth_rate = 0.2
                adjusted_val = (1 - smooth_rate) * current_val + smooth_rate * adjusted_val
                self.oppo_values[issue_idx].mapping[val] = round(adjusted_val, 5)

        # === 重新构建预测效用函数 ===
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )




    
    def evaluate_pearson_correlation(self):
        """
        Evaluates opponent model accuracy using Pearson correlation and
        Pareto frontier surface difference.
        
        Saves results as a CSV row with self/opp agent names, env name, method, etc.
        """
        try:
            outcomes = list(self.my_ufun.outcome_space.enumerate_or_sample())
        except AttributeError:
            # print("[WARNING] Outcome space not available.")
            return None

        # ## print(self.is_diverse_opponent)
        
        # --- 1. Pearson Correlation ---
        real_utils = [float(self.opponent_real_utilities(o)) for o in outcomes]
        pred_utils = [float(self.predicted_oppo_ufun(o)) for o in outcomes]

        real_avg = np.mean(real_utils)
        pred_avg = np.mean(pred_utils)

        numerator = sum((r - real_avg) * (p - pred_avg) for r, p in zip(real_utils, pred_utils))
        real_var = sum((r - real_avg) ** 2 for r in real_utils)
        pred_var = sum((p - pred_avg) ** 2 for p in pred_utils)
        denominator = np.sqrt(real_var * pred_var)

        pearson_corr = numerator / denominator if denominator != 0 else 0.0
        ## print(f"[INFO] Pearson correlation: {pearson_corr:.4f}")


        # --- Clean predicted_oppo_ufun to remove np.float64 ---
        clean_weights = [float(w) for w in self.predicted_oppo_ufun.weights]
        clean_values = []
        for vfun in self.predicted_oppo_ufun.values:
            clean_mapping = {k: float(v) for k, v in vfun.mapping.items()}
            clean_values.append(TableFun(clean_mapping))

        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=clean_values,
            weights=clean_weights,
            issues=self.predicted_oppo_ufun.issues
        )


       
       
       # --- 2. Pareto Frontier Surface Difference ---
        ufuns_real = (self.my_ufun, self.opponent_real_utilities)
        frontier_real, _ = pareto_frontier(ufuns_real, outcomes)

        ufuns_pred = (self.my_ufun, self.predicted_oppo_ufun)
        frontier_pred, _ = pareto_frontier(ufuns_pred, outcomes)

        # --- 2. Compute surface using trapezoidal integration ---
        def compute_surface(frontier):
            frontier_sorted = sorted(frontier, key=lambda x: x[0])  # sort by u_self
            area = 0.0
            for i in range(1, len(frontier_sorted)):
                x0, y0 = frontier_sorted[i - 1]
                x1, y1 = frontier_sorted[i]
                area += (x1 - x0) * (y0 + y1) / 2
            return area

        

        # 按 x 排序真实边界
        frontier_real_sorted = sorted(frontier_real, key=lambda x: x[0])
        frontier_pred_sorted = sorted(frontier_pred, key=lambda x: x[0])

        # 拆解坐标
        x_real, y_real = zip(*frontier_real_sorted)
        x_pred, y_pred = zip(*frontier_pred_sorted)

        # 对真实边界构造插值函数（用于采样真实的值）
        real_interp = interp1d(x_real, y_real, kind='linear', bounds_error=False, fill_value=(y_real[0], y_real[-1]))

        # 统一 x 轴采样（假设效用值都是 0~1）
        x_common = np.linspace(0, 1, 100)

        # 创建插值函数
        real_interp = interp1d(x_real, y_real, kind='linear', bounds_error=False, fill_value='extrapolate')
        pred_interp = interp1d(x_pred, y_pred, kind='linear', bounds_error=False, fill_value='extrapolate')

        # 计算两个曲线在统一 x 轴上的 y 值
        real_y = np.clip(real_interp(x_common), 0, 1)
        pred_y = np.clip(pred_interp(x_common), 0, 1)

        # 差值面积（绝对值）
        surface_diff = np.trapz(np.abs(real_y - pred_y), x_common)

        # 用单位正方形面积归一化
        pareto_surface_diff = surface_diff  # already normalized over [0,1]

        # ## print(f"[INFO] Pareto frontier surface difference: {pareto_surface_diff:.4f}")

       

        return pearson_corr, pareto_surface_diff
    
      

    

    
    def get_oppo_offer_history(self):
        """
        获取所有对手的出价记录
        :return: 以 (offer, utility) 形式返回对手的出价和对应的效用值
        """
        # return list(zip(self.opponent_offers, self.opponent_utilities))
        # ## print(f"[DEBUG] oppo_offers: {self.opponent_offers}")  # 确保数据是完整的
        # ## print(f"[DEBUG] oppo_utilities: {self.opponent_utilities}")  # 确保数据是完整的
        # input()
        return list(zip(self.opponent_offers, self.opponent_utilities))  # 返回整个列表

    def get_oppo_last_offer(self):
        """
        获取对手的最后一次出价及其效用
        :return: (offer, utility) 或 None
        """
        if self.opponent_offers:
            return self.opponent_offers[-1], self.opponent_utilities[-1]
        return None
    
    def record_self_offer(self, offer):
        """
        记录我方的出价，并计算其在我方效用函数下的效用值
        :param offer: 我方的出价
        """
        if offer is not None:
            utility = float(self.my_ufun(offer))
            self.self_offers.append(offer)
            self.self_utilities.append(utility)
            # ## print(f"[Self] 我方出价: {offer}, 计算的效用值: {utility}")
            # input("按 Enter 键继续...")  # 方便观察输入输出
    def get_self_offer_history(self):
        """
        获取所有我方的出价记录
        :return: 以 (offer, utility) 形式返回我方的出价和对应的效用值
        """
        # ## print(f"[DEBUG] self_offers: {self.self_offers}")  # 确保数据是完整的
        # ## print(f"[DEBUG] self_utilities: {self.self_utilities}")  # 确保数据是完整的
        # input()
        return list(zip(self.self_offers, self.self_utilities))  # 返回整个列表


    def get_self_last_offer(self):
        """
        获取我方的最后一次出价及其效用
        :return: (offer, utility) 或 None
        """
        if self.self_offers:
            return self.self_offers[-1], self.self_utilities[-1]
        return None

