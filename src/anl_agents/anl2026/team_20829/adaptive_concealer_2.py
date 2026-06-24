import random
import math
import numpy as np  # 統計量（標準偏差等）の計算に使用
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class AdaptiveConcealer2(SAOCallNegotiator):
    rational_outcomes = tuple()

    def on_preferences_changed(self, changes):
        """交渉開始時の初期化 (ANL 2026 仕様)"""
        if self.ufun is None:
            return

        # 1. 自分にとって合理的な（留保価格以上の）選択肢を抽出してソート
        ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) >= self.ufun.reserved_value
        ]
        self.rational_outcomes = tuple(_[1] for _ in sorted(ufun_outcome, reverse=True))

        # 2. 内部変数の初期化
        self.opponent_history = []
        self.my_history = []
        self.issue_value_counts = {}
        self.weight_hypotheses = {}
        
        self.n_issues = len(self.rational_outcomes[0]) if self.rational_outcomes else len(self.nmi.outcome_space.issues)
        
        # ベイズ初期確率（均等）
        for i in range(self.n_issues):
            self.weight_hypotheses[i] = 1.0 / self.n_issues

        # 【追加】ペルソナ分析用の変数
        self.opponent_persona = "UNKNOWN"
        self.estimated_opponent_utils_history = []  # 相手視点での提案の"価値"（擬似）の推移
        self.boulware_e = 40.0                      # デフォルトの強硬度

        # 3. ANL2026配点システムへの推測器の登録
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=self.estimate_opponent_utility)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is None:
            my_bid = self.concealing_bidding_strategy(state)
            self.my_history.append(my_bid)
            return SAOResponse(ResponseType.REJECT_OFFER, my_bid)

        # 相手モデルの更新 & ペルソナ分析
        self.update_opponent_model(state)

        # 受諾判定
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # カウンターオファー
        my_bid = self.concealing_bidding_strategy(state)
        self.my_history.append(my_bid)
        return SAOResponse(ResponseType.REJECT_OFFER, my_bid)

    def _get_target_utility(self, state: SAOState) -> float:
        """ペルソナに応じて動的に目標効用（譲歩カーブ）を変更"""
        if not self.rational_outcomes:
            return self.ufun.reserved_value

        u_max = self.ufun(self.rational_outcomes[0])
        u_res = self.ufun.reserved_value
        t = state.relative_time

        # --- ペルソナに応じた動的戦略分岐 ---
        
        # ペルソナA: 強硬・スナイパー型 (t=0.96まで一切譲歩しない超崖型)
        if self.opponent_persona == "A_HARD_SNIPER":
            e = 60.0  # 元の40.0よりさらに強硬に
            return u_max - (u_max - u_res) * (t ** e)

        # ペルソナB: 協調・線形譲歩型 (t=0.5から相手より少し高い位置をキープして滑らかに譲歩)
        elif self.opponent_persona == "B_COOPERATIVE_LINEAR":
            if t < 0.5:
                return u_max - 0.02  # 序盤は少しだけ高めで固定
            else:
                # t=0.5 から t=1.0 に向けて、緩やかな線形（またはマイルドなBoulware e=2.0）で譲歩
                # 決裂を防ぐため、最終盤には留保価値の少し上（+0.1）程度まで下りて歩み寄る
                target_at_end = max(u_res + 0.1, 0.75)
                progress = (t - 0.5) / 0.5  # 0.0 -> 1.0
                return (u_max - 0.02) - ((u_max - 0.02) - target_at_end) * progress

        # ペルソナC: ランダム・ノイズ型 (相手を信用せず、一定以上のオファーで即合意。カーブ自体は高め維持)
        elif self.opponent_persona == "C_RANDOM_NOISE":
            return max(0.85, u_res)

        # デフォルト (通常モード)
        else:
            e = self.boulware_e  # 40.0
            return u_max - (u_max - u_res) * (t ** e)

    def update_opponent_model(self, state: SAOState) -> None:
        """相手の欺瞞（Veil型）検知 + 序盤のペルソナ判定"""
        offer = state.current_offer
        if offer is None:
            return

        t = state.relative_time
        self.opponent_history.append(offer)

        # 相手の現時点での暫定推定効用（頻度ベース）を記録
        current_est_util = self.estimate_opponent_utility(offer)
        self.estimated_opponent_utils_history.append(current_est_util)

        # =================【新規】序盤のペルソナ判定（15ターン目に実行） =================
        if self.opponent_persona == "UNKNOWN" and len(self.opponent_history) == 15:
            self._classify_opponent_persona()

        # 【既存の嘘つき（Active Deception）検知ロジック】
        learning_weight = 1.0
        if t > 0.5 and len(self.opponent_history) > 10:
            early_history = self.opponent_history[:10]
            mismatch_count = 0
            for i, val in enumerate(offer):
                early_vals = [h[i] for h in early_history]
                if val not in early_vals:
                    mismatch_count += 1
            
            if mismatch_count > (self.n_issues / 2):
                for i in self.issue_value_counts:
                    for v in self.issue_value_counts[i]:
                        self.issue_value_counts[i][v] *= 0.2
                learning_weight = 2.0

        # 1. 頻度モデルの更新
        for i, val in enumerate(offer):
            if i not in self.issue_value_counts:
                self.issue_value_counts[i] = {}
            time_decay = (1.0 - t) * learning_weight
            self.issue_value_counts[i][val] = self.issue_value_counts[i].get(val, 0.0) + time_decay

        # 2. ベイズ推論による論点重要度の更新
        if len(self.opponent_history) > 1:
            last_offer = self.opponent_history[-2]
            for i in range(self.n_issues):
                if offer[i] != last_offer[i]:
                    self.weight_hypotheses[i] *= 0.95

            sum_posterior = sum(self.weight_hypotheses.values())
            if sum_posterior > 0:
                for i in range(self.n_issues):
                    self.weight_hypotheses[i] /= sum_posterior

    def _classify_opponent_persona(self):
        """15ターンのデータから相手の標準偏差とデルタ（変化量）を計算し分類"""
        utils = self.estimated_opponent_utils_history
        
        # 1. 各ターンの変化量（delta_u）の絶対値
        deltas = [abs(utils[i] - utils[i-1]) for i in range(1, len(utils))]
        
        mean_delta = np.mean(deltas)
        std_dev = np.std(utils)

        # --- 判定ロジック ---
        # 閾値（0.15や0.015）はANLの一般的な効用レンジ(0.0~1.0)を基準に調整
        
        # 判定C: 標準偏差が極めて高い、または毎ターンの変化量が激しく一貫性がない => ランダム
        if std_dev > 0.15 and mean_delta > 0.08:
            self.opponent_persona = "C_RANDOM_NOISE"
            
        # 判定A: 標準偏差がほぼゼロ（＝オファーが最初から固定されて動かない）=> 強硬・スナイパー
        elif std_dev < 0.015 and mean_delta < 0.005:
            self.opponent_persona = "A_HARD_SNIPER"
            
        # 判定B: 毎ターン一定のペースで変化している（標準偏差がそこそこあり、差分が安定）=> 線形譲歩
        elif mean_delta > 0.01 and std_dev >= 0.015:
            self.opponent_persona = "B_COOPERATIVE_LINEAR"
            
        else:
            self.opponent_persona = "NORMAL"  # どちらにも偏らない、または判定不能時

    def estimate_opponent_utility(self, outcome: Outcome) -> float:
        """ANLシステムに提出する、現在推測される相手の効用関数"""
        if not self.weight_hypotheses or not self.issue_value_counts:
            return 0.5

        estimated_utility = 0.0
        for i, val in enumerate(outcome):
            weight = self.weight_hypotheses.get(i, 1.0 / self.n_issues)
            counts = self.issue_value_counts.get(i, {})
            val_score = 0.5
            if counts:
                max_count = max(counts.values())
                if max_count > 0:
                    val_score = counts.get(val, 0.0) / max_count
            estimated_utility += weight * val_score
            
        return estimated_utility

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """逆頻度（Anti-Frequency）撹乱 ＆ パレート最適スナイピングのハイブリッド"""
        if not self.rational_outcomes:
            return None

        t = state.relative_time
        base_target = self._get_target_utility(state)

        # 自分にとって十分美味しい（目標クリア）提案プールを作成
        acceptable_outcomes = [o for o in self.rational_outcomes if self.ufun(o) >= base_target]
        if not acceptable_outcomes:
            return self.rational_outcomes[0]

        # 【ペルソナA専用の超終盤スナイパー対応】
        # 相手が強硬型の場合、t=0.96まで頑なに情報を隠蔽し、0.96以降に一撃必殺を狙う
        trigger_time = 0.96 if self.opponent_persona == "A_HARD_SNIPER" else 0.85

        # 【フェーズ1：隠蔽戦略】
        if t < trigger_time:
            def get_my_bid_frequency(outcome):
                return sum(1 for past_bid in self.my_history if past_bid == outcome)
            
            min_freq = min(get_my_bid_frequency(o) for o in acceptable_outcomes)
            best_concealing_pool = [o for o in acceptable_outcomes if get_my_bid_frequency(o) == min_freq]
            return random.choice(best_concealing_pool)

        # 【フェーズ2：終盤のパレート・スナイピング戦略】
        else:
            sniping_pool = [o for o in self.rational_outcomes if self.ufun(o) >= max(0.90, self.ufun.reserved_value)]
            if not sniping_pool:
                sniping_pool = acceptable_outcomes
            
            return max(sniping_pool, key=lambda o: self.estimate_opponent_utility(o))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """適応型受諾戦略"""
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return False

        offer_util = self.ufun(offer)
        t = state.relative_time

        # --- 【追加】ペルソナC（ランダム）への特別BOA対策 ---
        # 相手がランダムノイズ型なら、自分の効用が0.85以上（かつ留保価格以上）のオファーなら即座に妥協してセーフティネットを張る
        if self.opponent_persona == "C_RANDOM_NOISE":
            if offer_util >= 0.85 and offer_util >= self.ufun.reserved_value:
                return True

        # 1. パニックモード（決裂回避）
        if t >= 0.99 and offer_util >= self.ufun.reserved_value:
            return True

        # 2. 目標効用ベースの受諾
        target_util = self._get_target_utility(state)
        if offer_util >= target_util:
            return True

        # 3. 相手の「崖崩れ（不意の妥協）」検知
        if t > 0.95 and offer_util >= 0.93:
            return True

        return False