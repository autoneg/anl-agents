import math
from collections import Counter
import numpy as np
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from collections import defaultdict

class Snake(SAOCallNegotiator):
    rational_outcomes = tuple()

    def on_preferences_changed(self, changes):
        """交渉開始時の初期化 (ANL 2026 仕様準拠)"""
        if self.ufun is None:
            return

        # 自身の合理的な選択肢を抽出・ソート
        ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) >= self.ufun.reserved_value
        ]
        self.rational_outcomes = tuple(_[1] for _ in sorted(ufun_outcome, reverse=True))

        # 内部変数の初期化
        self.window_size = 20

        self.n_issues = len(self.rational_outcomes[0]) if self.rational_outcomes else len(self.nmi.outcome_space.issues)
        self.weight_hypotheses = {i: 1.0 / self.n_issues for i in range(self.n_issues)}

        # ペルソナとオンライン推定用変数
        self.opponent_persona = "UNKNOWN"
        self.estimated_opponent_utils_history = []
        self.boulware_e = 40.0           # 自身の強硬度

        self._initialize_state()

        # ANL配点システムへの推測器登録
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=self.estimate_opponent_utility)


    def _initialize_state(self):

        self.opponent_history = []
        self.opponent_history_window = []
        self.my_history = []

        self.concession_history = []

        self.estimated_boulware_e = 1.0     # 初期値

        self.issue_counters = {
            i: Counter()
            for i in range(self.n_issues)
        }

        self.offer_visit_count = Counter()

        self.total_offer_count = 1

        self.model_weights = {
            "freq":0.4,
            "recent":0.4,
            "transition":0.2,
        }

        self.pareto_outcomes = []

        self.opponent_utility_cache = {}

        self.transition_same = {
            i: Counter()
            for i in range(self.n_issues)
        }

        self.transition_total = 0

        self.pareto_outcomes = []
        self.pareto_initialized = False

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        offer = state.current_offer

        self.current_relative_time = state.relative_time

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is None:
            my_bid = self.concealing_bidding_strategy(state)
            self.my_history.append(my_bid)
            return SAOResponse(ResponseType.REJECT_OFFER, my_bid)

        # 相手モデルの更新 (Sliding Window + Entropy)
        self.update_opponent_model(state)

        # 自身の次の提案を事前計算 (AC_next判定用)
        next_bid = self.concealing_bidding_strategy(state)

        # 受諾判定
        if self.acceptance_strategy(state, next_bid):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        self.offer_visit_count[next_bid] += 1
        self.total_offer_count += 1

        # カウンターオファー
        self.my_history.append(next_bid)
        return SAOResponse(ResponseType.REJECT_OFFER, next_bid)

    def _get_target_utility(self, state: SAOState) -> float:
        """動的な目標効用（譲歩カーブ）の算出"""
        if not self.rational_outcomes:
            return self.ufun.reserved_value

        u_max = self.ufun(self.rational_outcomes[0])
        u_res = self.ufun.reserved_value
        t = state.relative_time
        
        e = 60.0
        return u_max - (u_max - u_res) * (t ** e)
    
        # オンライン推定した相手のBoulware指数に基づく動的戦略
        if self.estimated_boulware_e > 5.0 or self.opponent_persona == "A_HARD_SNIPER":
            # 相手が超強硬なら、こちらもギリギリまで譲歩しない
            e = 60.0
            return u_max - (u_max - u_res) * (t ** e)
        elif self.estimated_boulware_e < 0.5 or self.opponent_persona == "B_COOPERATIVE_LINEAR":
            # 相手が譲歩的(Conceder)なら、穏やかに歩み寄る
            if t < 0.5:
                return u_max - 0.02
            else:
                target_at_end = max(u_res + 0.1, 0.75)
                progress = (t - 0.5) / 0.5
                return (u_max - 0.02) - ((u_max - 0.02) - target_at_end) * progress
        else:
            e = self.boulware_e
            return u_max - (u_max - u_res) * (t ** e)
        

    def update_opponent_model(self, state: SAOState) -> None:
        """Sliding Window + Entropy ベースの Opponent Modeling"""
        offer = state.current_offer
        if offer is None:
            return

        self.opponent_history.append(offer)

        if len(self.opponent_history) >= 2:

            prev = self.opponent_history[-2]
            curr = self.opponent_history[-1]

            changed = sum(
                1
                for a, b in zip(prev, curr)
                if a != b
            )

            concession = changed / max(1, self.n_issues)

            self.concession_history.append(concession)

            self.transition_total += 1

            for i in range(self.n_issues):
                if prev[i] == curr[i]:
                    self.transition_same[i][curr[i]] += 1



        t = state.relative_time


        # スライディングウィンドウの更新
        self.opponent_history_window.append(offer)
        if len(self.opponent_history_window) > self.window_size:
            self.opponent_history_window.pop(0)

        # Counterを更新
        self.issue_counters = {}

        for i in range(self.n_issues):
            values = [o[i] for o in self.opponent_history_window]
            self.issue_counters[i] = Counter(values)

        # 序盤・中盤でオンラインBoulware指数(e)を推定
        if t > 0.1:
            self._estimate_boulware_exponent(t)

        # エントロピーを用いた重み(Issue Weights)の更新
        if len(self.opponent_history_window) > 5:
            new_weights = {}
            total_inverse_entropy = 0.0

            for i in range(self.n_issues):
                values = [o[i] for o in self.opponent_history_window]

                counter = Counter(values)

                entropy = 0

                for count in counter.values():

                    p = count / len(values)

                    entropy -= p * math.log(p + 1e-9)

                # エントロピーが低い（＝値が固定されている）ほど重みを大きくする
                # 完全固定（entropy=0）の場合を考慮して微小値を足す
                n_values = len(counter)

                max_entropy = math.log(max(2, n_values))

                normalized_entropy = entropy / max_entropy
                inv_entropy = 1.0 / (normalized_entropy + 0.05)

                new_weights[i] = inv_entropy
                total_inverse_entropy += inv_entropy


            # EMAで重みを更新
            if total_inverse_entropy > 0:

                alpha = min(
                    0.3,
                    len(self.opponent_history_window) / 50
                )

                for i in range(self.n_issues):

                    # 今回推定した重み
                    new_weight = (
                        new_weights[i] / total_inverse_entropy
                    )

                    # 前回の重み
                    old_weight = self.weight_hypotheses[i]

                    # EMA
                    self.weight_hypotheses[i] = (
                        (1 - alpha) * old_weight
                        + alpha * new_weight
                    )

                # EMA後にもう一度正規化
                total = sum(self.weight_hypotheses.values())

                if total > 0:
                    for i in range(self.n_issues):
                        self.weight_hypotheses[i] /= total

            # 20提案ごとにPareto Frontierを更新
            if (
                len(self.opponent_history) >= 20
                and len(self.opponent_history) % 20 == 0
            ):
                self.opponent_utility_cache.clear()
                self.pareto_outcomes = self._build_pareto_frontier()
                self.pareto_initialized = True


    def _estimate_boulware_exponent(self, t):

        if len(self.concession_history) < 10:
            return

        recent = self.concession_history[-10:]
        avg = np.mean(recent)

        if avg < 0.05:
            e = 80

        elif avg < 0.10:
            e = 40

        elif avg < 0.20:
            e = 15

        elif avg < 0.35:
            e = 5

        else:
            e = 1

        self.estimated_boulware_e = (
            0.8 * self.estimated_boulware_e
            + 0.2 * e
        )

        if self.estimated_boulware_e > 20:
            self.opponent_persona = "A_HARD_SNIPER"

        elif self.estimated_boulware_e < 5:
            self.opponent_persona = "B_COOPERATIVE_LINEAR"

        else:
            self.opponent_persona = "NORMAL"


    def estimate_opponent_utility_freq(self, outcome: Outcome) -> float:

        if not hasattr(self, "issue_counters"):
            return 0.5

        """エントロピーベースの重みを用いた相手効用の推定"""
        if not self.weight_hypotheses or not self.opponent_history_window:
            return 0.5

        estimated_utility = 0.0
        for i, val in enumerate(outcome):
            weight = self.weight_hypotheses.get(i, 1.0 / self.n_issues)

            # ウィンドウ内の頻度をスコア化
            counter = self.issue_counters.get(i, Counter())

            count = counter.get(val, 0)

            max_count = max(counter.values(), default=1)

            val_score = count / max_count
            estimated_utility += weight * val_score

        return estimated_utility

    def _calculate_nash_product(self, outcome):

        my_util = max(
            self.ufun(outcome) - self.ufun.reserved_value,
            1e-6
        )

        opp_util = max(
            self.estimate_opponent_utility(outcome),
            1e-6
        )

        t = getattr(
            self,
            "current_relative_time",
            0.5
        )

        # Dynamic alpha
        if t<0.5:
            alpha=0.65
        elif t<0.8:
            alpha=0.60
        else:
            alpha=0.55

        return (
            my_util ** alpha
        ) * (
            opp_util ** (1 - alpha)
        )

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """隠蔽 ＆ 推定パレート＋Nash Product スナイピング"""
        if not self.rational_outcomes:
            return None

        t = state.relative_time
        base_target = self._get_target_utility(state)

        # 現在の目標効用を満たす候補（Acceptable Outcomes）
        candidate_pool = (
            self.pareto_outcomes
            if len(self.pareto_outcomes) > 0
            else self.rational_outcomes
        )

        acceptable_outcomes = [
            o
            for o in candidate_pool
            if self.ufun(o) >= base_target
        ]

        
        trigger_time = min(
            0.98,
            0.80 + 0.003 * self.estimated_boulware_e
        ) if self.estimated_boulware_e > 2.0 else 0.98
        

        trigger_time = 0.98

        # 【フェーズ1：逆頻度による情報隠蔽】
        if t < trigger_time:
            def get_my_bid_frequency(outcome):
                #return sum(1 for past_bid in self.my_history if past_bid == outcome)
                return self.offer_visit_count[outcome]


            if not acceptable_outcomes:
                return self.rational_outcomes[0]
            
            min_freq = min(get_my_bid_frequency(o) for o in acceptable_outcomes)
            best_concealing_pool = [o for o in acceptable_outcomes if get_my_bid_frequency(o) == min_freq]
            #best_concealing_pool = acceptable_outcomes
            # 序盤だけ探索(UCB)
            if t < 0.4:
                return max(
                    best_concealing_pool,
                    key=lambda o: (
                        self._calculate_nash_product(o)
                        + 0.05 * self._ucb_bonus(o)
                    )
                )

            # 中盤以降は探索を止める
            return max(
                best_concealing_pool,
                key=lambda o: (
                    self._calculate_nash_product(o)
                    - 0.01 * self.offer_visit_count[o]
                )
            )
        # 【フェーズ2：推定パレート境界 ＆ Nash Product スナイピング】
        else:
            sniping_utility_limit = max(0.90, self.ufun.reserved_value)

            # 相手の過去の提案から探す (最優先)
            history_sniping_pool = [
                o
                for o in self.opponent_history
                if self.ufun(o) >= sniping_utility_limit
            ]
            
            candidate = set()

            candidate.update(history_sniping_pool)
            candidate.update(self.pareto_outcomes)

            candidate = [
                o
                for o in candidate
                if self.ufun(o) >= sniping_utility_limit
            ]

            if not candidate:
                candidate = self.pareto_outcomes
            
            return max(candidate, key=self._endgame_score)
            
            """
            if history_sniping_pool:
                return max(
                    history_sniping_pool,
                    key=self.ufun
                )
            
            # 履歴になければ、自分の合理的な選択肢から推定パレート解を狙う
            sniping_pool = [
                o
                for o in candidate_pool
                if self.ufun(o) >= sniping_utility_limit
            ]
            # Nash Product を最大化する提案（推定される交渉解）をスナイピング
            if not sniping_pool:
                sniping_pool = acceptable_outcomes
            
            if not sniping_pool:
                sniping_pool = candidate_pool

            if not sniping_pool:
                sniping_pool = self.rational_outcomes

            return max(sniping_pool, key=lambda o: self._calculate_nash_product(o))
            candidate = list(set(history_sniping_pool + sniping_pool))

            return max(
                candidate,
                key=self._endgame_score
            )
            """

    def acceptance_strategy(self, state: SAOState, next_planned_bid: Outcome | None) -> bool:
        """AC_next と適応型ロジックを統合した受諾戦略"""
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return False

        offer_util = self.ufun(offer)
        t = state.relative_time
        target_util = self._get_target_utility(state)

        # 1. AC_next ルール (自分の次の提案よりも、相手の現在の提案の方が良ければ受諾)
        if (
            next_planned_bid is not None
        ):
            if t < 0.7:

                margin = 0.03

            elif t < 0.9:

                margin = 0.015

            else:

                margin = 0.0
            if offer_util >= self.ufun(next_planned_bid) + margin:
                return True
        """
        offer_nash = self._calculate_nash_product(offer)

        if next_planned_bid is not None:

            next_nash = self._calculate_nash_product(
                next_planned_bid
            )

            if (
                offer_nash >= next_nash
                and
                offer_util >= target_util
            ):
                return True

        """
        # 2. パニックモード（時間切れによる決裂回避）
        if t >= 0.99 and offer_util >= self.ufun.reserved_value:
            return True
        

        if t < 0.8:
            margin = 0.02
        elif t < 0.95:
            margin = 0.01
        else:
            margin = 0

        # 3. 目標効用ベースの受諾
        if offer_util >= target_util + margin:
            return True
        

        return False

    def _build_pareto_frontier(self):

        outcomes = []

        for o in self.rational_outcomes:
            outcomes.append((
                self.ufun(o),
                self.estimate_opponent_utility(o),
                o
            ))

        # 自分効用で降順ソート
        outcomes.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True
        )

        frontier = []

        best_opp = -1

        for my_u, opp_u, outcome in outcomes:

            if opp_u > best_opp:

                frontier.append(outcome)

                best_opp = opp_u

        return frontier

    def _ucb_bonus(self, outcome):

        n = self.offer_visit_count[outcome]

        if n == 0:
            return 1.0

        return math.sqrt(
            math.log(self.total_offer_count) / n
        )

    def estimate_opponent_utility_recent(
        self,
        outcome,
    ):

        if not self.opponent_history_window:
            return 0.5

        score = 0

        for i, val in enumerate(outcome):

            weight = self.weight_hypotheses.get(
                i,
                1/self.n_issues
            )

            recent = [
                o[i]
                for o in self.opponent_history_window[-10:]
            ]

            count = recent.count(val)

            score += weight * (
                count / max(1, len(recent))
            )

        return score

    def estimate_opponent_utility_transition(
        self,
        outcome,
    ):

        if self.transition_total == 0:
            return 0.5

        score = 0.0

        for i, val in enumerate(outcome):

            weight = self.weight_hypotheses.get(
                i,
                1/self.n_issues
            )

            score += (
                weight *
                self.transition_same[i][val]
                / self.transition_total
            )

        return score

    def estimate_opponent_utility(self, outcome):

        # キャッシュ
        if outcome in self.opponent_utility_cache:
            return self.opponent_utility_cache[outcome]

        freq = self.estimate_opponent_utility_freq(outcome)
        recent = self.estimate_opponent_utility_recent(outcome)
        transition = self.estimate_opponent_utility_transition(outcome)

        utility = (
            self.model_weights["freq"] * freq
            + self.model_weights["recent"] * recent
            + self.model_weights["transition"] * transition
        )

        self.opponent_utility_cache[outcome] = utility

        return utility
    
    
    def _endgame_score(self, outcome):

        my = self.ufun(outcome)
        opp = self.estimate_opponent_utility(outcome)

        # 相手が受け入れそうか
        if opp < 0.6:
            return -1

        score = my

        if outcome in self.opponent_history:
            score += 0.1

        return score
    
