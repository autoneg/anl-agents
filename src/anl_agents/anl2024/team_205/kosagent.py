"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* KosAgent
*Authors* Kosuke Nakata s212131q@st.go.tuat.ac.jp

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random

from anl.anl2024.negotiators.base import ANLNegotiator
import numpy as np
from negmas.sao import ResponseType, SAOResponse, SAOState

__all__ = ["KosAgent"]


class KosAgent(ANLNegotiator):
    # 変数の初期設定
    step = 0
    current_time = 0
    next_offer = None
    e = 0
    current_value = 0
    max_value = 0
    min_value = 0
    allowable_line = 0
    current_value = 0
    best_offer__ = None

    def on_preferences_changed(self, changes) -> None:
        # 初期変数の設定
        self.step = 0
        self.current_time = 0
        self.next_offer = None
        self.e = 0
        self.current_value = 0
        self.max_value = 1
        self.allowable_line = 0
        self.current_value = 0

        # 最小値の設定
        assert self.ufun is not None
        if self.ufun.reserved_value >= 0.63:
            self.min_value = (
                self.ufun.reserved_value
            )  # この部分も最小値をうまいこと決められるようにしたい
        else:
            self.min_value = 0.63

        if self.ufun is None:
            return
        self.best_offer__ = self.ufun.best()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        # データの更新
        self.update(state)
        offer = state.current_offer
        self.next_offer = self.bidding_strategy(state)

        # Select action
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return SAOResponse(ResponseType.REJECT_OFFER, self.next_offer)

    # Make a counteroffer
    def bidding_strategy(self, state: SAOState):
        # ここもできれば相手によって時間調整をできるようにしたい
        if self.current_time < 0.85:
            self.e = 0.15
        elif self.current_time < 0.95:
            self.e = 0.01 * self.current_time + 0.15
        else:
            self.e = 0.5

        self.current_value = self.max_value - (self.max_value - self.min_value) * (
            self.current_time
        ) ** (1 / self.e)

        buf = self.choose_one(state, self.current_value, self.current_value + 0.03)
        if buf is not None:
            new_offer = buf

        buf = self.choose_one(state, self.current_value, self.current_value + 0.05)
        if buf is not None:
            new_offer = buf

        else:
            new_offer = self.choose_one(state, 100, 100)

        return new_offer

    # Accept the offer of the opponent agent
    def acceptance_strategy(self, state: SAOState):
        # 関数の初期化
        assert self.ufun
        offer = state.current_offer
        decision_time = 0.85

        if self.current_time < decision_time:
            decision_line = 1 - (1 - self.min_value) * (self.current_time**10)

        else:
            m_value = 1 - (1 - self.min_value) * (decision_time**10)
            decision_line = (
                1 / (1 + np.exp(40 * (2 * self.current_time - (1 + decision_time))))
            ) * (m_value - self.min_value) + self.min_value

        # 次の自分の提案よりもよかったら受け入れる(一応)
        if self.ufun(offer) > self.ufun(self.next_offer):
            return True

        # 受け入れるかどうかの決定
        if self.ufun(offer) >= decision_line:
            return True

        # 最後の一回になったら強制的に受け入れる

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        if nsteps__ - self.step == 1:
            if self.ufun(offer) > self.ufun.reserved_value:
                return True

        return False

    # one_in 与えられた範囲の中から相手にとって一番良いものを提案する
    def choose_one(self, state: SAOState, a: float, b: float):
        # 特別処理としてaとbの値が100ならば一番よい選択肢を出力(該当の範囲内に選択肢がなかった場合)
        assert self.ufun is not None
        if a == 100 and b == 100:
            if self.best_offer__ is None:
                self.best_offer__ = self.ufun.best()
            result_outcome = self.best_offer__
            return result_outcome

        self.choose_box = []
        self.choose_box = [
            p
            for p in self.nmi.outcome_space.enumerate_or_sample()
            if b >= self.ufun(p) and self.ufun(p) >= a
        ]
        if len(self.choose_box) == 0:
            return None

        # 特別処理としてstep数が5回ごとにchoose_boxからランダムで選択肢を生成(一応相手を撹乱)
        if self.step % 10 == 0:
            return random.choice(self.choose_box)
        else:
            # 最大値の探索
            assert self.opponent_ufun is not None
            max_num = self.opponent_ufun(self.choose_box[0])
            max_offer = self.choose_box[0]
            for i in self.choose_box:
                num = self.opponent_ufun(i)
                if num > max_num:
                    max_num = num
                    max_offer = i
                elif num == max_num:
                    if self.ufun(i) > self.ufun(max_offer):
                        max_num = num
                        max_offer = i
            return max_offer

    def update(self, state: SAOState):
        # ステップ数の増加
        self.step = self.step + 1
        # 現在時刻の更新

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        # assert self.nmi.n_steps is not None
        self.current_time = self.step / nsteps__


if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    # if you want to do a very small test,use the parameter small=True here. Otherwise, you can use the default parameters.
    run_a_tournament(KosAgent, small=True)
