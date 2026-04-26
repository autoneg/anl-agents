from typing import Literal
from random import choice
from anl2025.ufun import CenterUFun, SideUFun
from negmas import SAONMI, InverseUFun, PolyAspiration, PresortingInverseUtilityFunction

# 追加
from .helpers.helperfunctions import is_edge_agent

from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)
from negmas.outcomes import Outcome
from anl2025.negotiator import ANL2025Negotiator

__all__ = ["KAgent"]


class KAgent(ANL2025Negotiator):
    """
    Agent
    基本的に提案：("A","B","C")と
    応答：ACCEPT,REJECT,END_NEGOTIATIOM
    の二つで構成されている。
    プログラム内で、提案(Outcome)と応答(ResponseType.~~)を決定することでnegotiationは進んでいく。
    Propose関数：提案(Outcome)を決定する関数。
    Respond関数：応答(ResponseType)を決定する関数。
    """

    def __init__(
        self,
        *args,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | Literal["hardheaded"] =
        # | float = "boulware",
        None,
        deltas: tuple[float, ...] = (1e-3, 1e-1, 2e-1, 4e-1, 8e-1, 1.0),
        reject_exactly_as_reserved: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # 交渉履歴を保持（合意提案などを蓄積する想定）
        self.history: list[Outcome] = []
        self.ufunvalue = 0
        # 初期化（クラスの __init__ の中で追加）
        self._level_bias = 0.0

        # CenterUFunの初期化（ufunを使って生成）
        # ここではufunがNoneの場合もあるので例外処理を含める
        # if not is_edge_agent(self):
        # self.center_ufun = CenterUFun
        # print(self.center_ufun)
        self._opponent_offers = {}  # ← 追加
        self._opponent_issue_values = {}  # ← 追加
        if is_edge_agent(self):
            aspiration_type = "hardheaded"
            self._curve = PolyAspiration(0.9, aspiration_type)
        else:
            aspiration_type = "boulware"
            self._curve = PolyAspiration(1.0, aspiration_type)
        self._inverter: dict[str, InverseUFun] = (
            dict()
        )  # inverterっていうものを格納するためだけの変数
        # ここ三つはinverterから得られた値をいつでも使えるように格納するための変数
        self._best: list[Outcome] = None  # type: ignore  #ある程度スコアのいいやつを選んでしまっておくリスト。
        self._mx: float = 1.0  # 最大のスコアをしまっておく変数
        self._mn: float = 0.0  # 最小のスコアをしまっておく変数
        # ここはヒューリスティックな値を用意している場所。（例えば、if文の比較に>を使うか>=を使うか、みたいな）
        self._deltas = deltas
        self._best_margin = 1e-8
        self.reject_exactly_as_reserved = reject_exactly_as_reserved  # if文の比較をどうするか制御している。Trueなら>=で比較するし、Falseなら>で比較

    def ensure_inverter(self, negotiator_id) -> InverseUFun:
        """Ensures that utility inverter is available"""
        """効用値→提案を可能にするInverseUfunを呼び出すための関数"""
        if self._inverter.get(negotiator_id, None) is None:
            _, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]
            inverter = PresortingInverseUtilityFunction(ufun, rational_only=True)
            inverter.init()
            # breakpoint()
            """最大値とか最小値とかを取り出しておく"""
            self._mx, self._mn = inverter.max(), inverter.min()
            self._mn = max(self._mn, ufun(None))
            self._best = inverter.some(
                (
                    max(0.0, self._mn, ufun(None), self._mx - self._best_margin),
                    self._mx,
                ),
                normalized=True,
            )
            if not self._best:
                self._best = [inverter.best()]  # type: ignore
            self._inverter[negotiator_id] = inverter

        return self._inverter[negotiator_id]

    # def calc_level(self, nmi: SAONMI, state: SAOState, normalized: bool):
    # """時間によって減衰する値を計算するための関数"""
    # if state.step == 0:
    # level = 1.0
    # elif (
    # not self.reject_exactly_as_reserved
    # and
    # nmi.n_steps is not None and state.step >= nmi.n_steps - 1
    # ):
    # level = 0.0
    # else:
    # level = self._curve.utility_at(state.relative_time)
    # if not normalized:
    # level = level * (self._mx - self._mn) + self._mn
    # return level

    def calc_level(
        self, nmi: SAONMI, state: SAOState, normalized: bool, negotiator_id, source=None
    ):
        """時間によって減衰する値を計算するための関数"""
        if state.step == 0:
            level = 1.0
        elif (
            # not self.reject_exactly_as_reserved
            # and
            nmi.n_steps is not None and state.step >= nmi.n_steps - 1
        ):
            level = 0.0
        else:
            base_level = self._curve.utility_at(state.relative_time)
            adjustment = self.estimate_opponent_concession(
                self._opponent_offers.get(source, [])
            )
            # 譲歩傾向を見て level を調整
            # print(adjustment)
            if adjustment > 0:
                self._level_bias += 0.002  # 弱気：自分は強気に出る
                # print(self._level_bias)
                # print("強気に出ようかな")
            elif adjustment < 0:
                self._level_bias -= 0.001  # 強気：自分も少し譲歩する
                # print("強気です．．")
            # level に累積バイアスを加算（制限も必要）
            level = base_level + self._level_bias
            level = min(1.0, max(0.0, level))  # 0.0〜1.0 にクリップ
            # print(level)
        if not normalized:
            level = level * (self._mx - self._mn) + self._mn
        return level

        # 関数すべて追加

    def estimate_opponent_concession(self, offers: list[tuple[float, float]]) -> float:
        # print(offers)
        if len(offers) < 2:
            # print("なし")
            return 0.0
        # print("計算します")
        t1, u1 = offers[-2]
        t2, u2 = offers[-1]
        # print(u1, u2)
        return -(u2 - u1) / (t2 - t1 + 1e-6)  # 微小量でゼロ割り回避

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).
        提案を行うための関数：メイン！
        Remarks:
        """
        # 色々と変数を用意するエリア：ufun(効用値の計算に使う)だったり、nmi(交渉全体の色々なことが入っている)であったり。
        assert self.ufun
        negotiator, cntxt = self.negotiators[
            negotiator_id
        ]  # ufunとinverterを取り出すための準備
        inverter = self.ensure_inverter(
            negotiator_id
        )  # inverterの取り出し。これによって、適当な効用値から提案を引き出すことができる
        nmi = (
            negotiator.nmi
        )  # nmiを取り出しておく。nmiはこの交渉に関する色々な事柄が格納されている。
        # level = self.calc_level(nmi, state, normalized=True) #時間によって減衰する変数の取り出し。
        ufun: SideUFun = cntxt[
            "ufun"
        ]  # ufunを取り出しておく。これで今後、提案の効用値を知りたい時は、ufun(~~)でできるようになった。
        source = dest or state.current_partner_id
        level = self.calc_level(
            nmi, state, normalized=True, negotiator_id=negotiator_id, source=source
        )

        # 提案を実際に決めていくエリア。一応提案はNoneで通すこともできるため、Noneで初期化している。
        outcome = None
        if self._mx < float(ufun(None)):
            return None
        for d in self._deltas:  # 最初に決めたdeltaを用いて、提案の効用値を決める。
            mx = min(1.0, level + d)
            outcome = inverter.one_in(
                (level, mx), normalized=True
            )  # ここでかかった時間分の減衰を考えた提案を決める。
            # one_in関数は、引数によって定められた効用値の範囲から適当な提案を返す関数
            # print(f"{self.id} found {outcome} at level {(level, mx)}")
            if outcome:  # 提案が決まればbreak
                break
        if not outcome:  # もしdeltaの最後まで決まらなければ、事前に見繕っていたいい感じのものから適当に一個選ぶ。
            return choice(self._best)
        return outcome  # 最後に提案を返り値にして終了

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """Responds to the given partner (source) using the side negotiator (negotiator_id).
        相手からの提案に対する応答を決定するための関数：メイン！
        Remarks:
            - source: is the ID of the partner.
            - the mapping from negotiator_id to source is stable within a negotiation.

        """
        # 色々と変数を用意するエリア：ufun(効用値の計算に使う)だったり、nmi(交渉全体の色々なことが入っている)であったり。
        assert self.ufun
        _, cntxt = self.negotiators[negotiator_id]
        ufun: SideUFun = cntxt["ufun"]
        center_ufun: CenterUFun = self.ufun
        # print(ufun)
        # print(type(center_ufun))
        # print(center_ufun)
        # center_ufun: CenterUFun = cntxt["ufun"]
        nmi = self.negotiators[negotiator_id][0].nmi
        self.ensure_inverter(negotiator_id)
        # end the negotiation if there are no rational outcomes
        # level = self.calc_level(nmi, state, normalized=False)
        level = self.calc_level(
            nmi, state, normalized=True, negotiator_id=negotiator_id, source=source
        )
        offer = state.current_offer

        # 相手の提案を記録(3行追加)
        if source not in self._opponent_offers:
            self._opponent_offers[source] = []
        self._opponent_offers[source].append(
            (state.relative_time, ufun(state.current_offer))
        )

        # print(type(self.ufun))

        # 最終的な提案に対する応答を確定するエリア。
        if self._mx < float(ufun(None)):  # 流石に論外な提案に対して交渉を終了させる。
            return ResponseType.END_NEGOTIATION

        # センターエージェントであれば，CenterUFunで評価
        if not is_edge_agent(self):
            # assert isinstance(self.ufun, CenterUFun)

            # n_edgesはセンター効用関数が期待する結果の数
            n_edges = self.ufun.n_edges

            #   センター効用関数のために合意履歴 + 現在の提案をまとめて渡す
            combined_outcomes = tuple(self.history) + (state.current_offer,)
            # print(combined_outcomes)

            # self.ufun.n_edges と一致していない場合は，末尾を削るなどして調整（例として）
            if len(combined_outcomes) > n_edges:
                combined_outcomes = combined_outcomes[-self.ufun.n_edges :]
            elif len(combined_outcomes) < n_edges:
                # 不足分は None で埋める（交渉が始まったばかりの場合など）
                combined_outcomes = combined_outcomes + (None,) * (
                    self.ufun.n_edges - len(combined_outcomes)
                )
            # print(combined_outcomes)
            # print("ここからだよーーーーーーーーーーーーーーーーーー")
            # print(CenterUFun)

            # total_utility = ufun.getUtility.eval(combined_outcomes)
            # total_utility = CenterUFun.eval(combined_outcomes, 3)
            # total_utility = center_ufun.eval(combined_outcomes)
            # total_utility = center_ufun(combined_outcomes, use_expected=True)
            # total_utility = center_ufun(combined_outcomes)
            # total_utility = center_ufun(combined_outcomes)
            # print("結果発表ーーーーーーーーーーーーーー")
            # print(combined_outcomes)
            total_utility = center_ufun.eval(combined_outcomes)
            # print(total_utility)
            # print(state.current_offer)
            # print(ufun(state.current_offer))
            # print(center_ufun)
            # print(center_ufun.__class__)
            # print(center_ufun.eval)
            # print("combined_outcomes:", combined_outcomes)
            # print("state.current_offer:", state.current_offer)

            # 評価
            # try:
            # total_utility = center_ufun.eval(combined_outcomes)
            # print("結果発表ーーーーーーーーーーーーーー")
            # print(total_utility)
            # print(ufun(state.current_offer))
            # except Exception as e:
            # print("center_ufunでエラー発生:", e)

            # ★効用値が前回より下がっていればreject（初期は0.5基準）
            # prev = getattr(self, "ufunvalue", 0.5)  # 初期値0.5
            # if total_utility < prev:
            # return ResponseType.REJECT_OFFER

            # 閾値を上回っていればaccept（従来のルール）
            if (self.reject_exactly_as_reserved and level >= total_utility) or (
                not self.reject_exactly_as_reserved and level > total_utility
            ):
                return ResponseType.REJECT_OFFER
            return ResponseType.ACCEPT_OFFER

        # 閾値を上回っていればaccept（従来のルール）
        if (self.reject_exactly_as_reserved and level >= ufun(state.current_offer)) or (
            not self.reject_exactly_as_reserved and level > ufun(state.current_offer)
        ):
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        """交渉が終わった時に呼び出される関数"""
        _, cntxt = self.negotiators[negotiator_id]
        # ufun: SideUFun = cntxt["ufun"]
        center_ufun: CenterUFun = self.ufun
        # print("交渉が終了したーーーーーーーーーーーーーー")
        for side in self.negotiators.keys():
            if side == negotiator_id:
                continue
            if side in self._inverter:
                del self._inverter[side]

        # 合意した提案があれば履歴に記録（センターエージェントのみ対象）
        if not is_edge_agent(self):
            # print("記録しますーーーーーーーーーーーーーーーーーーー")
            self.history.append(state.agreement)
            # 現在の履歴をタプルに変換（センター効用評価用）
            record = tuple(self.history)

            # 評価（center_ufunはself.ufunとして取得済み）
            value = self.ufun.eval(record)

            # 提案セットとその効用を保存
            self.ufunvalue = value

            # ログ（任意）
            # print("履歴:", record)
            # print("評価値:", value)


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
# if __name__ == "__main__":
# from myagent.helpers.runner import run_a_tournament
# Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
# Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!
# run_a_tournament(MyAgent2000, small=False)
