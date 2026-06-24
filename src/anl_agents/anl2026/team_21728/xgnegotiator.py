import random
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun,BaseMultiFun,AffineMultiFun,LinearMultiFun
import traceback
from typing import Iterable,Callable,Any
from negmas.helpers.misc import (
    nonmonotonic_multi_minmax,
)
from negmas.outcomes.base_issue import Issue

class XGAgent(SAOCallNegotiator):
    """
    Your negotiator code. This is the ONLY class you need to implement.
    今回の交渉。重要な点は、「相手は自分にとってやや不利な提案も行う可能性がある」こと。
    つまり、不利な提案から、相手の主張の全体像が見やすくなる可能性がある。
    """

    rational_outcomes = tuple()

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2026, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """

        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return
        self.delta=0.1
        # create a list of all rational outcomes (i.e. outcomes with utility bigger than the reserved value) sorted by utility
        #基本固定のoutcomeのやつ
        self.enumerate_outcome=self.nmi.outcome_space.enumerate_or_sample()
        self.ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.enumerate_outcome  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.ufun_best=[
            (self.ufun(_), _)
            for _ in self.enumerate_outcome # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > 0.8
        ]
        self.rational_outcomes = tuple(_[1] for _ in sorted(self.ufun_outcome, reverse=True))
        print(f"N. rational: {len(self.rational_outcomes)}")


        #issue部分をベクトル化
        raw_data=[_ for _ in self.enumerate_outcome]

        data_value=[self.ufun(_) for _ in self.enumerate_outcome]
        opponent_initialize=[0.5 for _ in self.enumerate_outcome]


        #選択肢のデータモデル
        self.df= pd.DataFrame(raw_data,index=[tuple(x) for x in raw_data])
        self.df_encoded=pd.get_dummies(self.df)

        matrix_as_numpy= self.df_encoded.values

        data_value=np.array(data_value).reshape(-1,1)
        opponent_initialize=np.array(opponent_initialize).reshape(-1,1)

        #本題で使うベクトルの部分(選択肢とその評価値がhot-encodingされている)
        self.vectirize_self_ufun=np.concatenate([matrix_as_numpy,data_value],axis=1)
        self.vectirize_opponent_ufun=np.concatenate([matrix_as_numpy,opponent_initialize],axis=1)

        #model部分
        self.model = xgb.XGBRegressor(
            n_estimators=100,  # 作成する木の数
            learning_rate=1, # 学習率
            max_depth=5        # 木の深さ
            )
        #self.model = LassoCV(alphas=100, cv=3, max_iter=10000)
        #modelの初期化、とりあえず自分の評価値と真逆で仮定してある程度の初期化がいい感じかもしれない
        #テスト用データ（rev_data）の作成
        rev_data=np.array([1-self.ufun(_) for _ in self.enumerate_outcome])
        #インデックスデータ（その提案が何回呼び出されたか（y_trainか効用)）の作成
        self.opponent_index_data=[(_,1-self.ufun(_),0) for _ in self.enumerate_outcome]

        #おまじない（データを綺麗に整理するやつ）
        self.vectirize_self_ufun=np.array(self.vectirize_self_ufun,dtype=np.float64)
        rev_data=np.array(rev_data,dtype=np.float64)
        self.weights=np.ones(len(rev_data))

        try:
            self.model.fit(self.vectirize_self_ufun,rev_data,sample_weight=self.weights)
        except Exception:
            traceback.print_exc()
        # Initialize the opponent model, i.e. make a first guess for the opponent's utility function
        # Example: constant utility function
        try:
            self.private_info["opponent_ufun"] = PredictiveUfun(self.model,self.enumerate_outcome)
        except Exception:
            traceback.print_exc()
        try:
            self.private_info["opponent_ufun"].update_model(self.model)
            self.private_info["opponent_ufun"].update_oppnent(self.opponent_index_data)
            self.private_info["opponent_ufun"].update_df(self.df,self.ufun)
            self.private_info["opponent_ufun"].update_date(self.vectirize_self_ufun)
        except Exception:
            traceback.print_exc()
        self.my_offer=None #前回の自分の主張。相手が拒否したと言うことは効用値はそれほど高くないとわかる
        self.predict_outcome=self.model.predict(self.vectirize_self_ufun)
        self.opponent_outcome = [
                (p, o)
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.6
            ]
        self.deceptive_outcome = [
                o
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.8 and p>0.5
            ]
        self.deceptive_outcome_2 = [
                o
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.7 and p<0.3
            ]
        self.nash=None
        self.new_opp_index:list=[]
        self.new_y_train=[]


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
            - You can access your ufun using `self.ufun`.
            - You can access the opponent model using self.opponent_ufun
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """

        offer = state.current_offer
        #self.update_opponet_model(self.my_offer,state)
        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # If there is no offer yet (first call), make a counter offer
        if offer is None:
            self.my_offer=self.deceptive_bidding_strategy(state)
            return SAOResponse(
                ResponseType.REJECT_OFFER, self.my_offer
            )
        
        self.make_opponet_model(state)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the deceptive_bidding_strategy
        self.my_offer=self.deceptive_bidding_strategy(state)
        return SAOResponse(
            ResponseType.REJECT_OFFER, self.my_offer
        )

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """

        assert self.ufun

        offer = state.current_offer

        # Cannot accept a non-existent offer
        if offer is None:
            return False

        # Example: accept offer if utility is bigger than 80% of the maximum utility
        if self.ufun(offer) > self.ufun.max() * 0.8 > self.ufun.reserved_value:
            return True

        # Example: accept offer if utility is bigger than the reserved value
        if state.relative_time > 0.5 and self.ufun(offer) > 0.7:
            return True
        
        if state.relative_time > 0.8 and self.ufun(offer) >0.5:
            return True
        return False

    def deceptive_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the next deceptive counter offer.

        Returns: the counter offer as Outcome.
        """

        # Your opponent model can be accessed using self.private_info["opponent_ufun"], which is not used yet.
        #
        # Example: one of my best outcomes in the beginning of the negotiation
        if state.current_offer is None:
            if len(self.rational_outcomes[: min(len(self.rational_outcomes), 10)]) > 0 :
                return random.choice(
                    self.rational_outcomes[: min(len(self.rational_outcomes), 10)]
                )
            else:
                return random.choice(self.ufun_best)[1]
        # Example: random outcome in rational_outcomes
        if state.step % 30 == 0:
            self.predict_outcome=self.model.predict(self.vectirize_self_ufun)
            self.opponent_outcome = [
                (p, o)
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.6
            ]
            self.deceptive_outcome = [
                o
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.8 and p>0.5
            ]
            self.deceptive_outcome_2 = [
                o
                for p,o in zip(self.predict_outcome,self.enumerate_outcome)  # enumerates outcome space when finite, samples when infinite
                if self.ufun(o) > 0.4 and p<0.3
            ]
            self.nash=self.find_nash_bargaining_solution(self.ufun.reserved_value)
        make_a_choice=random.uniform(0,1)
        if self.nash is not None and state.relative_time>0.7 and self.ufun(self.nash)>0.7:
            return self.nash
        if make_a_choice >0.7 :
            if len(self.deceptive_outcome) > 0 :
                return random.choice(self.deceptive_outcome)
        elif make_a_choice >0.5 and state.relative_time>0.3 and state.relative_time<0.7:
            if len(self.deceptive_outcome_2) > 0:
                return random.choice(self.deceptive_outcome_2)
        elif make_a_choice >0.3 and state.relative_time < 0.5:
            if len(self.opponent_outcome) > 0:
                return random.choice(self.opponent_outcome)[1]
        
        if self.nash is not None and state.relative_time>0.3 and self.ufun(self.nash)>0.6:
            return self.nash
        return random.choice(self.ufun_best)[1]
            
    

        #とりあえず実装もしやすく事例が複数でも対応しやすい評価値制にしたい。



    def update_opponet_model(self, offer, state) -> None:
        """
        xgboostを用いた相手モデルの予測関数
        相手からの提案は高く、自分からの提案は低く見積もりたい
        """
        #データがなければ何もないので返却
        if offer is None:
            return
        time=state.step
        y_train=0.8
        y_train=y_train*(self.delta**state.relative_time)
        #新しい相手効用のタプルリスト
        self.opponent_index_data = [(y_train if x2 == offer else _,x2)for _, x2 in self.opponent_index_data]
        for i, (_,outcome) in enumerate(self.opponent_index_data):
            if outcome == offer:
                self.weights[i] +=10.0  # 10倍の重要度を持たせる
        #print(self.opponent_index_data
        
        if time % 30 == 0:
            opponent_predict=self.model.predict(self.vectirize_self_ufun)
            self.opponent_index_data = [( y_train if outcome == offer else x2,outcome) for (_,outcome),x2 in zip(self.opponent_index_data,opponent_predict)]
            y_train=np.array([ ufun for ufun,_ in self.opponent_index_data],dtype=np.float64)
            self.model.fit(self.vectirize_self_ufun,y_train,sample_weight=self.weights)
            self.private_info["opponent_ufun"].update_model(self.model)
            self.private_info["opponent_ufun"].update_date(self.vectirize_self_ufun)
        


    def make_opponet_model(self, state: SAOState) -> None:
        """
        xgboostを用いた相手モデルの予測関数
        相手からの提案は高く、自分からの提案は低く見積もりたい
        """
        #データがなければ何もないので返却
        if state.current_offer is None:
            return
        y_train=random.uniform(1.0,0.7)
        #新しい相手効用のタプルリスト
        self.opponent_index_data = [(outcome,_,count+1 if state.current_offer==outcome else count) for outcome,_,count in self.opponent_index_data]
        counts = [item[2] for item in self.opponent_index_data if item[1]>0]
        cotmen = np.mean(counts)
        for i, (outcome,_,count) in enumerate(self.opponent_index_data):
            if outcome == state.current_offer:
                if count>10 and self.ufun(outcome)<0.5:
                    y_train=1.0
                elif count>10 and self.ufun(outcome)>=0.5:
                    y_train=0.0
                elif cotmen>count:
                    y_train=random.uniform(0.3,0.1)
                else:
                    pass
        for _ in self.nmi.outcome_space.enumerate_or_sample():
                if _ == state.current_offer:
                    target = _
                    break  
        if target is not None:
                part_a = np.array(self.df_encoded.loc[[target]].values,dtype=np.float64) 
                part_b = np.array(self.ufun(target),dtype=np.float64).reshape(1,-1)
                new_row = np.hstack([part_a, part_b])
                new_row=np.array(new_row,np.float64).reshape(1,-1).flatten()
        found_index = -1
        for i, existing_row in enumerate(self.new_opp_index):
            # numpy配列同士の比較は np.array_equal を使うのが最も確実です
            if np.array_equal(existing_row, new_row):
                found_index = i
                break
        if found_index != -1:
            self.new_y_train[found_index] = y_train
        else:
            self.new_opp_index.append(new_row)
            self.new_y_train.append(y_train)
        if state.step % 30 == 0: 
            """
            opponent_predict=self.model.predict(self.vectirize_self_ufun)
            self.opponent_index_data = [( y_train if outcome == state.current_offer else x2,outcome) for (_,outcome),x2 in zip(self.opponent_index_data,opponent_predict)]
            y_train=np.array([ ufun for ufun,_ in self.opponent_index_data],dtype=np.float64)
            #print(y_train)
            self.model.fit(self.vectirize_self_ufun,y_train,sample_weight=self.weights)
            self.private_info["opponent_ufun"].update_model(self.model)
            self.private_info["opponent_ufun"].update_date(self.vectirize_self_ufun)
            """
            X_train=np.vstack(self.new_opp_index)
            #print(f"self.new_opp_index={X_train},y_train={self.new_y_train}")
            #self.new_opp_index=np.array(self.new_opp_index,dtype=np.float64).reshape(1,-1)
            #self.new_y_train=np.array(self.new_y_train,dtype=np.float64).reshape(1,-1)
            self.model.fit(X_train,self.new_y_train)


    def on_negotiation_end(self, state):
        if state.current_offer is None:
            return
        y_train=random.uniform(1.0,0.7)
        #新しい相手効用のタプルリスト
        
        for _ in self.nmi.outcome_space.enumerate_or_sample():
                if _ == state.current_offer:
                    target = _
                    break  
        if target is not None:
                part_a = np.array(self.df_encoded.loc[[target]].values,dtype=np.float64) 
                part_b = np.array(self.ufun(target),dtype=np.float64).reshape(1,-1)
                new_row = np.hstack([part_a, part_b])
                new_row=np.array(new_row,np.float64).reshape(1,-1).flatten()
                self.new_opp_index.append(new_row)
        self.new_y_train.append(y_train)
        X_train=np.vstack(self.new_opp_index)
        #print(f"self.new_opp_index={X_train},y_train={self.new_y_train}")
        #self.new_opp_index=np.array(self.new_opp_index,dtype=np.float64).reshape(1,-1)
        #self.new_y_train=np.array(self.new_y_train,dtype=np.float64).reshape(1,-1)
        self.model.fit(X_train,self.new_y_train)
        self.private_info["opponent_ufun"].update_model(self.model)
        return super().on_negotiation_end(state)
    
    def find_nash_bargaining_solution(self,reservation):
        """
        Args:
            outcomes: Outcomeのリスト
            utilities: 各Outcomeに対する、自分と相手の効用のリスト [((u_self, u_opp), ...)]
            reservation_self: 自分の保留価値 (これ以下なら交渉決裂)
            reservation_opp: 相手の保留価値
        Returns:
            Nash Bargaining Solution となる Outcome
        """
        max_nash_product = -float('inf')
        best_outcome = None
        u_selfs=[self.ufun(_) for _ in self.enumerate_outcome]
        u_opps=self.model.predict(self.vectirize_self_ufun)

        for  u_self,u_opp,outcome in zip(u_selfs,u_opps,self.enumerate_outcome):
            # 1. 保留価値を上回っているかチェック (交渉成立の条件)
            if u_self > reservation and  u_opp> reservation:
                # 2. ナッシュ積 (Nash Product) を計算
                # (自分の効用 - 自分の保留価値) * (相手の効用 - 相手の保留価値)
                nash_product = (u_self - reservation) * (u_opp - reservation)
                
                if nash_product > max_nash_product:
                    max_nash_product = nash_product
                    best_outcome = outcome
                    
        return best_outcome
        

    


class PredictiveUfun(BaseMultiFun):
    """
    予測モデルの出力を、neg_masのUfunとして機能させるクラス
    """
    f: Callable[[Any], float]
    min_value: float | None = None
    max_value: float | None = None

    def __init__(self, model, outcome_space):
        """
        :param model: 予測モデル (sklearn, XGBoost, etc.)
                    predict(X) メソッドを持つもの
        :param outcome_space: 評価対象となるアウトカムの集合
        """
        self.model = model
        self.outcome_space = outcome_space
        #データフレーム（outcome_space変換用のデータモデル）+対応するタプルの効用値
        self.df:pd.DataFrame | None = None
        self.ufun=None
        self.opponent_index_data=None

        # 親クラスの初期化
        # 予測モデルが返す値の範囲（報酬の最小・最大）をあらかじめ計算しておく必要がある
        # もしくは、とりあえず暫定的な範囲を渡す
        #super().__init__(f=self.__call__)

    def update_model(self,model):
        self.model=model
    
    def update_oppnent(self,opponent):
        self.opponent_index_data=opponent
    
    def update_df(self,df:pd.DataFrame,ufun):
        self.df=df
        self.ufun=ufun
    
    def update_date(self,data):
        opponent_predict=self.model.predict(data)
        self.opponent_index_data = [(outcome,x2,count) for (outcome,_,count),x2 in zip(self.opponent_index_data,opponent_predict)]



    
    def __call__(self, x: tuple) -> float:
        """
        単一のoutcome（タプル）を、モデルが理解できる形式（2D配列）に変換して予測
        データモデル(self.df)を使ってタプルを変換する
        """
        # xは(x1, x2, ...) というタプル
        for outcome,ufun,_ in self.opponent_index_data:
            if x==outcome:
                ret=ufun


        # 結果をスカラーとして返す
        return float(ret)
    
    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        """Find the min/max values, using cached bounds if available."""
        return self._minmax(input)


    def _minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        if self.min_value is not None and self.max_value is not None:
            return self.min_value, self.max_value
        mn, mx = nonmonotonic_multi_minmax(input, self.f)
        if self.min_value is not None:
            mn = min(mn, self.min_value)
        if self.max_value is not None:
            mx = min(mx, self.max_value)
        return mn, mx
    
    def dim(self) -> int:
        """Return the number of issues (not implemented)."""
        raise NotImplementedError()
    
    def shift_by(self, offset: float) -> AffineMultiFun:
        """Shift operation (not implemented for lambda functions)."""
        raise NotImplementedError()

    def scale_by(self, scale: float) -> LinearMultiFun:
        """Scale operation (not implemented for lambda functions)."""
        raise NotImplementedError()
    
    def xml(self, indx, issues, bias: float = 0) -> str:
        """Export to GENIUS XML format (not implemented)."""
        raise NotImplementedError()