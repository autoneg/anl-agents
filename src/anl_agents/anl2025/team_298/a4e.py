"""
**Submitted to ANAC 2025 Automated Negotiation League**

- Agent Name: a4e
- Team Name: Team 298
- Contact Email: s224631q@st.go.tuat.ac.jp
- Affiliation: Tokyo University of Agriculture and Technology
- Country: Japan
- Team Members:
  1. Kazuma Mochizuki<s224631q@st.go.tuat.ac.jp>

*Team* Team 298
*Authors* Kazuma Mochizuki
s224631q@st.go.tuat.ac.jp

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

from anl2025.ufun import SideUFun
from negmas import SAONMI, InverseUFun, PolyAspiration, PresortingInverseUtilityFunction
from negmas.sao.controllers import SAOState
from negmas import ResponseType, Outcome
import numpy as np
from typing import Literal, Dict, List, Tuple, Optional
from random import random, choice
from anl2025.negotiator import ANL2025Negotiator

__all__ = ["A4E"]


class A4E(ANL2025Negotiator):
    """
    攻撃的Individual戦略
    - 競争的相手に対して高い効用を追求
    - 協調的相手に対してのみNash均衡を考慮
    """

    def __init__(
        self,
        *args,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | Literal["hardheaded"]
        | float = "boulware",
        deltas: tuple[float, ...] = (1e-3, 5e-2, 1e-1, 2e-1, 4e-1, 6e-1),
        reject_exactly_as_reserved: bool = False,
        nash_weight: float = 0.2,
        individual_weight: float = 0.8,
        adaptation_rate: float = 0.2,
        risk_threshold: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._curve = PolyAspiration(1.0, aspiration_type)
        self._inverter: Dict[str, InverseUFun] = dict()
        self._nash_weight = nash_weight
        self._individual_weight = individual_weight
        self._adaptation_rate = adaptation_rate
        self._risk_threshold = risk_threshold
        self._best: List[Outcome] = None
        self._mx: float = 1.0
        self._mn: float = 0.0
        self._deltas = deltas
        self._best_margin = 1e-8
        self.reject_exactly_as_reserved = reject_exactly_as_reserved
        self._current_strategy: Dict[str, str] = {}
        self._opponent_type: Dict[str, str] = {}
        self._strategy_performance: Dict[str, Dict[str, float]] = {}
        self._opponent_utility_history: Dict[str, List[Tuple[Outcome, float]]] = {}
        self._nash_candidates: Dict[str, List[Outcome]] = {}
        self._pareto_frontier: Dict[str, List[Outcome]] = {}
        self._opponent_weakness_score: Dict[str, float] = {}
        self._aggression_level: Dict[str, float] = {}
        self._exploitation_opportunities: Dict[str, int] = {}
        self._negotiation_risk: Dict[str, float] = {}
        self._fallback_strategies: Dict[str, List[str]] = {}

    def ensure_inverter(self, negotiator_id) -> InverseUFun:
        if self._inverter.get(negotiator_id, None) is None:
            negotiator, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]
            inverter = PresortingInverseUtilityFunction(ufun, rational_only=True)
            inverter.init()
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
                self._best = [inverter.best()]
            self._inverter[negotiator_id] = inverter
            self._initialize_hybrid_strategy(negotiator_id, ufun, inverter)
        return self._inverter[negotiator_id]

    def _initialize_hybrid_strategy(
        self, negotiator_id: str, ufun: SideUFun, inverter: InverseUFun
    ):
        self._current_strategy[negotiator_id] = "hybrid"
        self._opponent_type[negotiator_id] = "unknown"
        self._strategy_performance[negotiator_id] = {
            "nash": 0.0,
            "individual": 0.0,
            "hybrid": 0.0,
        }
        self._opponent_utility_history[negotiator_id] = []
        self._pareto_frontier[negotiator_id] = []
        for utility_level in np.linspace(0.7, 1.0, 5):
            candidates = inverter.some(
                (utility_level - 0.1, utility_level), normalized=True
            )
            if candidates:
                self._pareto_frontier[negotiator_id].extend(candidates[:2])
        self._nash_candidates[negotiator_id] = self._pareto_frontier[
            negotiator_id
        ].copy()
        self._opponent_weakness_score[negotiator_id] = 0.5
        self._aggression_level[negotiator_id] = 0.6
        self._exploitation_opportunities[negotiator_id] = 0
        self._negotiation_risk[negotiator_id] = 0.0
        self._fallback_strategies[negotiator_id] = ["nash", "individual", "hybrid"]

    def _analyze_opponent_type(self, negotiator_id: str, state: SAOState) -> str:
        if state.step < 5:
            return "unknown"
        if negotiator_id not in self._opponent_utility_history:
            return "unknown"
        history = self._opponent_utility_history[negotiator_id]
        if len(history) < 3:
            return "unknown"
        recent_utilities = [util for _, util in history[-5:]]
        avg_utility = np.mean(recent_utilities)
        utility_trend = np.polyfit(range(len(recent_utilities)), recent_utilities, 1)[0]
        if avg_utility > 0.6 and utility_trend > 0:
            return "cooperative"
        elif avg_utility < 0.4 and utility_trend < 0:
            return "competitive"
        else:
            return "balanced"

    def _calculate_negotiation_risk(self, negotiator_id: str, state: SAOState) -> float:
        base_risk = 0.0
        if state.relative_time > 0.8:
            base_risk += 0.3
        elif state.relative_time > 0.6:
            base_risk += 0.1
        opponent_type = self._opponent_type.get(negotiator_id, "unknown")
        if opponent_type == "competitive":
            base_risk += 0.2
        elif opponent_type == "unknown":
            base_risk += 0.1
        history = self._opponent_utility_history.get(negotiator_id, [])
        if len(history) > 5:
            utilities = [util for _, util in history[-5:]]
            variability = np.std(utilities)
            base_risk += variability * 0.2  # 高い変動はリスク増加
        performance = self._strategy_performance[negotiator_id]
        current_strategy = self._current_strategy.get(negotiator_id, "hybrid")
        if performance[current_strategy] < -0.1:
            base_risk += 0.15
        return min(1.0, base_risk)

    def _select_optimal_strategy(self, negotiator_id: str, state: SAOState) -> str:
        opponent_type = self._analyze_opponent_type(negotiator_id, state)
        self._opponent_type[negotiator_id] = opponent_type
        risk_level = self._calculate_negotiation_risk(negotiator_id, state)
        self._negotiation_risk[negotiator_id] = risk_level
        if state.relative_time < 0.3:
            return "nash" if opponent_type == "cooperative" else "individual"
        elif state.relative_time < 0.7:
            if risk_level > self._risk_threshold:
                performance = self._strategy_performance[negotiator_id]
                return max(performance, key=performance.get)
            else:
                if opponent_type == "cooperative":
                    return "nash"
                else:
                    return "individual"
        else:
            return "individual" if risk_level > 0.5 else "hybrid"

    def _execute_nash_strategy(
        self,
        negotiator_id: str,
        state: SAOState,
        inverter: InverseUFun,
        ufun: SideUFun,
        level: float,
    ) -> Optional[Outcome]:
        nash_candidates = self._nash_candidates.get(negotiator_id, [])
        if not nash_candidates:
            return None
        valid_candidates = [
            outcome
            for outcome in nash_candidates
            if float(ufun(outcome)) >= level * 0.95
        ]
        if valid_candidates:
            best_outcome = None
            best_nash_product = 0.0
            for outcome in valid_candidates:
                my_utility = float(ufun(outcome))
                estimated_opponent_utility = self._estimate_opponent_utility(
                    negotiator_id, outcome
                )
                nash_product = my_utility * estimated_opponent_utility
                if nash_product > best_nash_product:
                    best_nash_product = nash_product
                    best_outcome = outcome
            return best_outcome
        return None

    def _execute_individual_strategy(
        self,
        negotiator_id: str,
        state: SAOState,
        inverter: InverseUFun,
        ufun: SideUFun,
        level: float,
    ) -> Optional[Outcome]:
        aggression = self._aggression_level.get(negotiator_id, 0.6)
        history = self._opponent_utility_history.get(negotiator_id, [])
        if len(history) >= 5:
            utilities = [util for _, util in history[-5:]]
            concession_rate = -np.polyfit(range(len(utilities)), utilities, 1)[0]
            self._aggression_level[negotiator_id] = min(
                1.0, aggression + concession_rate * 0.5
            )
        weakness_score = self._opponent_weakness_score.get(negotiator_id, 0.5)
        aggressive_level = level + (aggression * 0.4) + (weakness_score - 0.5) * 0.3
        aggressive_level = min(1.3, max(0.1, aggressive_level))
        for d in [0.1, 0.2, 0.3]:
            mx = min(1.3, aggressive_level + d)
            outcome = inverter.one_in((aggressive_level, mx), normalized=True)
            if outcome:
                return outcome
        return None

    def _execute_hybrid_strategy(
        self,
        negotiator_id: str,
        state: SAOState,
        inverter: InverseUFun,
        ufun: SideUFun,
        level: float,
    ) -> Optional[Outcome]:
        nash_outcome = self._execute_nash_strategy(
            negotiator_id, state, inverter, ufun, level
        )
        individual_outcome = self._execute_individual_strategy(
            negotiator_id, state, inverter, ufun, level
        )
        if nash_outcome and individual_outcome:
            nash_utility = float(ufun(nash_outcome))
            individual_utility = float(ufun(individual_outcome))
            current_nash_weight = self._nash_weight
            current_individual_weight = self._individual_weight
            opponent_type = self._opponent_type.get(negotiator_id, "balanced")
            if opponent_type == "cooperative":
                current_nash_weight += 0.3
                current_individual_weight -= 0.3
            nash_score = nash_utility * current_nash_weight
            individual_score = individual_utility * current_individual_weight
            return (
                nash_outcome if nash_score >= individual_score else individual_outcome
            )
        return nash_outcome or individual_outcome

    def _estimate_opponent_utility(self, negotiator_id: str, outcome: Outcome) -> float:
        history = self._opponent_utility_history.get(negotiator_id, [])
        if not history:
            return 0.5
        similar_utilities = [
            util
            for hist_outcome, util in history
            if self._calculate_similarity(outcome, hist_outcome) > 0.6
        ]
        if similar_utilities:
            return np.mean(similar_utilities)
        else:
            return np.mean([util for _, util in history])

    def _calculate_similarity(self, outcome1: Outcome, outcome2: Outcome) -> float:
        if outcome1 == outcome2:
            return 1.0
        return random() * 0.4 + 0.3

    def calc_level(self, nmi: SAONMI, state: SAOState, normalized: bool):
        if state.step == 0:
            level = 1.0
        elif nmi.n_steps is not None and state.step >= nmi.n_steps - 1:
            level = 0.2
        else:
            level = self._curve.utility_at(state.relative_time)
        if not normalized:
            level = level * (self._mx - self._mn) + self._mn
        return level

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        assert self.ufun
        negotiator, cntxt = self.negotiators[negotiator_id]
        inverter = self.ensure_inverter(negotiator_id)
        nmi = negotiator.nmi
        level = self.calc_level(nmi, state, normalized=True)
        ufun: SideUFun = cntxt["ufun"]
        if self._mx < float(ufun(None)):
            return None
        selected_strategy = self._select_optimal_strategy(negotiator_id, state)
        self._current_strategy[negotiator_id] = selected_strategy
        outcome = None
        if selected_strategy == "nash":
            outcome = self._execute_nash_strategy(
                negotiator_id, state, inverter, ufun, level
            )
        elif selected_strategy == "individual":
            outcome = self._execute_individual_strategy(
                negotiator_id, state, inverter, ufun, level
            )
        else:
            outcome = self._execute_hybrid_strategy(
                negotiator_id, state, inverter, ufun, level
            )
        if not outcome:
            for d in self._deltas:
                mx = min(1.0, level + d)
                outcome = inverter.one_in((level, mx), normalized=True)
                if outcome:
                    break
        if not outcome:
            outcome = choice(self._best)
        return outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        assert self.ufun
        negotiator, cntxt = self.negotiators[negotiator_id]
        ufun: SideUFun = cntxt["ufun"]
        nmi = self.negotiators[negotiator_id][0].nmi
        self.ensure_inverter(negotiator_id)
        current_offer = state.current_offer
        if current_offer is None:
            return ResponseType.REJECT_OFFER
        my_utility = float(ufun(current_offer))
        estimated_opponent_utility = self._estimate_opponent_utility(
            negotiator_id, current_offer
        )
        self._opponent_utility_history[negotiator_id].append(
            (current_offer, estimated_opponent_utility)
        )
        if self._mx < float(ufun(None)):
            return ResponseType.END_NEGOTIATION
        level = self.calc_level(nmi, state, normalized=False)
        current_strategy = self._current_strategy.get(negotiator_id, "hybrid")
        if current_strategy == "nash":
            nash_product = my_utility * estimated_opponent_utility
            nash_bonus = 0.1 * level if nash_product > 0.4 else 0.0
            effective_level = level - nash_bonus
        elif current_strategy == "individual":
            aggression = self._aggression_level.get(negotiator_id, 0.6)
            effective_level = level * (1 + aggression * 0.5)
        else:
            nash_component = my_utility * estimated_opponent_utility * 0.1
            individual_component = level * 0.15
            effective_level = level - (nash_component + individual_component) * 0.5
        if (self.reject_exactly_as_reserved and effective_level >= my_utility) or (
            not self.reject_exactly_as_reserved and effective_level > my_utility
        ):
            return ResponseType.REJECT_OFFER
        else:
            return ResponseType.ACCEPT_OFFER

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        for side in self.negotiators.keys():
            if side == negotiator_id:
                continue
            if side in self._inverter:
                del self._inverter[side]
        if negotiator_id in self._current_strategy:
            strategy = self._current_strategy[negotiator_id]
            performance_score = random() - 0.5
            self._strategy_performance[negotiator_id][strategy] += (
                performance_score * self._adaptation_rate
            )


if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(A4E, small=True)
