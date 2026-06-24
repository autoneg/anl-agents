from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

from negmas.outcomes import Outcome
from negmas.sao.controllers import SAOState
from negmas import ResponseType
from negmas.sao import SAONegotiator
from negmas.preferences import (
    WeightedUtilityFunction,
    ConstUtilityFunction,
    MappingUtilityFunction,
)

__all__ = ["TjAgent"]


EXCEPTIONAL_U = 0.95
MICRO_SPACE_MAX = 3
ANTICIP_MIN_TIME = 0.85
ANTICIP_U_FRAC = 0.12
LAST_N_STEPS = 1
STALEMATE_MIN_TIME = 0.92
STALEMATE_U_ABOVE_RV = 0.08

#
PARETO_MIN_TIME = 0.70
PARETO_MIN_SPACE = 40
PARETO_MIN_CANDIDATES = 5
LATE_OPP_TIME = 0.97
LATE_OPP_POW = 1.15


MAPPING_EMA_ALPHA = 0.35
WEIGHT_EMA_ALPHA = 0.25


class TjAgent(SAONegotiator):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent_history: list[Outcome] = []
        self.issue_value_counts: list[defaultdict] = []
        self.n_issue_values: list[int] = []
        self.predicted_weights: list[float] = []
        self.sorted_outcomes: list[tuple[float, Outcome]] = []
        self.all_outcomes: list[Outcome] = []
        self.max_u = 1.0
        self.exponent = 6.5
        self.is_deadlocked = False
        self.space_size = 0
        self.num_issues = 0
        self.concession_rate = 0.0
        self._micro_min_rational_u = 0.0
        self._pending_bid: Outcome | None = None
        self._mapping_ema: dict[Outcome, float] | None = None

    def on_preferences_changed(self, changes):

        if self.ufun is None:
            return

        self.opponent_history = []
        self.is_deadlocked = False
        self.concession_rate = 0.0
        self._pending_bid = None
        self._mapping_ema = None

        self.all_outcomes = list(self.nmi.outcome_space.enumerate_or_sample())


        valid_outcomes = []
        for o in self.all_outcomes:
            u_val = float(self.ufun(o))
            if u_val >= self.ufun.reserved_value:
                valid_outcomes.append((u_val, o))

        self.sorted_outcomes = sorted(valid_outcomes, key=lambda x: x[0], reverse=True)

        if not self.sorted_outcomes:
            return

        self.max_u = self.sorted_outcomes[0][0]
        self.num_issues = len(self.sorted_outcomes[0][1])

        self.predicted_weights = []
        for _ in range(self.num_issues):
            self.predicted_weights.append(1.0 / self.num_issues)

        issues = self.nmi.outcome_space.issues
        self.n_issue_values = []
        for iss in issues:
            self.n_issue_values.append(len(iss.values))

        self.issue_value_counts = []
        for _ in range(self.num_issues):
            self.issue_value_counts.append(defaultdict(int))

        self.space_size = len(self.sorted_outcomes)
        rv = float(self.ufun.reserved_value)


        above_rv = []
        for u, _ in self.sorted_outcomes:
            if u > rv + 1e-9:
                above_rv.append(u)

        if above_rv:
            lowest_above = above_rv[0]
            for val in above_rv:
                if val < lowest_above:
                    lowest_above = val
            self._micro_min_rational_u = lowest_above
        else:
            self._micro_min_rational_u = rv

        self.exponent = 9.0 if self.space_size <= 50 else 6.5

        # 逆向映射反推对手基础 Ufun
        inv = self.ufun.invert()
        rng = inv.max() - inv.min()
        if rng == 0:
            self.private_info["opponent_ufun"] = self.ufun
        else:
            self.private_info["opponent_ufun"] = WeightedUtilityFunction(
                [
                    ConstUtilityFunction(inv.max(), outcome_space=self.ufun.outcome_space),
                    self.ufun,
                ],
                weights=(1 / rng, -1 / rng),
                reserved_value=self.ufun.reserved_value,
            )

    def _remaining_steps(self, state: SAOState) -> int | None:
        nsteps = self.nmi.n_steps
        if nsteps is None:
            return None
        return max(0, nsteps - state.step)

    def _is_last_n_steps(self, state: SAOState, n: int = 1) -> bool:
        rem = self._remaining_steps(state)
        if rem is not None:
            return rem <= n
        t = state.relative_time
        if n <= 1:
            return t >= 0.995
        return t >= max(0.0, 1.0 - 0.01 * n)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(state, offer)
        self._pending_bid = self._concealing_bidding_strategy(state)

        if self._acceptance_strategy(state, offer, self._pending_bid):
            self._pending_bid = None
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if self._pending_bid is not None:
            bid = self._pending_bid
            self._pending_bid = None
            return bid
        return self._concealing_bidding_strategy(state)

    def _get_dynamic_exponent(self) -> float:

        base_exp = 9.0 if self.space_size <= 50 else 6.5

        if self.concession_rate <= 1e-5:

            return base_exp + 3.0
        elif self.concession_rate > 0.015:

            return max(3.5, base_exp - 2.5)

        return base_exp

    def _get_target_utility(self, state: SAOState) -> float:

        t = state.relative_time
        p0 = self.max_u
        rv = float(self.ufun.reserved_value)

        self.exponent = self._get_dynamic_exponent()
        span = p0 - rv

        target_u = p0 - span * (t ** self.exponent)
        if self.is_deadlocked:
            target_u *= 0.85


        if t >= 0.90:
            progress = min(1.0, (t - 0.90) / (0.995 - 0.90))
            start_frac = 0.20


            rate_factor = self.concession_rate / 0.015
            if rate_factor > 1.0:
                rate_factor = 1.0
            if rate_factor < 0.0:
                rate_factor = 0.0

            if self.is_deadlocked:
                end_frac = 0.15
            else:
                end_frac = 0.15 - (0.15 - 0.005) * rate_factor

            current_frac = start_frac - (start_frac - end_frac) * progress
            dynamic_floor = rv + current_frac * span

            if target_u < dynamic_floor:
                target_u = dynamic_floor

        buffer = 0.05 if t > 0.95 else 0.0
        return max(target_u, rv + buffer)

    def _reject_micro_corner_offer(self, u: float, t: float) -> bool:
        if self.space_size > MICRO_SPACE_MAX or t < 0.995:
            return False
        return u < self._micro_min_rational_u - 1e-9

    def _anticip_min_u(self) -> float:
        rv = float(self.ufun.reserved_value)
        span = max(0.0, self.max_u - rv)
        return rv + ANTICIP_U_FRAC * span

    def _acceptance_strategy(
            self,
            state: SAOState,
            offer: Outcome,
            next_bid: Outcome | None = None,
    ) -> bool:

        u = float(self.ufun(offer))
        rv = float(self.ufun.reserved_value)
        t = state.relative_time


        if u < rv:
            return False


        if self._reject_micro_corner_offer(u, t):
            return False


        if u >= EXCEPTIONAL_U:
            return True


        if not self.sorted_outcomes:
            return u >= rv


        target_now = self._get_target_utility(state)
        if u >= target_now:
            return True


        if next_bid is not None and t > ANTICIP_MIN_TIME:
            u_next = float(self.ufun(next_bid))
            anticip_floor = self._anticip_min_u()
            buffer = 0.05 if t > 0.95 else 0.0
            if u >= u_next and u >= anticip_floor and u >= rv + buffer:
                return True


        if (
                t > STALEMATE_MIN_TIME
                and len(self.opponent_history) >= 2
                and self.opponent_history[-1] == offer
                and u >= rv + STALEMATE_U_ABOVE_RV
        ):
            return True


        if t >= 0.90:
            span = self.max_u - rv
            progress = min(1.0, (t - 0.90) / (0.995 - 0.90))
            start_frac = 0.20

            rate_factor = self.concession_rate / 0.015
            if rate_factor > 1.0:
                rate_factor = 1.0
            if rate_factor < 0.0:
                rate_factor = 0.0

            if self.is_deadlocked:
                end_frac = 0.15
            else:
                end_frac = 0.15 - (0.15 - 0.005) * rate_factor

            current_frac = start_frac - (start_frac - end_frac) * progress
            dynamic_floor = rv + current_frac * span
            if u >= dynamic_floor:
                return True


        if self._is_last_n_steps(state, LAST_N_STEPS) and u > rv:
            return True


        future_state = copy.deepcopy(state)
        future_state.relative_time = min(1.0, t + 0.02)
        next_target = self._get_target_utility(future_state)
        margin = 0.05 if t > 0.9 else 0.0
        return u >= (next_target - margin)

    def _compute_shannon_weights(self) -> list[float]:

        new_w = []
        for _ in range(self.num_issues):
            new_w.append(0.0)

        for i in range(self.num_issues):
            total = 0
            for c in self.issue_value_counts[i].values():
                total += c

            if total == 0:
                new_w[i] = 1.0 / self.num_issues
                continue

            entropy = 0.0
            n_vals = max(1, self.n_issue_values[i])
            observed_count = 0

            for _val, c in self.issue_value_counts[i].items():
                p = (c + 1.0) / (total + n_vals)
                entropy -= p * math.log(p + 1e-9)
                observed_count += 1

            unobserved_count = n_vals - observed_count
            if unobserved_count > 0:
                p_unobserved = 1.0 / (total + n_vals)
                entropy -= unobserved_count * (p_unobserved * math.log(p_unobserved + 1e-9))

            new_w[i] = 1.0 / (entropy + 0.5)

        w_sum = 0.0
        for w in new_w:
            w_sum += w

        if w_sum > 0:
            normalized_w = []
            for w in new_w:
                normalized_w.append(w / w_sum)
            new_w = normalized_w
        else:
            new_w = []
            for _ in range(self.num_issues):
                new_w.append(1.0 / self.num_issues)

        alpha = WEIGHT_EMA_ALPHA
        if len(self.predicted_weights) == self.num_issues:
            blended = []
            for i in range(self.num_issues):
                blended_val = (1.0 - alpha) * self.predicted_weights[i] + alpha * new_w[i]
                blended.append(blended_val)
        else:
            blended = new_w

        s = 0.0
        for w in blended:
            s += w

        if s > 0:
            final_w = []
            for w in blended:
                final_w.append(w / s)
            return final_w
        return new_w

    def _update_opponent_model(self, state: SAOState, offer: Outcome):

        self.opponent_history.append(offer)
        for i, val in enumerate(offer):
            self.issue_value_counts[i][val] += 1


        opp_utils = []
        for hist_offer in self.opponent_history:
            opp_utils.append(float(self.ufun(hist_offer)))

        window_size = 5
        if len(opp_utils) >= window_size * 2:
            recent_sum = 0.0
            for i in range(len(opp_utils) - window_size, len(opp_utils)):
                recent_sum += opp_utils[i]
            older_sum = 0.0
            for i in range(len(opp_utils) - window_size * 2, len(opp_utils) - window_size):
                older_sum += opp_utils[i]
            self.concession_rate = (recent_sum - older_sum) / window_size
        else:
            self.concession_rate = 0.0

        if not self.sorted_outcomes:
            return

        self.predicted_weights = self._compute_shannon_weights()

        if self.all_outcomes:
            raw = {}
            first_o = self.all_outcomes[0]
            first_est = float(self._est_opp_u(first_o))
            min_raw = first_est
            max_raw = first_est

            for o in self.all_outcomes:
                est = float(self._est_opp_u(o))
                raw[o] = est
                if est < min_raw:
                    min_raw = est
                if est > max_raw:
                    max_raw = est

            span = max_raw - min_raw if max_raw > min_raw else 1.0

            mapping_dict = {}
            for o in self.all_outcomes:
                mapping_dict[o] = (raw[o] - min_raw) / span

            alpha = MAPPING_EMA_ALPHA
            if self._mapping_ema is None:
                self._mapping_ema = {}
                for k, v in mapping_dict.items():
                    self._mapping_ema[k] = v
            else:
                for o in self.all_outcomes:
                    old = self._mapping_ema.get(o, mapping_dict[o])
                    self._mapping_ema[o] = (1.0 - alpha) * old + alpha * mapping_dict[o]

            try:
                self.private_info["opponent_ufun"] = MappingUtilityFunction(
                    dict(self._mapping_ema),
                    outcome_space=self.ufun.outcome_space,
                )
            except TypeError:
                self.private_info["opponent_ufun"] = MappingUtilityFunction(
                    dict(self._mapping_ema)
                )

        # 僵局触发器
        if (
                2 <= self.space_size <= 15
                and state.relative_time > 0.85
                and len(self.opponent_history) >= 6
        ):
            recent_set = set()
            for hist_o in self.opponent_history[-6:]:
                recent_set.add(hist_o)

            if len(recent_set) <= 2 and not self.is_deadlocked:
                self.is_deadlocked = True

    def _est_opp_u(self, outcome: Outcome) -> float:
        res = 0.0
        total = max(1, len(self.opponent_history))
        for i, v in enumerate(outcome):
            n_vals = max(1, self.n_issue_values[i])
            val_freq = (self.issue_value_counts[i].get(v, 0) + 1.0) / (total + n_vals)
            res += val_freq * self.predicted_weights[i]
        return res

    def _filter_estimated_pareto(self, candidates: list[Outcome]) -> list[Outcome]:
        if len(candidates) <= 1:
            return candidates

        scored = []
        for o in candidates:
            scored.append((float(self.ufun(o)), self._est_opp_u(o), o))

        pareto: list[Outcome] = []
        for i, (u1, e1, o1) in enumerate(scored):
            dominated = False
            for j, (u2, e2, _o2) in enumerate(scored):
                if i == j:
                    continue
                if u2 >= u1 and e2 >= e1 and (u2 > u1 or e2 > e1):
                    dominated = True
                    break
            if not dominated:
                pareto.append(o1)
        return pareto

    def _concealing_bidding_strategy(self, state: SAOState) -> Outcome:
        """带安全随机柔性噪声的自适应出价机制。"""
        if not self.sorted_outcomes:
            return state.current_offer or self.nmi.outcome_space.sample()

        target = self._get_target_utility(state)
        t = state.relative_time

        # 核心改进 2：基于空间密集程度和谈判进度的动态自适应搜索带宽
        if self.space_size <= 20:
            current_band = 0.08  # 稀疏场景：大带宽扩张防止候选池为空
        else:
            current_band = max(0.015, 0.035 * (1.0 - t))  # 稠密场景：随时间向目标精准收窄

        candidates = []
        for u, o in self.sorted_outcomes:
            if u >= target - 0.005 and u <= target + current_band:
                candidates.append(o)

        if not candidates:
            valid_above = []
            for u, o in self.sorted_outcomes:
                if u >= target - 0.02:
                    valid_above.append((u, o))

            if valid_above:
                closest = valid_above[0]
                min_diff = abs(valid_above[0][0] - target)
                for item in valid_above:
                    diff = abs(item[0] - target)
                    if diff < min_diff:
                        min_diff = diff
                        closest = item
                candidates = [closest[1]]
            else:
                candidates = [self.sorted_outcomes[0][1]]

        if (
                t > PARETO_MIN_TIME
                and self.space_size > PARETO_MIN_SPACE
                and len(candidates) >= PARETO_MIN_CANDIDATES
        ):
            pareto_c = self._filter_estimated_pareto(candidates)
            if pareto_c:
                candidates = pareto_c

        temp = max(0.035, 0.1 * (1.0 - t))
        opp_pow = LATE_OPP_POW if t > LATE_OPP_TIME else 1.0

        probs = []
        for c in candidates:
            eu = self._est_opp_u(c) + 0.01
            score = (float(self.ufun(c)) * (eu ** opp_pow)) / temp
            probs.append(math.exp(min(score, 500)))

        return random.choices(candidates, weights=probs, k=1)[0]