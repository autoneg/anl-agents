from __future__ import annotations

import math
from dataclasses import dataclass

from negmas.preferences.ops import nash_points, pareto_frontier
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState

try:
    from .agent_utils import ToolkitNegotiator, build_additive_model, normalize_mapping
except ImportError:
    from agent_utils import ToolkitNegotiator, build_additive_model, normalize_mapping


@dataclass
class BeliefHypothesis:
    value_tables: list[dict]
    weights: list[float]
    score: float = 1.0


class HybridNegotiator(ToolkitNegotiator):
    """A hybrid agent combining frequency learning, mirror priors, and adaptive concealment."""

    def __init__(
        self,
        *args,
        self_weight: float = 0.48,
        opp_weight: float = 0.16,
        diversity_weight: float = 0.07277078888051938,
        reciprocity_weight: float = 0.08659223556421602,
        balance_weight: float = 0.09,
        reveal_weight: float = 0.138,
        mirror_blend: float = 0.21,
        acceptance_slack_scale: float = 0.033,
        nash_weight: float = 0.16,
        pareto_weight: float = 0.11,
        future_lookahead_weight: float = 0.15,
        reserve_floor_bias: float = 0.009,
        belief_blend: float = 0.0,
        robust_weight: float = 0.0,
        zone_weight: float = 0.0,
        probe_weight: float = 0.0,
        cooperation_weight: float = 0.01,
        transition_blend: float = 0.11,
        plateau_weight: float = 0.0,
        dynamic_belief_weight: float = 0.03,
        pool_mix_weight: float = 0.22,
        small_domain_relaxation: float = 0.0,
        leakage_weight: float = 0.0,
        exploration_width: float = 0.0,
        single_issue_belief_blend: float = 0.0,
        single_issue_acceptance_scale: float = 0.028,
        exact_frontier_threshold: int = 0,
        frontier_refresh_interval: int = 2,
        frontier_reference_size: int = 180,
        late_alignment_weight: float = 0.03,
        late_alignment_start: float = 0.84,
        near_best_acceptance_margin: float = 0.012,
        recency_alignment_weight: float = 0.06,
        recent_offer_pool_weight: float = 0.12,
        target_toughness_bias: float = 0.0,
        stubborn_acceptance_bias: float = 0.0,
        stubborn_match_weight: float = 0.015,
        reciprocal_pool_weight: float = 0.05,
        reciprocal_alignment_scale: float = 0.16,
        hardline_target_discount: float = 0.035,
        hardline_acceptance_scale: float = 0.006,
        deadlock_offer_weight: float = 0.022,
        deadlock_acceptance_scale: float = 0.005,
        hardline_alignment_weight: float = 0.0,
        micron_acceptance_scale: float = 0.0,
        reciprocal_endgame_weight: float = 0.0,
        reciprocal_target_discount: float = 0.0,
        reciprocal_acceptance_scale: float = 0.004,
        recent_pool_start: float = 0.52,
        hardline_pool_start: float = 0.78,
        large_domain_pool_scale: float = 0.65,
        value_decay: float = 1.0,
        recent_bonus_decay: float = 1.0,
        transition_bonus_decay: float = 1.0,
        adaptive_mirror_scale: float = 0.0,
        camouflage_weight: float = 0.028,
        camouflage_start: float = 0.0,
        camouflage_end: float = 0.58,
        camouflage_style_weight: float = 0.0,
        early_reveal_boost: float = 0.0,
        reveal_decay_power: float = 1.0,
        patience_target_discount: float = 0.022,
        patience_acceptance_scale: float = 0.006,
        patience_style_weight: float = 0.5,
        opportunism_target_bonus: float = 0.028,
        reservation_guard_margin: float = 0.0,
        target_late: float = 0.74,
        target_floor: float = 0.10,
        reported_smith_blend: float = 1.0,
        issue_lock_weight: float = 0.0,
        issue_lock_start: float = 0.58,
        best_received_pool_weight: float = 0.0,
        best_received_pool_start: float = 0.72,
        single_issue_compromise_start: float = 0.82,
        single_issue_accept_compromise_start: float = 0.88,
        single_issue_deadlock_window: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_weight = self_weight
        self.opp_weight = opp_weight
        self.diversity_weight = diversity_weight
        self.reciprocity_weight = reciprocity_weight
        self.balance_weight = balance_weight
        self.reveal_weight = reveal_weight
        self.mirror_blend = mirror_blend
        self.acceptance_slack_scale = acceptance_slack_scale
        self.nash_weight = nash_weight
        self.pareto_weight = pareto_weight
        self.future_lookahead_weight = future_lookahead_weight
        self.reserve_floor_bias = reserve_floor_bias
        self.belief_blend = belief_blend
        self.robust_weight = robust_weight
        self.zone_weight = zone_weight
        self.probe_weight = probe_weight
        self.cooperation_weight = cooperation_weight
        self.transition_blend = transition_blend
        self.plateau_weight = plateau_weight
        self.dynamic_belief_weight = dynamic_belief_weight
        self.pool_mix_weight = pool_mix_weight
        self.small_domain_relaxation = small_domain_relaxation
        self.leakage_weight = leakage_weight
        self.exploration_width = exploration_width
        self.single_issue_belief_blend = single_issue_belief_blend
        self.single_issue_acceptance_scale = single_issue_acceptance_scale
        self.exact_frontier_threshold = exact_frontier_threshold
        self.frontier_refresh_interval = max(1, frontier_refresh_interval)
        self.frontier_reference_size = max(32, frontier_reference_size)
        self.late_alignment_weight = late_alignment_weight
        self.late_alignment_start = late_alignment_start
        self.near_best_acceptance_margin = near_best_acceptance_margin
        self.recency_alignment_weight = recency_alignment_weight
        self.recent_offer_pool_weight = recent_offer_pool_weight
        self.target_toughness_bias = target_toughness_bias
        self.stubborn_acceptance_bias = stubborn_acceptance_bias
        self.stubborn_match_weight = stubborn_match_weight
        self.reciprocal_pool_weight = reciprocal_pool_weight
        self.reciprocal_alignment_scale = reciprocal_alignment_scale
        self.hardline_target_discount = hardline_target_discount
        self.hardline_acceptance_scale = hardline_acceptance_scale
        self.deadlock_offer_weight = deadlock_offer_weight
        self.deadlock_acceptance_scale = deadlock_acceptance_scale
        self.hardline_alignment_weight = hardline_alignment_weight
        self.micron_acceptance_scale = micron_acceptance_scale
        self.reciprocal_endgame_weight = reciprocal_endgame_weight
        self.reciprocal_target_discount = reciprocal_target_discount
        self.reciprocal_acceptance_scale = reciprocal_acceptance_scale
        self.recent_pool_start = recent_pool_start
        self.hardline_pool_start = hardline_pool_start
        self.large_domain_pool_scale = large_domain_pool_scale
        self.value_decay = max(0.7, min(1.0, value_decay))
        self.recent_bonus_decay = max(0.5, min(1.0, recent_bonus_decay))
        self.transition_bonus_decay = max(0.5, min(1.0, transition_bonus_decay))
        self.adaptive_mirror_scale = max(0.0, adaptive_mirror_scale)
        self.camouflage_weight = camouflage_weight
        self.camouflage_start = camouflage_start
        self.camouflage_end = max(camouflage_start + 1e-6, camouflage_end)
        self.camouflage_style_weight = max(0.0, min(1.0, camouflage_style_weight))
        self.early_reveal_boost = max(0.0, early_reveal_boost)
        self.reveal_decay_power = max(0.2, reveal_decay_power)
        self.patience_target_discount = max(0.0, patience_target_discount)
        self.patience_acceptance_scale = max(0.0, patience_acceptance_scale)
        self.patience_style_weight = max(0.0, min(1.0, patience_style_weight))
        self.opportunism_target_bonus = max(0.0, opportunism_target_bonus)
        self.reservation_guard_margin = max(0.0, reservation_guard_margin)
        self.target_late = max(0.08, min(0.95, target_late))
        self.target_floor = max(0.02, min(self.target_late, target_floor))
        self.reported_smith_blend = max(0.0, min(1.0, reported_smith_blend))
        self.issue_lock_weight = max(0.0, issue_lock_weight)
        self.issue_lock_start = issue_lock_start
        self.best_received_pool_weight = max(0.0, best_received_pool_weight)
        self.best_received_pool_start = best_received_pool_start
        self.single_issue_compromise_start = single_issue_compromise_start
        self.single_issue_accept_compromise_start = single_issue_accept_compromise_start
        self.single_issue_deadlock_window = single_issue_deadlock_window

    def initialize_strategy(self) -> None:
        self._value_counts = [{value: 1.0 for value in issue.all} for issue in self._issues]
        self._recent_bonus = [{value: 0.0 for value in issue.all} for issue in self._issues]
        self._transition_bonus = [{value: 0.0 for value in issue.all} for issue in self._issues]
        self._same = [1.0] * len(self._issues)
        self._change = [1.0] * len(self._issues)
        self._mirror_tables = [normalize_mapping(table, invert=True) for table in self._self_value_means]
        self._normalized_self_value_means = [normalize_mapping(table) for table in self._self_value_means]
        self._learned_tables = [{value: 0.5 for value in issue.all} for issue in self._issues]
        self._weights = [1.0 / max(1, len(self._issues))] * len(self._issues)
        self._reserve_estimate = 0.0
        self._recent_estimated_utilities: list[float] = []
        self._pareto_hint: dict[Outcome, tuple[float, float]] = {}
        self._frontier_outcomes: set[Outcome] = set()
        self._nash_target: tuple[float, float] | None = None
        self._hypotheses = self._build_hypotheses()
        self._belief_tables = [{value: 0.5 for value in issue.all} for issue in self._issues]
        self._belief_weights = [1.0 / max(1, len(self._issues))] * len(self._issues)
        self._best_opponent_offer: Outcome | None = None
        self._best_opponent_offer_utility = float("-inf")
        self._opponent_concession_rate = 0.0
        self._opponent_volatility = 0.0
        self._opponent_reciprocity = 0.0
        self._opponent_stubbornness = 0.5
        self._opponent_repeat_rate = 0.0
        self._deadlock_level = 0.0
        self._last_relative_time = 0.0
        self._no_improvement_streak = 0.0
        self._issue_lock_values = [None] * len(self._issues)
        self._issue_lock_strengths = [0.0] * len(self._issues)
        self._opponent_model_version = 0
        self._frontier_model_snapshot = -1
        self._frontier_refresh_count = 0
        self._frontier_reference_outcomes = self._build_frontier_reference_outcomes()
        self._cached_frontier_pairs: dict[Outcome, tuple[float, float]] = {}
        self._cached_frontier_outcomes: set[Outcome] = set()
        self._cached_nash_target: tuple[float, float] | None = None
        self._domain_density = len(self.rational_outcomes) / max(1, self.sample_limit)
        self._is_small_domain = len(self.rational_outcomes) <= min(800, self.sample_limit // 4)
        self._single_issue_mode = len(self._issues) == 1
        self._rebuild_model()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        _ = dest
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        self._last_relative_time = state.relative_time
        if offer is not None:
            prev = self._received_offers[-1] if self._received_offers else None
            self.register_received_offer(offer)
            self._update_model(offer, prev, state)
            next_offer = self.propose_offer(state)
            next_utility = self._utility_by_outcome.get(next_offer, self.reservation()) if next_offer is not None else self.reservation()
            if self.should_accept_offer(state, next_utility):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        else:
            next_offer = self.propose_offer(state)

        if next_offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        self.register_sent_offer(next_offer)
        return SAOResponse(ResponseType.REJECT_OFFER, next_offer)

    def on_negotiation_end(self, state: SAOState) -> None:
        super().on_negotiation_end(state)
        if self.reported_smith_blend <= 1e-9 or self.ufun is None:
            return
        reported_tables = []
        for index, counts in enumerate(self._value_counts):
            maximum = max(counts.values(), default=1.0)
            reported_tables.append(
                {
                    value: (1.0 - self.reported_smith_blend)
                    * self._strategy_model_tables[index][value]
                    + self.reported_smith_blend * counts[value] / max(1e-9, maximum)
                    for value in counts
                }
            )
        reported_weights = [
            (1.0 - self.reported_smith_blend) * weight
            + self.reported_smith_blend
            for weight in self._strategy_model_weights
        ]
        self.private_info["opponent_ufun"] = build_additive_model(
            self._issues,
            reported_tables,
            reported_weights,
            self.ufun.outcome_space,
            reserved_value=self._reserve_estimate,
        )

    def propose_offer(self, state: SAOState) -> Outcome | None:
        if self._single_issue_mode:
            return self._propose_single_issue_offer(state)
        t = state.relative_time
        target = self._adaptive_target_utility(t)
        pool = self._build_offer_pool(target, t)
        if not pool:
            return self.rational_outcomes[0] if self.rational_outcomes else None
        self._refresh_frontier_guidance(pool)

        scored = []
        for outcome in pool:
            myu = self.normalized_self_utility(outcome)
            opp = self.estimated_opponent_utility(outcome)
            diversity = self.diversity_score(outcome)
            reciprocity = self.reciprocity_score(outcome)
            reveal = self.reveal_penalty(outcome, t)
            balanced = 1.0 - abs(myu - opp)
            robust = self.robust_opponent_utility(outcome)
            pareto_score, nash_score = self._frontier_scores(outcome, myu, opp)
            zone = self._agreement_zone_score(outcome)
            probe = self._probe_score(outcome, t)
            plateau = self._plateau_offer_score(outcome, t)
            leakage = self._self_preference_leakage(outcome)
            late_alignment = self._late_alignment_score(outcome, t)
            recency_alignment = self._recent_offer_alignment(outcome)
            stubborn_match = self._stubborn_match_score(outcome, t)
            deadlock_offer = self._deadlock_offer_score(outcome, t)
            hardline_alignment = self._hardline_alignment_score(outcome, t)
            reciprocal_endgame = self._reciprocal_endgame_score(outcome, t)
            camouflage = self._camouflage_score(outcome, t)
            issue_lock = self._issue_lock_score(outcome, t)
            score = (
                self.self_weight * myu
                + self.opp_weight * opp
                + self.diversity_weight * diversity
                + self.reciprocity_weight * reciprocity
                + self.balance_weight * balanced
                + self.robust_weight * robust
                + self.pareto_weight * pareto_score
                + self.nash_weight * nash_score
                + self.zone_weight * zone
                + self.probe_weight * probe
                + self.plateau_weight * plateau
                + self.late_alignment_weight * late_alignment
                + self._effective_recency_weight(t) * recency_alignment
                + self.stubborn_match_weight * stubborn_match
                + self.deadlock_offer_weight * deadlock_offer
                + self.hardline_alignment_weight * hardline_alignment
                + self.reciprocal_endgame_weight * reciprocal_endgame
                + self.camouflage_weight * camouflage
                + self.issue_lock_weight * issue_lock
                - self.leakage_weight * leakage
                - self._effective_reveal_weight(t) * reveal
            )
            if t > 0.85:
                score += (0.12 + 0.10 * self.future_lookahead_weight) * myu * opp
            scored.append((score, outcome))
        scored.sort(key=lambda x: x[0], reverse=True)
        finalist_count = min(
            len(scored),
            max(3, 5 + int((1.0 - t) * 8.0 * self.exploration_width)),
        )
        return self._rng.choice([o for _, o in scored[:finalist_count]])

    def should_accept_offer(self, state: SAOState, next_offer_utility: float) -> bool:
        if self._single_issue_mode:
            offer = state.current_offer
            if offer is None or self.ufun is None:
                return False
            utility = float(self.ufun(offer))
            if utility < self._safe_acceptance_floor():
                return False
            compromise = self._single_issue_compromise_outcome()
            if (
                compromise is not None
                and state.relative_time >= self.single_issue_accept_compromise_start
                and offer == compromise
            ):
                compromise_utility = self._utility_by_outcome.get(compromise, self.reservation())
                if utility >= compromise_utility - 1e-9:
                    return True
            return self.should_accept(
                state,
                next_offer_utility,
                slack=self.single_issue_acceptance_scale * self.utility_span(),
            )
        assert self.ufun is not None
        offer = state.current_offer
        if offer is None:
            return False

        utility = float(self.ufun(offer))
        safe_floor = self._safe_acceptance_floor()
        if utility < safe_floor:
            return False
        base_slack = self.acceptance_slack_scale * self.utility_span()
        t = state.relative_time
        aspiration = self._adaptive_target_utility(t)
        future_best = self._expected_future_offer_utility(t)
        waiting_value = self.future_lookahead_weight * max(0.0, future_best - utility)
        plateau_discount = self.plateau_weight * self._plateau_acceptance_bonus(t)
        threshold = max(
            self.reservation(),
            min(aspiration, next_offer_utility, future_best)
            - base_slack
            + 0.35 * waiting_value
            - plateau_discount * self.utility_span(),
        )
        if t > 0.80 and self.stubborn_acceptance_bias > 0.0:
            threshold -= (
                self.stubborn_acceptance_bias
                * self._opponent_stubbornness
                * (t - 0.80)
                / 0.20
                * self.utility_span()
            )
        if t > 0.72 and self.patience_acceptance_scale > 0.0:
            threshold -= (
                self.patience_acceptance_scale
                * self._effective_patience_signal()
                * (t - 0.72)
                / 0.28
                * self.utility_span()
            )
        threshold = max(safe_floor, threshold)

        if utility >= threshold:
            return True
        if self._best_received_utility > float("-inf"):
            close_to_best = utility >= self._best_received_utility - self.near_best_acceptance_margin * self.utility_span()
            if t >= 0.88 and close_to_best and self._plateau_acceptance_bonus(t) > 0.35:
                return True
        if t >= 0.92 and utility >= next_offer_utility - self.near_best_acceptance_margin * self.utility_span():
            return True
        if (
            t >= 0.90
            and self._best_received_utility > float("-inf")
            and self._opponent_stubbornness > 0.55
            and utility >= self._best_received_utility - 0.005 * self.utility_span()
        ):
            return True
        if (
            t >= 0.86
            and self.deadlock_acceptance_scale > 0.0
            and self._deadlock_level > 0.45
            and self._best_received_utility > float("-inf")
            and utility >= self._best_received_utility - self.deadlock_acceptance_scale * self.utility_span()
        ):
            return True
        if (
            t >= 0.90
            and self.hardline_acceptance_scale > 0.0
            and self._hardline_signal() > 0.65
            and self._best_received_utility > float("-inf")
            and utility >= self._best_received_utility - self.hardline_acceptance_scale * self.utility_span()
        ):
            return True
        if (
            t >= 0.90
            and self.micron_acceptance_scale > 0.0
            and self._opponent_repeat_rate > 0.45
            and self._best_received_utility > float("-inf")
            and utility >= self._best_received_utility - self.micron_acceptance_scale * self.utility_span()
        ):
            return True
        if (
            t >= 0.88
            and self.reciprocal_acceptance_scale > 0.0
            and self._opponent_reciprocity > 0.55
            and self._best_received_utility > float("-inf")
            and utility >= self._best_received_utility - self.reciprocal_acceptance_scale * self.utility_span()
        ):
            return True
        if t >= 0.97 and utility >= self.reservation() + 0.008 * self.utility_span():
            return True
        return False

    def _safe_acceptance_floor(self) -> float:
        return self.reservation() + self.reservation_guard_margin * self.utility_span()

    def estimated_opponent_utility(self, outcome: Outcome) -> float:
        values = self.outcome_values(outcome)
        totalw = sum(self._weights) or 1.0
        total = 0.0
        for i, value in enumerate(values):
            total += (self._weights[i] / totalw) * self._learned_tables[i].get(value, 0.5)
        learned = max(0.0, min(1.0, total))
        belief = self.belief_opponent_utility(outcome)
        blend = self.belief_blend + self.dynamic_belief_weight * self._belief_activation()
        blend = max(0.0, min(0.55, blend))
        return (1.0 - blend) * learned + blend * belief

    def belief_opponent_utility(self, outcome: Outcome) -> float:
        if not getattr(self, "_hypotheses", None):
            return 0.5
        return sum(
            hypothesis.score * self._hypothesis_utility(hypothesis, outcome)
            for hypothesis in self._hypotheses
        )

    def robust_opponent_utility(self, outcome: Outcome) -> float:
        if not getattr(self, "_hypotheses", None):
            return 0.5
        values = [self._hypothesis_utility(hypothesis, outcome) for hypothesis in self._hypotheses]
        return min(values) if values else 0.5

    def _update_model(self, offer: Outcome, prev: Outcome | None, state: SAOState) -> None:
        values = self.outcome_values(offer)
        if self.ufun is not None:
            my_utility = float(self.ufun(offer))
            if my_utility > self._best_opponent_offer_utility:
                self._best_opponent_offer = offer
                self._best_opponent_offer_utility = my_utility
        if self.value_decay < 0.999999:
            for counts in self._value_counts:
                for value in counts:
                    counts[value] = 1.0 + (counts[value] - 1.0) * self.value_decay
        if self.recent_bonus_decay < 0.999999:
            for recent_bonus in self._recent_bonus:
                for value in recent_bonus:
                    recent_bonus[value] *= self.recent_bonus_decay
        if self.transition_bonus_decay < 0.999999:
            for transition_bonus in self._transition_bonus:
                for value in transition_bonus:
                    transition_bonus[value] *= self.transition_bonus_decay
        for i, value in enumerate(values):
            self._value_counts[i][value] += 1.0 + 1.3 * state.relative_time
            self._recent_bonus[i][value] += 0.6 + state.relative_time
            if prev is not None:
                prev_value = self.outcome_values(prev)[i]
                if prev_value == value:
                    self._same[i] += 1.0 + state.relative_time
                else:
                    self._change[i] += 1.0 + state.relative_time
                    self._transition_bonus[i][value] += 0.7 + 1.1 * state.relative_time

        issue_weights = []
        for i, counts in enumerate(self._value_counts):
            total = sum(counts.values())
            probs = [v / total for v in counts.values()]
            entropy = -sum(p * math.log(max(1e-12, p)) for p in probs)
            max_entropy = math.log(max(1, len(probs)))
            concentration = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
            stickiness = self._same[i] / (self._same[i] + self._change[i])
            raw = {
                value: counts[value]
                + 0.4 * self._recent_bonus[i].get(value, 0.0)
                + self.transition_blend * self._transition_bonus[i].get(value, 0.0)
                for value in counts
            }
            learned = normalize_mapping(raw)
            issue_signal = 0.55 * concentration + 0.45 * stickiness
            mirror_blend = self.mirror_blend * max(
                0.0,
                1.0 - self.adaptive_mirror_scale * issue_signal * (0.35 + 0.65 * state.relative_time),
            )
            self._learned_tables[i] = {
                value: (1.0 - mirror_blend) * learned.get(value, 0.5)
                + mirror_blend * self._mirror_tables[i].get(value, 0.5)
                for value in counts
            }
            issue_weights.append(issue_signal + 1e-4)

        self._weights = issue_weights
        current_estimate = self.estimated_opponent_utility(offer)
        self._recent_estimated_utilities.append(current_estimate)
        if len(self._recent_estimated_utilities) > 12:
            self._recent_estimated_utilities = self._recent_estimated_utilities[-12:]
        self._update_concession_statistics()
        self._refresh_patience_signal()
        self._refresh_issue_locks()
        self._update_beliefs(offer, state)
        self._estimate_reserved_value(state)
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        if self.ufun is None:
            return
        tables = []
        weights = []
        for index, issue in enumerate(self._issues):
            table = {}
            for value in issue.all:
                learned = self._learned_tables[index].get(value, 0.5)
                belief = self._belief_tables[index].get(value, 0.5)
                table[value] = (1.0 - self.belief_blend) * learned + self.belief_blend * belief
            tables.append(table)
            weights.append(
                (1.0 - self.belief_blend) * self._weights[index]
                + self.belief_blend * self._belief_weights[index]
            )
        self._strategy_opponent_ufun = build_additive_model(
            self._issues,
            tables,
            weights,
            self.ufun.outcome_space,
            reserved_value=self._reserve_estimate,
        )
        self._strategy_model_tables = tables
        self._strategy_model_weights = weights
        self.private_info["opponent_ufun"] = self._strategy_opponent_ufun
        self._opponent_model_version += 1


    def _estimate_reserved_value(self, state: SAOState) -> None:
        if len(self._recent_estimated_utilities) < 4:
            self._reserve_estimate = 0.0
            return
        tail = self._recent_estimated_utilities[-8:]
        floor = min(tail)
        latest = tail[-1]
        trend = max(0.0, tail[0] - latest)
        volatility = sum(abs(b - a) for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
        late_factor = 0.5 + 0.5 * state.relative_time
        estimate = floor - 0.35 * trend - 0.15 * volatility + self.reserve_floor_bias * late_factor
        self._reserve_estimate = max(0.0, min(0.92, estimate))

    def _expected_future_offer_utility(self, relative_time: float) -> float:
        if len(self._received_offers) < 2 or self.ufun is None:
            return self._adaptive_target_utility(relative_time)
        tail = [float(self.ufun(o)) for o in self._received_offers[-5:]]
        last = tail[-1]
        velocity = 0.0
        if len(tail) >= 2:
            diffs = [b - a for a, b in zip(tail, tail[1:])]
            velocity = sum(diffs) / len(diffs)
        remaining = max(0.0, 1.0 - relative_time)
        projected = last + velocity * (2.0 + 4.0 * remaining)
        aspiration = self._adaptive_target_utility(relative_time)
        return max(self.reservation(), min(self.best_utility(), max(projected, aspiration - 0.06 * self.utility_span())))

    def _refresh_frontier_guidance(self, pool: list[Outcome]) -> None:
        if self.ufun is None:
            return
        self._ensure_frontier_cache()
        self._pareto_hint = {}
        self._frontier_outcomes = set()
        for outcome in pool:
            pair = self._cached_frontier_pairs.get(outcome)
            if pair is None:
                pair = (self._utility_by_outcome[outcome], self.estimated_opponent_utility(outcome))
            else:
                self._frontier_outcomes.add(outcome)
            self._pareto_hint[outcome] = pair
        self._nash_target = self._cached_nash_target

    def _ensure_frontier_cache(self) -> None:
        if self.ufun is None:
            return
        if (
            self._frontier_model_snapshot >= 0
            and self._opponent_model_version - self._frontier_model_snapshot
            < self.frontier_refresh_interval
        ):
            return

        reference_outcomes = list(self._frontier_reference_outcomes)
        if self.exact_frontier_threshold > 0 and len(self.rational_outcomes) <= self.exact_frontier_threshold:
            reference_outcomes = list(self.rational_outcomes)
        try:
            frontier_utils, frontier_indices = pareto_frontier(
                (self.ufun, self._strategy_opponent_ufun),
                outcomes=reference_outcomes,
                sort_by_welfare=False,
            )
            frontier_outcomes = [reference_outcomes[i] for i in frontier_indices]
            self._cached_frontier_outcomes = set(frontier_outcomes)
            self._cached_frontier_pairs = {
                outcome: (utils[0], utils[1])
                for outcome, utils in zip(frontier_outcomes, frontier_utils)
            }
            if frontier_utils:
                nash = nash_points(
                    (self.ufun, self._strategy_opponent_ufun),
                    frontier_utils,
                    outcomes=frontier_outcomes,
                )
                self._cached_nash_target = nash[0][0] if nash else None
            else:
                self._cached_nash_target = None
        except Exception:
            self._cached_frontier_outcomes = set()
            self._cached_frontier_pairs = {}
            self._cached_nash_target = None
        self._frontier_model_snapshot = self._opponent_model_version
        self._frontier_refresh_count += 1

    def _build_frontier_reference_outcomes(self) -> tuple[Outcome, ...]:
        if not self.rational_outcomes:
            return tuple()
        if len(self.rational_outcomes) <= self.frontier_reference_size:
            return tuple(self.rational_outcomes)
        anchors = []
        top_block = min(len(self.rational_outcomes), max(24, self.frontier_reference_size // 3))
        anchors.extend(self.rational_outcomes[:top_block])
        remaining = self.frontier_reference_size - len(anchors)
        if remaining <= 0:
            return tuple(dict.fromkeys(anchors))
        stride = max(1, len(self.rational_outcomes) // remaining)
        anchors.extend(self.rational_outcomes[::stride][:remaining])
        return tuple(dict.fromkeys(anchors[: self.frontier_reference_size]))

    def _frontier_scores(self, outcome: Outcome, myu: float, opp: float) -> tuple[float, float]:
        pair = self._pareto_hint.get(outcome, (myu, opp))
        frontier_bonus = 1.0 if outcome in self._frontier_outcomes else 0.0
        welfare = 0.5 * pair[0] + 0.5 * pair[1]
        if self._is_small_domain:
            frontier_bonus = min(1.0, frontier_bonus + 0.15)
        pareto_score = 0.55 * frontier_bonus + 0.45 * welfare
        if self._nash_target is None:
            return pareto_score, myu * opp
        dx = abs(pair[0] - self._nash_target[0])
        dy = abs(pair[1] - self._nash_target[1])
        distance = max(0.0, 1.0 - 0.5 * (dx + dy))
        return pareto_score, max(0.0, distance)

    def _build_offer_pool(self, target: float, relative_time: float) -> list[Outcome]:
        if self.pool_mix_weight <= 1e-9:
            base_pool = self.select_candidates(target, lower_slack=0.038, upper_slack=0.14)
            merged = list(base_pool)
            seen = set(base_pool)
            if (
                self.recent_offer_pool_weight > 1e-9
                and self._last_offer_from_opponent is not None
                and self._use_recent_pool(relative_time)
            ):
                for outcome in self._near_reference_pool(
                    self._last_offer_from_opponent, cap=140, limit=self._scaled_style_limit(30)
                ):
                    if outcome not in seen:
                        seen.add(outcome)
                        merged.append(outcome)
            if (
                self.recent_offer_pool_weight > 1e-9
                and self._best_opponent_offer is not None
                and relative_time >= self.hardline_pool_start
            ):
                for outcome in self._near_reference_pool(
                    self._best_opponent_offer, cap=200, limit=self._scaled_style_limit(24)
                ):
                    if outcome not in seen:
                        seen.add(outcome)
                        merged.append(outcome)
            self._append_style_pools(merged, seen, relative_time)
            if len(merged) > self.ranking_pool_size:
                step = max(1, len(merged) // self.ranking_pool_size)
                merged = merged[::step][: self.ranking_pool_size]
            return merged
        pools = [
            self.select_candidates(target, lower_slack=0.038, upper_slack=0.14),
            self.select_candidates(
                max(self.reservation(), target - self.pool_mix_weight * 0.05 * self.utility_span()),
                lower_slack=max(0.02, 0.038 - 0.01 * self.pool_mix_weight),
                upper_slack=0.10 + 0.02 * self.pool_mix_weight,
            ),
        ]
        if relative_time >= 0.65:
            compromise_target = self.reservation() + self.utility_span() * max(
                0.48,
                0.78 - (0.25 + 0.09 * self.pool_mix_weight) * relative_time,
            )
            pools.append(self.select_candidates(compromise_target, lower_slack=0.025, upper_slack=0.085))
        if self._best_opponent_offer is not None and self.zone_weight > 0.0:
            pools.append(self._near_opponent_best_pool())
        if (
            self.recent_offer_pool_weight > 0.0
            and self._last_offer_from_opponent is not None
            and self._use_recent_pool(relative_time)
        ):
            pools.append(
                self._near_reference_pool(
                    self._last_offer_from_opponent, cap=140, limit=self._scaled_style_limit(30)
                )
            )

        merged: list[Outcome] = []
        seen: set[Outcome] = set()
        for pool in pools:
            for outcome in pool:
                if outcome not in seen:
                    seen.add(outcome)
                    merged.append(outcome)
        self._append_style_pools(merged, seen, relative_time)
        if len(merged) > self.ranking_pool_size:
            step = max(1, len(merged) // self.ranking_pool_size)
            merged = merged[::step][: self.ranking_pool_size]
        return merged

    def _append_style_pools(self, merged: list[Outcome], seen: set[Outcome], relative_time: float) -> None:
        if (
            self.reciprocal_pool_weight > 0.0
            and self._opponent_reciprocity > 0.45
            and self._last_offer_from_opponent is not None
            and self._use_recent_pool(relative_time)
        ):
            limit = self._scaled_style_limit(max(10, min(40, int(12 + 24 * self.reciprocal_pool_weight))))
            for outcome in self._near_reference_pool(self._last_offer_from_opponent, cap=180, limit=limit):
                if outcome not in seen:
                    seen.add(outcome)
                    merged.append(outcome)
        if self._deadlock_level > 0.40 and relative_time >= 0.68:
            for outcome in self._near_best_received_pool(limit=self._scaled_style_limit(28)):
                if outcome not in seen:
                    seen.add(outcome)
                    merged.append(outcome)
        if (
            self.best_received_pool_weight > 0.0
            and self._best_received_utility > float("-inf")
            and relative_time >= self.best_received_pool_start
        ):
            limit = self._scaled_style_limit(
                max(10, min(36, int(10 + 22 * self.best_received_pool_weight)))
            )
            for outcome in self._near_best_received_pool(limit=limit):
                if outcome not in seen:
                    seen.add(outcome)
                    merged.append(outcome)
        if (
            self._hardline_signal() > 0.68
            and self._best_opponent_offer is not None
            and relative_time >= self.hardline_pool_start
        ):
            for outcome in self._near_reference_pool(
                self._best_opponent_offer, cap=220, limit=self._scaled_style_limit(24)
            ):
                if outcome not in seen:
                    seen.add(outcome)
                    merged.append(outcome)

    def _propose_single_issue_offer(self, state: SAOState) -> Outcome | None:
        if not self.rational_outcomes:
            return None
        t = state.relative_time
        compromise = self._single_issue_compromise_outcome()
        if (
            compromise is not None
            and t >= self.single_issue_compromise_start
            and self._single_issue_deadlock_detected()
        ):
            return compromise
        opponent_progress = 0.0
        if len(self._received_offers) >= 2 and self.ufun is not None:
            last = float(self.ufun(self._received_offers[-1]))
            first = float(self.ufun(self._received_offers[0]))
            opponent_progress = max(0.0, last - first) / self.utility_span()
        heuristic_target = self.target_utility(
            t,
            early=0.992,
            middle=0.91 - 0.04 * opponent_progress,
            late=0.73 - 0.08 * opponent_progress,
            floor=0.14,
        )
        belief_target = self.target_utility(
            t,
            early=0.991,
            middle=0.90,
            late=0.72,
            floor=0.10,
        )
        target = (
            (1.0 - self.single_issue_belief_blend) * heuristic_target
            + self.single_issue_belief_blend * belief_target
        )
        lower_slack = (
            (1.0 - self.single_issue_belief_blend) * (0.04 + 0.06 * (1.0 - t))
            + self.single_issue_belief_blend * 0.045
        )
        upper_slack = (
            (1.0 - self.single_issue_belief_blend) * 0.14
            + self.single_issue_belief_blend * 0.16
        )
        pool = self.select_candidates(target, lower_slack=lower_slack, upper_slack=upper_slack)
        if not pool:
            return self.rational_outcomes[0]
        scored = []
        for outcome in pool:
            myu = self.normalized_self_utility(outcome)
            reciprocity = self.reciprocity_score(outcome)
            diversity = self.diversity_score(outcome)
            reveal = self.reveal_penalty(outcome, t)
            heuristic_score = 0.69 * myu + 0.18 * reciprocity + 0.16 * diversity - 0.18 * reveal
            belief_score = (
                0.53 * myu
                + 0.22 * self.belief_opponent_utility(outcome)
                + 0.12 * self.robust_opponent_utility(outcome)
                + 0.12 * diversity
                - 0.12 * reveal
            )
            score = (
                (1.0 - self.single_issue_belief_blend) * heuristic_score
                + self.single_issue_belief_blend * belief_score
            )
            if t > 0.85:
                score += 0.08 * reciprocity
            scored.append((score, outcome))
        scored.sort(key=lambda item: item[0], reverse=True)
        return self._rng.choice([outcome for _, outcome in scored[: min(6, len(scored))]])

    def _single_issue_compromise_outcome(self) -> Outcome | None:
        if not self.rational_outcomes:
            return None
        best = None
        best_score = float("-inf")
        for outcome in self.rational_outcomes:
            myu = self.normalized_self_utility(outcome)
            opp = self.belief_opponent_utility(outcome)
            score = 0.65 * min(myu, opp) + 0.35 * myu * opp
            if score > best_score:
                best_score = score
                best = outcome
        return best

    def _single_issue_deadlock_detected(self) -> bool:
        window = self.single_issue_deadlock_window
        if len(self._sent_offers) < window or len(self._received_offers) < window:
            return False
        sent_tail = self._sent_offers[-window:]
        recv_tail = self._received_offers[-window:]
        same_sent = len({self.outcome_values(o) for o in sent_tail}) == 1
        same_recv = len({self.outcome_values(o) for o in recv_tail}) == 1
        if not same_sent or not same_recv:
            return False
        return sent_tail[-1] != recv_tail[-1]

    def _near_opponent_best_pool(self) -> list[Outcome]:
        if self._best_opponent_offer is None:
            return []
        return self._near_reference_pool(self._best_opponent_offer, cap=220, limit=40)

    def _near_reference_pool(self, reference_offer: Outcome, cap: int, limit: int) -> list[Outcome]:
        best_values = self.outcome_values(reference_offer)
        scored = []
        cap = min(len(self.rational_outcomes), cap)
        for outcome in self.rational_outcomes[:cap]:
            values = self.outcome_values(outcome)
            same = sum(1 for a, b in zip(values, best_values) if a == b) / max(1, len(values))
            scored.append((same, outcome))
        scored.sort(key=lambda x: (x[0], self._utility_by_outcome[x[1]]), reverse=True)
        return [outcome for _, outcome in scored[:limit]]

    def _near_best_received_pool(self, limit: int) -> list[Outcome]:
        if self._best_received_utility == float("-inf"):
            return []
        scored = []
        cap = min(len(self.rational_outcomes), 220)
        target = self._best_received_utility
        for outcome in self.rational_outcomes[:cap]:
            utility = self._utility_by_outcome[outcome]
            gap = abs(target - utility) / self.utility_span()
            similarity = self._recent_offer_alignment(outcome)
            scored.append((1.0 - gap + 0.3 * similarity, outcome))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [outcome for _, outcome in scored[:limit]]

    def _adaptive_target_utility(self, relative_time: float) -> float:
        aspiration = self.target_utility(
            relative_time,
            early=0.993,
            middle=0.905,
            late=self.target_late,
            floor=self.target_floor,
        )
        if self.cooperation_weight <= 1e-9 and self.small_domain_relaxation <= 1e-9:
            normalized = (aspiration - self.reservation()) / self.utility_span()
        else:
            cooperation = self._opponent_concession_rate
            adjustment = self.cooperation_weight * (0.30 - cooperation)
            if self._is_small_domain and relative_time > 0.75:
                adjustment -= self.small_domain_relaxation
            normalized = (aspiration - self.reservation()) / self.utility_span()
            normalized = normalized + adjustment
        if self.target_toughness_bias > 0.0 and relative_time > 0.55:
            normalized -= (
                self.target_toughness_bias
                * self._opponent_stubbornness
                * (relative_time - 0.55)
                / 0.45
            )
        if self.hardline_target_discount > 0.0 and relative_time > 0.72:
            normalized -= (
                self.hardline_target_discount
                * self._opponent_stubbornness
                * (relative_time - 0.72)
                / 0.28
            )
        if self.deadlock_offer_weight > 0.0 and self._deadlock_level > 0.45 and relative_time > 0.70:
            normalized -= (
                0.5
                * self.deadlock_offer_weight
                * self._deadlock_level
                * (relative_time - 0.70)
                / 0.30
            )
        if self.reciprocal_target_discount > 0.0 and relative_time > 0.72:
            normalized -= (
                self.reciprocal_target_discount
                * self._opponent_reciprocity
                * max(self._opponent_repeat_rate, self._deadlock_level)
                * (relative_time - 0.72)
                / 0.28
            )
        if self.patience_target_discount > 0.0 and relative_time > 0.70:
            normalized -= (
                self.patience_target_discount
                * self._effective_patience_signal()
                * (relative_time - 0.70)
                / 0.30
            )
        if self.opportunism_target_bonus > 0.0 and relative_time < 0.72:
            normalized += self.opportunism_target_bonus * self._opportunism_signal(relative_time)
        if self._opponent_reciprocity > 0.35 and relative_time < 0.75:
            normalized += 0.025 * self._opponent_reciprocity
        normalized = max(0.08, min(0.995, normalized))
        return self.reservation() + normalized * self.utility_span()

    def _agreement_zone_score(self, outcome: Outcome) -> float:
        if self._best_opponent_offer is None:
            return 0.5
        values = self.outcome_values(outcome)
        best_values = self.outcome_values(self._best_opponent_offer)
        totalw = sum(self._weights) or 1.0
        same = 0.0
        for index, (left, right) in enumerate(zip(values, best_values)):
            if left == right:
                same += self._weights[index] / totalw
        return same

    def _probe_score(self, outcome: Outcome, relative_time: float) -> float:
        if relative_time > 0.45 or len(getattr(self, "_hypotheses", [])) < 2:
            return 0.0
        values = [self._hypothesis_utility(h, outcome) for h in self._hypotheses]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return min(1.0, 3.0 * variance)

    def _plateau_offer_score(self, outcome: Outcome, relative_time: float) -> float:
        if relative_time < 0.70:
            return 0.0
        if self._best_received_utility == float("-inf"):
            return 0.0
        utility = self._utility_by_outcome.get(outcome, self.reservation())
        gap = max(0.0, self._best_received_utility - utility) / self.utility_span()
        return max(0.0, 1.0 - 2.2 * gap)

    def _plateau_acceptance_bonus(self, relative_time: float) -> float:
        if relative_time < 0.75 or len(self._received_offers) < 5 or self.ufun is None:
            return 0.0
        tail = [float(self.ufun(outcome)) for outcome in self._received_offers[-5:]]
        improvements = [max(0.0, b - a) for a, b in zip(tail, tail[1:])]
        average_improvement = sum(improvements) / max(1, len(improvements))
        stagnation = max(0.0, 1.0 - average_improvement / max(1e-9, 0.04 * self.utility_span()))
        return max(0.0, min(1.0, stagnation))

    def _self_preference_leakage(self, outcome: Outcome) -> float:
        if not self._sent_offers or not self._self_value_means:
            return 0.0
        values = self.outcome_values(outcome)
        penalty = 0.0
        for index, value in enumerate(values):
            preference = self._normalized_self_value_means[index].get(value, 0.5)
            repeated = self._sent_value_counts[index].get(value, 0) / max(1, len(self._sent_offers))
            penalty += preference * repeated
        return penalty / max(1, len(values))

    def _camouflage_score(self, outcome: Outcome, relative_time: float) -> float:
        if self.camouflage_weight <= 0.0:
            return 0.0
        if relative_time < self.camouflage_start or relative_time > self.camouflage_end:
            return 0.0
        phase = 1.0 - (
            (relative_time - self.camouflage_start)
            / max(1e-9, self.camouflage_end - self.camouflage_start)
        )
        values = self.outcome_values(outcome)
        concealment = 0.0
        history = max(1, len(self._sent_offers))
        for index, value in enumerate(values):
            preference = self._normalized_self_value_means[index].get(value, 0.5)
            midpoint = 1.0 - min(1.0, 2.0 * abs(preference - 0.5))
            exposure = self._sent_value_counts[index].get(value, 0) / history
            concealment += 0.7 * midpoint + 0.3 * (1.0 - exposure)
        return (
            phase
            * self._camouflage_style_signal()
            * concealment
            / max(1, len(values))
        )

    def _effective_reveal_weight(self, relative_time: float) -> float:
        weight = self.reveal_weight - 0.08 * relative_time
        if self.early_reveal_boost > 0.0 and relative_time < 0.55:
            weight += self.early_reveal_boost * (
                (0.55 - relative_time) / 0.55
            ) ** self.reveal_decay_power
        if relative_time > 0.75 and self._hardline_signal() > 0.65:
            weight *= 0.92
        return max(0.02, weight)

    def _late_alignment_score(self, outcome: Outcome, relative_time: float) -> float:
        if self._best_opponent_offer is None or self.late_alignment_weight <= 0.0:
            return 0.0
        if relative_time < self.late_alignment_start:
            return 0.0
        return self._agreement_zone_score(outcome)

    def _recent_offer_alignment(self, outcome: Outcome) -> float:
        if not self._received_offers:
            return 0.5
        values = self.outcome_values(outcome)
        recent = self._received_offers[-3:]
        weights = [0.2, 0.3, 0.5][-len(recent):]
        total = 0.0
        norm = 0.0
        for weight, offer in zip(weights, recent):
            ref = self.outcome_values(offer)
            same = sum(1 for left, right in zip(values, ref) if left == right) / max(1, len(values))
            total += weight * same
            norm += weight
        return total / max(1e-9, norm)

    def _effective_recency_weight(self, relative_time: float) -> float:
        base = self.recency_alignment_weight
        if base <= 0.0:
            return 0.0
        boost = 1.0 + self.reciprocal_alignment_scale * self._opponent_reciprocity
        if self._deadlock_level > 0.35 and relative_time > 0.55:
            boost += 0.75 * self._deadlock_level
        return base * boost

    def _stubborn_match_score(self, outcome: Outcome, relative_time: float) -> float:
        if self._opponent_stubbornness <= 0.0:
            return 0.0
        anchor = self._best_opponent_offer if relative_time >= 0.7 else self._last_offer_from_opponent
        if anchor is None:
            return 0.0
        values = self.outcome_values(outcome)
        anchor_values = self.outcome_values(anchor)
        same = sum(1 for left, right in zip(values, anchor_values) if left == right) / max(1, len(values))
        return self._opponent_stubbornness * same

    def _deadlock_offer_score(self, outcome: Outcome, relative_time: float) -> float:
        if self._deadlock_level <= 0.0 or relative_time < 0.55:
            return 0.0
        alignment = self._recent_offer_alignment(outcome)
        plateau = self._plateau_offer_score(outcome, relative_time)
        return self._deadlock_level * (0.65 * alignment + 0.35 * plateau)

    def _issue_lock_score(self, outcome: Outcome, relative_time: float) -> float:
        if self.issue_lock_weight <= 0.0 or relative_time < self.issue_lock_start:
            return 0.0
        if not any(self._issue_lock_strengths):
            return 0.0
        values = self.outcome_values(outcome)
        total = 0.0
        norm = 0.0
        for index, value in enumerate(values):
            strength = self._issue_lock_strengths[index]
            locked = self._issue_lock_values[index]
            if strength <= 0.0 or locked is None:
                continue
            norm += strength
            if value == locked:
                total += strength
        if norm <= 1e-9:
            return 0.0
        return total / norm

    def _hardline_alignment_score(self, outcome: Outcome, relative_time: float) -> float:
        if self._best_opponent_offer is None or relative_time < self.hardline_pool_start:
            return 0.0
        return self._hardline_signal() * self._agreement_zone_score(outcome)

    def _reciprocal_endgame_score(self, outcome: Outcome, relative_time: float) -> float:
        if relative_time < 0.72:
            return 0.0
        if self._opponent_reciprocity <= 0.45:
            return 0.0
        return self._opponent_reciprocity * max(self._opponent_repeat_rate, self._deadlock_level) * self._recent_offer_alignment(outcome)

    def _use_recent_pool(self, relative_time: float) -> bool:
        if relative_time >= self.recent_pool_start:
            return True
        return self._opponent_reciprocity > 0.45 or self._deadlock_level > 0.35

    def _scaled_style_limit(self, base_limit: int) -> int:
        if not self._is_small_domain:
            return max(8, int(base_limit * self.large_domain_pool_scale))
        return base_limit

    def _patience_signal(self) -> float:
        return max(0.0, min(1.0, self._no_improvement_streak / 4.0))

    def _effective_patience_signal(self) -> float:
        base = self._patience_signal()
        if self.patience_style_weight <= 0.0:
            return base
        style = max(self._hardline_signal(), self._deadlock_level, 0.8 * self._opponent_repeat_rate)
        style = max(0.0, min(1.0, style))
        return base * ((1.0 - self.patience_style_weight) + self.patience_style_weight * style)

    def _camouflage_style_signal(self) -> float:
        if self.camouflage_style_weight <= 0.0:
            return 1.0
        style = 0.55 * self._opponent_stubbornness + 0.25 * self._deadlock_level + 0.20 * self._opponent_repeat_rate
        style = max(0.0, min(1.0, style))
        return (1.0 - self.camouflage_style_weight) + self.camouflage_style_weight * style

    def _opportunism_signal(self, relative_time: float) -> float:
        softness = max(0.0, 1.0 - self._hardline_signal())
        concession = min(1.0, self._opponent_concession_rate / 0.18)
        volatility = min(1.0, self._opponent_volatility / 0.16)
        timing = max(0.0, 1.0 - relative_time / 0.72)
        return max(0.0, min(1.0, softness * concession * (0.75 + 0.25 * volatility) * timing))

    def _hardline_signal(self) -> float:
        return max(0.0, min(1.0, 0.72 * self._opponent_stubbornness + 0.28 * self._opponent_repeat_rate))

    def _belief_activation(self) -> float:
        if len(self._received_offers) < 4:
            return 0.0
        confidence = self._model_confidence()
        if confidence < 0.10:
            return 0.0
        return max(0.0, min(1.0, 0.35 + 0.65 * confidence))

    def _model_confidence(self) -> float:
        if len(self._recent_estimated_utilities) < 4:
            return 0.0
        tail = self._recent_estimated_utilities[-6:]
        spread = max(tail) - min(tail)
        concentration = sum(self._weights) / max(1e-9, len(self._weights))
        stability = max(0.0, 1.0 - spread / 0.35)
        return max(0.0, min(1.0, 0.6 * stability + 0.4 * concentration))

    def _update_concession_statistics(self) -> None:
        if self.ufun is None or len(self._received_offers) < 2:
            return
        tail = [float(self.ufun(outcome)) for outcome in self._received_offers[-6:]]
        diffs = [b - a for a, b in zip(tail, tail[1:])]
        if diffs:
            positives = [max(0.0, diff) for diff in diffs]
            self._opponent_concession_rate = sum(positives) / max(1e-9, self.utility_span() * len(positives))
            self._opponent_volatility = sum(abs(diff) for diff in diffs) / max(1e-9, self.utility_span() * len(diffs))
        if self._sent_offers and self._received_offers:
            sent = self.outcome_values(self._sent_offers[-1])
            received = self.outcome_values(self._received_offers[-1])
            matching = sum(1 for left, right in zip(sent, received) if left == right) / max(1, len(sent))
            self._opponent_reciprocity = 0.65 * self._opponent_reciprocity + 0.35 * matching
        if len(self._received_offers) >= 3:
            recent_received = self._received_offers[-3:]
            self._opponent_repeat_rate = sum(
                1.0
                for left, right in zip(recent_received, recent_received[1:])
                if self.outcome_values(left) == self.outcome_values(right)
            ) / max(1, len(recent_received) - 1)
        stubbornness = 1.0 - min(1.0, self._opponent_concession_rate + 0.35 * self._opponent_volatility)
        self._opponent_stubbornness = max(0.0, min(1.0, stubbornness))
        self._deadlock_level = self._estimate_deadlock_level()

    def _refresh_patience_signal(self) -> None:
        if self.ufun is None or not self._received_offers:
            self._no_improvement_streak = 0.0
            return
        latest_utility = float(self.ufun(self._received_offers[-1]))
        if self._best_received_utility - latest_utility <= 0.004 * self.utility_span():
            self._no_improvement_streak = 0.0
            return
        repeated = 0.0
        if len(self._received_offers) >= 2 and self.outcome_values(self._received_offers[-1]) == self.outcome_values(self._received_offers[-2]):
            repeated = 0.6
        stagnation = self._plateau_acceptance_bonus(self._last_relative_time)
        self._no_improvement_streak = min(
            4.0,
            0.65 * self._no_improvement_streak + 0.75 + repeated + 0.45 * stagnation,
        )

    def _refresh_issue_locks(self) -> None:
        if len(self._received_offers) < 3:
            self._issue_lock_values = [None] * len(self._issues)
            self._issue_lock_strengths = [0.0] * len(self._issues)
            return
        recent = self._received_offers[-4:]
        values_by_offer = [self.outcome_values(offer) for offer in recent]
        locked_values = []
        strengths = []
        for index in range(len(self._issues)):
            counts: dict[object, int] = {}
            for offer_values in values_by_offer:
                value = offer_values[index]
                counts[value] = counts.get(value, 0) + 1
            value, count = max(counts.items(), key=lambda item: item[1])
            strength = count / len(values_by_offer)
            if strength >= 0.75:
                locked_values.append(value)
                strengths.append(strength)
            else:
                locked_values.append(None)
                strengths.append(0.0)
        self._issue_lock_values = locked_values
        self._issue_lock_strengths = strengths

    def _estimate_deadlock_level(self) -> float:
        if len(self._received_offers) < 3 or len(self._sent_offers) < 2 or self.ufun is None:
            return 0.0
        recent_received = self._received_offers[-3:]
        recent_sent = self._sent_offers[-3:]
        repeated_received = 0.0
        if len(recent_received) >= 2:
            repeated_received = sum(
                1.0
                for left, right in zip(recent_received, recent_received[1:])
                if self.outcome_values(left) == self.outcome_values(right)
            ) / max(1, len(recent_received) - 1)
        repeated_sent = 0.0
        if len(recent_sent) >= 2:
            repeated_sent = sum(
                1.0
                for left, right in zip(recent_sent, recent_sent[1:])
                if self.outcome_values(left) == self.outcome_values(right)
            ) / max(1, len(recent_sent) - 1)
        tail = [float(self.ufun(outcome)) for outcome in recent_received]
        gain = max(0.0, tail[-1] - tail[0]) / self.utility_span()
        stagnation = max(0.0, 1.0 - gain / 0.03)
        deadlock = 0.35 * repeated_received + 0.25 * repeated_sent + 0.40 * stagnation
        return max(0.0, min(1.0, deadlock))

    def _build_hypotheses(self) -> list[BeliefHypothesis]:
        hypotheses: list[BeliefHypothesis] = []
        base_weights = []
        for table in self._self_value_means:
            vals = list(table.values())
            base_weights.append(max(vals) - min(vals) if vals else 1.0)
        mirror_tables = [normalize_mapping(table, invert=True) for table in self._self_value_means]
        hypotheses.append(BeliefHypothesis(mirror_tables, base_weights, 1.0))

        flat_tables = [{value: 0.5 for value in issue.all} for issue in self._issues]
        hypotheses.append(BeliefHypothesis(flat_tables, [1.0] * len(self._issues), 0.8))

        if self.rational_outcomes:
            best = self.outcome_values(self.rational_outcomes[0])
            for focus in range(min(3, len(self._issues))):
                tables = []
                weights = []
                for index, issue in enumerate(self._issues):
                    base = {}
                    for value in issue.all:
                        if value == best[index]:
                            base[value] = 0.2 if index == focus else 0.4
                        else:
                            base[value] = 0.8 if index == focus else 0.6
                    tables.append(normalize_mapping(base))
                    weights.append(1.8 if index == focus else 1.0)
                hypotheses.append(BeliefHypothesis(tables, weights, 0.7))
        return hypotheses

    def _update_beliefs(self, offer: Outcome, state: SAOState) -> None:
        prev_offer = self._received_offers[-2] if len(self._received_offers) >= 2 else None
        for hypothesis in self._hypotheses:
            current = self._hypothesis_utility(hypothesis, offer)
            plausibility = 0.55 + 0.45 * current
            if prev_offer is not None:
                previous = self._hypothesis_utility(hypothesis, prev_offer)
                change = current - previous
                plausibility *= 1.0 + 0.3 * max(-0.15, min(0.15, -change))
            if state.relative_time > 0.7:
                plausibility *= 0.95 + 0.1 * current
            hypothesis.score *= max(0.35, plausibility)
        total = sum(hypothesis.score for hypothesis in self._hypotheses) or 1.0
        for hypothesis in self._hypotheses:
            hypothesis.score /= total

        tables = []
        weights = []
        for index, issue in enumerate(self._issues):
            table = {value: 0.0 for value in issue.all}
            for hypothesis in self._hypotheses:
                for value, score in hypothesis.value_tables[index].items():
                    table[value] += hypothesis.score * score
            tables.append(normalize_mapping(table))
            weights.append(sum(hypothesis.score * hypothesis.weights[index] for hypothesis in self._hypotheses))
        self._belief_tables = tables
        self._belief_weights = weights

    def _hypothesis_utility(self, hypothesis: BeliefHypothesis, outcome: Outcome) -> float:
        values = self.outcome_values(outcome)
        totalw = sum(hypothesis.weights) or 1.0
        total = 0.0
        for index, value in enumerate(values):
            total += (hypothesis.weights[index] / totalw) * hypothesis.value_tables[index].get(value, 0.5)
        return total
