"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Natures
*Authors* Jumpei Kawahara(s250312x@st.go.tuat.ac.jp)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""

import numpy as np

from .helpers.helperfunctions import (
    get_current_negotiation_index,
    get_agreement_at_index,
    get_number_of_subnegotiations,
    all_possible_bids_with_agreements_fixed,
    is_edge_agent,
    get_nmi_from_index,
)

from anl2025.negotiator import ANL2025Negotiator
from negmas import ResponseType
from negmas.outcomes import Outcome
from negmas.sao.controllers import SAOState

__all__ = ["RivAgent"]


class Range:
    def __init__(self, mx=None, mn=None):
        self.mx = mx
        self.mn = mn

    @property
    def size(self):
        return self.mx - self.mn

    def get_v(self, r):
        return self.mn + self.size * r

    def get_r(self, v):
        return (v - self.mn) / self.size

    def __str__(self):
        return f"[{self.mn}~{self.mx}]"


def normalize(ary):
    total = sum(ary)
    if total == 0:
        return [1.0 / len(ary) for x in ary]
    return [x / total for x in ary]


class ScoreNode:
    def __init__(self, agent, max_depth, oap, depth, parent, agreements):
        self.agent = agent

        self.max_depth = max_depth
        self.oap = oap

        self.depth = depth
        self.rest_depth = self.max_depth - self.depth
        self.is_leaf = self.rest_depth == 0

        self.parent = parent
        self.is_root = self.parent is None

        self.neg_index = self.agent.neg_index + self.depth - 1
        if not self.agent.use_single_neg:
            self.rest_neg_num = (self.agent.neg_num - 1) - self.neg_index

        if not self.is_leaf:
            self.all_next_outcomes = self.agent.get_all_outcomes(self.neg_index + 1)

        self.agreements = agreements
        if not self.is_root:
            self.outcome = self.agreements[-1]

        if self.agent.use_single_neg:
            self.reserved_value = self.agent.my_ufun(None)
        else:
            self.reserved_value = self.agent.my_ufun(
                self.agreements + [None] * self.rest_neg_num
            )

        if not self.is_leaf:
            self.next_outcome_2_child = {}
            for next_outcome in self.all_next_outcomes:
                child = ScoreNode(
                    self.agent,
                    self.max_depth,
                    self.oap,
                    depth=self.depth + 1,
                    parent=self,
                    agreements=self.agreements + [next_outcome],
                )
                self.next_outcome_2_child[str(next_outcome)] = child

            target_next_outcomes = []
            for next_outcome in self.all_next_outcomes:
                if self.judge_append_as_child(next_outcome):
                    target_next_outcomes.append(next_outcome)

            self.descend_next_outcomes = []
            for curr_next_outcome in target_next_outcomes:
                curr_score = self.get_child(curr_next_outcome).score

                insert_index = len(self.descend_next_outcomes)
                while insert_index > 0:
                    compared_next_outcome = self.descend_next_outcomes[insert_index - 1]
                    compared_score = self.get_child(compared_next_outcome).score
                    if curr_score <= compared_score:
                        break
                    insert_index -= 1

                self.descend_next_outcomes.insert(insert_index, curr_next_outcome)

            if not self.is_root:
                self.next_outcome_2_prob = {}
                rest_prob_total = 1.0
                for i, next_outcome in enumerate(self.descend_next_outcomes):
                    if i + 1 < len(self.descend_next_outcomes):
                        prop = rest_prob_total * oap
                        self.next_outcome_2_prob[str(next_outcome)] = prop
                        rest_prob_total -= prop
                    else:
                        self.next_outcome_2_prob[str(next_outcome)] = rest_prob_total

    def judge_append_as_child(self, next_outcome):
        if next_outcome is None:
            return True
        return self.get_child(next_outcome).score > self.get_child(None).score

    @property
    def score(self):
        assert not self.is_root
        if self.is_leaf:
            bid = self.outcome if self.agent.use_single_neg else self.agreements
            return self.agent.my_ufun(bid)
        else:
            ret = 0.0
            for next_outcome in self.descend_next_outcomes:
                ret += (
                    self.next_outcome_2_prob[str(next_outcome)]
                    * self.get_child(next_outcome).score
                )
            if len(self.descend_next_outcomes) > 1 and self.outcome is None:
                ret *= self.agent.coeff["branch_end_neg_prob"]
            return ret

    @property
    def max_child_score(self):
        assert not self.is_leaf
        next_outcome = self.descend_next_outcomes[0]
        return self.get_child(next_outcome).score

    @property
    def end_neg_child_score(self):
        assert not self.is_leaf
        return self.get_child(None).score

    def get_child(self, next_outcome):
        return self.next_outcome_2_child[str(next_outcome)]


class ScoreRoot(ScoreNode):
    def __init__(self, agent, oap):
        super().__init__(
            agent,
            max_depth=1 if agent.use_single_neg else agent.rest_neg_num,
            oap=oap,
            depth=0,
            parent=None,
            agreements=[] if agent.use_single_neg else agent.agreements,
        )


class ScoreSpace:
    def __init__(self, agent):
        self.root = ScoreRoot(agent, oap=self.collect_decayed_oap_sum(agent))

    def collect_decayed_oap_sum(self, agent):
        if len(agent.oap_history) == 0:
            return agent.coeff["oap_init"]
        gamma = agent.coeff["oap_gamma"]
        weighted_sum, weight_sum = 0.0, 0.0
        for i, oap in enumerate(reversed(agent.oap_history)):
            weighted_sum += gamma**i * oap
            weight_sum += gamma**i
        ratio = weighted_sum / weight_sum

        oap_size = agent.coeff["oap_max"] - agent.coeff["oap_min"]
        return agent.coeff["oap_min"] + oap_size * ratio

    @property
    def rng(self):
        return Range(mx=self.root.max_child_score, mn=0.0)

    @property
    def end_neg_score(self):
        return self.root.end_neg_child_score

    @property
    def descend_outcomes(self):
        return self.root.descend_next_outcomes

    def get(self, oc):
        return self.root.get_child(oc).score

    @property
    def descend_offers(self):
        return [o for o in self.root.descend_next_outcomes if o is not None]


class CurveArea:
    def __init__(self, mx):
        self.rng = Range(mx=mx, mn=None)
        self.r_rng = Range(mx=None, mn=None)

    def get_v(self, r):
        return self.rng.get_v(r=self.r_rng.get_r(v=r))


class ThresholdSpace:
    def __init__(self, agent):
        self.rng = Range(
            mx=agent.score_space.rng.mx,
            mn=max(
                agent.score_space.end_neg_score,
                agent.score_space.rng.mx * agent.coeff["th_min_ratio"],
            ),
        )
        agent.set_have_to_end_neg(self.rng.mx <= self.rng.mn)
        if agent.have_to_end_neg:
            return

        self.delta = agent.score_space.rng.get_v(r=agent.coeff["th_delta_r"])


class ThresholdAreaSpace:
    def __init__(self, agent):
        th_space = ThresholdSpace(agent)

        if agent.have_to_end_neg:
            return

        self.area_min_size = 1e-6

        self.areas = [CurveArea(mx=th_space.rng.mx)]
        prev_score = th_space.rng.mx
        for offer in agent.score_space.descend_offers[1:]:
            score = agent.score_space.get(offer)
            if score < th_space.rng.mn:
                area_mn = th_space.rng.mn
                if prev_score - area_mn > th_space.delta:
                    area_mn = prev_score - th_space.delta
                self.areas[-1].rng.mn = area_mn
                break

            if prev_score - score > th_space.delta:
                if self.areas[-1].rng.mx == prev_score:
                    self.areas[-1].rng.mn = self.areas[-1].rng.mx - self.area_min_size
                else:
                    self.areas[-1].rng.mn = prev_score
                self.areas.append(CurveArea(mx=score))
            prev_score = score
        else:
            self.areas[-1].rng.mn = agent.score_space.get(
                agent.score_space.descend_offers[-1]
            )

        self.size = sum([area.rng.size for area in self.areas])

        self.areas[0].r_rng.mx = 1.0
        self.areas[-1].r_rng.mn = 0.0
        for i in range(len(self.areas) - 1):
            r = self.areas[i].r_rng.mx - (self.areas[i].rng.size / self.size)
            self.areas[i].r_rng.mn = r
            self.areas[i + 1].r_rng.mx = r

        self.r_delta = (th_space.delta / self.size) if self.size > 0.0 else 0.0

    def get_v(self, r):
        for area in self.areas:
            if r >= area.r_rng.mn:
                return area.get_v(r)


class Threshold:
    def __init__(self, agent):
        self.agent = agent
        self.area_space = ThresholdAreaSpace(agent)

    def calc_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_r_mn(self, state):
        relative_time = state.step / (self.agent.n_steps - 1)
        th_exp = (
            self.agent.coeff["th_exp_concession"]
            if self.agent.opponent_model.judge_aggressive()
            else self.agent.coeff["th_exp_aggressive"]
        )
        return 1 - relative_time ** (self.agent.coeff["th_exp_aggressive"])

    def calc(self, state):
        r_mn = self.calc_r_mn(state)
        return self.area_space.get_v(r=r_mn)

    def calc_rng(self, state):
        r_mn = self.calc_r_mn(state)
        mn = self.area_space.get_v(r=r_mn)

        r_mx = r_mn + self.area_space.r_delta
        mx = self.area_space.get_v(r=r_mx)

        return Range(mx=mx, mn=mn)


class OpponentModel:
    def __init__(self, agent):
        self.agent = agent
        self.offer_history = []

    def update(self, offer):
        if offer is None:
            return
        self.offer_history.append(offer)

    def calc_oap(self):
        return len(set(self.offer_history)) / self.agent.n_offers

    def calc_preferences(self, target_offers):
        issue_value_score = {
            i: {value: 0.0 for value in values}
            for i, values in self.agent.issue_values_dict.items()
        }
        for t, offer in enumerate(reversed(self.offer_history)):
            base_score = self.agent.coeff["pref_gamma"] ** t
            for i, value in enumerate(offer):
                if value not in issue_value_score[i].keys():
                    issue_value_score[i][value] = 0.0
                issue_value_score[i][value] += base_score

        non_norm_weight_dict = {}
        for i, value_dict in issue_value_score.items():
            n_values = self.agent.issue_n_values_dict[i]
            non_norm_weight_dict[i] = 1 - len(value_dict) / n_values
        weight_dict = {
            k: v
            for k, v in zip(
                non_norm_weight_dict.keys(), normalize(non_norm_weight_dict.values())
            )
        }

        preferences = []
        for offer in target_offers:
            preference = 0.0
            for i, value in enumerate(offer):
                preference += weight_dict[i] * issue_value_score[i][value]
            preferences.append(preference)

        return np.array(preferences, dtype=float)

    def judge_aggressive(self):
        window_size = self.agent.coeff["opp_aggressive_window_size"]
        if len(self.offer_history) < window_size:
            return False
        return (
            len(set(self.offer_history[-window_size:]))
            <= self.agent.coeff["opp_aggressive_th"]
        )


class RivAgent(ANL2025Negotiator):
    def init(self):
        self.coeff = {
            "branch_end_neg_prob": 0.9,  # 0.0 <= branch_end_neg_prob <= 1.0
            "pref_gamma": 0.8,  # 0.0 <= pref_gamma <= 1.0
            "oap_init": 0.5,  # 0.0 <= opa_init <= 1.0
            "oap_rt_min1": 0.25,  # 0.0 <= opa_needed_rt_min <= 1.0
            "oap_rt_min2_n_offer": 2.0,  # opa_needed_rt_min >= 1.0
            "oap_min": 0.45,  # 0.0 <= opa_min <= 1.0
            "oap_max": 0.55,  # 0.0 <= opa_max <= 1.0
            "oap_gamma": 0.5,  # 0.0 <= oap_gamma <= 1.0
            "opp_aggressive_window_size": 5,  # 1 <= opp_aggressive_window_size <= n_steps
            "opp_aggressive_th": 1,  # 1 <= opp_aggressive_th <= opp_aggressive_window_size
            "th_min_ratio": 0.5,  # 0.0 <= th_min_ratio <= 1.0
            "th_exp_aggressive": 1.7,  # th_exp_aggressive > 0.0
            "th_exp_concession": 1.3,  # th_exp_concession > 0.0
            "th_delta_r": 0.1,  # 0.0 < proposal_delta <= 1.0
        }

        self.id_dict = {}
        for neg_id in self.negotiators.keys():
            index = self.negotiators[neg_id].context["index"]
            self.id_dict[index] = neg_id

        self.neg_num = get_number_of_subnegotiations(self)

        self.is_edge = is_edge_agent(self)

        self.first_neg_index = self.collect_first_neg_index()

        self.first_side_ufun = self.collect_first_side_ufun()

        self.is_multi_agree = self.collect_is_multi_agree()
        self.use_single_neg = self.is_edge or self.is_multi_agree

        self.neg_index = -1

        self.oap_history = []

    def collect_first_neg_index(self):
        return list(self.id_dict.keys())[0] if self.is_edge else 0

    def collect_first_side_ufun(self):
        first_neg_id = self.id_dict[self.first_neg_index]
        return self.negotiators[first_neg_id].context["ufun"]

    def collect_is_multi_agree(self):
        if self.is_edge:
            return False

        sample_size, sample_count = 10, 0
        all_bids_tmp = all_possible_bids_with_agreements_fixed(self)
        all_accept_bids_tmp = [
            bid
            for bid in all_bids_tmp
            if sum([int(o is not None) for o in bid]) == self.neg_num
        ]
        for bid in all_accept_bids_tmp:
            if sum([int(o is not None) for o in bid]) < self.neg_num:
                continue
            side_us = [self.first_side_ufun(o) for i, o in enumerate(bid)]
            center_u = self.center_ufun(bid)

            if np.max(side_us) != center_u:
                return False

            sample_count += 1
            if sample_count >= sample_size:
                return True

    @property
    def rest_neg_num(self):
        return self.neg_num - self.neg_index

    @property
    def n_steps(self):
        if self.is_edge:
            return get_nmi_from_index(self, self.first_neg_index).n_steps
        else:
            return get_nmi_from_index(self, self.neg_index).n_steps

    def get_all_offers(self, neg_index):
        if self.is_edge:
            return self.ufun.outcome_space.enumerate_or_sample()
        else:
            return self.ufun.outcome_spaces[neg_index].enumerate_or_sample()

    @property
    def n_offers(self):
        return len(self.get_all_offers(self.neg_index))

    def get_all_outcomes(self, neg_index):
        return self.get_all_offers(neg_index) + [None]

    @property
    def n_issues(self):
        return len(self.get_all_offers(self.neg_index)[0])

    @property
    def issue_values_dict(self):
        issue_values_dict = {i: set() for i in range(self.n_issues)}
        for offer in self.get_all_offers(self.neg_index):
            for i, value in enumerate(offer):
                issue_values_dict[i].add(value)
        return issue_values_dict

    @property
    def issue_n_values_dict(self):
        issue_values_dict = self.issue_values_dict
        return {i: len(issue_values_dict[i]) for i in range(self.n_issues)}

    @property
    def all_bids(self):
        if self.use_single_neg:
            return self.get_all_outcomes(self.neg_index)
        else:
            return all_possible_bids_with_agreements_fixed(self)

    @property
    def agreements(self):
        assert not self.is_edge
        return [get_agreement_at_index(self, i) for i in range(self.neg_index)]

    def center_ufun(self, bid):
        return self.ufun.eval_with_expected(bid, use_expected=False)

    def my_ufun(self, bid):
        if self.is_edge:
            return self.first_side_ufun(bid)
        elif self.is_multi_agree:
            return self.center_ufun(
                [None] * self.neg_index + [bid] + [None] * (self.rest_neg_num)
            )
        else:
            return self.center_ufun(bid)

    def update_neg_if_needed(self):
        if self.neg_index >= len(self.finished_negotiators):
            return

        # finalize negotiation
        if self.neg_index >= 0:
            total_steps = len(self.opponent_model.offer_history)
            if (
                total_steps > self.n_steps * self.coeff["oap_rt_min1"]
                or total_steps > self.n_offers * self.coeff["oap_rt_min2_n_offer"]
            ):
                self.oap_history.append(self.opponent_model.calc_oap())

        # setup negotiation
        self.neg_index = get_current_negotiation_index(self)

        self.score_space = ScoreSpace(self)
        self.threshold = Threshold(self)
        self.opponent_model = OpponentModel(self)

    def set_have_to_end_neg(self, arg):
        self.have_to_end_neg = arg

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self.update_neg_if_needed()

        if self.have_to_end_neg:
            return None

        th_rng = self.threshold.calc_rng(state)

        target_offers = []
        for offer in reversed(self.score_space.descend_offers):
            score = self.score_space.get(offer)
            if score < th_rng.mn:
                continue
            elif score > th_rng.mx:
                # 緊急エラー対策
                if len(target_offers) == 0:
                    target_offers.append(offer)
                break

            if offer not in target_offers:
                target_offers.append(offer)

        if len(target_offers) == 1:
            target_offer = target_offers[0]
        else:
            preferences = self.opponent_model.calc_preferences(target_offers)
            if preferences.sum() == 0:
                selected_index = np.random.choice(len(target_offers))
            else:
                selected_index = np.argmax(preferences)
            target_offer = target_offers[selected_index]

        return target_offer

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self.update_neg_if_needed()

        self.opponent_model.update(state.current_offer)

        if self.have_to_end_neg:
            return ResponseType.END_NEGOTIATION

        th = self.threshold.calc(state)
        opponent_score = self.score_space.get(state.current_offer)
        if opponent_score >= th:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


if __name__ == "__main__":
    from anl2025.negotiator import Boulware2025, Random2025, Linear2025, Conceder2025

    do_tournament = True

    if not do_tournament:
        from .helpers.runner import run_negotiation

        results = run_negotiation(
            center_agent=RivAgent,
            edge_agents=[
                # Random2025,
                # Boulware2025,
                Linear2025,
                # Conceder2025,
            ],
            scenario_name="dinners",
            # scenario_name = 'target-quantity',
            # scenario_name = 'job-hunt',
            n_steps=10,
        )

    else:
        from .helpers.runner import run_tournament

        results = run_tournament(
            my_agent=RivAgent,
            opponent_agents=[
                Random2025,
                Boulware2025,
                Linear2025,
                Conceder2025,
            ],
            scenario_names=[
                "dinners",
                # 'target-quantity',
                # 'job-hunt'
            ],
        )
