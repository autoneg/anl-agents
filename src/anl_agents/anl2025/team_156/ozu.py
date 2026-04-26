"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Ozu
*Authors* Emre Kuru <emre.kuru@ozu.edu.tr>
"""

from negmas import ResponseType
from negmas.outcomes import Outcome
from negmas.sao.controllers import SAOState
from anl2025.negotiator import ANL2025Negotiator
from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    is_edge_agent,
    get_agreement_at_index,
    get_outcome_space_from_index,
    get_number_of_subnegotiations,
    find_best_bid_in_outcomespace,
    get_current_negotiation_index,
)
import random
import os
from pathlib import Path
import anl2025

logs_dir = Path(os.getcwd()) / "logs"
logs_dir.mkdir(exist_ok=True)
anl2025.DEFAULT_TOURNAMENT_PATH = logs_dir

__all__ = ["OzUAgent"]


class OzUAgent(ANL2025Negotiator):
    def init(self):
        self.current_neg_index = -1
        self.id_dict = {}
        set_id_dict(self)

        self.W = {
            1: [1],
            2: [0.25, 0.75],
            3: [0.11, 0.22, 0.66],
            4: [0.05, 0.15, 0.3, 0.5],
        }

        self.epsilon = 0.05
        self.target_bid = None

        self.my_last_bids = {}
        self.last_received_bids = {}

        self._set_parameters()
        self._update_target_bid()

    def _set_parameters(self):
        self.p0 = 1.0
        self.p1 = 0.8
        self.p3 = 0.5

        size = len(
            get_outcome_space_from_index(self, get_current_negotiation_index(self))
        )

        if size < 450:
            self.p2 = 0.80
        elif size < 1500:
            self.p2 = 0.775
        elif size < 4500:
            self.p2 = 0.75
        elif size < 18000:
            self.p2 = 0.725
        elif size < 33000:
            self.p2 = 0.70
        else:
            self.p2 = 0.675

        index = get_current_negotiation_index(self)
        total_negotiations = get_number_of_subnegotiations(self)

        if not is_edge_agent(self):
            f = index / total_negotiations
            decay_f = f**2
            self.p2 -= min(0.3, decay_f * 0.2)

        try:
            best_bid = find_best_bid_in_outcomespace(self)[index]
            self.p0 = self._get_offer_utility(best_bid, index)
            self.p2 = (self.p0 / 1) * self.p2
        except:
            best_bid = find_best_bid_in_outcomespace(self)
            self.p0 = self._get_offer_utility(best_bid, index)
            self.p2 = (self.p0 / 1) * self.p2

    def _update_target_bid(self):
        self.target_bid = find_best_bid_in_outcomespace(self)

    def _reset_subnegotiation(self, negotiator_id):
        self.my_last_bids[negotiator_id] = []
        self.last_received_bids[negotiator_id] = []

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if did_negotiation_end(self):
            self._update_target_bid()

        if dest not in self.my_last_bids:
            self._reset_subnegotiation(dest)
            self._set_parameters()

        t = state.relative_time or 0.0
        index = self.negotiators[negotiator_id].context["index"]

        target_util = self._hybrid_target_utility(t, dest)
        window = self._get_bids_within_window(target_util, self.epsilon, index)
        bid = random.choice(window)
        self.my_last_bids[dest].append((bid, self._get_offer_utility(bid, index)))
        return bid

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if did_negotiation_end(self):
            self._update_target_bid()

        if source not in self.last_received_bids:
            self._reset_subnegotiation(source)

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        t = state.relative_time or 0.0
        index = self.negotiators[negotiator_id].context["index"]

        util = self._get_offer_utility(offer, index)
        self.last_received_bids[source].append((offer, util))

        next_bid = self.propose(negotiator_id, state, source)

        if util >= self._get_offer_utility(next_bid, index):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _get_offer_utility(self, offer, index):
        if is_edge_agent(self):
            return self.ufun(offer)

        outcome = []
        for i in range(get_number_of_subnegotiations(self)):
            if i < index:
                outcome.append(get_agreement_at_index(self, i))
            elif i == index:
                outcome.append(offer)
            else:
                outcome.append(None)

        return self.ufun(tuple(outcome))

    def _time_based_utility(self, t):
        return (1 - t) ** 2 * self.p0 + 2 * (1 - t) * t * self.p1 + t**2 * self.p2

    def _behaviour_based_utility(self, t, negotiator_id):
        if (
            len(self.last_received_bids[negotiator_id]) < 2
            or not self.my_last_bids[negotiator_id]
        ):
            return self._time_based_utility(t)

        diffs = [
            self.last_received_bids[negotiator_id][i + 1][1]
            - self.last_received_bids[negotiator_id][i][1]
            for i in range(len(self.last_received_bids[negotiator_id]) - 1)
        ]

        window_size = min(len(diffs), 4)
        weights = self.W[window_size]
        delta = sum(d * w for d, w in zip(diffs[-window_size:], weights))

        latest_util = self.my_last_bids[negotiator_id][-1][1]
        return latest_util - (self.p3 + self.p3 * t) * delta

    def _hybrid_target_utility(self, t, negotiator_id):
        tb = self._time_based_utility(t)
        if len(self.last_received_bids[negotiator_id]) <= 2:
            return tb
        bb = self._behaviour_based_utility(t, negotiator_id)

        return max((1 - t**2) * bb + t**2 * tb, 0)

    def _get_bids_within_window(self, target_util, epsilon, index):
        all_bids = get_outcome_space_from_index(self, index)
        while True:
            in_window = [
                b
                for b in all_bids
                if abs(self._get_offer_utility(b, index) - target_util) <= epsilon
            ]
            if in_window:
                return in_window
            epsilon += 0.01


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    run_a_tournament(OzUAgent, small=True)
