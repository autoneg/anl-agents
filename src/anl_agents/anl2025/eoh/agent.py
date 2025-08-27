"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* EOH Team
*Authors* Advanced Negotiation Agent with Adaptive Concession Strategy

"""

from negmas.outcomes import Outcome

from .helpers.helperfunctions import (
    set_id_dict,
    did_negotiation_end,
    get_target_bid_at_current_index,
    is_edge_agent,
    find_best_bid_in_outcomespace,
    get_current_negotiation_index,
    get_agreement_at_index,
    get_number_of_subnegotiations,
)

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)

__all__ = ["EOHAgent"]


class EOHAgent(ANL2025Negotiator):
    """
    Enhanced negotiation agent with adaptive concession strategy for ANL 2025.
    """

    def init(self):
        """Initialize the agent with enhanced strategy parameters."""
        self.current_neg_index = -1
        self.target_bid = None
        self.id_dict = {}
        set_id_dict(self)

        self.concession_rate = 0.02
        self.risk_tolerance = 0.3
        self.min_acceptable_utility = 0.1
        self.opponent_concession_rate = 0.05

        self.opponent_offers = {}
        self.opponent_utilities = {}
        self.negotiation_history = []

        self.center_aggressiveness = 0.8
        self.edge_cooperation = 0.6

        self.success_rate = 0.5
        self.total_negotiations = 0
        self.successful_negotiations = 0

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """
        Enhanced proposal strategy with time-based concession and opponent modeling.
        """
        if self.target_bid is None:
            self._update_strategy()
        if did_negotiation_end(self):
            self._update_strategy()

        nmi = self.negotiators[negotiator_id][0].nmi
        current_round = state.step
        max_rounds = nmi.n_steps if nmi.n_steps else 100
        time_pressure = current_round / max(1, max_rounds)

        if is_edge_agent(self):
            return self._propose_as_edge(negotiator_id, state, time_pressure)
        else:
            return self._propose_as_center(negotiator_id, state, time_pressure)

    def _propose_as_center(
        self, negotiator_id: str, state: SAOState, time_pressure: float
    ) -> Outcome | None:
        """Proposal strategy when acting as center agent."""
        if self.target_bid is None:
            return None
        target_bid_part = get_target_bid_at_current_index(self)
        if target_bid_part is None:
            return None

        side_ufun = self.negotiators[negotiator_id][1]["ufun"]
        target_utility = side_ufun(target_bid_part)

        negotiation_progress = self.current_neg_index / max(
            1, get_number_of_subnegotiations(self) - 1
        )
        concession_factor = self._calculate_concession_factor(
            time_pressure, is_center=True
        )

        if negotiation_progress < 0.5:
            concession_factor *= 0.7
        else:
            concession_factor *= 1.2

        acceptable_utility = target_utility * (1 - concession_factor)
        proposed_bid = self._find_bid_with_utility_from_current_space(
            acceptable_utility, target_bid_part, negotiator_id
        )

        return proposed_bid

    def _propose_as_edge(
        self, negotiator_id: str, state: SAOState, time_pressure: float
    ) -> Outcome | None:
        """Proposal strategy when acting as edge agent."""
        if self.target_bid is None:
            return None
        target_bid_part = get_target_bid_at_current_index(self)
        if target_bid_part is None:
            return None

        target_utility = self.ufun(target_bid_part)
        concession_factor = self._calculate_concession_factor(
            time_pressure, is_center=False
        )
        acceptable_utility = target_utility * (
            1 - concession_factor * self.edge_cooperation
        )

        return self._find_bid_with_utility_from_current_space(
            acceptable_utility, target_bid_part, negotiator_id
        )

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """
        Enhanced response strategy with opponent modeling and adaptive acceptance.
        """
        if self.target_bid is None:
            self._update_strategy()
        if did_negotiation_end(self):
            self._update_strategy()

        current_offer = state.current_offer
        if current_offer is None:
            return ResponseType.REJECT_OFFER

        side_ufun = (
            self.negotiators[negotiator_id][1]["ufun"]
            if not is_edge_agent(self)
            else self.ufun
        )

        if negotiator_id not in self.opponent_offers:
            self.opponent_offers[negotiator_id] = []
        self.opponent_offers[negotiator_id].append(current_offer)

        offer_utility = side_ufun(current_offer)

        if self.target_bid is None:
            return ResponseType.REJECT_OFFER
        target_bid_part = get_target_bid_at_current_index(self)
        if target_bid_part is None:
            return ResponseType.REJECT_OFFER

        target_utility = side_ufun(target_bid_part)

        nmi = self.negotiators[negotiator_id][0].nmi
        current_round = state.step
        max_rounds = nmi.n_steps if nmi.n_steps else 100
        time_pressure = current_round / max(1, max_rounds)

        if is_edge_agent(self):
            acceptance_threshold = self._calculate_edge_acceptance_threshold(
                target_utility, time_pressure, negotiator_id
            )
        else:
            acceptance_threshold = self._calculate_center_acceptance_threshold(
                target_utility, time_pressure, negotiator_id
            )

        if offer_utility >= acceptance_threshold:
            return ResponseType.ACCEPT_OFFER
        if time_pressure < 0.9 and offer_utility < self.min_acceptable_utility:
            return ResponseType.REJECT_OFFER
        if time_pressure > 0.95 and offer_utility > self.min_acceptable_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_concession_factor(
        self, time_pressure: float, is_center: bool
    ) -> float:
        base_concession = time_pressure**2 * self.concession_rate
        if is_center:
            return (
                base_concession * self.center_aggressiveness
                if time_pressure < 0.3
                else base_concession * (1 + time_pressure)
            )
        else:
            return base_concession * (1 + self.edge_cooperation * time_pressure)

    def _calculate_center_acceptance_threshold(
        self, target_utility: float, time_pressure: float, negotiator_id: str
    ) -> float:
        base_threshold = target_utility * (1 - time_pressure * self.concession_rate)
        if self.success_rate < 0.3:
            base_threshold *= 0.8
        elif self.success_rate > 0.7:
            base_threshold *= 1.1
        return max(base_threshold, self.min_acceptable_utility)

    def _calculate_edge_acceptance_threshold(
        self, target_utility: float, time_pressure: float, negotiator_id: str
    ) -> float:
        base_threshold = target_utility * (
            1 - time_pressure * self.concession_rate * 1.5
        )
        if (
            negotiator_id in self.opponent_offers
            and len(self.opponent_offers[negotiator_id]) > 1
        ):
            recent_offers = self.opponent_offers[negotiator_id][-2:]
            if len(recent_offers) == 2:
                recent_utilities = [self.ufun(offer) for offer in recent_offers]
                if recent_utilities[1] > recent_utilities[0]:
                    base_threshold *= 1.1
                else:
                    base_threshold *= 0.9
        return max(base_threshold, self.min_acceptable_utility)

    def _find_bid_with_utility_from_current_space(
        self, target_utility: float, fallback_bid: Outcome, negotiator_id: str
    ) -> Outcome:
        """Find a bid that achieves approximately the target utility from current negotiation space."""
        try:
            side_ufun = (
                self.negotiators[negotiator_id][1]["ufun"]
                if not is_edge_agent(self)
                else self.ufun
            )
            outcomes = list(side_ufun.outcome_space.enumerate_or_sample(200))

            best_match = fallback_bid
            min_diff = float("inf")

            for outcome in outcomes:
                if outcome is not None:
                    utility = side_ufun(outcome)
                    diff = abs(utility - target_utility)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = outcome

            return best_match
        except Exception:
            return fallback_bid

    def _update_strategy(self) -> None:
        """Update strategy based on negotiation outcomes and performance."""
        current_index = get_current_negotiation_index(self)
        if current_index > 0:
            prev_agreement = get_agreement_at_index(self, current_index - 1)
            if prev_agreement is not None:
                if (
                    not self.negotiation_history
                    or self.negotiation_history[-1] != prev_agreement
                ):
                    self.successful_negotiations += 1
                    self.negotiation_history.append(prev_agreement)

        self.total_negotiations = current_index

        if self.total_negotiations > 0:
            self.success_rate = self.successful_negotiations / self.total_negotiations

        if self.success_rate < 0.2:
            self.concession_rate = min(0.1, self.concession_rate * 1.2)
            self.min_acceptable_utility = max(0.05, self.min_acceptable_utility * 0.9)
        elif self.success_rate > 0.8:
            self.concession_rate = max(0.01, self.concession_rate * 0.8)
            self.min_acceptable_utility = min(0.3, self.min_acceptable_utility * 1.1)

        if not is_edge_agent(self):
            total_negotiations = get_number_of_subnegotiations(self)
            if current_index < total_negotiations / 2:
                self.risk_tolerance = min(0.5, self.risk_tolerance * 1.1)
            else:
                self.risk_tolerance = max(0.1, self.risk_tolerance * 0.9)

        if is_edge_agent(self):
            _, best_bid = self.ufun.extreme_outcomes()
        else:
            best_bid = find_best_bid_in_outcomespace(self)

        self.target_bid = best_bid
