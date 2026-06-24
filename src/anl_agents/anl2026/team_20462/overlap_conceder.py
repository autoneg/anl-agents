import random
import numpy as np
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class OverlapConceder(SAOCallNegotiator):
    rational_outcomes = tuple()

    def __init__(
        self,
        *args,
        # Concession curve parameters
        beta_a=2.7403,  # lower bound time exponent
        beta_b=3.8587,  # upper bound time exponent
        target_scale=1.0623,  # scale factor for the target utility range
        target_shift=0.4116,  # additive shift for the target utility range
        target_shift_b=0.0680,  # additive shift for the upper bound
        # Acceptance parameters
        advant=0.7968,  # advantage threshold over opponent estimate
        # Opponent modelling parameters
        gamma=0.9753,  # decay factor for older offers
        opp_util_min=0.1326,  # minimum estimated opponent utility for unseen outcomes
        opp_util_max=0.5321,  # maximum estimated opponent utility for an exact match
        # Bidding parameters
        top_k=7,  # number of best welfare outcomes to choose from
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.target_scale = target_scale
        self.target_shift = target_shift
        self.target_shift_b = target_shift_b
        self.advant = advant
        self.gamma = gamma
        self.opp_util_min = opp_util_min
        self.opp_util_max = opp_util_max
        self.top_k = top_k

        self.opponent_offers = []

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.rational_outcomes = tuple(_[1] for _ in sorted(ufun_outcome, reverse=True))

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda x: 0.5)

    def _target_bounds(self, t: float):
        delta = self.ufun.max() - self.ufun.reserved_value
        lower = (
            self.ufun.max() - delta * (t**self.beta_a)
        ) / self.target_scale + self.target_shift
        upper = (
            (self.ufun.max() - delta * (t**self.beta_b)) / self.target_scale
            + self.target_shift
            + self.target_shift_b
        )
        return lower, upper

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        offer = state.current_offer
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is None:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state)
            )

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(
            ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state)
        )

    def acceptance_strategy(self, state: SAOState) -> bool:
        if self.ufun is None or state.current_offer is None:
            return False

        offer = state.current_offer
        U_low, _ = self._target_bounds(state.relative_time)

        my_utility = self.ufun(offer)
        if my_utility >= U_low:
            return True

        opponent_ufun = self.private_info.get("opponent_ufun")
        if opponent_ufun is not None:
            opponent_utility = opponent_ufun(offer)
            if my_utility > opponent_utility + self.advant:
                return True

        return False

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        U_low, U_high = self._target_bounds(state.relative_time)
        target = random.uniform(U_low, U_high)

        opponent_ufun = self.private_info.get("opponent_ufun")
        if opponent_ufun is not None and len(self.opponent_offers) > 0:
            candidate_outcomes = []
            for outcome in self.rational_outcomes:
                my_utility = self.ufun(outcome)
                if U_low <= my_utility <= U_high:
                    opponent_utility = opponent_ufun(outcome)
                    combined_score = my_utility * opponent_utility
                    candidate_outcomes.append((combined_score, outcome))

            if candidate_outcomes:
                candidate_outcomes.sort(reverse=True, key=lambda x: x[0])
                top_n = min(self.top_k, len(candidate_outcomes))
                return random.choice(candidate_outcomes[:top_n])[1]

        best_conceding = None
        for outcome in reversed(self.rational_outcomes):
            if self.ufun(outcome) >= target:
                best_conceding = outcome
                break

        if best_conceding is None:
            return self.rational_outcomes[0]
        return best_conceding

    def update_opponent_model(self, state: SAOState) -> None:
        if state.current_offer is None:
            return

        self.opponent_offers.append(state.current_offer)

        gamma = self.gamma
        opp_min = self.opp_util_min
        opp_max = self.opp_util_max

        def opponent_utility(x):
            if not self.opponent_offers:
                return 0.5

            n_issues = len(x)
            similarities = []
            weights = []

            for idx, offer in enumerate(reversed(self.opponent_offers)):
                matches = sum(1 for i in range(n_issues) if x[i] == offer[i])
                sim = matches / n_issues
                w = gamma**idx
                similarities.append(sim)
                weights.append(w)

            avg_sim = np.average(similarities, weights=weights)
            return opp_min + (opp_max - opp_min) * avg_sim

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=opponent_utility)
