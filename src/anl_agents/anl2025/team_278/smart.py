from negmas import SAOState, Outcome, ResponseType
from anl2025 import ANL2025Negotiator
from .helpers.helperfunctions import *
import random
import numpy as np
from collections import defaultdict

__all__ = ["SmartNegotiator"]


class SmartNegotiator(ANL2025Negotiator):
    def __init__(
        self, beta=2, k=0.7, min_acceptable_utility=0.60, decay=0.03, **kwargs
    ):
        self._beta = beta
        self._k = k
        self.min_acceptable_utility = min_acceptable_utility
        self.decay = decay
        # Remove your custom params from kwargs before passing to superclass
        kwargs.pop("beta", None)
        kwargs.pop("k", None)
        kwargs.pop("min_acceptable_utility", None)
        kwargs.pop("decay", None)
        super().__init__(**kwargs)

    def init(self):
        self.current_neg_index = -1
        self.target_bid = None
        self.id_dict = {}
        set_id_dict(self)
        self.opponent_history = defaultdict(list)

    def _update_strategy(self):
        if is_edge_agent(self):
            _, best_bid = self.ufun.extreme_outcomes()
        else:
            best_bid = find_best_bid_in_outcomespace(self)
        self.target_bid = best_bid

    def _safe_eval_with_offer(self, offer):
        if is_edge_agent(self):
            return self.ufun(offer)

        if offer is None:
            return 0.0

        outcome_space = self.ufun.outcome_space.enumerate_or_sample()

        possible_outcomes = []

        neg_index = get_current_negotiation_index(self)
        n = get_number_of_subnegotiations(self)

        # Add fixed agreements for past negotiations
        for i in range(neg_index):
            possible_outcomes.append([get_agreement_at_index(self, i)])

        possible_outcomes.append([offer])

        # Add possible values for current and future negotiations
        for i in range(neg_index + 1, n):
            values_at_i = [outcome[i] for outcome in outcome_space]
            possible_outcomes.append(
                list(set(values_at_i))
            )  # Use set to remove duplicates

        # Cartesian product to build all valid combinations
        adapted_outcomes = cartesian_product(possible_outcomes)

        mn, mx = float("inf"), float("-inf")
        worst, best = None, None
        for o in adapted_outcomes:
            # print(o)
            u = self.ufun(o)
            if u < mn:
                worst, mn = o, u
            if u > mx:
                best, mx = o, u

        return mx

    def estimate_concession_rate(self, negotiator_id, alpha=0.5):
        """
        Estimate the opponent's concession rate using exponentially weighted moving average (EWMA).

        Parameters:
        - negotiator_id: ID of the opponent.
        - alpha (float): Smoothing factor in (0, 1]. Higher alpha = more weight on recent changes.

        Returns:
        - Float: Estimated average concession rate (positive = conceding, negative = retracting).
        """
        history = self.opponent_history[negotiator_id]

        # Need at least two offers to compute a difference
        if len(history) < 2:
            return 0.0

        ewma = 0.0

        # Loop through consecutive offer differences
        for i in range(1, len(history)):
            diff = history[i] - history[i - 1]  # Change in utility offered by opponent
            # Update EWMA: combine current diff and previous EWMA based on alpha
            ewma = alpha * diff + (1 - alpha) * ewma

        return ewma

    def _boulware_threshold(self, beta=1.2, k=0.6):
        t = (self.current_neg_index + 1) / (max(1, len(self.negotiators)))

        return 1.0 - (1.0 - k) * (t ** (1 / beta))

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if did_negotiation_end(self):
            self._update_strategy()

        # At first neogtiation step, propose best possible possible bid
        if len(self.opponent_history[negotiator_id]) == 0:
            target_bid = get_target_bid_at_current_index(self)
            if target_bid is None:
                return ResponseType.END_NEGOTIATION
            else:
                return target_bid

        # Step is number of offers in opponent history
        step = len(self.opponent_history[negotiator_id])

        # local threshold based on step in current negotiation
        local_threshold = max(1 - step * self.decay, self.min_acceptable_utility)
        # global threshold based on the idx of current negotiation out of all negotiations
        global_threshold = self._boulware_threshold()

        if is_edge_agent(self):
            threshold = local_threshold  # if edge agents use only local threshold
        else:
            threshold = (
                local_threshold + global_threshold
            ) / 2  # if center combine both local and global thresholds

        idx = get_current_negotiation_index(self)

        concession_rate = self.estimate_concession_rate(negotiator_id)

        # adjust threshold based on opponent concession rate
        adjusted_threshold = threshold * (1.0 + np.clip(concession_rate, -0.2, 0.2))

        adjusted_threshold = min(
            adjusted_threshold, 1.0
        )  # make sure threshold doesnt exceed 1

        # get list of possilbe bids sorted by their utility
        best_bids = find_sorted_bids_in_outcomespace(self)

        max_utility = best_bids[0][1]

        adjusted_threshold *= max_utility  # scale best utility by threshold

        # filter for utilities that exceed threshold
        filtered = [
            (val, util) for val, util in best_bids if util >= adjusted_threshold
        ]

        if not filtered:
            return get_target_bid_at_current_index(self)

        values, weights = zip(*filtered)

        # normalize utility weights for random selection
        weights = np.array(weights)
        weights = weights / weights.sum()

        # select random utility from options based on utility weights
        selected = random.choices(values, weights=weights, k=1)[0][idx]

        if selected is None:
            return get_target_bid_at_current_index(self)

        return selected if isinstance(selected, tuple) else (selected,)

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        if did_negotiation_end(self):
            self._update_strategy()

        target_bid = get_target_bid_at_current_index(self)
        if target_bid is None:
            return ResponseType.END_NEGOTIATION

        offer = state.current_offer

        utility = self._safe_eval_with_offer(offer)

        ## update opponent history of utilities
        self.opponent_history[negotiator_id].append(utility)

        # step is the length of opponenet history
        step = len(self.opponent_history[negotiator_id])

        local_threshold = max(1 - step * self.decay, self.min_acceptable_utility)
        global_threshold = self._boulware_threshold()

        if is_edge_agent(self):
            threshold = local_threshold
        else:
            threshold = (local_threshold + global_threshold) / 2

        concession_rate = self.estimate_concession_rate(negotiator_id)

        # adjust threshold based on opponent concession rate
        adjusted_threshold = threshold * (1.0 + np.clip(concession_rate, -0.2, 0.2))

        adjusted_threshold = min(
            adjusted_threshold, 1.0
        )  # make sure threshold doesnt exceed 1

        best_utility = self._safe_eval_with_offer(get_target_bid_at_current_index(self))

        # scale best utility by threshold
        min_utility = adjusted_threshold * best_utility

        # if utility is greater of equal to minimum accept, else reject
        if utility >= min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
