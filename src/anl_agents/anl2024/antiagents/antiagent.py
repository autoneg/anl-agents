"""
**Submitted to ANAC 2024 Automated Negotiation League**
Agent Name: AntiAgent
Team Name: AntiAgents
Contact Email: p.aronis@students.uu.nl
Affiliation: Utrecht University, Department of Information and Computing Sciences
Country: Netherlands
Team Members:
    1. Panagiotis Aronis <p.aronis@students.uu.nl>
    2. Sander van Bennekom <s.vanbennekom1@students.uu.nl>
    3. Mats Buis <m.p.buis@students.uu.nl>
    4. Vedant Puneet Singh <v.p.singh@students.uu.nl>
    5. Collin de Wit <c.r.dewit@students.uu.nl>
"""

from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState
from negmas.preferences import pareto_frontier
from scipy.optimize import curve_fit
from scipy.special import erf

__all__ = ["AntiAgent"]


class AntiAgent(ANLNegotiator):
    """
    Class to instantiate the implemented negotiating agent.
    """

    def on_preferences_changed(self, changes):
        """
        Agent intialization. For ANL2024 will only be called once before the negotiation starts. Changes is ignored.
        """
        assert self.ufun and self.opponent_ufun
        # are we bidding first according to the negotiation protocol?
        self.first = not self.nmi.negotiator_index(self.id)
        # number of negotiation rounds

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.deadline = nsteps__
        # all opponent offer timestamps
        self.opp_times = []
        # all opponent offer own utilities
        self.opp_utils = []
        # maximum opponent offered own utility
        self.opp_util_max = 0
        # minimum opponent offered own utility
        self.opp_util_min = 1
        # relative time to start using rv-fitting technique
        self.rvfit_time = 0.90
        # number of unique opponent offers before we start curve estimation
        self.min_unique_utils = 25
        # when both of the above is met (then rv estimate is from rv-fit)
        self.enough_data = False
        # when opp_rv estimation changed (to avoid unnecessary computation)
        self.rv_changed = False
        # estimate of opponent's reservation value
        self.opp_rv = 0.5
        # estimate of opponent's assumed time dependent tactic concession exponent
        self.opp_e = None
        # rational pareto optimal outcomes and their own and opp utilities
        oo = list(self.nmi.outcomes) if self.nmi.outcomes else []
        pareto_outcomes = pareto_frontier((self.ufun, self.opponent_ufun), oo)  # type: ignore
        self.outcomes = [
            (oo[index], utils[0], utils[1]) for utils, index in zip(*pareto_outcomes)
        ]
        # sort rational pareto outcomes by own utility
        self.outcomes.sort(key=lambda outcome: outcome[1], reverse=True)
        # optimal future bids that maximize expected own utility
        self.optimal_bids = []
        # current outcome to offer
        self.current_offer = self.ufun.best()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called to accept / reject current offer and (counter-)offer.
        Args:
            state: the SAOState containing the offer from your partner (None if you are just starting the negotiation)
            and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type SAOResponse which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.
        """
        assert self.ufun and self.opponent_ufun
        # number of remaining negotiation rounds where we can make a bid (one extra round if we are first)
        self.bids_left = self.deadline - 1 - state.step + self.first
        # update the opponent reservation value and concession exponent
        self.update_partner_reserved_value(state)
        # compute the optimal bidding strategy countering the modeled opponent acceptance strategy
        self.optimal_bids = self.maximize_expected_utility()
        # determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        # if it is not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Assume that the opponent bidding history follows a time dependent tactic and
        find the reservation value and concession exponent that best fit their bidding curve.
        Initially make a rough estimation of the opponent reservation value given the current negotiation gap.
        """
        # current opponent offer
        offer = state.current_offer
        if not offer:
            return
        # opponent utility of the outcome
        assert self.opponent_ufun
        opp_util = float(self.opponent_ufun(offer))
        # update opponent offer timestamps and own utilities
        self.opp_times.append(state.relative_time)
        self.opp_utils.append(opp_util)
        self.opp_util_max = max(self.opp_util_max, opp_util)
        self.opp_util_min = min(self.opp_util_min, opp_util)
        # only use RV-fitting when close to deadline to get the best estimate and avoid computation
        if state.relative_time < self.rvfit_time:
            # otherwise make a rough opp_rv estimation given the current negotiation gap
            # assume opponent does not offer anything below their rv, so take the midpoint as rv
            mid_rv = self.opp_util_min / 2
            if abs(mid_rv - self.opp_rv) > 0.01:
                self.opp_rv = mid_rv
                self.rv_changed = True
            return
        # if there is still not enough data for an accurate estimation
        # stay with the initial naive assumption for the opponent
        if not self.enough_data:
            n_unique_utils = len(set(round(util, 3) for util in self.opp_utils))
            self.enough_data = n_unique_utils >= self.min_unique_utils
        if not self.enough_data:
            return
        # use curve fitting to estimate the opponent reservation value and concession exponent
        # assuming that opp_rv is in [0, opp_util_min] and opp_e is in [0.1, 10.0]
        try:
            (opt_rv, opt_e), _ = curve_fit(
                lambda t, rv, e: (self.opp_util_max - rv) * (1.0 - t**e) + rv,
                self.opp_times,
                self.opp_utils,
                bounds=((0.0, 0.1), (self.opp_util_min, 10.0)),
            )
            if abs(opt_rv - self.opp_rv) > 0.01:
                self.opp_rv, self.opp_e = opt_rv, opt_e
                self.rv_changed = True
        except Exception:
            pass

    def opponent_acceptance_probability(self, opp_util: float) -> float:
        """
        Assume that the opponent will accept anything above their real reservation value for any turn.
        Return the probability of accepting an outcome of opponent utility opp_util given the opp_rv estimate.
        """
        # assume the opponent will always accept something better for them than what they already offered
        if opp_util > self.opp_util_min:
            return 1
        # we assume that our current estimate for the opponent reservation value is a gaussian distribution
        # with mu = opp_rv and sigma such that the whole range of possible rv values exactly fits [0, opp_util_min]
        sigma = self.opp_util_min / 6
        # we get the probability of opponent acceptance (opp_util > rv) from the corresponding error function
        prob = 0.5 + 0.5 * erf(
            (opp_util - self.opp_rv) / (sigma * 1.42)
        )  # P = Φ((x-μ)/σ)
        return prob

    def maximize_expected_utility(self) -> list[tuple[float, Outcome]]:
        """
        Compute the optimal bidding strategy assuming the modeled opponent acceptance strategy.
        Returns the outcomes to offer to maximize expected own utility for each remaining round.
        """
        # recompute the bidding strategy only when necessary
        if self.optimal_bids and not self.rv_changed:
            return self.optimal_bids
        self.rv_changed = False
        # with no bidding rounds left we always get our reservation value
        self.optimal_bids = [(self.reserved_value, None)]
        # bidding history is monotonic (to avoid some computation)
        last_index = len(self.outcomes) - 1
        # probability that the opponent will accept each outcome according to the opp model
        prob_opp_accs = [
            self.opponent_acceptance_probability(opp_util)
            for _, _, opp_util in self.outcomes
        ]
        # compute (previous) bids that maximize expected own utility using backwards induction
        for _turns_left in range(1, self.bids_left + 1):
            # compute the optimal outcome to offer with one extra bid remaining
            best_extra_util = 0
            best_index = last_index
            prev_expected_util = self.optimal_bids[-1][0]
            # find outcome that maximizes our expected utility
            for index, ((outcome, own_util, opp_util), prob_opp_acc) in enumerate(
                zip(self.outcomes[: last_index + 1], prob_opp_accs)
            ):
                # extra utility achieved if we offer this outcome
                extra_util = (own_util - prev_expected_util) * prob_opp_acc
                # update current best for this turn
                if extra_util > best_extra_util:
                    best_extra_util = extra_util
                    best_index = index
            # found best offer for this turn
            self.optimal_bids.append(
                (prev_expected_util + best_extra_util, self.outcomes[best_index][0])
            )
            last_index = best_index
        return self.optimal_bids

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determine whether or not to accept the current opponent offer.
        """
        # current opponent offer
        offer = state.current_offer
        # reject a non-offer
        if not offer:
            return False
        # own utility of the outcome
        assert self.ufun
        own_util = float(self.ufun(offer))
        # current own expected utility
        expected_util, _ = self.optimal_bids[self.bids_left]
        # set the acceptance threshold at our current expected utility
        # but if thats higher than what we gonna offer next lower it to that amount
        threshold = min(expected_util, float(self.ufun(self.current_offer)))
        # accept the offer if it is better than the utility threshold
        accept = own_util > threshold
        return accept

    def bidding_strategy(self, state: SAOState) -> Outcome:
        """
        Determine our current (counter-)offer.
        """
        # current optimal outcome to offer
        _, optimal_offer = self.optimal_bids[self.bids_left]
        # keep our bidding curve (not strictly) monotonic
        assert self.ufun
        if float(self.ufun(optimal_offer)) < float(self.ufun(self.current_offer)):
            # go with the optimal offer only if it is a new needed concession
            self.current_offer = optimal_offer
        return self.current_offer
