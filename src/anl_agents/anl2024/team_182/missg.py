"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator

from negmas.common import MechanismState

# from helpers.runner import run_a_tournament
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState


from scipy.stats import norm

__all__ = ["MissG"]


class MissG(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0

    rv_history = []
    partner_behavior_scores = []
    agent_behavior_scores = []
    reservation_estimates = []
    behavior_difference = -0.2
    previous_offer = None
    previous_bid = None

    # Bidding
    # Define when tactic change happen, where negotiation start at time 0 and end at 1.
    tactic_change_threshold = 0.9
    # epsilons for bidding
    explore_epsilon = 1
    exploit_epsilon = 0
    explore_threshold_history = []

    # geeg part
    acceptance_threshold_history = []
    threshold_history = []
    acceptance_threshold = 0.8

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent is the average value
        self.partner_reserved_value = 0.5
        self.rv_history.append(self.partner_reserved_value)
        # Calculate pareto frontier
        self.pareto_front = self._find_pareto_front()
        self.best_offer__ = self.ufun.best()

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
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)
        self.update_partner_behavior(state)
        self.previous_offer = offer

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def update_acceptance_threshold(self):
        try:
            # based on the opponent behaviour lower or higher the threshold
            if self.get_behavior_difference() < 0:
                if all(score == 1 for score in self.partner_behavior_scores[-5:]):
                    for n in range(4):
                        self.acceptance_threshold += 0.0005
                else:
                    self.acceptance_threshold -= 0.008
                self.acceptance_threshold = max(
                    0.0, min(1.0, self.acceptance_threshold)
                )
            if self.get_behavior_difference() >= 0:
                self.acceptance_threshold += 0.01
                self.acceptance_threshold = max(
                    0.0, min(1.0, self.acceptance_threshold)
                )
        except Exception:
            pass  # print("error in updating the acceptance threshold", e)
        self.acceptance_threshold = max(self.acceptance_threshold, self.reserved_value)
        self.acceptance_threshold_history.append(self.acceptance_threshold)
        return self.acceptance_threshold

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determine whether to accept or reject an offer based on the negotiation phase and the offer's utility.

        Args:
            state: The current state of the negotiation.

        Returns:
            bool: True if the offer should be accepted, False otherwise.
        """
        try:
            if (
                self.nmi.n_steps is not None and state.step / self.nmi.n_steps < 0.9
            ) or (self.nmi.n_steps is None and state.relative_time < 0.9):
                return False
            # Calculate the acceptance threshold based on the current state, which includes the negotiation phase, the offer's utility and reservation value
            assert self.ufun and self.opponent_ufun
            offer = state.current_offer
            offer_utility = float(self.ufun(offer))
            assert self.ufun
            acceptance_threshold = self.update_acceptance_threshold()
            threshold = (
                self.ufun.reserved_value
                + (1 - self.ufun.reserved_value) * acceptance_threshold
            )
            self.threshold_history.append(threshold)

            opponent_rv = self.partner_reserved_value

            # Adjust acceptance threshold based on the difference between agent's and opponent's reservation values
            # Make the agent more cautious if opponent's RV is close to agent's RV
            if abs(opponent_rv - self.ufun.reserved_value) < 0.1:
                cautious_factor = 1.4  # if below then lets the opponent win (but almost 1 negotiation time) and if above takes too much time
                acceptance_threshold *= cautious_factor
            self.acceptance_threshold_history.append(self.acceptance_threshold)

            if offer_utility >= threshold:
                return True

            if offer_utility >= acceptance_threshold:
                return True

        except Exception:
            pass  # print("error in accepting a bid", e)
        return False

    def plot_acceptance_thresholds(self):
        plt.plot(self.acceptance_threshold_history, label="Acceptance Threshold")
        plt.plot(self.threshold_history, label="Threshold")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Evolution of Acceptance Threshold and Threshold")
        plt.legend()
        plt.savefig("acceptance_thresholds.png")
        plt.close()

    def _is_pareto_optimal(self, outcome: Outcome) -> bool:
        """
        Determine whether the outcome is pareto optimal in the given negotiation.
        """
        # By definition, outcome is pareto optimal when utility of either
        # agent cannot be increase without decreasing the utility of the other.
        assert self.ufun and self.opponent_ufun
        for other_outcome in self.rational_outcomes:
            if float(self.ufun(other_outcome)) > float(self.ufun(outcome)) and float(
                self.opponent_ufun(other_outcome)
            ) > float(self.opponent_ufun(outcome)):
                return False
        return True

    def _find_pareto_front(self) -> list:
        """
        Find all the pareto optimal points in the given negotiation.

        Returns: list of pareto optimal points.
        """
        pareto_front = []
        # Enumerate all rational outcomes and keep track of one that is pareto optimal.
        for outcome in self.rational_outcomes:
            if self._is_pareto_optimal(outcome):
                pareto_front.append(outcome)

        # Sort the pareto front from best to worst utility for our agent
        pareto_front.sort(key=lambda x: float(self.ufun(x)), reverse=True)  # type: ignore
        return pareto_front

    def _get_explore_domain(self, state: SAOState) -> list:
        """
        Calcuate the feasible exploration domain at the current moment.
        Exploration domain start conservative (start with points that best
        utility for our agent but may not be for the opponent) and then expand
        to include points with lesser utility as time progress.

        Returns: list of possible offers.
        """
        # Set the possible region of exploration domain
        # Vertical threshold that slide to the left (decrease over time).
        assert self.ufun and self.opponent_ufun
        rel_ = (
            state.step / self.nmi.n_steps
            if self.nmi.n_steps is not None
            else state.relative_time
        )
        explore_threshold_vertical = (
            1  # Initial threshold
            - (1 - self.ufun.reserved_value)  # adjust range
            * (rel_) ** 2  # exponential decay
        )
        self.explore_threshold_history.append(explore_threshold_vertical)

        # Horizontal threshold that slide to the top (increase over time).
        explore_threshold_horizontal = (
            0  # Initial threshold
            + self.partner_reserved_value  # adjust range
            * (rel_) ** 2  # exponential growth
        )
        # Select all outcomes that within the current threshold region
        explore_domain = sorted(
            [
                _
                for _ in self.rational_outcomes
                if (float(self.ufun(_)) > explore_threshold_vertical)
                and (float(self.opponent_ufun(_)) < explore_threshold_horizontal)
            ],
            key=lambda x: float(self.ufun(x)) + float(self.opponent_ufun(x)),  # type: ignore
            reverse=True,
        )

        # If explore domain is empty, then select the best offer from pareto front.
        if not explore_domain:
            explore_domain.append(self.pareto_front[0])

        return explore_domain

    def plot_explore_thresholds(self):
        plt.plot(self.explore_threshold_history, label="Explore Threshold")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Evolution of Explore Threshold")
        plt.legend()
        plt.savefig("explore_thresholds.png")
        plt.close()

    def _epsilon_greedy(self, epsilon, state: SAOState) -> Outcome:
        """
        Epsilon greedy determine action of whether to explore or exploit opponent.
        High epsilon lead to higher chance of exploration whereas
        low epsilon lead to higher chance of exploitation.

        Returns: Bidding offer for the opponent.
        """
        # Explore
        # Select one of the points from the exploration domain.
        if random.uniform(0, 1) < epsilon:
            possible_offers = self._get_explore_domain(state)
            return possible_offers[random.randint(0, round(len(possible_offers) / 10))]

        # Exploit

        # The opponent might be stubborn in two situation:
        # (1) it have high reservation value
        # (2) it want us to think that it have high reservation value
        # Either case, our estimation of opponent reservation value would be high.
        # So, we also want to try out reservation value that is below the estimation.
        n_steps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps is not None
            else min(self.nmi.time_limit * state.step / state.time, self.nmi.n_outcomes)
        )
        adjusted_opp_res_val = (
            0  # minimum value
            + (self.partner_reserved_value)  # maximum value
            # exponentially grow reservation value from minimum to maximum
            * (
                (state.step + 1 - n_steps__ * self.tactic_change_threshold)
                / (n_steps__ * (1 - self.tactic_change_threshold))
            )
            ** 2
        )

        if adjusted_opp_res_val <= 0:
            adjusted_opp_res_val = self.partner_reserved_value / 2

        # Remove any offer that is below the reservation values.
        assert self.ufun and self.opponent_ufun
        possible_offers = [
            _
            for _ in self.pareto_front
            if float(self.opponent_ufun(_)) > adjusted_opp_res_val
        ]
        # If there is no possible offers (e.g. our estimation of opponent rv is wrong)
        # then revert to offering our top offer
        if not possible_offers:
            return self.best_offer__

        # Otherwise, select from the best possible offer
        return possible_offers[0]

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        # Draft for bidding strategy

        # Phase 1: exploration
        # In the first phase, diverse set of bidding is offered to learn about
        # reservation value and strategy of the opponent as well as concealing
        # the strategy of our agent.

        n_steps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps is not None
            else min(self.nmi.time_limit * state.step / state.time, self.nmi.n_outcomes)
        )
        if state.step < n_steps__ * self.tactic_change_threshold:
            bid = self._epsilon_greedy(self.explore_epsilon, state)

        # Phase 2: exploitation
        # In the second pahse, search for a offer that most advantagous to
        # our agent but still beneficial to the opponent.
        else:
            bid = self._epsilon_greedy(self.exploit_epsilon, state)

        # Dont remove this code - Matteo
        self.update_agent_behavior(bid)
        self.previous_bid = bid

        return bid

    # The code below is about the opponent modeling.

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Using the information of the new offers, you can update the estimated reservation value of the opponent.
        This function estimates the opponent function by applying Bayesian Learning.
        The initial estimation is the same as our actual reservation value.
        The prior standard deviation is a hyperparameter.
        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer
        opponent_value = self.opponent_ufun(offer)

        prior_mean = self.partner_reserved_value
        prior_std = (
            0.25  # Hyperparameter, we can initiate this to something else -Marijn
        )
        # Higher values make the rv-history graph smoother, opposite happening for lower values

        # Compute the likelihood using the opponent's utility function
        likelihood = norm.pdf(opponent_value, loc=prior_mean, scale=prior_std)

        # Update mean based on Bayesian inference
        posterior_mean = (prior_mean + opponent_value * likelihood) / (1 + likelihood)

        # Updated opponent's reserved value
        self.partner_reserved_value = posterior_mean

        self.rv_history.append(self.partner_reserved_value)

    def update_partner_behavior(self, state: SAOState) -> None:
        """Measures the concession rate of the opponent.
        Return: 1 - if a concession is made.
                0 - if no concession is made.
        """
        assert self.opponent_ufun and self.ufun
        if self.opponent_ufun(state.current_offer) < self.opponent_ufun(
            self.previous_offer
        ):
            self.partner_behavior_scores.append(1)
        else:
            self.partner_behavior_scores.append(0)

    def update_agent_behavior(self, current_bid) -> None:
        """Measures the concession rate of the agent.
        Return: 1 - if a concession is made.
                0 - if no concession is made.
        """
        assert self.opponent_ufun and self.ufun
        if self.ufun(current_bid) < self.ufun(self.previous_bid):
            self.agent_behavior_scores.append(1)
        else:
            self.agent_behavior_scores.append(0)

    def get_partner_behavior(self):
        """Calculates the average of concessions of the opponent made between 0 and 1.
        The more concessions it makes, the higher the value."""
        if len(self.partner_behavior_scores) > 0:
            return np.mean(self.partner_behavior_scores)
        else:
            return 0

    def get_agent_behavior(self):
        """Calculates the average of concessions of the agent made between 0 and 1.
        The more concessions it makes, the higher the value."""
        if len(self.agent_behavior_scores) > 0:
            return np.mean(self.agent_behavior_scores)
        else:
            return 0

    def get_behavior_difference(self):
        """
        Returns the difference in concessions in that time-step
        neutral = 0
        more conceder = negative value
        more stubborn = positive value
        """
        behavior_difference = self.get_agent_behavior() - self.get_partner_behavior()
        return behavior_difference

    def plot_behaviors(self):
        """Plots the concession rate of each agent and the difference between them."""
        partner_values = []
        agent_values = []
        differences = []
        for i in range(len(self.partner_behavior_scores)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                partner_values.append(np.mean(self.partner_behavior_scores[:i]))
                agent_values.append(np.mean(self.agent_behavior_scores[:i]))
                differences.append(
                    np.mean(self.agent_behavior_scores[:i])
                    - np.mean(self.partner_behavior_scores[:i])
                )

        x = list(range(len(self.partner_behavior_scores)))
        plt.plot(x, np.asarray(partner_values), color="blue", label="Partner")
        plt.plot(x, np.asarray(agent_values), color="red", label="Agent")
        plt.plot(x, np.asarray(differences), color="green", label="Differences")

        # Adding labels and legend
        plt.xlabel("Time")
        plt.ylabel("Mean")
        plt.title("Behavior of the agents")
        plt.legend()
        plt.savefig("behaviors.png")
        plt.close()

    def _on_negotiation_end(self, state: MechanismState) -> None:
        # Plots the estimated RV values throughout the negotiation.
        # plt.plot(self.rv_history)
        # plt.title('RV History')
        # plt.grid(True)
        # lt.savefig('rv_history.png')
        # plt.close()

        # print(self.agent_behavior_scores)
        # print(self.get_agent_behavior())
        # print(self.partner_behavior_scores)
        # print(self.get_partner_behavior())

        # self.plot_behaviors()
        # self.plot_acceptance_thresholds()
        # self.plot_explore_thresholds()

        return super()._on_negotiation_end(state)  # type: ignore


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
# if __name__ == "__main__":
#    run_a_tournament(MissG)
