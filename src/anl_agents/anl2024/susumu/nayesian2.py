import numpy as np
from anl.anl2024.negotiators.base import ANLNegotiator
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAOResponse, SAOState


# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.simplefilter("ignore", ConvergenceWarning)

__all__ = ["Nayesian2"]


class Nayesian2(ANLNegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes = tuple()

    partner_reserved_value = 0

    def __init__(self, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        # keeps track of times at which the opponent offers

        # keeps track of the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        # self._rational: list[tuple[float, float, Outcome]] = []

    def initialize(self):
        self.opponent_times = np.array([])
        # keeps track of opponent utilities of its offers
        self.opponent_utilities = np.array([])
        # keeps track of our last estimate of the opponent reserved value
        self.oppnent_rv_range = [0, 1]

        # self.offered_u_min = 1

        # kernel = C(0.2, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
        # Create Gaussian Process model
        # self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    def get_pareto_outcomes(self, ordered_outcomes):
        """
        Extracts Pareto outcomes from a list of ordered outcomes.

        Parameters:
        - ordered_outcomes: A list of tuples, where each tuple is (your_utility, opponent_utility, outcome),
                            ordered by your_utility in descending order.

        Returns:
        - A list of tuples representing Pareto outcomes.
        """
        pareto_outcomes = []
        max_opponent_util = float("-inf")

        for your_util, opponent_util, outcome in ordered_outcomes:
            # If the current outcome has a higher opponent utility than all previously seen outcomes,
            # it is Pareto optimal (because your utility is already sorted).
            if opponent_util > max_opponent_util:
                pareto_outcomes.append((your_util, opponent_util, outcome))
                max_opponent_util = opponent_util

        return pareto_outcomes

    def on_preferences_changed(self, changes):
        """
        Called just after the ufun is set and before the negotiation starts.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # print('preference changed')
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.initialize()

        outcomes = self.nmi.outcome_space.enumerate_or_sample()

        ordered_outcomes = sorted(
            [
                (self.ufun(outcome), self.opponent_ufun(outcome), outcome)
                for outcome in outcomes
            ],
            key=lambda x: x[0],
            reverse=True,
        )  # sort from high to low according my utils

        ordered_outcomes = self.get_pareto_outcomes(ordered_outcomes)

        self.rational_outcomes = []
        pair_utilities = []
        prod_utilities = []
        for o in ordered_outcomes:
            if o[0] > self.ufun.reserved_value:
                self.rational_outcomes.append(o[2])
                pair_utilities.append([o[0], o[1]])
                prod_utilities.append((o[0] - self.ufun.reserved_value) * o[1])

        self.pair_utilities = np.array(pair_utilities)

        # self.u_max = self.pair_utilities[0, 0]
        self.u_max = 1

        prod_utilities = np.array(prod_utilities)

        best_nash_outcome_i = np.argmax(prod_utilities)
        self.best_nash_u = self.pair_utilities[best_nash_outcome_i, 0]

        self.opp_highest_rv = self.pair_utilities[-1, 1]
        self.opp_lowest_rv = 0
        self.opp_rv_hypotheses = np.linspace(
            self.opp_lowest_rv, self.opp_highest_rv - 0.001, 10
        )

        self.predicted_rv = 0

        self.predicted_nash_u = self.best_nash_u
        self.my_worst_u = self.pair_utilities[-1, 0]
        self.predicted_best_u = self.pair_utilities[0, 0]

        self.acceptable_utility = 1

        nsteps__ = (
            self.nmi.n_steps
            if self.nmi.n_steps
            else int(
                (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.total_steps = nsteps__
        self.proposed_offers = 0
        self.recieved_offers = 0
        # print('ufun_initilize')

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
        # print('called')
        offer = state.current_offer
        # print('called', offer)
        if offer is not None:
            self._update_self_state(
                state
            )  # update time sequence, update reservation value estimate, update Nash

        self.to_propose = self.bidding_strategy(state)
        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            # print('return None')
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            # print('accept')
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.to_propose)

    def _update_self_state(self, state):
        # print('update self state')

        offer = state.current_offer
        self.recieved_offers += 1
        # time = state.relative_time
        time = self.recieved_offers / self.total_steps
        # print(time, step)
        if offer is None:
            return

        # offer_u_on_me = self.ufun(offer)
        offer_u_on_opp = self.opponent_ufun(offer)
        u_on_pareto_i = np.searchsorted(
            self.pair_utilities[:, 1], offer_u_on_opp, side="right"
        )
        # print(u_on_pareto_i, offer_u_on_opp, self.pair_utilities[u_on_pareto_i-1, 1], self.pair_utilities[u_on_pareto_i-1, 0])
        offer_u_on_me = self.pair_utilities[u_on_pareto_i - 1, 0]

        if offer_u_on_opp < self.opp_highest_rv:
            self.opp_highest_rv = offer_u_on_opp
        if offer_u_on_me > self.my_worst_u:
            self.my_worst_u = offer_u_on_me

        self.opponent_utilities = np.append(self.opponent_utilities, offer_u_on_opp)
        self.opponent_times = np.append(self.opponent_times, time)
        self.opp_rv_hypotheses = np.linspace(
            self.opp_lowest_rv, self.opp_highest_rv - 0.001, 10
        )

        self.predict_rv()
        self.predict_nash_u()
        # print(self.predicted_nash_u, self.predicted_best_u, self.predicted_rv, self.opp_highest_rv)

    def linear_predictor(self):
        t = self.opponent_times[-5:].mean()
        u = self.opponent_utilities[-5:].mean()
        p_u = u + (u - 1) * (1 - t) / t
        return p_u

    def predict_rv(self):
        if self.opponent_times.size >= 5:
            times_reshaped = self.opponent_times[-5:]
            # future_times = np.array([1.0])  # Example future times
            # future_times_reshaped = future_times.reshape(-1, 1)
            # predicted_utilities, std_dev = self.gp.predict(future_times_reshaped, return_std=True)
            bayesian_predicted_rv = self.Bayesian_predictor(
                times_reshaped, self.opponent_utilities[-5:]
            )
            # ln_predicted_rv = self.linear_predictor()
            # if (gp_predicted_rv <= 0) or (gp_predicted_rv >= self.opp_highest_rv) or (std_dev >= 0.25):
            #     if (ln_predicted_rv <= 0) or (ln_predicted_rv >= self.opp_highest_rv):
            #         self.predicted_rv = np.random.uniform(0, self.opp_highest_rv)
            #     else:
            #         self.predicted_rv = np.random.uniform(0, ln_predicted_rv)
            # else:
            self.predicted_rv = bayesian_predicted_rv
            pass  # print(bayesian_predicted_rv, self.opp_highest_rv)
        else:
            self.predicted_rv = np.random.uniform(0, self.opp_highest_rv)

    def Bayesian_predictor(self, u, t):
        # Hypothess parameter sets
        a_values = self.opp_rv_hypotheses
        b_values = np.array([1, 2, 5])

        # Prior (uniform prior over 'a' and 'b' values)
        prior_a = np.ones(len(a_values)) / len(a_values)
        prior_b = np.ones(len(b_values)) / len(b_values)

        # Likelihood function for observing data 'u' given 'a' and 't'
        def likelihood(u, t, a, b):
            predicted_u = (1 - a) * (1 - np.power(t, b)) + a
            # Assuming Gaussian noise with std deviation sigma
            sigma = 0.1
            return np.exp(-0.5 * np.sum((np.array(u) - predicted_u) ** 2) / sigma**2)

        # Compute the posterior for 'a' by marginalizing over 'b'
        posteriors_a = np.zeros(len(a_values))

        for i, a in enumerate(a_values):
            # Calculate likelihood for each 'b' and sum (marginalize over 'b')
            likelihood_sum = 0
            for j, b in enumerate(b_values):
                likelihood_sum += likelihood(u, t, a, b) * prior_b[j]
            posteriors_a[i] = likelihood_sum * prior_a[i]
        # Normalize the posteriors
        if np.all(posteriors_a == 0):
            posteriors_a = np.ones_like(a_values)
        posteriors_a /= np.sum(posteriors_a)
        predicted_a = np.sum(a_values * posteriors_a)

        return predicted_a

    def predict_nash_u(self):
        u_i = np.searchsorted(
            self.pair_utilities[:, 1], self.predicted_rv, side="right"
        )
        self.predicted_best_u = self.pair_utilities[u_i - 1, 0]

        predict_nash_i = np.argmax(
            (self.pair_utilities[:, 0] - self.ufun.reserved_value)
            * (self.pair_utilities[:, 1] - self.predicted_rv)
        )
        self.predicted_nash_u = self.pair_utilities[predict_nash_i, 0]

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        cur_step = state.step + 1
        t = cur_step / self.total_steps
        self.total_steps - cur_step

        # opp_left_turns = self.total_steps - self.recieved_offers
        # my_left_turns = self.total_steps - self.proposed_offers
        # print('bidding:', my_left_turns, opp_left_turns, len(self.opponent_utilities), state.step)

        # asp_1 = (1 - self.best_nash_u) * (1.0 - np.power(t, 5)) + self.best_nash_u
        asp_1 = (self.u_max - self.predicted_nash_u) * (
            1 - np.power(t, 5)
        ) + self.predicted_nash_u
        (self.u_max - self.my_worst_u) * (1 - np.power(t, 5)) + self.my_worst_u

        asp_3_1 = (self.u_max - self.predicted_best_u) * (
            1 - np.power(t, 5)
        ) + self.predicted_best_u
        asp_3_2 = (self.u_max - self.best_nash_u) * (
            1 - np.power(t, 5)
        ) + self.best_nash_u
        if t < 0.8:
            noise = np.random.uniform(0, 0.005)
        else:
            noise = 0

        # asp = t * asp_2 + (1-t) * ((t) * asp_1 + (1 - t) * max(asp_3_1, asp_3_2)) + noise
        asp = ((t) * asp_1 + (1 - t) * max(asp_3_1, asp_3_2)) + noise

        index = np.searchsorted(self.pair_utilities[:, 0][::-1], asp, side="right")
        # print('searched:', self.pair_utilities[:,0][::-1][index-1])
        index = self.pair_utilities[:, 0].size - index
        if index == self.pair_utilities.shape[0]:
            index = index - 1

        offer_utility = self.pair_utilities[index, 0]

        # if offer_utility < self.acceptable_utility:
        self.acceptable_utility = offer_utility

        offer = self.rational_outcomes[index]
        # print(offer_utility, asp, self.ufun(offer), asp_1, asp_3_1, asp_3_2)
        # print('indexted:', self.pair_utilities[:,0][index])
        # print('bidding_u:', self.ufun(self.rational_outcomes[index]))
        return offer

    def acceptance_strategy(self, state: SAOState) -> bool:
        offered_u = self.ufun(state.current_offer)
        to_propose_u = self.ufun(self.to_propose)
        cur_step = state.step + 1
        left_step = self.total_steps - cur_step
        if left_step <= 0:
            if offered_u > self.ufun.reserved_value:
                return True
        # elif left_step <= 2:
        #     if (offered_u >= self.my_worst_u):
        #         return True
        #     else:
        #         return False
        elif offered_u >= to_propose_u:
            return True
        else:
            return False
        # else:
        #     return False

        # if predicted_nash_u_new > self.worst_nash_u:
        #     self.predicted_nash_u = predicted_nash_u_new


if __name__ == "__main__":
    from .helpers.runner import run_a_tournament

    # if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
    run_a_tournament(Nayesian2, small=True)
