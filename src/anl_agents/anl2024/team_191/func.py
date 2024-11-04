import numpy as np
import copy
import scipy

__all__ = ["update_partner_reserved_value"]


def update_partner_reserved_value(self, state) -> None:
    """This is one of the functions you can implement.
    Using the information of the new offers, you can update the estimated reservation value of the opponent.

    returns: None.
    """
    assert self.ufun and self.opponent_ufun

    self.current_step += 1
    if self.current_step == 1:
        self.new_proposal = self.ufun(self.initial_proposal)
        # initialize rv probabilities with possible bias
        self.probability_prev = [1 / self.num_of_hyp] * self.num_of_hyp
        self.probability_prev[0] += 0.4
        self.probability_prev = np.array(self.probability_prev) / np.sum(
            self.probability_prev
        )
        return

    opp_offer_val = self.opponent_ufun(self.opp_offer)
    mean = 1 - opp_offer_val

    # Adjust std_dev to control the spread of the distribution
    # making it dynamic might not be optimal, need to test

    nsteps__ = (
        self.nmi.n_steps
        if self.nmi.n_steps
        else int(
            (self.nmi.state.time + 1e-6) / (self.nmi.state.relative_time + 1e-6) + 0.5
        )
    )
    std_dev = np.max(np.array([(nsteps__ - self.current_step) / nsteps__ - 0.6, 0.05]))

    # Calculate bucket boundaries
    bucket_boundaries = np.linspace(0, 1, self.num_of_hyp + 1)

    # Calculate probabilities for each bucket
    probabilities = copy.deepcopy(self.probability_prev)

    for index, value in enumerate(probabilities):
        if opp_offer_val < self.opponents_reserved_value[index]:
            # it should never be below, and the ones close to it should also be penalized
            probabilities[index - 3] /= 2
            probabilities[index - 2] /= 3
            probabilities[index - 1] /= 4
            probabilities[index] = 0

    for i in range(self.num_of_hyp):
        if probabilities[index] == 0:
            break
        lower_bound = bucket_boundaries[i]
        upper_bound = bucket_boundaries[i + 1]

        probability = scipy.stats.norm.cdf(
            upper_bound, mean, std_dev
        ) - scipy.stats.norm.cdf(lower_bound, mean, std_dev)
        probabilities[i] += probability

    self.probability_prev /= np.sum(probabilities)
    self.partner_reserved_value = np.dot(
        self.probability_prev, self.opponents_reserved_value
    )
