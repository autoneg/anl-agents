import math
import warnings

import numpy as np
from scipy.optimize import curve_fit

from .helpers.config import DEFAULT_SETTINGS

OPPONENT_MODEL_SETTINGS = DEFAULT_SETTINGS["opponent_model"]


# single variable (step number) quadratic_polynomial function definition for nonlinear regression
def quadratic_polynomial(x, a, b, c):
    return a * x**2 + b * x + c


# two-feature quadratic polynomial function definition for nonlinear regression
def quadratic_2D(x, a, b, c, d, e, f):
    x1, x2 = x[0, :], x[1, :]
    return a * x1**2 + b * x2**2 + c * x1 * x2 + d * x1 + e * x2 + f


# Function to find Pareto Distance of an offer
def pareto_distance(current_offer_utils, front):
    try:
        return min(math.dist(current_offer_utils, pp) for pp in front)
    except Exception:
        pass


# Function to find Nash difference
# def nash_diff(current_offer_utils, welfare):
# return welfare - sum(current_offer_utils)


def nash_diff(opponent_util, opponent_nash_util):
    return opponent_util - opponent_nash_util


def update_reserved_value(agent, offer):
    """Learns the reserved value of the partner"""
    if offer is None:
        return

    # current offer utilities:
    current_self_utility = float(agent.ufun(offer))
    current_opponent_utility = float(float(agent.opponent_ufun(offer)))

    # save to the list of utilities received from the opponent and their times
    agent.opponent_utilities.append(current_opponent_utility)

    if len(agent.opponent_utilities) == 1:
        agent.opponent_ufun.reserved_value = 1
    agent.past_opponent_rv = agent.opponent_ufun.reserved_value

    # We are going to try the interpolation with 2 additional features:
    # 0 - Offer number
    # 1 - Distance to Nash Equilibrium
    # 2 - Distance to the Pareto Frontier
    # We are going to try all combinations of the three and choose the best

    # Add current pareto distance to list
    if agent.pareto_front_exists:
        agent.pareto_distances.append(
            pareto_distance(
                [current_self_utility, current_opponent_utility], agent.frontier_utils
            )
        )
    # Add current Nash Welfare difference to list
    if agent.nash_exists:
        # welfare of both
        # agent.nash_diffs.append(nash_diff([current_self_utility,current_opponent_utility], agent.nash_welfare))
        # welfare of the opponent alone
        agent.nash_diffs.append(
            nash_diff(current_opponent_utility, agent.nash_utils[1])
        )

    # Find window size
    # Find neg_phase

    nsteps__ = (
        agent.nmi.n_steps
        if agent.nmi.n_steps
        else int(
            (agent.nmi.state.time + 1e-6) / (agent.nmi.state.relative_time + 1e-6) + 0.5
        )
    )
    if (agent.nmi.state.step / nsteps__) <= 4 / 8:
        agent.neg_phase = 0
    elif (agent.nmi.state.step / nsteps__) <= 6 / 8:
        agent.neg_phase = 1
    elif (agent.nmi.state.step / nsteps__) <= 7 / 8:
        agent.neg_phase = 2
    else:
        agent.neg_phase = 3

    # Find step_class
    step_ths = [100, 200, 500, 1000, 2500, 5000, 10000]

    for i, threshold in enumerate(step_ths):
        if nsteps__ <= threshold:
            agent.step_class = i
            break

    window_size = OPPONENT_MODEL_SETTINGS["step_class"][
        "step_" + str(agent.step_class)
    ]["phase_" + str(agent.neg_phase)]

    # if there are not enough bids to fill the window, go with what you've got.
    if len(agent.opponent_utilities) < window_size:
        window_size = len(agent.opponent_utilities)
    if window_size < 3:
        window_size = 3

    # based on the value of rv_feature_mode, use 1D, 2D
    if OPPONENT_MODEL_SETTINGS["rv_feature_mode"] == 1 and agent.nash_exists:
        if window_size < 6:
            window_size = 6
        agent.opponent_ufun.reserved_value = interpolation_1(
            agent, window_size, agent.nash_diffs
        )
        agent.effective_rv_feature_mode = 1
    else:
        if window_size < 3:
            window_size = 3
        agent.opponent_ufun.reserved_value = interpolation_0(agent, window_size)
        agent.effective_rv_feature_mode = 0


# interpolation function for rv_feature_mode = 0 (round)
def interpolation_0(agent, window_size):
    # if we did not reach window size, assume opponent rv is 1, highest.
    if len(agent.opponent_utilities) < window_size:
        # agent.past_opponent_rv_list.append(1.0)
        return 1.0

    # else, xs are the last N (window_size) offer numbers of the opponent.
    # ys are the last N offers.
    else:
        x = np.array(
            list(
                range(
                    len(agent.opponent_utilities) - window_size,
                    len(agent.opponent_utilities),
                )
            ),
            dtype=float,
        )
        y = np.array(agent.opponent_utilities[-window_size:], dtype=float)

        try:
            # Fit the curve.
            warnings.filterwarnings("ignore")
            params, covariance = curve_fit(quadratic_polynomial, x, y, p0=[1, 1, 1])
            warnings.filterwarnings("default")

            optimized_a, optimized_b, optimized_c = params

            # predict next offer of the opponent
            x_pred = len(agent.opponent_utilities)

            # i'll check if y_pred decreased inside update_reserved_value func
            # use fitted curve parameters to predict next offer of the opponent (the reservation value).
            y_pred = quadratic_polynomial(x_pred, optimized_a, optimized_b, optimized_c)
            if y_pred[0] < 0:
                y_pred[0] = 0
            elif y_pred[0] > 1:
                y_pred[0] = 1
            agent.past_opponent_rv_list.append(y_pred)

            return y_pred
        except Exception:
            # If we for any reason (e.g. non-convergence) then we use last opponent offer utility as opponent RV
            # Such a thing happens extremely rarely, almost never.
            agent.past_opponent_rv_list.append(agent.opponent_utilities[-1])
            return agent.opponent_utilities[-1]


# interpolation function for rv_feature_mode = 1 (round + Nash point distance) or rv_feature_mode = 2 (round + PF)
def interpolation_1(agent, window_size, f1):
    # if we did not reach window size, assume opponent rv is 1, highest.
    if len(agent.opponent_utilities) < window_size:
        # agent.past_opponent_rv_list.append(1.0)
        return 1.0

    # else, xs are the last N (window_size) offer numbers of the opponent.
    # ys are the last N offers.
    else:
        if len(f1) == window_size:
            x = np.array(
                [
                    list(
                        range(
                            len(agent.opponent_utilities) - window_size,
                            len(agent.opponent_utilities),
                        )
                    ),
                    [f1[0]] + f1[-window_size:-1],
                ],
                dtype=float,
            )
        else:
            x = np.array(
                [
                    list(
                        range(
                            len(agent.opponent_utilities) - window_size,
                            len(agent.opponent_utilities),
                        )
                    ),
                    f1[-window_size - 1 : -1],
                ],
                dtype=float,
            )
        y = np.array(agent.opponent_utilities[-window_size:], dtype=float)

        try:
            # Fit the curve.
            warnings.filterwarnings("ignore")
            params, covariance = curve_fit(quadratic_2D, x, y)
            warnings.filterwarnings("default")

            (
                optimized_a,
                optimized_b,
                optimized_c,
                optimized_d,
                optimized_e,
                optimized_f,
            ) = params

            # predict next offer of the opponent
            x_pred = np.array([[len(agent.opponent_utilities)], [f1[len(f1) - 1]]])
            # print(x_pred)
            # i'll check if y_pred decreased inside update_reserved_value func
            # use fitted curve parameters to predict next offer of the opponent (the reservation value).
            y_pred = quadratic_2D(
                x_pred,
                optimized_a,
                optimized_b,
                optimized_c,
                optimized_d,
                optimized_e,
                optimized_f,
            )
            if y_pred[0] < 0:
                y_pred[0] = 0
            elif y_pred[0] > 1:
                y_pred[0] = 1
            agent.past_opponent_rv_list.append(y_pred[0])

            # print(y_pred)
            return y_pred[0]
        except Exception:
            # If we for any reason (e.g. non-convergence) then we use last opponent offer utility as opponent RV
            # Such a thing happens extremely rarely, almost never.
            agent.past_opponent_rv_list.append(agent.opponent_utilities[-1])
            return agent.opponent_utilities[-1]
