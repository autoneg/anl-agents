import math

from .helpers.config import DEFAULT_SETTINGS


# time-dependent function
def calculate_discount_factor(t, k, e):
    # Calculate the discount factor F(t) at time t.
    """
    t: time in (0, 1)
    k: initial threshold in (0, 1)
    e: concession factor
    """
    F_t = k + (1 - k) * math.pow(t, 1 / e)
    return F_t


def calculate_target_utility(t, R, Pmax, k, e):
    # Calculate the target utility function at time t.
    """
    t: time in (0, 1)
    R: reservation value
    Pmax: maximum utility value
    k: initial threshold in (0, 1)
    e: concession factor
    """
    F_t = calculate_discount_factor(t, k, e)
    u_t = R + (Pmax - R) * (1 - F_t)
    return u_t


# exceptional case
def exceptional_acceptance_condition(current_offer_utlity):
    exceptional_utility_threshold = 0.95
    return current_offer_utlity >= exceptional_utility_threshold


def should_accept_offer(current_offer_utlity: float, R: float, t: float) -> bool:
    """
    current_offer_utlity: the utility of the current offer
    R: reservation value
    t: time in (0, 1)
    """
    Pmax = 1
    k = DEFAULT_SETTINGS["acceptance"]["k"]  # remain to be defined
    e = DEFAULT_SETTINGS["acceptance"]["e"]  # remain to be defined
    u_t = calculate_target_utility(t, R, Pmax, k, e)

    # For the first 95% of the session, we only accept offers that are exceptional case.
    # After that, we accept offers that are greater than the target utility.
    if t < DEFAULT_SETTINGS["acceptance"]["time_threshold"]:
        return exceptional_acceptance_condition(current_offer_utlity)
    elif current_offer_utlity >= u_t or exceptional_acceptance_condition(
        current_offer_utlity
    ):
        return True


if __name__ == "__main__":
    # just begin negotiation, should be false
    # print(should_accept_offer(0.7, 0.6, 0))

    # half way through, should be true
    # print(should_accept_offer(0.7, 0.6, 0.5))

    # almost end, should be true
    # print(should_accept_offer(0.7, 0.6, 0.99))

    # just begin negotiation, should be true due to exceptional acceptance condition
    # print(should_accept_offer(0.96, 0.9, 0))
    pass


# Should we accept the offer?
# accept = should_accept_offer(P_offer, t, R, Pmax, k, e, previous_offers, offer_times, threshold_time=10, threshold_increase=10)
# print(f"Accept offer: {accept}")  # This will print whether the offer should be accepted or not.
