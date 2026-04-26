#### all functions below are hulp functions for the functions above. They are not necessary to implement the agent, but they can be useful." ####

from negmas.sao.controllers import SAOState
import itertools
# from .MCQTD3.MCQTD3PolicyLoader import MCQTD3PolicyLoader
import numpy as np
import random
from bisect import bisect_left
import os
import joblib
import time
import pathlib
import pandas as pd

def set_id_dict(self):
    """Creates a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
    This dictionary allows us to find the right id for easy access to further information about the specific negotiation."""
    for id in self.negotiators.keys():
        index = self.negotiators[id].context['index']
        self.id_dict[index] = id


def did_negotiation_end(self):
    """The function finished_negotiators checks how many threads are finished. If that number changes, then the next negotiation has started."""
    if self.current_neg_index != len(self.finished_negotiators):
        self.current_neg_index = len(self.finished_negotiators)
        return True
    return False

def get_current_negotiation_index(self):
    """Returns the index of the current negotiation. An index 0 means the first negotiation in the sequence, index 1 the second, etc."""
    return len(self.finished_negotiators)


def get_agreement_at_index(self, index):
    """The nmi is the negotiator mechanism interface, available for each subnegotiation.
    Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids."""
    nmi = get_nmi_from_index(self, index)
    agreement = nmi.state.agreement
    # print(agreement)
    return agreement

def get_outcome_space_from_index(self, index):
    """This function returns the outcome space of the subnegotiation with the given index."""
    outcomes = self.ufun.outcome_spaces[index].enumerate_or_sample()
    #  print(outcomes)

    # Each subnegotiation can also end in disagreement aka outcome None. Therefore, we add this to all outcomes.
    outcomes.append(None)

    return outcomes

def get_number_of_subnegotiations(self):
    """Returns the total number of (sub)negotiations that the agent is involved in. For edge agents, this is 1."""
    return len(self.negotiators)

def cartesian_product(arrays):
    """This function returns the cartesian product of the given arrays."""

    if cartesian_product_count(arrays) > 600000:

        cartesian_product = cartesian_product_generator_v2(arrays,target_count = 600000)

    else:
        cartesian_product = list(itertools.product(*arrays))


    return cartesian_product


def cartesian_product_generator_v2(arrays, target_count):

    num_arrays = len(arrays)


    generators = []
    max_combinations = []
    for i in range(num_arrays):

        new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
        generators.append(itertools.product(*new_arrays))

        current_max = 1
        for j in range(num_arrays):
            current_max *= len(arrays[j])
        max_combinations.append(current_max)


    count = 0
    generator_index = 0
    generator_counts = [0] * num_arrays

    while count < target_count:
        i = generator_index % num_arrays


        if generator_counts[i] >= max_combinations[i]:
            new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
            generators[i] = itertools.product(*new_arrays)
            generator_counts[i] = 0

        try:

            yield next(generators[i])
            generator_counts[i] += 1
            count += 1
        except StopIteration:

            new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
            generators[i] = itertools.product(*new_arrays)
            generator_counts[i] = 1
            yield next(generators[i])
            count += 1

        generator_index += 1




def cartesian_product_generator(arrays, target_count):

    num_arrays = len(arrays)


    generators = []
    max_combinations = [1] * num_arrays
    for i in range(num_arrays):

        new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
        generators.append(itertools.product(*new_arrays))
        max_combinations[i] = len(arrays[i])
        for j in range(num_arrays):
            if j != i:
                max_combinations[i] *= len(arrays[j])


    count = 0
    generator_counts = [0] * num_arrays
    while count < target_count:
        for i in range(num_arrays):
            if count >= target_count:
                break
            try:

                if generator_counts[i] >= max_combinations[i]:
                    new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
                    generators[i] = itertools.product(*new_arrays)
                    generator_counts[i] = 0


                yield next(generators[i])
                generator_counts[i] += 1
                count += 1
            except StopIteration:

                new_arrays = [arrays[i]] + [arrays[j] for j in range(num_arrays) if j != i]
                generators[i] = itertools.product(*new_arrays)
                generator_counts[i] = 1
                yield next(generators[i])
                count += 1


def fixed_number_is_bigger_target(arrays, target_count):
    result = list(cartesian_product_generator(arrays, target_count))
    return result


def is_edge_agent(self):
    """Returns True if the agent is an edge agent, False otherwise, then the agent is a center agent."""
    if self.id.__contains__("edge"):
        return True
    return False


def find_best_bid_in_outcomespace(self):
    """Fixing previous agreements, this functions returns the best bid that can still be achieved."""
    # get outcome space with all bids with fixed agreements
    updated_outcomes = all_possible_bids_with_agreements_fixed(self)

    # this function is also in self.ufun.extreme_outcomes()
    mn, mx = float("inf"), float("-inf")
    worst, best = None, None
    for o in updated_outcomes:
        u = self.ufun(o)
        if u < mn:
            worst, mn = o, u
        if u > mx:
            best, mx = o, u
    return best

def get_target_bid_at_current_index(self):
    """ Returns the bid for the current subnegotiation, with the target_bid as source.
        """
    index = get_current_negotiation_index(self)
    #An edge agents bid is only one bid, not a tuple of bids. Therefore, we can just return the target bid.
    if is_edge_agent(self):
        return self.target_bid
    return self.target_bid[index]

def get_nmi_from_index(self, index):
    """
    This function returns the nmi of the subnegotiation with the given index.
    The nmi is the negotiator mechanism interface per subnegotiation. Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids.
    """
    negotiator_id = get_negid_from_index(self,index)
    return self.negotiators[negotiator_id].negotiator.nmi

def get_negid_from_index(self, index):
    """ This function returns the negotiator id of the subnegotiation with the given index. """
    return self.id_dict[index]


def all_possible_bids_with_agreements_fixed(self):
    """ This function returns all the bids that are still possible to achieve, given the agreements that were made in the previous negotiations."""

    if is_edge_agent(self):
        return self.ufun.outcome_space.enumerate_or_sample()

    possible_outcomes = []
    neg_index = get_current_negotiation_index(self)

    for i in range(neg_index):
        possible_outcomes.append([get_agreement_at_index(self, i)])

    n = get_number_of_subnegotiations(self)

    for i in range(neg_index, n):
        possible_outcomes.append(get_outcome_space_from_index(self, i))


    adapted_outcomes = cartesian_product(possible_outcomes)

    return adapted_outcomes



def get_sorted_outcomes_with_utility(self):

    if is_edge_agent(self):

        outcomes = self.ufun.outcome_space.enumerate_or_sample()

        sorted_outcomes = sorted(
            outcomes,
            key=lambda o: self.ufun(o) if o is not None else float('-inf'),
            reverse=True
        )
        outcomes_with_utility = []
        for outcome in sorted_outcomes:
            utility = self.ufun(outcome)
            outcomes_with_utility.append([utility, outcome])


        max_u = outcomes_with_utility[0][0] if outcomes_with_utility else 0

        return outcomes_with_utility, max_u
    else:

        outcomes_with_utility = []
        sorted_outcomes = all_possible_bids_with_agreements_fixed(self)




        for outcome in sorted_outcomes:
            utility = self.ufun(outcome)
            outcomes_with_utility.append([utility, outcome])


        outcomes_with_utility.sort(key=lambda x: x[0], reverse=True)
        max_u = outcomes_with_utility[0][0] if outcomes_with_utility else 0
        # min_u = outcomes_with_utility[-1][0] if outcomes_with_utility else 0



        neg_index = get_current_negotiation_index(self)
        # [[0.9851451712232603, ('v2', 'v2')],]
        check_compromise_bid_list = [[outcome[0],outcome[1][neg_index]] for outcome in outcomes_with_utility[:50]]

        # print(check_compromise_bid_list)




        high_utility_outcomes = [outcome for outcome in outcomes_with_utility if outcome[0] >= 0.7]
        # if len(high_utility_outcomes)<3:
        #     high_utility_outcomes = outcomes_with_utility[:3]


        if not high_utility_outcomes:
            high_utility_outcomes = outcomes_with_utility[:3]



        return high_utility_outcomes,max_u,check_compromise_bid_list





def inverse_utility_with_outcomes(self, cur_utility, outcomes_with_utility, max_u):

    target_u = (cur_utility + 0.00001) / (max_u+0.00001)


    utilities = [item[0] for item in outcomes_with_utility]


    idx = bisect_left(utilities, target_u)


    candidates = []
    if idx > 0:
        candidates.append(outcomes_with_utility[idx - 1])
    if idx < len(utilities):
        candidates.append(outcomes_with_utility[idx])


    if candidates:
        closest = min(candidates, key=lambda x: abs(x[0] - target_u))
        if abs(closest[0] - target_u) <= 0.05:
            return closest[1]


    return outcomes_with_utility[0][1]


def deal_state(self,own_history,partner_history):
    deal_history = []


    min_length = min(len(own_history), len(partner_history))
    for i in range(0, min_length):
        if i < min_length:
            deal_history.append([partner_history[i], own_history[i]])
    return deal_history


def get_state(self,deal_history):
    # deal_history = self.deal_state()
    state = []
    if len(deal_history) == 1:
        state = [0, 0, 0, 0, deal_history[0][0], deal_history[0][1]]
    if len(deal_history) == 2:
        state = [0, 0, deal_history[0][0], deal_history[0][1], deal_history[1][0], deal_history[1][1], ]

    if len(deal_history) >= 3:
        state = [deal_history[-3][0], deal_history[-3][1], deal_history[-2][0], deal_history[-2][1],
                 deal_history[-1][0], deal_history[-1][1], ]
    return state





def edge_td3_model(self):
    if is_edge_agent(self):
        current_dir = pathlib.Path(__file__).parent



        model_path1 = current_dir / 'MCQTD3' / 'load_model_file' / 'td3policy.pth'
        model_path2 = current_dir / 'MCQTD3' / 'load_model_file' / 'svm_model.pkl'
        # td3_policy_loader = MCQTD3PolicyLoader(model_path = str(model_path1))
        # svm_model = joblib.load(str(model_path2))

        # return td3_policy_loader, svm_model
    return None, None


def edge_predict_is_high(self, features):

    feature_names = [f"offer{i + 1}" for i in range(20)]
    features_df = pd.DataFrame([features], columns=feature_names)


    prediction = self.edge_svm_model.predict(features_df)


    return bool(prediction[0])


def create_combined_list(self,own_history,partner_history):

    own_last_10 = own_history[-10:]
    partner_last_10 = partner_history[-10:]


    combined_list = []
    for own, partner in zip(own_last_10, partner_last_10):
        combined_list.append(own)
        combined_list.append(partner)
    return combined_list




def select_from_top_5_percent(self, sorted_utilities):

    random.seed(time.time())


    n = max(1, int(0.05 * len(sorted_utilities)))


    top_n = sorted_utilities[:n]

    bid = random.choice(top_n)[0]

    if bid < 0.9:

        return 1.0


    return random.choice(top_n)[0]




def center_get_possible_bids(self):

    sorted_outcomes = None
    outcomes = None
    if is_edge_agent(self):

        pass
    else:

        outcomes = all_possible_bids_with_agreements_fixed(self)

    sorted_outcomes = sorted(
        outcomes,
        key=lambda o: self.ufun(o) if o is not None else float('-inf'),
        reverse=True
    )

    return sorted_outcomes



def center_get_current_id_bids(self,high_utility_outcomes,max_u):
    curr_neg_index = get_current_negotiation_index(self)

    useful_bids = []
    seen = set()
    if len(high_utility_outcomes) > 1:
        for outcome in high_utility_outcomes:

            item = outcome[1][curr_neg_index]
            if item not in seen:
                seen.add(item)
                useful_bids.append(item)

    else:
        return [None],1

    return useful_bids,len(useful_bids)



def center_bid_select(self,select_index,high_utility_outcomes,max_u):
    useful_bids,len_useful_bids = center_get_current_id_bids(self,high_utility_outcomes,max_u)
    if len_useful_bids == 1:
        return useful_bids[0]
    else:

        if select_index >= len_useful_bids:

            current_time = time.time()

            time_factor = current_time % 1

            random.seed(int(time_factor * 1000000))
            select_index = random.randint(0, len_useful_bids - 1)
            return useful_bids[select_index]
        return useful_bids[select_index]



def find_bid_bigger_than_exception_valeue(self,oppo_bid):
    adapted_outcomes = all_possible_bids_with_agreements_fixed(self)
    curr_neg_index = get_current_negotiation_index(self)


    for bid_fixed in adapted_outcomes:
        if bid_fixed[curr_neg_index] == oppo_bid:
            if self.ufun(bid_fixed) >= self.center_exception_values:
                return True
    return False


def center_find_bid_5best_of_one(self,sorted_utilities):
    # adapted_outcomes = all_possible_bids_with_agreements_fixed(self)

    curr_neg_index = get_current_negotiation_index(self)

    if all(outcome[0] < 0.4 for outcome in sorted_utilities):
        selected_outcomes = [sorted_utilities[0]] if sorted_utilities else []
    else:
        selected_outcomes = [outcome for outcome in sorted_utilities if outcome[0] > 0.4]

    if not selected_outcomes:
        return None
    return random.choice(selected_outcomes)[1][curr_neg_index]


def cartesian_product_count(arrays):
    count = 1
    for array in arrays:
        count *= len(array)
    return count





def based_compomise_select_bid(prob, check_compromise_bid_list, oppo_bid_expection_for_ownu):
    """
    Select an appropriate bid from the compromise list based on the probability
    and adjust according to the opponent's expected bid for the user.

    Parameters:
    prob (float): Probability of compromise, range 0-1
    check_compromise_bid_list (list): List of compromise bids, sorted in ascending order
    oppo_bid_expection_for_ownu (float): Opponent's expected bid for the user

    Returns:
    Selected bid or None (if list is empty or no suitable bid found)
    """
    if not check_compromise_bid_list:
        return None

    n = len(check_compromise_bid_list)

    # Select initial bid based on probability
    if prob >= 0.7 or oppo_bid_expection_for_ownu >= 0.7:
        upper = max(1, int(n * 0.05))
        selected_bid = random.choice(check_compromise_bid_list[:upper])
    elif prob >= 0.5 or oppo_bid_expection_for_ownu >= 0.45:
        lower = max(1, int(n * 0.05))
        upper = max(lower + 1, int(n * 0.20))
        selected_bid = random.choice(check_compromise_bid_list[lower:upper])
    else:
        upper = max(1, int(n * 0.3))
        selected_bid = random.choice(check_compromise_bid_list[:upper])

    # Check if it's the first element and lower than expected
    if selected_bid[0] < oppo_bid_expection_for_ownu and selected_bid == check_compromise_bid_list[0]:
        return selected_bid[1]

    # Traverse from end to find the first bid higher than expected
    for bid in reversed(check_compromise_bid_list):
        if bid[0] > oppo_bid_expection_for_ownu:
            return bid[1]

    # Return original selection if no higher bid found
    # print(oppo_bid_expection_for_ownu,selected_bid)


    return selected_bid[1]