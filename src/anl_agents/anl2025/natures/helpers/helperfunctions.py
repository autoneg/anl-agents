#### all functions below are hulp functions for the functions above. They are not necessary to implement the agent, but they can be useful." ####

from negmas.sao.controllers import SAOState
import itertools

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
    cartesian_product = list(itertools.product(*arrays))
    #  print(cartesian_product)
    return cartesian_product

def is_edge_agent(self):
    """Returns True if the agent is an edge agent, False otherwise, then the agent is a center agent."""
    if self.id.__contains__("edge"):
        return True
    return False

def all_possible_bids_with_agreements_fixed(self):
    """ This function returns all the bids that are still possible to achieve, given the agreements that were made in the previous negotiations."""

    # Once a negotiation has ended, the bids in the previous negotiations cannot be changed.
    # Therefore, this function helps to construct all the bids that can still be achieved, fixing the agreements of the previous negotiations.

    # If the agent is an edge agent, there is just one bid to be made, so we can just return the outcome space of the utility function.
    # Watch out, the structure of the outcomes for an edge agent is different from for a center agent.


    if is_edge_agent(self):
        return self.ufun.outcome_space.enumerate_or_sample()

    possible_outcomes = []
    neg_index = get_current_negotiation_index(self)

    #As the previous agreements are fixed, these are added first.
    for i in range(neg_index):
        possible_outcomes.append([get_agreement_at_index(self,i)])

    # The coming negotiations (with index higher than the current negotiation index) can still be anything, so we just list all possibilities there.
    n = get_number_of_subnegotiations(self)
    for i in range(neg_index, n):
        possible_outcomes.append(get_outcome_space_from_index(self,i))
    #print(possible_outcomes)

    #The cartesian product constructs the combinations of all possible outcomes.
    adapted_outcomes = cartesian_product(possible_outcomes)
    return adapted_outcomes

def get_nmi_from_index(self, index):
    """
    This function returns the nmi of the subnegotiation with the given index.
    The nmi is the negotiator mechanism interface per subnegotiation. Here you can find any information about the ongoing or ended negotiation, like the agreement or the previous bids.
    """
    negotiator_id = self.id_dict[index]
    return self.negotiators[negotiator_id].negotiator.nmi